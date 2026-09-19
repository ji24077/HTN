"""Jack desktop/iOS protocol on the existing durable Python task scheduler.

The public listener deliberately exposes signed, paired device connections here.
The separate bearer-token worker protocol stays on the private worker listener.
"""

import asyncio
import logging
import secrets
import time
from collections import OrderedDict
from contextlib import suppress
from dataclasses import dataclass
from datetime import datetime, timezone
from typing import Literal
from uuid import uuid4

import sentry_sdk
from fastapi import APIRouter, Depends, HTTPException, Request, WebSocket
from fastapi.responses import JSONResponse
from pydantic import BaseModel, ConfigDict, Field, ValidationError
from starlette.websockets import WebSocketDisconnect, WebSocketState

from ..shared.dwp import (
    assertion_identity,
    parse_frame,
    validate_public_key,
    verify_assertion,
    verify_result,
)
from ..shared.protocol import (
    HEARTBEAT_INTERVAL,
    JSON_LIMIT,
    LEASE_SECONDS,
    MESSAGE_LIMIT,
    UNHEALTHY_AFTER,
    Capabilities,
    JsonTooLarge,
    Machine,
    Ref,
    bounded_json,
    task_ref,
)
from ..shared.worker_telemetry import worker_telemetry
from .auth import require_admin
from .db.store import (
    Conflict,
    EnrollmentLimit,
    NotFound,
    StaleAssignment,
    StaleSession,
    Store,
    ingest_execution_events,
)
from .dwp_assets import release_info

log = logging.getLogger(__name__)
router = APIRouter()
ADAPTERS = frozenset({"echo", "walker_evolution", "cpu_inference_batch"})


class PairRequest(BaseModel):
    model_config = ConfigDict(extra="forbid")
    code: str = Field(min_length=4, max_length=64, repr=False)
    publicKey: str = Field(min_length=16, max_length=256)
    label: str = Field(
        default="New device", min_length=1, max_length=60, pattern=r"^[^\x00-\x1f\x7f]+$"
    )


class Frame(BaseModel):
    v: Literal[1]
    id: str = Field(min_length=1, max_length=128)
    ts: str = Field(max_length=64)
    type: str = Field(min_length=1, max_length=64)
    payload: dict


class Consent(BaseModel):
    paused: bool = Field(strict=True)
    allowCompute: bool = Field(strict=True)
    allowBrowser: bool = Field(strict=True)
    maxConcurrency: int = Field(ge=1, le=1024, strict=True)


class Capability(BaseModel):
    """What the device reports about itself at hello.

    The field names are the agent's, not ours — this model sits on the wire boundary, so
    it mirrors `CapabilityRecord` in packages/protocol. Everything but `adapters` is
    optional: an older device, or the iOS build, may send a subset, and a missing core
    count must not stop a machine joining.
    """

    adapters: list[str] = Field(min_length=1, max_length=32)
    agentVersion: str | None = Field(default=None, max_length=64)
    os: str | None = Field(default=None, max_length=32)
    arch: str | None = Field(default=None, max_length=32)
    cpuModel: str | None = Field(default=None, max_length=128)
    logicalCores: int | None = Field(default=None, ge=1, le=4096)
    totalRamMb: int | None = Field(default=None, ge=0)
    freeRamMb: int | None = Field(default=None, ge=0)


class Hello(BaseModel):
    capability: Capability
    consent: Consent


class Heartbeat(BaseModel):
    freeRamMb: float = Field(ge=0, allow_inf_nan=False)
    running: int = Field(ge=0, le=1024, strict=True)


class LeaseRef(BaseModel):
    taskId: str = Field(min_length=1, max_length=128)
    leaseId: str = Field(min_length=1, max_length=128)


class TaskError(LeaseRef):
    errorClass: str = Field(min_length=1, max_length=128)
    message: str = Field(default="", max_length=2048)


def origin(request: Request) -> str:
    return request.app.state.config.public_origin or str(request.base_url).rstrip("/")


def limit_pairing(request: Request) -> None:
    """Bound unauthenticated attempts and memory without trusting forwarded IPs."""
    limits = getattr(request.app.state, "device_pair_limits", None)
    if limits is None:
        limits = request.app.state.device_pair_limits = OrderedDict()
    now = time.monotonic()
    for key in list(limits):
        if now - limits[key][0] >= 300:
            del limits[key]
    peer = request.client.host if request.client else "unknown"
    for key, cap in (("global", 1000), ("peer:" + peer, 20)):
        started, count = limits.get(key, (now, 0))
        if count >= cap or (key not in limits and len(limits) >= 1024):
            raise HTTPException(429, "Too many pairing attempts", headers={"Retry-After": "300"})
        limits[key] = (started, count + 1)


@router.post("/v1/device-invites", dependencies=[Depends(require_admin)])
async def invite(request: Request):
    claims = getattr(request.state, "auth_claims", None)
    try:
        code = await request.app.state.store.create_pair_code(claims["sub"] if claims else None)
    except EnrollmentLimit:
        raise HTTPException(429, "Device invite limit reached") from None
    return JSONResponse(
        {"code": code, "expires_in": 600, "server": origin(request)},
        status_code=201,
        headers={"Cache-Control": "no-store"},
    )


@router.post("/hosts/pair")
async def pair(request: Request):
    limit_pairing(request)
    body = bytearray()
    try:
        async with asyncio.timeout(10):
            async for chunk in request.stream():
                body.extend(chunk)
                if len(body) > 4096:
                    raise HTTPException(413, "Pairing request too large")
        data = PairRequest.model_validate_json(body)
        public_key = validate_public_key(data.publicKey)
    except (ValidationError, ValueError):
        raise HTTPException(400, "Invalid pairing request") from None
    except TimeoutError:
        raise HTTPException(408, "Pairing request timed out") from None
    label = data.label.strip() or "New device"
    try:
        worker_id = await request.app.state.store.pair_device(
            data.code.strip().upper(), public_key, label
        )
    except Conflict:
        raise HTTPException(400, "Invalid or expired pairing code") from None
    except EnrollmentLimit:
        raise HTTPException(429, "Device enrollment limit reached") from None
    except ValueError:
        raise HTTPException(400, "Invalid pairing request") from None
    return JSONResponse(
        {
            "hostId": worker_id,
            "label": label,
            "wsUrl": origin(request).replace("http", "ws", 1) + "/agent/connect",
            "telemetry": worker_telemetry(),
            **release_info(),
        },
        headers={"Cache-Control": "no-store"},
    )


async def authenticate_device(store: Store, authorization: str | None) -> str:
    """Validate bounded Ed25519 credentials before database identity lookup."""
    if not authorization or len(authorization) > 4096 or not authorization.startswith("Bearer "):
        raise ValueError("missing device assertion")
    token = authorization[7:]
    worker_id = assertion_identity(token)
    public_key = await store.device_key(worker_id)
    if not public_key:
        raise ValueError("unknown device")
    claims = verify_assertion(token, public_key)
    if not await store.use_assertion_jti(worker_id, claims["jti"], claims["exp"]):
        raise ValueError("replayed or revoked device assertion")
    return worker_id


async def send(socket: WebSocket, kind: str, payload: dict, reply_to: str | None = None):
    frame = {
        "v": 1,
        "id": str(uuid4()),
        "ts": datetime.now(timezone.utc).isoformat(),
        "type": kind,
        "payload": payload,
    }
    if reply_to:
        frame["replyTo"] = reply_to
    async with asyncio.timeout(5):
        await socket.send_json(frame)


async def receive(socket: WebSocket):
    text = await socket.receive_text()
    if len(text.encode("utf-8")) > MESSAGE_LIMIT:
        raise ValueError("device message too large")
    frame, output = parse_frame(text)
    if type(frame.get("v")) is not int:
        raise ValueError("invalid protocol version")
    return Frame.model_validate(frame).model_dump(), output


@dataclass
class Assignment:
    ref: Ref
    lease: str
    accepted: bool = False
    job_id: str = ""

    def matches(self, payload: LeaseRef) -> bool:
        return payload.taskId == self.ref.task_id and secrets.compare_digest(
            payload.leaseId, self.lease
        )


class Connection:
    def __init__(self, socket: WebSocket, store: Store, worker_id: str, public_key: str):
        self.socket, self.store = socket, store
        self.worker_id, self.public_key = worker_id, public_key
        self.session = secrets.token_hex(16)
        self.registered = False
        self.paused = True
        self.active: Assignment | None = None

    async def cancel(self, task_id: str, reason: str = "assignment no longer active"):
        await send(self.socket, "task.cancel", {"taskId": task_id, "reason": reason})

    async def reject_result(self, active: Assignment) -> None:
        """Fail the one task whose result we will not store, and keep the connection.

        Not retryable, and that is the point. These adapters are deterministic — the same
        slice on another machine produces byte-identical output — so re-queueing an
        oversized result just hands the same poison pill to the next worker. Together
        with the close that used to follow, that is how one 1500-gait slice took a whole
        fleet down: every machine that picked the slice up computed it, had its socket
        closed on the way back, was marked unhealthy, reconnected, and lost whatever else
        it was holding as `worker_reconnected`.

        The device is told the assignment is over so it stops renewing a lease against a
        task that has already failed.
        """
        log.warning(
            "device result rejected as too large worker=%s task=%s limit=%d",
            self.worker_id,
            active.ref.task_id,
            JSON_LIMIT,
        )
        if not active.accepted:
            # `finish` only accepts a running task, and a result can arrive before the
            # accept we asked for has been processed.
            await self.store.ack(self.worker_id, self.session, active.ref)
        await self.store.finish(
            self.worker_id, self.session, active.ref, None, "result_too_large", False
        )
        self.active = None
        await self.cancel(
            active.ref.task_id,
            f"result exceeds the {JSON_LIMIT}-byte limit; submit this work in smaller slices",
        )

    async def refresh(self):
        # Presence is evidence of a live authenticated connection; a task lease is
        # renewed separately, only by a message containing the current opaque lease.
        if await self.store.device_key(self.worker_id) != self.public_key:
            await send(self.socket, "revoked", {"reason": "device revoked"})
            raise StaleSession("device revoked")
        await self.store.heartbeat(self.worker_id, self.session, [], self.paused)
        if self.active:
            ref = self.active.ref
            try:
                task = await self.store.task(ref.task_id)
                now = datetime.now(timezone.utc)
                valid = (
                    task_ref(task) == ref
                    and task.worker_id == self.worker_id
                    and task.session_id == self.session
                    and task.state in {"assigned", "running"}
                    and task.lease_until is not None
                    and task.lease_until > now
                    and task.deadline is not None
                    and task.deadline > now
                )
            except NotFound:
                valid = False
            if not valid:
                self.active = None
                await self.cancel(ref.task_id)
        if self.active is None and not self.paused:
            task = await self.store.claim(self.worker_id, self.session)
            if task:
                self.active = Assignment(
                    task_ref(task), secrets.token_hex(24), job_id=task.spec.job_id
                )
                await send(
                    self.socket,
                    "task.offer",
                    {
                        "taskId": task.spec.id,
                        "jobId": task.spec.job_id,
                        "adapter": task.spec.kind,
                        "attempt": task.generation,
                        "input": task.spec.payload,
                        "leaseId": self.active.lease,
                        "leaseSeconds": LEASE_SECONDS,
                        "wallClockMs": task.spec.timeout_seconds * 1000,
                    },
                )

    async def hello(self, frame: dict):
        if frame["type"] != "hello":
            raise ValueError("expected hello")
        hello = Hello.model_validate(frame["payload"])
        kinds = sorted(set(hello.capability.adapters) & ADAPTERS)
        if not kinds:
            raise ValueError("no supported adapters")
        await self.store.register(
            self.worker_id,
            self.session,
            Capabilities(
                # Still "cpu" with no VRAM, and deliberately so: these adapters are pure
                # JavaScript and no device path here dispatches to a GPU. Reporting
                # otherwise would let the scheduler match work against hardware that is
                # never used. Measured, it would also be wrong to prefer: the GPU backends
                # ran slower than the CPU for models this size.
                runtime="cpu",
                vram_mib=0,
                kinds=kinds,
                machine=Machine(
                    os=hello.capability.os,
                    arch=hello.capability.arch,
                    # Vendor strings arrive padded — "…Radeon Graphics         " — and the
                    # padding survives into every log line and dashboard that shows it.
                    cpu_model=(hello.capability.cpuModel or "").strip() or None,
                    logical_cores=hello.capability.logicalCores,
                    total_ram_mb=hello.capability.totalRamMb,
                    agent_version=hello.capability.agentVersion,
                    max_concurrency=hello.consent.maxConcurrency,
                ),
            ),
            expected_device_key=self.public_key,
        )
        self.registered = True
        self.paused = hello.consent.paused or not hello.consent.allowCompute
        await send(
            self.socket,
            "hello.ack",
            {
                "hostId": self.worker_id,
                "serverTime": datetime.now(timezone.utc).isoformat(),
                "heartbeatSeconds": HEARTBEAT_INTERVAL,
                "releaseVersion": release_info()["releaseVersion"],
                "telemetry": worker_telemetry(),
                "executionEvents": True,
            },
            frame["id"],
        )
        await self.refresh()

    async def message(self, frame: dict, raw_output: str | None):
        kind, payload = frame["type"], frame["payload"]
        if kind == "task.events":
            ack = await ingest_execution_events(self.store, self.worker_id, self.session, payload)
            if ack is not None:
                await send(self.socket, "task.events.ack", ack)
            return
        if kind == "heartbeat":
            Heartbeat.model_validate(payload)
        elif kind == "consent.update":
            consent = Consent.model_validate(payload)
            self.paused = consent.paused or not consent.allowCompute
        elif kind in {"task.accept", "task.decline", "lease.renew", "task.result", "task.error"}:
            ref = LeaseRef.model_validate(payload)
            active = self.active
            if active is None or not active.matches(ref):
                # Jack's cancellation frame names only taskId. Do not let a late
                # result cancel a newer generation of that same task.
                if active is None or active.ref.task_id != ref.taskId:
                    await self.cancel(ref.taskId, "stale lease")
                return
            try:
                if kind == "task.accept":
                    await self.store.ack(self.worker_id, self.session, active.ref)
                    active.accepted = True
                elif kind == "lease.renew":
                    if not active.accepted:
                        raise StaleAssignment("task not accepted")
                    accepted = await self.store.heartbeat(
                        self.worker_id, self.session, [active.ref], self.paused
                    )
                    if active.ref not in accepted:
                        raise StaleAssignment("lease expired")
                elif kind == "task.result":
                    try:
                        evidence = verify_result(
                            payload,
                            raw_output,
                            self.public_key,
                            task_id=active.ref.task_id,
                            attempt=active.ref.generation,
                            worker_id=self.worker_id,
                        )
                        result = bounded_json(payload.get("output"))
                    except JsonTooLarge:
                        # Deliberately not the same as a forged result. A signature that
                        # does not verify means this connection is lying and closing it is
                        # right; a result that is merely too big means this *slice* was
                        # too big, and the connection is doing exactly what it was told.
                        await self.reject_result(active)
                    else:
                        with sentry_sdk.start_span(
                            op="device.result", name="Accept signed device result"
                        ):
                            await self.store.finish(
                                self.worker_id,
                                self.session,
                                active.ref,
                                result,
                                "",
                                False,
                                attestation=evidence,
                            )
                        self.active = None
                else:
                    retryable = True
                    if kind == "task.error":
                        reported = TaskError.model_validate(payload)
                        # Device-provided free text and task data never enter telemetry.
                        failure = "device_execution_failed"
                        if reported.errorClass == "result_too_large":
                            # The one device-reported class that is a fact about the task
                            # rather than about the machine: the slice is too big for a
                            # result frame, and it will be too big on the next machine too.
                            # Not a Sentry error for the same reason — nothing is broken.
                            failure, retryable = "result_too_large", False
                            log.warning(
                                "device refused to send an oversized result "
                                "worker=%s task=%s limit=%d",
                                self.worker_id,
                                active.ref.task_id,
                                JSON_LIMIT,
                            )
                    else:
                        failure = "device_declined"
                    if not active.accepted:
                        await self.store.ack(self.worker_id, self.session, active.ref)
                    await self.store.finish(
                        self.worker_id, self.session, active.ref, None, failure, retryable
                    )
                    if kind == "task.error" and failure != "result_too_large":
                        with sentry_sdk.new_scope() as scope:
                            scope.set_tag("device.protocol", "dwp-v1")
                            scope.set_tag("device.id", self.worker_id)
                            scope.set_tag("task.id", active.ref.task_id)
                            scope.set_tag("task.generation", active.ref.generation)
                            scope.set_tag("worker_id", self.worker_id)
                            scope.set_tag("task_id", active.ref.task_id)
                            scope.set_tag("job_id", active.job_id)
                            scope.set_tag("attempt", active.ref.generation)
                            scope.set_tag(
                                "reservation_id", f"{active.ref.task_id}:{active.ref.generation}"
                            )
                            scope.set_tag(
                                "execution_id", f"{active.ref.task_id}:{active.ref.generation}"
                            )
                            sentry_sdk.capture_message(
                                "Paired device execution failed", level="error"
                            )
                    self.active = None
            except StaleAssignment:
                self.active = None
                await self.cancel(ref.taskId)
        elif kind != "task.progress":
            raise ValueError("unexpected device message")
        await self.refresh()


@router.websocket("/agent/connect")
async def connect(socket: WebSocket):
    connection = None
    pending = None
    close_code = 1008
    try:
        store = socket.app.state.store
        worker_id = await authenticate_device(store, socket.headers.get("authorization"))
        public_key = await store.device_key(worker_id)
        if not public_key:
            raise ValueError("device revoked")
        await socket.accept()
        connection = Connection(socket, store, worker_id, public_key)
        async with asyncio.timeout(UNHEALTHY_AFTER):
            frame, _ = await receive(socket)
        await connection.hello(frame)
        last_received = time.monotonic()
        while True:
            if pending is None:
                pending = asyncio.create_task(receive(socket))
            done, _ = await asyncio.wait({pending}, timeout=HEARTBEAT_INTERVAL)
            if done:
                frame, raw_output = pending.result()
                pending = None
                last_received = time.monotonic()
                await connection.message(frame, raw_output)
            elif time.monotonic() - last_received >= UNHEALTHY_AFTER:
                raise TimeoutError
            else:
                await connection.refresh()
    except WebSocketDisconnect:
        pass
    except StaleSession:
        # Both Jack clients treat 4000 as an intentional identity handoff. A
        # generic close would make two copies reconnect and evict each other.
        close_code = 4000
        log.info("device session superseded")
    except (ValueError, ValidationError, TimeoutError):
        log.info("device connection rejected, timed out, or superseded")
    except Exception as exc:
        # Exceptions are scrubbed by the shared Sentry setup; never log frames or keys.
        sentry_sdk.capture_exception(exc)
        log.error("device connection failed: %s", type(exc).__name__)
    finally:
        if pending is not None:
            pending.cancel()
            await asyncio.gather(pending, return_exceptions=True)
        if connection is not None and connection.registered:
            try:
                await connection.store.disconnect(connection.worker_id, connection.session)
            except Exception as exc:
                sentry_sdk.capture_exception(exc)
                log.error("device disconnect failed: %s", type(exc).__name__)
        if socket.application_state != WebSocketState.DISCONNECTED:
            with suppress(Exception):
                async with asyncio.timeout(2):
                    await socket.close(code=close_code)
