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
    Accelerator,
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
    accelerator: "AcceleratorReport | None" = None
    runtimePreference: Literal["auto", "cpu"] | None = None


class AcceleratorReport(BaseModel):
    """The device's own account of what it can compute on.

    Mirrors `AcceleratorReport` in packages/protocol, so the field names are the agent's.
    Every field is optional except the verdict: an agent may know it has no device
    without being able to name what it looked for.
    """

    model_config = ConfigDict(extra="ignore")
    runtime: Literal["cpu", "cuda", "mps"] = "cpu"
    vramMib: int = Field(default=0, ge=0, le=1_048_576)
    available: bool = False
    reason: str = Field(default="", max_length=200)
    device: str | None = Field(default=None, max_length=120)
    providers: list[str] = Field(default_factory=list, max_length=8)


class Hello(BaseModel):
    capability: Capability
    consent: Consent


class Heartbeat(BaseModel):
    freeRamMb: float = Field(ge=0, allow_inf_nan=False)
    running: int = Field(ge=0, le=1024, strict=True)


class LeaseRef(BaseModel):
    taskId: str = Field(min_length=1, max_length=128)
    leaseId: str = Field(min_length=1, max_length=128)


class SettingsAck(BaseModel):
    """What a device did about a settings.update, in its own words."""

    model_config = ConfigDict(extra="ignore")
    runtimePreference: Literal["auto", "cpu"]
    applied: bool
    detail: str = Field(default="", max_length=200)
    #: The device's re-probe after applying. Carried so the stored capability follows the
    #: switch immediately: without it the dashboard kept showing the runtime from the last
    #: hello, so a machine moved to CPU still displayed "Metal" until it reconnected.
    accelerator: AcceleratorReport | None = None


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


@router.post("/v1/join-requests")
async def join_request(request: Request):
    """Mint an invite for whoever asked, with no admin token.

    Every invite until now came from someone holding the admin token, which meant adding
    a machine needed two people in the same conversation: one to mint and one to paste.
    That is the right shape for a fleet of four and the wrong one for asking a room to
    contribute laptops, where the admin becomes a queue.

    What is *not* relaxed here is anything after the code. A self-serve invite is the
    same single-use ten-minute code `/v1/device-invites` returns, redeemed by the same
    `/hosts/pair`, and the machine that redeems it still has to hold the Ed25519 key it
    enrolled with for every later connection. This widens who may ask for a code; it does
    not widen what a code is or what holding one lets a machine do.

    Three limits stand between this and an open tap, and the third is the one that binds:
    `limit_pairing` caps a single peer at 20 attempts per five minutes, the global cap is
    1000, and `create_pair_code` allows a bounded number of *unowned* codes per hour
    across the whole network -- self-serve invites have no owner, so they all share that
    one quota. It is the difference between a page that recruits a room and a page that
    enrolls a botnet.

    The default of ten an hour is right for a link left on the internet and wrong for a
    room at an event: measured, the eleventh person to ask is told to come back in an
    hour, which reads as the network being broken rather than as a quota. Set
    DWP_SELF_SERVE_MAX_PER_HOUR for the room, and leave it alone for the internet.
    """
    if not request.app.state.config.self_serve_join:
        # 404 rather than 403: an endpoint that is switched off should not confirm it
        # exists and would work for someone better credentialed. There is no credential
        # that opens this one -- it is either on for everybody or absent.
        raise HTTPException(404, "Not found")
    limit_pairing(request)
    try:
        config = request.app.state.config
        code = await request.app.state.store.create_pair_code(
            None,
            max_per_hour=config.self_serve_max_per_hour,
            max_devices=config.self_serve_max_devices,
        )
    except EnrollmentLimit:
        # The reader is a volunteer with a laptop, not an operator reading a log. Say
        # what they should do rather than which internal quota they met.
        raise HTTPException(
            429,
            "This network has taken on as many new machines as it can for now. Try again in an hour, or ask for an invite directly.",
            headers={"Retry-After": "3600"},
        ) from None
    return JSONResponse(
        {"code": code, "expires_in": 600, "server": origin(request)},
        status_code=201,
        headers={"Cache-Control": "no-store"},
    )


@router.post("/v1/machines/{worker_id}/runtime", dependencies=[Depends(require_admin)])
async def set_runtime(worker_id: str, request: Request):
    """Set whether a machine may use a device beyond its CPU.

    The value is stored first and pushed second, and that order is the whole design. A
    machine that is asleep, restarting, or simply not connected still has to arrive at
    the setting, which it does by the server replaying the stored value at hello. Pushing
    without storing would make the switch work only for machines that happened to be
    online, which is the failure mode an operator would discover at the worst moment.

    The response says whether the machine was reachable, not whether it complied --
    compliance arrives asynchronously as `settings.ack` and lands in `runtime_applied`.
    Reporting a push as success would mean claiming a machine had switched before it had
    said anything at all.
    """
    try:
        body = await request.json()
    except ValueError:
        raise HTTPException(400, "invalid request body") from None
    preference = (body or {}).get("runtimePreference")
    if preference not in {"auto", "cpu"}:
        raise HTTPException(400, "runtimePreference must be 'auto' or 'cpu'")
    if not await request.app.state.store.set_runtime_preference(worker_id, preference):
        raise HTTPException(404, "unknown or revoked machine")
    connection = request.app.state.device_connections.get(worker_id)
    delivered = False
    if connection is not None:
        try:
            await connection.push_runtime_preference(preference)
            delivered = True
        except Exception:
            # A socket that died between the lookup and the send. The setting is already
            # stored, so the machine still picks it up when it reconnects; saying
            # delivered=false is the honest answer rather than an error.
            log.info("runtime preference stored but not delivered to %s", worker_id)
    return JSONResponse(
        {"runtimePreference": preference, "delivered": delivered},
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
        self.runtime_preference = "auto"
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
        report = hello.capability.accelerator
        accelerator = (
            Accelerator(
                available=report.available,
                reason=report.reason,
                device=report.device,
                providers=report.providers,
            )
            if report
            else None
        )
        # The stored preference wins over whatever the machine arrived believing: the
        # operator may have set it while this machine was asleep, and the machine has no
        # other way to learn that. Sent after the ack below, once the socket is live.
        preference = await self.store.runtime_preference(self.worker_id)
        self.runtime_preference = preference
        await self.store.register(
            self.worker_id,
            self.session,
            Capabilities(
                # What the machine says, not what we assume.
                #
                # This was hard-coded to cpu/0 for every device that ever connected. That
                # was true of the whole fleet and unfalsifiable: a machine with a real GPU
                # and one whose image cannot reach a GPU produced identical rows, so no
                # dashboard could tell them apart. An agent that says nothing -- every
                # agent built before this field -- still lands on cpu/0, but now as a
                # recorded absence of a claim rather than as an assertion of ours.
                runtime=report.runtime if report else "cpu",
                vram_mib=report.vramMib if report else 0,
                accelerator=accelerator,
                runtime_preference=preference,
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
        # Reachable by an operator for as long as this socket lives. Registered after
        # hello rather than at accept, so a half-open connection that never identified
        # itself is never a push target.
        self.socket.app.state.device_connections[self.worker_id] = self
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
        # Replay the operator's setting to a machine that disagrees with it. A machine
        # that already matches is left alone: re-sending on every reconnect would rewrite
        # the stored ack each time and make the dashboard flicker for no change.
        if hello.capability.runtimePreference not in (None, preference):
            await self.push_runtime_preference(preference)
        await self.refresh()

    async def push_runtime_preference(self, preference: str) -> None:
        """Ask this machine to change how it runs work. It answers with settings.ack."""
        self.runtime_preference = preference
        await send(self.socket, "settings.update", {"runtimePreference": preference})

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
        elif kind == "settings.ack":
            # The device's answer is the record, including when it refused. Writing what
            # was asked instead would let the dashboard show "GPU" on a machine that had
            # just finished explaining it has no GPU.
            ack = SettingsAck.model_validate(payload)
            self.runtime_preference = ack.runtimePreference
            await self.store.record_runtime_ack(
                self.worker_id,
                ack.runtimePreference,
                ack.applied,
                ack.detail,
                runtime=ack.accelerator.runtime if ack.accelerator else None,
                vram_mib=ack.accelerator.vramMib if ack.accelerator else None,
            )
            log.info(
                "device runtime preference %s (applied=%s)", ack.runtimePreference, ack.applied
            )
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
            # Only if this connection is still the registered one: a supersession puts the
            # newer connection in the map under the same id, and the loser's cleanup must
            # not evict the winner.
            live = socket.app.state.device_connections
            if live.get(connection.worker_id) is connection:
                live.pop(connection.worker_id, None)
            try:
                await connection.store.disconnect(connection.worker_id, connection.session)
            except Exception as exc:
                sentry_sdk.capture_exception(exc)
                log.error("device disconnect failed: %s", type(exc).__name__)
        if socket.application_state != WebSocketState.DISCONNECTED:
            with suppress(Exception):
                async with asyncio.timeout(2):
                    await socket.close(code=close_code)
