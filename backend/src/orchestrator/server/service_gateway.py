"""Bounded reverse HTTP channels owned by the worker-gateway process."""

import asyncio
import hmac
import logging
import os
import re
from contextlib import suppress
from dataclasses import dataclass, field
from datetime import UTC, datetime
from urllib.parse import urlsplit

import httpx
from fastapi import APIRouter, Depends, HTTPException, Request, WebSocket
from fastapi.responses import StreamingResponse
from starlette.websockets import WebSocketDisconnect, WebSocketState

from ..shared.services import BODY_LIMIT, CHUNK_SIZE, ServiceAction, safe_headers
from .auth import require_admin
from .services import ServiceStore

public_router = APIRouter(dependencies=[Depends(require_admin)])
worker_router = APIRouter()
SECURITY_HEADERS = {
    "Content-Security-Policy": "sandbox; default-src 'none'; frame-ancestors 'none'",
    "X-Content-Type-Options": "nosniff",
    "Cache-Control": "no-store",
}


def forwarded(headers):
    return [
        (k, v)
        for k, v in safe_headers(headers)
        if not k.lower().startswith("access-control-")
        and k.lower() not in {"content-security-policy", "x-content-type-options", "cache-control"}
    ]


def valid_response_headers(headers):
    return (
        isinstance(headers, list)
        and len(str(headers)) <= 16384
        and all(
            isinstance(pair, list)
            and len(pair) == 2
            and all(isinstance(value, str) for value in pair)
            and re.fullmatch(r"[!#$%&'*+.^_`|~0-9a-zA-Z-]+", pair[0])
            and all(32 <= ord(c) <= 255 and ord(c) != 127 for c in pair[1])
            for pair in headers
        )
    )


def stream_response(chunks, status, headers):
    response = StreamingResponse(chunks, status_code=status, headers=SECURITY_HEADERS)
    response.raw_headers.extend(
        (key.lower().encode("latin-1"), value.encode("latin-1"))
        for key, value in forwarded(headers)
    )
    return response


async def close_socket(socket):
    """Response cleanup and channel teardown may race; close is idempotent here."""
    if socket.application_state != WebSocketState.DISCONNECTED:
        with suppress(RuntimeError, WebSocketDisconnect, OSError):
            await socket.close()


class ClientGone(Exception):
    pass


async def before_headers(operation, request):
    """ASGI does not cancel a handler when its peer leaves before headers exist."""

    async def disconnected():
        while not await request.is_disconnected():
            await asyncio.sleep(0.05)

    pending = asyncio.create_task(operation)
    watcher = asyncio.create_task(disconnected())
    try:
        done, _ = await asyncio.wait([pending, watcher], return_when=asyncio.FIRST_COMPLETED)
        if pending in done:
            return pending.result()
        watcher.result()
        raise ClientGone()
    finally:
        pending.cancel()
        watcher.cancel()
        await asyncio.gather(pending, watcher, return_exceptions=True)


@dataclass(eq=False)
class Slot:
    socket: WebSocket
    task_id: str
    generation: int
    session: str
    request: asyncio.Future = field(
        default_factory=lambda: asyncio.get_running_loop().create_future()
    )
    frames: asyncio.Queue = field(default_factory=lambda: asyncio.Queue(maxsize=2))
    done: asyncio.Event = field(default_factory=asyncio.Event)


class Gateway:
    def __init__(self, store):
        self.services = ServiceStore(store)
        self.slots: dict[str, list[Slot]] = {}
        self.controls: set[WebSocket] = set()

    async def authenticate(self, socket):
        task_id = socket.path_params["task_id"]
        try:
            generation = int(socket.headers.get("x-task-attempt", "0"))
        except ValueError:
            return None
        row = await self.services.active(
            task_id, generation, socket.headers.get("x-worker-session")
        )
        token = socket.headers.get("authorization", "").removeprefix("Bearer ")
        if (
            not row
            or row["worker_id"] != socket.headers.get("x-worker-id")
            or not hmac.compare_digest(token, row["spec"]["payload"]["artifact_token"])
        ):
            return None
        return row

    async def control(self, socket):
        row = await self.authenticate(socket)
        if row is None:
            await socket.close(code=1008)
            return
        await socket.accept()
        self.controls.add(socket)
        try:
            while True:
                async with asyncio.timeout(12):
                    message = await socket.receive_json()
                if set(message) != {"ready"} or type(message["ready"]) is not bool:
                    break
                if not await self.services.active(row["id"], row["generation"], row["session_id"]):
                    break
                await self.services.health(
                    row["id"], row["generation"], row["session_id"], message["ready"]
                )
                await socket.send_json({"accepted": True})
        except Exception:
            pass
        finally:
            self.controls.discard(socket)
            try:
                await self.services.health(row["id"], row["generation"], row["session_id"], False)
            except Exception:
                pass
            await close_socket(socket)

    async def data(self, socket):
        row = await self.authenticate(socket)
        if row is None:
            await socket.close(code=1008)
            return
        slots = self.slots.setdefault(row["id"], [])
        if len(slots) >= row["config"]["concurrency"]:
            await socket.close(code=1013)
            return
        await socket.accept()
        # Accept yields; the old pool may have emptied or another connection filled it.
        slots = self.slots.setdefault(row["id"], [])
        if len(slots) >= row["config"]["concurrency"]:
            await close_socket(socket)
            return
        slot = Slot(socket, row["id"], row["generation"], row["session_id"])
        slots.append(slot)

        async def guard():
            while True:
                await asyncio.sleep(1)
                current = await self.services.active(slot.task_id, slot.generation, slot.session)
                if current is None or (
                    slot.request.done()
                    and (
                        not current["health_at"]
                        or (datetime.now(UTC) - current["health_at"]).total_seconds() > 12
                    )
                ):
                    raise ConnectionError("Service ownership or readiness lost")

        async def transfer():
            # Detect an idle peer closing without stealing reads from a dispatched response.
            closed = asyncio.create_task(socket.receive())
            try:
                done, _ = await asyncio.wait(
                    [closed, slot.request], return_when=asyncio.FIRST_COMPLETED
                )
                if closed in done:
                    raise ConnectionError("Idle service channel disconnected")
            finally:
                closed.cancel()
                await asyncio.gather(closed, return_exceptions=True)
            meta, body = slot.request.result()
            async with asyncio.timeout(row["config"]["request_timeout_seconds"]):
                await socket.send_json(meta)
                for offset in range(0, len(body), CHUNK_SIZE):
                    await socket.send_bytes(body[offset : offset + CHUNK_SIZE])
                await socket.send_bytes(b"")
                head = await socket.receive_json()
                if (
                    type(head.get("status")) is not int
                    or not 200 <= head["status"] <= 599
                    or not valid_response_headers(head.get("headers"))
                ):
                    raise ValueError("Invalid service response headers")
                await slot.frames.put(head)
                while True:
                    message = await socket.receive()
                    if message["type"] == "websocket.disconnect":
                        raise ConnectionError("Upstream disconnected")
                    if message.get("bytes") is not None:
                        chunk = message["bytes"]
                        if len(chunk) > CHUNK_SIZE:
                            raise ValueError("Oversized service frame")
                        await slot.frames.put(chunk)
                    elif message.get("text") == "end":
                        await slot.frames.put(None)
                        await slot.done.wait()
                        return
                    else:
                        raise ValueError("Invalid service frame")

        transfer_task, guard_task = asyncio.create_task(transfer()), asyncio.create_task(guard())
        try:
            done, _ = await asyncio.wait(
                [transfer_task, guard_task], return_when=asyncio.FIRST_COMPLETED
            )
            for task in done:
                task.result()
        except BaseException as exc:
            slot.done.set()
            logging.getLogger(__name__).debug("Service data channel ended: %s", exc)
            # A failed stream must not be stuck behind queued bytes from a slow consumer.
            while not slot.frames.empty():
                slot.frames.get_nowait()
            slot.frames.put_nowait(
                exc if isinstance(exc, Exception) else ConnectionError("Channel cancelled")
            )
        finally:
            transfer_task.cancel()
            guard_task.cancel()
            await asyncio.gather(transfer_task, guard_task, return_exceptions=True)
            slots.remove(slot)
            if not slots:
                self.slots.pop(row["id"], None)
            await close_socket(socket)

    async def response(self, job_id, path, request, body):
        service = await self.services.pool.fetchrow(
            "SELECT task_id FROM hosted_services WHERE job_id=$1", job_id
        )
        row = (
            await self.services.active(service["task_id"])
            if service and service["task_id"]
            else None
        )
        if (
            not row
            or not row["health_at"]
            or (datetime.now(UTC) - row["health_at"]).total_seconds() > 12
        ):
            raise HTTPException(503, "Service is not ready")
        live = [s for s in self.slots.get(row["id"], []) if not s.done.is_set()]
        slot = next((s for s in live if not s.request.done()), None)
        if slot is None:
            raise HTTPException(429 if live else 503, "No service capacity available")
        # No await between selecting and reserving a slot.
        slot.request.set_result(
            (
                {
                    "method": request.method,
                    "path": "/" + path,
                    "query": request.url.query,
                    "headers": forwarded(request.headers.items()),
                },
                body,
            )
        )
        try:
            async with asyncio.timeout(row["config"]["request_timeout_seconds"]):
                head = await before_headers(slot.frames.get(), request)
            if isinstance(head, Exception):
                raise head
        except BaseException as exc:
            slot.done.set()
            await close_socket(slot.socket)
            if isinstance(exc, asyncio.CancelledError):
                raise
            raise HTTPException(
                499
                if isinstance(exc, ClientGone)
                else 504
                if isinstance(exc, TimeoutError)
                else 502,
                "Service request failed",
            ) from exc

        async def chunks():
            try:
                while True:
                    chunk = await slot.frames.get()
                    if isinstance(chunk, Exception):
                        raise chunk
                    if chunk is None:
                        return
                    yield chunk
            finally:
                slot.done.set()
                await close_socket(slot.socket)

        return stream_response(chunks(), head["status"], head["headers"])


def gateway(app):
    if not hasattr(app.state, "service_gateway"):
        app.state.service_gateway = Gateway(app.state.store)
    return app.state.service_gateway


async def bounded_body(request):
    body = bytearray()
    async for chunk in request.stream():
        body.extend(chunk)
        if len(body) > BODY_LIMIT:
            raise HTTPException(413, "Service request exceeds 1 MiB")
    return bytes(body)


@worker_router.websocket("/v1/service-control/{task_id}")
async def control(socket: WebSocket, task_id: str):
    await gateway(socket.app).control(socket)


@worker_router.websocket("/v1/service-data/{task_id}")
async def data(socket: WebSocket, task_id: str):
    await gateway(socket.app).data(socket)


@public_router.post("/v1/jobs/{job_id}/service-actions")
async def action(job_id: str, body: ServiceAction, request: Request):
    return await ServiceStore(request.app.state.store).action(job_id, body)


METHODS = ["GET", "HEAD", "POST", "PUT", "PATCH", "DELETE", "OPTIONS"]


@worker_router.api_route("/internal/serve/{job_id}/{path:path}", methods=METHODS)
async def internal(job_id: str, path: str, request: Request):
    key = os.getenv("SERVICE_BRIDGE_TOKEN", "")
    supplied = request.headers.get("x-dispatch-bridge", "")
    if not key or not hmac.compare_digest(key, supplied):
        raise HTTPException(401, "Invalid bridge credential")
    return await gateway(request.app).response(job_id, path, request, await bounded_body(request))


@public_router.api_route("/serve/{job_id}/{path:path}", methods=METHODS)
async def serve(job_id: str, path: str, request: Request):
    body = await bounded_body(request)
    if request.app.state.surface == "combined":
        return await gateway(request.app).response(job_id, path, request, body)
    origin = os.getenv("SERVICE_BRIDGE_URL", "http://127.0.0.1:8081").rstrip("/")
    key = os.getenv("SERVICE_BRIDGE_TOKEN", "")
    parsed = urlsplit(origin)
    if (
        not key
        or (
            parsed.scheme != "https"
            and not (
                parsed.scheme == "http" and parsed.hostname in {"127.0.0.1", "localhost", "::1"}
            )
        )
        or parsed.username
        or parsed.query
        or parsed.fragment
    ):
        raise HTTPException(503, "Service bridge is not configured")
    client = httpx.AsyncClient(
        timeout=httpx.Timeout(3610, connect=10), follow_redirects=False, trust_env=False
    )
    try:
        # Re-encode paths with the HTTP client's URL model, never interpret an uploaded destination.
        from urllib.parse import quote

        url = origin + "/internal/serve/" + quote(job_id, safe="") + "/" + quote(path, safe="/")
        if request.url.query:
            url += "?" + request.url.query
        upstream = await before_headers(
            client.send(
                client.build_request(
                    request.method,
                    url,
                    content=body,
                    headers=[*forwarded(request.headers.items()), ("x-dispatch-bridge", key)],
                ),
                stream=True,
            ),
            request,
        )
    except BaseException as exc:
        await client.aclose()
        if isinstance(exc, asyncio.CancelledError):
            raise
        raise HTTPException(
            499 if isinstance(exc, ClientGone) else 502, "Service bridge unavailable"
        ) from exc

    async def chunks():
        try:
            async for chunk in upstream.aiter_raw():
                yield chunk
        finally:
            await upstream.aclose()
            await client.aclose()

    return stream_response(chunks(), upstream.status_code, upstream.headers.multi_items())


@worker_router.websocket("/v1/service-bundle/{task_id}")
async def bundle(socket: WebSocket, task_id: str):
    row = await gateway(socket.app).authenticate(socket)
    if row is None:
        await socket.close(code=1008)
        return
    await socket.accept()
    try:
        content = await socket.app.state.store.pool.fetchval(
            "SELECT content FROM simulation_artifacts WHERE job_id=$1 AND digest=$2",
            row["spec"]["job_id"],
            row["spec"]["payload"]["bundle_hash"],
        )
        if content is None:
            return
        raw = content.encode() if isinstance(content, str) else bytes(content)
        for offset in range(0, len(raw), CHUNK_SIZE):
            await socket.send_bytes(raw[offset : offset + CHUNK_SIZE])
        await socket.send_text("end")
    finally:
        await close_socket(socket)
