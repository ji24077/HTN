"""Authenticated worker protocol sessions and assignment delivery."""

import asyncio
import logging
import secrets
from contextlib import suppress

from fastapi import WebSocket
from pydantic import ValidationError
from redis.asyncio import Redis
from starlette.websockets import WebSocketDisconnect, WebSocketState

from ..shared.protocol import MESSAGE_LIMIT, UNHEALTHY_AFTER, VERSION, Message, json_loads
from .db.store import StaleAssignment, StaleSession, Store

log = logging.getLogger(__name__)


async def send(socket: WebSocket, message: Message) -> None:
    async with asyncio.timeout(5):
        await socket.send_text(message.model_dump_json(exclude_none=True))


async def receive(socket: WebSocket) -> Message:
    async with asyncio.timeout(UNHEALTHY_AFTER):
        text = await socket.receive_text()
    if len(text.encode()) > MESSAGE_LIMIT:
        raise ValueError("worker message exceeds 128 KiB")
    message = Message.model_validate(json_loads(text))
    if message.version != VERSION:
        raise ValueError("unsupported protocol version")
    return message


async def cache_presence(cache: Redis | None, worker_id: str, session: str) -> None:
    if cache is None:
        return
    try:
        # Session-qualified keys cannot refresh a replacement session's presence.
        async with asyncio.timeout(0.2):
            await cache.set(
                f"orchestrator:presence:{worker_id}:{session}", "alive", ex=UNHEALTHY_AFTER
            )
    except Exception:
        log.warning("presence cache unavailable; durable ownership unaffected")


async def serve_worker(
    socket: WebSocket, store: Store, cache: Redis | None, worker_id: str, enrolled: bool = False
) -> None:
    session = None
    try:
        await socket.accept()
        hello = await receive(socket)
        if hello.type != "hello" or hello.capabilities is None:
            raise ValueError("expected hello with capabilities")
        new_session = secrets.token_hex(16)
        await store.register(worker_id, new_session, hello.capabilities, enrolled=enrolled)
        session = new_session
        await send(socket, Message(type="welcome", session_id=session))
        log.info("worker connected worker=%s session=%s", worker_id, session)
        while True:
            message = await receive(socket)
            try:
                match message.type:
                    case "heartbeat":
                        if message.sequence < 1:
                            raise ValueError("heartbeat requires a positive sequence")
                        accepted = await store.heartbeat(
                            worker_id, session, message.active, message.paused, message.progress
                        )
                        response = Message(
                            type="heartbeat_ack", sequence=message.sequence, active=accepted
                        )
                    case "ack":
                        if message.ref is None:
                            raise ValueError("ack requires an assignment reference")
                        await store.ack(worker_id, session, message.ref)
                        response = Message(type="ack_accepted", ref=message.ref)
                    case "complete" | "failed":
                        if message.ref is None:
                            raise ValueError("completion requires an assignment reference")
                        if (message.type == "failed") != bool(message.error):
                            raise ValueError(
                                "failed requires an error; complete must not contain one"
                            )
                        await store.finish(
                            worker_id,
                            session,
                            message.ref,
                            message.result,
                            message.error,
                            message.retryable,
                        )
                        response = Message(type="result_accepted", ref=message.ref)
                    case _:
                        raise ValueError("unexpected worker message")
            except StaleAssignment as exc:
                log.info("stale assignment rejected worker=%s ref=%s", worker_id, message.ref)
                await send(socket, Message(type="revoke", ref=message.ref, error=str(exc)))
                continue
            await send(socket, response)
            if message.type == "heartbeat":
                await cache_presence(cache, worker_id, session)
                # Keep the slot occupied while a result is awaiting acknowledgment.
                if not message.active and not message.paused:
                    task = await store.claim(worker_id, session)
                    if task is not None:
                        await send(socket, Message(type="assign", task=task))
    except WebSocketDisconnect:
        pass
    except (TimeoutError, StaleSession):
        log.info("worker timed out or session superseded worker=%s", worker_id)
    except (ValueError, ValidationError):
        log.info("invalid worker protocol message worker=%s", worker_id)
    except Exception:
        log.exception("worker connection failed worker=%s", worker_id)
    finally:
        if session is not None:
            try:
                await store.disconnect(worker_id, session)
            except Exception:
                log.exception("could not record worker disconnect worker=%s", worker_id)
        if socket.application_state == WebSocketState.CONNECTED:
            with suppress(Exception):
                async with asyncio.timeout(2):
                    await socket.close(code=1001)
