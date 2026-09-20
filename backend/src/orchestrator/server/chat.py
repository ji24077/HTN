"""Authenticated chat routes; orchestration lives in orchestrator.agent."""

import asyncio
import hashlib
from uuid import UUID

from fastapi import APIRouter, Depends, HTTPException, Request, Response
from pydantic import Field, field_validator

from ..agent import Turn
from ..agent.loop import MAX_HISTORY_BYTES, finish_interrupted
from ..shared.protocol import Model, json_loads, json_text
from ..shared.security import authorized
from .auth import require_admin, same_origin
from .chat_tools import FleetTools
from .db.store import Conflict

router = APIRouter(prefix="/v1/chat", dependencies=[Depends(require_admin)])


class ChatMessage(Model):
    request_id: UUID
    message: str = Field(min_length=1, max_length=4000)

    @field_validator("message")
    @classmethod
    def nonblank(cls, value):
        if not value.strip():
            raise ValueError("Message must not be blank")
        return value.strip()


def owner(request):
    token = request.app.state.config.admin_token
    if authorized(request.headers.get("authorization"), token):
        return "automation:" + hashlib.sha256(token.encode()).hexdigest()
    claims = getattr(request.state, "auth_claims", None)
    if claims:
        return "user:" + claims["sub"]
    return "demo:" + hashlib.sha256(request.app.state.ui_session.encode()).hexdigest()


@router.get("/config")
async def config(request: Request, response: Response):
    response.headers["Cache-Control"] = "no-store"
    return {"enabled": getattr(request.app.state, "chat_agent", None) is not None}


@router.get("/{conversation_id}")
async def conversation(request: Request, conversation_id: UUID, response: Response):
    response.headers["Cache-Control"] = "no-store"
    state = await request.app.state.chat_store.read(owner(request), conversation_id)
    return {"id": conversation_id, "turns": state.turns}


@router.post("/{conversation_id}/messages", response_model=Turn)
async def send(request: Request, conversation_id: UUID, response: Response):
    response.headers["Cache-Control"] = "no-store"
    # Unlike the older loopback API, cookie-driven chat mutations require Origin
    # even in demo mode. Bearer callers are not subject to browser CSRF.
    if not request.headers.get("authorization") and not same_origin(request, mutation=True):
        raise HTTPException(403, "Chat messages require a same-origin request")
    agent = getattr(request.app.state, "chat_agent", None)
    if agent is None:
        raise HTTPException(503, "Chat is not configured on this server")
    body = bytearray()
    try:
        async with asyncio.timeout(10):
            async for chunk in request.stream():
                body.extend(chunk)
                if len(body) > 32 * 1024:
                    raise HTTPException(413, "Chat message is too large")
        message = ChatMessage.model_validate(json_loads(bytes(body)))
    except TimeoutError:
        raise HTTPException(408, "Chat request body timed out") from None
    except (ValueError, RecursionError):
        raise HTTPException(400, "Invalid chat message") from None
    slots = request.app.state.chat_slots
    if slots.locked():
        raise HTTPException(429, "Chat is busy. Try again shortly.")
    async with slots:
        async with request.app.state.chat_store.edit(owner(request), conversation_id) as (
            state,
            save,
        ):
            # A held database lock means another request is still working. If we
            # acquired it and see 'running', its process stopped before completion.
            if state.turns and state.turns[-1].status == "running":
                finish_interrupted(
                    state,
                    "The previous message was interrupted. Its tools were not replayed; inspect their task IDs before requesting more work.",
                )
                await save(state)
            previous = next(
                (turn for turn in state.turns if turn.request_id == message.request_id), None
            )
            if previous:
                if previous.message != message.message:
                    raise Conflict("Message ID already used with different text")
                return previous
            if (
                len(state.turns) >= 40
                or len(json_text(state.history).encode()) > MAX_HISTORY_BYTES - 32000
            ):
                raise Conflict("This conversation is full. Please start a new chat.")
            turn = Turn(request_id=message.request_id, message=message.message)
            state.turns.append(turn)
            state.history.append({"role": "user", "content": message.message})
            await save(state)
            result = await agent.run(state, FleetTools(request), save)
            await require_admin(request)
            return result
