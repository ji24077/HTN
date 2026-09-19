"""Authenticated task, worker, audit, and push-update HTTP routes."""

import asyncio

import asyncpg
from fastapi import APIRouter, Depends, HTTPException, Query, Request
from fastapi.encoders import jsonable_encoder
from fastapi.responses import StreamingResponse
from pydantic import ValidationError

from ..shared.protocol import Identifier, Submission, Task, Worker, json_loads, json_text
from .auth import require_admin

router = APIRouter(prefix="/v1", dependencies=[Depends(require_admin)])


async def read_snapshot(request: Request):
    store = request.app.state.store
    # A single consistent view of worker/task state and its audit events.
    async with store.pool.acquire() as conn:
        async with conn.transaction(isolation="repeatable_read", readonly=True):
            workers = await conn.fetch("SELECT * FROM workers ORDER BY id LIMIT 500")
            tasks = await conn.fetch("SELECT * FROM tasks ORDER BY created_at DESC,id LIMIT 500")
            events = await conn.fetch("SELECT * FROM events ORDER BY id DESC LIMIT 40")
    return {
        "workers": [dict(w) for w in workers],
        "tasks": [dict(t) for t in tasks],
        "events": [dict(e) for e in events],
    }


@router.get("/snapshot")
async def snapshot(request: Request):
    return await read_snapshot(request)


@router.get("/updates")
async def updates(request: Request):
    async def stream():
        feed = request.app.state.updates
        async with feed.subscribe() as changed:
            yield "retry: 2000\n\n"
            while True:
                try:
                    await require_admin(request)
                except HTTPException:
                    return  # Expired browser sessions must authenticate again.
                try:
                    await asyncio.wait_for(changed.wait(), timeout=15)
                except TimeoutError:
                    yield ": keepalive\n\n"
                    continue
                # Clear before reading so a concurrent commit schedules
                # another snapshot instead of losing its notification.
                changed.clear()
                if not feed.connected:
                    yield "event: unavailable\ndata: {}\n\n"
                    continue
                try:
                    data = json_text(jsonable_encoder(await read_snapshot(request)))
                except (asyncpg.PostgresError, asyncpg.InterfaceError, OSError, TimeoutError):
                    yield "event: unavailable\ndata: {}\n\n"
                    return  # EventSource reconnects and requests fresh state.
                yield f"event: snapshot\ndata: {data}\n\n"

    return StreamingResponse(
        stream(),
        media_type="text/event-stream",
        headers={
            "Cache-Control": "no-cache, no-transform",
            "X-Accel-Buffering": "no",
        },
    )


@router.post("/tasks", response_model=list[Task])
async def submit(request: Request):
    data = bytearray()
    try:
        async with asyncio.timeout(10):
            async for chunk in request.stream():
                data.extend(chunk)
                if len(data) > 1024 * 1024:
                    raise HTTPException(status_code=413, detail="request exceeds 1 MiB")
    except TimeoutError as exc:
        raise HTTPException(status_code=408, detail="request body timeout") from exc
    try:
        submission = Submission.model_validate(json_loads(bytes(data)))
    except (ValueError, ValidationError) as exc:
        raise HTTPException(status_code=400, detail="invalid task submission") from exc
    for task in submission.tasks:
        if (
            task.target_worker_id
            and task.target_worker_id not in request.app.state.config.worker_tokens
            and not await request.app.state.store.enrolled_worker(task.target_worker_id)
        ):
            raise HTTPException(status_code=400, detail="unknown target worker")
    return await request.app.state.store.submit(submission.tasks)


@router.get("/tasks", response_model=list[Task])
async def tasks(request: Request):
    return await request.app.state.store.tasks()


@router.get("/tasks/{task_id}", response_model=Task)
async def task(task_id: Identifier, request: Request):
    return await request.app.state.store.task(task_id)


@router.post("/tasks/{task_id}/cancel", response_model=Task)
async def cancel(task_id: Identifier, request: Request):
    await request.app.state.store.cancel(task_id)
    return await request.app.state.store.task(task_id)


@router.get("/workers", response_model=list[Worker])
async def workers(request: Request):
    return await request.app.state.store.workers()


@router.get("/events")
async def events(request: Request, after: int = Query(default=0, ge=0, le=2**63 - 1)):
    return await request.app.state.store.events(after)
