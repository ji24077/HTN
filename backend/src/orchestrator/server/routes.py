"""Authenticated task, worker, audit, and push-update HTTP routes."""

import asyncio

import asyncpg
from fastapi import APIRouter, Depends, HTTPException, Query, Request
from fastapi.encoders import jsonable_encoder
from fastapi.responses import StreamingResponse
from pydantic import ValidationError

from ..shared.protocol import Identifier, Task, TaskSpec, Worker, json_loads, json_text
from ..shared.usage import MeteredSubmission
from .auth import require_admin
from .credits import account_credit, account_id
from .db.store import TASK_SUMMARY_COLUMNS

router = APIRouter(prefix="/v1", dependencies=[Depends(require_admin)])


async def read_snapshot(request: Request):
    store = request.app.state.store
    # A single consistent view of worker/task state and its audit events.
    async with store.pool.acquire() as conn:
        async with conn.transaction(isolation="repeatable_read", readonly=True):
            workers = await conn.fetch("SELECT * FROM workers ORDER BY id LIMIT 500")
            tasks = await conn.fetch(
                f"SELECT {TASK_SUMMARY_COLUMNS} FROM tasks "
                "WHERE spec->>'kind' != 'python_project' "
                "ORDER BY created_at DESC,id LIMIT 500"
            )
            events = await conn.fetch("SELECT * FROM events ORDER BY id DESC LIMIT 40")
            # The header covers all recorded runs, independently of the 500-task
            # list limit. Use the same snapshot and clock as other fleet data.
            usage = await conn.fetchrow(
                """SELECT (totals.cost_cad + (SELECT COALESCE(sum(estimated_cost_cad),0)
                    FROM usage_record_totals WHERE ended_at IS NULL))::text AS cost,
                    totals.attempts FROM usage_fleet_totals totals WHERE id"""
            )
            credit = await account_credit(conn, account_id(request))
    return {
        "account": credit,
        "workers": [dict(w) for w in workers],
        "tasks": [dict(t) for t in tasks],
        "events": [dict(e) for e in events],
        "usage": {"currency": "CAD", "estimated": True, **dict(usage)},
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
        submission = MeteredSubmission.model_validate(json_loads(bytes(data)))
    except (ValueError, ValidationError) as exc:
        raise HTTPException(status_code=400, detail="invalid task submission") from exc
    return await submit_specs(
        request, submission.tasks, submission.instructions, submission.usage_cap
    )


async def submit_specs(
    request: Request, specs: list[TaskSpec], instructions: str | None = None, usage_cap=None
) -> list[Task]:
    """Shared submission checks for the HTTP API and authenticated chat tools."""
    for task in specs:
        if (
            task.target_worker_id
            and task.target_worker_id not in request.app.state.config.worker_tokens
            and not await request.app.state.store.enrolled_worker(task.target_worker_id)
        ):
            raise HTTPException(status_code=400, detail="unknown target worker")
    account = account_id(request)
    if account is not None:
        return await request.app.state.store.submit(
            specs, instructions=instructions, usage_cap=usage_cap, account_id=account
        )
    if usage_cap is not None:
        return await request.app.state.store.submit(
            specs, instructions=instructions, usage_cap=usage_cap
        )
    if instructions is None:
        return await request.app.state.store.submit(specs)
    return await request.app.state.store.submit(specs, instructions=instructions)


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


@router.get("/tasks/{task_id}/execution-events")
async def execution_events(
    task_id: Identifier,
    request: Request,
    after: int = Query(default=0, ge=0, le=2**63 - 1),
    worker_id: Identifier | None = None,
    attempt: int | None = Query(default=None, ge=0, le=10),
):
    rows = await request.app.state.store.execution_events(task_id, after, worker_id, attempt)
    return {
        "events": rows,
        "next_cursor": rows[-1]["id"] if rows else after,
        "has_more": len(rows) == 200,
    }


@router.get("/workers", response_model=list[Worker])
async def workers(request: Request):
    return await request.app.state.store.workers()


@router.get("/events")
async def events(request: Request, after: int = Query(default=0, ge=0, le=2**63 - 1)):
    return await request.app.state.store.events(after)
