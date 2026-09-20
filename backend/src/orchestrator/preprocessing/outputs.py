"""Job-scoped binary outputs with bounded-memory transfers and durable chunks."""

import asyncio
import hashlib
import hmac
from contextlib import suppress
from datetime import UTC, datetime
from urllib.parse import quote
from uuid import UUID, uuid4

from fastapi import APIRouter, Depends, HTTPException, Request
from fastapi.responses import Response, StreamingResponse

from ..server.auth import require_admin
from ..server.db.store import Conflict, NotFound, task_from_row
from ..shared.protocol import Identifier, json_text
from .artifacts import safe_path

CHUNK_BYTES = 1024 * 1024
MAX_OUTPUT_FILES = 100
router = APIRouter(prefix="/v1/jobs", dependencies=[Depends(require_admin)])
worker_router = APIRouter()


def authorize(task, worker_id, attempt, token):
    now = datetime.now(UTC)
    payload = task.spec.payload if isinstance(task.spec.payload, dict) else {}
    if (
        task.state != "running"
        or task.worker_id != worker_id
        or task.generation != attempt
        or task.lease_until is None
        or task.lease_until <= now
        or task.deadline is None
        or task.deadline <= now
        or not payload.get("artifact_token")
        or not hmac.compare_digest(payload["artifact_token"], token)
    ):
        raise HTTPException(403, "Output upload is not authorized for this attempt")


async def authorized_task(conn, task_id, worker_id, attempt, token):
    row = await conn.fetchrow("SELECT * FROM tasks WHERE id=$1", task_id)
    if row is None:
        raise NotFound("Task not found")
    task = task_from_row(row)
    authorize(task, worker_id, attempt, token)
    control = await conn.fetchrow(
        "SELECT j.state,p.deadline FROM supervised_jobs j LEFT JOIN simulation_jobs p ON p.job_id=j.id WHERE j.id=$1",
        task.spec.job_id,
    )
    if control["state"] not in {"active", "paused"} or (
        control["deadline"] and control["deadline"] <= datetime.now(UTC)
    ):
        raise HTTPException(403, "Job has ended")
    return task


async def save_output_stream(store, task_id, worker_id, attempt, token, name, size, source):
    """Stage small chunks without holding the scheduler lock during transfer.

    An incomplete upload is invisible. Its metadata becomes ready only after the
    declared length, digest and current assignment have been checked. Legacy
    inline outputs remain readable; new files never occupy one large bytea.
    """
    safe_path(name)
    if not 0 <= size <= 2**63 - 1:
        raise ValueError("Invalid output length")
    async with store.change() as (conn, _):
        task = await authorized_task(conn, task_id, worker_id, attempt, token)
        # Recover chunks abandoned by a crashed server on an older attempt.
        await conn.execute(
            "DELETE FROM job_outputs WHERE task_id=$1 AND attempt<$2 AND NOT ready",
            task_id,
            attempt,
        )
        old = await conn.fetchrow(
            "SELECT id,digest,size,ready FROM job_outputs WHERE task_id=$1 AND attempt=$2 AND name=$3",
            task_id,
            attempt,
            name,
        )
        if old:
            if not old["ready"]:
                raise Conflict("This file is already being uploaded")
            if old["size"] != size:
                raise Conflict("Output name already committed with different content")
            identifier = old["id"]
        else:
            count = await conn.fetchval(
                "SELECT count(*) FROM job_outputs WHERE job_id=$1", task.spec.job_id
            )
            if count >= MAX_OUTPUT_FILES:
                raise ValueError("Job output limit exceeded (100 files)")
            identifier = uuid4()
            await conn.execute(
                "INSERT INTO job_outputs(id,job_id,task_id,attempt,name,digest,size,content,ready) VALUES($1,$2,$3,$4,$5,'',$6,NULL,false)",
                identifier,
                task.spec.job_id,
                task_id,
                attempt,
                name,
                size,
            )
    digest, received, part = hashlib.sha256(), 0, 0
    pending = bytearray()

    async def write_chunk(content):
        nonlocal part
        if part % 16 == 0:
            authorize(await store.task(task_id), worker_id, attempt, token)
        if old is None:
            await store.pool.execute(
                "INSERT INTO job_output_chunks(output_id,part,content) VALUES($1,$2,$3)",
                identifier,
                part,
                content,
            )
        part += 1

    try:
        async for incoming in source:
            received += len(incoming)
            if received > size:
                raise ValueError("Output exceeds its declared length")
            digest.update(incoming)
            # ASGI messages need not match storage chunks. Never buffer an entire file.
            for offset in range(0, len(incoming), CHUNK_BYTES):
                pending.extend(incoming[offset : offset + CHUNK_BYTES])
                if len(pending) >= CHUNK_BYTES:
                    await write_chunk(bytes(pending[:CHUNK_BYTES]))
                    del pending[:CHUNK_BYTES]
        if pending:
            await write_chunk(bytes(pending))
        if received != size:
            raise ValueError("Output transfer ended before its declared length")
        checksum = digest.hexdigest()
        async with store.change() as (conn, _):
            await authorized_task(conn, task_id, worker_id, attempt, token)
            if old and old["digest"] != checksum:
                raise Conflict("Output name already committed with different content")
            if old is None:
                await conn.execute(
                    "UPDATE job_outputs SET digest=$2,ready=true WHERE id=$1", identifier, checksum
                )
        return {"id": str(identifier), "name": name, "size": size, "sha256": checksum}
    except BaseException:
        if old is None:
            # Cascade removes partial chunks; a server crash is recovered next attempt.
            with suppress(Exception):
                await asyncio.shield(
                    store.pool.execute(
                        "DELETE FROM job_outputs WHERE id=$1 AND NOT ready", identifier
                    )
                )
        raise


@worker_router.put("/v1/execution-outputs/{task_id}")
async def upload(task_id: Identifier, name: str, request: Request):
    try:
        safe_path(name)
        attempt = int(request.headers.get("x-task-attempt", "0"))
        length = request.headers.get("content-length")
        if length is None:
            raise HTTPException(411, "Content-Length is required for output uploads")
        size = int(length)
        if not 0 <= size <= 2**63 - 1:
            raise ValueError("Invalid output length")
    except ValueError as exc:
        raise HTTPException(400, str(exc)) from exc
    store = request.app.state.store
    worker = request.headers.get("x-worker-id", "")
    token = request.headers.get("authorization", "").removeprefix("Bearer ")
    task = await store.task(task_id)
    authorize(task, worker, attempt, token)
    try:
        async with asyncio.timeout(max(1, (task.deadline - datetime.now(UTC)).total_seconds())):
            return await save_output_stream(
                store, task_id, worker, attempt, token, name, size, request.stream()
            )
    except ValueError as exc:
        raise HTTPException(400, str(exc)) from exc


@router.get("/{job_id}/outputs")
async def list_outputs(job_id: Identifier, request: Request):
    store = request.app.state.store
    if not await store.pool.fetchval("SELECT 1 FROM supervised_jobs WHERE id=$1", job_id):
        raise NotFound("Job not found")
    rows = await store.pool.fetch(
        """SELECT o.id,o.name,o.size,o.digest AS sha256,o.task_id,o.attempt
        FROM job_outputs o JOIN tasks t ON t.id=o.task_id
        WHERE o.job_id=$1 AND o.ready AND t.state='succeeded' AND t.generation=o.attempt
        ORDER BY o.created_at,o.name""",
        job_id,
    )
    return {"files": [dict(row) for row in rows]}


def attachment_headers(name):
    return {
        "Content-Disposition": "attachment; filename*=UTF-8''" + quote(name, safe=""),
        "Cache-Control": "no-store",
        "X-Content-Type-Options": "nosniff",
    }


@router.api_route("/{job_id}/outputs/{output_id}", methods=["GET", "HEAD"])
async def download(job_id: Identifier, output_id: UUID, request: Request):
    pool = request.app.state.store.pool
    row = await pool.fetchrow(
        """SELECT o.name,o.size,o.content IS NOT NULL AS inline FROM job_outputs o JOIN tasks t ON t.id=o.task_id
        WHERE o.job_id=$1 AND o.id=$2 AND o.ready AND t.state='succeeded' AND t.generation=o.attempt""",
        job_id,
        output_id,
    )
    if row is None:
        raise NotFound("Output not found in an accepted task attempt")
    headers = {**attachment_headers(row["name"].split("/")[-1]), "Content-Length": str(row["size"])}
    if request.method == "HEAD":
        return Response(headers=headers, media_type="application/octet-stream")

    async def chunks():
        if row["inline"]:
            for offset in range(0, row["size"], CHUNK_BYTES):
                yield bytes(
                    await pool.fetchval(
                        "SELECT substring(content FROM $2::int FOR $3::int) FROM job_outputs WHERE id=$1",
                        output_id,
                        offset + 1,
                        CHUNK_BYTES,
                    )
                )
        else:
            after = -1
            while True:
                page = await pool.fetch(
                    "SELECT part,content FROM job_output_chunks WHERE output_id=$1 AND part>$2 ORDER BY part LIMIT 8",
                    output_id,
                    after,
                )
                if not page:
                    break
                for item in page:
                    yield bytes(item["content"])
                after = page[-1]["part"]

    return StreamingResponse(chunks(), headers=headers, media_type="application/octet-stream")


@router.api_route("/{job_id}/result", methods=["GET", "HEAD"])
async def result(job_id: Identifier, request: Request):
    task = await request.app.state.store.task(job_id)
    if task.spec.job_id != job_id or task.state != "succeeded":
        raise NotFound("Completed job result not found")
    content = json_text(task.result).encode()
    headers = {**attachment_headers("result.json"), "Content-Length": str(len(content))}
    return Response(
        b"" if request.method == "HEAD" else content, media_type="application/json", headers=headers
    )
