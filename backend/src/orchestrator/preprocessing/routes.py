import asyncio
import hmac
import zipfile
from datetime import UTC, datetime

from fastapi import APIRouter, Depends, HTTPException, Request
from fastapi.responses import Response
from pydantic import ValidationError

from ..server.auth import require_admin
from ..server.credits import account_id
from ..server.db.store import Conflict
from ..shared.protocol import Identifier, json_loads
from .artifacts import unpack
from .models import Answer, Upload
from .store import SimulationStore

router = APIRouter(prefix="/v1/simulations", dependencies=[Depends(require_admin)])
worker_router = APIRouter()


@router.post("")
async def submit(request: Request):
    if getattr(request.app.state, "preprocessing", None) is None:
        raise HTTPException(503, "Preprocessing model is not configured")
    body = bytearray()
    async for chunk in request.stream():
        body.extend(chunk)
        if len(body) > 12 * 1024 * 1024:
            raise HTTPException(413, "Upload exceeds 12 MiB request limit")
    try:
        upload = Upload.model_validate(json_loads(bytes(body)))
        files = await asyncio.to_thread(unpack, upload.files)
    except (
        ValueError,
        ValidationError,
        zipfile.BadZipFile,
        RuntimeError,
        NotImplementedError,
    ) as exc:
        raise HTTPException(400, str(exc)[:300]) from exc
    return await SimulationStore(request.app.state.store).create(
        upload, files, account_id=account_id(request)
    )


@router.get("/{job_id}")
async def status(job_id: Identifier, request: Request):
    return await SimulationStore(request.app.state.store).status(job_id)


@router.post("/{job_id}/answer")
async def answer(job_id: Identifier, body: Answer, request: Request):
    store = SimulationStore(request.app.state.store)
    async with store.store.change() as (conn, _):
        job = await conn.fetchrow("SELECT * FROM simulation_jobs WHERE job_id=$1", job_id)
        if job is None or job["phase"] != "needs_input":
            raise Conflict("This job is not waiting for input")
        data = job["data"]
        data.setdefault("answers", []).append(body.message)
        if len(data["answers"]) > 10:
            raise Conflict("Clarification limit reached; submit a revised project")
        data.pop("question", None)
        data["message"] = "Answer received. Resuming preprocessing."
        await conn.execute(
            "UPDATE simulation_jobs SET phase=$2,data=$3,revision=revision+1,retry_after=clock_timestamp() WHERE job_id=$1",
            job_id,
            data.pop("resume_phase", "inspecting"),
            data,
        )
    return {"saved": True}


@router.get("/{job_id}/artifacts/{digest}")
async def artifact(job_id: Identifier, digest: str, request: Request):
    return {"files": await SimulationStore(request.app.state.store).files(job_id, digest)}


@worker_router.get("/v1/execution-bundles/{task_id}/{digest}")
async def execution_bundle(task_id: Identifier, digest: str, request: Request):
    store = request.app.state.store
    task = await store.task(task_id)
    payload = task.spec.payload if isinstance(task.spec.payload, dict) else {}
    token = request.headers.get("authorization", "").removeprefix("Bearer ")
    if (
        task.state not in {"assigned", "running"}
        or task.lease_until is None
        or task.lease_until <= datetime.now(UTC)
        or task.deadline is None
        or task.deadline <= datetime.now(UTC)
        or not payload.get("artifact_token")
        or not hmac.compare_digest(token, payload["artifact_token"])
        or digest != payload.get("bundle_hash")
        or request.headers.get("x-worker-id") != task.worker_id
    ):
        raise HTTPException(403, "Artifact is not authorized for this assignment")
    raw = await store.pool.fetchval(
        "SELECT content FROM simulation_artifacts WHERE job_id=$1 AND digest=$2",
        task.spec.job_id,
        digest,
    )
    if raw is None:
        raise HTTPException(404, "Artifact not found")
    return Response(
        bytes(raw), media_type="application/json", headers={"Cache-Control": "no-store"}
    )
