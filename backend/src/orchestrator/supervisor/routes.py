from fastapi import APIRouter, Depends, Query, Request
from pydantic import Field

from ..server.auth import require_admin
from ..server.db.store import Conflict
from ..server.usage import UsageStore
from ..shared.protocol import Identifier, Model
from ..shared.usage import UsageCap
from .models import Action
from .store import SupervisorStore

router = APIRouter(prefix="/v1/jobs", dependencies=[Depends(require_admin)])


class Instructions(Model):
    instructions: str = Field(max_length=8000)


@router.get("/{job_id}/usage")
async def usage(job_id: Identifier, request: Request, after: int = Query(0, ge=0, le=2**63 - 1)):
    return await UsageStore(request.app.state.store).read(job_id, after=after)


@router.put("/{job_id}/usage-cap")
async def usage_cap(job_id: Identifier, body: UsageCap, request: Request):
    return await UsageStore(request.app.state.store).set_cap(job_id, body.cap)


@router.get("/{job_id}/supervisor")
async def status(job_id: Identifier, request: Request):
    store = SupervisorStore(request.app.state.store)
    snapshot = await store.snapshot(job_id)
    rows = await store.pool.fetch(
        """SELECT id,status,created_at,conversation->'turns'->0 AS turn
           FROM supervisor_runs WHERE job_id=$1 ORDER BY created_at DESC LIMIT 10""",
        job_id,
    )
    snapshot["runs"] = [
        {
            "id": row["id"],
            "status": row["status"],
            "created_at": row["created_at"],
            "reply": row["turn"]["reply"],
            "tools": row["turn"]["tools"],
        }
        for row in rows
    ]
    snapshot["enabled"] = getattr(request.app.state, "supervisor", None) is not None
    snapshot["sentry_enabled"] = bool(getattr(request.app.state, "supervisor_sentry", None))
    return snapshot


@router.put("/{job_id}/instructions")
async def instructions(job_id: Identifier, body: Instructions, request: Request):
    store = SupervisorStore(request.app.state.store)
    await store.job(job_id)
    async with store.store.change() as (conn, _):
        state = await conn.fetchval("SELECT state FROM supervised_jobs WHERE id=$1", job_id)
        if state in {"succeeded", "failed", "cancelled"}:
            raise Conflict("job is terminal")
        await conn.execute(
            "UPDATE supervised_jobs SET instructions=$2,active_run=NULL,revision=revision+1 WHERE id=$1",
            job_id,
            body.instructions,
        )
        await conn.execute(
            "INSERT INTO supervisor_events(job_id,kind,data) VALUES($1,'instructions_changed','{}')",
            job_id,
        )
    return {"saved": True}


@router.post("/{job_id}/actions")
async def action(job_id: Identifier, body: Action, request: Request):
    return await SupervisorStore(request.app.state.store).action(job_id, body)
