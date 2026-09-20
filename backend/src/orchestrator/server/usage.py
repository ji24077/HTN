"""Durable run estimates and soft spending caps; no payment processing."""

from decimal import Decimal

from fastapi.encoders import jsonable_encoder

from ..shared.usage import UsageCap


async def usage_summary(conn, job_id):
    from .db.store import NotFound

    row = await conn.fetchrow(
        """SELECT j.id AS job_id,j.usage_cap_cad AS cap,j.usage_cap_reached_at AS cap_reached_at,
            COALESCE(sum(u.estimated_cost_cad),0) AS cost,
            COALESCE(sum(u.elapsed_seconds),0) AS duration_seconds,
            count(u.id) AS attempts,count(u.id) FILTER (WHERE u.ended_at IS NULL) AS active_attempts
            FROM supervised_jobs j LEFT JOIN usage_record_totals u ON u.job_id=j.id
            WHERE j.id=$1 GROUP BY j.id""",
        job_id,
    )
    if row is None:
        raise NotFound("run not found")
    result = dict(row)
    cap, cost = result["cap"], result["cost"]
    result.update(
        currency="CAD",
        estimated=True,
        remaining=None if cap is None else max(Decimal(0), cap - cost),
        cap_reached=result["cap_reached_at"] is not None or (cap is not None and cost >= cap),
    )
    # Money stays decimal through SQL and JSON; conversion to a number is display-only.
    return jsonable_encoder(result, custom_encoder={Decimal: str})


async def enforce_usage_caps(conn, *, worker_id=None, job_id=None):
    """Caller holds Store.change's cross-process scheduling lock."""
    from .db.store import cancel_job_tasks

    rows = await conn.fetch(
        """SELECT j.id FROM supervised_jobs j
            LEFT JOIN usage_job_totals totals ON totals.job_id=j.id
            WHERE j.state IN ('active','paused') AND j.usage_cap_cad IS NOT NULL
            AND j.usage_cap_reached_at IS NULL
            AND ($1::text IS NULL OR EXISTS(SELECT 1 FROM tasks t WHERE t.worker_id=$1
                AND t.state IN ('assigned','running') AND t.spec->>'job_id'=j.id))
            AND ($2::text IS NULL OR j.id=$2)
            AND EXISTS(SELECT 1 FROM tasks t WHERE t.spec->>'job_id'=j.id
                AND t.state IN ('queued','assigned','running'))
            AND j.usage_cap_cad <= COALESCE(totals.cost_cad,0)
                + (SELECT COALESCE(sum(u.estimated_cost_cad),0)
                   FROM usage_record_totals u WHERE u.job_id=j.id AND u.ended_at IS NULL)""",
        worker_id,
        job_id,
    )
    for row in rows:
        job_id = row["id"]
        reason = "Run usage cap reached"
        await conn.execute(
            """UPDATE supervised_jobs SET state='cancelled',usage_cap_reached_at=clock_timestamp(),
                active_run=NULL,revision=revision+1 WHERE id=$1""",
            job_id,
        )
        # Fence any in-flight planner, and keep the visible simulation root consistent.
        await conn.execute(
            """WITH simulation AS (
                UPDATE simulation_jobs SET phase='cancelled',revision=revision+1,
                    data=jsonb_set(data,'{message}',to_jsonb($2::text)) WHERE job_id=$1
                RETURNING job_id
            ) UPDATE tasks SET spec=jsonb_set(spec,'{payload,phase}','"cancelled"'::jsonb)
              WHERE id IN (SELECT job_id FROM simulation)""",
            job_id,
            reason,
        )
        await cancel_job_tasks(conn, job_id, reason, include_failed=True)
        await conn.execute("DELETE FROM job_reservations WHERE job_id=$1", job_id)
        await conn.execute(
            "INSERT INTO supervisor_events(job_id,kind,data) VALUES($1,'usage_cap_reached',$2)",
            job_id,
            await usage_summary(conn, job_id),
        )


class UsageStore:
    def __init__(self, store):
        self.store = store

    async def read(self, job_id, *, after=0):
        async with (
            self.store.pool.acquire() as conn,
            conn.transaction(isolation="repeatable_read", readonly=True),
        ):
            summary = await usage_summary(conn, job_id)
            rows = await conn.fetch(
                """SELECT id,task_id,attempt,worker_id,hourly_rate_cad AS hourly_rate,started_at,ended_at,
                    outcome,elapsed_seconds AS duration_seconds,estimated_cost_cad AS cost,pricing_basis
                    FROM usage_record_totals WHERE job_id=$1 AND id>$2 ORDER BY id LIMIT 101""",
                job_id,
                after,
            )
        records = jsonable_encoder([dict(r) for r in rows[:100]], custom_encoder={Decimal: str})
        return {
            **summary,
            "records": records,
            "has_more": len(rows) > 100,
            "next_cursor": records[-1]["id"] if records else after,
        }

    async def set_cap(self, job_id, cap):
        from .db.store import Conflict, NotFound

        cap = UsageCap(cap=cap).cap
        async with self.store.change() as (conn, _):
            job = await conn.fetchrow("SELECT * FROM supervised_jobs WHERE id=$1", job_id)
            if job is None:
                raise NotFound("run not found")
            # Repeated calls are harmless, including after the cap cancelled the run.
            if job["usage_cap_cad"] != cap:
                if job["state"] not in {"active", "paused"}:
                    raise Conflict("cannot change the cap of a terminal run")
                await conn.execute(
                    """UPDATE supervised_jobs SET usage_cap_cad=$2,active_run=NULL,
                        revision=revision+1 WHERE id=$1""",
                    job_id,
                    cap,
                )
                await conn.execute(
                    "INSERT INTO supervisor_events(job_id,kind,data) VALUES($1,'usage_cap_changed',$2)",
                    job_id,
                    {"cap": str(cap) if cap is not None else None},
                )
            await enforce_usage_caps(conn, job_id=job_id)
            return await usage_summary(conn, job_id)
