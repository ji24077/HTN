"""Job scope and recovery actions are enforced in database transactions."""

from datetime import timedelta

from fastapi.encoders import jsonable_encoder

from ..server.db.store import Conflict, NotFound, event, task_from_row
from ..shared.execution import scrub_execution
from ..shared.protocol import json_text

TERMINAL = {"succeeded", "failed", "cancelled"}


class SupervisorStore:
    def __init__(self, store):
        self.store = store
        self.pool = store.pool

    async def job(self, job_id):
        row = await self.pool.fetchrow("SELECT * FROM supervised_jobs WHERE id=$1", job_id)
        if row is None:
            raise NotFound("job not found")
        return dict(row)

    async def task(self, job_id, task_id):
        row = await self.pool.fetchrow(
            "SELECT * FROM tasks WHERE id=$1 AND spec->>'job_id'=$2", task_id, job_id
        )
        if row is None:
            raise NotFound("task not found in this job")
        return task_from_row(row)

    async def eligible_workers(self, job_id, conn=None):
        rows = await (conn or self.pool).fetch(
            """SELECT w.id,w.capabilities,w.last_seen FROM workers w
               WHERE w.state='alive' AND NOT w.paused
               AND w.last_seen>clock_timestamp()-interval '15 seconds'
               AND NOT EXISTS(SELECT 1 FROM tasks busy WHERE busy.worker_id=w.id
                   AND busy.state IN ('assigned','running'))
               AND NOT EXISTS(SELECT 1 FROM job_reservations r WHERE r.worker_id=w.id
                   AND r.job_id!=$1 AND r.expires_at>clock_timestamp())
               AND EXISTS(SELECT 1 FROM tasks t WHERE t.spec->>'job_id'=$1
                   AND t.state IN ('queued','failed') AND t.generation<(t.spec->>'max_attempts')::int
                   AND w.capabilities->'kinds' ? (t.spec->>'kind')
                   AND w.capabilities->>'runtime'=t.spec->'requirements'->>'runtime'
                   AND (w.capabilities->>'vram_mib')::int >= (t.spec->'requirements'->>'vram_mib')::int
                   AND (t.spec->>'target_worker_id' IS NULL OR t.spec->>'target_worker_id'=w.id
                       OR (t.generation>0 AND COALESCE((t.spec->>'allow_failover')::boolean,true))))
               ORDER BY w.id LIMIT 100""",
            job_id,
        )
        return jsonable_encoder([dict(r) for r in rows])

    async def snapshot(self, job_id):
        # Consistent snapshot; task payloads/results are fetched individually.
        async with (
            self.pool.acquire() as conn,
            conn.transaction(isolation="repeatable_read", readonly=True),
        ):
            job = await conn.fetchrow("SELECT * FROM supervised_jobs WHERE id=$1", job_id)
            if job is None:
                raise NotFound("job not found")
            counts = await conn.fetch(
                "SELECT state,count(*) AS count FROM tasks WHERE spec->>'job_id'=$1 GROUP BY state",
                job_id,
            )
            tasks = await conn.fetch(
                """SELECT id,state,generation,worker_id,failure,progress,deadline,
                   spec-'payload' AS spec FROM tasks WHERE spec->>'job_id'=$1
                   ORDER BY CASE WHEN state='failed' THEN 0 ELSE 1 END,created_at,id LIMIT 100""",
                job_id,
            )
            workers = await conn.fetch(
                """SELECT w.id,w.state,w.capabilities,w.last_seen,w.paused,t.id AS task_id,
                   t.generation,t.lease_until,t.id || ':' || t.generation AS reservation_id
                   FROM workers w JOIN tasks t ON t.worker_id=w.id
                   WHERE t.spec->>'job_id'=$1 AND t.state IN ('assigned','running')""",
                job_id,
            )
            events = await conn.fetch(
                "SELECT * FROM supervisor_events WHERE job_id=$1 AND id>$2 ORDER BY id LIMIT 100",
                job_id,
                job["event_cursor"],
            )
            actions = await conn.fetch(
                "SELECT * FROM supervisor_actions WHERE job_id=$1 ORDER BY at DESC LIMIT 20",
                job_id,
            )
            reserved = await conn.fetch(
                """SELECT r.worker_id,r.expires_at,w.state,w.capabilities FROM job_reservations r
                   JOIN workers w ON w.id=r.worker_id WHERE r.job_id=$1 AND r.expires_at>clock_timestamp()""",
                job_id,
            )
        result = jsonable_encoder(
            {
                "job": dict(job),
                "counts": [dict(r) for r in counts],
                "tasks": [dict(r) for r in tasks],
                "reservations": [dict(r) for r in workers],
                "events": [dict(r) for r in events],
                "actions": [dict(r) for r in actions],
                "reserved_workers": [dict(r) for r in reserved],
                "tasks_truncated": sum(r["count"] for r in counts) > len(tasks),
            }
        )
        return scrub_execution(result)

    async def logs(self, job_id, search):
        if search.task_id:
            await self.task(job_id, search.task_id)
        if search.trace_id or search.cursor:
            raise Conflict("trace_id and cursor are Sentry filters; execution logs use after")
        rows = await self.pool.fetch(
            """SELECT e.*,e.task_id || ':' || e.attempt::text AS execution_id
               FROM execution_events e JOIN tasks t ON t.id=e.task_id
               WHERE t.spec->>'job_id'=$1 AND e.id>$2
               AND ($3::text IS NULL OR e.task_id=$3)
               AND ($4::int IS NULL OR e.attempt=$4)
               AND ($5::text IS NULL OR e.worker_id=$5)
               AND ($6::timestamptz IS NULL OR e.occurred_at >= $6)
               AND ($7::timestamptz IS NULL OR e.occurred_at <= $7)
               AND ($8::text='' OR strpos(lower(e.data::text),lower($8))>0)
               AND ($9::text IS NULL OR CASE WHEN e.kind IN ('failed','error','stderr')
                    THEN 'error' ELSE 'info' END=$9)
               AND ($10::text IS NULL OR e.task_id || ':' || e.attempt::text=$10)
               ORDER BY e.id LIMIT $11""",
            job_id,
            search.after,
            search.task_id,
            search.attempt,
            search.worker_id,
            search.start,
            search.end,
            search.text,
            search.severity,
            search.reservation_id,
            search.limit + 1,
        )
        page = []
        size = 0
        for row in rows[: search.limit]:
            item = jsonable_encoder(dict(row))
            item_size = len(json_text(item).encode())
            if size + item_size > 48 * 1024:
                break
            page.append(item)
            size += item_size
        return jsonable_encoder(
            {
                "events": page,
                "has_more": len(rows) > len(page),
                "next_cursor": page[-1]["id"] if page else search.after,
            }
        )

    async def log_context(self, job_id, args):
        anchor = await self.pool.fetchrow(
            """SELECT e.* FROM execution_events e JOIN tasks t ON t.id=e.task_id
               WHERE e.id=$1 AND t.spec->>'job_id'=$2""",
            args.log_id,
            job_id,
        )
        if anchor is None:
            raise NotFound("log not found in this job")
        before = await self.pool.fetch(
            """SELECT * FROM execution_events WHERE task_id=$1 AND attempt=$2 AND id<$3
               ORDER BY id DESC LIMIT $4""",
            anchor["task_id"],
            anchor["attempt"],
            args.log_id,
            args.before,
        )
        after = await self.pool.fetch(
            """SELECT * FROM execution_events WHERE task_id=$1 AND attempt=$2 AND id>$3
               ORDER BY id LIMIT $4""",
            anchor["task_id"],
            anchor["attempt"],
            args.log_id,
            args.after,
        )
        rows = [dict(r) for r in [*reversed(before), anchor, *after]]
        # Context is bounded even when neighboring stdout entries are large.
        for row in rows:
            if len(json_text(row["data"]).encode()) > 1500:
                row["data"] = {"excerpt": json_text(row["data"])[:1500], "truncated": True}
        return jsonable_encoder({"events": rows})

    @staticmethod
    async def check_run(conn, job_id, run_id):
        if run_id is not None:
            active = await conn.fetchval(
                "SELECT active_run FROM supervised_jobs WHERE id=$1", job_id
            )
            if active != run_id:
                raise Conflict("supervisor run was superseded")

    async def remember(self, job_id, memory, run_id=None):
        async with self.store.change() as (conn, now):
            await self.check_run(conn, job_id, run_id)
            due = min([now + timedelta(seconds=120)] + [f.due_at for f in memory.followups])
            due = max(now + timedelta(seconds=30), due)
            await conn.execute(
                "UPDATE supervised_jobs SET memory=$2,next_check_at=$3 WHERE id=$1",
                job_id,
                scrub_execution(memory.model_dump(mode="json")),
                due,
            )

    async def action(self, job_id, action, run_id=None):
        request = action.model_dump(mode="json")
        async with self.store.change() as (conn, _):
            await self.check_run(conn, job_id, run_id)
            old = await conn.fetchrow(
                "SELECT request,result FROM supervisor_actions WHERE job_id=$1 AND action_id=$2",
                job_id,
                action.action_id,
            )
            if old:
                if old["request"] != request:
                    raise Conflict("action ID already used with different arguments")
                return old["result"]
            job = await conn.fetchrow("SELECT * FROM supervised_jobs WHERE id=$1", job_id)
            if job is None:
                raise NotFound("job not found")
            if job["state"] in TERMINAL:
                raise Conflict("job is terminal")
            if run_id is None:
                # A user's intervention supersedes any decision based on the
                # snapshot that preceded it, including whole-job actions.
                await conn.execute(
                    "UPDATE supervised_jobs SET active_run=NULL,revision=revision+1 WHERE id=$1",
                    job_id,
                )
            operation = action.operation
            if action.task_id:
                row = await conn.fetchrow(
                    "SELECT * FROM tasks WHERE id=$1 AND spec->>'job_id'=$2", action.task_id, job_id
                )
                if row is None:
                    raise NotFound("task not found in this job")
                task = task_from_row(row)
                if task.generation != action.expected_generation:
                    raise Conflict("task attempt changed; inspect current state")
                if operation == "retry_task":
                    if job["state"] != "active" or task.state != "failed":
                        raise Conflict("retry requires an active job and a failed task")
                    if task.generation >= task.spec.max_attempts:
                        raise Conflict("submitted attempt limit exhausted")
                    state = "queued"
                else:
                    if task.state == "succeeded":
                        raise Conflict("cannot cancel a successful task")
                    state = "cancelled"
                await conn.execute(
                    """UPDATE tasks SET state=$2,lease_until=NULL,deadline=NULL,
                       worker_id=NULL,session_id=NULL,progress=0 WHERE id=$1""",
                    task.spec.id,
                    state,
                )
                await event(
                    conn,
                    "task",
                    task.spec.id,
                    task.state,
                    state,
                    generation=task.generation,
                    worker_id=task.worker_id,
                    reason=action.reason,
                    supervisor_action=str(action.action_id),
                )
                result = {"task_id": task.spec.id, "state": state, "generation": task.generation}
            elif action.worker_id:
                if operation == "reserve_worker":
                    if job["state"] != "active":
                        raise Conflict("only active jobs can reserve workers")
                    eligible = await self.eligible_workers(job_id, conn)
                    if action.worker_id not in {w["id"] for w in eligible}:
                        raise Conflict("worker is unavailable or incompatible with this job")
                    count = await conn.fetchval(
                        "SELECT count(*) FROM job_reservations WHERE job_id=$1 AND expires_at>clock_timestamp()",
                        job_id,
                    )
                    if count >= 4:
                        raise Conflict("job reservation limit is four workers")
                    await conn.execute(
                        """INSERT INTO job_reservations(worker_id,job_id,expires_at)
                           VALUES($1,$2,clock_timestamp()+interval '5 minutes') ON CONFLICT(worker_id)
                           DO UPDATE SET job_id=$2,expires_at=EXCLUDED.expires_at""",
                        action.worker_id,
                        job_id,
                    )
                else:
                    # Release future reserved capacity; in-flight task leases are
                    # independently owned and only end on completion/cancellation.
                    deleted = await conn.fetchval(
                        "DELETE FROM job_reservations WHERE worker_id=$1 AND job_id=$2 RETURNING worker_id",
                        action.worker_id,
                        job_id,
                    )
                    if deleted is None:
                        raise NotFound("worker reservation not found in this job")
                result = {
                    "worker_id": action.worker_id,
                    "state": "reserved" if operation == "reserve_worker" else "released",
                }
            else:
                state = {"pause_job": "paused", "resume_job": "active", "cancel_job": "cancelled"}[
                    operation
                ]
                await conn.execute("UPDATE supervised_jobs SET state=$2 WHERE id=$1", job_id, state)
                if state in {"paused", "cancelled"}:
                    await conn.execute("DELETE FROM job_reservations WHERE job_id=$1", job_id)
                if state == "cancelled":
                    rows = await conn.fetch(
                        "SELECT * FROM tasks WHERE spec->>'job_id'=$1 AND state NOT IN ('succeeded','cancelled')",
                        job_id,
                    )
                    for row in rows:
                        task = task_from_row(row)
                        await conn.execute(
                            """UPDATE tasks SET state='cancelled',lease_until=NULL,deadline=NULL,
                               worker_id=NULL,session_id=NULL WHERE id=$1""",
                            task.spec.id,
                        )
                        await event(
                            conn,
                            "task",
                            task.spec.id,
                            task.state,
                            "cancelled",
                            generation=task.generation,
                            worker_id=task.worker_id,
                            reason=action.reason,
                        )
                result = {"job_id": job_id, "state": state}
            await conn.execute(
                "INSERT INTO supervisor_actions(job_id,action_id,request,result) VALUES($1,$2,$3,$4)",
                job_id,
                action.action_id,
                request,
                result,
            )
            await conn.execute(
                "INSERT INTO supervisor_events(job_id,kind,data) VALUES($1,'supervisor_action',$2)",
                job_id,
                {"action_id": str(action.action_id), **result},
            )
            return result

    async def settle(self, job_id, cursor, run_id=None):
        """Finish only after a successful run observed all events and terminal tasks."""
        async with self.store.change() as (conn, _):
            await self.check_run(conn, job_id, run_id)
            job = await conn.fetchrow("SELECT * FROM supervised_jobs WHERE id=$1", job_id)
            states = await conn.fetch(
                "SELECT DISTINCT state FROM tasks WHERE spec->>'job_id'=$1", job_id
            )
            pending = await conn.fetchval(
                "SELECT EXISTS(SELECT 1 FROM supervisor_events WHERE job_id=$1 AND id>$2)",
                job_id,
                cursor,
            )
            all_states = {r["state"] for r in states}
            state = job["state"]
            finished = bool(all_states) and all_states <= TERMINAL and not pending
            if finished:
                await conn.execute("DELETE FROM job_reservations WHERE job_id=$1", job_id)
                state = (
                    "cancelled"
                    if state == "cancelled"
                    else "failed"
                    if "failed" in all_states
                    else "cancelled"
                    if "cancelled" in all_states
                    else "succeeded"
                )
                if job["state"] not in TERMINAL:
                    # One more wake observes the authoritative terminal state and
                    # produces the final explanation before supervision closes.
                    await conn.execute(
                        "INSERT INTO supervisor_events(job_id,kind,data) VALUES($1,'job_terminal',$2)",
                        job_id,
                        {"state": state},
                    )
                    finished = False
                else:
                    await conn.execute(
                        "UPDATE supervised_jobs SET memory=jsonb_set(memory,'{followups}','[]') WHERE id=$1",
                        job_id,
                    )
            await conn.execute(
                "UPDATE supervised_jobs SET event_cursor=$2,state=$3,finalized=$4 WHERE id=$1",
                job_id,
                cursor,
                state,
                finished,
            )
