"""Persistent services reuse the task scheduler; PostgreSQL owns desired state."""

import hashlib
import secrets
from datetime import timedelta
from uuid import uuid4

from ..preprocessing.artifacts import bundle, safe_path
from ..shared.protocol import TaskSpec, json_text
from ..shared.services import ServiceConfig
from .db.store import Conflict, NotFound, cancel_job_tasks, event


class ServiceStore:
    def __init__(self, store):
        self.store, self.pool = store, store.pool

    async def exists(self, job_id):
        return await self.pool.fetchval(
            "SELECT EXISTS(SELECT 1 FROM hosted_services WHERE job_id=$1)", job_id
        )

    @staticmethod
    def validate(config, files):
        if config.entrypoint:
            if safe_path(config.entrypoint) not in files or not config.entrypoint.endswith(".py"):
                raise ValueError("Service entrypoint must be an uploaded Python file")
        if config.working_directory != ".":
            safe_path(config.working_directory)
            if not any(f.startswith(config.working_directory + "/") for f in files):
                raise ValueError("Working directory is not in the uploaded project")

    async def create(self, upload, files):
        job_id = "svc-" + upload.request_id.hex
        signature = hashlib.sha256(json_text(upload.model_dump(mode="json")).encode()).hexdigest()
        digest, content = bundle(files)
        config = upload.service or ServiceConfig()
        if config.entrypoint is None:
            candidates = [name for name in files if name.endswith(".py")]
            if len(candidates) == 1:
                config = config.model_copy(update={"entrypoint": candidates[0]})
        self.validate(config, files)
        phase = "pending" if config.entrypoint else "needs_input"
        message = (
            "Waiting for a compatible service worker."
            if config.entrypoint
            else "Which uploaded Python file starts the HTTP server? Reply with its path."
        )
        spec = TaskSpec(
            id=job_id,
            job_id=job_id,
            kind="simulation_job",
            payload={
                "value": {"label": upload.description[:100]},
                "execution_mode": "service",
                "phase": phase,
            },
            requirements=config.requirements,
            max_attempts=1,
            timeout_seconds=1,
        )
        async with self.store.change() as (conn, now):
            old = await conn.fetchval(
                "SELECT submission_hash FROM hosted_services WHERE job_id=$1", job_id
            )
            if old:
                if old != signature:
                    raise Conflict("Submission ID already used for another upload")
            else:
                await conn.execute(
                    "INSERT INTO supervised_jobs(id,instructions) VALUES($1,$2)",
                    job_id,
                    upload.description,
                )
                await conn.execute(
                    "INSERT INTO simulation_artifacts(job_id,digest,content) VALUES($1,$2,$3)",
                    job_id,
                    digest,
                    content,
                )
                await conn.execute(
                    "INSERT INTO tasks(id,spec,state) VALUES($1,$2,'running')",
                    job_id,
                    spec.model_dump(mode="json"),
                )
                await conn.execute(
                    """INSERT INTO hosted_services(job_id,submission_hash,bundle_hash,description,config,phase,message,expires_at)
                    VALUES($1,$2,$3,$4,$5,$6,$7,$8)""",
                    job_id,
                    signature,
                    digest,
                    upload.description,
                    config.model_dump(mode="json"),
                    phase,
                    message,
                    now + timedelta(seconds=config.lifetime_seconds)
                    if config.lifetime_seconds
                    else None,
                )
                await event(
                    conn, "task", job_id, "", "running", phase=phase, execution_mode="service"
                )
        return await self.store.task(job_id)

    async def answer(self, job_id, message):
        from ..preprocessing.store import SimulationStore

        async with self.store.change() as (conn, _):
            row = await conn.fetchrow("SELECT * FROM hosted_services WHERE job_id=$1", job_id)
            if row is None or row["phase"] != "needs_input" or row["desired"] != "running":
                raise Conflict("Service is not waiting for input")
            config = ServiceConfig.model_validate(row["config"]).model_copy(
                update={"entrypoint": message.strip()}
            )
            files = await SimulationStore(self.store).files(job_id, row["bundle_hash"])
            self.validate(config, files)
            await conn.execute(
                "UPDATE hosted_services SET config=$2,phase='pending',message='Waiting for a compatible service worker.',revision=revision+1 WHERE job_id=$1",
                job_id,
                config.model_dump(mode="json"),
            )
        return {"saved": True}

    async def status(self, job_id):
        row = await self.pool.fetchrow("SELECT * FROM hosted_services WHERE job_id=$1", job_id)
        if row is None:
            raise NotFound("Service not found")
        tasks = await self.pool.fetch(
            "SELECT id,state,generation,worker_id,failure,progress,'service' AS role FROM tasks WHERE spec->>'job_id'=$1 AND id!=$1 ORDER BY created_at",
            job_id,
        )
        return dict(
            job_id=job_id,
            execution_mode="service",
            workload="python",
            phase=row["phase"],
            description=row["description"],
            message=row["message"],
            deadline=row["expires_at"],
            original_hash=row["bundle_hash"],
            round=row["attempts"],
            limits={
                "adaptations": 0,
                "workers": 1,
                "runtime_seconds": row["config"].get("lifetime_seconds"),
            },
            tasks=[dict(t) for t in tasks],
            workers=[t["worker_id"] for t in tasks if t["id"] == row["task_id"] and t["worker_id"]],
            checks=[],
            versions=[],
            question=row["message"] if row["phase"] == "needs_input" else None,
            service={
                "endpoint": f"/serve/{job_id}",
                "task_id": row["task_id"],
                "desired": row["desired"],
                "config": row["config"],
                "ready_at": row["ready_at"],
                "health_at": row["health_at"],
                "restarts": max(0, row["attempts"] - 1),
            },
        )

    async def transition(self, conn, row, phase, message):
        if row["phase"] == phase and row["message"] == message:
            return
        await conn.execute(
            "UPDATE hosted_services SET phase=$2,message=$3,revision=revision+1 WHERE job_id=$1",
            row["job_id"],
            phase,
            message,
        )
        previous = await conn.fetchval("SELECT state FROM tasks WHERE id=$1", row["job_id"])
        state = {"stopped": "cancelled", "failed": "failed"}.get(phase, "running")
        await conn.execute(
            "UPDATE tasks SET state=$3,spec=jsonb_set(spec,'{payload,phase}',to_jsonb($2::text)) WHERE id=$1",
            row["job_id"],
            phase,
            state,
        )
        await event(
            conn,
            "task",
            row["job_id"],
            previous,
            state,
            job_id=row["job_id"],
            phase=phase,
            message=message,
        )

    async def action(self, job_id, action):
        request = action.model_dump(mode="json")
        async with self.store.change() as (conn, now):
            old = await conn.fetchrow(
                "SELECT request,result FROM supervisor_actions WHERE job_id=$1 AND action_id=$2",
                job_id,
                action.action_id,
            )
            if old:
                if old["request"] != request:
                    raise Conflict("Action ID already used")
                return old["result"]
            row = await conn.fetchrow("SELECT * FROM hosted_services WHERE job_id=$1", job_id)
            if row is None:
                raise NotFound("Service not found")
            if (
                action.operation == "restart"
                and row["expires_at"] is not None
                and row["expires_at"] <= now
            ):
                raise Conflict("Service lifetime has expired; submit a new service")
            await cancel_job_tasks(conn, job_id, "Service " + action.operation, exclude_root=True)
            phase = (
                "stopped"
                if action.operation == "stop"
                else ("pending" if row["config"].get("entrypoint") else "needs_input")
            )
            await conn.execute(
                """UPDATE hosted_services SET desired=$2,task_id=NULL,failures='[]',health_at=NULL,ready_at=NULL,
                retry_after=clock_timestamp() WHERE job_id=$1""",
                job_id,
                "stopped" if action.operation == "stop" else "running",
            )
            await self.transition(
                conn,
                row,
                phase,
                "Service stopped." if phase == "stopped" else "Service restart requested.",
            )
            state = "cancelled" if action.operation == "stop" else "active"
            await conn.execute(
                "UPDATE supervised_jobs SET state=$2,finalized=false,active_run=NULL,revision=revision+1 WHERE id=$1",
                job_id,
                state,
            )
            await conn.execute(
                "UPDATE tasks SET state=$2,failure='' WHERE id=$1",
                job_id,
                "cancelled" if state == "cancelled" else "running",
            )
            await conn.execute("DELETE FROM job_reservations WHERE job_id=$1", job_id)
            result = {"job_id": job_id, "state": phase}
            await conn.execute(
                "INSERT INTO supervisor_actions(job_id,action_id,request,result) VALUES($1,$2,$3,$4)",
                job_id,
                action.action_id,
                request,
                result,
            )
        return result

    async def reconcile(self):
        async with self.store.change(timeout=15) as (conn, now):
            rows = await conn.fetch(
                "SELECT h.*,j.state AS supervisor_state FROM hosted_services h JOIN supervised_jobs j ON j.id=h.job_id WHERE h.desired='running'"
            )
            for row in rows:
                job_id = row["job_id"]
                if row["supervisor_state"] == "cancelled" or (
                    row["expires_at"] and row["expires_at"] <= now
                ):
                    await cancel_job_tasks(conn, job_id, "Service stopped or expired")
                    await conn.execute(
                        "UPDATE hosted_services SET desired='stopped',health_at=NULL WHERE job_id=$1",
                        job_id,
                    )
                    await conn.execute(
                        "UPDATE supervised_jobs SET state='cancelled',active_run=NULL,revision=revision+1 WHERE id=$1",
                        job_id,
                    )
                    await self.transition(
                        conn, row, "stopped", "Service stopped or its lifetime expired."
                    )
                    continue
                if row["supervisor_state"] != "active" or row["phase"] in {"needs_input", "failed"}:
                    continue
                config = ServiceConfig.model_validate(row["config"])
                task = (
                    await conn.fetchrow("SELECT * FROM tasks WHERE id=$1", row["task_id"])
                    if row["task_id"]
                    else None
                )
                if task and task["state"] in {"queued", "assigned", "running"}:
                    if task["state"] == "running" and row["phase"] == "pending":
                        await self.transition(
                            conn,
                            row,
                            "starting",
                            "Starting the HTTP service and checking readiness.",
                        )
                    # Startup includes downloads and model loading. Heartbeats alone cannot keep a hung startup alive.
                    if (
                        task["started_at"]
                        and not row["ready_at"]
                        and (now - task["started_at"]).total_seconds()
                        > config.startup_timeout_seconds
                    ):
                        await cancel_job_tasks(
                            conn, job_id, "Service startup timeout", exclude_root=True
                        )
                    continue
                if task:
                    failures = [t for t in row["failures"] if t > now.timestamp() - 600]
                    failures.append(now.timestamp())
                    phase = "failed" if len(failures) >= 5 else "restarting"
                    await conn.execute(
                        """UPDATE hosted_services SET task_id=NULL,failures=$2,health_at=NULL,ready_at=NULL,
                        retry_after=$3 WHERE job_id=$1""",
                        job_id,
                        failures,
                        now + timedelta(seconds=min(60, 2 ** (len(failures) - 1))),
                    )
                    await self.transition(
                        conn,
                        row,
                        phase,
                        task["failure"]
                        or "Service process stopped; replacing its worker assignment.",
                    )
                    if phase == "failed":
                        await conn.execute(
                            "UPDATE tasks SET state='failed',failure=$2 WHERE id=$1",
                            job_id,
                            "Service crash-loop limit reached",
                        )
                        await conn.execute(
                            "UPDATE supervised_jobs SET state='failed',active_run=NULL,revision=revision+1 WHERE id=$1",
                            job_id,
                        )
                    continue
                if row["retry_after"] > now:
                    continue
                # The normal scheduler owns atomic worker placement and capacity. It waits when none fit.
                attempt_id = job_id + "-" + uuid4().hex[:12]
                spec = TaskSpec(
                    id=attempt_id,
                    job_id=job_id,
                    kind="python_service",
                    requirements=config.requirements,
                    max_attempts=1,
                    timeout_seconds=None,
                    payload={
                        "role": "service",
                        "bundle_hash": row["bundle_hash"],
                        "artifact_token": secrets.token_urlsafe(32),
                        "service": config.model_dump(mode="json"),
                    },
                )
                await conn.execute(
                    "INSERT INTO tasks(id,spec,state) VALUES($1,$2,'queued')",
                    attempt_id,
                    spec.model_dump(mode="json"),
                )
                await conn.execute(
                    "UPDATE hosted_services SET task_id=$2,attempts=attempts+1,health_at=NULL,ready_at=NULL WHERE job_id=$1",
                    job_id,
                    attempt_id,
                )
                await self.transition(
                    conn, row, "pending", "Waiting for a compatible service worker."
                )
                await event(conn, "task", attempt_id, "", "queued", job_id=job_id)

    async def active(self, task_id, generation=None, session=None):
        row = await self.pool.fetchrow(
            """SELECT t.*,h.config,h.desired,h.health_at,h.phase,h.expires_at,w.state AS worker_state,w.last_seen
            FROM tasks t JOIN hosted_services h ON h.task_id=t.id JOIN workers w ON w.id=t.worker_id
            JOIN supervised_jobs j ON j.id=h.job_id
            WHERE t.id=$1 AND t.state='running' AND t.lease_until>clock_timestamp()
            AND t.session_id=w.session_id AND h.desired='running' AND j.state='active'
            AND (h.expires_at IS NULL OR h.expires_at>clock_timestamp())
            AND w.state='alive' AND w.last_seen>clock_timestamp()-interval '15 seconds' """,
            task_id,
        )
        if (
            row
            and (generation is None or row["generation"] == generation)
            and (session is None or row["session_id"] == session)
        ):
            return row
        return None

    async def health(self, task_id, generation, session, ready):
        # Routine freshness reports must not wait behind unrelated scheduler mutations.
        row = await self.pool.fetchrow(
            """UPDATE hosted_services h SET
                health_at=CASE WHEN $4 THEN clock_timestamp() ELSE NULL END,
                ready_at=CASE WHEN $4 THEN COALESCE(ready_at,clock_timestamp()) ELSE ready_at END
            WHERE h.task_id=$1 AND h.desired='running'
            AND EXISTS(SELECT 1 FROM tasks t JOIN workers w ON w.id=t.worker_id
                JOIN supervised_jobs j ON j.id=h.job_id
                WHERE t.id=$1 AND t.generation=$2 AND t.session_id=$3
                AND w.session_id=$3 AND t.state='running' AND j.state='active'
                AND t.lease_until>clock_timestamp())
            AND (h.expires_at IS NULL OR h.expires_at>clock_timestamp())
            RETURNING h.*""",
            task_id,
            generation,
            session,
            ready,
        )
        if row is None:
            raise Conflict("Service assignment was revoked")
        phase = "ready" if ready else "starting"
        if row["phase"] == phase:
            return
        try:
            async with self.store.change(timeout=1) as (conn, _):
                # Recheck under the mutation lock: Stop/Restart may have won since the report.
                fresh = await conn.fetchrow(
                    "SELECT * FROM hosted_services WHERE task_id=$1 AND desired='running'", task_id
                )
                if (
                    fresh is not None
                    and (fresh["health_at"] is not None) == ready
                    and await self.active(task_id, generation, session)
                ):
                    await self.transition(
                        conn,
                        fresh,
                        phase,
                        "Service is ready."
                        if ready
                        else "Service is not ready; requests are paused.",
                    )
        except TimeoutError:
            # Freshness is already durable. Retry the UI/audit transition on the next report.
            pass
