import hashlib
from datetime import timedelta

from ..server.db.store import Conflict, NotFound, cancel_job_tasks, event
from ..server.usage import enforce_usage_caps
from ..shared.protocol import TaskSpec, json_loads, json_text
from .artifacts import bundle
from .models import CODE_KIND, ROOT_KIND, TERMINAL


class SimulationStore:
    def __init__(self, store):
        self.store, self.pool = store, store.pool

    async def create(self, upload, files, *, account_id=None):
        job_id = "sim-" + upload.request_id.hex
        digest, content = bundle(files)
        # Omit optional defaults introduced after existing submissions were hashed.
        excluded = {"execution_mode", "service"}
        if upload.execution_mode == "auto":
            excluded.remove("execution_mode")
        if upload.usage_cap is None:
            excluded.add("usage_cap")
        signature_data = upload.model_dump(mode="json", exclude=excluded)
        signature = hashlib.sha256(json_text(signature_data).encode()).hexdigest()

        data = {
            "planning_version": 2
            if upload.workload == "simulation" and upload.execution_mode != "auto"
            else 3,
            "execution_mode": upload.execution_mode,
            "workload": upload.workload,
            "description": upload.description,
            "limits": {
                "adaptations": upload.max_adaptations,
                "runtime_seconds": upload.max_runtime_seconds,
                "workers": upload.max_workers,
            },
            "round": 0,
            "versions": [],
            "checks": [],
            "tasks": [],
            "workers": [],
            "message": "Waiting for a preprocessing worker.",
            "model_failures": 0,
        }
        spec = TaskSpec(
            id=job_id,
            job_id=job_id,
            kind=ROOT_KIND,
            payload={"value": {"label": next(iter(files)).split("/")[-1]}, "phase": "submitted"},
            requirements={"runtime": "cpu", "vram_mib": 0},
            max_attempts=1,
            timeout_seconds=upload.max_runtime_seconds,
        )
        async with self.store.change() as (conn, now):
            hosted = await conn.fetchval(
                "SELECT submission_hash FROM hosted_services WHERE job_id=$1", job_id
            )
            if hosted:
                if hosted != signature:
                    raise Conflict("Submission ID already used for another upload")
                return await self.store.task(job_id)
            old = await conn.fetchrow(
                "SELECT submission_hash FROM simulation_jobs WHERE job_id=$1", job_id
            )
            if old:
                if old["submission_hash"] != signature:
                    raise Conflict("Submission ID already used for another upload")
                return await self.store.task(job_id)
            await conn.execute(
                "INSERT INTO supervised_jobs(id,instructions,usage_cap_cad,billing_account_id) VALUES($1,$2,$3,$4)",
                job_id,
                upload.description,
                upload.usage_cap,
                account_id,
            )
            await conn.execute(
                "INSERT INTO simulation_artifacts(job_id,digest,content) VALUES($1,$2,$3)",
                job_id,
                digest,
                content,
            )
            await conn.execute(
                "INSERT INTO tasks(id,spec,state) VALUES($1,$2,'queued')",
                job_id,
                spec.model_dump(mode="json"),
            )
            await conn.execute(
                "INSERT INTO simulation_jobs(job_id,submission_hash,original_hash,data,deadline) VALUES($1,$2,$3,$4,$5)",
                job_id,
                signature,
                digest,
                data,
                now + timedelta(seconds=upload.max_runtime_seconds),
            )
            await event(
                conn, "task", job_id, "", "queued", phase="submitted", message=data["message"]
            )
            if upload.usage_cap is not None:
                await enforce_usage_caps(conn, job_id=job_id)
        return await self.store.task(job_id)

    async def job(self, job_id):
        row = await self.pool.fetchrow("SELECT * FROM simulation_jobs WHERE job_id=$1", job_id)
        if row is None:
            raise NotFound("Uploaded simulation not found")
        return dict(row)

    async def files(self, job_id, digest):
        content = await self.pool.fetchval(
            "SELECT content FROM simulation_artifacts WHERE job_id=$1 AND digest=$2", job_id, digest
        )
        if content is None:
            raise NotFound("Artifact not found")
        return json_loads(content)["files"]

    async def eligible(self, job_id, conn=None):
        return [
            dict(row)
            for row in await (conn or self.pool).fetch(
                """SELECT w.id,w.capabilities,w.last_seen FROM workers w
            WHERE w.state='alive' AND NOT w.paused AND w.last_seen>clock_timestamp()-interval '15 seconds'
            AND w.capabilities->'kinds' ? $2
            AND w.capabilities->>'runtime'='cpu'
            AND (w.capabilities->'kinds' ? 'python_program' OR NOT EXISTS(
                SELECT 1 FROM simulation_jobs p WHERE p.job_id=$1
                AND p.data->>'planning_version'='3' AND p.data->>'workload'!='auto'))
            AND NOT EXISTS(SELECT 1 FROM job_reservations r WHERE r.worker_id=w.id AND r.job_id!=$1 AND r.expires_at>clock_timestamp())
            AND NOT EXISTS(SELECT 1 FROM tasks t WHERE t.worker_id=w.id AND t.state IN ('assigned','running') AND t.spec->>'job_id'!=$1)
            ORDER BY w.id""",
                job_id,
                CODE_KIND,
            )
        ]

    async def reserve(self, job, count, worker_ids=None):
        async with self.store.change() as (conn, _):
            fresh = await conn.fetchrow(
                "SELECT s.state,p.revision FROM supervised_jobs s JOIN simulation_jobs p ON p.job_id=s.id WHERE s.id=$1",
                job["job_id"],
            )
            if fresh["state"] != "active" or fresh["revision"] != job["revision"]:
                raise Conflict("Job changed while allocating workers")
            available = await self.eligible(job["job_id"], conn)
            ids = {w["id"] for w in available}
            selected = [w for w in job["data"].get("workers", []) if w in ids]
            preferred = job["data"].get("plan", {}).get("preferred_worker")
            for worker in sorted(available, key=lambda w: (w["id"] != preferred, w["id"])):
                if worker["id"] not in selected:
                    selected.append(worker["id"])
            selected = (
                selected[:count] if worker_ids is None else [w for w in worker_ids if w in ids]
            )
            if worker_ids is not None and len(selected) != len(worker_ids):
                return []  # Never silently substitute the agent's placement.
            if len(selected) > job["data"]["limits"]["workers"]:
                raise Conflict("Placement exceeds the submission worker budget")
            # While waiting for the second machine keep the original reservation.
            await conn.execute(
                """INSERT INTO job_reservations(worker_id,job_id,expires_at)
                SELECT worker,$2,clock_timestamp()+interval '5 minutes' FROM unnest($1::text[]) AS worker
                ON CONFLICT(worker_id) DO UPDATE SET job_id=EXCLUDED.job_id,expires_at=EXCLUDED.expires_at
                WHERE job_reservations.job_id!=EXCLUDED.job_id
                    OR job_reservations.expires_at<clock_timestamp()+interval '1 minute'""",
                selected,
                job["job_id"],
            )
            await conn.execute(
                "DELETE FROM job_reservations WHERE job_id=$1 AND NOT(worker_id=ANY($2::text[]))",
                job["job_id"],
                selected,
            )
            return selected

    async def save(self, job, phase, message, *, tasks=(), artifacts=(), result=None):
        data = job["data"]
        data["message"] = message
        terminal = phase in TERMINAL
        async with self.store.change() as (conn, now):
            if not terminal and job["deadline"] <= now:
                raise Conflict("Submission deadline elapsed; discard late proposal")
            current = await conn.fetchrow(
                "SELECT p.revision,p.deadline,s.state,t.state AS task_state FROM simulation_jobs p JOIN supervised_jobs s ON s.id=p.job_id JOIN tasks t ON t.id=p.job_id WHERE p.job_id=$1",
                job["job_id"],
            )
            if not terminal and current["deadline"] <= now:
                raise Conflict("Submission deadline elapsed; discard late proposal")
            if current["revision"] != job["revision"] or (
                (
                    (
                        current["state"] != "active"
                        and not (phase == "failed" and current["state"] == "paused")
                    )
                    or current["task_state"] == "cancelled"
                )
                and phase != "cancelled"
            ):
                raise Conflict("Job changed while preprocessing; discard stale work")
            for digest, content in artifacts:
                await conn.execute(
                    "INSERT INTO simulation_artifacts(job_id,digest,content) VALUES($1,$2,$3) ON CONFLICT DO NOTHING",
                    job["job_id"],
                    digest,
                    content,
                )
            if tasks:
                specs = [task.model_dump(mode="json") for task in tasks]
                await conn.execute(
                    "INSERT INTO tasks(id,spec,state) SELECT spec->>'id',spec,'queued' FROM jsonb_array_elements($1::jsonb) AS spec ON CONFLICT DO NOTHING",
                    specs,
                )
                await conn.execute(
                    """WITH added AS (
                    INSERT INTO events(entity,entity_id,previous_state,new_state,details)
                    SELECT 'task',id,'','queued',jsonb_build_object('phase',$2::text,'job_id',$3::text)
                    FROM tasks WHERE id=ANY($1::text[]) RETURNING id,entity_id,at,details)
                    INSERT INTO execution_events(task_id,attempt,worker_id,source,sequence,kind,occurred_at,data)
                    SELECT entity_id,0,NULL,'server',id,'queued',at,details FROM added""",
                    [task.id for task in tasks],
                    phase,
                    job["job_id"],
                )
            await conn.execute(
                "UPDATE simulation_jobs SET phase=$2,data=$3,revision=revision+1,retry_after=clock_timestamp()+interval '1 second' WHERE job_id=$1",
                job["job_id"],
                phase,
                data,
            )
            state = {"completed": "succeeded", "failed": "failed", "cancelled": "cancelled"}.get(
                phase, "running"
            )
            old = await conn.fetchval("SELECT state FROM tasks WHERE id=$1", job["job_id"])
            progress = {
                "submitted": 0,
                "inspecting": 5,
                "preparing": 5,
                "profiling": 15,
                "planning": 18,
                "sampling": 24,
                "checking_aggregation": 45,
                "scheduling": 50,
                "aggregation_wait": 94,
                "original": 10,
                "adapting": 20,
                "reference": 25,
                "testing": 30,
                "distributed_reference": 40,
                "distributed_validation": 45,
                "allocating": 50,
                "running": 60,
                "aggregating": 95,
                "completed": 100,
                "program_planning": 5,
                "program_preparing": 10,
                "program_probe": 20,
                "program_placement": 35,
                "program_ready": 40,
                "program_running": 60,
            }.get(phase, 0)
            await conn.execute(
                """UPDATE tasks SET state=$2,progress=$3,result=$4,failure=$5,
                spec=jsonb_set(spec,'{payload,phase}',to_jsonb($6::text)) WHERE id=$1""",
                job["job_id"],
                state,
                progress,
                result,
                message if phase == "failed" else "",
                phase,
            )
            await event(conn, "task", job["job_id"], old, state, phase=phase, message=message)
            if terminal:
                await cancel_job_tasks(conn, job["job_id"], message, exclude_root=True)
                await conn.execute("DELETE FROM job_reservations WHERE job_id=$1", job["job_id"])
                await conn.execute(
                    "UPDATE supervised_jobs SET state=$2,active_run=NULL,revision=revision+1 WHERE id=$1",
                    job["job_id"],
                    state,
                )
            elif phase in {"adapting", "needs_input"}:
                keep = data.get("workers", [])[:1] if phase == "adapting" else []
                await conn.execute(
                    "DELETE FROM job_reservations WHERE job_id=$1 AND NOT(worker_id=ANY($2::text[]))",
                    job["job_id"],
                    keep,
                )

    async def status(self, job_id):
        job = await self.job(job_id)
        data = job["data"]
        tasks = await self.pool.fetch(
            "SELECT id,state,generation,worker_id,failure,progress,spec->'payload'->>'role' AS role FROM tasks WHERE spec->>'job_id'=$1 AND id!=$1 ORDER BY created_at,id",
            job_id,
        )
        cleanup = await self.pool.fetch(
            """SELECT e.task_id,e.attempt,e.worker_id,
                EXISTS(SELECT 1 FROM execution_events c WHERE c.task_id=e.task_id
                    AND c.attempt=e.attempt AND c.worker_id=e.worker_id AND c.source='worker'
                    AND c.kind='cleaned' AND c.data->>'workspace_removed'='true'
                    AND c.data->>'processes_stopped'='true') AS confirmed
            FROM execution_events e JOIN tasks t ON t.id=e.task_id
            WHERE t.spec->>'job_id'=$1 AND e.source='server' AND e.kind='cancelled'
                AND e.data->>'cleanup_required'='true'""",
            job_id,
        )
        counts = await self.pool.fetch(
            """SELECT state,sum(COALESCE((spec->'payload'->'seed_range'->>'count')::bigint,jsonb_array_length(spec->'payload'->'seeds'))) AS trials
            FROM tasks WHERE spec->>'job_id'=$1 AND spec->'payload'->>'role' LIKE 'batch-%'
            GROUP BY state""",
            job_id,
        )
        return {
            "job_id": job_id,
            "phase": job["phase"],
            "trial_counts": {r["state"]: r["trials"] for r in counts},
            "cleanup": {
                "required": len(cleanup),
                "confirmed": sum(r["confirmed"] for r in cleanup),
                "pending_workers": sorted({r["worker_id"] for r in cleanup if not r["confirmed"]}),
            },
            "deadline": job["deadline"],
            "original_hash": job["original_hash"],
            "description": data["description"],
            "workload": data.get("workload", "simulation"),
            "program_plan": data.get("program_plan"),
            "message": data["message"],
            "round": data["round"],
            "limits": data["limits"],
            "plan": data.get("plan"),
            "policy": data.get("policy"),
            "measurements": data.get("measurements", []),
            "decisions": data.get("decisions", []),
            "schedule": data.get("schedule"),
            "versions": data["versions"],
            "checks": data["checks"],
            "workers": data["workers"],
            "question": data.get("question"),
            "tasks": [dict(t) for t in tasks],
            "validated_hash": data.get("validated_hash"),
        }
