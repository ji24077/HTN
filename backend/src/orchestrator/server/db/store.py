"""PostgreSQL owns task state, capacity, leases, and transactional audit events."""

import asyncio
import hashlib
import hmac
import logging
import math
import re
import secrets
from collections.abc import AsyncIterator
from contextlib import asynccontextmanager
from datetime import UTC, datetime, timedelta
from importlib.resources import files
from uuid import UUID, uuid4

import asyncpg
from pydantic import JsonValue, ValidationError

from ...shared.dwp import validate_public_key
from ...shared.execution import ExecutionBatch, rejected_ack, scrub_execution
from ...shared.protocol import (
    ACK_SECONDS,
    LEASE_SECONDS,
    UNHEALTHY_AFTER,
    Capabilities,
    Ref,
    Submission,
    Task,
    TaskSpec,
    Worker,
    bounded_json,
    json_loads,
    json_text,
    task_ref,
)
from .connection import connection_options

# A reservation older than this never receives its provider key; the reconciler closes it.
PENDING_ENROLLMENT_SECONDS = 300


class Conflict(Exception):
    pass


class StaleAssignment(Exception):
    pass


class StaleSession(Exception):
    pass


class NotFound(Exception):
    pass


class EnrollmentLimit(Exception):
    pass


log = logging.getLogger(__name__)


def task_from_row(row: asyncpg.Record) -> Task:
    values = dict(row)
    values.pop("id")
    return Task.model_validate(values)


async def configure_connection(conn: asyncpg.Connection) -> None:
    for typename in ("json", "jsonb"):
        await conn.set_type_codec(
            typename, schema="pg_catalog", encoder=json_text, decoder=json_loads, format="text"
        )


async def event(
    conn,
    entity: str,
    entity_id: str,
    before: str,
    after: str,
    *,
    spec: dict | None = None,
    **details,
) -> None:
    audit_id = await conn.fetchval(
        """INSERT INTO events(entity,entity_id,previous_state,new_state,details)
           VALUES($1,$2,$3,$4,$5) RETURNING id""",
        entity,
        entity_id,
        before,
        after,
        details,
    )
    if entity == "task":
        if spec is None:
            spec = await conn.fetchval("SELECT spec FROM tasks WHERE id=$1", entity_id)
        data = {
            **details,
            "state": after,
            "adapter": spec["kind"],
            "job_id": spec["job_id"],
            "runtime": spec["requirements"]["runtime"],
            "task_hash": hashlib.sha256(json_text(spec).encode()).hexdigest(),
        }
        data.pop("session_id", None)
        if after == "succeeded":
            data["result_url"] = f"/v1/tasks/{entity_id}"
        kind = "failed" if after == "queued" and before != "" else after
        await conn.execute(
            """INSERT INTO execution_events
               (task_id,attempt,worker_id,source,sequence,kind,occurred_at,data)
               VALUES($1,$2,$3,'server',$4,$5,clock_timestamp(),$6)""",
            entity_id,
            details.get("generation", 0),
            details.get("worker_id"),
            audit_id,
            kind,
            scrub_execution(data),
        )
        await conn.execute(
            """INSERT INTO supervisor_events(job_id,kind,data)
               SELECT id,$2,$3 FROM supervised_jobs WHERE id=$1""",
            spec["job_id"],
            "task_" + kind,
            scrub_execution({"task_id": entity_id, **data}),
        )
    elif entity == "worker":
        await conn.execute(
            """INSERT INTO supervisor_events(job_id,kind,data)
               SELECT j.id,'worker_health',$2::jsonb FROM supervised_jobs j
               WHERE EXISTS(SELECT 1 FROM tasks t WHERE t.spec->>'job_id'=j.id
                   AND t.worker_id=$1 AND t.state IN ('assigned','running'))
               OR EXISTS(SELECT 1 FROM job_reservations r WHERE r.job_id=j.id
                   AND r.worker_id=$1 AND r.expires_at>clock_timestamp())""",
            entity_id,
            {"worker_id": entity_id, "state": after},
        )


async def ingest_execution_events(
    store, worker_id: str, session: str, payload: object
) -> dict | None:
    """Store a device's diagnostics batch and build its acknowledgement.

    Diagnostics are best-effort: a batch this server cannot store must never
    close the connection carrying the task itself. Returns None when the device
    should simply retry the batch after its resend window.
    """
    try:
        batch = ExecutionBatch.model_validate(payload)
    except ValidationError as exc:
        log.warning("dropping malformed execution batch worker=%s: %s", worker_id, exc)
        return rejected_ack(payload)
    try:
        sequences = await store.append_execution_events(worker_id, session, batch)
        rejected = False
    except StaleAssignment:
        sequences, rejected = [item.sequence for item in batch.events], True
    except (TimeoutError, asyncpg.PostgresError) as exc:
        log.warning(
            "deferring execution batch worker=%s task=%s attempt=%s: %s",
            worker_id,
            batch.taskId,
            batch.attempt,
            exc,
        )
        return None
    return {
        "taskId": batch.taskId,
        "attempt": batch.attempt,
        "sequences": sequences,
        "rejected": rejected,
    }


class Store:
    def __init__(self, pool: asyncpg.Pool):
        self.pool = pool

    @classmethod
    async def open(cls, url: str, *, schema: str = "public") -> "Store":
        if not re.fullmatch(r"[a-z][a-z0-9_]{0,62}", schema):
            raise ValueError("Invalid database schema")
        pool = await asyncpg.create_pool(
            url,
            min_size=1,
            max_size=10,
            timeout=10,
            command_timeout=5,
            init=configure_connection,
            server_settings={"search_path": f'"{schema}"'},
            **connection_options(url),
        )
        store = cls(pool)
        try:
            # Remote schema setup makes many round trips and may wait for the
            # other listener's startup lock; keep normal mutations at 5 seconds.
            async with store.change(timeout=60) as (conn, _):
                if schema != "public":
                    await conn.execute(f'CREATE SCHEMA IF NOT EXISTS "{schema}"')
                    await conn.execute(f'REVOKE ALL ON SCHEMA "{schema}" FROM PUBLIC')
                await conn.execute(
                    files("orchestrator.server.db").joinpath("schema.sql").read_text()
                )
                if schema != "public":
                    # Supabase browser roles must not bypass scheduler transactions.
                    roles = ["PUBLIC"] + [
                        row["rolname"]
                        for row in await conn.fetch(
                            "SELECT rolname FROM pg_roles WHERE rolname IN ('anon', 'authenticated')"
                        )
                    ]
                    for role in roles:
                        for kind in ("TABLES", "SEQUENCES", "FUNCTIONS"):
                            await conn.execute(
                                f'REVOKE ALL ON ALL {kind} IN SCHEMA "{schema}" FROM {role}'
                            )
                            await conn.execute(
                                f'ALTER DEFAULT PRIVILEGES IN SCHEMA "{schema}" '
                                f"REVOKE ALL ON {kind} FROM {role}"
                            )
                        await conn.execute(f'REVOKE ALL ON SCHEMA "{schema}" FROM {role}')
                    for table in (
                        "workers",
                        "tasks",
                        "events",
                        "worker_enrollments",
                        "dwp_pair_codes",
                        "dwp_devices",
                        "dwp_assertions",
                        "chat_conversations",
                        "execution_events",
                        "supervised_jobs",
                        "supervisor_events",
                        "supervisor_runs",
                        "supervisor_actions",
                        "job_reservations",
                    ):
                        await conn.execute(
                            f'ALTER TABLE "{schema}".{table} ENABLE ROW LEVEL SECURITY'
                        )
        except BaseException:
            await store.close()
            raise
        return store

    async def close(self) -> None:
        try:
            async with asyncio.timeout(5):
                await self.pool.close()
        except TimeoutError:
            self.pool.terminate()

    @asynccontextmanager
    async def change(
        self, *, timeout: float = 5
    ) -> AsyncIterator[tuple[asyncpg.Connection, datetime]]:
        # A coarse database lock deliberately serializes prototype mutations,
        # including across processes. Move to finer row locks before scaling.
        async with asyncio.timeout(timeout):
            async with self.pool.acquire() as conn:
                async with conn.transaction():
                    await conn.execute("SELECT pg_advisory_xact_lock(71420931)", timeout=timeout)
                    now = await conn.fetchval("SELECT clock_timestamp()")
                    yield conn, now

    async def task(self, task_id: str) -> Task:
        row = await self.pool.fetchrow("SELECT * FROM tasks WHERE id=$1", task_id)
        if row is None:
            raise NotFound("task not found")
        return task_from_row(row)

    async def tasks(self) -> list[Task]:
        rows = await self.pool.fetch("SELECT * FROM tasks ORDER BY created_at DESC,id LIMIT 500")
        return [task_from_row(row) for row in rows]

    async def execution_events(
        self, task_id: str, after: int = 0, worker_id: str | None = None, attempt: int | None = None
    ):
        await self.task(task_id)
        rows = await self.pool.fetch(
            """SELECT *, task_id || ':' || attempt::text AS execution_id
               FROM execution_events WHERE task_id=$1 AND id>$2
               AND ($3::text IS NULL OR worker_id=$3)
               AND ($4::int IS NULL OR attempt=$4) ORDER BY id LIMIT 200""",
            task_id,
            after,
            worker_id,
            attempt,
        )
        return [dict(row) for row in rows]

    async def append_execution_events(self, worker_id: str, session: str, batch: ExecutionBatch):
        async with self.change() as (conn, _):
            await self._worker(conn, worker_id, session)
            # Replay may arrive after cancellation, completion, or a newer attempt.
            # Historical assignment ownership permits diagnostics, never task mutation.
            owned = await conn.fetchval(
                """SELECT EXISTS(SELECT 1 FROM events WHERE entity='task' AND entity_id=$1
                   AND new_state='assigned' AND details->>'worker_id'=$2
                   AND (details->>'generation')::int=$3)""",
                batch.taskId,
                worker_id,
                batch.attempt,
            )
            if not owned:
                raise StaleAssignment("execution was not assigned to this worker")
            # Everything here runs under the global lock; keep it to two round trips.
            await conn.executemany(
                """INSERT INTO execution_events
                   (task_id,attempt,worker_id,source,sequence,kind,occurred_at,data)
                   VALUES($1,$2,$3,'worker',$4,$5,$6,$7)
                   ON CONFLICT(task_id,attempt,source,sequence) DO NOTHING""",
                [
                    (
                        batch.taskId,
                        batch.attempt,
                        worker_id,
                        item.sequence,
                        item.kind,
                        item.at,
                        scrub_execution(item.data),
                    )
                    for item in batch.events
                ],
            )
            percents = [
                float(percent)
                for item in batch.events
                if item.kind == "progress"
                and isinstance(percent := item.data.get("percent"), int | float)
                and not isinstance(percent, bool)
                and math.isfinite(percent)
                and 0 <= percent <= 100
            ]
            if percents:
                await conn.execute(
                    """UPDATE tasks SET progress=GREATEST(progress,$4)
                       WHERE id=$1 AND worker_id=$2 AND generation=$3 AND state='running'""",
                    batch.taskId,
                    worker_id,
                    batch.attempt,
                    max(percents),
                )
            return [item.sequence for item in batch.events]

    async def workers(self) -> list[Worker]:
        rows = await self.pool.fetch("SELECT * FROM workers ORDER BY id LIMIT 500")
        return [Worker.model_validate(dict(row)) for row in rows]

    async def enrolled_worker(self, worker_id: str) -> bool:
        return await self.pool.fetchval(
            """SELECT EXISTS(
                SELECT 1 FROM worker_enrollments WHERE worker_id=$1 AND state='active'
                UNION ALL
                SELECT 1 FROM dwp_devices WHERE worker_id=$1 AND revoked_at IS NULL
            )""",
            worker_id,
        )

    async def create_pair_code(self, owner_id: str | None) -> str:
        owner = UUID(owner_id) if owner_id and owner_id != "local-admin" else None
        code = secrets.token_hex(16)
        async with self.change() as (conn, now):
            # Keep used/expired codes for one hour to enforce the issuance quota.
            await conn.execute(
                "DELETE FROM dwp_pair_codes WHERE created_at < $1", now - timedelta(hours=1)
            )
            recent = await conn.fetchval(
                "SELECT count(*) FROM dwp_pair_codes WHERE owner_id IS NOT DISTINCT FROM $1::uuid",
                owner,
            )
            active = await conn.fetchval(
                """SELECT count(*) FROM dwp_devices
                   WHERE owner_id IS NOT DISTINCT FROM $1::uuid AND revoked_at IS NULL""",
                owner,
            )
            if recent >= 10 or active >= 100:
                raise EnrollmentLimit
            await conn.execute(
                """INSERT INTO dwp_pair_codes(code_hash,owner_id,created_at,expires_at)
                   VALUES($1,$2,$3,$4)""",
                hashlib.sha256(code.encode()).hexdigest(),
                owner,
                now,
                now + timedelta(minutes=10),
            )
        return code

    async def pair_device(self, code: str, public_key: str, label: str) -> str:
        if not isinstance(code, str) or not re.fullmatch(r"[0-9a-fA-F]{32}", code.strip()):
            raise Conflict("Invalid or expired pairing code")
        public_key = validate_public_key(public_key)
        if not isinstance(label, str) or not 1 <= len(label.strip()) <= 128:
            raise ValueError("Device label must contain 1 to 128 characters")
        worker_id = str(uuid4())
        async with self.change() as (conn, now):
            invite = await conn.fetchrow(
                """SELECT owner_id FROM dwp_pair_codes
                   WHERE code_hash=$1 AND used_at IS NULL AND expires_at>$2 FOR UPDATE""",
                hashlib.sha256(code.strip().lower().encode()).hexdigest(),
                now,
            )
            if invite is None:
                raise Conflict("Invalid or expired pairing code")
            if await conn.fetchval("SELECT 1 FROM dwp_devices WHERE public_key=$1", public_key):
                raise Conflict("Device key is already paired")
            active = await conn.fetchval(
                """SELECT count(*) FROM dwp_devices
                   WHERE owner_id IS NOT DISTINCT FROM $1::uuid AND revoked_at IS NULL""",
                invite["owner_id"],
            )
            if active >= 100:
                raise EnrollmentLimit
            await conn.execute(
                """INSERT INTO dwp_devices(worker_id,owner_id,public_key,display_name)
                   VALUES($1,$2,$3,$4)""",
                worker_id,
                invite["owner_id"],
                public_key,
                label.strip(),
            )
            await conn.execute(
                "UPDATE dwp_pair_codes SET used_at=$2 WHERE code_hash=$1",
                hashlib.sha256(code.strip().lower().encode()).hexdigest(),
                now,
            )
            await event(conn, "enrollment", worker_id, "", "active", protocol="dwp")
        return worker_id

    async def device_key(self, worker_id: str) -> str | None:
        return await self.pool.fetchval(
            "SELECT public_key FROM dwp_devices WHERE worker_id=$1 AND revoked_at IS NULL",
            worker_id,
        )

    async def use_assertion_jti(self, worker_id: str, jti: str, expires_at: float) -> bool:
        try:
            nonce = UUID(jti)
            if type(expires_at) not in (int, float) or not math.isfinite(expires_at):
                return False
            expiry = datetime.fromtimestamp(expires_at, UTC)
        except (ValueError, TypeError, OverflowError, OSError):
            return False
        async with self.change() as (conn, now):
            if not now < expiry <= now + timedelta(minutes=4):
                return False
            await conn.execute("DELETE FROM dwp_assertions WHERE expires_at<=$1", now)
            return bool(
                await conn.fetchval(
                    """INSERT INTO dwp_assertions(worker_id,jti,expires_at)
                   SELECT worker_id,$2,$3 FROM dwp_devices
                   WHERE worker_id=$1 AND revoked_at IS NULL
                   ON CONFLICT(worker_id,jti) DO NOTHING RETURNING true""",
                    worker_id,
                    nonce,
                    expiry,
                )
            )

    async def worker_authorized(self, worker_id: str, header: str | None) -> bool:
        if not header or not header.startswith("Bearer ") or len(header) > 512:
            return False
        digest = await self.pool.fetchval(
            "SELECT token_hash FROM worker_enrollments WHERE worker_id=$1 AND state='active'",
            worker_id,
        )
        return bool(digest) and hmac.compare_digest(
            digest, hashlib.sha256(header[7:].encode()).hexdigest()
        )

    async def reserve_enrollment(self, worker_id, request_id, user_id, name, token):
        async with self.change() as (conn, _):
            if await conn.fetchval(
                "SELECT 1 FROM worker_enrollments WHERE request_id=$1", UUID(request_id)
            ):
                raise Conflict("Enrollment request already used; credentials cannot be replayed")
            recent = await conn.fetchval(
                "SELECT count(*) FROM worker_enrollments WHERE user_id=$1 AND created_at > clock_timestamp()-interval '1 hour'",
                UUID(user_id),
            )
            active = await conn.fetchval(
                "SELECT count(*) FROM worker_enrollments WHERE user_id=$1 AND state IN ('pending','active')",
                UUID(user_id),
            )
            if recent >= 10 or active >= 100:
                raise EnrollmentLimit
            await conn.execute(
                """INSERT INTO worker_enrollments
                   (worker_id,request_id,user_id,display_name,token_hash,state)
                   VALUES($1,$2,$3,$4,$5,'pending')""",
                worker_id,
                UUID(request_id),
                UUID(user_id),
                name,
                hashlib.sha256(token.encode()).hexdigest(),
            )

    async def finish_enrollment(
        self, worker_id: str, key_id: str | None, *, retain_key: str | None = None
    ) -> bool:
        """Activate or fail a pending reservation; False once the reconciler already closed it.

        On failure, `retain_key` records a provider key that could not be revoked so a later
        withdrawal can retry.
        """
        state = "active" if key_id else "failed"
        async with self.change() as (conn, _):
            row = await conn.fetchrow(
                """UPDATE worker_enrollments SET state=$2,tailscale_key_id=$3
                   WHERE worker_id=$1 AND state='pending' RETURNING worker_id""",
                worker_id,
                state,
                key_id or retain_key,
            )
            if row is None:
                if retain_key:
                    # The reconciler already failed this reservation; still record the key
                    # that could not be revoked, without another state transition.
                    await conn.execute(
                        """UPDATE worker_enrollments SET tailscale_key_id=COALESCE(tailscale_key_id,$2)
                           WHERE worker_id=$1 AND state='failed'""",
                        worker_id,
                        retain_key,
                    )
                return False
            await event(conn, "enrollment", worker_id, "pending", state)
            return True

    async def has_active_enrollments(self) -> bool:
        return await self.pool.fetchval(
            """SELECT EXISTS(
                SELECT 1 FROM worker_enrollments WHERE state='active'
                UNION ALL
                SELECT 1 FROM dwp_devices WHERE revoked_at IS NULL
            )"""
        )

    async def cancel_enrollment(self, worker_id: str, user_id: str) -> str | None:
        """Withdraw the caller's unused enrollment; returns the provider key ID to revoke."""
        async with self.change() as (conn, _):
            row = await conn.fetchrow(
                """SELECT state,tailscale_key_id FROM worker_enrollments
                   WHERE worker_id=$1 AND user_id=$2""",
                worker_id,
                UUID(user_id),
            )
            if row is None:
                raise NotFound("enrollment not found")
            if row["state"] == "failed":
                # Already closed; repeating only retries a revocation that failed earlier.
                return row["tailscale_key_id"]
            # A worker that has ever registered is fleet state, not an unused enrollment.
            if await conn.fetchval("SELECT 1 FROM workers WHERE id=$1", worker_id):
                raise Conflict("worker already connected; remove it through fleet management")
            await conn.execute(
                "UPDATE worker_enrollments SET state='failed' WHERE worker_id=$1", worker_id
            )
            await event(conn, "enrollment", worker_id, row["state"], "failed", reason="cancelled")
            return row["tailscale_key_id"]

    async def clear_enrollment_key(self, worker_id: str) -> None:
        """Forget a provider key once its revocation succeeded."""
        await self.pool.execute(
            "UPDATE worker_enrollments SET tailscale_key_id=NULL WHERE worker_id=$1 AND state='failed'",
            worker_id,
        )

    @staticmethod
    async def _expire_enrollments(conn, now: datetime) -> None:
        # One round trip: reconciliation shares a five-second transaction with lease recovery.
        await conn.execute(
            """WITH expired AS (
                   UPDATE worker_enrollments SET state='failed' WHERE worker_id IN (
                       SELECT worker_id FROM worker_enrollments
                       WHERE state='pending' AND created_at <= $1 LIMIT 100)
                   RETURNING worker_id)
               INSERT INTO events(entity,entity_id,previous_state,new_state,details)
               SELECT 'enrollment', worker_id, 'pending', 'failed', $2::jsonb FROM expired""",
            now - timedelta(seconds=PENDING_ENROLLMENT_SECONDS),
            {"reason": "expired"},
        )

    async def events(self, after: int) -> list[dict]:
        rows = await self.pool.fetch(
            "SELECT * FROM events WHERE id>$1 ORDER BY id LIMIT 500", after
        )
        return [dict(row) for row in rows]

    async def submit(self, specs: list[TaskSpec], *, instructions: str | None = None) -> list[Task]:
        # Validate the in-process planner boundary as well as the HTTP boundary.
        specs = Submission(tasks=specs, instructions=instructions).tasks
        if instructions is not None and len({spec.job_id for spec in specs}) != 1:
            raise Conflict("instructions apply to a single submitted job")
        result = []
        async with self.change() as (conn, _):
            for spec in specs:
                encoded = spec.model_dump(mode="json")
                row = await conn.fetchrow("SELECT * FROM tasks WHERE id=$1", spec.id)
                if row is not None:
                    if json_text(
                        TaskSpec.model_validate(row["spec"]).model_dump(mode="json")
                    ) != json_text(encoded):
                        raise Conflict("task ID already exists with a different specification")
                else:
                    await conn.execute(
                        "INSERT INTO supervised_jobs(id,instructions) VALUES($1,$2) ON CONFLICT DO NOTHING",
                        spec.job_id,
                        instructions or "",
                    )
                    job_state = await conn.fetchval(
                        "SELECT state FROM supervised_jobs WHERE id=$1", spec.job_id
                    )
                    if job_state not in {"active", "paused"}:
                        raise Conflict("cannot add tasks to a terminal job")
                    row = await conn.fetchrow(
                        "INSERT INTO tasks(id,spec,state) VALUES($1,$2,'queued') RETURNING *",
                        spec.id,
                        encoded,
                    )
                    await event(
                        conn, "task", spec.id, "", "queued", spec=encoded, job_id=spec.job_id
                    )
                result.append(task_from_row(row))
        return result

    @staticmethod
    async def _worker(conn, worker_id: str, session: str) -> Worker:
        row = await conn.fetchrow("SELECT * FROM workers WHERE id=$1", worker_id)
        if row is None or row["session_id"] != session:
            raise StaleSession("worker session superseded")
        return Worker.model_validate(dict(row))

    @staticmethod
    async def _worker_event(conn, worker: Worker, state: str) -> None:
        rows = await conn.fetch(
            "SELECT id,generation FROM tasks WHERE worker_id=$1 AND state IN ('assigned','running')",
            worker.id,
        )
        await event(
            conn,
            "worker",
            worker.id,
            worker.state,
            state,
            session_id=worker.session_id,
            tasks=[{"task_id": row["id"], "generation": row["generation"]} for row in rows],
        )

    async def register(
        self,
        worker_id: str,
        session: str,
        capabilities: Capabilities,
        *,
        enrolled: bool = False,
        expected_device_key: str | None = None,
    ) -> None:
        async with self.change() as (conn, now):
            # Database authentication ran outside this lock; an enrollment withdrawn since must not
            # register. Static WORKER_TOKENS credentials are unaffected by enrollment history.
            if enrolled:
                state = await conn.fetchval(
                    "SELECT state FROM worker_enrollments WHERE worker_id=$1", worker_id
                )
                if state != "active":
                    raise StaleSession("worker enrollment withdrawn")
            if expected_device_key is not None:
                # Signed-device authentication also precedes this transaction;
                # recheck before superseding a session or recovering its tasks.
                key = await conn.fetchval(
                    "SELECT public_key FROM dwp_devices WHERE worker_id=$1 AND revoked_at IS NULL",
                    worker_id,
                )
                if key != expected_device_key:
                    raise StaleSession("device revoked or identity changed")
            previous = await conn.fetchval("SELECT state FROM workers WHERE id=$1", worker_id)
            rows = await conn.fetch(
                "SELECT * FROM tasks WHERE worker_id=$1 AND state IN ('assigned','running')",
                worker_id,
            )
            tasks = [task_from_row(row) for row in rows]
            for task in tasks:
                await self._fail(conn, task, "worker_reconnected", retryable=True)
            await conn.execute(
                """INSERT INTO workers(id,session_id,capabilities,state,last_seen,paused)
                   VALUES($1,$2,$3,'alive',$4,false) ON CONFLICT(id) DO UPDATE
                   SET session_id=$2,capabilities=$3,state='alive',last_seen=$4,paused=false""",
                worker_id,
                session,
                capabilities.model_dump(mode="json"),
                now,
            )
            await event(
                conn,
                "worker",
                worker_id,
                previous or "",
                "alive",
                session_id=session,
                reason="connected",
                superseded_tasks=[task_ref(t).model_dump() for t in tasks],
            )

    async def disconnect(self, worker_id: str, session: str) -> None:
        async with self.change() as (conn, _):
            try:
                worker = await self._worker(conn, worker_id, session)
            except StaleSession:
                return  # An old socket must not change its replacement's state.
            if worker.state == "alive":
                await self._worker_event(conn, worker, "unhealthy")
                await conn.execute("UPDATE workers SET state='unhealthy' WHERE id=$1", worker_id)

    @staticmethod
    def _valid(task: Task, worker_id: str, session: str, ref: Ref, now: datetime) -> bool:
        return (
            task.state in {"assigned", "running"}
            and task_ref(task) == ref
            and task.worker_id == worker_id
            and task.session_id == session
            and task.lease_until is not None
            and task.lease_until > now
            and task.deadline is not None
            and task.deadline > now
        )

    @staticmethod
    async def _renew(conn, task: Task, now: datetime) -> None:
        until = min(now + timedelta(seconds=LEASE_SECONDS), task.deadline)
        await conn.execute("UPDATE tasks SET lease_until=$2 WHERE id=$1", task.spec.id, until)

    async def heartbeat(
        self,
        worker_id: str,
        session: str,
        active: list[Ref],
        paused: bool,
        progress: float | None = None,
    ) -> list[Ref]:
        accepted = []
        async with self.change() as (conn, now):
            worker = await self._worker(conn, worker_id, session)
            if worker.state != "alive":
                await self._worker_event(conn, worker, "alive")
            await conn.execute(
                "UPDATE workers SET state='alive',last_seen=$2,paused=$3 WHERE id=$1",
                worker_id,
                now,
                paused,
            )
            for ref in active:
                row = await conn.fetchrow("SELECT * FROM tasks WHERE id=$1", ref.task_id)
                if row is None:
                    continue
                task = task_from_row(row)
                if task.state == "running" and self._valid(task, worker_id, session, ref, now):
                    await self._renew(conn, task, now)
                    if progress is not None:
                        await conn.execute(
                            "UPDATE tasks SET progress=$2 WHERE id=$1", ref.task_id, progress
                        )
                    accepted.append(ref)
        return accepted

    async def claim(self, worker_id: str, session: str) -> Task | None:
        async with self.change() as (conn, now):
            worker = await self._worker(conn, worker_id, session)
            if (
                worker.paused
                or worker.state != "alive"
                or (now - worker.last_seen).total_seconds() >= UNHEALTHY_AFTER
            ):
                return None
            occupied = await conn.fetchval(
                "SELECT EXISTS(SELECT 1 FROM tasks WHERE worker_id=$1 AND state IN ('assigned','running'))",
                worker_id,
            )
            if occupied:
                return None
            caps = worker.capabilities
            row = await conn.fetchrow(
                """SELECT * FROM tasks WHERE state='queued'
                   AND NOT EXISTS (SELECT 1 FROM supervised_jobs j
                       WHERE j.id=tasks.spec->>'job_id' AND j.state!='active')
                   AND NOT EXISTS (SELECT 1 FROM job_reservations r
                       WHERE r.worker_id=$4 AND r.job_id!=tasks.spec->>'job_id'
                       AND r.expires_at>clock_timestamp())
                   AND spec->>'kind'=ANY($1::text[])
                   AND spec->'requirements'->>'runtime'=$2
                   AND (spec->'requirements'->>'vram_mib')::int <= $3
                   AND (spec->>'target_worker_id' IS NULL OR spec->>'target_worker_id'=$4
                        OR (generation>0 AND COALESCE((spec->>'allow_failover')::boolean,true)))
                   ORDER BY created_at,id LIMIT 1""",
                caps.kinds,
                caps.runtime,
                caps.vram_mib,
                worker_id,
            )
            if row is None:
                return None
            task = task_from_row(row)
            deadline = now + timedelta(seconds=task.spec.timeout_seconds)
            lease = min(now + timedelta(seconds=ACK_SECONDS), deadline)
            row = await conn.fetchrow(
                """UPDATE tasks SET state='assigned',generation=generation+1,worker_id=$2,
                   session_id=$3,lease_until=$4,deadline=$5,result=NULL,attestation=NULL,
                   failure='',progress=0,started_at=NULL
                   WHERE id=$1 RETURNING *""",
                task.spec.id,
                worker_id,
                session,
                lease,
                deadline,
            )
            task = task_from_row(row)
            await event(
                conn,
                "task",
                task.spec.id,
                "queued",
                "assigned",
                spec=task.spec.model_dump(mode="json"),
                worker_id=worker_id,
                session_id=session,
                generation=task.generation,
                lease_until=lease.isoformat(),
            )
            return task

    async def _owned(self, conn, worker_id: str, session: str, ref: Ref) -> Task:
        await self._worker(conn, worker_id, session)
        row = await conn.fetchrow("SELECT * FROM tasks WHERE id=$1", ref.task_id)
        if row is None:
            raise StaleAssignment("assignment no longer exists")
        return task_from_row(row)

    async def ack(self, worker_id: str, session: str, ref: Ref) -> None:
        async with self.change() as (conn, now):
            task = await self._owned(conn, worker_id, session, ref)
            if not self._valid(task, worker_id, session, ref, now):
                raise StaleAssignment("assignment expired or superseded")
            await self._renew(conn, task, now)
            if task.state == "running":
                return
            await conn.execute(
                "UPDATE tasks SET state='running',started_at=$2 WHERE id=$1", ref.task_id, now
            )
            await event(
                conn,
                "task",
                ref.task_id,
                "assigned",
                "running",
                spec=task.spec.model_dump(mode="json"),
                worker_id=worker_id,
                generation=ref.generation,
            )

    @staticmethod
    async def _fail(conn, task: Task, reason: str, retryable: bool) -> None:
        state = "queued" if retryable and task.generation < task.spec.max_attempts else "failed"
        await conn.execute(
            """UPDATE tasks SET state=$2,worker_id=NULL,session_id=NULL,lease_until=NULL,
               deadline=NULL,failure=$3,progress=0,started_at=NULL WHERE id=$1""",
            task.spec.id,
            state,
            reason,
        )
        await event(
            conn,
            "task",
            task.spec.id,
            task.state,
            state,
            spec=task.spec.model_dump(mode="json"),
            worker_id=task.worker_id,
            session_id=task.session_id,
            generation=task.generation,
            reason=reason,
        )

    async def finish(
        self,
        worker_id: str,
        session: str,
        ref: Ref,
        result: JsonValue,
        failure: str = "",
        retryable: bool = False,
        *,
        attestation: dict[str, JsonValue] | None = None,
    ) -> None:
        bounded_json(result)
        async with self.change() as (conn, now):
            task = await self._owned(conn, worker_id, session, ref)
            if (
                task.state == "succeeded"
                and task_ref(task) == ref
                and task.worker_id == worker_id
                and task.session_id == session
                and not failure
            ):
                return  # Never overwrite an already accepted result.
            if task.state != "running" or not self._valid(task, worker_id, session, ref, now):
                raise StaleAssignment("assignment expired, cancelled, or superseded")
            if failure:
                await self._fail(conn, task, failure, retryable)
                return
            await conn.execute(
                """UPDATE tasks SET state='succeeded',result=$2,attestation=$3,
                   lease_until=NULL,deadline=NULL,progress=100 WHERE id=$1""",
                ref.task_id,
                result,
                attestation,
            )
            await event(
                conn,
                "task",
                ref.task_id,
                task.state,
                "succeeded",
                spec=task.spec.model_dump(mode="json"),
                worker_id=worker_id,
                generation=ref.generation,
            )

    async def cancel(self, task_id: str) -> None:
        async with self.change() as (conn, _):
            row = await conn.fetchrow("SELECT * FROM tasks WHERE id=$1", task_id)
            if row is None:
                raise NotFound("task not found")
            task = task_from_row(row)
            if task.state in {"succeeded", "failed", "cancelled"}:
                return
            await conn.execute(
                "UPDATE tasks SET state='cancelled',lease_until=NULL,deadline=NULL WHERE id=$1",
                task_id,
            )
            await event(
                conn,
                "task",
                task_id,
                task.state,
                "cancelled",
                spec=task.spec.model_dump(mode="json"),
                worker_id=task.worker_id,
                generation=task.generation,
            )

    async def reconcile(self) -> None:
        async with self.change() as (conn, now):
            expired = await conn.fetch(
                "DELETE FROM job_reservations WHERE expires_at<=$1 RETURNING job_id,worker_id", now
            )
            for reservation in expired:
                await conn.execute(
                    "INSERT INTO supervisor_events(job_id,kind,data) VALUES($1,'reservation_expired',$2)",
                    reservation["job_id"],
                    {"worker_id": reservation["worker_id"]},
                )
            # Capture affected tasks before expiry clears their worker assignment.
            rows = await conn.fetch("SELECT * FROM workers WHERE state!='offline'")
            for row in rows:
                worker = Worker.model_validate(dict(row))
                elapsed = (now - worker.last_seen).total_seconds()
                state = worker.state
                if elapsed >= LEASE_SECONDS:
                    state = "offline"
                elif elapsed >= UNHEALTHY_AFTER:
                    state = "unhealthy"
                if state != worker.state:
                    await self._worker_event(conn, worker, state)
                    await conn.execute("UPDATE workers SET state=$2 WHERE id=$1", worker.id, state)
            rows = await conn.fetch(
                """SELECT * FROM tasks WHERE state IN ('assigned','running')
                   AND (lease_until <= $1 OR deadline <= $1)""",
                now,
            )
            for row in rows:
                task = task_from_row(row)
                reason = "lease_expired"
                if task.deadline <= now:
                    reason = "execution_deadline"
                elif task.state == "assigned":
                    reason = "ack_timeout"
                await self._fail(conn, task, reason, retryable=True)
            await self._expire_enrollments(conn, now)
