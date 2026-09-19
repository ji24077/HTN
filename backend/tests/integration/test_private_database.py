"""Optional local PostgreSQL integration; never uses DATABASE_URL or Supabase."""

import asyncio
import base64
import hashlib
import os
import tempfile
import unittest
from datetime import UTC, datetime, timedelta
from pathlib import Path
from uuid import UUID, uuid4

import asyncpg
from cryptography.hazmat.primitives import serialization
from cryptography.hazmat.primitives.asymmetric.ed25519 import Ed25519PrivateKey

from orchestrator.server.db.store import (
    Conflict,
    EnrollmentLimit,
    NotFound,
    StaleAssignment,
    StaleSession,
    Store,
)
from orchestrator.server.updates import ChangeFeed
from orchestrator.shared.execution import ExecutionBatch
from orchestrator.shared.protocol import Capabilities, TaskSpec, task_ref


@unittest.skipUnless(
    os.getenv("RUN_DATABASE_TESTS") == "1", "Set RUN_DATABASE_TESTS=1; needs demo extra"
)
class PrivateDatabaseTests(unittest.IsolatedAsyncioTestCase):
    async def test_execution_replay_is_owned_deduplicated_and_cannot_change_new_attempt(self):
        import pgserver

        with tempfile.TemporaryDirectory(prefix="execution-db-test-") as directory:
            database = pgserver.get_server(Path(directory) / "postgres", cleanup_mode="stop")
            store = await Store.open(database.get_uri(), schema="execution_test")
            try:
                caps = Capabilities(runtime="cpu", vram_mib=0, kinds=["stub"])
                await store.register("worker-a", "session-1", caps)
                await store.register("worker-b", "session-b", caps)
                spec = TaskSpec(
                    id="tracked-task",
                    job_id="test",
                    kind="stub",
                    payload={},
                    requirements={"runtime": "cpu", "vram_mib": 0},
                    max_attempts=3,
                    timeout_seconds=60,
                )
                await store.submit([spec])
                task = await store.claim("worker-a", "session-1")
                await store.ack("worker-a", "session-1", task_ref(task))
                batch = ExecutionBatch(
                    taskId=spec.id,
                    attempt=1,
                    events=[
                        {
                            "sequence": 1,
                            "at": datetime.now(UTC),
                            "kind": "stdout",
                            "data": {
                                "text": "success output Bearer synthetic-secret",
                                "token": "hidden",
                            },
                        }
                    ],
                )
                with self.assertRaises(StaleAssignment):
                    await store.append_execution_events("worker-b", "session-b", batch)
                await store.append_execution_events("worker-a", "session-1", batch)
                await store.append_execution_events("worker-a", "session-1", batch)
                # Reconnect changes assignment state, but late diagnostics still belong to attempt 1.
                await store.register("worker-a", "session-2", caps)
                second = await store.claim("worker-a", "session-2")
                await store.ack("worker-a", "session-2", task_ref(second))
                batch.events[0].sequence = 2
                batch.events[0].kind = "progress"
                batch.events[0].data = {"percent": 99}
                await store.append_execution_events("worker-a", "session-2", batch)
                self.assertEqual((await store.task(spec.id)).progress, 0)
                batch.attempt = 2
                await store.append_execution_events("worker-a", "session-2", batch)
                await store.heartbeat("worker-a", "session-2", [task_ref(second)], False)
                self.assertEqual((await store.task(spec.id)).progress, 99)
                rows = await store.execution_events(spec.id, worker_id="worker-a", attempt=1)
                worker_rows = [r for r in rows if r["source"] == "worker"]
                self.assertEqual(len(worker_rows), 2)
                self.assertNotIn("synthetic-secret", str(worker_rows))
                self.assertNotIn("hidden", str(worker_rows))
                self.assertEqual(await store.execution_events(spec.id, worker_id="worker-b"), [])
                self.assertEqual(
                    await store.execution_events(spec.id, after=rows[-1]["id"], attempt=1), []
                )
                with self.assertRaises(StaleSession):
                    await store.append_execution_events("worker-a", "session-1", batch)
            finally:
                await store.close()
                database.cleanup()

    async def test_private_schema_shared_state_notifications_and_browser_denial(self):
        import pgserver

        with tempfile.TemporaryDirectory(prefix="orchestrator-db-test-") as directory:
            database = pgserver.get_server(Path(directory) / "postgres", cleanup_mode="stop")
            first = second = admin = feed = None
            try:
                url = database.get_uri()
                admin = await asyncpg.connect(url)
                await admin.execute("CREATE ROLE anon; CREATE ROLE authenticated")
                first = await Store.open(url, schema="orchestrator")
                second = await Store.open(url, schema="orchestrator")
                self.assertFalse(await first.has_active_enrollments())
                self.assertIsNone(await admin.fetchval("SELECT to_regclass('public.tasks')"))
                for role in ("anon", "authenticated"):
                    self.assertFalse(
                        await admin.fetchval(
                            "SELECT has_schema_privilege($1, 'orchestrator', 'USAGE')", role
                        )
                    )
                    for table in (
                        "tasks",
                        "workers",
                        "events",
                        "worker_enrollments",
                        "dwp_pair_codes",
                        "dwp_devices",
                        "dwp_assertions",
                        "execution_events",
                    ):
                        self.assertFalse(
                            await admin.fetchval(
                                "SELECT has_table_privilege($1, $2, 'SELECT')",
                                role,
                                f"orchestrator.{table}",
                            )
                        )
                        self.assertTrue(
                            await admin.fetchval(
                                "SELECT relrowsecurity FROM pg_class WHERE oid = $1::regclass",
                                f"orchestrator.{table}",
                            )
                        )
                feed = ChangeFeed(url)
                feed.start()
                async with asyncio.timeout(5):
                    while not feed.connected:
                        await asyncio.sleep(0.02)
                async with feed.subscribe() as changed:
                    changed.clear()
                    spec = TaskSpec(
                        id="private-schema-task",
                        job_id="test",
                        kind="stub",
                        payload={},
                        requirements={"runtime": "cpu", "vram_mib": 0},
                        max_attempts=3,
                        timeout_seconds=60,
                    )
                    await first.submit([spec])
                    await asyncio.wait_for(changed.wait(), 3)
                    self.assertEqual((await second.task(spec.id)).state, "queued")
                    await second.cancel(spec.id)
                    self.assertEqual((await first.task(spec.id)).state, "cancelled")

                # A freshly enrolled worker is accepted by the already-open gateway Store.
                request_id, user_id = str(uuid4()), str(uuid4())
                token = "integration-worker-secret-123456789"
                await first.reserve_enrollment(
                    "enrolled-worker", request_id, user_id, "Test", token
                )
                self.assertFalse(
                    await second.worker_authorized("enrolled-worker", f"Bearer {token}")
                )
                await first.finish_enrollment("enrolled-worker", "test-key-id")
                self.assertTrue(await second.enrolled_worker("enrolled-worker"))
                self.assertTrue(
                    await second.worker_authorized("enrolled-worker", f"Bearer {token}")
                )
                self.assertFalse(await second.worker_authorized("enrolled-worker", "Bearer wrong"))
                row = await first.pool.fetchrow(
                    "SELECT * FROM worker_enrollments WHERE worker_id='enrolled-worker'"
                )
                self.assertEqual(row["token_hash"], hashlib.sha256(token.encode()).hexdigest())
                self.assertNotIn(token, str(dict(row)))
                with self.assertRaises(Conflict):
                    await second.reserve_enrollment("another", request_id, user_id, "Test", token)
                # Concurrent reservations cannot exceed the shared database rate limit.
                attempts = await asyncio.gather(
                    *[
                        first.reserve_enrollment(
                            f"limited-{i}", str(uuid4()), user_id, "Test", token
                        )
                        for i in range(12)
                    ],
                    return_exceptions=True,
                )
                self.assertEqual(sum(isinstance(x, EnrollmentLimit) for x in attempts), 3)

                # DWP pairing is durable, one-use and visible to another gateway.
                private_key = Ed25519PrivateKey.generate()
                public_key = base64.b64encode(
                    private_key.public_key().public_bytes(
                        serialization.Encoding.DER, serialization.PublicFormat.SubjectPublicKeyInfo
                    )
                ).decode()
                owner = str(uuid4())
                code = await first.create_pair_code(owner)
                self.assertEqual(len(code), 32)
                invite = await second.pool.fetchrow(
                    "SELECT * FROM dwp_pair_codes WHERE owner_id=$1", UUID(owner)
                )
                self.assertNotIn(code, str(dict(invite)))
                self.assertEqual(invite["code_hash"], hashlib.sha256(code.encode()).hexdigest())
                pairs = await asyncio.gather(
                    first.pair_device(code.upper(), public_key, "Device"),
                    second.pair_device(code, public_key, "Device"),
                    return_exceptions=True,
                )
                self.assertEqual(sum(isinstance(x, Conflict) for x in pairs), 1)
                device_id = next(x for x in pairs if isinstance(x, str))
                self.assertTrue(await second.enrolled_worker(device_id))
                self.assertEqual(await second.device_key(device_id), public_key)
                self.assertFalse(await second.worker_authorized(device_id, "Bearer fake"))
                expiry = (datetime.now(UTC) + timedelta(seconds=120)).timestamp()
                nonce = str(uuid4())
                uses = await asyncio.gather(
                    first.use_assertion_jti(device_id, nonce, expiry),
                    second.use_assertion_jti(device_id, nonce, expiry),
                )
                self.assertEqual(sorted(uses), [False, True])
                self.assertFalse(await second.use_assertion_jti(device_id, nonce, expiry))
                self.assertFalse(await first.use_assertion_jti(device_id, str(uuid4()), 1))

                expired_code = await first.create_pair_code(owner)
                await first.pool.execute(
                    "UPDATE dwp_pair_codes SET expires_at=clock_timestamp()-interval '1 second' WHERE code_hash=$1",
                    hashlib.sha256(expired_code.encode()).hexdigest(),
                )
                with self.assertRaises(Conflict):
                    await second.pair_device(expired_code, public_key, "Expired")
                await first.pool.execute(
                    "UPDATE dwp_devices SET revoked_at=clock_timestamp() WHERE worker_id=$1",
                    device_id,
                )
                self.assertIsNone(await second.device_key(device_id))
                self.assertFalse(await second.enrolled_worker(device_id))
                self.assertFalse(await second.use_assertion_jti(device_id, str(uuid4()), expiry))
                caps = Capabilities(runtime="cpu", vram_mib=0, kinds=["echo"])
                with self.assertRaises(StaleSession):
                    await second.register(
                        device_id, "revoked-session", caps, expected_device_key=public_key
                    )
                self.assertFalse(
                    await first.pool.fetchval(
                        "SELECT EXISTS(SELECT 1 FROM workers WHERE id=$1)", device_id
                    )
                )
                await first.pool.execute(
                    "UPDATE dwp_devices SET revoked_at=NULL WHERE worker_id=$1", device_id
                )

                # The existing scheduler owns DWP tasks, generations and attestations.
                session = str(uuid4())
                await first.register(device_id, session, caps, expected_device_key=public_key)
                await first.submit(
                    [
                        TaskSpec(
                            id="signed-device-task",
                            job_id="signed-job",
                            kind="echo",
                            payload={},
                            requirements={"runtime": "cpu", "vram_mib": 0},
                            max_attempts=2,
                            timeout_seconds=60,
                        )
                    ]
                )
                task = await second.claim(device_id, session)
                self.assertEqual(task.spec.id, "signed-device-task")
                self.assertIsNone(await first.claim(device_id, session))
                ref = task_ref(task)
                await first.ack(device_id, session, ref)
                with self.assertRaises(StaleSession):
                    await second.register(
                        device_id, "wrong-key-session", caps, expected_device_key="wrong-key"
                    )
                unchanged = await first.task(task.spec.id)
                self.assertEqual(unchanged.session_id, session)
                self.assertEqual(unchanged.state, "running")
                proof = {"signature": "verified-by-gateway", "attempt": ref.generation}
                await first.finish(device_id, session, ref, {"ok": True}, attestation=proof)
                await second.finish(device_id, session, ref, {"wrong": True}, attestation={})
                accepted = await second.task(task.spec.id)
                self.assertEqual(accepted.result, {"ok": True})
                self.assertEqual(accepted.attestation, proof)
                self.assertEqual(accepted.state, "succeeded")

                await first.submit(
                    [
                        TaskSpec(
                            id="superseded-device-task",
                            job_id="signed-job",
                            kind="echo",
                            payload={},
                            requirements={"runtime": "cpu", "vram_mib": 0},
                            max_attempts=2,
                            timeout_seconds=60,
                        )
                    ]
                )
                stale = await first.claim(device_id, session)
                await first.ack(device_id, session, task_ref(stale))
                await first.finish(device_id, session, task_ref(stale), None, "retry", True)
                replacement = await second.claim(device_id, session)
                await second.ack(device_id, session, task_ref(replacement))
                with self.assertRaises(StaleAssignment):
                    await first.finish(
                        device_id, session, task_ref(stale), {"late": True}, attestation=proof
                    )
                self.assertIsNone((await second.task(stale.spec.id)).attestation)
                await second.finish(device_id, session, task_ref(replacement), {"new": True})
                self.assertIsNone((await first.task(stale.spec.id)).attestation)

                # Issuance limits are shared across processes and retain consumed codes.
                codes = await asyncio.gather(
                    *[first.create_pair_code(owner) for _ in range(11)], return_exceptions=True
                )
                self.assertEqual(sum(isinstance(x, EnrollmentLimit) for x in codes), 3)

                # A stale reservation is expired by reconciliation and can no longer activate.
                owner = str(uuid4())
                await first.reserve_enrollment("stale-worker", str(uuid4()), owner, "Test", token)
                await first.pool.execute(
                    """UPDATE worker_enrollments SET created_at=created_at-interval '1 hour'
                       WHERE worker_id='stale-worker'"""
                )
                await second.reconcile()
                self.assertFalse(await first.finish_enrollment("stale-worker", "late-key"))
                # A key that then fails revocation is still recorded on the expired row.
                self.assertFalse(
                    await first.finish_enrollment("stale-worker", None, retain_key="late-key")
                )
                self.assertEqual(await first.cancel_enrollment("stale-worker", owner), "late-key")
                self.assertEqual(
                    await first.pool.fetchval(
                        "SELECT state FROM worker_enrollments WHERE worker_id='stale-worker'"
                    ),
                    "failed",
                )
                self.assertEqual(
                    [
                        row["details"]
                        for row in await first.pool.fetch(
                            "SELECT details FROM events WHERE entity='enrollment' AND entity_id='stale-worker'"
                        )
                    ],
                    [{"reason": "expired"}],
                )
                # Withdrawal is owner-checked, idempotent, and refused once the worker registered.
                self.assertTrue(await first.has_active_enrollments())
                with self.assertRaises(NotFound):
                    await first.cancel_enrollment("enrolled-worker", str(uuid4()))
                self.assertEqual(
                    await first.cancel_enrollment("enrolled-worker", user_id), "test-key-id"
                )
                # Until revocation is confirmed, repeating returns the same key to retry.
                self.assertEqual(
                    await first.cancel_enrollment("enrolled-worker", user_id), "test-key-id"
                )
                await first.clear_enrollment_key("enrolled-worker")
                self.assertIsNone(await first.cancel_enrollment("enrolled-worker", user_id))
                self.assertFalse(
                    await second.worker_authorized("enrolled-worker", f"Bearer {token}")
                )
                self.assertEqual(
                    await first.pool.fetchval(
                        "SELECT count(*) FROM worker_enrollments WHERE state='active'"
                    ),
                    0,
                )
                # A fleet containing only paired DWP devices can still authenticate.
                self.assertTrue(await second.has_active_enrollments())
                capabilities = Capabilities(runtime="cpu", vram_mib=0, kinds=["stub"])
                with self.assertRaises(StaleSession):
                    await second.register(
                        "enrolled-worker", "session-1", capabilities, enrolled=True
                    )
                # A static WORKER_TOKENS credential for that ID is unaffected by the withdrawal.
                await second.register("enrolled-worker", "session-static", capabilities)
                await first.reserve_enrollment("joined-worker", str(uuid4()), owner, "Test", token)
                self.assertTrue(await first.finish_enrollment("joined-worker", "joined-key"))
                await second.register("joined-worker", "session-2", capabilities, enrolled=True)
                with self.assertRaises(Conflict):
                    await first.cancel_enrollment("joined-worker", owner)
            finally:
                if feed:
                    await feed.close()
                for store in (first, second):
                    if store:
                        await store.close()
                if admin:
                    await admin.close()
                database.cleanup()
