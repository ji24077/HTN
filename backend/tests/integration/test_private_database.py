"""Optional local PostgreSQL integration; never uses DATABASE_URL or Supabase."""

import asyncio
import hashlib
import os
import tempfile
import unittest
from pathlib import Path
from uuid import uuid4

import asyncpg

from orchestrator.server.db.store import Conflict, EnrollmentLimit, NotFound, StaleSession, Store
from orchestrator.server.updates import ChangeFeed
from orchestrator.shared.protocol import Capabilities, TaskSpec


@unittest.skipUnless(
    os.getenv("RUN_DATABASE_TESTS") == "1", "Set RUN_DATABASE_TESTS=1; needs demo extra"
)
class PrivateDatabaseTests(unittest.IsolatedAsyncioTestCase):
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
                self.assertIsNone(await admin.fetchval("SELECT to_regclass('public.tasks')"))
                for role in ("anon", "authenticated"):
                    self.assertFalse(
                        await admin.fetchval(
                            "SELECT has_schema_privilege($1, 'orchestrator', 'USAGE')", role
                        )
                    )
                    for table in ("tasks", "workers", "events", "worker_enrollments"):
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
