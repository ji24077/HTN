"""Optional local PostgreSQL integration; never uses DATABASE_URL or Supabase."""

import asyncio
import os
import tempfile
import unittest
from pathlib import Path

import asyncpg

from orchestrator.server.db.store import Store
from orchestrator.server.updates import ChangeFeed
from orchestrator.shared.protocol import TaskSpec


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
                    for table in ("tasks", "workers", "events"):
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
            finally:
                if feed:
                    await feed.close()
                for store in (first, second):
                    if store:
                        await store.close()
                if admin:
                    await admin.close()
                database.cleanup()
