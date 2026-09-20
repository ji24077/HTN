"""Usage accounting and caps against isolated local PostgreSQL, never Supabase."""

import asyncio
import os
import tempfile
import unittest
from decimal import Decimal
from pathlib import Path
from types import SimpleNamespace
from unittest.mock import patch
from uuid import uuid4

import httpx

from orchestrator.preprocessing.models import Upload
from orchestrator.preprocessing.store import SimulationStore
from orchestrator.server.app import create_app
from orchestrator.server.chat_tools import FleetTools
from orchestrator.server.credits import account_credit, grant_credit
from orchestrator.server.db.store import Conflict, NotFound, StaleAssignment, Store
from orchestrator.server.routes import read_snapshot
from orchestrator.server.services import ServiceStore
from orchestrator.server.usage import UsageStore
from orchestrator.shared.protocol import Capabilities, TaskSpec, task_ref
from orchestrator.shared.services import ServiceAction
from orchestrator.supervisor.models import Action
from orchestrator.supervisor.store import SupervisorStore


@unittest.skipUnless(os.getenv("RUN_DATABASE_TESTS") == "1", "needs local PostgreSQL/demo extra")
class UsageTests(unittest.IsolatedAsyncioTestCase):
    @classmethod
    def setUpClass(cls):
        import pgserver

        cls.directory = tempfile.TemporaryDirectory(prefix="usage-test-")
        cls.database = pgserver.get_server(
            Path(cls.directory.name) / "postgres", cleanup_mode="stop"
        )

    @classmethod
    def tearDownClass(cls):
        cls.database.cleanup()
        cls.directory.cleanup()

    async def asyncSetUp(self):
        self.schema = "test_" + uuid4().hex
        self.store = await Store.open(self.database.get_uri(), schema=self.schema)
        self.addAsyncCleanup(self.store.close)
        self.usage = UsageStore(self.store)
        # Most lifecycle tests use a known flat fixture rate. Spec-pricing tests
        # below explicitly select the production default (null override).
        await self.store.pool.execute("UPDATE usage_pricing SET hourly_rate_cad=1")
        self.spec = TaskSpec(
            id="task-a",
            job_id="job-a",
            kind="stub",
            payload={},
            requirements={"runtime": "cpu", "vram_mib": 0},
            max_attempts=3,
            timeout_seconds=600,
        )
        self.caps = Capabilities(runtime="cpu", vram_mib=0, kinds=["stub"])
        for worker in ("a", "b", "c"):
            await self.store.register(worker, worker, self.caps)

    async def start(self, worker="a"):
        task = await self.store.claim(worker, worker)
        self.assertIsNotNone(task)
        await self.store.ack(worker, worker, task_ref(task))
        return task

    async def accrue(self, seconds, task_id="task-a"):
        # Advance observed execution time deterministically without sleeping or
        # depending on the machine's speed. Keep live leases valid.
        await self.store.pool.execute(
            """UPDATE usage_records SET started_at=clock_timestamp()-$2*interval '1 second'
                WHERE task_id=$1 AND ended_at IS NULL""",
            task_id,
            float(seconds),
        )

    async def test_queue_assignment_and_duplicate_messages_do_not_double_count(self):
        await self.store.submit([self.spec])
        self.assertEqual((await self.usage.read("job-a"))["attempts"], 0)
        task = await self.store.claim("a", "a")
        self.assertEqual((await self.usage.read("job-a"))["attempts"], 0)
        await self.store.ack("a", "a", task_ref(task))
        await self.accrue(60)
        await self.store.ack("a", "a", task_ref(task))
        live = await self.usage.read("job-a")
        self.assertEqual(live["attempts"], 1)
        self.assertEqual(live["active_attempts"], 1)
        self.assertAlmostEqual(float(live["cost"]), 1 / 60, places=4)
        await self.store.finish("a", "a", task_ref(task), {"ok": True})
        finished = await self.usage.read("job-a")
        await self.store.finish("a", "a", task_ref(task), {"different": True})
        await self.store.reconcile()
        self.assertEqual(await self.usage.read("job-a"), finished)
        self.assertEqual(finished["active_attempts"], 0)
        self.assertEqual(finished["records"][0]["outcome"], "succeeded")

    async def test_persistent_service_is_metered_and_cap_stops_it_without_restart(self):
        services = ServiceStore(self.store)
        account = uuid4()
        upload = Upload(
            request_id=uuid4(),
            description="Host an HTTP service",
            execution_mode="service",
            usage_cap="0.01",
            files=[{"name": "server.py", "content": "cGFzcw=="}],
        )
        root = await services.create(upload, {"server.py": "cGFzcw=="}, account_id=account)
        job_id = root.spec.job_id
        await self.store.register(
            "a", "a", Capabilities(runtime="cpu", vram_mib=0, kinds=["python_service"])
        )
        await services.reconcile()
        task = await self.start()
        self.assertEqual(task.spec.kind, "python_service")
        self.assertIsNone(task.deadline)
        await self.accrue(60, task.spec.id)
        before = await self.usage.read(job_id)
        self.assertEqual(before["active_attempts"], 1)
        self.assertGreater(Decimal(before["cost"]), Decimal("0.01"))
        async with self.store.pool.acquire() as conn:
            self.assertGreater(Decimal((await account_credit(conn, account))["spent"]), 0)

        await self.store.reconcile()
        status = await services.status(job_id)
        self.assertEqual(status["phase"], "stopped")
        self.assertEqual(status["service"]["desired"], "stopped")
        self.assertIn("cap reached", status["message"])
        self.assertEqual((await self.store.task(task.spec.id)).state, "cancelled")
        after = await self.usage.read(job_id)
        self.assertTrue(after["cap_reached"])
        self.assertEqual(after["active_attempts"], 0)
        self.assertIsNotNone(after["records"][0]["ended_at"])
        with self.assertRaisesRegex(Conflict, "usage cap reached"):
            await services.action(job_id, ServiceAction(action_id=uuid4(), operation="restart"))
        await services.reconcile()
        self.assertIsNone(await self.store.claim("a", "a"))
        self.assertEqual((await self.usage.read(job_id))["cost"], after["cost"])

    async def test_zero_service_cap_blocks_dispatch_and_submission_replay_is_idempotent(self):
        services = ServiceStore(self.store)
        upload = Upload(
            request_id=uuid4(),
            description="Host an HTTP service",
            execution_mode="service",
            usage_cap=0,
            files=[{"name": "server.py", "content": "cGFzcw=="}],
        )
        root = await services.create(upload, {"server.py": "cGFzcw=="})
        self.assertEqual(root.state, "cancelled")
        replay = await services.create(upload, {"server.py": "cGFzcw=="})
        self.assertEqual(replay.spec.id, root.spec.id)
        self.assertEqual(replay.state, "cancelled")
        await services.reconcile()
        self.assertEqual((await services.status(root.spec.job_id))["tasks"], [])
        with self.assertRaises(Conflict):
            await services.create(
                upload.model_copy(update={"usage_cap": Decimal(1)}), {"server.py": "cGFzcw=="}
            )

    async def test_retry_preserves_rate_history_and_restart_preserves_totals(self):
        await self.store.submit([self.spec], usage_cap="1")
        first = await self.start()
        await self.accrue(30)
        await self.store.finish(
            "a", "a", task_ref(first), None, failure="temporary", retryable=True
        )
        await self.store.pool.execute("UPDATE usage_pricing SET hourly_rate_cad=2")
        second = await self.start()
        await self.accrue(30)
        with self.assertRaises(StaleAssignment):
            await self.store.finish("a", "a", task_ref(first), {"late": True})
        await self.store.cancel(self.spec.id)
        final = await self.usage.read("job-a")
        self.assertEqual([r["attempt"] for r in final["records"]], [1, 2])
        self.assertEqual([r["hourly_rate"] for r in final["records"]], ["1.000000", "2.000000"])
        self.assertEqual([r["outcome"] for r in final["records"]], ["retry", "cancelled"])
        self.assertAlmostEqual(float(final["cost"]), 0.025, places=4)
        restarted = await Store.open(self.database.get_uri(), schema=self.schema)
        self.addAsyncCleanup(restarted.close)
        self.assertEqual(await UsageStore(restarted).read("job-a"), final)
        self.assertEqual(second.generation, 2)

    async def test_expired_lease_stops_accrual_even_before_reconciliation(self):
        await self.store.submit([self.spec])
        await self.start()
        await self.accrue(100)
        await self.store.pool.execute(
            """UPDATE tasks SET lease_until=(SELECT started_at+interval '10 seconds'
                FROM usage_records WHERE task_id=tasks.id AND ended_at IS NULL) WHERE id='task-a'"""
        )
        before = await self.usage.read("job-a")
        self.assertEqual(Decimal(before["duration_seconds"]), 10)
        await self.store.reconcile()
        after = await self.usage.read("job-a")
        self.assertEqual(after["cost"], before["cost"])
        self.assertEqual(Decimal(after["records"][0]["duration_seconds"]), 10)
        self.assertEqual(after["active_attempts"], 0)

    async def test_ack_timeout_is_free_and_reconnect_finalizes_running_attempt(self):
        await self.store.submit([self.spec])
        await self.store.claim("a", "a")
        await self.store.pool.execute(
            "UPDATE tasks SET lease_until=clock_timestamp()-interval '1 second'"
        )
        await self.store.reconcile()
        self.assertEqual((await self.usage.read("job-a"))["attempts"], 0)
        await self.start()
        await self.accrue(10)
        await self.store.register("a", "replacement", self.caps)
        usage = await self.usage.read("job-a")
        self.assertEqual(usage["active_attempts"], 0)
        self.assertEqual(usage["records"][0]["attempt"], 2)

    async def test_cap_stops_parallel_tasks_and_blocks_concurrent_dispatch(self):
        specs = [self.spec.model_copy(update={"id": f"task-{n}"}) for n in ("a", "b", "c")]
        await self.store.submit(specs, usage_cap="0.02")
        a, b = await asyncio.gather(self.start("a"), self.start("b"))
        await self.accrue(40, a.spec.id)
        await self.accrue(40, b.spec.id)
        # Each attempt individually costs less than the cap; their sum exceeds it.
        claimed, accepted, _ = await asyncio.gather(
            self.store.claim("c", "c"),
            self.store.heartbeat("a", "a", [task_ref(a)], False),
            self.store.reconcile(),
        )
        self.assertIsNone(claimed)
        self.assertEqual(accepted, [])
        self.assertTrue(all(t.state == "cancelled" for t in await self.store.tasks()))
        usage = await self.usage.read("job-a")
        self.assertTrue(usage["cap_reached"])
        self.assertIsNotNone(usage["cap_reached_at"])
        self.assertEqual(usage["active_attempts"], 0)
        self.assertEqual(Decimal(usage["remaining"]), 0)
        self.assertEqual(
            await self.store.pool.fetchval(
                "SELECT count(*) FROM supervisor_events WHERE kind='usage_cap_reached'"
            ),
            1,
        )
        with self.assertRaises(StaleAssignment):
            await self.store.finish("b", "b", task_ref(b), {"late": True})
        with self.assertRaises(Conflict):
            await self.usage.set_cap("job-a", "10")
        with self.assertRaises(Conflict):
            await SupervisorStore(self.store).action(
                "job-a",
                Action(action_id=uuid4(), operation="resume_job", reason="Try to bypass cap"),
            )

    async def test_cap_update_remove_zero_and_unrelated_run(self):
        await self.store.submit([self.spec])
        await self.store.submit([self.spec.model_copy(update={"id": "other", "job_id": "other"})])
        self.assertIsNone((await self.usage.read("job-a"))["cap"])
        await self.usage.set_cap("job-a", "1.25")
        await self.usage.set_cap("job-a", "1.25")
        self.assertEqual(
            await self.store.pool.fetchval(
                "SELECT count(*) FROM supervisor_events WHERE kind='usage_cap_changed'"
            ),
            1,
        )
        await self.usage.set_cap("job-a", None)
        task = await self.start()
        await self.accrue(40)
        await self.usage.set_cap("job-a", "0.001")
        self.assertEqual((await self.store.task(task.spec.id)).state, "cancelled")
        self.assertEqual((await self.store.task("other")).state, "queued")
        await self.usage.set_cap("other", 0)
        self.assertEqual((await self.store.task("other")).state, "cancelled")
        self.assertEqual((await self.usage.read("other"))["attempts"], 0)
        with self.assertRaises(NotFound):
            await self.usage.set_cap("missing", 1)

    async def test_atomic_submission_cap_replay_and_zero_dispatch(self):
        await self.store.submit([self.spec], usage_cap=0)
        await self.store.submit([self.spec], usage_cap=0)
        # Retrying an existing task never applies a new cap implicitly.
        await self.store.submit([self.spec], usage_cap=1)
        self.assertEqual(Decimal((await self.usage.read("job-a"))["cap"]), 0)
        self.assertIsNone(await self.store.claim("a", "a"))
        self.assertEqual((await self.store.task("task-a")).state, "cancelled")
        with self.assertRaises(Conflict):
            await self.store.submit(
                [
                    self.spec.model_copy(update={"id": "x", "job_id": "x"}),
                    self.spec.model_copy(update={"id": "y", "job_id": "y"}),
                ],
                usage_cap=1,
            )
        self.assertIsNone(await self.store.pool.fetchval("SELECT id FROM tasks WHERE id='x'"))

    async def test_simulation_cancellation_fences_planner_and_meters_only_children(self):
        simulation = SimulationStore(self.store)
        upload = Upload(
            request_id=uuid4(), description="Example", files=[{"name": "main.py", "content": ""}]
        )
        root = await simulation.create(upload, {"main.py": "print('hello')"})
        job_id = root.spec.job_id
        await self.store.pool.execute("UPDATE tasks SET state='running' WHERE id=$1", root.spec.id)
        await self.store.submit([self.spec.model_copy(update={"job_id": job_id})])
        await self.start()
        await self.accrue(30)
        await self.store.pool.execute(
            "INSERT INTO job_reservations(worker_id,job_id,expires_at) VALUES('a',$1,clock_timestamp()+interval '1 minute')",
            job_id,
        )
        before = await simulation.job(job_id)
        await self.usage.set_cap(job_id, 0)
        after = await simulation.job(job_id)
        self.assertEqual(after["phase"], "cancelled")
        self.assertGreater(after["revision"], before["revision"])
        self.assertEqual((await self.store.task(root.spec.id)).spec.payload["phase"], "cancelled")
        self.assertEqual(await self.store.pool.fetchval("SELECT count(*) FROM job_reservations"), 0)
        usage = await self.usage.read(job_id)
        self.assertEqual(usage["attempts"], 1)
        self.assertEqual(usage["records"][0]["task_id"], "task-a")

    async def test_supervisor_cancellation_uses_same_accounting(self):
        await self.store.submit([self.spec])
        await self.start()
        await self.accrue(20)
        await SupervisorStore(self.store).action(
            "job-a", Action(action_id=uuid4(), operation="cancel_job", reason="User request")
        )
        usage = await self.usage.read("job-a")
        self.assertEqual(usage["active_attempts"], 0)
        self.assertGreater(Decimal(usage["cost"]), 0)

    async def test_authenticated_api_assistant_and_validation(self):
        app = create_app()
        app.state.store = self.store
        app.state.config = SimpleNamespace(
            admin_token="test-admin-long-credential", worker_tokens={}
        )
        headers = {"Authorization": "Bearer test-admin-long-credential"}
        async with httpx.AsyncClient(
            transport=httpx.ASGITransport(app), base_url="http://localhost"
        ) as client:
            self.assertEqual((await client.get("/v1/jobs/job-a/usage")).status_code, 401)
            self.assertEqual(
                (await client.put("/v1/jobs/job-a/usage-cap", json={"cap": 0})).status_code, 401
            )
            response = await client.post(
                "/v1/tasks",
                headers=headers,
                json={"tasks": [self.spec.model_dump(mode="json")], "usage_cap": "2.5"},
            )
            self.assertEqual(response.status_code, 200, response.text)
            response = await client.get("/v1/jobs/job-a/usage", headers=headers)
            self.assertEqual(response.json()["cap"], "2.500000")
            for body in (
                {},
                {"cap": -1},
                {"cap": "NaN"},
                {"cap": "Infinity"},
                {"cap": True},
                {"cap": "0.0000001"},
                {"cap": 1, "job_id": "other"},
            ):
                self.assertEqual(
                    (
                        await client.put("/v1/jobs/job-a/usage-cap", headers=headers, json=body)
                    ).status_code,
                    422,
                    body,
                )
            self.assertEqual(
                (await client.get("/v1/jobs/missing/usage", headers=headers)).status_code, 404
            )
            self.assertEqual(
                (await client.get("/v1/jobs/job-a/usage?after=-1", headers=headers)).status_code,
                422,
            )
            request = SimpleNamespace(app=app)
            tools = FleetTools(request)
            with patch("orchestrator.server.chat_tools.require_admin", return_value=None):
                result = await tools.call("set_run_usage_cap", {"job_id": "job-a", "cap": None})
                self.assertTrue(result["ok"])
                self.assertIsNone(result["result"]["cap"])
                result = await tools.call("get_run_usage", {"job_id": "job-a"})
                self.assertTrue(result["ok"])
                self.assertFalse((await tools.call("set_run_usage_cap", {"job_id": "job-a"}))["ok"])
                self.assertFalse((await tools.call("get_run_usage", {"job_id": "missing"}))["ok"])
                spec = self.spec.model_copy(update={"id": "chat-task", "job_id": "chat-job"})
                result = await tools.call(
                    "submit_tasks",
                    {"tasks": [spec.model_dump(mode="json")], "usage_cap": "0.75"},
                )
                self.assertTrue(result["ok"], result)
                self.assertEqual((await self.usage.read("chat-job"))["cap"], "0.750000")
            snapshot = (await client.get("/v1/jobs/job-a/supervisor", headers=headers)).json()
            self.assertIn("usage", snapshot)

    async def test_usage_tables_are_private(self):
        rows = await self.store.pool.fetch(
            "SELECT relname,relrowsecurity FROM pg_class WHERE relnamespace=$1::regnamespace AND relname=ANY($2::text[])",
            self.schema,
            ["usage_records", "usage_pricing", "usage_job_totals", "usage_fleet_totals"],
        )
        self.assertEqual(len(rows), 4)
        self.assertTrue(all(row["relrowsecurity"] for row in rows))

    async def test_header_total_includes_runs_outside_the_task_list(self):
        request = SimpleNamespace(app=SimpleNamespace(state=SimpleNamespace(store=self.store)))
        empty = await read_snapshot(request)
        self.assertEqual(Decimal(empty["usage"]["cost"]), 0)
        await self.store.submit([self.spec])
        task = await self.start()
        await self.accrue(60)
        await self.store.finish("a", "a", task_ref(task), {"ok": True})
        expected = (await self.usage.read("job-a"))["cost"]
        await self.store.pool.execute("INSERT INTO supervised_jobs(id) VALUES('new-job')")
        await self.store.pool.execute(
            """INSERT INTO tasks(id,spec,state,created_at)
                SELECT 'new-'||n,jsonb_set(jsonb_set($1::jsonb,'{id}',to_jsonb('new-'||n)),
                    '{job_id}','"new-job"'),'queued',clock_timestamp()+interval '1 second'
                FROM generate_series(1,501) n""",
            self.spec.model_dump(mode="json"),
        )
        snapshot = await read_snapshot(request)
        self.assertEqual(len(snapshot["tasks"]), 500)
        self.assertNotIn("task-a", [t["id"] for t in snapshot["tasks"]])
        self.assertEqual(snapshot["usage"]["cost"], expected)
        self.assertEqual(snapshot["usage"]["attempts"], 1)

    async def test_legacy_task_without_supervision_still_executes(self):
        await self.store.pool.execute(
            "INSERT INTO tasks(id,spec,state) VALUES($1,$2,'queued')",
            self.spec.id,
            self.spec.model_dump(mode="json"),
        )
        task = await self.start()
        await self.store.finish("a", "a", task_ref(task), {"ok": True})
        self.assertEqual((await self.store.task(self.spec.id)).state, "succeeded")
        with self.assertRaises(NotFound):
            await self.usage.read(self.spec.job_id)

    async def test_uncapped_runs_continue_and_free_rate_does_not_spend_allowance(self):
        await self.store.submit([self.spec])
        task = await self.start()
        await self.accrue(7200)
        await self.store.reconcile()
        self.assertEqual((await self.store.task("task-a")).state, "running")
        self.assertEqual(
            await self.store.heartbeat("a", "a", [task_ref(task)], False), [task_ref(task)]
        )
        await self.store.cancel("task-a")
        await self.store.pool.execute("UPDATE usage_pricing SET hourly_rate_cad=0")
        await self.store.submit(
            [self.spec.model_copy(update={"id": "free", "job_id": "free"})], usage_cap="0.01"
        )
        await self.start()
        await self.accrue(7200, "free")
        await self.store.reconcile()
        usage = await self.usage.read("free")
        self.assertEqual(Decimal(usage["cost"]), 0)
        self.assertFalse(usage["cap_reached"])
        self.assertEqual((await self.store.task("free")).state, "running")

    async def test_records_paginate_without_truncating_run_total(self):
        await self.store.submit([self.spec])
        # Historical fixtures model 105 distinct acknowledged attempts without
        # running 105 workers; the query must sum all pages, not only the first.
        await self.store.pool.execute(
            """INSERT INTO usage_records(job_id,task_id,attempt,worker_id,hourly_rate_cad,
                started_at,metering_until,ended_at,outcome,duration_seconds,cost_cad)
                SELECT 'job-a','task-a',n,'a',1,clock_timestamp()-interval '36 seconds',
                    clock_timestamp(),clock_timestamp(),'succeeded',36,.01
                FROM generate_series(1,105) n"""
        )
        first = await self.usage.read("job-a")
        self.assertEqual(len(first["records"]), 100)
        self.assertTrue(first["has_more"])
        self.assertEqual(first["attempts"], 105)
        self.assertEqual(Decimal(first["cost"]), Decimal("1.05"))
        second = await self.usage.read("job-a", after=first["next_cursor"])
        self.assertEqual(len(second["records"]), 5)
        self.assertFalse(second["has_more"])
        self.assertEqual(second["cost"], first["cost"])
        self.assertGreater(second["records"][0]["id"], first["next_cursor"])

    async def test_upgrade_starts_tracking_existing_work_without_inventing_history(self):
        await self.store.submit([self.spec])
        await self.start()
        await self.accrue(1000)
        # Model an in-flight task from before usage tracking was installed.
        await self.store.pool.execute("DELETE FROM usage_records")
        restarted = await Store.open(self.database.get_uri(), schema=self.schema)
        self.addAsyncCleanup(restarted.close)
        usage = await UsageStore(restarted).read("job-a")
        self.assertEqual(usage["attempts"], 1)
        self.assertLess(Decimal(usage["duration_seconds"]), 5)
        started_at = usage["records"][0]["started_at"]
        another = await Store.open(self.database.get_uri(), schema=self.schema)
        self.addAsyncCleanup(another.close)
        self.assertEqual(
            (await UsageStore(another).read("job-a"))["records"][0]["started_at"], started_at
        )

    async def test_spec_rates_defaults_bounds_and_no_unused_gpu_premium(self):
        await self.store.pool.execute("UPDATE usage_pricing SET hourly_rate_cad=NULL")
        for machine, expected, inferred in (
            ({"logical_cores": 4, "total_ram_mb": 8192}, "0.016000", False),
            ({"logical_cores": 8, "total_ram_mb": 16384}, "0.032000", False),
            ({"logical_cores": 16, "total_ram_mb": 32768}, "0.064000", False),
            ({"logical_cores": 4096, "total_ram_mb": 10**12}, "0.500000", False),
            ({}, "0.010000", True),
            ({"logical_cores": 8, "total_ram_mb": 0}, "0.017000", True),
        ):
            with self.subTest(machine=machine):
                quote = await self.store.pool.fetchrow(
                    "SELECT * FROM worker_usage_quote($1)",
                    {"machine": machine, "runtime": "cpu", "vram_mib": 0},
                )
                self.assertEqual(quote["hourly_rate_cad"], Decimal(expected))
                self.assertEqual(quote["pricing_basis"]["inferred"], inferred)
                self.assertEqual(quote["pricing_basis"]["model"], "specs-v1")
        # CPU work must not get a premium merely because the host has GPU memory.
        cpu = {"machine": {"logical_cores": 8, "total_ram_mb": 16384}, "runtime": "cpu"}
        self.assertEqual(
            await self.store.pool.fetchval(
                "SELECT hourly_rate_cad FROM worker_usage_quote($1)", cpu
            ),
            await self.store.pool.fetchval(
                "SELECT hourly_rate_cad FROM worker_usage_quote($1)", {**cpu, "vram_mib": 24576}
            ),
        )

    async def test_attempt_snapshots_specs_and_coefficients_across_heartbeats_and_retries(self):
        await self.store.pool.execute("UPDATE usage_pricing SET hourly_rate_cad=NULL")
        caps = Capabilities(
            runtime="cpu",
            vram_mib=0,
            kinds=["stub"],
            machine={"logical_cores": 8, "total_ram_mb": 16384},
        )
        await self.store.register("a", "a", caps)
        await self.store.submit([self.spec])
        task = await self.start()
        first = (await self.usage.read("job-a"))["records"][0]
        self.assertEqual(first["hourly_rate"], "0.032000")
        self.assertEqual(first["pricing_basis"]["logical_cores"], 8)
        await self.store.pool.execute("UPDATE usage_pricing SET core_hour_cad=0.004")
        await self.store.pool.execute(
            "UPDATE workers SET capabilities=jsonb_set(capabilities,'{machine,logical_cores}','16') WHERE id='a'"
        )
        await self.store.heartbeat("a", "a", [task_ref(task)], False)
        unchanged = (await self.usage.read("job-a"))["records"][0]
        self.assertEqual(unchanged["hourly_rate"], first["hourly_rate"])
        self.assertEqual(unchanged["pricing_basis"], first["pricing_basis"])
        await self.store.finish("a", "a", task_ref(task), None, failure="retry", retryable=True)
        await self.start()
        second = (await self.usage.read("job-a"))["records"][1]
        self.assertEqual(second["hourly_rate"], "0.080000")
        self.assertEqual(second["pricing_basis"]["logical_cores"], 16)
        self.assertEqual(second["pricing_basis"]["core_hour_rate"], "0.004000")

    async def test_cap_aggregates_different_machine_rates(self):
        await self.store.pool.execute("UPDATE usage_pricing SET hourly_rate_cad=NULL")
        for worker, cores, ram in (("a", 4, 8192), ("b", 16, 32768)):
            caps = Capabilities(
                runtime="cpu",
                vram_mib=0,
                kinds=["stub"],
                machine={"logical_cores": cores, "total_ram_mb": ram},
            )
            await self.store.register(worker, worker, caps)
        await self.store.submit(
            [
                self.spec,
                self.spec.model_copy(update={"id": "task-b"}),
            ],
            usage_cap="0.0008",
        )
        await self.start("a")
        await self.start("b")
        await self.accrue(40, "task-a")
        await self.accrue(40, "task-b")
        await self.store.reconcile()
        usage = await self.usage.read("job-a")
        self.assertTrue(usage["cap_reached"])
        self.assertEqual(usage["active_attempts"], 0)
        self.assertAlmostEqual(float(usage["cost"]), (0.016 + 0.064) * 40 / 3600, places=6)
        self.assertEqual([r["hourly_rate"] for r in usage["records"]], ["0.016000", "0.064000"])

    async def test_upgrade_replaces_old_default_once_without_repricing_history(self):
        await self.store.submit([self.spec])
        await self.start()
        before = (await self.usage.read("job-a"))["records"][0]
        await self.store.pool.execute("UPDATE usage_pricing SET spec_pricing_version=NULL")
        restarted = await Store.open(self.database.get_uri(), schema=self.schema)
        self.addAsyncCleanup(restarted.close)
        self.assertIsNone(
            await restarted.pool.fetchval("SELECT hourly_rate_cad FROM usage_pricing")
        )
        self.assertEqual(
            (await UsageStore(restarted).read("job-a"))["records"][0]["hourly_rate"],
            before["hourly_rate"],
        )
        # An intentional flat override made after migration must survive restart.
        await restarted.pool.execute("UPDATE usage_pricing SET hourly_rate_cad=1")
        again = await Store.open(self.database.get_uri(), schema=self.schema)
        self.addAsyncCleanup(again.close)
        self.assertEqual(await again.pool.fetchval("SELECT hourly_rate_cad FROM usage_pricing"), 1)

    async def test_upload_cap_is_atomic_and_retry_keeps_original_cap(self):
        simulation = SimulationStore(self.store)
        for cap in (None, "1.25", "0"):
            upload = Upload(
                request_id=uuid4(),
                description="Example",
                usage_cap=cap,
                files=[{"name": "main.py", "content": ""}],
            )
            files = {"main.py": "print('hello')"}
            root = await simulation.create(upload, files)
            retry = await simulation.create(upload, files)
            self.assertEqual(root.spec.id, retry.spec.id)
            usage = await self.usage.read(root.spec.job_id)
            self.assertEqual(usage["currency"], "CAD")
            self.assertEqual(usage["cap"], None if cap is None else f"{Decimal(cap):.6f}")
            if cap == "0":
                self.assertEqual(root.state, "cancelled")
                self.assertEqual((await simulation.job(root.spec.job_id))["phase"], "cancelled")
            else:
                self.assertEqual(root.state, "queued")
            with self.assertRaises(Conflict):
                await simulation.create(upload.model_copy(update={"usage_cap": Decimal(2)}), files)

    async def test_original_currency_columns_migrate_once(self):
        await self.store.submit([self.spec], usage_cap="2.5")
        task = await self.start()
        await self.accrue(30)
        await self.store.finish("a", "a", task_ref(task), {"ok": True})
        before = await self.usage.read("job-a")
        # Recreate the previous prototype schema, including its function return name.
        schema = Path(__file__).parents[2] / "src/orchestrator/server/db/schema.sql"
        sql = schema.read_text()
        quote = sql[
            sql.index("CREATE OR REPLACE FUNCTION worker_usage_quote") : sql.index(
                "-- Task transitions"
            )
        ]
        async with self.store.pool.acquire() as conn:
            await conn.execute("DROP FUNCTION worker_usage_quote(jsonb)")
            columns = await conn.fetch(
                "SELECT table_name,column_name FROM information_schema.columns WHERE table_schema=$1 AND column_name LIKE '%_cad' AND table_name IN ('usage_pricing','usage_records','usage_record_totals','supervised_jobs')",
                self.schema,
            )
            for row in columns:
                kind = "VIEW" if row["table_name"] == "usage_record_totals" else "TABLE"
                await conn.execute(
                    f"ALTER {kind} {row['table_name']} RENAME COLUMN {row['column_name']} TO {row['column_name'].replace('_cad', '_usd')}"
                )
            await conn.execute(quote.replace("_cad", "_usd"))
        for _ in range(2):
            reopened = await Store.open(self.database.get_uri(), schema=self.schema)
            try:
                self.assertEqual(await UsageStore(reopened).read("job-a"), before)
                self.assertEqual(
                    await reopened.pool.fetchval(
                        "SELECT count(*) FROM information_schema.columns WHERE table_schema=$1 AND column_name LIKE '%_usd'",
                        self.schema,
                    ),
                    0,
                )
            finally:
                await reopened.close()

    async def test_capped_replay_after_cap_changes_does_not_restore_old_cap(self):
        await self.store.submit([self.spec], usage_cap="2.5")
        for cap in ("1", "5", None):
            await self.usage.set_cap("job-a", cap)
            replay = await self.store.submit([self.spec], usage_cap="2.5")
            self.assertEqual(replay[0].spec.id, self.spec.id)
            current = (await self.usage.read("job-a"))["cap"]
            self.assertEqual(current, None if cap is None else f"{Decimal(cap):.6f}")
            # A mixed replay/new submission must not bypass the current run limit.
            with self.assertRaises(Conflict):
                await self.store.submit(
                    [self.spec, self.spec.model_copy(update={"id": "extra"})], usage_cap="2.5"
                )
            self.assertIsNone(
                await self.store.pool.fetchval("SELECT id FROM tasks WHERE id='extra'")
            )
        await self.usage.set_cap("job-a", 0)
        self.assertEqual(
            (await self.store.submit([self.spec], usage_cap="2.5"))[0].state, "cancelled"
        )
        self.assertEqual(Decimal((await self.usage.read("job-a"))["cap"]), 0)

    async def test_missing_pricing_uses_default_specs_and_still_enforces_cap(self):
        await self.store.pool.execute("DELETE FROM usage_pricing")
        await self.store.submit([self.spec], usage_cap="0.0001")
        task = await self.start()
        usage = await self.usage.read("job-a")
        self.assertEqual(usage["attempts"], 1)
        self.assertEqual(usage["records"][0]["hourly_rate"], "0.010000")
        await self.accrue(60)
        self.assertEqual(await self.store.heartbeat("a", "a", [task_ref(task)], False), [])
        self.assertEqual((await self.store.task("task-a")).state, "cancelled")

    async def test_migration_rebuilds_derived_legacy_shapes_without_losing_totals(self):
        await self.store.submit([self.spec])
        task = await self.start()
        await self.accrue(30)
        await self.store.finish("a", "a", task_ref(task), {})
        expected = await self.usage.read("job-a")
        await self.store.pool.execute("""
            DROP VIEW usage_record_totals;
            CREATE VIEW usage_record_totals AS SELECT id,pricing_basis,cost_cad FROM usage_records;
            DROP FUNCTION worker_usage_quote(jsonb);
            CREATE FUNCTION worker_usage_quote(jsonb) RETURNS numeric LANGUAGE sql AS 'SELECT 1::numeric';
            DROP TRIGGER usage_records_totals ON usage_records;
            DROP TABLE usage_job_totals;
            DROP TABLE usage_fleet_totals;
        """)
        request = SimpleNamespace(app=SimpleNamespace(state=SimpleNamespace(store=self.store)))
        for _ in range(2):
            restarted = await Store.open(self.database.get_uri(), schema=self.schema)
            try:
                self.assertEqual(await UsageStore(restarted).read("job-a"), expected)
                snapshot = await read_snapshot(request)
                self.assertEqual(Decimal(snapshot["usage"]["cost"]), Decimal(expected["cost"]))
                self.assertEqual(snapshot["usage"]["attempts"], 1)
            finally:
                await restarted.close()

    async def test_totals_follow_finalization_rollback_and_ledger_deletion(self):
        await self.store.submit([self.spec])
        task = await self.start()
        await self.accrue(30)
        request = SimpleNamespace(app=SimpleNamespace(state=SimpleNamespace(store=self.store)))
        # Live cost contributes to the header even while finalized total is zero.
        live = (await read_snapshot(request))["usage"]
        self.assertGreater(Decimal(live["cost"]), 0)
        self.assertEqual(
            await self.store.pool.fetchval("SELECT cost_cad FROM usage_fleet_totals"), 0
        )
        async with self.store.pool.acquire() as conn:
            with self.assertRaisesRegex(RuntimeError, "rollback"):
                async with conn.transaction():
                    await conn.execute("UPDATE tasks SET state='cancelled' WHERE id='task-a'")
                    self.assertGreater(
                        await conn.fetchval("SELECT cost_cad FROM usage_fleet_totals"), 0
                    )
                    raise RuntimeError("rollback")
        self.assertEqual(
            await self.store.pool.fetchval("SELECT cost_cad FROM usage_fleet_totals"), 0
        )
        await self.store.finish("a", "a", task_ref(task), {})
        final = await self.usage.read("job-a")
        self.assertEqual(
            Decimal((await read_snapshot(request))["usage"]["cost"]), Decimal(final["cost"])
        )
        await self.store.pool.execute("DELETE FROM usage_records")
        empty = (await read_snapshot(request))["usage"]
        self.assertEqual(Decimal(empty["cost"]), 0)
        self.assertEqual(empty["attempts"], 0)
        self.assertEqual(
            await self.store.pool.fetchval(
                "SELECT cost_cad FROM usage_job_totals WHERE job_id='job-a'"
            ),
            0,
        )

    async def test_heartbeat_only_enforces_its_run_then_reconcile_enforces_others(self):
        await self.store.submit([self.spec], usage_cap="0.01")
        first = await self.start("a")
        other = self.spec.model_copy(update={"id": "other", "job_id": "other"})
        await self.store.submit([other], usage_cap="0.01")
        await self.start("b")
        await self.accrue(60, "task-a")
        await self.accrue(60, "other")
        self.assertEqual(await self.store.heartbeat("a", "a", [task_ref(first)], False), [])
        self.assertEqual((await self.store.task("task-a")).state, "cancelled")
        self.assertEqual((await self.store.task("other")).state, "running")
        await self.store.reconcile()
        self.assertEqual((await self.store.task("other")).state, "cancelled")

    async def test_unchanged_lease_does_not_rewrite_usage(self):
        await self.store.submit([self.spec])
        await self.start()
        before = await self.store.pool.fetchval("SELECT ctid::text FROM usage_records")
        await self.store.pool.execute("UPDATE tasks SET lease_until=lease_until WHERE id='task-a'")
        self.assertEqual(
            await self.store.pool.fetchval("SELECT ctid::text FROM usage_records"), before
        )

    async def test_caps_count_finalized_history_without_scanning_it(self):
        await self.store.submit([self.spec], usage_cap="100")
        await self.store.pool.execute("""INSERT INTO usage_records(job_id,task_id,attempt,worker_id,hourly_rate_cad,
            started_at,metering_until,ended_at,outcome,duration_seconds,cost_cad)
            SELECT 'job-a','task-a',n,'a',1,now(),now(),now(),'succeeded',36,.01
            FROM generate_series(1,10000) n""")
        await self.store.pool.execute("ANALYZE usage_records")
        plan = await self.store.pool.fetchval(
            "EXPLAIN (FORMAT JSON) SELECT sum(estimated_cost_cad) FROM usage_record_totals WHERE ended_at IS NULL"
        )
        self.assertIn("usage_records_live", str(plan))
        self.assertIsNone(await self.store.claim("a", "a"))
        self.assertEqual((await self.store.task("task-a")).state, "cancelled")
        request = SimpleNamespace(app=SimpleNamespace(state=SimpleNamespace(store=self.store)))
        usage = (await read_snapshot(request))["usage"]
        self.assertEqual(Decimal(usage["cost"]), 100)
        self.assertEqual(usage["attempts"], 10000)

    async def test_credit_grants_are_account_scoped_and_idempotent(self):
        account, other, receipt = uuid4(), uuid4(), uuid4()
        results = await asyncio.gather(
            *[
                grant_credit(self.store, account, "60.00", receipt, reason="Prototype credit")
                for _ in range(2)
            ]
        )
        self.assertTrue(all(Decimal(r["balance"]) == 60 for r in results))
        self.assertEqual(await self.store.pool.fetchval("SELECT count(*) FROM account_credits"), 1)
        with self.assertRaises(Conflict):
            await grant_credit(self.store, account, "61", receipt, reason="Prototype credit")
        with self.assertRaises(Conflict):
            await grant_credit(self.store, other, "60", receipt, reason="Prototype credit")
        for amount in ("NaN", "Infinity", "0", "-1", "1.001", "1000000001"):
            with self.assertRaises(ValueError):
                await grant_credit(self.store, account, amount, uuid4(), reason="Invalid")
        async with self.store.pool.acquire() as conn:
            self.assertEqual(Decimal((await account_credit(conn, other))["balance"]), 0)
        reopened = await Store.open(self.database.get_uri(), schema=self.schema)
        self.addAsyncCleanup(reopened.close)
        async with reopened.pool.acquire() as conn:
            self.assertEqual(Decimal((await account_credit(conn, account))["balance"]), 60)
        self.assertTrue(
            await self.store.pool.fetchval(
                "SELECT relrowsecurity FROM pg_class WHERE oid='account_credits'::regclass"
            )
        )

    async def test_balance_decreases_only_for_owned_jobs_and_is_not_double_charged(self):
        account, other = uuid4(), uuid4()
        await grant_credit(self.store, account, 60, uuid4(), reason="Prototype credit")
        await self.store.submit([self.spec], account_id=account)
        first = await self.start("a")
        await self.accrue(60)
        await self.store.submit(
            [self.spec.model_copy(update={"id": "other", "job_id": "other"})], account_id=other
        )
        await self.start("b")
        await self.accrue(60, "other")
        async with self.store.pool.acquire() as conn:
            balance = await account_credit(conn, account)
            self.assertAlmostEqual(float(balance["spent"]), 1 / 60, places=4)
            self.assertAlmostEqual(float(balance["balance"]), 60 - 1 / 60, places=4)
        await self.store.finish("a", "a", task_ref(first), {})
        async with self.store.pool.acquire() as conn:
            finalized = await account_credit(conn, account)
        await self.store.finish("a", "a", task_ref(first), {})
        # A replay cannot move billing ownership, even when another admin sends it.
        await self.store.submit([self.spec], account_id=other)
        with self.assertRaises(Conflict):
            await self.store.submit(
                [self.spec.model_copy(update={"id": "extra"})], account_id=other
            )
        async with self.store.pool.acquire() as conn:
            self.assertEqual(await account_credit(conn, account), finalized)
        self.assertEqual(
            await self.store.pool.fetchval(
                "SELECT billing_account_id FROM supervised_jobs WHERE id='job-a'"
            ),
            account,
        )
        request = SimpleNamespace(
            app=SimpleNamespace(state=SimpleNamespace(store=self.store)),
            state=SimpleNamespace(auth_claims={"sub": str(account)}),
        )
        snapshot = await read_snapshot(request)
        self.assertEqual(snapshot["account"], finalized)
        self.assertGreater(Decimal(snapshot["usage"]["cost"]), Decimal(finalized["spent"]))
        request.state.auth_claims = {"sub": str(other)}
        self.assertEqual(Decimal((await read_snapshot(request))["account"]["credited"]), 0)
        request.state.auth_claims = None
        self.assertIsNone((await read_snapshot(request))["account"])

    async def test_authenticated_submission_binds_credit_account_and_internal_children_inherit(
        self,
    ):
        from orchestrator.server.routes import submit_specs

        account = uuid4()
        request = SimpleNamespace(
            app=SimpleNamespace(
                state=SimpleNamespace(store=self.store, config=SimpleNamespace(worker_tokens={}))
            ),
            state=SimpleNamespace(auth_claims={"sub": str(account)}),
        )
        await submit_specs(request, [self.spec])
        self.assertEqual(
            await self.store.pool.fetchval(
                "SELECT billing_account_id FROM supervised_jobs WHERE id='job-a'"
            ),
            account,
        )
        simulation = SimulationStore(self.store)
        upload = Upload(
            request_id=uuid4(), description="Example", files=[{"name": "main.py", "content": ""}]
        )
        root = await simulation.create(upload, {"main.py": "pass"}, account_id=account)
        # Preprocessing submits children internally, without any browser identity.
        await self.store.submit(
            [self.spec.model_copy(update={"id": "child", "job_id": root.spec.job_id})]
        )
        self.assertEqual(
            await self.store.pool.fetchval(
                "SELECT billing_account_id FROM supervised_jobs WHERE id=$1", root.spec.job_id
            ),
            account,
        )
        await self.store.cancel(self.spec.id)
        child = await self.start()
        self.assertEqual(child.spec.id, "child")
        await self.accrue(60, "child")
        async with self.store.pool.acquire() as conn:
            self.assertGreater(Decimal((await account_credit(conn, account))["spent"]), 0)

    async def test_old_unowned_work_does_not_reduce_new_account_credit(self):
        await self.store.submit([self.spec])
        await self.start()
        await self.accrue(60)
        account = uuid4()
        await grant_credit(self.store, account, 60, uuid4(), reason="Prototype credit")
        async with self.store.pool.acquire() as conn:
            self.assertEqual(Decimal((await account_credit(conn, account))["balance"]), 60)
