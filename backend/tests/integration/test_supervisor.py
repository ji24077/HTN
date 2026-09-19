"""Real PostgreSQL and executor; controlled model decisions make recovery reproducible."""

import asyncio
import os
import tempfile
import unittest
from pathlib import Path
from types import SimpleNamespace
from unittest.mock import AsyncMock
from uuid import uuid4

from orchestrator.agent import Conversation, Turn
from orchestrator.llm import ModelResponse, ToolCall
from orchestrator.server.db.store import Conflict, Store
from orchestrator.shared.protocol import Capabilities, TaskSpec, json_text, task_ref
from orchestrator.supervisor.models import Action, LogSearch, Memory
from orchestrator.supervisor.service import SupervisorService
from orchestrator.supervisor.store import SupervisorStore
from orchestrator.supervisor.tools import SupervisorTools
from orchestrator.worker.agent import execute
from orchestrator.worker.executors.stub import StubExecutor


def answer(text="Waiting for task progress."):
    return ModelResponse("response", "fixture", text, (), [{"role": "assistant", "content": text}])


def call(name, args):
    tool = ToolCall(str(uuid4()), name, json_text(args))
    return ModelResponse(
        "response",
        "fixture",
        "",
        (tool,),
        [
            {
                "type": "function_call",
                "call_id": tool.call_id,
                "name": name,
                "arguments": tool.arguments,
            }
        ],
    )


@unittest.skipUnless(
    os.getenv("RUN_SUPERVISOR_TESTS") == "1", "Set RUN_SUPERVISOR_TESTS=1; needs demo extra"
)
class SupervisorTests(unittest.IsolatedAsyncioTestCase):
    @classmethod
    def setUpClass(cls):
        import pgserver

        cls.directory = tempfile.TemporaryDirectory(prefix="supervisor-test-")
        cls.database = pgserver.get_server(
            Path(cls.directory.name) / "postgres", cleanup_mode="stop"
        )

    @classmethod
    def tearDownClass(cls):
        cls.database.cleanup()
        cls.directory.cleanup()

    async def asyncSetUp(self):
        self.store = await Store.open(self.database.get_uri(), schema="test_" + uuid4().hex)
        self.addAsyncCleanup(self.store.close)
        self.jobs = SupervisorStore(self.store)
        self.model = SimpleNamespace(respond=AsyncMock(return_value=answer()))
        self.service = SupervisorService(self.store, self.model)
        self.spec = TaskSpec(
            id="task-a",
            job_id="job-a",
            kind="stub",
            payload={"fail": True},
            requirements={"runtime": "cpu", "vram_mib": 0},
            max_attempts=2,
            timeout_seconds=60,
        )
        await self.store.submit([self.spec])
        await self.store.register(
            "worker-a", "session-a", Capabilities(runtime="cpu", vram_mib=0, kinds=["stub"])
        )

    async def ready(self):
        await self.store.pool.execute("UPDATE supervised_jobs SET retry_after=clock_timestamp()")

    async def fail(self, reason=None):
        task = await self.store.claim("worker-a", "session-a")
        await self.store.ack("worker-a", "session-a", task_ref(task))
        outcome = await execute(StubExecutor(), task, lambda _: None)
        await self.store.finish(
            "worker-a", "session-a", task_ref(task), outcome.result, failure=reason or outcome.error
        )
        return task

    async def test_failure_investigation_and_final_memory_survive_restart(self):
        await self.service.run_once("job-a")  # Submission is supervised before execution.
        await self.fail()
        action_id = str(uuid4())
        self.model.respond.side_effect = [
            call("get_task", {"task_id": "task-a"}),
            call("search_logs", {"task_id": "task-a", "attempt": 1}),
            call(
                "remember",
                {
                    "findings": [
                        {
                            "kind": "observation",
                            "text": "The payload requests a deterministic failure.",
                            "evidence": ["task-a:1"],
                        }
                    ]
                },
            ),
            call(
                "take_action",
                {
                    "action_id": action_id,
                    "operation": "cancel_job",
                    "reason": "The submitted payload will fail on every attempt.",
                },
            ),
            answer("Cancelled after inspecting the failure; no useful retry is possible."),
        ]
        await self.ready()
        await self.service.run_once("job-a")
        self.assertEqual((await self.store.task("task-a")).generation, 1)
        self.assertEqual((await self.store.task("task-a")).state, "cancelled")
        self.assertTrue((await self.jobs.job("job-a"))["memory"]["findings"])
        self.assertFalse((await self.jobs.job("job-a"))["finalized"])
        # A fresh supervisor must observe the cancellation result before finalizing.
        self.model.respond.side_effect = None
        self.model.respond.return_value = answer("Cancellation confirmed; no further work remains.")
        restarted = SupervisorService(self.store, self.model)
        await self.ready()
        await restarted.run_once("job-a")
        job = await self.jobs.job("job-a")
        self.assertTrue(job["finalized"])
        self.assertTrue(job["memory"]["findings"])
        self.assertEqual(await restarted.due(), [])
        self.assertEqual(
            await self.store.pool.fetchval("SELECT count(*) FROM supervisor_actions"), 1
        )

    async def test_transient_retry_and_accepted_result(self):
        await self.fail("temporary input service unavailable")
        self.model.respond.side_effect = [
            call("search_logs", {"task_id": "task-a", "severity": "error"}),
            call(
                "take_action",
                {
                    "action_id": str(uuid4()),
                    "operation": "retry_task",
                    "task_id": "task-a",
                    "expected_generation": 1,
                    "reason": "Temporary dependency recovered.",
                },
            ),
            answer("Retry queued within the original attempt limit."),
        ]
        await self.service.run_once("job-a")
        task = await self.store.claim("worker-a", "session-a")
        self.assertEqual(task.generation, 2)
        self.assertEqual(task.spec, self.spec)
        await self.store.ack("worker-a", "session-a", task_ref(task))
        await self.store.finish("worker-a", "session-a", task_ref(task), {"ok": True})
        self.model.respond.side_effect = None
        self.model.respond.return_value = answer("The retry succeeded.")
        await self.ready()
        await self.service.run_once("job-a")
        self.assertEqual((await self.jobs.job("job-a"))["state"], "succeeded")

    async def test_scope_filters_limits_idempotency_and_cancellation(self):
        other = self.spec.model_copy(update={"id": "task-b", "job_id": "job-b"})
        await self.store.submit([other])
        tools = SupervisorTools(self.jobs, "job-a")
        self.assertFalse((await tools.call("get_task", {"task_id": "task-b"}))["ok"])
        self.assertFalse((await tools.call("get_job", {"job_id": "job-b"}))["ok"])
        self.assertFalse((await tools.call("search_logs", {"task_id": "task-b"}))["ok"])
        await self.fail()
        page = await self.jobs.logs("job-a", LogSearch(limit=1))
        self.assertTrue(page["has_more"])
        next_page = await self.jobs.logs("job-a", LogSearch(after=page["next_cursor"]))
        self.assertTrue(all(e["task_id"] == "task-a" for e in next_page["events"]))
        errors = await self.jobs.logs(
            "job-a", LogSearch(attempt=1, severity="error", text="requested stub")
        )
        self.assertEqual(len(errors["events"]), 1)
        action = Action(
            action_id=uuid4(),
            operation="retry_task",
            task_id="task-a",
            expected_generation=1,
            reason="Test retry",
        )
        self.assertEqual(
            await self.jobs.action("job-a", action), await self.jobs.action("job-a", action)
        )
        task = await self.store.claim("worker-a", "session-a")
        await self.store.ack("worker-a", "session-a", task_ref(task))
        await self.store.finish("worker-a", "session-a", task_ref(task), None, failure="again")
        with self.assertRaises(Conflict):
            await self.jobs.action(
                "job-a", action.model_copy(update={"action_id": uuid4(), "expected_generation": 2})
            )
        with self.assertRaises(Conflict):
            await self.jobs.action("job-a", action.model_copy(update={"action_id": uuid4()}))
        await self.jobs.action(
            "job-a", Action(action_id=uuid4(), operation="cancel_job", reason="Stop")
        )
        with self.assertRaises(Conflict):
            await self.jobs.action("job-a", action.model_copy(update={"action_id": uuid4()}))
        self.assertEqual((await self.store.task("task-b")).state, "queued")

    async def test_timeout_pause_and_one_run_per_job(self):
        entered, release = asyncio.Event(), asyncio.Event()

        async def blocked(*args, **kwargs):
            entered.set()
            await release.wait()
            return answer()

        self.model.respond.side_effect = blocked
        running = asyncio.create_task(self.service.run_once("job-a"))
        await asyncio.wait_for(entered.wait(), 5)
        self.assertFalse(await self.service.run_once("job-a"))
        release.set()
        await running
        self.model.respond.side_effect = None
        await self.jobs.action(
            "job-a", Action(action_id=uuid4(), operation="pause_job", reason="Investigating")
        )
        self.assertIsNone(await self.store.claim("worker-a", "session-a"))
        await self.ready()
        await self.service.run_once("job-a")
        await self.ready()
        self.assertFalse(await self.service.run_once("job-a"))
        await self.store.pool.execute("UPDATE supervised_jobs SET next_check_at=clock_timestamp()")
        self.assertTrue(await self.service.run_once("job-a"))
        await self.jobs.action(
            "job-a", Action(action_id=uuid4(), operation="resume_job", reason="Ready")
        )
        self.assertIsNotNone(await self.store.claim("worker-a", "session-a"))

    async def test_interrupted_action_is_not_replayed_and_old_run_is_fenced(self):
        run_id = uuid4()
        state = Conversation(
            history=[{"role": "user", "content": "Inspect job"}],
            turns=[Turn(request_id=run_id, message="Inspect job")],
        )
        await self.store.pool.execute(
            "INSERT INTO supervisor_runs(id,job_id,event_cursor,conversation) VALUES($1,'job-a',0,$2)",
            run_id,
            state.model_dump(mode="json"),
        )
        await self.store.pool.execute("UPDATE supervised_jobs SET active_run=$1", run_id)
        await self.jobs.action(
            "job-a", Action(action_id=uuid4(), operation="pause_job", reason="Before crash"), run_id
        )
        await self.service.run_once("job-a")
        self.assertEqual(
            await self.store.pool.fetchval(
                "SELECT status FROM supervisor_runs WHERE id=$1", run_id
            ),
            "failed",
        )
        self.assertEqual(
            await self.store.pool.fetchval("SELECT count(*) FROM supervisor_actions"), 1
        )
        with self.assertRaises(Conflict):
            await self.jobs.action(
                "job-a",
                Action(action_id=uuid4(), operation="resume_job", reason="Stale process"),
                run_id,
            )
        with self.assertRaises(Conflict):
            await self.jobs.remember("job-a", Memory(), run_id)
        current_run = uuid4()
        await self.store.pool.execute("UPDATE supervised_jobs SET active_run=$1", current_run)
        await self.jobs.action(
            "job-a", Action(action_id=uuid4(), operation="pause_job", reason="User intervention")
        )
        with self.assertRaises(Conflict):
            await self.jobs.action(
                "job-a",
                Action(action_id=uuid4(), operation="cancel_job", reason="Outdated decision"),
                current_run,
            )
        self.assertEqual((await self.jobs.job("job-a"))["state"], "paused")

    async def test_worker_reservation_scope_expiry_and_health_wakeup(self):
        await self.store.submit([self.spec.model_copy(update={"id": "task-b", "job_id": "job-b"})])
        eligible = await self.jobs.eligible_workers("job-a")
        self.assertEqual([w["id"] for w in eligible], ["worker-a"])
        reservation = Action(
            action_id=uuid4(),
            operation="reserve_worker",
            worker_id="worker-a",
            reason="Compatible capacity",
        )
        await self.jobs.action("job-a", reservation)
        with self.assertRaises(Conflict):
            await self.jobs.action("job-b", reservation.model_copy(update={"action_id": uuid4()}))
        snapshot = await self.jobs.snapshot("job-a")
        self.assertEqual(snapshot["reserved_workers"][0]["worker_id"], "worker-a")
        await self.store.disconnect("worker-a", "session-a")
        snapshot = await self.jobs.snapshot("job-a")
        self.assertTrue(any(e["kind"] == "worker_health" for e in snapshot["events"]))
        await self.store.pool.execute("UPDATE job_reservations SET expires_at=clock_timestamp()")
        await self.store.reconcile()
        self.assertTrue(
            any(
                e["kind"] == "reservation_expired"
                for e in (await self.jobs.snapshot("job-a"))["events"]
            )
        )

    async def test_http_authentication_and_submission_instructions(self):
        import httpx

        from orchestrator.server.app import create_app

        app = create_app()
        app.state.store = self.store
        app.state.config = SimpleNamespace(
            admin_token="test-admin-long-credential", worker_tokens={}
        )
        app.state.supervisor = self.service
        async with httpx.AsyncClient(
            transport=httpx.ASGITransport(app), base_url="http://localhost"
        ) as client:
            self.assertEqual((await client.get("/v1/jobs/job-a/supervisor")).status_code, 401)
            headers = {"Authorization": "Bearer test-admin-long-credential"}
            spec = self.spec.model_copy(update={"id": "task-c", "job_id": "job-c"})
            response = await client.post(
                "/v1/tasks",
                headers=headers,
                json={
                    "tasks": [spec.model_dump(mode="json")],
                    "instructions": "Observe the first attempt.",
                },
            )
            self.assertEqual(response.status_code, 200)
            response = await client.get("/v1/jobs/job-c/supervisor", headers=headers)
            self.assertEqual(response.status_code, 200)
            self.assertEqual(response.json()["job"]["instructions"], "Observe the first attempt.")
            self.assertNotIn("history", response.json())
            self.assertEqual(
                (await client.get("/v1/jobs/missing/supervisor", headers=headers)).status_code, 404
            )
