"""Real database checkpoints and barriers proving concurrent child execution."""

import asyncio
import copy
import json
import os
import unittest
from unittest.mock import AsyncMock
from uuid import uuid4

from integration import test_preprocessing as fixtures
from orchestrator.preprocessing import analysis, planning, projects
from orchestrator.preprocessing.service import PreprocessingService
from orchestrator.server.db.store import Conflict
from orchestrator.supervisor.models import Action
from orchestrator.supervisor.store import SupervisorStore

proposal = fixtures.proposal

ROLES = ("dependencies", "parallelization", "validation")
REPORT = {"summary": "Independent seeded trials", "evidence": ["simulation.py"], "questions": []}


@unittest.skipUnless(os.getenv("RUN_SUPERVISOR_TESTS") == "1", "needs temporary PostgreSQL")
class ParallelAnalysisTests(unittest.IsolatedAsyncioTestCase):
    setUpClass = classmethod(fixtures.PreprocessingTests.setUpClass.__func__)
    tearDownClass = classmethod(fixtures.PreprocessingTests.tearDownClass.__func__)
    asyncSetUp = fixtures.PreprocessingTests.asyncSetUp
    upload = fixtures.PreprocessingTests.upload

    async def delegate(self, roles=ROLES):
        if self.id is None:
            await self.upload()
        job = await self.service.store.job(self.id)
        await analysis.delegate(
            self.service,
            job,
            {
                "rationale": "Independent questions",
                "tasks": [{"role": role, "question": f"Review {role}"} for role in roles],
            },
            "classify_project",
            {"files": {"simulation.py": "original source"}},
        )
        return job

    def blocking_model(self):
        started = {role: asyncio.Event() for role in ROLES}
        release = {role: asyncio.Event() for role in ROLES}
        stopped = set()

        async def respond(messages, *, tools, instructions):
            context = json.loads(messages[0]["content"])
            role = context["assignment"]["role"]
            self.assertEqual([tool["name"] for tool in tools], ["report_analysis"])
            self.assertNotIn("analysis", context["project"])
            started[role].set()
            try:
                await release[role].wait()
                return proposal("report_analysis", REPORT)
            finally:
                stopped.add(role)

        self.model.respond = AsyncMock(side_effect=respond)
        return started, release, stopped

    async def wait_started(self, started, roles=ROLES):
        async with asyncio.timeout(3):
            await asyncio.gather(*(started[role].wait() for role in roles))

    async def test_three_calls_overlap_and_checkpoint_individually(self):
        job = await self.delegate()
        started, release, _ = self.blocking_model()
        task = asyncio.create_task(analysis.advance(self.service, job))
        self.addAsyncCleanup(self.cancel, task)
        await self.wait_started(started)  # Cannot pass with sequential model calls.
        saved = await self.service.store.job(self.id)
        self.assertEqual(
            [c["status"] for c in saved["data"]["analysis"]["children"]], ["running"] * 3
        )
        release[ROLES[0]].set()
        async with asyncio.timeout(3):
            while True:
                saved = await self.service.store.job(self.id)
                if saved["data"]["analysis"]["children"][0]["status"] == "completed":
                    break
                await asyncio.sleep(0.01)
        self.assertFalse(task.done())
        for event in release.values():
            event.set()
        self.assertTrue(await task)
        status = await self.service.store.status(self.id)
        self.assertEqual(status["analysis"]["calls_used"], 3)
        self.assertNotIn("snapshot", status["analysis"])
        self.assertEqual(status["tasks"], [])
        self.assertTrue(all(c["status"] == "completed" for c in status["analysis"]["children"]))

    async def cancel(self, task):
        task.cancel()
        await asyncio.gather(task, return_exceptions=True)

    async def test_parent_controls_cancel_all_children(self):
        for state, expected in (("cancelled", "cancelled"), ("paused", "interrupted")):
            with self.subTest(state=state):
                self.id = None
                job = await self.delegate()
                started, _, stopped = self.blocking_model()
                task = asyncio.create_task(analysis.advance(self.service, job))
                self.addAsyncCleanup(self.cancel, task)
                await self.wait_started(started)
                await SupervisorStore(self.store).action(
                    self.id,
                    Action(
                        action_id=uuid4(),
                        operation="cancel_job" if state == "cancelled" else "pause_job",
                        reason="User control",
                    ),
                )
                await asyncio.wait_for(task, 3)
                self.assertEqual(stopped, set(ROLES))
                saved = await self.service.store.job(self.id)
                self.assertEqual(
                    [c["status"] for c in saved["data"]["analysis"]["children"]], [expected] * 3
                )
                self.assertEqual(
                    await self.store.pool.fetchval(
                        "SELECT state FROM supervised_jobs WHERE id=$1", self.id
                    ),
                    state,
                )

    async def test_restart_preserves_reports_without_repeating_unknown_calls(self):
        job = await self.delegate()
        children = job["data"]["analysis"]["children"]
        children[0].update(status="completed", report=REPORT)
        for child in children[1:]:
            child["status"] = "running"
        await self.service.store.save_analysis(job)
        restored = await self.service.store.job(self.id)
        self.assertTrue(await analysis.advance(self.service, restored))
        self.model.respond.assert_not_called()
        self.assertEqual(
            children[0]["report"], restored["data"]["analysis"]["children"][0]["report"]
        )
        self.assertEqual(
            [c["status"] for c in restored["data"]["analysis"]["children"]],
            ["completed", "interrupted", "interrupted"],
        )
        self.assertEqual(analysis.context(restored)["remaining_calls"], 3)

    async def test_failures_isolated_and_budget_cannot_be_reset(self):
        async def respond(messages, **kwargs):
            role = json.loads(messages[0]["content"])["assignment"]["role"]
            if role == "dependencies":
                raise RuntimeError("secret provider body")
            return proposal("report_analysis" if role == "validation" else "submit_task", REPORT)

        self.model.respond = AsyncMock(side_effect=respond)
        for _ in range(2):
            job = await self.delegate()
            await analysis.advance(self.service, job)
        self.assertEqual(analysis.definition(job, "classify_project"), [])
        self.assertEqual(analysis.context(job)["remaining_calls"], 0)
        self.assertEqual(
            [c["status"] for c in job["data"]["analysis"]["children"]],
            ["failed", "failed", "completed"] * 2,
        )
        self.assertNotIn("secret", json.dumps(job["data"]["analysis"]))
        with self.assertRaises(ValueError):
            await self.delegate(("validation",))
        self.assertEqual(self.model.respond.await_count, 6)

    async def test_deadline_stops_inflight_requests(self):
        job = await self.delegate()
        started, _, stopped = self.blocking_model()
        task = asyncio.create_task(analysis.advance(self.service, job))
        self.addAsyncCleanup(self.cancel, task)
        await self.wait_started(started)
        await self.store.pool.execute(
            "UPDATE simulation_jobs SET deadline=clock_timestamp() WHERE job_id=$1", self.id
        )
        await asyncio.wait_for(task, 3)
        self.assertEqual(stopped, set(ROLES))
        saved = await self.service.store.job(self.id)
        self.assertTrue(
            all(c["status"] == "timed_out" for c in saved["data"]["analysis"]["children"])
        )

    async def test_stale_checkpoint_and_late_results_rejected(self):
        job = await self.delegate()
        stale = copy.deepcopy(job)
        await self.service.store.save_analysis(job)
        with self.assertRaises(Conflict):
            await self.service.store.save_analysis(stale)
        await self.store.pool.execute(
            "UPDATE supervised_jobs SET state='cancelled' WHERE id=$1", self.id
        )
        with self.assertRaises(Conflict):
            await self.service.store.save_analysis(job)

    async def test_both_coordinators_delegate_and_receive_saved_reports(self):
        for coordinator, name in ((projects, "classify_project"), (planning, "inspect_project")):
            with self.subTest(coordinator=name):
                self.id = None
                await self.upload()
                job = await self.service.store.job(self.id)
                self.model.respond = AsyncMock(
                    return_value=proposal(
                        "delegate_analysis",
                        {
                            "rationale": "Need independent advice",
                            "tasks": [
                                {"role": "validation", "question": "What should be checked?"}
                            ],
                        },
                    )
                )
                result = await coordinator.decide(self.service, job, analysis.Report, name)
                self.assertIsNone(result)
                self.model.respond = AsyncMock(return_value=proposal("report_analysis", REPORT))
                await analysis.advance(self.service, job)
                self.model.respond = AsyncMock(return_value=proposal(name, REPORT))
                result = await coordinator.decide(self.service, job, analysis.Report, name)
                self.assertEqual(result.summary, REPORT["summary"])
                context = json.loads(self.model.respond.call_args.args[0][0]["content"])
                self.assertEqual(context["analysis"]["children"][0]["report"], REPORT)

    async def test_cancel_queued_children_without_running_service(self):
        await self.delegate()
        await self.store.cancel(self.id)
        saved = await self.service.store.job(self.id)
        self.assertEqual(saved["phase"], "cancelled")
        self.assertEqual(
            [c["status"] for c in saved["data"]["analysis"]["children"]], ["cancelled"] * 3
        )
        self.model.respond.assert_not_called()

    async def test_job_lock_prevents_duplicate_fanout_and_shutdown_does_not_replay(self):
        await self.delegate()
        started, _, stopped = self.blocking_model()
        tick = asyncio.create_task(self.service.run_once(self.id))
        self.addAsyncCleanup(self.cancel, tick)
        await self.wait_started(started)
        second = PreprocessingService(self.store, self.model)
        await asyncio.wait_for(second.run_once(self.id), 3)
        self.assertEqual(self.model.respond.await_count, 3)
        await self.cancel(tick)
        self.assertEqual(stopped, set(ROLES))
        await self.store.pool.execute(
            "UPDATE simulation_jobs SET retry_after=clock_timestamp() WHERE job_id=$1", self.id
        )
        await second.run_once(self.id)
        saved = await second.store.job(self.id)
        self.assertEqual(
            [c["status"] for c in saved["data"]["analysis"]["children"]], ["interrupted"] * 3
        )
        self.assertEqual(self.model.respond.await_count, 3)
