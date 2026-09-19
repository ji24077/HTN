"""Model-owned execution plans, measured workers, and real Python execution."""

import json
import os
import unittest
from unittest.mock import AsyncMock

from integration import test_preprocessing as legacy
from orchestrator.preprocessing.models import ExecutionPolicy, ProjectPlan, Schedule

PROJECT = {
    "preprocessing_worker": "worker-a",
    "summary": "Independently seeded trials",
    "entrypoint": "simulation.py",
    "working_directory": ".",
    "root_seed": 100,
    "smoke_args": ["--trials", "4"],
    "reference": legacy.REFERENCE,
    "aggregate": legacy.AGGREGATE,
    "parameters": {},
    "trials": 10000,
    "profile_cases": 8,
    "probe_timeout_seconds": 30,
}
POLICY = {
    "rationale": "Measured work is cheap; avoid coordination overhead.",
    "local_cases": 7,
    "independent_cases": 9,
    "validation_timeout_seconds": 45,
    "aggregation": "inline",
}
PARTIAL = (
    legacy.REFERENCE
    + """
def reduce(values, parameters):
    return {'sum': sum(values), 'count': len(values)}
def merge(partials, parameters):
    count = sum(p['count'] for p in partials)
    return {'mean': sum(p['sum'] for p in partials) / count, 'count': count}
"""
)


@unittest.skipUnless(os.getenv("RUN_SUPERVISOR_TESTS") == "1", "needs temporary PostgreSQL")
class PlanningTests(unittest.IsolatedAsyncioTestCase):
    setUpClass = classmethod(legacy.PreprocessingTests.setUpClass.__func__)
    tearDownClass = classmethod(legacy.PreprocessingTests.tearDownClass.__func__)
    asyncSetUp = legacy.PreprocessingTests.asyncSetUp
    step = legacy.PreprocessingTests.step
    complete = legacy.PreprocessingTests.complete

    async def start(self, *, policy=None, project=None, code=legacy.REFERENCE, schedule=None):
        self.project = {**PROJECT, **(project or {})}
        self.policy = {**POLICY, **(policy or {})}
        self.contexts = []
        self.code = code
        self.choose_schedule = schedule
        await legacy.PreprocessingTests.upload(self)
        await self.store.pool.execute(
            "UPDATE simulation_jobs SET data=jsonb_set(data,'{planning_version}','2'::jsonb) WHERE job_id=$1",
            self.id,
        )

        async def choose(messages, **kwargs):
            name = kwargs["tools"][0]["name"]
            context = json.loads(messages[0]["content"])
            self.contexts.append((name, context))
            if name == "inspect_project":
                value = self.project
            elif name == "plan_execution":
                value = self.policy
            elif name == "propose_candidate":
                value = {"explanation": "Preserve seeds", "code": self.code}
            elif name == "place_workers":
                value = {
                    "rationale": "Use available distinct workers",
                    "worker_ids": ["worker-a", "worker-b"][: context["required_count"]],
                }
            elif name == "schedule_wave":
                value = (
                    self.choose_schedule(context)
                    if self.choose_schedule
                    else {
                        "rationale": "10k trials fit in one fast CPU task, returning the aggregate only.",
                        "batches": [
                            {
                                "worker_id": "worker-b",
                                "trials": context["remaining_trials"],
                                "timeout_seconds": 37,
                            }
                        ],
                        "aggregation_worker": "worker-b",
                        "aggregation_timeout_seconds": 23,
                    }
                )
            else:
                raise AssertionError(name)
            return legacy.proposal(name, value)

        self.model.respond = AsyncMock(side_effect=choose)

    async def test_agent_runs_10000_trials_as_one_task_on_one_worker_after_profiling(self):
        await self.start()
        job = await self.complete()
        self.assertEqual(job["phase"], "completed", job["data"]["message"])
        self.assertEqual((await self.store.task(self.id)).result["output"]["count"], 10000)
        batches = await self.store.pool.fetch(
            "SELECT spec,result,worker_id FROM tasks WHERE spec->>'job_id'=$1 AND spec->'payload'->>'role' LIKE 'batch-%'",
            self.id,
        )
        self.assertEqual(len(batches), 1)
        self.assertEqual(batches[0]["worker_id"], "worker-b")
        self.assertEqual(batches[0]["spec"]["timeout_seconds"], 37)
        self.assertFalse(batches[0]["spec"]["allow_failover"])
        self.assertEqual(
            batches[0]["spec"]["payload"]["seed_range"], {"start": 100, "count": 10000}
        )
        self.assertNotIn("items", batches[0]["result"])
        self.assertEqual([c["cases"] for c in job["data"]["checks"]], [7, 9])
        planning = next(c for n, c in self.contexts if n == "plan_execution")
        measured = next(m for m in planning["measurements"] if m["stage"] == "profiling")
        self.assertEqual(measured["trials"], 8)
        self.assertGreater(measured["compute_seconds"], 0)
        self.assertGreater(measured["execution_seconds"], 0)
        self.assertGreater(measured["output_bytes"], 0)
        scheduling = next(c for n, c in self.contexts if n == "schedule_wave")
        self.assertTrue(any(m["worker_id"] == "worker-b" for m in scheduling["measurements"]))
        self.assertEqual(
            await self.store.pool.fetchval(
                "SELECT count(*) FROM job_reservations WHERE job_id=$1", self.id
            ),
            0,
        )

    async def test_agent_replaces_lost_full_run_worker_without_changing_trial_range(self):
        await self.start()
        for _ in range(30):
            job = await self.step()
            if job["phase"] == "allocating":
                break
        self.assertEqual(job["phase"], "allocating")
        await self.store.pool.execute(
            "UPDATE simulation_jobs SET retry_after=clock_timestamp() WHERE job_id=$1", self.id
        )
        await self.service.run_once(self.id)
        job = await self.service.store.job(self.id)
        task_id = job["data"]["tasks"][0]
        original = await self.store.task(task_id)
        await self.store.pool.execute("UPDATE workers SET state='offline' WHERE id='worker-b'")
        await self.step(workers=("worker-a",))
        relocated = await self.store.task(task_id)
        self.assertEqual(relocated.state, "succeeded")
        self.assertEqual(relocated.worker_id, "worker-a")
        self.assertEqual(relocated.spec.payload, original.spec.payload)
        replacement = [c for n, c in self.contexts if n == "place_workers"][-1]
        self.assertEqual(replacement["eligible_replacements"], ["worker-a"])
        completed = await self.step(workers=("worker-a",))
        self.assertEqual(completed["phase"], "completed")
        self.assertEqual((await self.store.task(self.id)).result["output"]["count"], 10000)

    async def test_agent_replans_wave_sizes_placement_and_timeouts_from_observed_results(self):
        def wave(context):
            batches = (
                [
                    {"worker_id": "worker-b", "trials": 3, "timeout_seconds": 17},
                    {"worker_id": "worker-a", "trials": 7, "timeout_seconds": 21},
                ]
                if context["completed_trials"] == 0
                else [
                    {
                        "worker_id": "worker-b",
                        "trials": context["remaining_trials"],
                        "timeout_seconds": 29,
                    }
                ]
            )
            return {
                "rationale": "Review the first wave before scaling.",
                "batches": batches,
                "aggregation_worker": "worker-a",
                "aggregation_timeout_seconds": 31,
            }

        await self.start(
            policy={"aggregation": "partial"}, project={"trials": 24}, code=PARTIAL, schedule=wave
        )
        job = await self.complete()
        self.assertEqual(job["phase"], "completed", job["data"]["message"])
        result = (await self.store.task(self.id)).result["output"]
        import random

        self.assertAlmostEqual(
            result["mean"], sum(random.Random(i).random() for i in range(100, 124)) / 24
        )
        self.assertEqual(result["count"], 24)
        plans = [c for n, c in self.contexts if n == "schedule_wave"]
        self.assertEqual([c["remaining_trials"] for c in plans], [24, 14])
        self.assertTrue(any(m["stage"] == "running" for m in plans[1]["measurements"]))
        rows = await self.store.pool.fetch(
            "SELECT spec,result FROM tasks WHERE spec->>'job_id'=$1 AND spec->'payload'->>'role' LIKE 'batch-%'",
            self.id,
        )
        spans = sorted([r["spec"]["payload"]["seed_range"] for r in rows], key=lambda s: s["start"])
        self.assertEqual(
            spans,
            [{"start": 100, "count": 3}, {"start": 103, "count": 7}, {"start": 110, "count": 14}],
        )
        self.assertTrue(all("partial" in r["result"] and "values" not in r["result"] for r in rows))

    async def test_invalid_partial_aggregation_never_reaches_full_run(self):
        await self.start(
            policy={"aggregation": "partial"}, code=PARTIAL.replace(" / count", " / count * 2")
        )
        job = await self.complete()
        self.assertEqual(job["phase"], "failed")
        self.assertEqual(job["data"]["round"], 3)
        self.assertEqual(len(job["data"]["checks"]), 3)
        self.assertTrue(all(not c["passed"] for c in job["data"]["checks"]))
        self.assertEqual(
            await self.store.pool.fetchval(
                "SELECT count(*) FROM tasks WHERE spec->>'job_id'=$1 AND spec->'payload'->>'role' LIKE 'batch-%'",
                self.id,
            ),
            0,
        )

    async def test_values_strategy_preserves_seed_order_with_unequal_batch_sizes(self):
        def wave(context):
            return {
                "rationale": "Keep ordered values for a non-reducible aggregate.",
                "batches": [
                    {"worker_id": "worker-b", "trials": 19, "timeout_seconds": 40},
                    {"worker_id": "worker-a", "trials": 5, "timeout_seconds": 40},
                ],
                "aggregation_worker": "worker-a",
                "aggregation_timeout_seconds": 40,
            }

        aggregate = "def aggregate(values, parameters):\n    return values\n"
        await self.start(
            policy={"aggregation": "values"},
            project={"trials": 24, "aggregate": aggregate},
            schedule=wave,
        )
        job = await self.complete()
        self.assertEqual(job["phase"], "completed", job["data"]["message"])
        import random

        self.assertEqual(
            (await self.store.task(self.id)).result["output"],
            [random.Random(i).random() for i in range(100, 124)],
        )

    def test_planning_decisions_are_required_not_hidden_defaults(self):
        for schema, fields in [
            (ProjectPlan, ["profile_cases", "probe_timeout_seconds", "preprocessing_worker"]),
            (
                ExecutionPolicy,
                ["local_cases", "independent_cases", "aggregation", "validation_timeout_seconds"],
            ),
            (Schedule, ["batches", "aggregation_worker", "aggregation_timeout_seconds"]),
        ]:
            for field in fields:
                self.assertIn(field, schema.model_json_schema()["required"])
