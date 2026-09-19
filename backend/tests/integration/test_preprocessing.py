"""Real PostgreSQL and Python executions with controlled model proposals."""

import asyncio
import base64
import os
import tempfile
import unittest
from contextlib import asynccontextmanager
from pathlib import Path
from types import SimpleNamespace
from unittest.mock import AsyncMock, patch
from uuid import uuid4

from orchestrator.llm import ModelResponse, ToolCall
from orchestrator.preprocessing.artifacts import unpack
from orchestrator.preprocessing.models import Upload
from orchestrator.preprocessing.service import PreprocessingService
from orchestrator.server.db.store import Conflict, StaleAssignment, Store
from orchestrator.shared.protocol import Capabilities, json_text, task_ref
from orchestrator.supervisor.models import Action
from orchestrator.supervisor.store import SupervisorStore
from orchestrator.worker.agent import execute
from orchestrator.worker.executors.python_project import PythonProjectExecutor

ORIGINAL = """import random, argparse
def simulate(seed):
    return random.Random(seed).random()
if __name__ == '__main__':
    parser = argparse.ArgumentParser()
    parser.add_argument('--trials', type=int, default=1000)
    args = parser.parse_args()
    print(sum(simulate(seed) for seed in range(args.trials)) / args.trials)
"""
REFERENCE = (
    "from simulation import simulate\ndef run(seed, parameters):\n    return simulate(seed)\n"
)
AGGREGATE = "def aggregate(values, parameters):\n    return {'mean': sum(values) / len(values), 'count': len(values)}\n"
PLAN = {
    "summary": "Independent seeded draws, aggregated into their mean.",
    "entrypoint": "simulation.py",
    "smoke_args": ["--trials", "4"],
    "reference": REFERENCE,
    "aggregate": AGGREGATE,
    "parameters": {},
    "trials": 24,
    "batch_size": 8,
    "workers": 2,
}


def proposal(name, args):
    return ModelResponse(
        "fixture", "fixture", "", (ToolCall(str(uuid4()), name, json_text(args)),), []
    )


def candidate(code=REFERENCE):
    return proposal("propose_candidate", {"explanation": "Preserve trial semantics", "code": code})


@unittest.skipUnless(os.getenv("RUN_SUPERVISOR_TESTS") == "1", "needs temporary PostgreSQL")
class PreprocessingTests(unittest.IsolatedAsyncioTestCase):
    @classmethod
    def setUpClass(cls):
        import pgserver

        cls.directory = tempfile.TemporaryDirectory(prefix="preprocessing-test-")
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
        self.model = SimpleNamespace(
            respond=AsyncMock(side_effect=[proposal("propose_plan", PLAN), candidate()])
        )
        self.service = PreprocessingService(self.store, self.model)
        self.id = None
        for worker in ("worker-a", "worker-b"):
            await self.store.register(
                worker, worker, Capabilities(runtime="cpu", vram_mib=0, kinds=["python_project"])
            )

    async def upload(self, source=ORIGINAL, **limits):
        upload = Upload(
            request_id=uuid4(),
            description="Run 24 independent trials and report their mean.",
            files=[
                {"name": "simulation.py", "content": base64.b64encode(source.encode()).decode()}
            ],
            **limits,
        )
        task = await self.service.store.create(upload, unpack(upload.files))
        self.id = task.spec.job_id
        await self.store.pool.execute(
            "UPDATE simulation_jobs SET data=jsonb_set(data,'{planning_version}','1'::jsonb) WHERE job_id=$1",
            self.id,
        )
        return task

    async def step(self, workers=("worker-a", "worker-b")):
        await self.store.pool.execute(
            "UPDATE simulation_jobs SET retry_after=clock_timestamp() WHERE job_id=$1", self.id
        )
        for worker in workers:
            await self.store.heartbeat(worker, worker, [], False)
        await self.service.run_once(self.id)
        for worker in workers:
            task = await self.store.claim(worker, worker)
            if task:
                await self.store.ack(worker, worker, task_ref(task))

                async def fetch(spec):
                    return await self.service.store.files(spec.job_id, spec.payload["bundle_hash"])

                result = await execute(
                    PythonProjectExecutor("ws://localhost:8080/v1/worker", worker, fetch=fetch),
                    task,
                    lambda _: None,
                )
                await self.store.finish(
                    worker, worker, task_ref(task), result.result, failure=result.error or None
                )
        return await self.service.store.job(self.id)

    async def complete(self):
        for _ in range(40):
            job = await self.step()
            if job["phase"] in {"completed", "failed", "cancelled"}:
                return job
        self.fail("pipeline did not finish")

    async def until(self, phase):
        for _ in range(30):
            job = await self.step()
            if job["phase"] == phase:
                return job
        self.fail(f"pipeline did not reach {phase}")

    async def test_adapt_fresh_references_two_workers_and_exact_package_handoff(self):
        await self.upload()
        self.model.respond.side_effect = [
            proposal("propose_plan", PLAN),
            candidate(REFERENCE.replace("return simulate(seed)", "return simulate(seed) * 2")),
            candidate(),
        ]
        job = await self.complete()
        self.assertEqual(job["phase"], "completed")
        self.assertEqual(job["data"]["round"], 2)
        self.assertEqual([c["passed"] for c in job["data"]["checks"]], [False, True, True])
        samples = [set(c["seeds"]) for c in job["data"]["checks"]]
        self.assertFalse(samples[0] & samples[1])
        self.assertFalse(samples[1] & samples[2])
        result = (await self.store.task(self.id)).result
        self.assertEqual(result["output"]["count"], 24)
        self.assertEqual(result["validated_hash"], job["data"]["candidate_hash"])
        rows = await self.store.pool.fetch(
            "SELECT spec FROM tasks WHERE spec->>'job_id'=$1 AND spec->'payload'->>'role' LIKE 'batch-%'",
            self.id,
        )
        self.assertTrue(
            all(r["spec"]["payload"]["bundle_hash"] == result["validated_hash"] for r in rows)
        )
        original = await self.service.store.files(self.id, job["original_hash"])
        self.assertEqual(base64.b64decode(original["simulation.py"]).decode(), ORIGINAL)
        self.assertEqual(
            await self.store.pool.fetchval(
                "SELECT count(*) FROM job_reservations WHERE job_id=$1", self.id
            ),
            0,
        )

    async def test_broken_original_stops_without_adaptation_or_repair(self):
        await self.upload("raise RuntimeError('original broken')")
        job = await self.complete()
        self.assertEqual(job["phase"], "failed")
        self.assertIn("Original code", job["data"]["message"])
        self.assertEqual(self.model.respond.call_count, 1)
        self.assertEqual(job["data"]["versions"], [])

    async def test_reference_failure_stops_instead_of_repairing_original(self):
        await self.upload()
        self.model.respond.side_effect = [
            proposal(
                "propose_plan",
                {
                    **PLAN,
                    "reference": "def run(seed, parameters):\n    raise RuntimeError('reference broken')",
                },
            ),
            candidate(),
        ]
        job = await self.complete()
        self.assertEqual(job["phase"], "failed")
        self.assertIn("Original reference", job["data"]["message"])
        self.assertEqual(self.model.respond.call_count, 2)

    async def test_retry_exhaustion_never_launches_full_run(self):
        await self.upload(max_adaptations=1)
        self.model.respond.side_effect = [
            proposal("propose_plan", PLAN),
            candidate("def run(seed, parameters):\n    return -1"),
        ]
        job = await self.complete()
        self.assertEqual(job["phase"], "failed")
        self.assertEqual(job["data"]["round"], 1)
        self.assertNotIn("validated_hash", job["data"])
        self.assertEqual(
            await self.store.pool.fetchval(
                "SELECT count(*) FROM tasks WHERE spec->>'job_id'=$1 AND spec->'payload'->>'role' LIKE 'batch-%'",
                self.id,
            ),
            0,
        )

    async def test_holds_for_second_worker_and_survives_service_restart(self):
        await self.upload()
        await self.store.pool.execute("UPDATE workers SET state='offline' WHERE id='worker-b'")
        for _ in range(15):
            job = await self.step(workers=("worker-a",))
            if job["phase"] == "validation_wait":
                break
        self.assertEqual(job["phase"], "validation_wait")
        self.service = PreprocessingService(self.store, self.model)
        self.assertEqual((await self.step(workers=("worker-a",)))["phase"], "validation_wait")
        job = await self.complete()
        self.assertEqual(job["phase"], "completed")
        self.assertEqual(self.model.respond.call_count, 2)

    async def test_cancel_during_model_call_cannot_resurrect_job(self):
        await self.upload()

        async def cancel_then_reply(*args, **kwargs):
            await self.store.cancel(self.id)
            return proposal("propose_plan", PLAN)

        self.model.respond.side_effect = cancel_then_reply
        await self.step()
        job = await self.step()
        self.assertEqual(job["phase"], "cancelled")
        self.assertEqual(
            await self.store.pool.fetchval(
                "SELECT count(*) FROM tasks WHERE spec->>'job_id'=$1 AND id!=$1", self.id
            ),
            0,
        )

    async def test_supervisor_cannot_bypass_gate_or_mutate_pipeline_tasks(self):
        await self.upload()
        for action in [
            Action(
                action_id=uuid4(), operation="reserve_worker", worker_id="worker-a", reason="bypass"
            ),
            Action(
                action_id=uuid4(),
                operation="retry_task",
                task_id=self.id,
                expected_generation=0,
                reason="bypass",
            ),
        ]:
            with self.assertRaises(Conflict):
                await SupervisorStore(self.store).action(self.id, action)

    async def test_failed_independent_gate_returns_to_adaptation_with_fresh_samples(self):
        await self.upload()
        self.model.respond.side_effect = [proposal("propose_plan", PLAN), candidate(), candidate()]
        job = await self.until("distributed_validation")
        task_id = job["data"]["tasks"][0]
        await self.store.pool.execute(
            "UPDATE tasks SET result=jsonb_set(result,'{items,0,value}','-1'::jsonb) WHERE id=$1",
            task_id,
        )
        job = await self.step()
        self.assertEqual(job["phase"], "adapting")
        self.assertEqual(
            await self.store.pool.fetchval(
                "SELECT count(*) FROM job_reservations WHERE job_id=$1", self.id
            ),
            1,
        )
        job = await self.complete()
        self.assertEqual(job["phase"], "completed")
        self.assertEqual(job["data"]["round"], 2)
        self.assertEqual([c["passed"] for c in job["data"]["checks"]], [True, False, True, True])
        self.assertNotEqual(job["data"]["checks"][1]["seeds"], job["data"]["checks"][3]["seeds"])

    async def test_upload_auth_idempotency_and_task_scoped_artifact_download(self):
        from fastapi import FastAPI
        from httpx import ASGITransport, AsyncClient

        from orchestrator.preprocessing.routes import router, worker_router

        app = FastAPI()
        app.state.store = self.store
        app.state.preprocessing = self.service
        app.include_router(router)
        app.include_router(worker_router)
        app.state.config = SimpleNamespace(admin_token="test-admin-token", public_origin="")
        upload = Upload(
            request_id=uuid4(),
            description="Run trials",
            files=[
                {"name": "simulation.py", "content": base64.b64encode(ORIGINAL.encode()).decode()}
            ],
        )
        async with AsyncClient(transport=ASGITransport(app=app), base_url="http://test") as client:
            self.assertEqual(
                (
                    await client.post("/v1/simulations", json=upload.model_dump(mode="json"))
                ).status_code,
                401,
            )
            client.headers["Authorization"] = "Bearer test-admin-token"
            first = await client.post("/v1/simulations", json=upload.model_dump(mode="json"))
            self.assertEqual(first.status_code, 200)
            self.id = first.json()["spec"]["job_id"]
            await self.store.pool.execute(
                "UPDATE simulation_jobs SET data=jsonb_set(data,'{planning_version}','1'::jsonb) WHERE job_id=$1",
                self.id,
            )
            second = await client.post("/v1/simulations", json=upload.model_dump(mode="json"))
            self.assertEqual(second.json()["spec"]["id"], self.id)
            await self.step()
            await self.store.pool.execute(
                "UPDATE simulation_jobs SET retry_after=clock_timestamp() WHERE job_id=$1", self.id
            )
            await self.service.run_once(self.id)
            task = await self.store.claim("worker-a", "worker-a")
            url = f"/v1/execution-bundles/{task.spec.id}/{task.spec.payload['bundle_hash']}"
            headers = {
                "Authorization": "Bearer " + task.spec.payload["artifact_token"],
                "X-Worker-ID": "worker-b",
            }
            self.assertEqual((await client.get(url, headers=headers)).status_code, 403)
            headers["X-Worker-ID"] = "worker-a"
            self.assertEqual((await client.get(url, headers=headers)).status_code, 200)
            await self.store.cancel(self.id)
            self.assertEqual((await client.get(url, headers=headers)).status_code, 403)

    async def test_worker_replacement_keeps_independent_validation_on_two_workers(self):
        await self.upload()
        await self.until("distributed_reference")
        await self.store.pool.execute(
            "UPDATE simulation_jobs SET retry_after=clock_timestamp() WHERE job_id=$1", self.id
        )
        await self.service.run_once(self.id)
        await self.store.register(
            "worker-c",
            "worker-c",
            Capabilities(runtime="cpu", vram_mib=0, kinds=["python_project"]),
        )
        await self.store.pool.execute("UPDATE workers SET state='offline' WHERE id='worker-a'")
        job = await self.step(workers=("worker-b", "worker-c"))
        assigned = await self.store.pool.fetch(
            "SELECT worker_id FROM tasks WHERE id=ANY($1::text[])", job["data"]["tasks"]
        )
        self.assertEqual({r["worker_id"] for r in assigned}, {"worker-b", "worker-c"})
        self.assertEqual((await self.step(workers=("worker-b", "worker-c")))["phase"], "allocating")

    async def test_pause_resume_restores_reservations_and_deadline_cannot_be_reset(self):
        await self.upload()
        await self.step()
        supervisor = SupervisorStore(self.store)
        await supervisor.action(
            self.id, Action(action_id=uuid4(), operation="pause_job", reason="Pause")
        )
        self.assertEqual(
            await self.store.pool.fetchval(
                "SELECT count(*) FROM job_reservations WHERE job_id=$1", self.id
            ),
            0,
        )
        await self.step()
        self.assertEqual(self.model.respond.call_count, 0)
        await supervisor.action(
            self.id, Action(action_id=uuid4(), operation="resume_job", reason="Resume")
        )
        self.assertEqual((await self.step())["phase"], "original")
        await supervisor.action(
            self.id, Action(action_id=uuid4(), operation="pause_job", reason="Pause")
        )
        await self.store.pool.execute(
            "UPDATE simulation_jobs SET deadline=clock_timestamp()-interval '1 second' WHERE job_id=$1",
            self.id,
        )
        self.assertEqual((await self.step())["phase"], "failed")

    async def full_run_ready(self):
        await self.upload()
        self.model.respond.side_effect = [
            proposal("propose_plan", {**PLAN, "trials": 10000, "batch_size": 32}),
            candidate(),
        ]
        job = await self.until("allocating")
        await self.service.advance(job)
        job = await self.service.store.job(self.id)
        self.assertEqual(len(job["data"]["tasks"]), 313)
        return job

    @asynccontextmanager
    async def delayed_changes(self):
        original = self.store.change
        calls = []

        class DelayedConnection:
            def __init__(self, conn):
                self.conn = conn

            def __getattr__(self, name):
                async def call(*args, **kwargs):
                    calls.append(name)
                    await asyncio.sleep(0.03)
                    return await getattr(self.conn, name)(*args, **kwargs)

                return call

        @asynccontextmanager
        async def change(**kwargs):
            async with original(**kwargs) as (conn, now):
                yield DelayedConnection(conn), now

        with patch.object(self.store, "change", change):
            yield calls

    async def test_cancel_313_batches_with_latency_preserves_audit_and_fences_results(self):
        job = await self.full_run_ready()
        task = await self.store.claim("worker-a", "worker-a")
        await self.store.ack("worker-a", "worker-a", task_ref(task))
        action = Action(action_id=uuid4(), operation="cancel_job", reason="User cancelled")
        async with self.delayed_changes() as calls:
            await SupervisorStore(self.store).action(self.id, action)
        self.assertLessEqual(len(calls), 16)
        self.assertEqual((await self.service.store.job(self.id))["phase"], "cancelled")
        self.assertEqual(
            await self.store.pool.fetchval(
                "SELECT count(*) FROM tasks WHERE id=ANY($1::text[]) AND state='cancelled'",
                job["data"]["tasks"],
            ),
            313,
        )
        self.assertEqual(
            await self.store.pool.fetchval(
                "SELECT count(*) FROM execution_events WHERE task_id=ANY($1::text[]) AND kind='cancelled'",
                job["data"]["tasks"],
            ),
            313,
        )
        self.assertEqual(
            await self.store.pool.fetchval(
                "SELECT count(*) FROM supervisor_events WHERE job_id=$1 AND kind='task_cancelled'",
                self.id,
            ),
            314,
        )
        self.assertEqual(
            await self.store.pool.fetchval(
                "SELECT count(*) FROM job_reservations WHERE job_id=$1", self.id
            ),
            0,
        )
        with self.assertRaises(StaleAssignment):
            await self.store.finish("worker-a", "worker-a", task_ref(task), {"ok": True})
        await SupervisorStore(self.store).action(self.id, action)
        self.assertEqual(
            await self.store.pool.fetchval(
                "SELECT count(*) FROM execution_events WHERE task_id=ANY($1::text[]) AND kind='cancelled'",
                job["data"]["tasks"],
            ),
            313,
        )

    async def test_exhausted_batch_after_queued_siblings_fails_and_cleans_up_in_bulk(self):
        job = await self.full_run_ready()
        failed = job["data"]["tasks"][-1]
        await self.store.pool.execute(
            "UPDATE tasks SET state='failed',generation=3,failure='ack_timeout' WHERE id=$1", failed
        )
        async with self.delayed_changes() as calls:
            await self.service.advance(job)
        self.assertLessEqual(len(calls), 12)
        self.assertEqual((await self.service.store.job(self.id))["phase"], "failed")
        self.assertEqual(
            await self.store.pool.fetchval(
                "SELECT count(*) FROM tasks WHERE id=ANY($1::text[]) AND state='cancelled'",
                job["data"]["tasks"],
            ),
            312,
        )
        self.assertEqual((await self.store.task(failed)).failure, "ack_timeout")

    async def test_bulk_retarget_preserves_seeds_and_allows_heartbeat(self):
        job = await self.full_run_ready()
        original = {
            r["id"]: r["spec"]
            for r in await self.store.pool.fetch(
                "SELECT id,spec FROM tasks WHERE id=ANY($1::text[])", job["data"]["tasks"]
            )
        }
        await self.store.pool.execute("UPDATE workers SET state='offline' WHERE id='worker-a'")
        async with self.delayed_changes() as calls:
            await asyncio.gather(
                self.service.recover_targets(job),
                self.store.heartbeat("worker-b", "worker-b", [], False),
            )
        self.assertLessEqual(len(calls), 16)
        for r in await self.store.pool.fetch(
            "SELECT id,spec,generation FROM tasks WHERE id=ANY($1::text[])", job["data"]["tasks"]
        ):
            self.assertEqual(r["spec"]["target_worker_id"], "worker-b")
            self.assertEqual(r["spec"]["payload"], original[r["id"]]["payload"])
            self.assertEqual(r["generation"], 0)
        self.assertEqual(
            await self.store.pool.fetchval(
                "SELECT count(*) FROM execution_events WHERE task_id=ANY($1::text[]) AND data->>'reason'='preprocessing worker replacement'",
                job["data"]["tasks"],
            ),
            157,
        )

    async def test_transient_database_timeout_does_not_consume_model_budget(self):
        job = await self.full_run_ready()
        await self.store.pool.execute(
            "UPDATE simulation_jobs SET retry_after=clock_timestamp() WHERE job_id=$1", self.id
        )
        with patch.object(self.service, "recover_targets", AsyncMock(side_effect=TimeoutError)):
            await self.service.run_once(self.id)
        fresh = await self.service.store.job(self.id)
        self.assertEqual(fresh["phase"], "running")
        self.assertEqual(fresh["data"]["model_failures"], 0)
        self.assertEqual(fresh["deadline"], job["deadline"])

    async def test_10000_trials_complete_with_exactly_one_result_per_seed(self):
        await self.full_run_ready()
        for _ in range(165):
            job = await self.step()
            if job["phase"] in {"completed", "failed"}:
                break
        self.assertEqual(job["phase"], "completed")
        result = (await self.store.task(self.id)).result
        self.assertEqual(result["trials"], 10000)
        self.assertEqual(result["output"]["count"], 10000)
        batches = await self.store.pool.fetch(
            "SELECT result FROM tasks WHERE spec->>'job_id'=$1 AND spec->'payload'->>'role' LIKE 'batch-%'",
            self.id,
        )
        seeds = [i["seed"] for b in batches for i in b["result"]["items"]]
        self.assertEqual(len(seeds), 10000)
        self.assertEqual(len(set(seeds)), 10000)

    async def test_cleanup_confirmation_arrives_after_cancel_without_changing_result(self):
        from datetime import UTC, datetime

        from orchestrator.shared.execution import ExecutionBatch, ExecutionEvent

        await self.full_run_ready()
        task = await self.store.claim("worker-a", "worker-a")
        await self.store.ack("worker-a", "worker-a", task_ref(task))
        await self.store.cancel(self.id)
        status = await self.service.store.status(self.id)
        self.assertEqual(status["cleanup"]["pending_workers"], ["worker-a"])
        await self.store.append_execution_events(
            "worker-a",
            "worker-a",
            ExecutionBatch(
                taskId=task.spec.id,
                attempt=task.generation,
                events=[
                    ExecutionEvent(
                        sequence=1,
                        at=datetime.now(UTC),
                        kind="cleaned",
                        data={"workspace_removed": True, "processes_stopped": True},
                    )
                ],
            ),
        )
        status = await self.service.store.status(self.id)
        self.assertEqual(status["cleanup"], {"required": 1, "confirmed": 1, "pending_workers": []})
        self.assertEqual(status["phase"], "cancelled")
        self.assertIsNone((await self.store.task(self.id)).result)

    async def test_proposal_finishing_after_deadline_cannot_dispatch_code(self):
        await self.upload()
        await self.step()

        async def late_proposal(*args, **kwargs):
            await self.store.pool.execute(
                "UPDATE simulation_jobs SET deadline=clock_timestamp()-interval '1 second' WHERE job_id=$1",
                self.id,
            )
            return proposal("propose_plan", PLAN)

        self.model.respond.side_effect = late_proposal
        await self.step()
        # The persisted deadline, not the pre-inference snapshot, must gate dispatch.
        self.assertEqual(
            await self.store.pool.fetchval(
                "SELECT count(*) FROM tasks WHERE spec->>'job_id'=$1 AND id!=$1", self.id
            ),
            0,
        )
        job = await self.step()
        self.assertEqual(job["phase"], "failed")

    async def test_slow_model_job_does_not_block_other_job_checks(self):
        second_started = asyncio.Event()

        class Updates:
            @asynccontextmanager
            async def subscribe(self):
                yield asyncio.Event()

        async def run_once(job_id):
            if job_id == "slow":
                await asyncio.Event().wait()
            else:
                second_started.set()

        pool = SimpleNamespace(
            fetch=AsyncMock(return_value=[{"job_id": "slow"}, {"job_id": "other"}])
        )
        with (
            patch.object(self.service.store, "pool", pool),
            patch.object(self.service, "run_once", run_once),
        ):
            loop = asyncio.create_task(self.service.run(Updates()))
            try:
                await asyncio.wait_for(second_started.wait(), 1)
            finally:
                loop.cancel()
                await asyncio.gather(loop, return_exceptions=True)
