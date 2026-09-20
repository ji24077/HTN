"""Real persistence/scheduling with synthetic GPU evidence; no hardware or model calls."""

import json
import os
import unittest
from types import SimpleNamespace
from unittest.mock import AsyncMock, patch
from uuid import uuid4

from unit.test_training_preparation import SOURCE, VALIDATOR, benchmark, plan, report

from integration import test_preprocessing as legacy
from integration.test_preprocessing import proposal
from orchestrator.llm import ModelClientError
from orchestrator.preprocessing.artifacts import encoded, unpack
from orchestrator.preprocessing.models import Upload
from orchestrator.preprocessing.portability.scripts import get_native_source, translate_script
from orchestrator.preprocessing.service import PreprocessingService
from orchestrator.preprocessing.training import source
from orchestrator.server.db.store import Store
from orchestrator.shared.protocol import Accelerator, Capabilities, task_ref


@unittest.skipUnless(os.getenv("RUN_SUPERVISOR_TESTS") == "1", "needs temporary PostgreSQL")
class TrainingPreparationTests(unittest.IsolatedAsyncioTestCase):
    setUpClass = classmethod(legacy.PreprocessingTests.setUpClass.__func__)
    tearDownClass = classmethod(legacy.PreprocessingTests.tearDownClass.__func__)

    async def asyncSetUp(self):
        self.store = await Store.open(self.database.get_uri(), schema="test_" + uuid4().hex)
        self.addAsyncCleanup(self.store.close)
        for worker, provider in (("worker-a", "cuda"), ("worker-b", "rocm")):
            await self.store.register(
                worker,
                worker,
                Capabilities(
                    runtime="cuda",
                    vram_mib=1000,
                    kinds=["python_project", "python_program"],
                    accelerator=Accelerator(available=True, providers=[provider]),
                ),
            )
        self.model = SimpleNamespace(respond=AsyncMock(side_effect=self.choose))
        self.service = PreprocessingService(self.store, self.model)
        self.config = plan()
        self.source = SOURCE
        self.contexts = []
        self.launched = []
        self.bad_static = False
        self.bad_migration = False
        self.bad_baseline = False
        self.slow_optimization = False
        self.model_failure = False
        self.kernel_calls = {"optimization": 0, "migration": 0}

    async def choose(self, messages, **kwargs):
        name = kwargs["tools"][0]["name"]
        context = json.loads(messages[0]["content"])
        self.contexts.append((name, context))
        if name == "plan_program":
            return proposal(name, self.config)
        self.assertEqual(name, "propose_training_kernel")
        stage = context["preparation_stage"]
        self.kernel_calls[stage] += 1
        if stage == "optimization" and self.model_failure:
            raise ModelClientError("provider_unavailable", "Synthetic provider unavailable")
        native = context["native_source"]
        if stage == "optimization" and self.bad_static and self.kernel_calls[stage] == 1:
            native += '\nvoid forbidden() { system("bad"); }\n'
        elif stage == "optimization" and self.bad_static:
            native = get_native_source(self.source) + "\n// repaired optimization fixture\n"
        else:
            native += "\n// " + stage + " fixture\n"
        return proposal(name, {"native_source": native, "reason": "Fixture " + stage})

    async def start(self, *, adaptations=3):
        upload = Upload(
            request_id=uuid4(),
            workload="training",
            description="Optimize if useful, migrate and train 16 steps; loss <= 1.",
            files=[
                {"name": "train.py", "content": encoded(self.source)},
                {"name": "validate.py", "content": encoded(VALIDATOR)},
            ],
            max_adaptations=adaptations,
            max_runtime_seconds=600,
        )
        task = await self.service.store.create(upload, unpack(upload.files))
        self.id = task.spec.job_id

    async def tick(self, *, execute=True):
        await self.store.pool.execute(
            "UPDATE simulation_jobs SET retry_after=clock_timestamp() WHERE job_id=$1", self.id
        )
        for worker in ("worker-a", "worker-b"):
            await self.store.heartbeat(worker, worker, [], False)
        await self.service.run_once(self.id)
        if execute:
            for worker in ("worker-a", "worker-b"):
                task = await self.store.claim(worker, worker)
                if task is None:
                    continue
                await self.store.ack(worker, worker, task_ref(task))
                self.launched.append(task)
                payload = task.spec.payload
                if "preparation" in payload:
                    stage = payload["preparation"]["stage"]
                    files = await self.service.store.files(self.id, payload["bundle_hash"])
                    candidate = source(files, "train.py")
                    value = report(candidate, payload["preparation"]["vendor"])
                    if (stage == "migration" and self.bad_migration) or (
                        stage == "baseline" and self.bad_baseline
                    ):
                        value["parameters_on_gpu"] = False
                    result = {
                        "ok": True,
                        "preparation": {"report": value},
                        "compute_seconds": 0.2,
                        "metrics": {"execution_seconds": 0.3, "output_bytes": 1024},
                    }
                    if stage == "optimization":
                        result["preparation"]["benchmark"] = benchmark(
                            self.source, candidate, factor=1.1 if self.slow_optimization else 0.8
                        )
                        result["preparation"]["benchmark"]["vendor"] = payload["preparation"][
                            "vendor"
                        ]
                else:
                    result = {"ok": True, "validation": {"metrics": {"loss": 0.3}}, "files": []}
                await self.store.finish(worker, worker, task_ref(task), result)
        return await self.service.store.job(self.id)

    async def until(self, phase):
        for _ in range(45):
            job = await self.tick()
            if job["phase"] == phase:
                return job
            if job["phase"] in {"failed", "cancelled"}:
                self.fail(str(job["data"]))
        self.fail("Did not reach " + phase)

    async def complete(self):
        for _ in range(65):
            job = await self.tick()
            if job["phase"] in {"completed", "failed", "cancelled"}:
                return job
        self.fail("Preparation did not terminate")

    async def test_order_feedback_persistence_and_exact_artifact_handoff(self):
        self.bad_static = True
        await self.start()
        job = await self.until("training_optimization_rejected")
        self.assertEqual(len(self.launched), 1)  # Baseline only; rejected code never ran.
        # Resume from persisted state, with no process-local candidate or feedback.
        self.service = PreprocessingService(self.store, self.model)
        job = await self.complete()
        self.assertEqual(job["phase"], "completed", job["data"])
        self.assertEqual(
            [t.spec.payload["role"] for t in self.launched],
            ["baseline", "optimization", "migration", "program"],
        )
        calls = [c for name, c in self.contexts if name == "propose_training_kernel"]
        self.assertIn("static_checks", calls[1]["preparation_feedback"])
        self.assertIn("repaired optimization", calls[2]["native_source"])
        state = job["data"]["training_preparation"]
        self.assertEqual(state["attempts"], {"optimization": 2, "migration": 1})
        final = self.launched[-1].spec
        self.assertEqual(final.target_worker_id, "worker-b")
        self.assertEqual(final.payload["bundle_hash"], state["accepted_hash"])
        self.assertEqual(final.payload["bundle_hash"], job["data"]["validated_hash"])
        final_files = await self.service.store.files(self.id, final.payload["bundle_hash"])
        original_files = await self.service.store.files(self.id, job["original_hash"])
        self.assertEqual(source(original_files, "train.py"), SOURCE)
        self.assertEqual(final_files["validate.py"], original_files["validate.py"])
        self.assertIn("hip/hip_runtime.h", source(final_files, "train.py"))
        self.assertEqual(final.payload["args"], self.config["run_args"])
        status = await self.service.store.status(self.id)
        self.assertEqual(status["training_preparation"]["status"], "ready")

    async def test_optional_optimization_falls_back_but_migration_still_runs(self):
        self.slow_optimization = True
        await self.start(adaptations=2)
        job = await self.complete()
        self.assertEqual(job["phase"], "completed", job["data"])
        self.assertEqual(job["data"]["training_preparation"]["optimization"], "kept_baseline")
        calls = [c for name, c in self.contexts if name == "propose_training_kernel"]
        self.assertIn("performance gate", calls[1]["preparation_feedback"]["error"])
        self.assertNotIn("optimization fixture", calls[2]["native_source"])
        self.assertIsNone(calls[2]["preparation_feedback"])

    async def test_amd_to_nvidia_uses_optimized_hip_then_validated_cuda(self):
        self.source = translate_script(SOURCE, "amd")["source"]
        self.config = plan(source_vendor="amd", target_vendor="nvidia")
        for worker, provider in (("worker-a", "rocm"), ("worker-b", "cuda")):
            await self.store.pool.execute(
                "UPDATE workers SET capabilities=jsonb_set(capabilities,'{accelerator,providers}',$2::jsonb) WHERE id=$1",
                worker,
                [provider],
            )
        with patch("orchestrator.preprocessing.training_telemetry.publish") as publish:
            await self.start()
            job = await self.complete()
        self.assertEqual(job["phase"], "completed", job["data"])
        self.assertEqual(
            [t.spec.payload["role"] for t in self.launched],
            ["baseline", "optimization", "migration", "program"],
        )
        calls = [c for name, c in self.contexts if name == "propose_training_kernel"]
        self.assertIn("hip/hip_runtime.h", calls[0]["native_source"])
        self.assertIn("optimization fixture", calls[1]["native_source"])
        self.assertIn("cuda_runtime.h", calls[1]["native_source"])
        state = job["data"]["training_preparation"]
        final_files = await self.service.store.files(self.id, state["accepted_hash"])
        self.assertIn("cuda_runtime.h", source(final_files, "train.py"))
        self.assertNotIn("hip/hip_runtime.h", source(final_files, "train.py"))
        self.assertEqual(self.launched[-1].spec.payload["bundle_hash"], state["accepted_hash"])
        emitted = [event for call in publish.call_args_list for event in call.args[0]]
        self.assertEqual(emitted, state["telemetry"])
        self.assertEqual(
            [(e["stage"], e["outcome"]) for e in emitted],
            [(stage, "passed") for stage in ("baseline", "optimization", "migration", "training")],
        )
        self.assertEqual(len({e["preparation_event_id"] for e in emitted}), len(emitted))

    async def test_optimization_model_failure_keeps_original_and_continues(self):
        self.model_failure = True
        await self.start(adaptations=2)
        job = await self.complete()
        self.assertEqual(job["phase"], "completed", job["data"])
        state = job["data"]["training_preparation"]
        self.assertEqual(state["optimization"], "kept_baseline")
        self.assertEqual(self.kernel_calls, {"optimization": 2, "migration": 1})
        failures = [e for e in state["telemetry"] if e["outcome"] == "rejected"]
        self.assertEqual(len(failures), 2)
        self.assertTrue(all(e["evidence_kind"] == "agent_error" for e in failures))
        self.assertEqual(
            [t.spec.payload["role"] for t in self.launched], ["baseline", "migration", "program"]
        )

    async def test_optimization_preserves_time_for_migration_and_full_run(self):
        await self.start()
        await self.until("training_optimization")
        await self.store.pool.execute(
            "UPDATE simulation_jobs SET deadline=clock_timestamp()+interval '500 seconds' WHERE job_id=$1",
            self.id,
        )
        job = await self.complete()
        self.assertEqual(job["phase"], "completed", job["data"])
        self.assertEqual(job["data"]["training_preparation"]["optimization"], "kept_baseline")
        self.assertEqual(self.kernel_calls, {"optimization": 0, "migration": 1})

    async def test_optimization_infrastructure_failure_does_not_fail_training(self):
        await self.start(adaptations=1)
        await self.until("training_optimization_ready")
        job = await self.tick(execute=False)
        await self.store.pool.execute(
            "UPDATE tasks SET state='failed',generation=1,failure='Synthetic GPU process lost' WHERE id=$1",
            job["data"]["tasks"][0],
        )
        job = await self.complete()
        self.assertEqual(job["phase"], "completed", job["data"])
        self.assertEqual(job["data"]["training_preparation"]["optimization"], "kept_baseline")

    async def test_required_migration_failure_blocks_full_training_and_reaches_model(self):
        self.config["preparation"]["optimize"] = False
        self.bad_migration = True
        await self.start(adaptations=2)
        job = await self.complete()
        self.assertEqual(job["phase"], "failed")
        self.assertEqual(
            [t.spec.payload["role"] for t in self.launched], ["baseline", "migration", "migration"]
        )
        calls = [c for name, c in self.contexts if name == "propose_training_kernel"]
        self.assertIn("parameters_on_gpu", calls[1]["preparation_feedback"]["error"])
        self.assertNotIn("validated_hash", job["data"])

    async def test_skip_unneeded_stages_and_train_on_tested_source(self):
        self.config = plan(optimize=False, target_vendor="nvidia", target="worker-a")
        await self.start()
        job = await self.complete()
        self.assertEqual(job["phase"], "completed", job["data"])
        self.assertEqual([t.spec.payload["role"] for t in self.launched], ["baseline", "program"])
        self.assertEqual(self.kernel_calls, {"optimization": 0, "migration": 0})
        self.assertEqual(job["data"]["training_preparation"]["migration"], "skipped")

    async def test_failed_original_blocks_all_candidate_and_training_calls(self):
        self.bad_baseline = True
        await self.start()
        job = await self.complete()
        self.assertEqual(job["phase"], "failed")
        self.assertEqual(len(self.launched), 1)
        self.assertEqual(self.kernel_calls, {"optimization": 0, "migration": 0})

    async def test_cancel_after_candidate_freeze_cannot_launch_it(self):
        await self.start()
        await self.until("training_optimization_ready")
        await self.store.cancel(self.id)
        job = await self.tick()
        self.assertEqual(job["phase"], "cancelled")
        self.assertEqual([t.spec.payload["role"] for t in self.launched], ["baseline"])

    async def test_validation_wait_does_not_advance_or_repeat_model_calls(self):
        await self.start()
        await self.until("training_optimization_ready")
        job = await self.tick(execute=False)
        self.assertEqual(job["phase"], "training_optimization_testing")
        calls = self.model.respond.await_count
        for _ in range(3):
            job = await self.tick(execute=False)
        self.assertEqual(job["phase"], "training_optimization_testing")
        self.assertEqual(self.model.respond.await_count, calls)
