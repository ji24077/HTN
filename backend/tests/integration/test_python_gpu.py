"""Opt-in real GPU execution, dependency setup, checkpoint validation and cleanup.

RUN_PYTHON_GPU_TESTS=1 PYTHON_GPU_RUNTIME=cuda (or mps) requires a working GPU.
"""

import asyncio
import json
import os
import unittest
from pathlib import Path

from unit.test_dependency_setup import Reporter

from orchestrator.preprocessing.artifacts import encoded
from orchestrator.shared.protocol import TaskSpec
from orchestrator.worker.devices import capabilities
from orchestrator.worker.executors.python_project import PythonProjectExecutor


@unittest.skipUnless(os.getenv("RUN_PYTHON_GPU_TESTS") == "1", "requires a real GPU")
class PythonGPUExecutionTests(unittest.IsolatedAsyncioTestCase):
    async def test_dependency_environment_trains_and_validates_on_requested_gpu(self):
        runtime = os.environ["PYTHON_GPU_RUNTIME"]
        self.assertIn(runtime, {"cuda", "mps"})
        detected = await asyncio.to_thread(capabilities, "python_project")
        self.assertEqual(detected.runtime, runtime, detected.accelerator.reason)
        source = Path(__file__).resolve().parents[3] / "examples/projects/pytorch"
        files = {path.name: encoded(path.read_text()) for path in source.glob("*.py")}
        outputs = {}
        paths = []

        async def fetch(_):
            return files

        async def publish(_spec, _report, name, path):
            paths.append(path)
            outputs[name] = path.read_bytes()
            return {"id": name, "size": len(outputs[name])}

        executor = PythonProjectExecutor("ws://localhost", "gpu", fetch=fetch, publish=publish)
        for probe, steps in ((True, 2), (False, 200)):
            reporter = Reporter()
            spec = TaskSpec(
                id="gpu-smoke",
                job_id="gpu-project",
                kind="python_project",
                requirements={"runtime": runtime, "vram_mib": 0},
                timeout_seconds=120,
                max_attempts=1,
                payload={
                    "mode": "program",
                    "entrypoint": "train.py",
                    "args": ["--steps", str(steps)],
                    "dependencies": ["torch"],
                    "program": {
                        "probe": probe,
                        "validator": "validate.py",
                        "validation_args": [],
                        "outputs": [{"path": "checkpoint.pt"}, {"path": "metrics.json"}],
                        "metrics": [{"name": "mse", "minimum": None, "maximum": 0.001}],
                    },
                },
            )
            result = await asyncio.wait_for(executor.execute(spec, reporter), 120)
            self.assertTrue(result["ok"], (result, reporter.text))
            self.assertIn(f"model device={runtime}", reporter.text)
            self.assertIn(f"tensor device={runtime}", reporter.text)
            self.assertGreater(result["metrics"]["dependency_seconds"], 0)
            self.assertTrue(any(data.get("workspace_removed") for _, data in reporter.events))
        metrics = json.loads(outputs["metrics.json"])
        self.assertEqual(metrics["device"], runtime)
        self.assertLessEqual(metrics["mse"], 0.001)
        self.assertGreater(len(outputs["checkpoint.pt"]), 0)
        self.assertTrue(all(not path.exists() for path in paths))
