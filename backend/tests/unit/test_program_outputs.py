"""Output contracts must reject unsafe paths and failed metrics before publishing."""

import os
import tempfile
import unittest
from pathlib import Path

from orchestrator.preprocessing.artifacts import encoded
from orchestrator.preprocessing.projects import ProgramPlan, validate_plan
from orchestrator.shared.protocol import TaskSpec
from orchestrator.worker.executors.program_runner import collect_outputs
from orchestrator.worker.executors.python_project import PythonProjectExecutor


class OutputPathTests(unittest.TestCase):
    def test_200_gib_sparse_checkpoint_has_no_output_byte_cap(self):
        with tempfile.TemporaryDirectory() as temporary:
            directory = Path(temporary)
            checkpoint = directory / "checkpoint.bin"
            with checkpoint.open("wb") as stream:
                stream.truncate(200 * 1024**3)
            outputs = collect_outputs(directory, ["checkpoint.bin"])
            self.assertEqual(outputs[0][1].stat().st_size, 200 * 1024**3)

    def test_project_input_accepts_binary_above_previous_limit(self):
        import base64

        from orchestrator.preprocessing.artifacts import inspect_files, unpack
        from orchestrator.preprocessing.models import UploadFile

        binary = base64.b64encode(b"x" * (9 * 1024 * 1024)).decode()
        files = unpack(
            [
                UploadFile(name="main.py", content=encoded("print('ok')")),
                UploadFile(name="data.bin", content=binary),
            ]
        )
        self.assertEqual(inspect_files(files)["data.bin"], "[data file: 9437184 bytes]")

    def test_escape_symlink_hardlink_and_empty_files_are_rejected(self):
        with tempfile.TemporaryDirectory() as temporary:
            root = Path(temporary)
            directory = root / "outputs"
            directory.mkdir()
            (root / "private").write_bytes(b"private")
            (directory / "empty").touch()
            for name in ("../private", "/private", "empty"):
                with self.subTest(name=name), self.assertRaises((ValueError, OSError)):
                    collect_outputs(directory, [name])
            (directory / "linked").symlink_to(root / "private")
            with self.assertRaises(ValueError):
                collect_outputs(directory, ["linked"])
            os.link(root / "private", directory / "hardlink")
            with self.assertRaises(ValueError):
                collect_outputs(directory, ["hardlink"])


class ProgramOutputTests(unittest.IsolatedAsyncioTestCase):
    async def run_program(self, *, validator, outputs=None, metrics=None):
        source = """import os
from pathlib import Path
out = Path(os.environ['DISPATCH_OUTPUT_DIR'])
(out / 'checkpoint.bin').write_bytes(b'weights')
"""
        files = {"train.py": encoded(source), "validate.py": encoded(validator)}
        plan = {
            "summary": "Train and evaluate.",
            "worker_id": "worker",
            "requirements": {"runtime": "cpu", "vram_mib": 0},
            "entrypoint": "train.py",
            "working_directory": ".",
            "probe_args": [],
            "run_args": [],
            "probe_timeout_seconds": 10,
            "run_timeout_seconds": 20,
            "validator": "validate.py",
            "validation_args": [],
            "outputs": outputs
            or [
                {"path": "checkpoint.bin", "kind": "checkpoint"},
                {"path": "metrics.json", "kind": "file"},
            ],
            "metrics": metrics or [{"name": "loss", "minimum": None, "maximum": 0.1}],
        }
        validate_plan(ProgramPlan.model_validate(plan), files, "training", 60)
        spec = TaskSpec(
            id="program",
            job_id="job",
            kind="python_project",
            payload={
                "mode": "program",
                "entrypoint": "train.py",
                "args": [],
                "program": {**plan, "probe": False, "workload": "training"},
            },
            requirements={"runtime": "cpu", "vram_mib": 0},
            max_attempts=1,
            timeout_seconds=30,
        )
        published = []

        async def fetch(_):
            return files

        async def publish(_spec, _report, name, path):
            content = path.read_bytes()
            published.append((name, content))
            return {"id": name, "name": name, "size": len(content)}

        result = await PythonProjectExecutor(
            "ws://localhost", "worker", fetch=fetch, publish=publish
        ).execute(spec, lambda _: None)
        return result, published

    async def test_server_defined_metric_gate_rejects_successful_validator_with_bad_metric(self):
        result, published = await self.run_program(
            validator="import os, pathlib\n(pathlib.Path(os.environ['DISPATCH_OUTPUT_DIR']) / 'metrics.json').write_text('{\"loss\": 2}')"
        )
        self.assertFalse(result["ok"])
        self.assertIn("exceeds", result["error"])
        self.assertEqual(published, [])

    async def test_nan_and_boolean_are_not_valid_evaluation_metrics(self):
        for value in ("NaN", "true"):
            with self.subTest(value=value):
                result, published = await self.run_program(
                    validator="import os, pathlib\n(pathlib.Path(os.environ['DISPATCH_OUTPUT_DIR']) / 'metrics.json').write_text('"
                    + '{"loss": '
                    + value
                    + "}')"
                )
                self.assertFalse(result["ok"])
                self.assertEqual(published, [])

    async def test_missing_declared_checkpoint_fails_before_upload(self):
        result, published = await self.run_program(
            validator="import os, pathlib\no = pathlib.Path(os.environ['DISPATCH_OUTPUT_DIR'])\n(o / 'metrics.json').write_text('{\"loss\": 0.01}')\n(o / 'checkpoint.bin').unlink()"
        )
        self.assertFalse(result["ok"])
        self.assertEqual(published, [])

    async def test_valid_outputs_are_uploaded_before_workspace_cleanup(self):
        result, published = await self.run_program(
            validator="import os, pathlib\n(pathlib.Path(os.environ['DISPATCH_OUTPUT_DIR']) / 'metrics.json').write_text('{\"loss\": 0.01}')"
        )
        self.assertTrue(result["ok"], result)
        self.assertEqual(dict(published)["checkpoint.bin"], b"weights")
        self.assertEqual(result["validation"]["metrics"], {"loss": 0.01})
