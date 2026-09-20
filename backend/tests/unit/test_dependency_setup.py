"""Real, offline pip installs exercise the worker's complete setup/cleanup path."""

import asyncio
import base64
import io
import json
import os
import tempfile
import unittest
import zipfile
from pathlib import Path
from unittest.mock import patch

from pydantic import ValidationError

from orchestrator.preprocessing.artifacts import encoded
from orchestrator.preprocessing.service import PreprocessingService
from orchestrator.shared.dependencies import DependencyPlan
from orchestrator.shared.protocol import TaskSpec
from orchestrator.shared.services import ServiceConfig
from orchestrator.worker.executors.dependency_setup import manifest
from orchestrator.worker.executors.python_project import PythonProjectExecutor

WHEEL_NAME = "dispatch_fixture-1.0-py3-none-any.whl"


def wheel():
    output = io.BytesIO()
    with zipfile.ZipFile(output, "w") as archive:
        archive.writestr("dispatch_fixture.py", "VALUE = 42\n")
        archive.writestr(
            "dispatch_fixture-1.0.dist-info/METADATA",
            "Metadata-Version: 2.1\nName: dispatch-fixture\nVersion: 1.0\n",
        )
        archive.writestr(
            "dispatch_fixture-1.0.dist-info/WHEEL",
            "Wheel-Version: 1.0\nGenerator: test\nRoot-Is-Purelib: true\nTag: py3-none-any\n",
        )
        archive.writestr("dispatch_fixture-1.0.dist-info/RECORD", "")
    return base64.b64encode(output.getvalue()).decode()


SOURCE = """import os, sys
import dispatch_fixture
import pydantic
def run(seed, parameters):
    assert 'TEST_WORKER_SECRET' not in os.environ
    return {'answer': dispatch_fixture.VALUE + seed, 'python': sys.executable,
            'root': os.environ['HOME'], 'package': dispatch_fixture.__file__}
"""


class Reporter:
    def __init__(self):
        self.text = ""
        self.events = []
        self.installing = asyncio.Event()

    def __call__(self, _):
        pass

    def stdout(self, text):
        self.text += text
        if "BUILD_STARTED=" in self.text:
            self.installing.set()

    stderr = stdout

    def emit(self, kind, data):
        self.events.append((kind, data))


class DependencyPlanTests(unittest.TestCase):
    def test_accepts_distribution_names_extras_and_constraints_not_commands(self):
        deps = ["Pillow", "scikit-learn>=1.3,<2", "uvicorn[standard]", "torch==2.5.1+cu124"]
        self.assertEqual(DependencyPlan(dependencies=deps).dependencies, deps)
        for value in (
            "--target=/tmp",
            "numpy; curl example.org",
            "./local",
            "x @ https://x",
            "a\nb",
        ):
            with self.subTest(value=value), self.assertRaises(ValidationError):
                DependencyPlan(dependencies=[value])

    def test_plan_survives_service_serialization_and_simulation_dispatch(self):
        config = ServiceConfig(dependencies=["Pillow"])
        self.assertEqual(ServiceConfig.model_validate(config.model_dump()).dependencies, ["Pillow"])
        service = PreprocessingService.__new__(PreprocessingService)
        job = {"job_id": "job", "revision": 1, "data": {"plan": {"dependencies": ["numpy"]}}}
        task = service.task(job, "probe", "worker", "digest")
        self.assertEqual(task.payload["dependencies"], ["numpy"])

    def test_nearest_manifest_and_requirements_precedence(self):
        with tempfile.TemporaryDirectory() as temporary:
            root = Path(temporary).resolve()
            nested = root / "project" / "src"
            nested.mkdir(parents=True)
            (root / "requirements.txt").write_text("numpy==1.0")
            project = nested.parent / "pyproject.toml"
            project.write_text('[project]\ndependencies = ["Pillow"]\n')
            self.assertEqual(manifest(root, nested), (project, ["Pillow"]))
            req = nested.parent / "requirements.txt"
            req.write_text("Pillow==11.0.0")
            self.assertEqual(manifest(root, nested), (req, ["-r", str(req)]))
            with self.assertRaisesRegex(ValueError, "working directory"):
                manifest(root, root.parent)


class DependencyExecutionTests(unittest.IsolatedAsyncioTestCase):
    async def execute(self, files, *, dependencies=None, working=".", reporter=None):
        async def fetch(_):
            return files

        task = TaskSpec(
            id="dependency-test",
            job_id="dependency-job",
            kind="python_project",
            requirements={"runtime": "cpu", "vram_mib": 0},
            timeout_seconds=30,
            max_attempts=1,
            payload={
                "mode": "map",
                "module": "simulation",
                "seeds": [1],
                "parameters": {},
                "working_directory": working,
                "dependencies": dependencies or [],
            },
        )
        reporter = reporter or Reporter()
        with patch.dict(os.environ, {"TEST_WORKER_SECRET": "must-not-inherit"}):
            result = await asyncio.wait_for(
                PythonProjectExecutor("ws://localhost", "worker", fetch=fetch).execute(
                    task, reporter
                ),
                30,
            )
        return result, reporter

    async def test_requirements_install_private_package_and_keep_worker_libraries(self):
        files = {
            "project/simulation.py": encoded(SOURCE),
            "project/requirements.txt": encoded("--no-index\n-r pinned.txt\n"),
            "project/pinned.txt": encoded(f"./{WHEEL_NAME}\n"),
            f"project/{WHEEL_NAME}": wheel(),
        }
        original = dict(files)
        result, reporter = await self.execute(files, working="project")
        self.assertTrue(result["ok"], reporter.text)
        value = result["items"][0]["value"]
        self.assertEqual(value["answer"], 43)
        self.assertIn("__dispatch_environment__", value["python"])
        self.assertIn("__dispatch_environment__", value["package"])
        self.assertFalse(Path(value["root"]).exists())
        self.assertEqual(files, original)
        self.assertGreater(result["metrics"]["dependency_seconds"], 0)
        self.assertTrue(any(kind == "cleaned" for kind, _ in reporter.events))
        self.assertIsNone(
            __import__("importlib.util", fromlist=["find_spec"]).find_spec("dispatch_fixture")
        )

    async def test_agent_selection_and_manifest_are_resolved_together(self):
        files = {
            "simulation.py": encoded(SOURCE),
            WHEEL_NAME: wheel(),
            "requirements.txt": encoded("--no-index\n--find-links .\n"),
        }
        result, reporter = await self.execute(files, dependencies=["dispatch-fixture==1.0"])
        self.assertTrue(result["ok"], reporter.text)
        self.assertIn("Agent-selected packages: dispatch-fixture==1.0", reporter.text)
        files["requirements.txt"] = encoded("--no-index\n--find-links .\ndispatch-fixture==1.0\n")
        result, reporter = await self.execute(files, dependencies=["dispatch-fixture==2.0"])
        self.assertFalse(result["ok"])
        self.assertIn("Dependency setup failed", result["error"])
        self.assertNotIn("Starting project", reporter.text)
        self.assertTrue(any(kind == "cleaned" for kind, _ in reporter.events))

    async def test_agent_only_plan_works_without_manifest(self):
        # pydantic exists only in the worker venv. Its inherited distribution must
        # satisfy the plan offline, without reinstalling into the worker itself.
        import pydantic

        files = {
            "simulation.py": encoded(
                "import pydantic, sys\ndef run(seed, parameters): return sys.executable\n"
            )
        }
        result, reporter = await self.execute(
            files, dependencies=[f"pydantic=={pydantic.__version__}"]
        )
        self.assertTrue(result["ok"], reporter.text)
        self.assertIn("__dispatch_environment__", result["items"][0]["value"])

    async def test_pyproject_dependencies_install_before_import(self):
        with tempfile.TemporaryDirectory() as directory:
            path = Path(directory) / WHEEL_NAME
            path.write_bytes(base64.b64decode(wheel()))
            files = {
                "simulation.py": encoded(SOURCE),
                "pyproject.toml": encoded(
                    '[project]\nname = "uploaded"\nversion = "1.0"\ndependencies = ['
                    + json.dumps("dispatch-fixture @ " + path.as_uri())
                    + "]\n"
                ),
            }
            result, reporter = await self.execute(files)
        self.assertTrue(result["ok"], reporter.text)
        self.assertEqual(result["items"][0]["value"]["answer"], 43)

    @unittest.skipIf(os.name == "nt", "POSIX process groups")
    async def test_cancel_during_install_reaps_build_children_and_workspace(self):
        marker_dir = tempfile.TemporaryDirectory()
        self.addCleanup(marker_dir.cleanup)
        marker = Path(marker_dir.name) / "build.json"
        backend = """import json, os, pathlib, subprocess, sys, time
def build_wheel(*args, **kwargs):
    child = subprocess.Popen([sys.executable, '-c', 'import time; time.sleep(60)'])
    pathlib.Path(MARKER).write_text(json.dumps({'root': os.environ['HOME'], 'pid': os.getpid(), 'child': child.pid}))
    time.sleep(60)
""".replace("MARKER", repr(str(marker)))
        files = {
            "simulation.py": encoded("raise AssertionError('must not execute')"),
            "requirements.txt": encoded("--no-index\n./package\n"),
            "package/pyproject.toml": encoded(
                '[build-system]\nrequires = []\nbuild-backend = "builder"\nbackend-path = ["."]\n'
            ),
            "package/builder.py": encoded(backend),
        }
        reporter = Reporter()
        task = asyncio.create_task(self.execute(files, reporter=reporter))
        try:
            async with asyncio.timeout(20):
                while not marker.exists():
                    if task.done():
                        self.fail(reporter.text)
                    await asyncio.sleep(0.05)
            info = json.loads(marker.read_text())
            task.cancel()
            with self.assertRaises(asyncio.CancelledError):
                await asyncio.wait_for(task, 5)
            self.assertFalse(Path(info["root"]).exists())
            from unit.test_python_project import dead

            for pid in (info["pid"], info["child"]):
                self.assertTrue(await dead(pid), f"Build process {pid} survived cancellation")
            self.assertTrue(any(kind == "cleaned" for kind, _ in reporter.events))
        finally:
            task.cancel()
            await asyncio.gather(task, return_exceptions=True)
