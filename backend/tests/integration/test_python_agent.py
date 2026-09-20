"""Real paired Docker agent, CPU PyTorch, public transfers and temporary PostgreSQL.

Build dwp-agent:python-cpu, then opt in with RUN_PYTHON_AGENT_E2E=1.
Planner proposals are controlled fixtures; no model API or production data is used.
"""

import asyncio
import base64
import hashlib
import json
import os
import secrets
import socket
import tempfile
import unittest
from contextlib import asynccontextmanager
from pathlib import Path
from types import SimpleNamespace
from unittest.mock import AsyncMock
from uuid import uuid4

import httpx
import uvicorn

from integration.test_preprocessing import proposal
from orchestrator.preprocessing.service import PreprocessingService
from orchestrator.server.app import create_app
from orchestrator.server.config import ServerConfig
from orchestrator.server.db.store import Store
from orchestrator.server.scheduler import reconcile_loop

ROOT = Path(__file__).resolve().parents[3]


@unittest.skipUnless(os.getenv("RUN_PYTHON_AGENT_E2E") == "1", "needs the Python Docker image")
class PythonAgentTests(unittest.IsolatedAsyncioTestCase):
    async def docker(self, *args, check=True):
        process = await asyncio.create_subprocess_exec(
            "docker", *args, stdout=asyncio.subprocess.PIPE, stderr=asyncio.subprocess.PIPE
        )
        out, err = await process.communicate()
        if check and process.returncode:
            self.fail(f"Docker command failed: {err.decode(errors='replace')}")
        return out.decode(errors="replace")

    async def asyncSetUp(self):
        import pgserver

        self.directory = tempfile.TemporaryDirectory(prefix="python-agent-e2e-")
        self.database = pgserver.get_server(
            Path(self.directory.name) / "postgres", cleanup_mode="stop"
        )
        self.store = await Store.open(self.database.get_uri(), schema="python_agent")
        self.destination = Path(
            os.getenv("PYTHON_AGENT_OUTPUT_DIR", str(ROOT / ".local/python-agent-smoke"))
        )
        self.destination.mkdir(parents=True, exist_ok=True)
        self.container = "dispatch-python-test-" + uuid4().hex[:12]
        self.model = SimpleNamespace(respond=AsyncMock(side_effect=self.choose))
        self.contexts = []
        self.service = PreprocessingService(self.store, self.model)
        self.app = create_app("public")

        @asynccontextmanager
        async def lifespan(_):
            yield

        self.app.router.lifespan_context = lifespan
        self.admin = secrets.token_urlsafe(32)
        self.app.state.config = ServerConfig(self.database.get_uri(), None, self.admin, {})
        self.app.state.store = self.store
        self.app.state.preprocessing = self.service
        self.app.state.device_connections = {}
        self.app.state.cache = None
        sock = socket.socket()
        sock.bind(("0.0.0.0", 0))
        self.port = sock.getsockname()[1]
        self.server = uvicorn.Server(
            uvicorn.Config(self.app, log_level="warning", ws="websockets", ws_max_size=131072)
        )
        self.server_task = asyncio.create_task(self.server.serve(sockets=[sock]))
        self.reconciler = asyncio.create_task(reconcile_loop(self.store))
        self.client = httpx.AsyncClient(
            base_url=f"http://127.0.0.1:{self.port}",
            headers={"Authorization": "Bearer " + self.admin},
            timeout=120,
        )
        self.addAsyncCleanup(self.cleanup)
        async with asyncio.timeout(15):
            while not self.server.started:
                await asyncio.sleep(0.05)
        invite = await self.client.post("/v1/device-invites")
        invite.raise_for_status()
        self.image = os.getenv("PYTHON_AGENT_IMAGE", "dwp-agent:python-cpu")
        await self.docker(
            "run",
            "-d",
            "--name",
            self.container,
            "--cpus",
            "4",
            "--memory",
            "4g",
            "--add-host",
            "host.docker.internal:host-gateway",
            "-e",
            f"DWP_SERVER=http://host.docker.internal:{self.port}",
            "-e",
            "DWP_CODE=" + invite.json()["code"],
            "-e",
            "DWP_LABEL=Python container test",
            "-e",
            "DWP_DNS_FALLBACK=0",
            self.image,
            "run",
        )
        async with asyncio.timeout(120):
            while True:
                workers = await self.store.workers()
                if workers and "python_program" in workers[0].capabilities.kinds:
                    self.worker = workers[0].id
                    break
                await asyncio.sleep(0.5)

    async def cleanup(self):
        logs = await self.docker("logs", self.container, check=False)
        (self.destination / "agent.log").write_text(logs)
        await self.docker("rm", "-f", "-v", self.container, check=False)
        await self.client.aclose()
        self.reconciler.cancel()
        await asyncio.gather(self.reconciler, return_exceptions=True)
        self.server.should_exit = True
        await self.server_task
        await self.store.close()
        self.database.cleanup()
        self.directory.cleanup()

    async def choose(self, messages, **kwargs):
        name = kwargs["tools"][0]["name"]
        self.contexts.append((name, json.loads(messages[0]["content"])))
        if name == "classify_project":
            return proposal(
                name, {"workload": "training", "rationale": "Train the CPU PyTorch model."}
            )
        if name == "plan_program":
            return proposal(name, self.plan)
        if name == "place_program":
            return proposal(name, {"worker_id": self.worker, "rationale": "The CPU probe passed."})
        self.fail("Unexpected planner call: " + name)

    async def test_cpu_training_dependencies_validation_and_signed_outputs(self):
        self.plan = {
            "summary": "Train and validate the uploaded CPU model; torch is inferred from imports.",
            "dependencies": ["torch"],
            "worker_id": self.worker,
            "requirements": {"runtime": "cpu", "vram_mib": 0},
            "entrypoint": "train.py",
            "working_directory": ".",
            "probe_args": ["--steps", "2"],
            "run_args": ["--steps", "200"],
            "probe_timeout_seconds": 60,
            "run_timeout_seconds": 60,
            "validator": "validate.py",
            "validation_args": [],
            "outputs": [
                {"path": "checkpoint.pt", "kind": "checkpoint"},
                {"path": "metrics.json", "kind": "file"},
            ],
            "metrics": [{"name": "mse", "minimum": None, "maximum": 0.001}],
        }
        workers = await self.store.workers()
        self.assertEqual(workers[0].capabilities.runtime, "cpu")
        self.assertIn("+cpu", workers[0].capabilities.python.pytorch)
        files = [
            {"name": path.name, "content": base64.b64encode(path.read_bytes()).decode()}
            for path in (ROOT / "examples/projects/pytorch").glob("*.py")
        ]
        response = await self.client.post(
            "/v1/jobs",
            json={
                "request_id": str(uuid4()),
                "workload": "auto",
                "execution_mode": "auto",
                "description": "Train on CPU for 200 steps, probe with 2, and validate held-out MSE <= 0.001.",
                "files": files,
                "max_adaptations": 1,
                "max_runtime_seconds": 180,
            },
        )
        self.assertEqual(response.status_code, 200, response.text)
        job_id = response.json()["spec"]["job_id"]
        async with asyncio.timeout(150):
            while True:
                await self.store.pool.execute(
                    "UPDATE simulation_jobs SET retry_after=clock_timestamp() WHERE job_id=$1",
                    job_id,
                )
                await self.service.run_once(job_id)
                job = await self.service.store.job(job_id)
                if job["phase"] in {"completed", "failed", "needs_input"}:
                    break
                await asyncio.sleep(0.25)
        self.assertEqual(job["phase"], "completed", job["data"])
        self.assertEqual(job["data"]["program_plan"]["dependencies"], ["torch"])
        listing = (await self.client.get(f"/v1/jobs/{job_id}/outputs")).json()["files"]
        self.assertEqual({item["name"] for item in listing}, {"checkpoint.pt", "metrics.json"})
        for item in listing:
            response = await self.client.get(f"/v1/jobs/{job_id}/outputs/{item['id']}")
            self.assertEqual(response.status_code, 200)
            self.assertEqual(hashlib.sha256(response.content).hexdigest(), item["sha256"])
            (self.destination / item["name"]).write_bytes(response.content)
            if item["name"] == "metrics.json":
                metrics = response.json()
                self.assertEqual(metrics["device"], "cpu")
                self.assertLessEqual(metrics["mse"], 0.001)
        tasks = await self.store.pool.fetch(
            "SELECT id,spec,result,attestation FROM tasks WHERE spec->>'job_id'=$1 AND spec->>'kind'='python_project'",
            job_id,
        )
        self.assertEqual(len(tasks), 2)
        for task in tasks:
            self.assertTrue(task["attestation"])
            self.assertEqual(task["spec"]["payload"]["dependencies"], ["torch"])
            self.assertGreater(task["result"]["metrics"]["dependency_seconds"], 0)
        denied = await self.client.get(
            f"/agent/v1/execution-bundles/{tasks[0]['id']}/{job['original_hash']}"
        )
        self.assertEqual(denied.status_code, 403)
