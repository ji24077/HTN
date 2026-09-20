"""Real PostgreSQL, split HTTP/WS gateways, and serving subprocess integration."""

import asyncio
import os
import socket
import tempfile
import time
import unittest
from pathlib import Path
from types import SimpleNamespace
from unittest.mock import patch
from uuid import uuid4

import httpx
import uvicorn

from integration import test_preprocessing as legacy
from orchestrator.preprocessing.artifacts import encoded
from orchestrator.server.app import create_app
from orchestrator.server.db.store import Store
from orchestrator.server.scheduler import reconcile_loop
from orchestrator.server.services import ServiceStore
from orchestrator.shared.protocol import Capabilities, TaskSpec
from orchestrator.worker.agent import Agent
from orchestrator.worker.config import WorkerConfig
from orchestrator.worker.executors.python_project import PythonProjectExecutor

SOURCE = Path(__file__).resolve().parents[3] / "examples/projects/service/server.py"


@unittest.skipUnless(os.getenv("RUN_SUPERVISOR_TESTS") == "1", "needs temporary PostgreSQL")
class HostedServicesTests(unittest.IsolatedAsyncioTestCase):
    setUpClass = classmethod(legacy.PreprocessingTests.setUpClass.__func__)
    tearDownClass = classmethod(legacy.PreprocessingTests.tearDownClass.__func__)

    async def asyncSetUp(self):
        self.directory = tempfile.TemporaryDirectory()
        self.addCleanup(self.directory.cleanup)
        self.env = patch.dict(
            os.environ,
            {
                "WORKER_EXECUTION_DIR": self.directory.name,
                "SERVICE_BRIDGE_TOKEN": "test-bridge-secret",
            },
        )
        self.env.start()
        self.addCleanup(self.env.stop)
        self.store = await Store.open(self.database.get_uri(), schema="test_" + uuid4().hex)
        self.addAsyncCleanup(self.store.close)
        self.services = ServiceStore(self.store)
        self.jobs = []
        self.runners = []
        self.servers = []
        self.apps = {}
        self.addAsyncCleanup(self.cleanup)
        self.worker_origin = await self.server("worker")
        os.environ["SERVICE_BRIDGE_URL"] = self.worker_origin
        self.public_origin = await self.server("public")
        self.client = httpx.AsyncClient(
            base_url=self.public_origin, timeout=15, headers={"Authorization": "Bearer test-admin"}
        )
        self.addAsyncCleanup(self.client.aclose)
        self.reconciler = asyncio.create_task(reconcile_loop(self.store))
        self.runners.append(self.reconciler)

    async def server(self, surface, port=0):
        app = create_app(surface)
        self.apps[surface] = app
        app.state.store, app.state.cache = self.store, None
        app.state.config = SimpleNamespace(
            admin_token="test-admin",
            worker_tokens={"worker-a": "test-worker-a", "worker-b": "test-worker-b"},
            public_origin="",
        )
        sock = socket.socket()
        sock.setsockopt(socket.SOL_SOCKET, socket.SO_REUSEADDR, 1)
        sock.bind(("127.0.0.1", port))
        port = sock.getsockname()[1]
        server = uvicorn.Server(
            uvicorn.Config(app, lifespan="off", log_level="critical", timeout_graceful_shutdown=2)
        )
        run = asyncio.create_task(server.serve(sockets=[sock]))
        self.servers.append((server, run, sock))
        for _ in range(100):
            if server.started:
                return f"http://127.0.0.1:{port}"
            await asyncio.sleep(0.02)
        self.fail("HTTP server did not start")

    async def worker(self, name):
        url = self.worker_origin.replace("http:", "ws:") + "/v1/worker"
        executor = PythonProjectExecutor(url, name)
        config = WorkerConfig(
            url,
            name,
            "test-" + name,
            Capabilities(runtime="cpu", vram_mib=0, kinds=list(executor.kinds)),
        )
        agent = Agent(config, executor)
        run = asyncio.create_task(agent.run())
        self.runners.append(run)
        return run

    async def cleanup(self):
        for job in self.jobs:
            await self.store.cancel(job)
        for run in self.runners:
            run.cancel()
        await asyncio.gather(*self.runners, return_exceptions=True)
        for server, run, sock in self.servers:
            server.should_exit = True
        for server, run, sock in self.servers:
            await run
            sock.close()

    async def upload(self, **service):
        body = {
            "request_id": str(uuid4()),
            "execution_mode": "service",
            "description": "Host this HTTP server",
            "service": {"startup_timeout_seconds": 15, **service},
            "files": [{"name": "server.py", "content": encoded(SOURCE.read_text())}],
        }
        response = await self.client.post("/v1/jobs", json=body)
        self.assertEqual(response.status_code, 200, response.text)
        job = response.json()["spec"]["job_id"]
        self.jobs.append(job)
        return job, body

    async def ready(self, job):
        for _ in range(150):
            response = await self.client.get("/serve/" + job + "/health")
            if response.status_code == 200:
                return await self.services.status(job)
            await asyncio.sleep(0.1)
        self.fail(str((response.status_code, response.text, await self.services.status(job))))

    async def test_split_gateway_streaming_restart_and_stop(self):
        worker = await self.worker("worker-a")
        job, body = await self.upload()
        status = await self.ready(job)
        self.assertEqual(status["workers"], ["worker-a"])
        self.assertEqual(
            await self.store.pool.fetchval(
                "SELECT new_state FROM events WHERE entity_id=$1 AND details->>'phase'='ready' ORDER BY id DESC LIMIT 1",
                job,
            ),
            "running",
        )
        self.assertIsNone((await self.store.task(status["tasks"][-1]["id"])).deadline)
        duplicate = await self.client.post("/v1/jobs", json=body)
        self.assertEqual(duplicate.status_code, 200)
        body["service"]["readiness_path"] = "/different"
        self.assertEqual((await self.client.post("/v1/jobs", json=body)).status_code, 409)
        pid = int((await self.client.get("/serve/" + job + "/pid")).text)
        started = time.monotonic()
        async with self.client.stream("GET", "/serve/" + job + "/stream") as response:
            self.assertEqual(response.status_code, 200)
            iterator = response.aiter_bytes()
            first = await anext(iterator)
            self.assertIn(b"data: 0", first)
            self.assertLess(time.monotonic() - started, 0.4)
            rest = b"".join([part async for part in iterator])
            self.assertIn(b"data: 2", rest)
        payload = bytes(range(256)) * 3000
        response = await self.client.post("/serve/" + job + "/echo", content=payload)
        self.assertEqual(response.content, payload)
        self.assertIn("sandbox", response.headers["content-security-policy"])
        # A worker loss is recovered without any model client.
        worker.cancel()
        await asyncio.gather(worker, return_exceptions=True)
        await self.worker("worker-b")
        await self.store.pool.execute(
            "UPDATE tasks SET lease_until=clock_timestamp()-interval '1 second' WHERE spec->>'job_id'=$1 AND worker_id='worker-a'",
            job,
        )
        status = await self.ready(job)
        self.assertEqual(status["workers"], ["worker-b"])
        self.assertGreaterEqual(status["service"]["restarts"], 1)
        with self.assertRaises(ProcessLookupError):
            os.kill(pid, 0)
        action = {"action_id": str(uuid4()), "operation": "stop"}
        self.assertEqual(
            (
                await self.client.post("/v1/jobs/" + job + "/service-actions", json=action)
            ).status_code,
            200,
        )
        self.assertEqual(
            (
                await self.client.post("/v1/jobs/" + job + "/service-actions", json=action)
            ).status_code,
            200,
        )
        self.assertEqual((await self.client.get("/serve/" + job + "/health")).status_code, 503)
        self.assertEqual((await self.services.status(job))["phase"], "stopped")

    async def test_auth_limits_and_expiry(self):
        await self.worker("worker-a")
        job, _ = await self.upload(lifetime_seconds=30)
        await self.ready(job)
        async with httpx.AsyncClient(base_url=self.public_origin) as anonymous:
            self.assertEqual((await anonymous.get("/serve/" + job + "/health")).status_code, 401)
        self.assertEqual(
            (
                await self.client.post("/serve/" + job + "/echo", content=b"x" * (1024 * 1024 + 1))
            ).status_code,
            413,
        )
        await self.store.pool.execute(
            "UPDATE hosted_services SET expires_at=clock_timestamp()-interval '1 second' WHERE job_id=$1",
            job,
        )
        self.assertEqual((await self.client.get("/serve/" + job + "/health")).status_code, 503)
        await self.services.reconcile()
        self.assertEqual((await self.services.status(job))["phase"], "stopped")

    async def test_disconnect_backpressure_headers_and_stale_identity(self):
        from websockets.asyncio.client import connect
        from websockets.exceptions import InvalidStatus

        await self.worker("worker-a")
        job, _ = await self.upload(concurrency=1)
        status = await self.ready(job)
        response = await self.client.get(
            "/serve/" + job + "/headers", headers={"Cookie": "private=secret"}
        )
        self.assertNotIn("set-cookie", response.headers)
        self.assertEqual(response.headers.get_list("link"), ["</a>; rel=first", "</b>; rel=next"])
        self.assertFalse(
            any(
                k.lower() in {"authorization", "cookie", "x-dispatch-bridge"}
                for k in response.json()
            )
        )
        async with self.client.stream("GET", "/serve/" + job + "/slow") as response:
            self.assertEqual(response.status_code, 200)
            chunks = response.aiter_bytes()
            await anext(chunks)
            overloaded = await self.client.get("/serve/" + job + "/health")
            self.assertEqual(overloaded.status_code, 429)
        # Disconnect releases the only slot promptly, not at the long response timeout.
        await self.ready(job)
        task = await self.store.task(status["tasks"][-1]["id"])
        with self.assertRaises(InvalidStatus):
            async with connect(
                self.worker_origin.replace("http:", "ws:") + "/v1/service-data/" + task.spec.id,
                additional_headers={
                    "Authorization": "Bearer " + task.spec.payload["artifact_token"],
                    "X-Worker-ID": "worker-a",
                    "X-Task-Attempt": str(task.generation + 1),
                    "X-Worker-Session": task.session_id,
                },
            ):
                self.fail("Stale attempt attached")
        await self.services.health(task.spec.id, task.generation, task.session_id, False)
        self.assertEqual((await self.client.get("/serve/" + job + "/health")).status_code, 503)

    async def test_full_gateway_restart_recovers_with_new_session_and_process(self):
        from urllib.parse import urlsplit

        await self.worker("worker-a")
        job, _ = await self.upload()
        await self.ready(job)
        pid = int((await self.client.get("/serve/" + job + "/pid")).text)
        server, run, _ = self.servers[0]
        server.should_exit = True
        await asyncio.wait_for(run, 5)
        await self.server("worker", urlsplit(self.worker_origin).port)
        await self.ready(job)
        self.assertNotEqual(int((await self.client.get("/serve/" + job + "/pid")).text), pid)
        with self.assertRaises(ProcessLookupError):
            os.kill(pid, 0)

    async def test_failed_health_probe_replaces_process_while_agent_stays_connected(self):
        await self.worker("worker-a")
        job, _ = await self.upload()
        await self.ready(job)
        pid = int((await self.client.get("/serve/" + job + "/pid")).text)
        await self.client.post("/serve/" + job + "/unhealthy")
        for _ in range(50):
            if (await self.client.get("/serve/" + job + "/pid")).status_code == 503:
                break
            await asyncio.sleep(0.1)
        else:
            self.fail("Failed readiness did not withdraw routing")
        await self.ready(job)
        self.assertNotEqual(int((await self.client.get("/serve/" + job + "/pid")).text), pid)

    async def test_startup_timeout_kills_process_and_stop_suppresses_replacement(self):
        await self.worker("worker-a")
        pid_file = Path(self.directory.name) / "startup.pid"
        source = f"import os,time,pathlib; pathlib.Path({str(pid_file)!r}).write_text(str(os.getpid())); time.sleep(60)"
        body = {
            "request_id": str(uuid4()),
            "execution_mode": "service",
            "description": "A server that never binds",
            "service": {"startup_timeout_seconds": 1},
            "files": [{"name": "server.py", "content": encoded(source)}],
        }
        response = await self.client.post("/v1/jobs", json=body)
        job = response.json()["spec"]["job_id"]
        self.jobs.append(job)
        for _ in range(100):
            if pid_file.exists():
                break
            await asyncio.sleep(0.05)
        pid = int(pid_file.read_text())
        for _ in range(100):
            status = await self.services.status(job)
            if status["tasks"][0]["state"] in {"failed", "cancelled"}:
                break
            await asyncio.sleep(0.05)
        else:
            self.fail("Startup timeout did not revoke attempt")
        await self.store.cancel(job)
        for _ in range(100):
            try:
                os.kill(pid, 0)
            except ProcessLookupError:
                break
            await asyncio.sleep(0.05)
        else:
            self.fail("Timed-out service process was not killed")
        await self.services.reconcile()
        self.assertEqual((await self.services.status(job))["phase"], "stopped")

    async def test_control_reconnect_keeps_process_and_health_bypasses_scheduler_lock(self):
        from orchestrator.server.db.store import Conflict
        from orchestrator.server.service_gateway import close_socket
        from orchestrator.supervisor.models import Action
        from orchestrator.supervisor.store import SupervisorStore

        await self.worker("worker-a")
        job, _ = await self.upload()
        status = await self.ready(job)
        task = await self.store.task(status["service"]["task_id"])
        pid = int((await self.client.get("/serve/" + job + "/pid")).text)
        async with self.store.change():
            # A stable ready report must complete without acquiring the advisory lock again.
            await asyncio.wait_for(
                self.services.health(task.spec.id, task.generation, task.session_id, True), 1
            )
        gateway = self.apps["worker"].state.service_gateway
        await close_socket(next(iter(gateway.controls)))
        await asyncio.sleep(0.2)
        await self.ready(job)
        self.assertEqual(int((await self.client.get("/serve/" + job + "/pid")).text), pid)
        self.assertEqual((await self.services.status(job))["service"]["restarts"], 0)
        with self.assertRaises(Conflict):
            await SupervisorStore(self.store).action(
                job, Action(action_id=uuid4(), operation="pause_job", reason="pause service")
            )
        # Generic cancellation of the currently running attempt must stop, not restart.
        await self.store.cancel(task.spec.id)
        await self.services.reconcile()
        self.assertEqual((await self.services.status(job))["phase"], "stopped")

    async def test_successful_request_and_disconnect_do_not_log_double_close(self):
        await self.worker("worker-a")
        job, _ = await self.upload()
        await self.ready(job)
        with self.assertNoLogs("uvicorn.error", level="ERROR"):
            for _ in range(3):
                response = await self.client.get("/serve/" + job + "/health")
                self.assertEqual(response.status_code, 200)
            async with self.client.stream("GET", "/serve/" + job + "/slow") as response:
                chunks = response.aiter_bytes()
                await anext(chunks)
            await asyncio.sleep(0.2)

    async def test_disconnect_before_headers_releases_capacity(self):
        await self.worker("worker-a")
        job, _ = await self.upload(concurrency=1)
        await self.ready(job)
        with self.assertRaises(httpx.ReadTimeout):
            await self.client.get("/serve/" + job + "/delayed", timeout=0.3)
        started = time.monotonic()
        await self.ready(job)
        self.assertLess(time.monotonic() - started, 2)

    async def test_slow_startup_does_not_repeat_lifecycle_events(self):
        await self.worker("worker-a")
        body = {
            "request_id": str(uuid4()),
            "execution_mode": "service",
            "description": "Load model before serving",
            "service": {"startup_timeout_seconds": 20},
            "files": [
                {
                    "name": "server.py",
                    "content": encoded("import time; time.sleep(5)\n" + SOURCE.read_text()),
                }
            ],
        }
        response = await self.client.post("/v1/jobs", json=body)
        job = response.json()["spec"]["job_id"]
        self.jobs.append(job)
        await self.ready(job)
        count = await self.store.pool.fetchval(
            "SELECT count(*) FROM events WHERE entity_id=$1 AND details->>'phase'='starting'", job
        )
        self.assertLessEqual(count, 2)

    async def test_clarification_crash_loop_and_manual_restart(self):
        from orchestrator.shared.services import ServiceAction

        body = {
            "request_id": str(uuid4()),
            "execution_mode": "service",
            "description": "Serve a project",
            "files": [
                {"name": "server.py", "content": encoded(SOURCE.read_text())},
                {"name": "helpers.py", "content": encoded("value = 1")},
            ],
        }
        response = await self.client.post("/v1/jobs", json=body)
        job = response.json()["spec"]["job_id"]
        self.jobs.append(job)
        self.assertEqual((await self.services.status(job))["phase"], "needs_input")
        self.assertEqual(
            (
                await self.client.post(
                    "/v1/jobs/" + job + "/answer", json={"message": "../outside.py"}
                )
            ).status_code,
            400,
        )
        self.assertEqual(
            (
                await self.client.post("/v1/jobs/" + job + "/answer", json={"message": "server.py"})
            ).status_code,
            200,
        )
        for _ in range(5):
            await self.store.pool.execute(
                "UPDATE hosted_services SET retry_after=clock_timestamp() WHERE job_id=$1", job
            )
            await self.services.reconcile()
            await self.store.pool.execute(
                "UPDATE tasks SET state='failed',failure='fixture startup failure' WHERE id=(SELECT task_id FROM hosted_services WHERE job_id=$1)",
                job,
            )
            await self.services.reconcile()
        self.assertEqual((await self.services.status(job))["phase"], "failed")
        count = len((await self.services.status(job))["tasks"])
        await self.services.reconcile()
        self.assertEqual(len((await self.services.status(job))["tasks"]), count)
        restart = ServiceAction(action_id=uuid4(), operation="restart")
        await asyncio.gather(self.services.action(job, restart), self.services.action(job, restart))
        await asyncio.gather(self.services.reconcile(), self.services.reconcile())
        active = await self.store.pool.fetchval(
            "SELECT count(*) FROM tasks WHERE spec->>'job_id'=$1 AND id!=$1 AND state IN ('queued','assigned','running')",
            job,
        )
        self.assertEqual(active, 1)


class ServiceContractTests(unittest.TestCase):
    def test_header_validation_rejects_malformed_frames(self):
        from orchestrator.server.service_gateway import valid_response_headers

        self.assertTrue(valid_response_headers([["Content-Type", "application/json"]]))
        for value in [
            [["X-Test", "one\r\nInjected: two"]],
            [["Bad Header", "value"]],
            [["x", 3]],
            ["invalid"],
            [["x", "y", "z"]],
        ]:
            self.assertFalse(valid_response_headers(value))

    def test_only_services_can_omit_deadline(self):
        fields = dict(
            id="one",
            job_id="one",
            payload={},
            requirements={"runtime": "cpu", "vram_mib": 0},
            max_attempts=1,
        )
        with self.assertRaises(ValueError):
            TaskSpec(kind="stub", **fields)
        self.assertIsNone(TaskSpec(kind="python_service", **fields).timeout_seconds)
