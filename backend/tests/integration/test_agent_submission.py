"""Automatic job/service decisions preserve scheduling, billing and cancellation."""

import asyncio
import os
import unittest
from types import SimpleNamespace
from unittest.mock import AsyncMock
from uuid import uuid4

from integration import test_hosted_services as hosted
from integration.test_preprocessing import proposal
from orchestrator.preprocessing.artifacts import encoded
from orchestrator.preprocessing.service import PreprocessingService
from orchestrator.shared.services import ServiceConfig


@unittest.skipUnless(os.getenv("RUN_SUPERVISOR_TESTS") == "1", "needs temporary PostgreSQL")
class AgentSubmissionTests(unittest.IsolatedAsyncioTestCase):
    setUpClass = classmethod(hosted.HostedServicesTests.setUpClass.__func__)
    tearDownClass = classmethod(hosted.HostedServicesTests.tearDownClass.__func__)
    server = hosted.HostedServicesTests.server
    worker = hosted.HostedServicesTests.worker
    cleanup = hosted.HostedServicesTests.cleanup
    ready = hosted.HostedServicesTests.ready

    async def asyncSetUp(self):
        await hosted.HostedServicesTests.asyncSetUp(self)
        self.model = SimpleNamespace(respond=AsyncMock())
        self.planner = PreprocessingService(self.store, self.model)
        self.apps["public"].state.preprocessing = self.planner
        await self.worker("worker-a")
        async with asyncio.timeout(10):
            while not await self.store.workers():
                await asyncio.sleep(0.05)

    async def submit(self, **changes):
        self.body = {
            "request_id": str(uuid4()),
            "execution_mode": "auto",
            "workload": "auto",
            "description": "Host this HTTP API for one hour. Choose its settings from the code.",
            "usage_cap": "2.50",
            "files": [{"name": "server.py", "content": encoded(hosted.SOURCE.read_text())}],
            **changes,
        }
        response = await self.client.post("/v1/jobs", json=self.body)
        self.assertEqual(response.status_code, 200, response.text)
        self.id = response.json()["spec"]["job_id"]
        self.jobs.append(self.id)

    async def step(self, name, value):
        self.model.respond.return_value = proposal(name, value)
        await self.store.pool.execute(
            "UPDATE simulation_jobs SET retry_after=clock_timestamp() WHERE job_id=$1", self.id
        )
        await self.planner.run_once(self.id)

    async def classify_service(self):
        await self.step(
            "classify_project",
            {
                "execution_mode": "service",
                "workload": "python",
                "rationale": "The user asked for a persistent HTTP API.",
            },
        )

    def config(self, runtime="cpu"):
        return ServiceConfig(
            entrypoint="server.py",
            readiness_path="/health",
            lifetime_seconds=3600,
            startup_timeout_seconds=15,
            requirements={"runtime": runtime, "vram_mib": 0},
        ).model_dump(mode="json")

    async def test_agent_selected_dependency_is_installed_before_service_readiness(self):
        from unit.test_dependency_setup import WHEEL_NAME, wheel

        await self.submit(
            files=[
                {
                    "name": "server.py",
                    "content": encoded(
                        "import dispatch_fixture\nassert dispatch_fixture.VALUE == 42\n"
                        + hosted.SOURCE.read_text()
                    ),
                },
                {"name": "requirements.txt", "content": encoded("--no-index\n--find-links .\n")},
                {"name": WHEEL_NAME, "content": wheel()},
            ]
        )
        await self.classify_service()
        await self.step(
            "plan_service",
            {
                "summary": "Host the API with its imported distribution.",
                "config": {**self.config(), "dependencies": ["dispatch-fixture==1.0"]},
            },
        )
        status = await self.ready(self.id)
        self.assertEqual(status["service"]["config"]["dependencies"], ["dispatch-fixture==1.0"])
        self.assertEqual((await self.client.get("/serve/" + self.id + "/health")).status_code, 200)

    async def test_automatic_service_starts_on_real_worker_and_retry_preserves_identity_and_cap(
        self,
    ):
        await self.submit()
        await self.classify_service()
        await self.step(
            "plan_service",
            {
                "summary": "Host server.py on the CPU with its /health endpoint.",
                "config": self.config(),
            },
        )
        status = await self.ready(self.id)
        self.assertEqual(status["workers"], ["worker-a"])
        self.assertEqual(status["execution_mode"], "service")
        self.assertEqual(status["service"]["config"]["lifetime_seconds"], 3600)
        self.assertEqual(
            [d["tool"] for d in status["decisions"]], ["classify_project", "plan_service"]
        )
        self.assertIsNone(
            await self.store.pool.fetchval(
                "SELECT job_id FROM simulation_jobs WHERE job_id=$1", self.id
            )
        )
        self.assertEqual(
            str(
                await self.store.pool.fetchval(
                    "SELECT usage_cap_cad FROM supervised_jobs WHERE id=$1", self.id
                )
            ),
            "2.500000",
        )
        retry = await self.client.post("/v1/jobs", json=self.body)
        self.assertEqual(retry.status_code, 200, retry.text)
        self.assertEqual(retry.json()["spec"]["job_id"], self.id)
        conflict = await self.client.post(
            "/v1/jobs", json={**self.body, "description": "Different request"}
        )
        self.assertEqual(conflict.status_code, 409)

    async def test_unsupported_gpu_plan_does_not_start_service(self):
        await self.submit()
        await self.classify_service()
        config = self.config()
        config["requirements"] = {"runtime": "cuda", "vram_mib": 0}
        await self.step("plan_service", {"summary": "Use CUDA", "config": config})
        self.assertFalse(await self.services.exists(self.id))
        job = await self.planner.store.job(self.id)
        self.assertIn("cpu", job["data"]["last_planning_error"])

    async def test_clarification_resumes_planning_without_manual_config(self):
        await self.submit(description="Do something with this project")
        await self.step(
            "ask_user", {"question": "Should the API stay running or do you want a one-off test?"}
        )
        status = (await self.client.get(f"/v1/jobs/{self.id}")).json()
        self.assertEqual(status["phase"], "needs_input")
        self.assertFalse(await self.services.exists(self.id))
        answer = await self.client.post(
            f"/v1/jobs/{self.id}/answer", json={"message": "Keep the API running for one hour."}
        )
        self.assertEqual(answer.status_code, 200, answer.text)
        await self.classify_service()
        self.assertEqual((await self.planner.store.job(self.id))["phase"], "service_planning")

    async def test_explicit_job_cannot_be_promoted_to_service(self):
        await self.submit(execution_mode="job")
        await self.classify_service()
        self.assertFalse(await self.services.exists(self.id))
        self.assertIn(
            "explicitly requests a job",
            (await self.planner.store.job(self.id))["data"]["last_planning_error"],
        )

    async def test_cancellation_during_service_planning_prevents_start(self):
        await self.submit()
        await self.classify_service()

        async def choose(*args, **kwargs):
            await self.store.cancel(self.id)
            return proposal("plan_service", {"summary": "Host server", "config": self.config()})

        self.model.respond.side_effect = choose
        await self.step("plan_service", {})
        self.assertFalse(await self.services.exists(self.id))
        self.assertEqual((await self.store.task(self.id)).state, "cancelled")

    async def test_agent_can_choose_finite_job_from_same_simple_submission(self):
        await self.submit(description="Run a one-off calculation and return the results")
        await self.step(
            "classify_project",
            {"execution_mode": "job", "workload": "python", "rationale": "The request is finite."},
        )
        job = await self.planner.store.job(self.id)
        self.assertEqual(job["phase"], "program_planning")
        self.assertEqual(job["data"]["execution_mode"], "job")
        self.assertFalse(await self.services.exists(self.id))
