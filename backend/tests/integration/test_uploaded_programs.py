"""Real program execution, persisted outputs, and the uploaded-job HTTP boundary."""

import hashlib
import json
import os
import tempfile
import unittest
from pathlib import Path
from types import SimpleNamespace
from unittest.mock import patch
from uuid import uuid4

import httpx

from integration import test_preprocessing as legacy
from orchestrator.preprocessing.artifacts import encoded
from orchestrator.server.app import create_app
from orchestrator.shared.protocol import Capabilities, task_ref
from orchestrator.supervisor.tools import SupervisorTools
from orchestrator.worker.agent import execute
from orchestrator.worker.executors.python_project import PythonProjectExecutor

EXAMPLES = Path(__file__).resolve().parents[3] / "examples/projects"


def plan(workload):
    training = workload == "training"
    return {
        "summary": "Run the original project and its uploaded validator.",
        "worker_id": "worker-b",
        "requirements": {"runtime": "cpu", "vram_mib": 0},
        "entrypoint": "train.py" if training else "render.py",
        "working_directory": ".",
        "probe_args": ["--steps", "2"] if training else ["--size", "16"],
        "run_args": ["--steps", "200"] if training else ["--size", "256"],
        "probe_timeout_seconds": 20,
        "run_timeout_seconds": 40,
        "validator": "validate.py",
        "validation_args": [],
        "outputs": (
            [
                {"path": "checkpoint.json", "kind": "checkpoint"},
                {"path": "metrics.json", "kind": "file"},
            ]
            if training
            else [{"path": "mandelbrot.ppm", "kind": "image"}]
        ),
        "metrics": [{"name": "mse", "minimum": None, "maximum": 0.001}] if training else [],
    }


@unittest.skipUnless(os.getenv("RUN_SUPERVISOR_TESTS") == "1", "needs temporary PostgreSQL")
class UploadedProgramTests(unittest.IsolatedAsyncioTestCase):
    setUpClass = classmethod(legacy.PreprocessingTests.setUpClass.__func__)
    tearDownClass = classmethod(legacy.PreprocessingTests.tearDownClass.__func__)

    async def asyncSetUp(self):
        await legacy.PreprocessingTests.asyncSetUp(self)
        for worker in ("worker-a", "worker-b"):
            await self.store.register(
                worker,
                worker,
                Capabilities(runtime="cpu", vram_mib=0, kinds=["python_project", "python_program"]),
            )
        app = self.app = create_app()
        app.state.store = self.store
        app.state.preprocessing = self.service
        app.state.config = SimpleNamespace(admin_token="program-test-token", worker_tokens={})
        self.client = httpx.AsyncClient(
            transport=httpx.ASGITransport(app=app), base_url="http://localhost"
        )
        self.addAsyncCleanup(self.client.aclose)
        self.admin = {"Authorization": "Bearer program-test-token"}
        self.contexts = []
        self.uploads = []

    async def start(
        self, workload="rendering", *, automatic=False, override=None, folder=None, extra_files=()
    ):
        self.workload = workload
        self.proposal = {**plan(workload), **(override or {})}

        async def choose(messages, **kwargs):
            name = kwargs["tools"][0]["name"]
            self.contexts.append((name, json.loads(messages[0]["content"])))
            if name == "classify_project":
                value = {"workload": workload, "rationale": "The project writes a deliverable."}
            elif name == "plan_program":
                value = self.proposal
            elif name == "place_program":
                value = {
                    "worker_id": "worker-a",
                    "rationale": "Use compatible available capacity after the probe.",
                }
            else:
                raise AssertionError(name)
            return legacy.proposal(name, value)

        self.model.respond.side_effect = choose
        body = {
            "request_id": str(uuid4()),
            "description": "Render 256x256"
            if workload == "rendering"
            else "Train 200 steps; held-out MSE <= 0.001",
            "files": [
                {"name": path.name, "content": encoded(path.read_text())}
                for path in (EXAMPLES / (folder or workload)).glob("*.py")
            ]
            + list(extra_files),
        }
        if not automatic:
            body["workload"] = workload
        response = await self.client.post("/v1/jobs", json=body, headers=self.admin)
        self.assertEqual(response.status_code, 200, response.text)
        self.id = response.json()["spec"]["job_id"]
        self.submission = body

    async def step(self, workers=("worker-a", "worker-b")):
        await self.store.pool.execute(
            "UPDATE simulation_jobs SET retry_after=clock_timestamp() WHERE job_id=$1", self.id
        )
        for worker in workers:
            await self.store.heartbeat(worker, worker, [], False)
        await self.service.run_once(self.id)
        for worker in workers:
            task = await self.store.claim(worker, worker)
            if not task:
                continue
            await self.store.ack(worker, worker, task_ref(task))

            async def fetch(spec):
                return await self.service.store.files(spec.job_id, spec.payload["bundle_hash"])

            async def publish(spec, report, name, path, worker=worker, task=task):
                content = path.read_bytes()
                headers = {
                    "Authorization": "Bearer " + spec.payload["artifact_token"],
                    "X-Worker-ID": worker,
                    "X-Task-Attempt": str(task.generation),
                }
                response = await self.client.put(
                    f"/v1/execution-outputs/{spec.id}",
                    params={"name": name},
                    content=content,
                    headers=headers,
                )
                self.assertEqual(response.status_code, 200, response.text)
                self.uploads.append((spec, headers, name, content, response.json()))
                return response.json()

            outcome = await execute(
                PythonProjectExecutor("ws://localhost", worker, fetch=fetch, publish=publish),
                task,
                lambda _: None,
            )
            await self.store.finish(
                worker, worker, task_ref(task), outcome.result, failure=outcome.error or None
            )
        return await self.service.store.job(self.id)

    async def complete(self):
        for _ in range(25):
            job = await self.step()
            if job["phase"] in {"completed", "failed", "cancelled"}:
                return job
        self.fail("Program did not complete")

    async def test_auto_render_selects_workers_and_downloads_binary_larger_than_json_limit(self):
        await self.start(automatic=True)
        job = await self.complete()
        self.assertEqual(job["phase"], "completed", job["data"]["message"])
        self.assertEqual([check["passed"] for check in job["data"]["checks"]], [True, True])
        placement = next(context for name, context in self.contexts if name == "place_program")
        self.assertGreater(placement["measurements"][0]["compute_seconds"], 0)
        self.assertEqual(placement["measurements"][0]["worker_id"], "worker-b")
        self.assertTrue(all("capabilities" in w for w in placement["workers"]))
        listing = await self.client.get(f"/v1/jobs/{self.id}/outputs", headers=self.admin)
        files = listing.json()["files"]
        self.assertEqual(len(files), 1)
        self.assertGreater(files[0]["size"], 64 * 1024)
        url = f"/v1/jobs/{self.id}/outputs/{files[0]['id']}"
        downloaded = await self.client.get(url, headers=self.admin)
        self.assertEqual(downloaded.content, self.uploads[0][3])
        self.assertTrue(downloaded.content.startswith(b"P6\n256 256\n255\n"))
        self.assertIn("attachment", downloaded.headers["content-disposition"])
        self.assertEqual((await self.client.get(url)).status_code, 401)
        self.assertEqual(
            (
                await self.client.get(url.replace(self.id, "another-job"), headers=self.admin)
            ).status_code,
            404,
        )
        spec, headers, name, content, _ = self.uploads[0]
        late = await self.client.put(
            f"/v1/execution-outputs/{spec.id}",
            params={"name": name},
            content=content,
            headers=headers,
        )
        self.assertEqual(late.status_code, 403)
        repeated = await self.client.post("/v1/jobs", json=self.submission, headers=self.admin)
        self.assertEqual(repeated.json()["spec"]["job_id"], self.id)
        result = await self.client.get(f"/v1/jobs/{self.id}/result", headers=self.admin)
        self.assertEqual(result.json()["workload"], "rendering")

    async def test_training_reloads_checkpoint_and_enforces_metric_gate(self):
        await self.start("training")
        job = await self.complete()
        self.assertEqual(job["phase"], "completed", job["data"]["message"])
        result = (await self.store.task(self.id)).result
        self.assertLessEqual(result["output"]["metrics"]["mse"], 0.001)
        self.assertEqual(len(result["files"]), 2)
        listing = await self.client.get(f"/v1/jobs/{self.id}/outputs", headers=self.admin)
        self.assertEqual(
            {f["name"] for f in listing.json()["files"]}, {"checkpoint.json", "metrics.json"}
        )

    async def test_agent_dependency_plan_reaches_probe_full_run_and_validator(self):
        from unit.test_dependency_setup import WHEEL_NAME, wheel

        with tempfile.TemporaryDirectory() as temporary:
            directory = Path(temporary)
            for name in ("render.py", "validate.py"):
                (directory / name).write_text(
                    "import dispatch_fixture\nassert dispatch_fixture.VALUE == 42\n"
                    + (EXAMPLES / "rendering" / name).read_text()
                )
            await self.start(
                folder=directory,
                override={"dependencies": ["dispatch-fixture==1.0"]},
                extra_files=[
                    {"name": WHEEL_NAME, "content": wheel()},
                    {
                        "name": "requirements.txt",
                        "content": encoded("--no-index\n--find-links .\n"),
                    },
                ],
            )
        job = await self.complete()
        self.assertEqual(job["phase"], "completed", job["data"])
        self.assertEqual(job["data"]["program_plan"]["dependencies"], ["dispatch-fixture==1.0"])
        for check in job["data"]["checks"]:
            self.assertTrue(check["passed"], check)
            self.assertGreater(check["details"]["metrics"]["dependency_seconds"], 0)
        self.assertTrue(self.uploads)

    async def test_failed_full_validation_does_not_publish_or_weaken_contract(self):
        await self.start("training", override={"run_args": ["--steps", "1"]})
        job = await self.complete()
        self.assertEqual(job["phase"], "failed")
        self.assertIn("MSE", job["data"]["message"])
        self.assertEqual(self.uploads, [])
        self.assertEqual(sum(name == "plan_program" for name, _ in self.contexts), 1)

    async def test_missing_validator_asks_and_resumes_without_dispatch(self):
        await self.start()
        await self.step()
        self.model.respond.side_effect = None
        self.model.respond.return_value = legacy.proposal(
            "ask_user", {"question": "Which output dimensions should validation require?"}
        )
        job = await self.step()
        self.assertEqual(job["phase"], "needs_input")
        self.assertEqual(job["data"]["tasks"], [])
        response = await self.client.post(
            f"/v1/jobs/{self.id}/answer", json={"message": "256 by 256"}, headers=self.admin
        )
        self.assertEqual(response.status_code, 200)
        self.model.respond.return_value = legacy.proposal("plan_program", self.proposal)
        job = await self.step()
        self.assertEqual(job["phase"], "program_preparing")

    async def test_old_python_workers_are_not_given_new_program_modes(self):
        await self.start()
        for worker in ("worker-a", "worker-b"):
            await self.store.register(
                worker, worker, Capabilities(runtime="cpu", vram_mib=0, kinds=["python_project"])
            )
        await self.step()
        job = await self.step()
        self.assertEqual(job["phase"], "program_planning")
        self.assertEqual(job["data"]["tasks"], [])
        self.assertIn("Waiting for", job["data"]["message"])

    async def running_output_task(self):
        await self.start()
        for _ in range(10):
            job = await self.step()
            if job["phase"] == "program_ready":
                break
        self.assertEqual(job["phase"], "program_ready")
        await self.store.pool.execute(
            "UPDATE simulation_jobs SET retry_after=clock_timestamp() WHERE job_id=$1", self.id
        )
        await self.service.run_once(self.id)
        task = await self.store.claim("worker-a", "worker-a")
        self.assertIsNotNone(task)
        await self.store.ack("worker-a", "worker-a", task_ref(task))
        headers = {
            "Authorization": "Bearer " + task.spec.payload["artifact_token"],
            "X-Worker-ID": "worker-a",
            "X-Task-Attempt": str(task.generation),
        }
        return task, headers

    async def test_output_upload_is_idempotent_attempt_scoped_and_hidden_before_acceptance(self):
        task, headers = await self.running_output_task()
        url = f"/v1/execution-outputs/{task.spec.id}"

        async def put(content=b"image", **kwargs):
            return await self.client.put(
                url,
                params={"name": kwargs.get("name", "frame.ppm")},
                content=content,
                headers={**headers, **kwargs.get("headers", {})},
            )

        first = await put()
        self.assertEqual(first.status_code, 200, first.text)
        self.assertEqual((await put()).json()["id"], first.json()["id"])
        self.assertEqual((await put(b"changed")).status_code, 409)
        self.assertEqual((await put(headers={"X-Task-Attempt": "0"})).status_code, 403)
        self.assertEqual((await put(headers={"X-Worker-ID": "worker-b"})).status_code, 403)
        self.assertEqual((await put(name="../escape")).status_code, 400)
        files = await self.client.get(f"/v1/jobs/{self.id}/outputs", headers=self.admin)
        self.assertEqual(files.json()["files"], [])
        download = await self.client.get(
            f"/v1/jobs/{self.id}/outputs/{first.json()['id']}", headers=self.admin
        )
        self.assertEqual(download.status_code, 404)
        await self.client.post(f"/v1/tasks/{self.id}/cancel", headers=self.admin)
        self.assertEqual((await put()).status_code, 403)

    async def test_worker_loss_replans_placement_without_changing_the_program(self):
        await self.start()
        await self.step()
        job = await self.step()
        self.assertEqual(job["phase"], "program_preparing")
        await self.store.heartbeat("worker-b", "worker-b", [], True)
        job = await self.step(workers=("worker-a",))
        self.assertEqual(job["data"]["program_plan"]["worker_id"], "worker-a")
        self.assertEqual(job["data"]["program_plan"]["run_args"], ["--size", "256"])
        job = await self.complete()
        self.assertEqual(job["phase"], "completed", job["data"]["message"])
        self.assertEqual(sum(name == "plan_program" for name, _ in self.contexts), 1)

    async def test_empty_model_working_directory_means_project_root(self):
        await self.start(override={"working_directory": ""})
        job = await self.complete()
        self.assertEqual(job["phase"], "completed", job["data"]["message"])
        self.assertEqual(job["data"]["program_plan"]["working_directory"], ".")

    async def test_large_output_streams_through_worker_storage_and_download(self):
        task, _ = await self.running_output_task()
        block = bytes(range(256)) * 4096
        digest = hashlib.sha256()
        with tempfile.TemporaryDirectory() as directory:
            path = Path(directory) / "checkpoint.bin"
            with path.open("wb") as stream:
                for _ in range(64):
                    stream.write(block)
                    digest.update(block)
            client = httpx.AsyncClient(
                transport=httpx.ASGITransport(app=self.app), base_url="http://localhost"
            )
            with patch(
                "orchestrator.worker.executors.python_project.httpx.AsyncClient",
                return_value=client,
            ):
                saved = await PythonProjectExecutor("ws://localhost", "worker-a").upload_output(
                    task.spec, SimpleNamespace(task=task), path.name, path
                )
        self.assertEqual(saved["size"], 64 * 1024 * 1024)
        self.assertEqual(saved["sha256"], digest.hexdigest())
        row = await self.store.pool.fetchrow(
            "SELECT count(*) AS parts,max(octet_length(content)) AS largest FROM job_output_chunks WHERE output_id=$1::uuid",
            saved["id"],
        )
        self.assertEqual((row["parts"], row["largest"]), (64, 1024 * 1024))
        self.assertIsNone(
            await self.store.pool.fetchval(
                "SELECT content FROM job_outputs WHERE id=$1::uuid", saved["id"]
            )
        )
        await self.store.finish("worker-a", "worker-a", task_ref(task), {"ok": True})
        url = f"/v1/jobs/{self.id}/outputs/{saved['id']}"
        head = await self.client.head(url, headers=self.admin)
        self.assertEqual(head.status_code, 200)
        self.assertEqual(head.content, b"")
        self.assertEqual(int(head.headers["content-length"]), saved["size"])
        downloaded = await self.client.get(url, headers=self.admin)
        self.assertEqual(hashlib.sha256(downloaded.content).hexdigest(), digest.hexdigest())

    async def test_truncated_transfer_removes_staged_metadata_and_chunks(self):
        task, headers = await self.running_output_task()
        response = await self.client.put(
            f"/v1/execution-outputs/{task.spec.id}",
            params={"name": "partial.bin"},
            headers={**headers, "Content-Length": str(1024 * 1024 + 1)},
            content=b"x" * (1024 * 1024),
        )
        self.assertEqual(response.status_code, 400)
        self.assertEqual(await self.store.pool.fetchval("SELECT count(*) FROM job_outputs"), 0)
        self.assertEqual(
            await self.store.pool.fetchval("SELECT count(*) FROM job_output_chunks"), 0
        )

    async def test_legacy_inline_outputs_still_download(self):
        task, _ = await self.running_output_task()
        identifier = uuid4()
        await self.store.pool.execute(
            "INSERT INTO job_outputs(id,job_id,task_id,attempt,name,digest,size,content) VALUES($1,$2,$3,$4,'legacy.bin',$5,7,$6)",
            identifier,
            self.id,
            task.spec.id,
            task.generation,
            hashlib.sha256(b"weights").hexdigest(),
            b"weights",
        )
        await self.store.finish("worker-a", "worker-a", task_ref(task), {"ok": True})
        response = await self.client.get(
            f"/v1/jobs/{self.id}/outputs/{identifier}", headers=self.admin
        )
        self.assertEqual(response.content, b"weights")

    async def test_non_python_upload_is_rejected_before_planning(self):
        for name in ("input.custom", "scene.blend"):
            response = await self.client.post(
                "/v1/jobs",
                json={
                    "request_id": str(uuid4()),
                    "execution_mode": "auto",
                    "description": "Run the uploaded project",
                    "files": [{"name": name, "content": encoded("data")}],
                },
                headers=self.admin,
            )
            self.assertEqual(response.status_code, 400, response.text)
            self.assertIn("Python project", response.json()["detail"])
        self.model.respond.assert_not_called()

    async def test_planner_can_reject_insufficient_hardware_without_dispatch(self):
        await self.start("training")
        await self.step()
        self.model.respond.side_effect = None
        self.model.respond.return_value = legacy.proposal(
            "reject_job",
            {
                "reason": "The requested training requires a CUDA GPU with 80 GiB VRAM.",
                "evidence": "Both available workers report CPU runtime and 0 MiB VRAM.",
            },
        )
        job = await self.step()
        self.assertEqual(job["phase"], "failed")
        self.assertEqual(job["data"]["tasks"], [])
        self.assertIn("80 GiB", job["data"]["message"])
        self.assertTrue(job["data"]["rejection"]["evidence"])
        self.assertEqual((await self.store.task(self.id)).state, "failed")

    async def test_post_probe_agent_can_reject_full_run_and_release_reservations(self):
        await self.start("training")
        for _ in range(10):
            job = await self.step()
            if job["phase"] == "program_placement":
                break
        self.assertEqual(job["phase"], "program_placement")
        self.model.respond.side_effect = None
        self.model.respond.return_value = legacy.proposal(
            "reject_job",
            {
                "reason": "The full run exceeds available memory despite the small probe passing.",
                "evidence": "Test fixture: full-run estimate is 80 GiB; available memory is 16 GiB.",
            },
        )
        job = await self.step()
        self.assertEqual(job["phase"], "failed")
        self.assertEqual(len(job["data"]["checks"]), 1)
        self.assertEqual(self.uploads, [])
        self.assertEqual(
            await self.store.pool.fetchval(
                "SELECT count(*) FROM job_reservations WHERE job_id=$1", self.id
            ),
            0,
        )

    async def test_running_supervisor_can_stop_an_oversized_job(self):
        task, _ = await self.running_output_task()
        from orchestrator.supervisor.store import SupervisorStore

        tools = SupervisorTools(SupervisorStore(self.store), self.id)
        result = await tools.call(
            "take_action",
            {
                "action_id": str(uuid4()),
                "operation": "cancel_job",
                "reason": "Full training exceeds available VRAM; stop without reducing the model.",
            },
        )
        self.assertTrue(result["ok"], result)
        self.assertEqual((await self.store.task(task.spec.id)).state, "cancelled")
        self.assertEqual((await self.service.store.job(self.id))["phase"], "cancelled")

    async def test_gpu_plan_is_rejected_before_dispatch(self):
        await self.start(
            "training", override={"requirements": {"runtime": "cuda", "vram_mib": 81920}}
        )
        job = await self.complete()
        self.assertEqual(job["phase"], "failed")
        self.assertEqual(job["data"]["tasks"], [])
        self.assertIn("cpu", job["data"]["last_planning_error"])
