"""Exercise real model planning, HTTP uploads/downloads and local Python workers.

uv run --env-file .env --project backend --python 3.12 --extra demo python examples/uploaded_project_smoke.py
Only the model credentials are used; the database, server, and workers are temporary.
"""

import asyncio
import hashlib
import json
import os
import secrets
import socket
import tempfile
from contextlib import asynccontextmanager
from datetime import UTC, datetime
from pathlib import Path
from uuid import uuid4

import httpx
import uvicorn
from orchestrator.llm import OpenAIClient
from orchestrator.preprocessing.artifacts import encoded
from orchestrator.preprocessing.service import PreprocessingService
from orchestrator.server.app import create_app
from orchestrator.server.config import ServerConfig
from orchestrator.server.db.store import Store
from orchestrator.server.scheduler import reconcile_loop
from orchestrator.server.updates import ChangeFeed
from orchestrator.shared.protocol import Capabilities, Machine
from orchestrator.worker.agent import Agent
from orchestrator.worker.config import WorkerConfig
from orchestrator.worker.executors.python_project import PythonProjectExecutor

ROOT = Path(__file__).resolve().parents[1]
CASES = [
    (
        "hardware-rejected-training",
        "training",
        "training",
        "Run train.py for 200 steps and validate mse <= 0.001, but this job has a mandatory hardware requirement: a CUDA GPU with at least 80 GiB VRAM. CPU fallback is forbidden. If no available worker meets that requirement, reject the job with the reported hardware evidence before executing anything. Return checkpoint.json and metrics.json only on success.",
        "failed",
    ),
    (
        "render",
        "rendering",
        "auto",
        "Render mandelbrot.ppm at 256 by 256 pixels using render.py. Use --size 16 for a cheap probe. Run validate.py --size 256 on the full output and return the image. Use the uploaded scripts unchanged.",
        "completed",
    ),
    (
        "train",
        "training",
        "auto",
        "Train with train.py for 200 steps, using --steps 2 for a probe. Reload the checkpoint through validate.py and require held-out mse <= 0.001 as documented in that script. Return checkpoint.json and metrics.json. Use the uploaded scripts unchanged.",
        "completed",
    ),
    (
        "rejected-training",
        "training",
        "training",
        "This is an intentional validation-failure test. Run train.py for exactly ONE step (--steps 1), with a one-step probe. Then run validate.py unchanged and require mse <= 0.001. It is expected to fail the quality gate. Do execute it; do not change the number of steps, source, validator, or quality bound to make it pass. Return checkpoint.json and metrics.json only if validation succeeds.",
        "failed",
    ),
]


async def main():
    import pgserver

    os.environ["SENTRY_DSN"] = ""
    destination = (
        ROOT / ".local/project-smoke" / datetime.now(UTC).strftime("%Y%m%dT%H%M%SZ")
    )
    destination.mkdir(parents=True)
    model = OpenAIClient.from_env()
    report = {"model": model.model, "cases": {}, "all_passed": False}
    print(f"Live model: {model.model}; results: {destination}", flush=True)
    with tempfile.TemporaryDirectory(prefix="uploaded-project-smoke-") as directory:
        os.environ["WORKER_EXECUTION_DIR"] = str(Path(directory) / "executions")
        database = pgserver.get_server(
            Path(directory) / "postgres", cleanup_mode="stop"
        )
        store = await Store.open(database.get_uri(), schema="project_smoke")
        feed = ChangeFeed(database.get_uri())
        service = PreprocessingService(store, model)
        admin = secrets.token_urlsafe(32)
        tokens = {name: secrets.token_urlsafe(32) for name in ("local-a", "local-b")}
        app = create_app()

        @asynccontextmanager
        async def lifespan(_app):
            yield

        app.router.lifespan_context = lifespan
        app.state.config = ServerConfig(database.get_uri(), None, admin, tokens)
        app.state.store, app.state.cache, app.state.updates = store, None, feed
        app.state.preprocessing = service
        app.state.ui_session = "unused"
        sock = socket.socket()
        sock.bind(("127.0.0.1", 0))
        port = sock.getsockname()[1]
        server = uvicorn.Server(uvicorn.Config(app, log_level="error", ws="websockets"))
        server_task = asyncio.create_task(server.serve(sockets=[sock]))
        jobs = []
        try:
            async with asyncio.timeout(10):
                while not server.started:
                    await asyncio.sleep(0.05)
            feed.start()
            jobs.append(asyncio.create_task(service.run(feed)))
            jobs.append(asyncio.create_task(reconcile_loop(store)))
            for name, token in tokens.items():
                url = f"ws://127.0.0.1:{port}/v1/worker"
                executor = PythonProjectExecutor(url, name)
                config = WorkerConfig(
                    url,
                    name,
                    token,
                    Capabilities(
                        runtime="cpu",
                        vram_mib=0,
                        kinds=list(executor.kinds),
                        machine=Machine(
                            os=os.uname().sysname,
                            arch=os.uname().machine,
                            logical_cores=os.cpu_count(),
                            max_concurrency=1,
                        ),
                    ),
                )
                jobs.append(asyncio.create_task(Agent(config, executor).run()))
            async with httpx.AsyncClient(
                base_url=f"http://127.0.0.1:{port}",
                headers={"Authorization": f"Bearer {admin}"},
                timeout=30,
            ) as client:
                async with asyncio.timeout(10):
                    while len(await store.workers()) != 2:
                        await asyncio.sleep(0.1)
                pending = {}
                for name, folder, workload, description, expected in CASES:
                    files = [
                        {"name": path.name, "content": encoded(path.read_text())}
                        for path in (ROOT / "examples/projects" / folder).glob("*.py")
                    ]
                    response = await client.post(
                        "/v1/jobs",
                        json={
                            "request_id": str(uuid4()),
                            "description": description,
                            "workload": workload,
                            "files": files,
                            "max_runtime_seconds": 600,
                        },
                    )
                    response.raise_for_status()
                    identifier = response.json()["spec"]["job_id"]
                    pending[name] = (identifier, expected)
                    report["cases"][name] = {"job_id": identifier, "expected": expected}
                last = {}
                async with asyncio.timeout(650):
                    while pending:
                        for name, (identifier, expected) in list(pending.items()):
                            response = await client.get(f"/v1/jobs/{identifier}")
                            response.raise_for_status()
                            status = response.json()
                            phase = status["phase"]
                            if last.get(name) != phase:
                                print(
                                    f"{name}: {phase} — {status['message']}", flush=True
                                )
                                last[name] = phase
                            if phase not in {
                                "completed",
                                "failed",
                                "cancelled",
                                "needs_input",
                            }:
                                continue
                            case_dir = destination / name
                            case_dir.mkdir()
                            response = await client.get(
                                f"/v1/jobs/{identifier}/outputs"
                            )
                            response.raise_for_status()
                            outputs = response.json()["files"]
                            for output in outputs:
                                response = await client.get(
                                    f"/v1/jobs/{identifier}/outputs/{output['id']}"
                                )
                                response.raise_for_status()
                                assert (
                                    hashlib.sha256(response.content).hexdigest()
                                    == output["sha256"]
                                )
                                assert len(response.content) == output["size"]
                                target = case_dir / output["name"]
                                target.parent.mkdir(parents=True, exist_ok=True)
                                target.write_bytes(response.content)
                            if phase == "completed":
                                response = await client.get(
                                    f"/v1/jobs/{identifier}/result"
                                )
                                response.raise_for_status()
                                (case_dir / "result.json").write_bytes(response.content)
                            (case_dir / "status.json").write_text(
                                json.dumps(status, indent=2) + "\n"
                            )
                            passed = phase == expected and (
                                bool(outputs)
                                if expected == "completed"
                                else not outputs
                            )
                            if name == "hardware-rejected-training" and passed:
                                passed = any(
                                    d["tool"] == "reject_job"
                                    for d in status.get("decisions", [])
                                ) and not status.get("checks")
                            if name == "render" and passed:
                                raw = (case_dir / "mandelbrot.ppm").read_bytes()
                                passed = (
                                    raw.startswith(b"P6\n256 256\n255\n")
                                    and len(raw.split(b"\n", 3)[3]) == 256 * 256 * 3
                                )
                            if name == "train" and passed:
                                metrics = json.loads(
                                    (case_dir / "metrics.json").read_text()
                                )
                                passed = metrics["mse"] <= 0.001
                            report["cases"][name].update(
                                phase=phase,
                                passed=passed,
                                outputs=outputs,
                                message=status["message"],
                                question=status.get("question"),
                                decisions=status.get("decisions", []),
                                measurements=status.get("measurements", []),
                            )
                            print(
                                f"{name}: {'PASS' if passed else 'FAIL'}; downloaded {len(outputs)} files",
                                flush=True,
                            )
                            del pending[name]
                        await asyncio.sleep(1)
                report["all_passed"] = all(
                    case["passed"] for case in report["cases"].values()
                )
        finally:
            (destination / "report.json").write_text(
                json.dumps(report, indent=2) + "\n"
            )
            print(f"Saved {destination / 'report.json'}", flush=True)
            for job in jobs:
                job.cancel()
            await asyncio.gather(*jobs, return_exceptions=True)
            server.should_exit = True
            await server_task
            await feed.close()
            await model.aclose()
            await store.close()
            database.cleanup()
    if not report["all_passed"]:
        raise SystemExit("Some live project checks failed; see the saved report.")


if __name__ == "__main__":
    asyncio.run(main())
