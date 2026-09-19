"""Live model + real local server/worker + isolated temporary PostgreSQL.

uv run --env-file .env --project backend --python 3.12 --extra demo python examples/supervisor_failure.py
Only model credentials are used from .env. No configured database or fleet is touched.
"""

import asyncio
import json
import os
import secrets
import socket
import tempfile
from contextlib import asynccontextmanager
from pathlib import Path

import httpx
import uvicorn
from orchestrator.llm import OpenAIClient
from orchestrator.server.app import create_app
from orchestrator.server.config import ServerConfig
from orchestrator.server.db.store import Store
from orchestrator.server.updates import ChangeFeed
from orchestrator.shared.protocol import Capabilities
from orchestrator.supervisor.service import SupervisorService
from orchestrator.worker.agent import Agent
from orchestrator.worker.config import WorkerConfig
from orchestrator.worker.executors.stub import StubExecutor


async def main():
    import pgserver

    # The smoke test intentionally fails; keep its telemetry and journals local.
    os.environ["SENTRY_DSN"] = ""
    with tempfile.TemporaryDirectory(prefix="supervisor-live-") as directory:
        os.environ["WORKER_EXECUTION_DIR"] = str(Path(directory) / "executions")
        database = pgserver.get_server(
            Path(directory) / "postgres", cleanup_mode="stop"
        )
        store = await Store.open(database.get_uri(), schema="supervisor_live")
        feed = ChangeFeed(database.get_uri())
        model = OpenAIClient.from_env()
        service = SupervisorService(store, model)
        admin, token = secrets.token_urlsafe(32), secrets.token_urlsafe(32)
        app = create_app()

        @asynccontextmanager
        async def lifespan(_app):
            yield

        app.router.lifespan_context = lifespan
        app.state.config = ServerConfig(
            database.get_uri(), None, admin, {"test-worker": token}
        )
        app.state.store, app.state.cache, app.state.updates = store, None, feed
        app.state.supervisor = service
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
            async with httpx.AsyncClient(
                base_url=f"http://127.0.0.1:{port}",
                headers={"Authorization": f"Bearer {admin}"},
            ) as client:
                response = await client.post(
                    "/v1/tasks",
                    json={
                        "instructions": "This is an intentional failure-handling test. Allow the first execution attempt to run even if you can predict its failure; investigate the observed failure afterward. Do not change the payload.",
                        "tasks": [
                            {
                                "id": "intentional-failure",
                                "job_id": "failure-investigation",
                                "kind": "stub",
                                "payload": {"duration_seconds": 1, "fail": True},
                                "requirements": {"runtime": "cpu", "vram_mib": 0},
                                "max_attempts": 3,
                                "timeout_seconds": 60,
                            }
                        ],
                    },
                )
                response.raise_for_status()
                print(
                    "Submitted failure-investigation; supervisor reviewing submission.",
                    flush=True,
                )
                await service.run_once("failure-investigation")
                config = WorkerConfig(
                    f"ws://127.0.0.1:{port}/v1/worker",
                    "test-worker",
                    token,
                    Capabilities(runtime="cpu", vram_mib=0, kinds=["stub"]),
                )
                jobs = [
                    asyncio.create_task(Agent(config, StubExecutor()).run()),
                    asyncio.create_task(service.run(feed)),
                ]
                async with asyncio.timeout(240):
                    while True:
                        response = await client.get(
                            "/v1/jobs/failure-investigation/supervisor"
                        )
                        response.raise_for_status()
                        data = response.json()
                        if data["job"]["finalized"]:
                            break
                        await asyncio.sleep(1)
                result = {
                    "model": model.model,
                    "job_state": data["job"]["state"],
                    "task": (await store.task("intentional-failure")).model_dump(
                        mode="json"
                    ),
                    "memory": data["job"]["memory"],
                    "actions": data["actions"],
                    "runs": [
                        {
                            "status": r["status"],
                            "reply": r["reply"],
                            "tools": [t["name"] for t in r["tools"]],
                        }
                        for r in data["runs"]
                    ],
                }
                output = Path(".local/supervisor-live-result.json")
                output.parent.mkdir(exist_ok=True)
                output.write_text(json.dumps(result, indent=2) + "\n")
                print(json.dumps(result, indent=2), flush=True)
                print(f"Saved {output}", flush=True)
                assert result["task"]["generation"] == 1, (
                    "deterministic failure should not be retried"
                )
                assert any("search_logs" in r["tools"] for r in result["runs"]), (
                    "failure was not investigated"
                )
                assert result["memory"]["findings"], "findings were not persisted"
        finally:
            for job in jobs:
                job.cancel()
            await asyncio.gather(*jobs, return_exceptions=True)
            server.should_exit = True
            await server_task
            await feed.close()
            await model.aclose()
            await store.close()
            database.cleanup()


if __name__ == "__main__":
    asyncio.run(main())
