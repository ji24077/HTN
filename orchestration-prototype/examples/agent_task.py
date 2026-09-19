"""Run from the prototype directory: uv run python examples/agent_task.py --demo.

Requires the server and workers to be running. A stable --task-id makes reruns
idempotent. Use a different ID when you actually want a new task.
"""

import argparse
import asyncio
import json

from orchestrator.client import AgentTools, OrchestratorClient
from orchestrator.client.cli import connection


async def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--demo", action="store_true")
    parser.add_argument("--url")
    parser.add_argument("--task-id", default="agent-example-001")
    args = parser.parse_args()
    url, token = connection(args)
    async with OrchestratorClient(url, token) as client:
        tools = AgentTools(client)
        workers = await tools.call("list_workers", {})
        print(json.dumps(workers))
        if not workers["ok"]:
            return
        submitted = await tools.call(
            "submit_tasks",
            {
                "tasks": [
                    {
                        "id": args.task_id,
                        "job_id": "agent-example",
                        "kind": "stub",
                        "payload": {
                            "duration_seconds": 15,
                            "value": {"label": "Agent-submitted task"},
                        },
                        "requirements": {"runtime": "cpu", "vram_mib": 0},
                        "max_attempts": 3,
                        "timeout_seconds": 60,
                        "target_worker_id": "worker-a",
                        "allow_failover": True,
                    }
                ]
            },
        )
        print(json.dumps(submitted))
        if submitted["ok"]:
            print(
                json.dumps(
                    await tools.call(
                        "wait_task",
                        {
                            "task_id": args.task_id,
                            "timeout_seconds": 120,
                        },
                    )
                )
            )


asyncio.run(main())
