"""Bind the existing dispatcher to the current fleet user's authorization."""

import asyncpg
from fastapi import HTTPException, Request

from ..agent.loop import failure
from ..client import AgentTools, ClientError, tool_definitions
from .auth import require_admin
from .db.store import Conflict, NotFound
from .dwp_assets import workloads
from .routes import submit_specs

KINDS = {"stub", "echo", "walker_evolution", "cpu_inference_batch"}


class ServerTaskClient:
    def __init__(self, request: Request):
        self.request = request
        self.store = request.app.state.store

    async def list_workers(self):
        return await self.store.workers()

    async def list_tasks(self):
        return await self.store.tasks()

    async def get_task(self, task_id):
        return await self.store.task(task_id)

    async def submit_tasks(self, tasks):
        if len(tasks) > 10 or any(task.kind not in KINDS for task in tasks):
            raise ClientError(
                "invalid_arguments", "Submit at most 10 tasks using existing workloads."
            )
        return await submit_specs(self.request, tasks)

    async def cancel_task(self, task_id):
        await self.store.cancel(task_id)
        return await self.store.task(task_id)

    async def list_events(self, after=0):
        return await self.store.events(after)

    async def wait_task(self, task_id, timeout_seconds=300):
        raise ClientError("unknown_tool", "Use get_task in a follow-up to check progress.")


class FleetTools:
    def __init__(self, request: Request):
        self.request = request
        self.dispatcher = AgentTools(ServerTaskClient(request))

    def definitions(self):
        return [tool for tool in tool_definitions() if tool["name"] != "wait_task"] + [
            {
                "name": "list_workloads",
                "description": "List supported workloads and example payloads. Use before submitting tasks.",
                "input_schema": {"type": "object", "properties": {}, "additionalProperties": False},
            }
        ]

    async def call(self, name, arguments):
        try:
            # Recheck expiry/revocation before each action, including reads. Never
            # substitute the server's automation token for the user's credentials.
            await require_admin(self.request)
            if name == "list_workloads":
                if arguments:
                    return failure("invalid_arguments", "list_workloads takes no arguments.")
                catalog = await workloads()
                return {
                    "ok": True,
                    "result": {
                        "requirements": {"runtime": "cpu", "vram_mib": 0},
                        "payloads": {
                            "stub": {"duration_seconds": 2, "value": {"label": "Connection test"}},
                            "echo": {"nonce": "Connection test", "sleepMs": 1000},
                            "walker_evolution": {
                                "generation": 0,
                                "parent": [0] * 308,
                                "sigma": 0.1,
                                "seeds": list(range(1, 9)),
                                "steps": 600,
                            },
                            "cpu_inference_batch": catalog["inference"],
                        },
                        "notes": "Null inference means unavailable. Match worker kinds. No rendering, custom code, or GPU execution. Use max_attempts=3 and timeout_seconds=120 (300 for inference).",
                    },
                }
            if name not in {tool["name"] for tool in self.definitions()}:
                return failure("unknown_tool", "This tool is not available.")
            return await self.dispatcher.call(name, arguments)
        except HTTPException as exc:
            code = {401: "unauthorized", 403: "forbidden"}.get(exc.status_code, "invalid_arguments")
            return failure(
                code, "Access denied." if exc.status_code in {401, 403} else str(exc.detail)
            )
        except (Conflict, NotFound) as exc:
            return failure("conflict" if isinstance(exc, Conflict) else "not_found", str(exc))
        except (asyncpg.PostgresError, asyncpg.InterfaceError, OSError, TimeoutError):
            return failure(
                "connection_error", "Control plane unavailable; an action may have committed."
            )
