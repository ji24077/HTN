"""Bind the existing dispatcher to the current fleet user's authorization."""

import asyncpg
from fastapi import HTTPException, Request

from ..agent.loop import failure
from ..client import AgentTools, ClientError
from .auth import require_admin
from .db.store import Conflict, NotFound
from .dwp_assets import workloads
from .routes import submit_specs

# One catalog drives both what the assistant is told it can submit and what
# submission accepts. Each entry maps a workload kind to an example payload,
# built from the live workload catalog so inference reflects availability.
EXAMPLE_PAYLOADS = {
    "stub": lambda _catalog: {"duration_seconds": 2, "value": {"label": "Connection test"}},
    "echo": lambda _catalog: {"nonce": "Connection test", "sleepMs": 1000},
    "walker_evolution": lambda _catalog: {
        "generation": 0,
        "parent": [0] * 308,
        "sigma": 0.1,
        "seeds": list(range(1, 9)),
        "steps": 600,
    },
    "cpu_inference_batch": lambda catalog: catalog["inference"],
}
KINDS = frozenset(EXAMPLE_PAYLOADS)

# wait_task blocks on pushed updates; chat turns are bounded, so the model is
# told to call get_task in a follow-up instead.
FLEET_TOOLS = (
    "list_workers",
    "list_tasks",
    "get_task",
    "submit_tasks",
    "cancel_task",
    "list_events",
)

LIST_WORKLOADS = {
    "name": "list_workloads",
    "description": "List supported workloads and example payloads. Use before submitting tasks.",
    "input_schema": {"type": "object", "properties": {}, "additionalProperties": False},
}
SUMMARY_NOTE = " Payloads and results are omitted; use get_task for one task's full record."


def task_summary(task) -> dict:
    """A list entry without the payload/result blobs that can each reach 64 KiB."""
    data = task.model_dump(mode="json")
    data["spec"] = {key: value for key, value in data["spec"].items() if key != "payload"}
    data.pop("result", None)
    data.pop("attestation", None)
    return data


class ServerTaskClient:
    def __init__(self, request: Request):
        self.request = request
        self.store = request.app.state.store

    async def list_workers(self):
        return await self.store.workers()

    async def list_tasks(self):
        return [task_summary(task) for task in await self.store.tasks()]

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
        # Event details may embed task snapshots; keep the audit trail compact.
        return [
            {key: value for key, value in event.items() if key != "details"}
            for event in await self.store.events(after)
        ]


class FleetTools:
    def __init__(self, request: Request):
        self.request = request
        self.dispatcher = AgentTools(ServerTaskClient(request), FLEET_TOOLS)
        self._definitions = self.dispatcher.definitions() + [LIST_WORKLOADS]
        for definition in self._definitions:
            if definition["name"] in {"list_tasks", "list_events"}:
                definition["description"] += SUMMARY_NOTE
        self.names = frozenset(definition["name"] for definition in self._definitions)

    def definitions(self):
        return self._definitions

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
                            kind: example(catalog) for kind, example in EXAMPLE_PAYLOADS.items()
                        },
                        "notes": "Null inference means unavailable. Match worker kinds. No rendering, custom code, or GPU execution. Use max_attempts=3 and timeout_seconds=120 (300 for inference).",
                    },
                }
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
