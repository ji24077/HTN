"""Framework-neutral JSON tool definitions and a validated dispatcher."""

from collections.abc import Iterable
from typing import Protocol

from pydantic import Field, ValidationError

from ..shared.protocol import Identifier, Model, Submission, Task, TaskSpec, Worker
from .http import ClientError


class TaskClient(Protocol):
    async def list_workers(self) -> list[Worker]: ...
    async def list_tasks(self) -> list[Task]: ...
    async def get_task(self, task_id: str) -> Task: ...
    async def submit_tasks(self, tasks: list[TaskSpec]) -> list[Task]: ...
    async def cancel_task(self, task_id: str) -> Task: ...
    async def wait_task(self, task_id: str, timeout_seconds: float = 300) -> Task: ...
    async def list_events(self, after: int = 0) -> list[dict]: ...


class EmptyArgs(Model):
    pass


class TaskArgs(Model):
    task_id: Identifier


class WaitArgs(TaskArgs):
    timeout_seconds: float = Field(default=300, gt=0, le=3600)


class EventsArgs(Model):
    after: int = Field(default=0, ge=0, le=2**63 - 1, strict=True)


# No credentials in schemas or tool arguments: the host binds a client once.
TOOLS = {
    "list_workers": (EmptyArgs, "List up to 500 workers, their state and reported capabilities."),
    "list_tasks": (EmptyArgs, "List the newest 500 tasks and their current state."),
    "get_task": (TaskArgs, "Inspect one task, including its accepted result or failure."),
    "submit_tasks": (
        Submission,
        "Submit 1–100 already-split tasks. Reuse stable task IDs on retries; identical submissions are idempotent. Requirements still apply to targeted workers.",
    ),
    "cancel_task": (
        TaskArgs,
        "Cancel a task. This changes server state; the worker stops on its next heartbeat.",
    ),
    "wait_task": (
        WaitArgs,
        "Wait via pushed updates for succeeded, failed, or cancelled. A wait timeout does not cancel the task. Inspect the returned task state.",
    ),
    "list_events": (
        EventsArgs,
        "Read up to 500 audit events after an event ID. Use the last returned ID as the next cursor.",
    ),
}


def select_tools(names: Iterable[str] | None = None) -> dict[str, tuple[type[Model], str]]:
    """Return the registry limited to ``names`` (all tools when omitted), in registry order."""
    if names is None:
        return dict(TOOLS)
    unknown = set(names) - TOOLS.keys()
    if unknown:
        raise ValueError(f"Unknown tools: {sorted(unknown)}")
    return {name: TOOLS[name] for name in TOOLS if name in names}


def tool_definitions(names: Iterable[str] | None = None) -> list[dict]:
    return [
        {"name": name, "description": description, "input_schema": model.model_json_schema()}
        for name, (model, description) in select_tools(names).items()
    ]


class AgentTools:
    def __init__(self, client: TaskClient, names: Iterable[str] | None = None):
        """Bind a client; ``names`` limits which registry tools this dispatcher exposes."""
        self.client = client
        self.tools = select_tools(names)

    def definitions(self) -> list[dict]:
        return tool_definitions(self.tools)

    async def call(self, name: str, arguments: dict) -> dict:
        """Return a JSON-serializable envelope; ok describes the call, not task success."""
        if name not in self.tools:
            return {"ok": False, "error": {"code": "unknown_tool", "message": "Unknown tool name"}}
        try:
            args = self.tools[name][0].model_validate(arguments)
        except ValidationError as exc:
            return {
                "ok": False,
                "error": {
                    "code": "invalid_arguments",
                    "message": "Invalid tool arguments",
                    "details": exc.errors(
                        include_input=False, include_context=False, include_url=False
                    ),
                },
            }
        try:
            match name:
                case "list_workers":
                    result = await self.client.list_workers()
                case "list_tasks":
                    result = await self.client.list_tasks()
                case "get_task":
                    result = await self.client.get_task(args.task_id)
                case "submit_tasks":
                    result = await self.client.submit_tasks(args.tasks)
                case "cancel_task":
                    result = await self.client.cancel_task(args.task_id)
                case "wait_task":
                    result = await self.client.wait_task(args.task_id, args.timeout_seconds)
                case "list_events":
                    result = await self.client.list_events(args.after)
            if isinstance(result, Model):
                result = result.model_dump(mode="json")
            elif isinstance(result, list):
                result = [v.model_dump(mode="json") if isinstance(v, Model) else v for v in result]
            return {"ok": True, "result": result}
        except ValidationError as exc:
            return {
                "ok": False,
                "error": {
                    "code": "invalid_response",
                    "message": "Unexpected control plane response",
                    "details": exc.errors(
                        include_input=False, include_context=False, include_url=False
                    ),
                },
            }
        except ClientError as exc:
            return {"ok": False, "error": exc.as_dict()}
