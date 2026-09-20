from typing import ClassVar

from pydantic import Field, ValidationError

from ..agent.loop import failure
from ..server.db.store import Conflict, NotFound
from ..shared.execution import scrub_execution
from ..shared.protocol import Model
from .models import Action, Empty, LogContext, LogSearch, Memory, TaskLookup
from .sentry import SentryUnavailable


class AlertLookup(Model):
    event_id: str = Field(pattern=r"^[a-fA-F0-9]{32}$")


class SupervisorTools:
    SCHEMAS: ClassVar[dict] = {
        "get_job": (Empty, "Read this job's current progress, reservations, events and memory."),
        "list_eligible_workers": (
            Empty,
            "List up to 100 available machines compatible with this job's unfinished tasks.",
        ),
        "get_task": (
            TaskLookup,
            "Read one task's full specification, failure and result within this job.",
        ),
        "search_logs": (
            LogSearch,
            "Search bounded job logs. Execution logs are always available; Sentry requires configuration. Use after/cursor to paginate.",
        ),
        "get_log_context": (
            LogContext,
            "Read surrounding execution logs from the same task attempt by log ID, within this job.",
        ),
        "get_alert_details": (
            AlertLookup,
            "Read a Sentry error event within this job by event ID.",
        ),
        "take_action": (
            Action,
            "Perform an authorized recovery action. Use a unique UUID action_id; reuse it after an uncertain outcome. Retry never increases max_attempts or changes payloads. Reserve compatible idle workers for five minutes, at most four per job; release removes future capacity reservation. Pause prevents new assignments; in-flight tasks finish.",
        ),
        "remember": (
            Memory,
            "Replace this job's findings, questions and follow-ups. Preserve unresolved items. Action history is saved automatically.",
        ),
    }

    def __init__(self, store, job_id, sentry=None, run_id=None):
        self.store, self.job_id, self.sentry = store, job_id, sentry
        self.run_id = run_id

    def definitions(self):
        return [
            {"name": name, "description": desc, "input_schema": schema.model_json_schema()}
            for name, (schema, desc) in self.SCHEMAS.items()
        ]

    async def call(self, name, arguments):
        if name not in self.SCHEMAS:
            return failure("unknown_tool", "tool unavailable")
        try:
            args = self.SCHEMAS[name][0].model_validate(arguments)
            if name == "get_job":
                result = await self.store.snapshot(self.job_id)
            elif name == "list_eligible_workers":
                result = await self.store.eligible_workers(self.job_id)
            elif name == "get_task":
                result = (await self.store.task(self.job_id, args.task_id)).model_dump(mode="json")
            elif name == "get_log_context":
                result = await self.store.log_context(self.job_id, args)
            elif name == "search_logs":
                if args.task_id:
                    await self.store.task(self.job_id, args.task_id)
                if args.source == "execution":
                    result = await self.store.logs(self.job_id, args)
                else:
                    if self.sentry is None:
                        raise SentryUnavailable("Sentry API is not configured; use execution logs")
                    result = await self.sentry.search(self.job_id, args)
            elif name == "get_alert_details":
                if self.sentry is None:
                    raise SentryUnavailable("Sentry API is not configured")
                result = await self.sentry.search(
                    self.job_id, LogSearch(), dataset="errors", event_id=args.event_id
                )
            elif name == "take_action":
                result = await self.store.action(self.job_id, args, self.run_id)
            else:
                await self.store.remember(self.job_id, args, self.run_id)
                result = {"saved": True}
            return {"ok": True, "result": scrub_execution(result)}
        except ValidationError:
            return failure("invalid_arguments", "Arguments do not match the tool schema")
        except (NotFound, Conflict, SentryUnavailable) as exc:
            code = (
                "not_found"
                if isinstance(exc, NotFound)
                else "unavailable"
                if isinstance(exc, SentryUnavailable)
                else "conflict"
            )
            return failure(code, str(exc))
