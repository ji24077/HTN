from datetime import datetime
from typing import Annotated, Literal
from uuid import UUID

from pydantic import Field, model_validator

from ..shared.protocol import Identifier, Model

Text = Annotated[str, Field(min_length=1, max_length=2000)]


class Finding(Model):
    kind: Literal["observation", "hypothesis"]
    text: Text
    evidence: list[Annotated[str, Field(max_length=256)]] = Field(
        default_factory=list, max_length=20
    )


class Followup(Model):
    check: Text
    expected_outcome: Text
    due_at: datetime

    @model_validator(mode="after")
    def timezone(self):
        if self.due_at.tzinfo is None:
            raise ValueError("due_at must have a timezone")
        return self


class Memory(Model):
    findings: list[Finding] = Field(default_factory=list, max_length=30)
    questions: list[Text] = Field(default_factory=list, max_length=20)
    followups: list[Followup] = Field(default_factory=list, max_length=20)


class Empty(Model):
    pass


class TaskLookup(Model):
    task_id: Identifier


class LogContext(Model):
    log_id: int = Field(ge=1, le=2**63 - 1)
    before: int = Field(default=5, ge=0, le=10)
    after: int = Field(default=5, ge=0, le=10)


class LogSearch(Model):
    task_id: Identifier | None = None
    attempt: int | None = Field(default=None, ge=0, le=2_147_483_647)
    worker_id: Identifier | None = None
    reservation_id: Annotated[str, Field(max_length=160)] | None = None
    severity: Literal["trace", "debug", "info", "warning", "error", "fatal"] | None = None
    text: Annotated[str, Field(max_length=256)] = ""
    trace_id: Annotated[str, Field(pattern=r"^[a-fA-F0-9]{32}$")] | None = None
    start: datetime | None = None
    end: datetime | None = None
    after: int = Field(default=0, ge=0, le=2**63 - 1)
    cursor: Annotated[str, Field(max_length=256)] | None = None
    limit: int = Field(default=20, ge=1, le=50)
    source: Literal["execution", "sentry"] = "execution"

    @model_validator(mode="after")
    def time_range(self):
        for value in (self.start, self.end):
            if value and value.tzinfo is None:
                raise ValueError("timestamps must have a timezone")
        if self.start and self.end and self.end <= self.start:
            raise ValueError("end must follow start")
        return self


class Action(Model):
    action_id: UUID
    operation: Literal[
        "retry_task",
        "cancel_task",
        "pause_job",
        "resume_job",
        "cancel_job",
        "reserve_worker",
        "release_worker",
    ]
    reason: Text
    task_id: Identifier | None = None
    expected_generation: int | None = Field(default=None, ge=0, le=2_147_483_647)
    worker_id: Identifier | None = None

    @model_validator(mode="after")
    def task_boundary(self):
        if self.operation.endswith("_worker"):
            if (
                self.worker_id is None
                or self.task_id is not None
                or self.expected_generation is not None
            ):
                raise ValueError("worker actions require only worker_id")
            return self
        if self.worker_id is not None:
            raise ValueError("worker_id is only valid for worker actions")
        if self.operation.endswith("_task"):
            if self.task_id is None or self.expected_generation is None:
                raise ValueError("task actions require task_id and expected_generation")
        elif self.task_id is not None or self.expected_generation is not None:
            raise ValueError("job actions do not accept task fields")
        return self
