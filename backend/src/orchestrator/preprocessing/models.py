from typing import Annotated
from uuid import UUID

from pydantic import Field, JsonValue, model_validator

from ..shared.protocol import Model

MAX_UPLOAD = 8 * 1024 * 1024
MAX_SOURCE = 128 * 1024
CODE_KIND = "python_project"
ROOT_KIND = "simulation_job"
TERMINAL = {"completed", "failed", "cancelled"}


class UploadFile(Model):
    name: str = Field(min_length=1, max_length=240)
    content: str = Field(max_length=12 * 1024 * 1024)


class Upload(Model):
    request_id: UUID
    description: str = Field(min_length=1, max_length=8000)
    files: list[UploadFile] = Field(min_length=1, max_length=100)
    max_adaptations: int = Field(default=3, ge=1, le=5)
    max_runtime_seconds: int = Field(default=1800, ge=60, le=7200)
    max_workers: int = Field(default=4, ge=2, le=4)


class Plan(Model):
    summary: str = Field(min_length=1, max_length=3000)
    entrypoint: str = Field(min_length=1, max_length=240)
    working_directory: str = Field(default=".", max_length=240)
    root_seed: int | None = Field(default=None, ge=0, le=2**31 - 10000)
    smoke_args: list[Annotated[str, Field(max_length=256)]] = Field(
        default_factory=list, max_length=32
    )
    reference: str = Field(min_length=1, max_length=32000)
    aggregate: str = Field(min_length=1, max_length=16000)
    parameters: dict[str, JsonValue] = Field(default_factory=dict)
    trials: int = Field(ge=1, le=10000)
    batch_size: int = Field(default=32, ge=1, le=64)
    workers: int = Field(default=2, ge=2, le=4)
    preferred_worker: str | None = Field(default=None, max_length=128)

    @model_validator(mode="after")
    def batch_limit(self):
        if (self.trials + self.batch_size - 1) // self.batch_size > 512:
            raise ValueError("Plan must fit in 512 batches; increase batch_size")
        return self


class Candidate(Model):
    explanation: str = Field(min_length=1, max_length=3000)
    code: str = Field(min_length=1, max_length=48000)


class Question(Model):
    question: str = Field(min_length=1, max_length=2000)


class Answer(Model):
    message: str = Field(min_length=1, max_length=8000)


# Version 2 proposals have no execution-policy defaults. The model must choose
# each setting explicitly, inside transport, capacity and submission budgets.
class ProjectPlan(Model):
    preprocessing_worker: str = Field(min_length=1, max_length=128)
    summary: str = Field(min_length=1, max_length=3000)
    entrypoint: str = Field(min_length=1, max_length=240)
    working_directory: str = Field(max_length=240)
    root_seed: int | None = Field(ge=0, le=2**31 - 1)
    smoke_args: list[Annotated[str, Field(max_length=256)]] = Field(max_length=32)
    reference: str = Field(min_length=1, max_length=32000)
    aggregate: str = Field(min_length=1, max_length=16000)
    parameters: dict[str, JsonValue]
    trials: int = Field(ge=1, le=2**31)
    profile_cases: int = Field(ge=1, le=1024)
    probe_timeout_seconds: int = Field(ge=1, le=7200)

    @model_validator(mode="after")
    def seed_space(self):
        if (self.root_seed or 0) + self.trials > 2**31:
            raise ValueError("Requested trials exceed the independent seed space")
        return self


class ExecutionPolicy(Model):
    rationale: str = Field(min_length=1, max_length=3000)
    local_cases: int = Field(ge=1, le=1024)
    independent_cases: int = Field(ge=2, le=1024)
    validation_timeout_seconds: int = Field(ge=1, le=7200)
    aggregation: Annotated[str, Field(pattern=r"^(inline|values|partial)$")]


class Batch(Model):
    worker_id: str = Field(min_length=1, max_length=128)
    trials: int = Field(ge=1, le=2**31)
    timeout_seconds: int = Field(ge=1, le=7200)


class Schedule(Model):
    rationale: str = Field(min_length=1, max_length=3000)
    batches: list[Batch] = Field(min_length=1, max_length=512)
    aggregation_worker: str = Field(min_length=1, max_length=128)
    aggregation_timeout_seconds: int = Field(ge=1, le=7200)


class Placement(Model):
    rationale: str = Field(min_length=1, max_length=3000)
    worker_ids: list[str] = Field(min_length=1, max_length=4)

    @model_validator(mode="after")
    def distinct(self):
        if len(set(self.worker_ids)) != len(self.worker_ids):
            raise ValueError("Worker IDs must be distinct")
        return self
