"""Shared models and the two boundaries for future planners and executors."""

import json
from collections.abc import Callable
from datetime import datetime
from typing import Annotated, Literal, Protocol

from pydantic import BaseModel, ConfigDict, Field, JsonValue, field_validator

VERSION = 1
HEARTBEAT_INTERVAL = 5
UNHEALTHY_AFTER = 15
LEASE_SECONDS = 45
ACK_SECONDS = 10
MESSAGE_LIMIT = 128 * 1024
JSON_LIMIT = 64 * 1024

Identifier = Annotated[str, Field(pattern=r"^[a-zA-Z0-9][a-zA-Z0-9_.-]{0,127}$")]
Runtime = Literal["cpu", "cuda", "mps"]


def json_text(value: object) -> str:
    """Canonical JSON for storage and comparison; never emit NaN or Infinity."""
    return json.dumps(
        value, sort_keys=True, separators=(",", ":"), ensure_ascii=True, allow_nan=False
    )


def reject_constant(value: str) -> None:
    raise ValueError(f"non-finite JSON number: {value}")


def json_loads(value: str | bytes) -> object:
    return json.loads(value, parse_constant=reject_constant)


def bounded_json(value: JsonValue) -> JsonValue:
    if len(json_text(value).encode()) > JSON_LIMIT:
        raise ValueError("JSON payload/result must be at most 64 KiB")
    return value


class Model(BaseModel):
    model_config = ConfigDict(extra="forbid", allow_inf_nan=False)


class Requirements(Model):
    runtime: Runtime
    vram_mib: int = Field(ge=0, le=2_147_483_647, strict=True)


class TaskSpec(Model):
    """One already-split task. The orchestrator does not interpret the payload."""

    id: Identifier
    job_id: Identifier
    kind: Identifier
    payload: JsonValue
    requirements: Requirements
    max_attempts: int = Field(ge=1, le=10, strict=True)
    timeout_seconds: int = Field(ge=1, le=86400, strict=True)
    target_worker_id: Identifier | None = None
    allow_failover: bool = True

    @field_validator("payload")
    @classmethod
    def check_payload(cls, value: JsonValue) -> JsonValue:
        return bounded_json(value)


class Machine(Model):
    """What a worker *is*, as opposed to what it can run.

    Every field here already crossed the wire before this model existed — the agent has
    always reported them and the server discarded them, so a scheduler could tell an
    8-core laptop from a 15-core desktop only by watching how fast work came back. None
    of this is load-bearing for correctness: it exists so allocation can be better than
    round-robin, and every field is optional because a client that predates it, or a
    platform that cannot answer, must still be able to register.
    """

    os: str | None = Field(default=None, max_length=32)
    arch: str | None = Field(default=None, max_length=32)
    cpu_model: str | None = Field(default=None, max_length=128)
    logical_cores: int | None = Field(default=None, ge=1, le=4096)
    total_ram_mb: int | None = Field(default=None, ge=0)
    agent_version: str | None = Field(default=None, max_length=64)
    #: What the owner consented to run at once. Not a hardware fact, and deliberately
    #: lower than the core count on machines someone is sitting in front of.
    max_concurrency: int | None = Field(default=None, ge=1, le=1024)


class Capabilities(Requirements):
    kinds: list[Identifier] = Field(min_length=1, max_length=32)
    #: Optional because rows written before this field existed must still load, and
    #: because `extra="forbid"` would otherwise reject them outright.
    machine: Machine | None = None


class Task(Model):
    spec: TaskSpec
    state: Literal["queued", "assigned", "running", "succeeded", "failed", "cancelled"]
    generation: int
    worker_id: str | None = None
    session_id: str | None = None
    lease_until: datetime | None = None
    deadline: datetime | None = None
    result: JsonValue = None
    attestation: dict[str, JsonValue] | None = None
    failure: str = ""
    created_at: datetime
    progress: float = 0
    started_at: datetime | None = None


class Worker(Model):
    id: Identifier
    session_id: str
    capabilities: Capabilities
    state: Literal["alive", "unhealthy", "offline"]
    last_seen: datetime
    paused: bool


class Ref(Model):
    task_id: Identifier
    generation: int = Field(ge=1, strict=True)


def task_ref(task: Task) -> Ref:
    return Ref(task_id=task.spec.id, generation=task.generation)


class Message(Model):
    type: Literal[
        "hello",
        "welcome",
        "heartbeat",
        "heartbeat_ack",
        "assign",
        "ack",
        "ack_accepted",
        "complete",
        "failed",
        "result_accepted",
        "revoke",
    ]
    sequence: int = Field(default=0, ge=0, strict=True)
    version: int = Field(default=VERSION, strict=True)
    capabilities: Capabilities | None = None
    session_id: str | None = None
    task: Task | None = None
    ref: Ref | None = None
    active: list[Ref] = Field(default_factory=list, max_length=1)
    paused: bool = Field(default=False, strict=True)
    result: JsonValue = None
    error: str = Field(default="", max_length=2048)
    retryable: bool = Field(default=False, strict=True)
    progress: float = Field(default=0, ge=0, le=100)

    @field_validator("result")
    @classmethod
    def check_result(cls, value: JsonValue) -> JsonValue:
        return bounded_json(value)


class Submission(Model):
    tasks: list[TaskSpec] = Field(min_length=1, max_length=100)


class TaskQueue(Protocol):
    async def submit(self, specs: list[TaskSpec]) -> list[Task]:
        """Atomically submit tasks; identical IDs/specs are idempotent."""
        ...


class Executor(Protocol):
    kind: str

    async def execute(self, spec: TaskSpec, report: Callable[[float], None]) -> JsonValue:
        """Honor asyncio cancellation; do not publish canonical outputs here."""
        ...
