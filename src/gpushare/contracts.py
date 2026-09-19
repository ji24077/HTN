"""
Contracts. THE shared file. All three of you import from here.

RULES
  1. Frozen at hour 0. Changing anything here requires all three of you, together.
  2. Never duplicate these shapes anywhere else. Import them.
  3. `extra="forbid"` is deliberate: if a sender adds a field the receiver
     doesn't know about, we want a loud error, not a silent drop.
"""

from __future__ import annotations

import sys
import time
from typing import Annotated, Literal

from pydantic import BaseModel, ConfigDict, Field, TypeAdapter, ValidationError, model_validator

# ─────────────────────────────────────────────────────────────────────────────
# Contract 5 — chip_class
#
# Jack writes it (worker.register, SQLite). Ethan tags probes with it.
# Ji dispatches the router on it. A typo used to mean per-chip learning would
# silently never happen; as a Literal it's a validation error instead.
# ─────────────────────────────────────────────────────────────────────────────
ChipClass = Literal[
    "ampere_24gb",   # RTX 3090
    "ada_24gb",      # RTX 4090
    "turing_16gb",   # V100 and friends
    "cdna_amd",      # MI250, 7900
    "default",       # unknown — cold start
]

Vendor = Literal["nvidia", "amd"]
DType = Literal["bf16", "fp16", "fp32"]
Attention = Literal["sdpa", "eager"]
Backend = Literal["gloo"]           # NCCL is not an option. See handbook.
LeaveReason = Literal["timeout", "user", "game"]
MigrationPhase = Literal["waiting_sync", "saving", "transferring", "loading", "resumed"]
ConfigName = Literal["baseline", "optimized"]
WorkerRole = Literal["train", "preprocess"]


def classify_chip(*, gpu_name: str, cc: float, vram_gb: float, is_amd: bool) -> ChipClass:
    """Single source of truth. Jack calls it; Ji must not reimplement it."""
    if is_amd:
        return "cdna_amd"
    if cc >= 8.9 and vram_gb >= 20:
        return "ada_24gb"
    if cc >= 8.0 and vram_gb >= 20:
        return "ampere_24gb"
    if cc >= 7.0:
        return "turing_16gb"
    return "default"


def is_trainable(cc: float) -> bool:
    """cc < 7.0 has no fp16/bf16 tensor cores (e.g. GTX 970). Preprocessing only."""
    return cc >= 7.0


# ─────────────────────────────────────────────────────────────────────────────
# Contract 3 — job config.  Ji's output = Ethan's input. The only interface.
# ─────────────────────────────────────────────────────────────────────────────
class JobConfig(BaseModel):
    model_config = ConfigDict(extra="forbid")

    job_id: str
    model: str
    total_steps: int
    seq_len: int

    dtype: DType
    attention: Attention
    micro_batch: int
    grad_accum: int
    global_batch_tokens: int
    compile: bool = False

    H: int = Field(ge=50, le=500)     # clamped range — outside it we can't claim convergence
    backend: Backend = "gloo"
    workers: list[str]

    @model_validator(mode="after")
    def _fixed_work_invariant(self):
        """
        Optimization means doing the same work faster, not doing less work.
        If this fires, the before/after story is invalid — fix the config,
        don't relax the check.
        """
        actual = self.micro_batch * self.grad_accum * len(self.workers) * self.seq_len
        if actual != self.global_batch_tokens:
            raise ValueError(
                f"fixed-work invariant broken: micro_batch({self.micro_batch}) × "
                f"grad_accum({self.grad_accum}) × workers({len(self.workers)}) × "
                f"seq_len({self.seq_len}) = {actual}, "
                f"but global_batch_tokens = {self.global_batch_tokens}"
            )
        return self


class ShareRules(BaseModel):
    model_config = ConfigDict(extra="forbid")
    enabled: bool = True
    not_gaming: bool = True
    always_between: tuple[str, str] | None = None     # ("00:00", "08:00")


# ─────────────────────────────────────────────────────────────────────────────
# Contract 1 — worker → server events
# ─────────────────────────────────────────────────────────────────────────────
class _Event(BaseModel):
    model_config = ConfigDict(extra="forbid")
    ts: float = Field(default_factory=time.time)


class WorkerRegister(_Event):
    type: Literal["worker.register"] = "worker.register"
    worker_id: str
    owner: str
    vendor: Vendor
    gpu: str
    vram_gb: float
    cc: float
    chip_class: ChipClass
    share: ShareRules
    credits_per_hour: float


class WorkerHeartbeat(_Event):
    type: Literal["worker.heartbeat"] = "worker.heartbeat"
    worker_id: str
    gpu_util: float = Field(ge=0.0, le=1.0)
    mem_used_gb: float


class WorkerLeft(_Event):
    type: Literal["worker.left"] = "worker.left"
    worker_id: str
    reason: LeaveReason


class TrainStep(_Event):
    type: Literal["train.step"] = "train.step"
    worker_id: str
    step: int
    loss: float
    step_time_s: float
    tokens: int


class TrainSync(_Event):
    type: Literal["train.sync"] = "train.sync"
    round: int
    participants: list[str]          # shorter list == someone left. That's the demo.
    bytes: int
    duration_s: float
    global_loss: float               # fixed held-out batch. The migration proof.


class CreditsUpdate(_Event):
    type: Literal["credits.update"] = "credits.update"
    owner: str
    earned_gpu_hours: float
    balance: float


class MigrationProgress(_Event):
    type: Literal["migration.progress"] = "migration.progress"
    job_id: str
    from_worker: str = Field(alias="from")
    to_worker: str = Field(alias="to")
    phase: MigrationPhase
    pct: float = Field(default=0.0, ge=0.0, le=1.0)

    model_config = ConfigDict(extra="forbid", populate_by_name=True)


class ProbeResult(_Event):
    type: Literal["probe.result"] = "probe.result"
    worker_id: str
    chip_class: ChipClass
    config_name: ConfigName
    t_step_median_s: float           # MEDIAN, warmup dropped. Not the mean.
    tokens_per_s: float              # compare on this, never steps/sec
    peak_vram_gb: float
    gpu_util: float
    t_sync_s: float


WorkerEvent = Annotated[
    WorkerRegister
    | WorkerHeartbeat
    | WorkerLeft
    | TrainStep
    | TrainSync
    | CreditsUpdate
    | MigrationProgress
    | ProbeResult,
    Field(discriminator="type"),
]
WORKER_EVENT = TypeAdapter(WorkerEvent)


# ─────────────────────────────────────────────────────────────────────────────
# Contract 2 — server → worker commands
# ─────────────────────────────────────────────────────────────────────────────
class JobStart(_Event):
    type: Literal["job.start"] = "job.start"
    job_id: str
    config: JobConfig


class JobStop(_Event):
    type: Literal["job.stop"] = "job.stop"
    job_id: str


class CheckpointSave(_Event):
    type: Literal["checkpoint.save"] = "checkpoint.save"
    job_id: str
    round: int


class CheckpointLoad(_Event):
    type: Literal["checkpoint.load"] = "checkpoint.load"
    job_id: str
    uri: str


class ConfigUpdate(_Event):
    type: Literal["config.update"] = "config.update"
    micro_batch: int | None = None
    H: int | None = None


Command = Annotated[
    JobStart | JobStop | CheckpointSave | CheckpointLoad | ConfigUpdate,
    Field(discriminator="type"),
]
COMMAND = TypeAdapter(Command)


# ─────────────────────────────────────────────────────────────────────────────
# Contract 4 — checkpoint layout
# ─────────────────────────────────────────────────────────────────────────────
class CheckpointMeta(BaseModel):
    model_config = ConfigDict(extra="forbid")
    round: int
    global_step: int
    global_loss: float
    config_hash: str
    ts: float = Field(default_factory=time.time)


def ckpt_path(job_id: str, round_: int) -> str:
    return f"ckpt/{job_id}/round_{round_}.safetensors"


def ckpt_meta_path(job_id: str, round_: int) -> str:
    return f"ckpt/{job_id}/round_{round_}.meta.json"


# ─────────────────────────────────────────────────────────────────────────────
# stdout line protocol  (Ethan's trainer → Jack's daemon)
#
# ⚠️ EVENTS GO TO STDOUT. EVERYTHING HUMAN GOES TO STDERR.
# torch and friends will print warnings; if those land on stdout the parser
# chokes. Ethan: logging.basicConfig(stream=sys.stderr). Run with PYTHONUNBUFFERED=1.
# ─────────────────────────────────────────────────────────────────────────────
def emit(event: _Event) -> None:
    """Trainer side. One JSON object per line, flushed."""
    print(event.model_dump_json(by_alias=True), file=sys.stdout, flush=True)


def parse_event(line: str) -> WorkerEvent | None:
    """Daemon side. Returns None for junk (stray warnings) — skip, don't crash."""
    line = line.strip()
    if not line or not line.startswith("{"):
        return None
    try:
        return WORKER_EVENT.validate_json(line)
    except ValidationError:
        return None


def parse_command(raw: str | bytes) -> Command:
    """Worker side. Malformed commands SHOULD raise — that's a real bug."""
    return COMMAND.validate_json(raw)
