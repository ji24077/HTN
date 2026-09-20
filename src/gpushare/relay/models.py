from __future__ import annotations

from datetime import datetime
from enum import StrEnum
from typing import Any, Literal

from pydantic import BaseModel, Field, model_validator


class Workload(StrEnum):
    fine_tune = "fine_tune"
    inference = "inference"
    rendering = "rendering"


class ApprovalKind(StrEnum):
    spend = "spend"
    migration = "migration"
    traffic = "traffic"


class AvailabilityWindow(BaseModel):
    """A local-time recurring window; weekday is Monday=0 through Sunday=6."""

    weekdays: list[int] = Field(default_factory=lambda: [0, 1, 2, 3, 4])
    start: str = "19:00"
    end: str = "07:00"

    @model_validator(mode="after")
    def validate_window(self):
        if not self.weekdays or any(day < 0 or day > 6 for day in self.weekdays):
            raise ValueError("weekdays must contain values from 0 through 6")
        for value in (self.start, self.end):
            try:
                hour, minute = (int(part) for part in value.split(":"))
            except (ValueError, TypeError) as exc:
                raise ValueError("times must use HH:MM") from exc
            if not 0 <= hour <= 23 or not 0 <= minute <= 59:
                raise ValueError("times must use HH:MM")
        if self.start == self.end:
            raise ValueError("an availability window cannot cover zero hours")
        return self


class LeasePolicy(BaseModel):
    owner_id: str = Field(min_length=1)
    timezone: str = "UTC"
    minimum_hourly_price_usd: float = Field(default=0, ge=0)
    windows: list[AvailabilityWindow] = Field(default_factory=lambda: [AvailabilityWindow()])
    allowed_workloads: set[Workload] = Field(default_factory=lambda: set(Workload))
    public_data_only: bool = True
    allowed_regions: set[str] = Field(default_factory=set)
    max_runtime_hours: float | None = Field(default=None, gt=0)
    max_gpu_memory_fraction: float = Field(default=0.9, gt=0, le=1)
    paused: bool = False


class GPUOffer(BaseModel):
    id: str
    provider: str
    listing_type: Literal["managed", "community"]
    evidence: Literal["verified", "simulated"]
    resource_id: str | None = None
    chip: str
    vendor: Literal["nvidia", "amd"]
    vram_gb: float = Field(gt=0)
    hourly_price_usd: float = Field(ge=0)
    performance_index: float = Field(gt=0)
    render_frames_per_hour: float = Field(gt=0)
    network_mbps: float = Field(gt=0)
    latency_ms: float = Field(ge=0)
    trust_score: float = Field(ge=0, le=1)
    failure_rate: float = Field(ge=0, le=1)
    region: str
    supported_workloads: set[Workload] = Field(default_factory=lambda: set(Workload))
    accepts_private_data: bool = True
    available: bool = True
    healthy: bool = True
    lease_policy: LeasePolicy | None = None


class RelayGoal(BaseModel):
    text: str
    workload: Workload
    model: str | None = None
    budget_usd: float = Field(gt=0)
    deadline_hours: float | None = Field(default=None, gt=0)
    estimated_gpu_hours: float | None = Field(default=None, gt=0)
    render_frames: int | None = Field(default=None, gt=0)
    minimum_vram_gb: float = Field(default=16, gt=0)
    data_classification: Literal["public", "private"] = "public"
    required_region: str | None = None
    minimum_trust_score: float = Field(default=0.75, ge=0, le=1)
    preserve_quality: bool = True
    quality_tolerance: float = Field(default=0.02, ge=0, le=1)
    deploy: bool = True


class AllocationCandidate(BaseModel):
    offer: GPUOffer
    eligible: bool
    reasons: list[str] = Field(default_factory=list)
    estimated_hours: float
    estimated_cost_usd: float
    score: float | None = None
    score_components: dict[str, float] = Field(default_factory=dict)


class AllocationDecision(BaseModel):
    selected: AllocationCandidate | None
    candidates: list[AllocationCandidate]
    strategy: str


class QualityMetrics(BaseModel):
    json_validity: float = Field(ge=0, le=1)
    exact_match: float = Field(ge=0, le=1)
    safety_passed: bool
    hallucination_rate: float | None = Field(default=None, ge=0, le=1)
    omission_rate: float | None = Field(default=None, ge=0, le=1)
    latency_ms: float | None = Field(default=None, ge=0)
    throughput: float | None = Field(default=None, ge=0)
    peak_vram_gb: float | None = Field(default=None, ge=0)
    cost_usd: float | None = Field(default=None, ge=0)
    evaluation_set_hash: str = Field(min_length=8)


class VerificationReport(BaseModel):
    status: Literal["pass", "reject"]
    reasons: list[str]
    reference: QualityMetrics
    candidate: QualityMetrics
    comparable: bool
    rollback_required: bool = False


class Approval(BaseModel):
    kind: ApprovalKind
    status: Literal["pending", "approved", "rejected"] = "pending"
    reason: str
    maximum_cost_usd: float | None = None
    actor: str | None = None
    decided_at: str | None = None


class ExecutionRecord(BaseModel):
    agent: str
    job_id: str
    status: str = "dispatched"
    created_at: str


class AuditEvent(BaseModel):
    at: str
    event: str
    detail: dict[str, Any] = Field(default_factory=dict)


class RelayRun(BaseModel):
    id: str
    created_at: str
    status: Literal[
        "awaiting_spend_approval",
        "ready",
        "running",
        "awaiting_migration_approval",
        "awaiting_traffic_approval",
        "deployed",
        "rejected",
        "rolled_back",
    ]
    goal: RelayGoal
    allocation: AllocationDecision
    training_plan: dict[str, Any]
    inference_plan: dict[str, Any]
    migration_plan: dict[str, Any]
    approvals: dict[ApprovalKind, Approval]
    verification: VerificationReport | None = None
    executions: list[ExecutionRecord] = Field(default_factory=list)
    active_deployment: dict[str, Any] | None = None
    previous_deployment: dict[str, Any] | None = None
    audit: list[AuditEvent] = Field(default_factory=list)


def utc_now() -> str:
    return datetime.now().astimezone().isoformat(timespec="seconds")
