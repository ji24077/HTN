from __future__ import annotations

import hashlib
import json
import re
import threading
import uuid
from pathlib import Path
from typing import Any

from .marketplace import JobAllocationAgent
from .models import (
    Approval,
    ApprovalKind,
    AuditEvent,
    ExecutionRecord,
    GPUOffer,
    QualityMetrics,
    RelayGoal,
    RelayRun,
    Workload,
    utc_now,
)
from .skills import SkillRegistry
from .verification import QualityVerifier


class RelayError(RuntimeError):
    pass


class ApprovalRequired(RelayError):
    pass


def parse_goal(text: str) -> RelayGoal:
    """Parse the controlled MVP requests without outsourcing policy to an LLM."""
    lowered = text.lower()
    budget_match = re.search(r"(?:under|below|budget|예산|이하)\s*\$?\s*(\d+(?:\.\d+)?)", lowered)
    if not budget_match:
        budget_match = re.search(r"\$\s*(\d+(?:\.\d+)?)", lowered)
    budget = float(budget_match.group(1)) if budget_match else 20.0
    frame_match = re.search(r"(\d[\d,]*)\s*(?:blender\s*)?frames?", lowered)
    if "blender" in lowered or "render" in lowered or "렌더" in text:
        workload = Workload.rendering
    elif "fine-tune" in lowered or "finetune" in lowered or "train" in lowered or "학습" in text:
        workload = Workload.fine_tune
    else:
        workload = Workload.inference
    model_match = re.search(r"(qwen(?:2\.5|3)?[-/\w.]+)", text, re.IGNORECASE)
    deadline = 24.0 if "tomorrow" in lowered or "내일" in text else None
    hour_match = re.search(r"(?:within|in|under|이내)\s*(\d+(?:\.\d+)?)\s*(?:hours?|시간)", lowered)
    if hour_match:
        deadline = float(hour_match.group(1))
    private = any(word in lowered for word in ("private", "confidential", "sensitive")) or "비공개" in text
    unchanged = "unchanged" in lowered or "동일" in text or "변경 없이" in text
    return RelayGoal(
        text=text,
        workload=workload,
        model=model_match.group(1) if model_match else ("Qwen/Qwen2.5-0.5B" if workload == Workload.fine_tune else None),
        budget_usd=budget,
        deadline_hours=deadline,
        render_frames=int(frame_match.group(1).replace(",", "")) if frame_match else None,
        minimum_vram_gb=16 if workload != Workload.rendering else 8,
        data_classification="private" if private else "public",
        preserve_quality=True,
        quality_tolerance=0.0 if unchanged else 0.02,
        deploy="deploy" in lowered or "배포" in text or workload != Workload.rendering,
    )


class RelayService:
    """Persistent human-approved workflow around the six specialized agents."""

    def __init__(self, *, skills_root: Path, state_path: Path | None = None):
        self.skills = SkillRegistry(skills_root)
        self.state_path = state_path
        self.allocator = JobAllocationAgent()
        self.verifier = QualityVerifier()
        self._runs: dict[str, RelayRun] = {}
        self._lock = threading.RLock()
        self._load()

    def _load(self) -> None:
        if self.state_path is None or not self.state_path.is_file():
            return
        try:
            payload = json.loads(self.state_path.read_text(encoding="utf-8"))
            self._runs = {key: RelayRun.model_validate(value) for key, value in payload.items()}
        except (OSError, ValueError, json.JSONDecodeError):
            self._runs = {}

    def _save(self) -> None:
        if self.state_path is None:
            return
        self.state_path.parent.mkdir(parents=True, exist_ok=True)
        temp = self.state_path.with_suffix(".tmp")
        temp.write_text(
            json.dumps(
                {key: run.model_dump(mode="json") for key, run in self._runs.items()},
                indent=2,
                ensure_ascii=False,
            ),
            encoding="utf-8",
        )
        temp.replace(self.state_path)

    def create_plan(self, text: str, offers: list[GPUOffer]) -> RelayRun:
        self.skills.load()  # fail closed if an agent contract is absent
        goal = parse_goal(text)
        if goal.workload == Workload.fine_tune:
            requested = (goal.model or "").lower()
            if any(name in text.lower() for name in ("llama", "mistral", "gemma")) or requested not in {
                "qwen2.5-0.5b",
                "qwen/qwen2.5-0.5b",
            }:
                raise RelayError("the MVP training workflow supports Qwen/Qwen2.5-0.5B only")
        allocation = self.allocator.allocate(goal, offers)
        selected = allocation.selected
        run_id = uuid.uuid4().hex
        approvals = {
            ApprovalKind.spend: Approval(
                kind=ApprovalKind.spend,
                reason="Authorize a maximum charge before any paid GPU job starts.",
                maximum_cost_usd=selected.estimated_cost_usd if selected else goal.budget_usd,
            ),
            ApprovalKind.migration: Approval(
                kind=ApprovalKind.migration,
                reason="Authorize copying a checkpoint to, and testing it on, a candidate chip.",
            ),
            ApprovalKind.traffic: Approval(
                kind=ApprovalKind.traffic,
                reason="Authorize switching live inference traffic only after verification passes.",
            ),
        }
        chip = selected.offer.chip if selected else None
        vendor = selected.offer.vendor if selected else None
        training = {
            "agent": "training-optimizer",
            "model": goal.model,
            "gpu": chip,
            "candidates": ["LoRA", "QLoRA"] if vendor == "nvidia" else ["LoRA"],
            "batch_search": "constant tokens/step across batch×accum candidates",
            "checkpoint_policy": "quarterly-and-final",
            "selection": "lowest measured cost among finite-loss candidates that pass held-out quality",
        }
        inference = {
            "agent": "inference-optimizer",
            "gpu": chip,
            "tests": {
                "runtimes": ["eager", "torch.compile"],
                "batching": [1, 8, 16],
                "kv_cache": [False, True],
                "quantization": ["bf16", "int8", "nf4"] if vendor == "nvidia" else ["bf16"],
                "routing": "lowest cost/1M tokens among quality-preserving candidates",
            },
        }
        migration = {
            "agent": "chip-migration",
            "candidate_chips": ["RTX 4090", "RTX A5000", "L40S", "MI300X"],
            "metrics": ["latency", "cost", "VRAM", "exact match", "JSON validity"],
            "gate": f"same held-out hash; quality delta >= -{goal.quality_tolerance:.3f}",
            "on_failure": "keep current deployment or roll back if candidate was applied",
        }
        now = utc_now()
        status = "awaiting_spend_approval" if selected else "rejected"
        audit = [AuditEvent(at=now, event="plan.created", detail={"offer_count": len(offers)})]
        if selected is None:
            audit.append(AuditEvent(at=now, event="allocation.rejected", detail={"reason": "no eligible GPU"}))
        elif selected.offer.evidence == "simulated":
            audit.append(
                AuditEvent(
                    at=now,
                    event="allocation.simulated",
                    detail={"warning": "catalog estimates require a verified provider before execution"},
                )
            )
        run = RelayRun(
            id=run_id,
            created_at=now,
            status=status,
            goal=goal,
            allocation=allocation,
            training_plan=training,
            inference_plan=inference,
            migration_plan=migration,
            approvals=approvals,
            audit=audit,
        )
        with self._lock:
            self._runs[run.id] = run
            self._save()
        return run

    def list(self) -> list[RelayRun]:
        return sorted(self._runs.values(), key=lambda item: item.created_at, reverse=True)

    def get(self, run_id: str) -> RelayRun:
        try:
            return self._runs[run_id]
        except KeyError as exc:
            raise RelayError(f"Relay run {run_id} was not found") from exc

    def approve(self, run_id: str, kind: ApprovalKind, *, actor: str, approve: bool = True) -> RelayRun:
        with self._lock:
            run = self.get(run_id)
            approval = run.approvals[kind]
            approval.status = "approved" if approve else "rejected"
            approval.actor = actor
            approval.decided_at = utc_now()
            run.audit.append(
                AuditEvent(at=approval.decided_at, event=f"approval.{approval.status}", detail={"kind": kind.value, "actor": actor})
            )
            if not approve:
                run.status = "rejected" if kind == ApprovalKind.spend else "rolled_back"
            elif kind == ApprovalKind.spend and run.status == "awaiting_spend_approval":
                run.status = "ready"
            elif kind == ApprovalKind.migration and run.status == "awaiting_migration_approval":
                run.status = "ready"
            elif kind == ApprovalKind.traffic and run.status == "awaiting_traffic_approval":
                run.status = "ready"
            self._save()
            return run

    def authorize_execution(self, run_id: str, agent: str) -> RelayRun:
        run = self.get(run_id)
        if run.allocation.selected is None:
            raise RelayError("no eligible GPU was allocated")
        if run.allocation.selected.offer.evidence != "verified":
            raise RelayError("the selected provider card is simulated; choose a connected verified GPU")
        if run.approvals[ApprovalKind.spend].status != "approved":
            raise ApprovalRequired("spend approval is required before starting a paid GPU job")
        if agent == "chip-migration" and run.approvals[ApprovalKind.migration].status != "approved":
            run.status = "awaiting_migration_approval"
            self._save()
            raise ApprovalRequired("migration approval is required before copying the checkpoint")
        if agent == "deploy":
            if run.verification is None or run.verification.status != "pass":
                raise RelayError("deployment requires a passing verification report")
            if run.approvals[ApprovalKind.traffic].status != "approved":
                run.status = "awaiting_traffic_approval"
                self._save()
                raise ApprovalRequired("traffic approval is required before switching inference traffic")
        return run

    def record_execution(self, run_id: str, agent: str, job_id: str) -> RelayRun:
        with self._lock:
            run = self.get(run_id)
            run.executions.append(ExecutionRecord(agent=agent, job_id=job_id, created_at=utc_now()))
            run.status = "running"
            run.audit.append(AuditEvent(at=utc_now(), event="agent.dispatched", detail={"agent": agent, "job_id": job_id}))
            self._save()
            return run

    def reconcile_execution(
        self,
        run_id: str,
        job_id: str,
        *,
        status: str,
        result: dict[str, Any] | None = None,
        error: str | None = None,
    ) -> RelayRun:
        """Fold a real runner result into the independent Relay verification gate."""
        with self._lock:
            run = self.get(run_id)
            record = next((item for item in run.executions if item.job_id == job_id), None)
            if record is None:
                raise RelayError(f"job {job_id} is not part of Relay run {run_id}")
            if record.status == status and status in {
                "complete",
                "failed",
                "cancelled",
                "interrupted",
            }:
                return run
            record.status = status
            if status in {"failed", "cancelled", "interrupted"}:
                run.status = "rolled_back" if record.agent in {"chip-migration", "deploy"} else "rejected"
                if run.status == "rolled_back":
                    run.active_deployment = run.previous_deployment
                run.audit.append(
                    AuditEvent(
                        at=utc_now(),
                        event="agent.failed",
                        detail={"agent": record.agent, "job_id": job_id, "error": error or status},
                    )
                )
                self._save()
                return run
            if status != "complete":
                self._save()
                return run
            if record.agent in {"deploy", "rollback"}:
                run.audit.append(
                    AuditEvent(
                        at=utc_now(),
                        event="agent.complete",
                        detail={"agent": record.agent, "job_id": job_id},
                    )
                )
                self._save()
                return run

            pair = _quality_pair(record.agent, result or {})
            validation = (result or {}).get("validation") or {}
            if pair is None:
                run.status = "rejected"
                run.audit.append(
                    AuditEvent(
                        at=utc_now(),
                        event="verification.reject",
                        detail={"reasons": ["completed job returned no comparable quality reports"]},
                    )
                )
                self._save()
                return run
            reference_report, candidate_report = pair
            reference = _quality_metrics(reference_report, result or {}, safety_passed=True)
            candidate = _quality_metrics(
                candidate_report,
                result or {},
                safety_passed=bool(validation.get("safety_passed", validation.get("status") == "ok")),
            )
            if reference is None or candidate is None:
                run.status = "rejected"
                run.audit.append(
                    AuditEvent(
                        at=utc_now(),
                        event="verification.reject",
                        detail={"reasons": ["completed job did not identify its evaluation set"]},
                    )
                )
                self._save()
                return run
            report = self.verifier.verify(
                reference,
                candidate,
                tolerance=run.goal.quality_tolerance,
                candidate_applied=False,
            )
            run.verification = report
            run.status = (
                "awaiting_traffic_approval"
                if report.status == "pass" and run.goal.deploy
                else "ready"
                if report.status == "pass"
                else "rejected"
            )
            run.audit.append(
                AuditEvent(
                    at=utc_now(),
                    event=f"verification.{report.status}",
                    detail={"agent": record.agent, "job_id": job_id, "reasons": report.reasons},
                )
            )
            self._save()
            return run

    def verify(
        self,
        run_id: str,
        reference: QualityMetrics,
        candidate: QualityMetrics,
        *,
        candidate_applied: bool = False,
    ) -> RelayRun:
        with self._lock:
            run = self.get(run_id)
            report = self.verifier.verify(
                reference,
                candidate,
                tolerance=run.goal.quality_tolerance,
                candidate_applied=candidate_applied,
            )
            run.verification = report
            if report.status == "pass":
                run.status = "awaiting_traffic_approval" if run.goal.deploy else "ready"
            else:
                run.status = "rolled_back" if report.rollback_required else "rejected"
                if report.rollback_required:
                    run.active_deployment = run.previous_deployment
            run.audit.append(
                AuditEvent(
                    at=utc_now(),
                    event=f"verification.{report.status}",
                    detail={"reasons": report.reasons, "rollback": report.rollback_required},
                )
            )
            self._save()
            return run

    def mark_deployed(self, run_id: str, deployment: dict[str, Any]) -> RelayRun:
        with self._lock:
            run = self.authorize_execution(run_id, "deploy")
            run.previous_deployment = run.active_deployment
            run.active_deployment = deployment
            run.status = "deployed"
            run.audit.append(AuditEvent(at=utc_now(), event="traffic.switched", detail=deployment))
            self._save()
            return run

    def record_current_deployment(self, run_id: str, deployment: dict[str, Any]) -> RelayRun:
        """Snapshot the known-good route before any candidate can replace it."""
        with self._lock:
            run = self.get(run_id)
            run.active_deployment = deployment
            run.audit.append(AuditEvent(at=utc_now(), event="deployment.snapshot", detail=deployment))
            self._save()
            return run

    def rollback(self, run_id: str, reason: str) -> RelayRun:
        with self._lock:
            run = self.get(run_id)
            run.active_deployment = run.previous_deployment
            run.status = "rolled_back"
            run.audit.append(AuditEvent(at=utc_now(), event="deployment.rolled_back", detail={"reason": reason}))
            self._save()
            return run


def _quality_pair(agent: str, result: dict[str, Any]) -> tuple[dict[str, Any], dict[str, Any]] | None:
    if agent == "training-optimizer":
        quality = result.get("quality") or {}
        before, after = quality.get("reference"), quality.get("chosen")
    elif agent == "inference-optimizer":
        before, after = result.get("baseline"), result.get("optimized")
    elif agent == "chip-migration":
        before, after = result.get("before"), result.get("after")
    else:
        return None
    return (before, after) if isinstance(before, dict) and isinstance(after, dict) else None


def _quality_metrics(
    report: dict[str, Any],
    result: dict[str, Any],
    *,
    safety_passed: bool,
) -> QualityMetrics | None:
    identity = report.get("evaluation") or result.get("evaluation")
    if not isinstance(identity, dict):
        return None
    encoded = json.dumps(identity, sort_keys=True, separators=(",", ":")).encode("utf-8")
    runtime = report.get("runtime") if isinstance(report.get("runtime"), dict) else {}
    elapsed = report.get("elapsed_s")
    return QualityMetrics(
        json_validity=float(report.get("json_parse_rate", 0)),
        exact_match=float(report.get("exact_match_rate", 0)),
        safety_passed=safety_passed,
        hallucination_rate=report.get("hallucination_rate"),
        omission_rate=report.get("omission_rate"),
        latency_ms=float(elapsed) * 1000 if elapsed is not None else None,
        throughput=report.get("tokens_per_second"),
        peak_vram_gb=runtime.get("peak_allocated_gb"),
        evaluation_set_hash=hashlib.sha256(encoded).hexdigest(),
    )
