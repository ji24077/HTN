"""Persisted preparation outcomes exported as structured Sentry Logs after commit."""

import hashlib
import logging
from contextlib import suppress
from datetime import UTC, datetime

from ..shared.protocol import json_text
from ..shared.telemetry import Scrubber, secret_values

log = logging.getLogger("orchestrator.training_preparation")
# Success is part of the dataset, even when the application's root level is WARNING.
log.setLevel(logging.INFO)


def record(job, stage, outcome, *, evidence=None):
    evidence = evidence or {}
    data = job["data"]
    state = data["training_preparation"]
    plan = data["program_plan"]
    prep = plan["preparation"]
    attempt = state["attempts"].get(stage, 0)
    candidate = state.get("candidate_hash", job["original_hash"])
    if outcome in {"skipped", "kept_baseline"} or stage in {"training", "preparation"}:
        candidate = state["accepted_hash"]
    if "error_type" in evidence:
        candidate = ""  # No new artifact exists when proposal generation fails.
    executed = bool(evidence.get("result") or evidence.get("report"))
    identity = f"{job['job_id']}:{stage}:{attempt}:{candidate}:{outcome}"
    identifier = hashlib.sha256(identity.encode()).hexdigest()
    events = state.setdefault("telemetry", [])
    if any(item["preparation_event_id"] == identifier for item in events):
        return
    now = datetime.now(UTC)
    event = {
        "event_name": "training.preparation.outcome",
        "preparation_event_id": identifier,
        "job_id": job["job_id"],
        "stage": stage,
        "outcome": outcome,
        "attempt": attempt,
        "recorded_at": now.isoformat(),
        "source_worker_id": prep["source_worker_id"],
        "target_worker_id": plan["worker_id"],
        "source_vendor": prep["source_vendor"],
        "target_vendor": prep["target_vendor"],
        "candidate_hash": candidate,
        "original_hash": job["original_hash"],
        "accepted_hash": state["accepted_hash"],
        "worker_id": (data.get("workers") or [""])[0] if executed else "",
        "task_id": (data.get("tasks") or [""])[0] if executed else "",
        "max_attempts": data["limits"]["adaptations"],
        "measurement_scope": "native_preprocessing",
        "evidence_kind": "worker_execution" if executed else "lifecycle",
    }
    if state.get("started_at"):
        event["elapsed_seconds"] = (
            now - datetime.fromisoformat(state["started_at"])
        ).total_seconds()
    if data.get("decisions"):
        decision = data["decisions"][-1]
        event["agent_model"] = decision.get("model", "unknown")
        # These are counters, not credentials; avoid token-named keys in the scrubber.
        for key, value in (decision.get("usage") or {}).items():
            event["model_" + key.replace("tokens", "units")] = value
    event["evidence_sha256"] = hashlib.sha256(json_text(evidence).encode()).hexdigest()
    event["reason"] = str(evidence.get("error", evidence.get("reason", "")))[:2000]
    if "error_type" in evidence:
        event["evidence_kind"] = "agent_error"
        event["error_type"] = evidence["error_type"]
    if "static_checks" in evidence:
        event["evidence_kind"] = "static_checks"
        event["findings_json"] = json_text(evidence["static_checks"])[:8000]
    result = evidence.get("result", {})
    report = evidence.get("report") or result.get("preparation", {}).get("report", {})
    for key in (
        "gpu",
        "torch_version",
        "training_seconds",
        "peak_allocated_bytes",
        "peak_reserved_bytes",
        "global_step",
        "parameters_on_gpu",
    ):
        if key in report:
            event[key] = report[key]
    if report.get("loss_history"):
        event["loss_final"] = report["loss_history"][-1]
    if "compute_seconds" in result:
        event["compute_seconds"] = result["compute_seconds"]
    performance = evidence.get("performance")
    if performance:
        event["native_improvement_min"] = min(
            item["paired_median_improvement_fraction"] for item in performance
        )
        event["performance_json"] = json_text(performance)
    elif result.get("preparation", {}).get("benchmark"):
        benchmark = result["preparation"]["benchmark"]
        event["benchmark_json"] = json_text(benchmark)[:16000]
    events.append(Scrubber(secret_values()).scrub(event))


def publish(events):
    """Sentry failures cannot fail a job; persisted IDs allow later export/deduplication."""
    for item in events:
        # Telemetry integrations are third-party handlers; preserve the durable
        # job outcome even if a handler raises an unexpected exception.
        with suppress(Exception):
            event = Scrubber(secret_values()).scrub(item)
            level = (
                logging.WARNING
                if event["outcome"] in {"rejected", "failed", "cancelled"}
                else logging.INFO
            )
            log.log(
                level, "Training preparation %s: %s", event["stage"], event["outcome"], extra=event
            )
