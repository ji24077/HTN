"""Expose the frozen hardware campaign without launching or mutating a workload."""

from __future__ import annotations

import hashlib
import json
from datetime import UTC, datetime
from pathlib import Path

from gpushare.agent.gpu_matrix import BY_KEY, device_matches
from gpushare.agent.latency import compare
from gpushare.agent.prediction_validation import compare_predictions

FOLDER = "demo/results/hardware-matrix-2026-09-19"
SUITE = "demo/validation/cases300.jsonl"
# Pin the frozen campaign's manifest content, not its platform-dependent JSON
# whitespace. Updating this campaign requires an intentional reviewed pin update.
MANIFEST_SHA256 = "0a2e922e32c207141a011a78d9e60e831f9dc8d56c80c634360496fcb9ec2882"


class EvidenceUnavailable(ValueError):
    """The requested recorded comparison does not exist or is incomplete."""


def _manifest(root: Path) -> dict:
    path = (root / FOLDER / "manifest.json").resolve()
    if not path.is_relative_to(root.resolve()):
        raise ValueError("Recorded manifest path leaves the project")
    manifest = json.loads(path.read_text(encoding="utf-8"))
    canonical = json.dumps(manifest, sort_keys=True, separators=(",", ":"), allow_nan=False).encode()
    if hashlib.sha256(canonical).hexdigest() != MANIFEST_SHA256:
        raise ValueError("Recorded campaign manifest differs from its frozen identity")
    return manifest


def _checked(root: Path, name: str, manifest: dict) -> bytes:
    if not isinstance(name, str) or not name:
        raise ValueError("Recorded evidence path is missing")
    path = (root / name).resolve()
    if not path.is_relative_to(root.resolve()):
        raise ValueError("Recorded evidence path leaves the project")
    data = path.read_bytes()
    expected = manifest.get(name)
    if not expected or hashlib.sha256(data).hexdigest() != expected:
        raise ValueError(f"Recorded evidence differs from its manifest: {name}")
    return data


def _decision(value: dict | None, accepted: bool, cases: list[dict]) -> dict:
    if not value:
        return {"status": "not_validated", "cases": 0, "changed_cases": None,
                "reference_correct": None, "candidate_correct": None, "examples": []}
    changes = value.get("changed_or_invalid_cases", [])
    examples = []
    for case in changes:
        index = case.get("source_index")
        if type(index) is not int or not 0 <= index < len(cases):
            raise ValueError("Recorded change does not identify a frozen evaluation case")
        source = cases[index]
        if case.get("id") != source.get("id", str(index)):
            raise ValueError("Recorded change refers to a different source case")
        examples.append({
            "source_index": index, "id": source.get("id", str(index)), "sentence": source["sentence"],
            **{key: case.get(key) for key in ("fields", "before", "after", "expected")},
        })
    return {
        "status": "passed" if accepted and value.get("passed") is True else "rejected",
        "cases": value["n"],
        "changed_cases": len(changes),
        "reference_correct": value.get("reference_correct"),
        "candidate_correct": value.get("candidate_correct"),
        "examples": examples,
    }


def recorded_evidence(root: Path) -> dict:
    manifest = _manifest(root)
    summary = json.loads(_checked(root, f"{FOLDER}/summary.json", manifest))
    suite = _checked(root, SUITE, manifest)
    cases = [json.loads(line) for line in suite.decode("utf-8").splitlines() if line.strip()]
    rows = []
    for row in summary["rows"]:
        baseline = row.get("baseline", {}).get("median_s")
        candidate = row.get("optimized", {}).get("median_s")
        rows.append({
            "key": row["key"], "gpu": row["gpu_id"], "vendor": row["vendor"],
            "measured": row["measured"], "baseline_s": baseline, "candidate_s": candidate,
            "latency_ratio": baseline / candidate if baseline and candidate else None,
            "optimization": _decision(row.get("optimization_300"), row["accepted_optimization"], cases),
            "migration_from_4090": _decision(
                row.get("migration_optimized_300"), row["accepted_migration_from_4090"], cases
            ),
        })
    return {
        "kind": "recorded_hardware_evidence", "recorded_at": "2026-09-19",
        "checkpoint": "Qwen2.5-0.5B + experimental v3b",
        "model_sha256": summary["model_sha256"],
        "dataset_sha256": hashlib.sha256(suite).hexdigest(),
        "model_status": summary["model_status"], "timing_scope": summary["timing_scope"],
        "sampling_note": summary["sampling_note"],
        "quality_note": "Experimental checkpoint; not an approved v2 upgrade. "
                        "Output preservation does not imply perfect accuracy.",
        "rows": rows,
    }


def recheck_evidence(root: Path, gpu_key: str, comparison: str = "optimization") -> dict:
    """Recompute one decision from frozen outputs and already measured timings.

    Every file is hashed before parsing, with a pinned manifest as the trust
    anchor. Only the selected GPU and the RTX 4090 reference are read. No model,
    subprocess, provider, artifact writer or GPU runtime is invoked.
    """
    if gpu_key not in BY_KEY:
        raise EvidenceUnavailable("Choose a GPU key listed in the recorded evidence.")
    if comparison not in {"optimization", "migration_from_4090"}:
        raise EvidenceUnavailable("Comparison must be optimization or migration_from_4090.")
    manifest = _manifest(root)
    index = json.loads(_checked(root, f"{FOLDER}/index.json", manifest))
    suite = _checked(root, SUITE, manifest)
    dataset_hash = hashlib.sha256(suite).hexdigest()
    cases = [json.loads(line) for line in suite.decode("utf-8").splitlines() if line.strip()]
    if len(cases) != 300 or dataset_hash != index["dataset_file_sha256"]:
        raise ValueError("Recheck requires the complete frozen 300-case evaluation set")
    files = index["targets"].get(gpu_key, {})
    required = {"reference", "candidate", "validation", "latency"}
    if not required.issubset(files):
        raise EvidenceUnavailable("This GPU has no complete recorded candidate to recheck; no GPU run was started.")

    cache = {}

    def read(name: str) -> dict:
        if name not in cache:
            cache[name] = json.loads(_checked(root, name, manifest))
        return cache[name]

    anchor_files = index["targets"]["rtx4090"]
    anchor_validation = read(anchor_files["validation"])
    model_hash = index["model_sha256"]

    def checked_target(key: str) -> tuple[dict, dict, dict]:
        source = index["targets"][key]
        validation = read(source["validation"])
        reference = read(source["reference"])
        candidate = read(source["candidate"])
        latency = read(source["latency"])
        target = BY_KEY[key]
        if (validation["model_identity"]["sha256"] != model_hash
                or validation["n"] != len(cases)
                or validation["reference_code_unchanged"] is not True
                or validation["dataset_file_sha256"] != dataset_hash
                or validation["reference_evaluator_sha256"] != anchor_validation["reference_evaluator_sha256"]):
            raise ValueError("Recorded evaluator, model or dataset identity does not match")
        if any(not device_matches(target, report.get("runtime", {}).get("gpu"))
               for report in (reference, candidate)):
            raise ValueError("Recorded evaluation ran on a different GPU")
        if latency.get("status") not in {"measured", "baseline_only"} or "optimized" not in latency.get("summaries", {}):
            raise EvidenceUnavailable("This GPU has no complete recorded candidate timing to recheck.")
        if (latency["model_identity"]["sha256"] != model_hash or latency.get("gpu_verified") is not True
                or latency["environment"]["vendor"] != target.vendor
                or not device_matches(target, latency["environment"]["gpu"])):
            raise ValueError("Recorded performance lacks matching model and GPU provenance")
        return reference, candidate, latency

    reference, candidate, candidate_latency = checked_target(gpu_key)
    reference_latency = candidate_latency
    if comparison == "migration_from_4090":
        reference, _, reference_latency = checked_target("rtx4090")
    elif anchor_validation["model_identity"]["sha256"] != model_hash:
        raise ValueError("The frozen reference evaluator uses a different model")
    quality = compare_predictions(reference, candidate, cases)
    speed = compare(reference_latency, candidate_latency, before_engine="baseline", after_engine="optimized")
    verdict = _decision(quality, speed["speedup_verified"] is True, cases)
    return {
        "gpu_key": gpu_key, "comparison": comparison, "source": "saved_outputs",
        "checked_at": datetime.now(UTC).isoformat(), "verdict": verdict, "status": verdict["status"],
        "model_sha256": model_hash, "dataset_sha256": dataset_hash,
        "recorded_speed_gate_passed": speed["speedup_verified"],
        "detail": "Recomputed all 300 saved outputs and the recorded repeated latency gate. "
                  "No model was run, no new performance measurement was made, and no GPU was rented.",
    }
