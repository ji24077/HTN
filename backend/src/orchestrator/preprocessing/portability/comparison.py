"""Compare demo reports using only Python's standard library.

Usage: python demo/compare.py baseline.json candidate.json
Exit codes: 0 = equivalent within tolerance, 1 = mismatch, 2 = invalid input.
This checks reported measurements; it does not execute either GPU program.
"""
# Preserve the upstream comparator's ValueError contract for malformed evidence.
# ruff: noqa: TRY004

from __future__ import annotations

import argparse
import json
import math
import re
from pathlib import Path

KERNEL_ABSOLUTE_ERROR_LIMIT = 1e-5


def _number(value, label: str, *, minimum: float | None = None) -> float:
    if isinstance(value, bool) or not isinstance(value, (int, float)):
        raise ValueError(f"{label} must be a finite number")
    try:
        value = float(value)
    except OverflowError as exc:
        raise ValueError(f"{label} is outside the supported numeric range") from exc
    if not math.isfinite(value) or (minimum is not None and value < minimum):
        raise ValueError(f"{label} must be finite and at least {minimum}")
    return value


def _integer(value, label: str, *, minimum: int = 0) -> int:
    if type(value) is not int or value < minimum:
        raise ValueError(f"{label} must be an integer at least {minimum}")
    return value


def _text(value, label: str) -> str:
    if not isinstance(value, str) or not value.strip():
        raise ValueError(f"{label} must be a nonempty string")
    return value


def _vector(value, label: str, *, allow_empty: bool = False) -> list[float]:
    if not isinstance(value, list) or (not value and not allow_empty):
        raise ValueError(f"{label} must be a {'possibly empty' if allow_empty else 'nonempty'} numeric list")
    return [_number(item, f"{label}[{index}]") for index, item in enumerate(value)]


def _validate(report, label: str) -> None:
    if not isinstance(report, dict):
        raise ValueError(f"{label} must be a JSON object")
    for key in ("operation", "gpu", "memory_scope", "timing_scope"):
        _text(report.get(key), f"{label}.{key}")
    if report.get("vendor") not in ("nvidia", "amd"):
        raise ValueError(f"{label}.vendor must be nvidia or amd")
    if report.get("parameters_on_gpu") is not True:
        raise ValueError(f"{label}.parameters_on_gpu must be true")
    if report.get("precision") != "float32":
        raise ValueError(f"{label}.precision must be float32")
    digest = report.get("input_sha256")
    if not isinstance(digest, str) or not re.fullmatch(r"[a-f0-9]{64}", digest):
        raise ValueError(f"{label}.input_sha256 must be a lowercase SHA256 digest")
    step = _integer(report.get("global_step"), f"{label}.global_step")
    start = _integer(report.get("start_step"), f"{label}.start_step")
    executed = _integer(report.get("steps_executed"), f"{label}.steps_executed")
    if start > step or executed != step - start:
        raise ValueError(f"{label} has inconsistent start/global/executed steps")
    config = report.get("training_config")
    if not isinstance(config, dict) or set(config) != {"seed", "learning_rate", "batch_size"}:
        raise ValueError(f"{label}.training_config must contain seed, learning_rate, batch_size")
    _integer(config["seed"], f"{label}.training_config.seed")
    _integer(config["batch_size"], f"{label}.training_config.batch_size", minimum=1)
    if _number(config["learning_rate"], f"{label}.training_config.learning_rate", minimum=0) == 0:
        raise ValueError(f"{label}.training_config.learning_rate must be positive")
    _vector(report.get("predictions"), f"{label}.predictions")
    if "initial_predictions" in report:
        _vector(report["initial_predictions"], f"{label}.initial_predictions")
    losses = _vector(report.get("loss_history"), f"{label}.loss_history", allow_empty=executed == 0)
    if len(losses) != executed:
        raise ValueError(f"{label}.loss_history length must equal steps_executed")
    allocated = _integer(report.get("peak_allocated_bytes"), f"{label}.peak_allocated_bytes")
    reserved = _integer(report.get("peak_reserved_bytes"), f"{label}.peak_reserved_bytes")
    if allocated > reserved or (executed > 0 and (allocated == 0 or reserved == 0)):
        raise ValueError(f"{label} has inconsistent or zero training allocator memory")
    seconds = _number(report.get("training_seconds"), f"{label}.training_seconds", minimum=0)
    if executed > 0 and seconds == 0:
        raise ValueError(f"{label}.training_seconds must be positive when steps were executed")
    checks = report.get("custom_kernel_shape_checks")
    if not isinstance(checks, list) or not checks:
        raise ValueError(f"{label}.custom_kernel_shape_checks must be a nonempty list")
    max_error = _number(report.get("custom_kernel_max_abs_error"), f"{label}.custom_kernel_max_abs_error", minimum=0)
    if max_error > KERNEL_ABSOLUTE_ERROR_LIMIT:
        raise ValueError(f"{label}.custom_kernel_max_abs_error exceeds this fixed demo's {KERNEL_ABSOLUTE_ERROR_LIMIT} absolute error limit")
    seen = set()
    for index, check in enumerate(checks):
        if not isinstance(check, dict):
            raise ValueError(f"{label}.custom_kernel_shape_checks[{index}] must be an object")
        count = _integer(check.get("elements"), f"{label}.kernel_check.elements")
        error = _number(check.get("max_abs_error"), f"{label}.kernel_check.max_abs_error", minimum=0)
        if count in seen or error > max_error:
            raise ValueError(f"{label} has duplicated or inconsistent kernel shape checks")
        seen.add(count)
    if 0 not in seen or not any(count > 0 for count in seen):
        raise ValueError(f"{label} must check both empty and nonempty custom-kernel inputs")
    _integer(report.get("custom_kernel_calls"), f"{label}.custom_kernel_calls", minimum=len(checks) + 1)


def _compare_vector(baseline: list, candidate: list, atol: float, rtol: float) -> dict:
    if len(baseline) != len(candidate):
        return {"passed": False, "reason": "different lengths", "baseline_count": len(baseline), "candidate_count": len(candidate)}
    differences = [abs(float(a) - float(b)) for a, b in zip(baseline, candidate, strict=True)]
    failures = sum(
        not math.isfinite(error) or error > atol + rtol * abs(float(reference))
        for error, reference in zip(differences, baseline, strict=True)
    )
    maximum = max(differences, default=0.0)
    return {"passed": failures == 0, "count": len(baseline), "mismatch_count": failures,
            "max_abs_error": maximum if math.isfinite(maximum) else None,
            "difference_overflow": not math.isfinite(maximum)}


def compare_reports(baseline: dict, candidate: dict, *, atol: float = 1e-5, rtol: float = 1e-4) -> dict:
    """Raise ValueError for invalid evidence; otherwise return passed or mismatch."""
    atol = _number(atol, "atol", minimum=0)
    rtol = _number(rtol, "rtol", minimum=0)
    _validate(baseline, "baseline")
    _validate(candidate, "candidate")
    identity = ("operation", "global_step", "input_sha256", "precision", "training_config")
    differences = [key for key in identity if baseline[key] != candidate[key]]
    baseline_shapes = {check["elements"] for check in baseline["custom_kernel_shape_checks"]}
    candidate_shapes = {check["elements"] for check in candidate["custom_kernel_shape_checks"]}
    if baseline_shapes != candidate_shapes:
        differences.append("custom_kernel_shape_checks")
    result = {"status": "mismatch" if differences else "passed", "atol": atol, "rtol": rtol,
              "identity_mismatches": differences,
              "verification_scope": "Comparison of supplied reports, not a new GPU execution",
              "warnings": ["Timing is diagnostic, not a benchmark."]}
    if differences:
        return result
    result["predictions"] = _compare_vector(baseline["predictions"], candidate["predictions"], atol, rtol)
    same_start = baseline["start_step"] == candidate["start_step"]
    if same_start:
        result["loss_history"] = _compare_vector(baseline["loss_history"], candidate["loss_history"], atol, rtol)
        result["comparison_scope"] = "final predictions and loss history over the same step range"
    else:
        result["loss_history"] = {"compared": False, "reason": "different start_step"}
        result["comparison_scope"] = "final predictions only; training segments start at different steps"
    same_gpu = baseline["gpu"] == candidate["gpu"] and baseline["vendor"] == candidate["vendor"]
    timing_reasons = []
    numerical_match = result["predictions"]["passed"] and (not same_start or result["loss_history"]["passed"])
    if not numerical_match:
        timing_reasons.append("numerical mismatch")
    if not same_gpu:
        timing_reasons.append("different GPU or vendor")
        result["warnings"].append("Cross-GPU comparison makes no optimization or speedup claim.")
    if baseline["timing_scope"] != candidate["timing_scope"]:
        timing_reasons.append("different timing_scope")
    if not same_start:
        timing_reasons.append("different start_step")
    if baseline["steps_executed"] <= 0 or baseline["steps_executed"] != candidate["steps_executed"]:
        timing_reasons.append("steps_executed must be positive and equal")
    timing = {"comparable": not timing_reasons, "reasons": timing_reasons, "speedup": None}
    if not timing_reasons:
        speedup = baseline["training_seconds"] / candidate["training_seconds"]
        if math.isfinite(speedup):
            timing["speedup"] = speedup
        else:
            timing.update(comparable=False, reasons=["timing ratio overflow"])
    result["timing"] = timing
    if not numerical_match:
        result["status"] = "mismatch"
    return result


class _Parser(argparse.ArgumentParser):
    def error(self, message):
        raise ValueError(message)


def main(argv: list[str] | None = None) -> int:
    parser = _Parser(description=__doc__)
    parser.add_argument("baseline", type=Path)
    parser.add_argument("candidate", type=Path)
    parser.add_argument("--atol", type=float, default=1e-5)
    parser.add_argument("--rtol", type=float, default=1e-4)
    try:
        args = parser.parse_args(argv)
        baseline = json.loads(args.baseline.read_text(encoding="utf-8"))
        candidate = json.loads(args.candidate.read_text(encoding="utf-8"))
        result = compare_reports(baseline, candidate, atol=args.atol, rtol=args.rtol)
    except (OSError, ValueError, TypeError) as exc:
        result = {"status": "invalid", "error": str(exc)}
    print(json.dumps(result, indent=2, allow_nan=False))
    return {"passed": 0, "mismatch": 1, "invalid": 2}[result["status"]]


if __name__ == "__main__":
    raise SystemExit(main())
