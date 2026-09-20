"""CPU-only comparison of independently constructed demo report evidence."""

import copy
import importlib.util
import json
import subprocess
import sys
from pathlib import Path

import pytest

SCRIPT = Path(__file__).resolve().parents[1] / "demo" / "compare.py"
SPEC = importlib.util.spec_from_file_location("demo_compare", SCRIPT)
MODULE = importlib.util.module_from_spec(SPEC)
SPEC.loader.exec_module(MODULE)
compare_reports = MODULE.compare_reports


@pytest.fixture
def report():
    return {
        "operation": "axpy", "vendor": "nvidia", "gpu": "RTX 4090",
        "parameters_on_gpu": True, "precision": "float32",
        "global_step": 2, "start_step": 0, "steps_executed": 2,
        "input_sha256": "a" * 64,
        "training_config": {"seed": 42, "learning_rate": 0.005, "batch_size": 257},
        "predictions": [0.0, 0.5, -2.0], "loss_history": [0.5, 0.25],
        "peak_allocated_bytes": 1024, "peak_reserved_bytes": 4096,
        "memory_scope": "PyTorch training allocator only; excludes native kernel allocations",
        "timing_scope": "training loop including correctness checks and step logging",
        "training_seconds": 2.0,
        "custom_kernel_max_abs_error": 1e-7,
        "custom_kernel_shape_checks": [
            {"elements": 0, "max_abs_error": 0.0},
            {"elements": 257, "max_abs_error": 1e-7},
        ],
        "custom_kernel_calls": 3,
    }


def test_same_gpu_validated_timing_and_tolerance(report):
    candidate = copy.deepcopy(report)
    candidate["predictions"][0] = 0.000009
    candidate["training_seconds"] = 1.0
    result = compare_reports(report, candidate)
    assert result["status"] == "passed"
    assert result["predictions"]["max_abs_error"] == 0.000009
    assert result["loss_history"]["passed"]
    assert result["timing"]["speedup"] == 2.0
    assert "not a benchmark" in result["warnings"][0]


def test_cross_gpu_never_reports_speedup(report):
    candidate = {**report, "vendor": "amd", "gpu": "MI300X", "training_seconds": 0.01}
    result = compare_reports(report, candidate)
    assert result["status"] == "passed"
    assert result["timing"]["speedup"] is None
    assert any("no optimization" in warning for warning in result["warnings"])


def test_resumed_segment_compares_final_predictions_only(report):
    candidate = {**report, "start_step": 1, "steps_executed": 1, "loss_history": [99.0]}
    result = compare_reports(report, candidate)
    assert result["status"] == "passed"
    assert result["loss_history"]["compared"] is False
    assert result["timing"]["speedup"] is None
    assert "predictions only" in result["comparison_scope"]


@pytest.mark.parametrize("field,value", [
    ("input_sha256", "b" * 64), ("operation", "different"),
    ("training_config", {"seed": 43, "learning_rate": 0.005, "batch_size": 257}),
])
def test_identity_changes_are_mismatches(report, field, value):
    result = compare_reports(report, {**report, field: value})
    assert result["status"] == "mismatch"
    assert field in result["identity_mismatches"]
    assert "predictions" not in result


@pytest.mark.parametrize("field,value", [
    ("predictions", []), ("predictions", [float("nan")]),
    ("predictions", [True]), ("predictions", ["1"]),
    ("loss_history", [float("inf"), 1.0]), ("loss_history", []),
    ("peak_allocated_bytes", 0), ("peak_reserved_bytes", 0),
    ("peak_allocated_bytes", 1024.0), ("parameters_on_gpu", 1),
    ("precision", "bf16"), ("vendor", []), ("vendor", None),
    ("global_step", True), ("training_seconds", 0),
    ("custom_kernel_shape_checks", []), ("custom_kernel_shape_checks", None),
    ("custom_kernel_max_abs_error", float("nan")), ("custom_kernel_calls", 0),
])
def test_invalid_evidence_is_rejected(report, field, value):
    with pytest.raises(ValueError):
        compare_reports(report, {**report, field: value})


@pytest.mark.parametrize("field,value", [
    ("predictions", [0.0, 0.5, -1.0]), ("predictions", [0.0]),
    ("loss_history", [0.5, 0.9]),
])
def test_numerical_or_shape_mismatch_never_claims_speedup(report, field, value):
    result = compare_reports(report, {**report, field: value, "training_seconds": 0.1})
    assert result["status"] == "mismatch"
    assert result["timing"]["speedup"] is None


def test_zero_step_validation_and_different_timing_scopes_have_no_speedup(report):
    no_steps = {**report, "start_step": 2, "steps_executed": 0, "loss_history": [], "training_seconds": 0}
    assert compare_reports(no_steps, no_steps)["timing"]["speedup"] is None
    candidate = {**report, "timing_scope": "different work"}
    assert compare_reports(report, candidate)["timing"]["speedup"] is None


@pytest.mark.parametrize("kind,expected_code", [("passed", 0), ("mismatch", 1), ("invalid", 2)])
def test_standalone_cli_json_stdout_and_exit_codes(tmp_path, report, kind, expected_code):
    baseline = tmp_path / "baseline.json"
    candidate = tmp_path / "candidate.json"
    baseline.write_text(json.dumps(report), encoding="utf-8")
    changed = copy.deepcopy(report)
    if kind == "mismatch":
        changed["predictions"][0] = 10
    elif kind == "invalid":
        del changed["custom_kernel_shape_checks"]
    candidate.write_text(json.dumps(changed), encoding="utf-8")
    proc = subprocess.run([sys.executable, "-I", str(SCRIPT), str(baseline), str(candidate)],
                          capture_output=True, text=True, timeout=10)
    assert proc.returncode == expected_code
    assert json.loads(proc.stdout)["status"] == kind
    assert proc.stderr == ""


def test_invalid_tolerance_and_overflow_still_produce_json_safe_results(report):
    with pytest.raises(ValueError):
        compare_reports(report, report, atol=float("nan"))
    large = {**report, "predictions": [1e308]}
    opposite = {**report, "predictions": [-1e308]}
    result = compare_reports(large, opposite)
    assert result["status"] == "mismatch"
    assert result["predictions"]["difference_overflow"]
    json.dumps(result, allow_nan=False)


def test_kernel_error_cap_cannot_be_relaxed_with_prediction_tolerance(report):
    candidate = copy.deepcopy(report)
    candidate["custom_kernel_max_abs_error"] = 100.0
    candidate["custom_kernel_shape_checks"][1]["max_abs_error"] = 100.0
    with pytest.raises(ValueError, match="absolute error limit"):
        compare_reports(report, candidate, atol=1000.0, rtol=1000.0)


def test_removing_kernel_edge_case_blocks_comparison(report):
    baseline = copy.deepcopy(report)
    baseline["custom_kernel_shape_checks"].append({"elements": 1, "max_abs_error": 0.0})
    baseline["custom_kernel_calls"] = 4
    result = compare_reports(baseline, report)
    assert result["status"] == "mismatch"
    assert "custom_kernel_shape_checks" in result["identity_mismatches"]
    assert "predictions" not in result
