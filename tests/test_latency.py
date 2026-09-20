import copy
import json

import pytest

from gpushare.agent.latency import (
    benchmark_exit_code,
    compare,
    contract,
    digest,
    model_identity,
    read_cases,
    summarize,
)

RECORD = {"name": "Finn", "age": 18, "org": "University of Toronto", "role": "student", "year": 2026}
CASES = [{"id": "finn", "sentence": "In 2026, Finn, 18, became a student at the University of Toronto.", "record": RECORD}]


def report():
    return {"contract": contract(CASES, {"sha256": "a" * 64}, max_new=128, rounds=3, warmups=2),
            "cases": copy.deepcopy(CASES), "gpu_verified": True, "environment": {"gpu_uuid": "gpu-1"},
            "samples": [{"engine": engine, "case_id": "finn", "round": r, "latency_s": time,
                         "raw_output": json.dumps(RECORD)}
                        for engine, time in (("baseline", 1.), ("optimized", .5)) for r in range(3)]}


def test_compares_same_values_and_complete_repeated_latency():
    r = report()
    c = compare(r, r, before_engine="baseline", after_engine="optimized")
    assert c["same_json_values"] and c["speedup_verified"] and c["same_device"]
    assert c["median_latency_ratio"] == 2


def test_rejected_quality_is_a_process_failure_even_when_measurements_complete():
    r = report()
    r["status"] = "measured"
    r["samples"][-1]["raw_output"] = json.dumps({**RECORD, "role": "student and teacher"})
    r["comparison"] = compare(r, r, before_engine="baseline", after_engine="optimized")
    assert benchmark_exit_code(r) == 2
    assert benchmark_exit_code(r, measure_only=True) == 0


@pytest.mark.parametrize("status", ["failed", "starting", "warming", "measuring"])
def test_incomplete_runs_never_exit_success_even_in_measure_only_mode(status):
    assert benchmark_exit_code({"status": status}, measure_only=True) == 1


def test_baseline_only_is_not_an_accepted_optimization():
    assert benchmark_exit_code({"status": "baseline_only"}) == 2
    assert benchmark_exit_code({"status": "measured", "comparison": {"speedup_verified": True}}) == 0


def test_one_changed_answer_fails_even_when_aggregate_accuracy_is_unchanged():
    r = report()
    r["samples"][-1]["raw_output"] = json.dumps({**RECORD, "name": "Fin"})
    c = compare(r, r, before_engine="baseline", after_engine="optimized")
    assert not c["same_json_values"] and not c["speedup_verified"]
    assert len(c["mismatches"]) == 1


def test_matching_invalid_outputs_never_count_as_parity():
    r = report()
    for s in r["samples"]:
        s["raw_output"] = "not JSON"
    assert not compare(r, r, before_engine="baseline", after_engine="optimized")["same_json_values"]


@pytest.mark.parametrize("value", [0, -1, float("nan"), float("inf"), True])
def test_bad_timings_rejected(value):
    r = report()
    r["samples"][0]["latency_s"] = value
    with pytest.raises(ValueError, match="positive"):
        summarize(r, "baseline")


@pytest.mark.parametrize("mutation", ["missing", "duplicate", "extra"])
def test_missing_or_duplicate_trials_cannot_appear_faster(mutation):
    r = report()
    if mutation == "missing":
        r["samples"].pop()
    elif mutation == "duplicate":
        r["samples"].append(r["samples"][-1])
    else:
        r["samples"].append({**r["samples"][-1], "round": 3})
    with pytest.raises(ValueError, match="measurements"):
        compare(r, r, before_engine="baseline", after_engine="optimized")


def test_hardware_migration_is_labelled_separately():
    r, s = report(), report()
    s["environment"]["gpu_uuid"] = "gpu-2"
    c = compare(r, s, before_engine="baseline", after_engine="optimized")
    assert c["comparison_kind"] == "hardware migration" and not c["same_device"]


def test_different_precision_or_model_is_not_comparable():
    for field, value in (("precision", "fp16"), ("model_sha256", "b" * 64)):
        r, s = report(), report()
        s["contract"][field] = value
        with pytest.raises(ValueError, match="different model"):
            compare(r, s, before_engine="baseline", after_engine="optimized")


def test_whitespace_and_key_order_may_change_but_values_cannot():
    r = report()
    r["samples"][-1]["raw_output"] = json.dumps(RECORD, sort_keys=True, indent=2)
    assert compare(r, r, before_engine="baseline", after_engine="optimized")["same_json_values"]


def test_cases_have_strict_schema_and_unique_ids(tmp_path):
    p = tmp_path / "cases.json"
    p.write_text(json.dumps(CASES), encoding="utf-8")
    assert digest(read_cases(p)) == digest(CASES)
    p.write_text(json.dumps(CASES * 2), encoding="utf-8")
    with pytest.raises(ValueError, match="unique"):
        read_cases(p)


def test_model_identity_ignores_download_metadata_but_detects_weight_changes(tmp_path):
    base, adapter = tmp_path / "base", tmp_path / "adapter"
    for path in (base, adapter):
        path.mkdir()
        (path / "model.safetensors").write_bytes(b"frozen-test-weights")
        (path / "config.json").write_text('{}', encoding="utf-8")
    frozen = model_identity(base, adapter)
    metadata = base / ".cache/huggingface/trees"
    metadata.mkdir(parents=True)
    (metadata / "download.json").write_text('{"download_time":123}', encoding="utf-8")
    assert model_identity(base, adapter) == frozen
    (adapter / "model.safetensors").write_bytes(b"different-weights")
    assert model_identity(base, adapter)["sha256"] != frozen["sha256"]
