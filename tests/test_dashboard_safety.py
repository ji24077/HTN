"""Acceptance and serving regressions, using local evidence and fake hosts only."""

import copy
import io
import json
import urllib.request
from types import SimpleNamespace

import pytest

from gpushare.agent.evaluation import evaluation_identity
from gpushare.agent.task import REQUIRED_FIELDS
from gpushare.dashboard import migration, runner


@pytest.fixture
def reports():
    rows = [{"sentence": f"{name}, 30, joined Lab in 2020 as engineer.",
             "record": {"name": name, "age": 30, "org": "Lab", "role": "engineer", "year": 2020}}
            for name in ("Ada", "Bea")]
    report = {"n": len(rows), "evaluation": evaluation_identity(rows, max_new_tokens=128, seq_len=192),
              "inference": {"batch": 1, "dtype": "bf16", "fuse_adapter": False, "length_bucketing": False},
              "json_parse_rate": 1.0, "exact_match_rate": 1.0,
              "field_accuracy": dict.fromkeys(REQUIRED_FIELDS, 1.0),
              "samples": [{"source_index": i, "sentence": row["sentence"], "expected": row["record"],
                           "raw_output": json.dumps(row["record"])} for i, row in enumerate(rows)]}
    return report, copy.deepcopy(report)


def test_strict_gate_checks_each_answer_even_when_aggregate_scores_match(reports):
    before, after = reports
    for report, index in ((before, 0), (after, 1)):
        value = json.loads(report["samples"][index]["raw_output"])
        report["samples"][index]["raw_output"] = json.dumps({**value, "year": 2021})
        report["exact_match_rate"] = 0.5
        report["field_accuracy"]["year"] = 0.5
    gate = runner._quality(before, after)
    assert gate["status"] == "regressed"
    assert len(gate["comparison"]["changed_or_invalid_cases"]) == 2
    assert not gate["output_preservation_verified"]


def test_batch_optimization_can_pass_when_every_retained_answer_matches(reports):
    before, after = reports
    after["inference"]["batch"] = 16
    after["samples"].reverse()
    gate = runner._quality(before, after)
    assert gate["status"] == "ok"
    assert gate["output_preservation_verified"]
    assert gate["comparison"]["n"] == 2


@pytest.mark.parametrize("damage", ["no_samples", "duplicate", "different_suite", "nan", "missing_field", "wrong_count"])
def test_incomplete_or_incomparable_evidence_never_passes(reports, damage):
    before, after = reports
    if damage == "no_samples":
        del after["samples"]
    elif damage == "duplicate":
        after["samples"][1]["source_index"] = 0
    elif damage == "different_suite":
        after["evaluation"]["dataset_sha256"] = "different"
    elif damage == "nan":
        after["exact_match_rate"] = float("nan")
    elif damage == "missing_field":
        del after["field_accuracy"]["year"]
    else:
        after["n"] = 1
    assert runner._quality(before, after)["status"] == "not_validated"


def test_identical_invalid_outputs_fail_preservation(reports):
    before, after = reports
    for report in reports:
        report["samples"][0]["raw_output"] = "not JSON"
    assert runner._quality(before, after)["status"] == "regressed"


def test_training_may_change_answers_without_claiming_output_preservation(reports):
    before, after = reports
    before["exact_match_rate"] = 0.5
    before["field_accuracy"]["year"] = 0.5
    before["samples"][0]["raw_output"] = json.dumps({**before["samples"][0]["expected"], "year": 2021})
    gate = runner._quality(before, after, preserve_outputs=False)
    assert gate["status"] == "ok"
    assert gate["policy"] == "aggregate_no_regression"
    assert not gate["output_preservation_verified"]
    assert runner._quality(before, after)["status"] == "regressed"


def test_training_no_longer_allows_two_percent_regression(reports):
    before, after = reports
    after["exact_match_rate"] = 0.99
    assert runner._quality(before, after, preserve_outputs=False)["status"] == "regressed"
    assert runner._quality(before, after, tol=0.02)["status"] == "not_validated"


def test_transfer_requires_outputs_but_continued_training_has_separate_policy(reports):
    before, after = reports
    del after["samples"]
    with pytest.raises(runner.JobError, match="quality gate"):
        migration._gate(before, after)
    gate = migration._gate(before, after, preserve_outputs=False)
    assert gate["policy"] == "aggregate_no_regression"
    assert not gate["output_preservation_verified"]


@pytest.fixture
def synchronous_jobs(monkeypatch):
    monkeypatch.setattr(runner.JOBS, "update", lambda *args: None)
    monkeypatch.setattr(runner.JOBS, "log", lambda *args: None)
    monkeypatch.setattr(runner.JOBS, "create", lambda kind, params, work: work(runner.Job(id="test-job", kind=kind, params=params)))


@pytest.mark.parametrize("outcome", ["passed", "regressed", "unvalidated"])
def test_training_publishes_latest_only_after_acceptance(tmp_path, monkeypatch, reports, synchronous_jobs, outcome):
    before, after = reports
    if outcome == "regressed":
        after["exact_match_rate"] = 0.99
    elif outcome == "unvalidated":
        del after["evaluation"]
    (tmp_path / "data").mkdir()
    (tmp_path / "data/heldout.jsonl").write_text("{}\n", encoding="utf-8")
    monkeypatch.setattr(runner, "ROOT", tmp_path)
    monkeypatch.setattr(runner, "RUN_ROOT", tmp_path / "runs")
    pod = {"id": "pod-test", "vendor": "nvidia"}
    monkeypatch.setattr(runner, "_place", lambda *a, **k: (pod, {}))
    for name in ("_sync_project", "_setup_pod", "_run", "_remote", "_pull"):
        monkeypatch.setattr(runner, name, lambda *a, **k: 0)
    monkeypatch.setattr(runner, "_ssh_args", lambda *a: [])
    monkeypatch.setattr(runner, "_json", lambda path: before if path.name == "base.json" else after if path.name == "after.json" else {"steps": 10})
    published = []
    monkeypatch.setattr(runner, "_write_latest", published.append)
    result = runner.start_training(pod_id="pod-test", steps=10, dtype="bf16", attention="sdpa", micro_batch=1, grad_accum=1)
    assert bool(published) == (outcome == "passed")
    if published:
        assert published[0]["validation"] == result["validation"]
        assert published[0]["inference"] == after["inference"]


def test_legacy_and_rejected_checkpoints_are_not_offered_for_serving(monkeypatch):
    monkeypatch.setattr(runner, "latest_run", lambda: {"remote_checkpoint": "/old", "pod_id": "pod-old"})
    monkeypatch.setattr(runner.JOBS, "list", lambda: [{"id": "rejected", "kind": "optimize-training-speed", "status": "complete",
                                                    "result": {"remote_checkpoint": "/candidate", "validation": {"status": "regressed"}}}])
    assert {item["id"] for item in runner.available_models()} == {"base", "longctx"}


def test_strict_acceptance_requires_explicit_successful_output_comparison(reports):
    before, after = reports
    gate = runner._quality(before, after)
    assert runner._accepted_validation({"validation": gate})
    del gate["output_preservation_verified"]
    assert not runner._accepted_validation({"validation": gate})


@pytest.fixture
def serving_state(tmp_path, monkeypatch, synchronous_jobs):
    active = {"model_id": "base", "model_ref": runner.MODEL_ID_FOR_SERVE,
              "pod_id": "pod-live", "dtype": "bf16"}
    monkeypatch.setattr(runner, "_serve", active)
    monkeypatch.setattr(runner, "SERVING_PATH", tmp_path / "serving.json")
    monkeypatch.setattr(runner, "latest_run", lambda: None)
    monkeypatch.setattr(runner.JOBS, "list", lambda: [])
    monkeypatch.setattr(runner, "_server_health", lambda **k: {"model": runner.MODEL_ID_FOR_SERVE, "dtype": "bf16"})
    def forbidden(*args, **kwargs):
        pytest.fail("candidate must not touch active baseline or contact hosts")
    for name in ("_pod", "_ssh_info", "_sync_project", "_run"):
        monkeypatch.setattr(runner, name, forbidden)
    return active


def test_different_model_cannot_implicitly_replace_active_baseline(serving_state):
    with pytest.raises(runner.JobError, match="active baseline is retained"):
        runner.start_inference_server(pod_id="pod-other", model_id="longctx")
    assert serving_state["model_id"] == "base"


def test_rejected_model_selection_leaves_baseline_running(serving_state):
    with pytest.raises(runner.JobError, match="unknown model"):
        runner.start_inference_server(pod_id="pod-other", model_id="finetuned")
    assert serving_state["model_id"] == "base"


def test_same_running_model_is_idempotent(serving_state):
    result = runner.start_inference_server(pod_id="pod-live", model_id="base")
    assert result["already_running"]
    assert serving_state["model_id"] == "base"


def test_stale_unhealthy_baseline_is_not_silently_destroyed(serving_state, monkeypatch):
    monkeypatch.setattr(runner, "_server_health", lambda **k: None)
    with pytest.raises(runner.JobError, match="active baseline is retained"):
        runner.start_inference_server(pod_id="pod-live", model_id="base")
    assert serving_state["model_id"] == "base"


def test_explicit_stop_stops_remote_model_and_releases_owned_tunnel(serving_state, monkeypatch):
    calls = []
    serving_state["tunnel"] = SimpleNamespace(terminate=lambda: calls.append("terminate"), wait=lambda **k: calls.append("wait"))
    monkeypatch.setattr(runner, "_ssh_info", lambda *a: {})
    monkeypatch.setattr(runner, "_ssh_args", lambda info, command: [command])
    monkeypatch.setattr(runner, "_capture", lambda argv, **k: calls.append("remote-stop"))
    assert runner.stop_inference_server() == {"stopped": "base"}
    assert calls == ["remote-stop", "terminate", "wait"]
    assert not serving_state


def test_failed_explicit_remote_stop_retains_baseline_record(serving_state, monkeypatch):
    monkeypatch.setattr(runner, "_ssh_info", lambda *a: {})
    monkeypatch.setattr(runner, "_ssh_args", lambda *a: [])
    def fail(*args, **kwargs):
        raise OSError("fake SSH failure")
    monkeypatch.setattr(runner, "_capture", fail)
    with pytest.raises(runner.JobError, match="active serving record retained"):
        runner.stop_inference_server()
    assert serving_state["model_id"] == "base"


@pytest.mark.parametrize("target", ["pod-live", "pod-other"])
def test_restored_orphan_forward_does_not_block_explicit_stop_and_reload(serving_state, monkeypatch, target):
    # Adopted legacy sessions have no Popen handle; port 8100 may remain bound
    # by that orphan SSH process even after the remote model is stopped.
    calls = []
    monkeypatch.setattr(runner, "_ssh_info", lambda pod: {"key": "fake", "ip": "127.0.0.1", "port": 22})
    monkeypatch.setattr(runner, "_ssh_args", lambda info, cmd: [cmd])
    monkeypatch.setattr(runner, "_capture", lambda *a, **k: calls.append("remote-stop"))
    assert runner.stop_inference_server() == {"stopped": "base"}
    monkeypatch.setattr(runner, "_pod", lambda pod: {"id": pod, "vendor": "nvidia"})
    for name in ("_sync_project", "_setup_pod", "_run"):
        monkeypatch.setattr(runner, name, lambda *a, **k: 0)
    monkeypatch.setattr(runner, "_available_forward_port", lambda: 19123)
    forwards = []
    def popen(argv, **kwargs):
        forwards.append(argv[argv.index("-L") + 1])
        return SimpleNamespace(poll=lambda: None, terminate=lambda: None, wait=lambda **k: None)
    monkeypatch.setattr(runner.subprocess, "Popen", popen)
    health_ports = []
    def health(timeout=3.0, *, local_port=None):
        port = local_port if local_port is not None else runner._serve.get("local_port", runner.SERVE_PORT)
        health_ports.append(port)
        assert port == 19123, "readiness must not trust the legacy/orphan forward"
        return {"model": runner.MODEL_ID_FOR_SERVE, "dtype": "bf16"}
    monkeypatch.setattr(runner, "_server_health", health)
    result = runner.start_inference_server(pod_id=target, model_id="base")
    assert result["port"] == runner._serve["local_port"] == 19123
    assert forwards == ["127.0.0.1:19123:127.0.0.1:8100"]
    assert calls == ["remote-stop"]
    assert json.loads(runner.SERVING_PATH.read_text())["local_port"] == 19123
    monkeypatch.setattr(runner, "_serve", {})
    runner.restore_serving()
    assert runner._serve["local_port"] == 19123
    assert runner._serve["pod_id"] == target
    assert runner.serving()["running"]
    assert health_ports == [19123, 19123, 19123]


def test_health_and_all_request_proxies_use_recorded_local_forward(monkeypatch):
    monkeypatch.setattr(runner, "_serve", {"model_id": "base", "local_port": 19123})
    urls = []
    def urlopen(request, **kwargs):
        url = request if isinstance(request, str) else request.full_url
        urls.append(url)
        response = io.BytesIO(b'data: {"done": true}\n\n' if url.endswith("/stream") else b'{}')
        response.status = 200
        return response
    monkeypatch.setattr(urllib.request, "urlopen", urlopen)
    assert runner._server_health() == {}
    runner.generate(sentence="input")
    runner.set_prefix(prefix="policy")
    assert list(runner.generate_stream(sentence="input")) == ['{"done": true}']
    assert urls == [f"http://127.0.0.1:19123/{path}" for path in ("health", "generate", "prefix", "generate/stream")]


def test_wrong_health_identity_cannot_certify_candidate(tmp_path, monkeypatch, synchronous_jobs):
    monkeypatch.setattr(runner, "_serve", {})
    monkeypatch.setattr(runner, "latest_run", lambda: None)
    monkeypatch.setattr(runner.JOBS, "list", lambda: [])
    monkeypatch.setattr(runner, "SERVING_PATH", tmp_path / "serving.json")
    pod = {"id": "pod-test", "vendor": "nvidia"}
    monkeypatch.setattr(runner, "_pod", lambda *a: pod)
    monkeypatch.setattr(runner, "_ssh_info", lambda *a: {"key": "fake", "ip": "127.0.0.1", "port": 22})
    for name in ("_sync_project", "_setup_pod", "_run"):
        monkeypatch.setattr(runner, name, lambda *a, **k: 0)
    monkeypatch.setattr(runner, "_ssh_args", lambda *a: [])
    monkeypatch.setattr(runner, "_capture", lambda *a, **k: "candidate failed")
    monkeypatch.setattr(runner.time, "sleep", lambda *a: None)
    terminated = []
    monkeypatch.setattr(runner.subprocess, "Popen", lambda *a, **k: SimpleNamespace(terminate=lambda: terminated.append(True), poll=lambda: None))
    health_calls = []
    def health(**kwargs):
        health_calls.append(True)
        return None if len(health_calls) == 1 else {"model": "wrong-model", "dtype": "bf16"}
    monkeypatch.setattr(runner, "_server_health", health)
    with pytest.raises(runner.JobError, match="never became ready"):
        runner.start_inference_server(pod_id="pod-test", model_id="base")
    assert terminated
    assert not runner._serve
    assert not runner.SERVING_PATH.exists()
