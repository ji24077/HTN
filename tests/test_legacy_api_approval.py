from __future__ import annotations

import json
from typing import Any

import pytest
from fastapi.testclient import TestClient

from gpushare.dashboard import app as dashboard_app
from gpushare.dashboard import runner
from gpushare.dashboard.workspaces import WorkspaceRegistry


def _client(tmp_path, monkeypatch) -> TestClient:
    monkeypatch.setattr(dashboard_app, "restore_serving", lambda: None)
    monkeypatch.setattr(dashboard_app, "serving", lambda: {"running": False})
    monkeypatch.setattr(dashboard_app.JOBS, "states", lambda: {})
    registry = WorkspaceRegistry(tmp_path / "workspaces.json")
    return TestClient(dashboard_app.build_app(registry))


MUTATING_JOB_ROUTES = [
    (
        "/api/jobs/data",
        {"total": 100, "heldout": 20, "workers": 1},
        "start_data_generation",
    ),
    (
        "/api/jobs/train",
        {"pod_id": "pod-3090", "save_as": "trained", "steps": 10},
        "start_training",
    ),
    (
        "/api/serve",
        {"pod_id": "pod-3090", "model_id": "base", "dtype": "bf16"},
        "start_inference_server",
    ),
    (
        "/api/prefix",
        {"prefix": "shared policy"},
        "runner_set_prefix",
    ),
    (
        "/api/jobs/action/optimize-training",
        {"pod_id": "pod-4090", "task": "extraction"},
        "start_training_optimization",
    ),
    (
        "/api/jobs/action/optimize-runtime",
        {"pod_id": "pod-4090", "task": "extraction"},
        "start_runtime_optimization",
    ),
    (
        "/api/jobs/action/optimize-inference",
        {"pod_id": "pod-4090", "task": "extraction", "model_id": "base"},
        "start_inference_optimization",
    ),
    (
        "/api/jobs/action/migrate-amd-nvidia",
        {
            "source_pod_id": "pod-4090",
            "target_pod_id": "pod-mi300x",
            "total_steps": 8,
            "stop_after": 4,
            "eval_n": 2,
            "prepare_pods": True,
        },
        "start_migration",
    ),
]


@pytest.mark.parametrize(("path", "payload", "runner_name"), MUTATING_JOB_ROUTES)
def test_legacy_paid_or_remote_job_routes_require_explicit_approval(
    path: str,
    payload: dict[str, Any],
    runner_name: str,
    tmp_path,
    monkeypatch,
):
    calls: list[dict[str, Any]] = []

    def fake_runner(*_args, **kwargs):
        calls.append(kwargs)
        return runner.Job(id="approved-job", kind=runner_name, params=kwargs)

    monkeypatch.setattr(dashboard_app, runner_name, fake_runner)
    client = _client(tmp_path, monkeypatch)

    denied = client.post(path, json=payload)

    assert denied.status_code == 403
    assert "approval" in denied.json()["detail"].lower()
    assert calls == [], "the runner was reached before approval was checked"

    accepted = client.post(path, json={**payload, "approved": True})

    assert accepted.status_code == 200
    assert len(calls) == 1
    assert "approved" not in calls[0], "transport consent leaked into the runner contract"


def test_legacy_stop_and_cancel_require_approval(tmp_path, monkeypatch):
    stopped: list[bool] = []
    cancelled: list[str] = []

    def fake_stop():
        stopped.append(True)
        return {"stopped": True}

    def fake_cancel(job_id: str):
        cancelled.append(job_id)
        return runner.Job(id=job_id, kind="test", params={})

    monkeypatch.setattr(dashboard_app, "stop_inference_server", fake_stop)
    monkeypatch.setattr(dashboard_app.JOBS, "cancel", fake_cancel)
    client = _client(tmp_path, monkeypatch)

    assert client.post("/api/serve/stop", json={}).status_code == 403
    assert client.post("/api/jobs/job-1/cancel", json={}).status_code == 403
    assert not stopped and not cancelled

    assert client.post("/api/serve/stop", json={"approved": True}).status_code == 200
    assert client.post("/api/jobs/job-1/cancel", json={"approved": True}).status_code == 200
    assert stopped == [True]
    assert cancelled == ["job-1"]


def test_legacy_planning_action_requires_approval_before_optional_llm_use(tmp_path, monkeypatch):
    calls: list[str] = []

    def fake_action(name: str):
        calls.append(name)
        return dashboard_app.ActionResult(
            action=name,
            from_chip=None,
            to_chip=None,
            decision={},
            reason="test",
            decided_by="rules",
            projection={},
            validation={"status": "pending"},
        )

    monkeypatch.setattr(dashboard_app, "run_action", fake_action)
    client = _client(tmp_path, monkeypatch)

    assert client.post("/api/action/optimize-training", json={}).status_code == 403
    assert calls == []

    response = client.post("/api/action/optimize-training", json={"approved": True})
    assert response.status_code == 200
    assert calls == ["optimize-training"]


def test_legacy_job_errors_are_sanitized_before_they_reach_the_browser(tmp_path, monkeypatch):
    secret = "sk-proj-AbCdEfGhIjKlMnOpQrStUv123456"

    def fail(**_kwargs):
        raise runner.JobError(
            f"ssh root@203.0.113.42 failed reading /Users/demo/private/key: {secret}"
        )

    monkeypatch.setattr(dashboard_app, "start_inference_server", fail)
    client = _client(tmp_path, monkeypatch)

    response = client.post(
        "/api/serve",
        json={"pod_id": "pod-3090", "model_id": "base", "approved": True},
    )

    assert response.status_code == 400
    detail = response.json()["detail"]
    assert "203.0.113.42" not in detail
    assert "/Users/demo" not in detail
    assert secret not in detail
    assert "REDACTED" in detail


def test_build_app_reconciles_persisted_state_after_restoring_serving(tmp_path, monkeypatch):
    registry = WorkspaceRegistry(tmp_path / "workspaces.json")
    jobs = {"job-1": {"id": "job-1", "status": "running"}}
    active = {"running": True, "stale": False, "model_id": "version-1", "pod_id": "pod-1"}
    events: list[tuple[str, Any]] = []

    monkeypatch.setattr(dashboard_app, "restore_serving", lambda: events.append(("restore", None)))
    monkeypatch.setattr(
        dashboard_app, "serving", lambda: events.append(("serving", None)) or active
    )
    monkeypatch.setattr(dashboard_app.JOBS, "states", lambda: jobs)

    def reconcile(job_states, active_serving):
        events.append(("reconcile", (job_states, active_serving)))
        return {}

    monkeypatch.setattr(registry, "reconcile_runtime", reconcile)

    dashboard_app.build_app(registry)

    assert events == [
        ("restore", None),
        ("serving", None),
        ("reconcile", (jobs, active)),
    ]


def test_control_plane_rejects_untrusted_host_and_cross_origin_mutations(tmp_path, monkeypatch):
    calls: list[dict[str, Any]] = []

    def fake_save(**fields):
        calls.append(fields)
        return {**fields, "prompt": "private template"}

    monkeypatch.setattr(dashboard_app, "save_model", fake_save)
    client = _client(tmp_path, monkeypatch)
    payload = {"name": "base", "ref": runner.MODEL_ID_FOR_SERVE, "kind": "base"}

    bad_host = client.get("/api/health", headers={"Host": "attacker.example:8080"})
    bad_origin = client.post(
        "/api/models/save",
        json=payload,
        headers={"Origin": "https://attacker.example"},
    )

    assert bad_host.status_code == 400
    assert bad_origin.status_code == 403
    assert calls == []

    local = client.post(
        "/api/models/save",
        json=payload,
        headers={"Origin": "http://127.0.0.1:8080"},
    )
    assert local.status_code == 200
    assert calls == [payload | {"pod_id": None, "base": None}]
    assert "prompt" not in local.json()


def test_legacy_public_endpoints_strip_paths_prompts_and_private_samples(tmp_path, monkeypatch):
    client = _client(tmp_path, monkeypatch)
    secret = "sk-proj-AbCdEfGhIjKlMnOpQrStUv123456"
    monkeypatch.setattr(
        dashboard_app,
        "available_models",
        lambda: [
            {
                "id": "trained",
                "label": "trained",
                "ref": "/workspace/private/adapter",
                "prompt": "private prompt",
                "metrics": {
                    "detail": f"failed with {secret} at /Users/demo/key",
                    "samples": [{"sentence": "private metric sample"}],
                },
            }
        ],
    )
    monkeypatch.setattr(
        dashboard_app,
        "saved_models",
        lambda: [{"name": "trained", "ref": "/workspace/private/adapter"}],
    )
    monkeypatch.setattr(
        dashboard_app,
        "serving",
        lambda: {
            "running": True,
            "model_id": "trained",
            "pod_id": "pod-3090",
            "model_ref": "/workspace/private/adapter",
            "live_model_ref": "/workspace/private/adapter",
            "prompt_template": "private prompt",
        },
    )
    monkeypatch.setattr(
        dashboard_app,
        "experiment",
        lambda: {
            "model_id": runner.MODEL_ID_FOR_SERVE,
            "prompt": "private prompt",
            "fields": ["name"],
            "example": {"sentence": "repository fixture"},
            "before": {
                "n": 1,
                "json_parse_rate": 0.0,
                "samples": [{"sentence": "private evaluation row", "raw_output": "secret"}],
            },
            "after": None,
            "train": None,
            "data": {"train": 1, "heldout": 1},
            "cost_model": None,
            "latest_run": {
                "job_id": "job-1",
                "pod_id": "pod-3090",
                "local_dir": "/Users/demo/private/run",
                "remote_checkpoint": "/workspace/private/adapter",
            },
        },
    )

    models = client.get("/api/models").json()
    state = client.get("/api/state").json()
    combined = json.dumps({"models": models, "state": state})

    assert "/workspace/private" not in combined
    assert "/Users/demo" not in combined
    assert "private prompt" not in combined
    assert "private evaluation row" not in combined
    assert "private metric sample" not in combined
    assert secret not in combined
    assert models["serving"]["model_id"] == "trained"
    assert state["experiment"]["before"] == {"n": 1, "json_parse_rate": 0.0}


def test_generation_result_serializer_removes_effective_prompt_and_internal_model_path(
    tmp_path, monkeypatch
):
    secret = "sk-proj-AbCdEfGhIjKlMnOpQrStUv123456"
    monkeypatch.setattr(
        dashboard_app,
        "generate",
        lambda **_kwargs: {
            "prompt": "private effective prompt",
            "model": "/workspace/private/model",
            "raw_output": f"answer {secret}",
            "error": "ssh root@203.0.113.1 read /Users/demo/key",
            "latency_s": 1.0,
        },
    )
    client = _client(tmp_path, monkeypatch)

    response = client.post("/api/generate", json={"sentence": "hello"})

    assert response.status_code == 200
    body = response.json()
    assert "prompt" not in body and "model" not in body
    assert secret not in body["raw_output"]
    assert "203.0.113.1" not in body["error"]
    assert "/Users/demo" not in body["error"]
