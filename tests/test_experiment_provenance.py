import hashlib
import json
from pathlib import Path

import pytest
from fastapi.testclient import TestClient

from gpushare.dashboard import app as dashboard
from gpushare.dashboard.evidence import recorded_evidence


def write(path, value):
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(json.dumps(value), encoding="utf-8")


def test_incomplete_latest_run_never_borrows_an_unrelated_result(tmp_path, monkeypatch):
    write(tmp_path / "eval/base.json", {"n": 100, "exact_match_rate": 0.0})
    write(tmp_path / "eval/after.json", {"n": 200, "exact_match_rate": 0.91})
    run = tmp_path / "run"
    write(run / "eval/after.json", {"n": 50, "exact_match_rate": 0.80})
    monkeypatch.setattr(dashboard, "ROOT", tmp_path)
    monkeypatch.setattr(dashboard, "latest_run", lambda: {"local_dir": str(run)})
    result = dashboard.experiment()
    assert result["before"] is None
    assert result["after"]["n"] == 50
    assert result["comparison"]["status"] == "not_validated"


@pytest.mark.parametrize("mode", ["different_suite", "different_count", "no_identity"])
def test_unmatched_evaluations_are_not_presented_as_a_before_after_pair(tmp_path, monkeypatch, mode):
    before = {"n": 200, "evaluation": {"dataset_sha256": "suite-a", "n": 200}}
    after = {"n": 200, "evaluation": {"dataset_sha256": "suite-a", "n": 200}}
    if mode == "different_suite":
        after["evaluation"]["dataset_sha256"] = "suite-b"
    elif mode == "different_count":
        before["n"] = 100
    else:
        before.pop("evaluation")
        after.pop("evaluation")
    write(tmp_path / "eval/base.json", before)
    write(tmp_path / "eval/after.json", after)
    monkeypatch.setattr(dashboard, "ROOT", tmp_path)
    monkeypatch.setattr(dashboard, "latest_run", lambda: None)
    result = dashboard.experiment()
    assert result["before"] is None
    assert result["after"] == after
    assert result["comparison"]["status"] == "not_comparable"


def test_matching_evaluations_remain_visible(tmp_path, monkeypatch):
    report = {"n": 200, "evaluation": {"dataset_sha256": "suite-a", "n": 200}}
    write(tmp_path / "eval/base.json", report)
    write(tmp_path / "eval/after.json", report)
    monkeypatch.setattr(dashboard, "ROOT", tmp_path)
    monkeypatch.setattr(dashboard, "latest_run", lambda: None)
    result = dashboard.experiment()
    assert result["before"] == report
    assert result["comparison"]["status"] == "comparable"


def test_saved_evidence_distinguishes_local_optimization_from_migration():
    result = recorded_evidence(Path(__file__).resolve().parents[1])
    rows = {row["key"]: row for row in result["rows"]}
    assert result["kind"] == "recorded_hardware_evidence"
    assert rows["a5000"]["optimization"]["status"] == "passed"
    assert rows["a5000"]["migration_from_4090"]["status"] == "rejected"
    assert rows["mi300x"]["optimization"]["changed_cases"] == 13
    assert rows["mi300x"]["migration_from_4090"]["changed_cases"] == 15
    assert rows["rtx5090"]["optimization"]["status"] == "not_validated"


def test_evidence_api_fails_closed_when_recorded_files_change(tmp_path, monkeypatch):
    folder = tmp_path / "demo/results/hardware-matrix-2026-09-19"
    write(folder / "summary.json", {"rows": []})
    write(folder / "manifest.json", {
        "demo/results/hardware-matrix-2026-09-19/summary.json": hashlib.sha256(b"original").hexdigest()
    })
    monkeypatch.setattr(dashboard, "ROOT", tmp_path)
    monkeypatch.setattr(dashboard, "restore_serving", lambda: None)
    client = TestClient(dashboard.build_app())
    response = client.get("/api/evidence")
    assert response.status_code == 503
    assert "integrity check" in response.json()["detail"]


def test_evidence_api_never_discovers_or_starts_cloud_resources(monkeypatch):
    def forbidden(*args, **kwargs):
        raise AssertionError("Evidence must come only from local files")

    monkeypatch.setattr(dashboard, "restore_serving", lambda: None)
    monkeypatch.setattr(dashboard, "list_pods", forbidden)
    monkeypatch.setattr(dashboard, "start_inference_server", forbidden)
    client = TestClient(dashboard.build_app())
    response = client.get("/api/evidence")
    assert response.status_code == 200
    assert len(response.json()["rows"]) == 12


def test_recorded_demo_has_no_cloud_discovery_or_mutations(monkeypatch):
    def forbidden(*args, **kwargs):
        raise AssertionError("Recorded demo must not touch cloud resources")

    for name in ("restore_serving", "list_pods", "serving", "start_inference_server"):
        monkeypatch.setattr(dashboard, name, forbidden)
    client = TestClient(dashboard.build_app(read_only_demo=True))
    assert client.get("/api/health").json()["demo_read_only"] is True
    assert client.get("/api/pods").json() == {"pods": []}
    assert client.get("/api/models").json()["serving"]["running"] is False
    assert client.get("/api/jobs").json() == {"jobs": []}
    assert client.get("/api/evidence").status_code == 200
    assert client.post("/api/serve", json={"pod_id": "demo", "model_id": "base"}).status_code == 403
    assert client.post("/api/serve/stop").status_code == 403
    assert client.post("/api/jobs/train", json={"pod_id": "demo"}).status_code == 403
