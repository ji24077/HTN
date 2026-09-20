"""Recheck frozen evidence using raw outputs, without model or provider access."""

import hashlib
import json
import shutil
import subprocess
from datetime import datetime
from pathlib import Path

import pytest
from fastapi.testclient import TestClient

from gpushare.dashboard import app as dashboard
from gpushare.dashboard import evidence

ROOT = Path(__file__).resolve().parents[1]


@pytest.mark.parametrize("comparison", ["optimization", "migration_from_4090"])
def test_every_measured_recheck_matches_frozen_decision_and_complete_case_details(comparison):
    saved = evidence.recorded_evidence(ROOT)
    for row in saved["rows"]:
        if not row["measured"]:
            continue
        result = evidence.recheck_evidence(ROOT, row["key"], comparison)
        assert result["verdict"] == row[comparison]
        assert result["model_sha256"] == saved["model_sha256"]
        assert result["dataset_sha256"] == saved["dataset_sha256"]
        assert result["source"] == "saved_outputs"
        assert datetime.fromisoformat(result["checked_at"]).tzinfo is not None
        assert "No model was run" in result["detail"]
        for case in result["verdict"]["examples"]:
            assert isinstance(case["source_index"], int)
            assert case["id"] and case["sentence"]


def test_every_changed_case_is_available_with_original_sentence():
    saved = evidence.recorded_evidence(ROOT)
    mi300x = next(row for row in saved["rows"] if row["key"] == "mi300x")
    suite = [json.loads(line) for line in (ROOT / evidence.SUITE).read_text(encoding="utf-8").splitlines() if line.strip()]
    for comparison, expected in (("optimization", 13), ("migration_from_4090", 15)):
        changes = mi300x[comparison]["examples"]
        assert len(changes) == mi300x[comparison]["changed_cases"] == expected
        for change in changes:
            assert change["sentence"] == suite[change["source_index"]]["sentence"]
            assert change["id"] == suite[change["source_index"]]["id"]


def test_recheck_recomputes_without_loading_summary_or_other_gpu_reports(monkeypatch):
    checked = evidence._checked
    read_names = []
    def check(root, name, manifest):
        read_names.append(name)
        assert not name.endswith("summary.json"), "a saved verdict is not a recomputation"
        assert not any(f"/{key}/" in name for key in ("a5000", "l40s", "h100", "a100"))
        return checked(root, name, manifest)
    monkeypatch.setattr(evidence, "_checked", check)
    result = evidence.recheck_evidence(ROOT, "mi300x", "migration_from_4090")
    assert result["verdict"]["changed_cases"] == 15
    assert any(name.endswith("compile-strict.json") for name in read_names)
    assert any(name.endswith("reference.json") for name in read_names)


@pytest.fixture
def copied_bundle(tmp_path):
    index_name = f"{evidence.FOLDER}/index.json"
    manifest_name = f"{evidence.FOLDER}/manifest.json"
    index = json.loads((ROOT / index_name).read_text(encoding="utf-8"))
    names = {index_name, manifest_name, evidence.SUITE}
    for key in ("mi300x", "rtx4090"):
        names.update(index["targets"][key][kind] for kind in ("reference", "candidate", "validation", "latency"))
    for name in names:
        destination = tmp_path / name
        destination.parent.mkdir(parents=True, exist_ok=True)
        shutil.copy2(ROOT / name, destination)
    return tmp_path, index


@pytest.mark.parametrize("tamper_manifest", [False, True])
def test_modified_raw_outputs_fail_even_when_manifest_is_rewritten(copied_bundle, tamper_manifest):
    root, index = copied_bundle
    candidate_name = index["targets"]["mi300x"]["candidate"]
    candidate = root / candidate_name
    candidate.write_bytes(candidate.read_bytes() + b" ")
    if tamper_manifest:
        manifest_path = root / evidence.FOLDER / "manifest.json"
        manifest = json.loads(manifest_path.read_text(encoding="utf-8"))
        manifest[candidate_name] = hashlib.sha256(candidate.read_bytes()).hexdigest()
        manifest_path.write_text(json.dumps(manifest), encoding="utf-8")
    with pytest.raises(ValueError, match="manifest"):
        evidence.recheck_evidence(root, "mi300x")


def test_modified_suite_fails_before_comparing(copied_bundle):
    root, _ = copied_bundle
    suite = root / evidence.SUITE
    suite.write_bytes(suite.read_bytes() + b"\n")
    with pytest.raises(ValueError, match="manifest"):
        evidence.recheck_evidence(root, "mi300x")


def test_checked_paths_cannot_leave_project(tmp_path):
    with pytest.raises(ValueError, match="leaves the project"):
        evidence._checked(tmp_path, "../outside.json", {"../outside.json": "unused"})


@pytest.mark.parametrize("key,comparison", [
    ("rtx5090", "optimization"), ("unknown", "optimization"),
    ("mi300x", "live"), ("..\\secrets", "optimization"),
])
def test_missing_or_invalid_comparisons_return_clear_422(key, comparison):
    client = TestClient(dashboard.build_app(read_only_demo=True))
    response = client.get(f"/api/evidence/{key}/recheck", params={"comparison": comparison})
    assert response.status_code == 422
    assert response.json()["detail"]


def test_recheck_api_is_read_only_and_has_no_cloud_side_effects(monkeypatch):
    client = TestClient(dashboard.build_app(read_only_demo=True))
    def forbidden(*args, **kwargs):
        pytest.fail("saved-output recheck must not launch processes, contact providers, or write files")
    for name in ("restore_serving", "list_pods", "start_inference_server"):
        monkeypatch.setattr(dashboard, name, forbidden)
    monkeypatch.setattr(subprocess, "Popen", forbidden)
    monkeypatch.setattr(Path, "write_text", forbidden)
    monkeypatch.setattr(Path, "write_bytes", forbidden)
    response = client.get("/api/evidence/a5000/recheck", params={"comparison": "optimization"})
    assert response.status_code == 200
    result = response.json()
    assert result["status"] == result["verdict"]["status"] == "passed"
    assert result["recorded_speed_gate_passed"] is True
    assert result["verdict"]["cases"] == 300


def test_recheck_api_reports_integrity_failure_without_a_verdict(copied_bundle, monkeypatch):
    root, index = copied_bundle
    (root / index["targets"]["mi300x"]["latency"]).write_text("{}", encoding="utf-8")
    monkeypatch.setattr(dashboard, "ROOT", root)
    client = TestClient(dashboard.build_app(read_only_demo=True))
    response = client.get("/api/evidence/mi300x/recheck")
    assert response.status_code == 503
    assert "integrity" in response.json()["detail"]
    assert "verdict" not in response.json()
