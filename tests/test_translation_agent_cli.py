"""Offline CLI integration: instructions load; lessons never self-promote."""

import hashlib
import importlib.util
import json
from pathlib import Path

import pytest

from gpushare.portability.memory import SkillStore
from gpushare.portability.providers import ModelError

ROOT = Path(__file__).resolve().parents[1]


@pytest.fixture
def cli(tmp_path, monkeypatch):
    spec = importlib.util.spec_from_file_location("_translation_agent_cli", ROOT / "scripts/translation_agent.py")
    module = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(module)
    monkeypatch.setattr(module, "ROOT", tmp_path)
    source = tmp_path / "source.py"
    source.write_text("# Offline CLI fixture; never executed.\n", encoding="utf-8")
    observed = {"model_clients": [], "runs": [], "gpu_verified": False}

    class StubModelClient:
        def __init__(self, provider, env_file, *, budget_usd):
            observed["model_clients"].append((provider, env_file, budget_usd))

    class StubAgent:
        def __init__(self, client, skills, *, max_attempts, event):
            observed["skills"] = skills
            observed["max_attempts"] = max_attempts

        def run(self, source, out, *, target, validator):
            observed["runs"].append({"source": source, "target": target, "validator": validator})
            out.mkdir(parents=True, exist_ok=False)
            verified = observed["gpu_verified"]
            status = "verified_correctness" if verified else "prepared_unverified"
            lesson = observed.get("lesson", "A proposed project-specific launch setting.")
            result = {
                "status": status,
                "source_sha256": hashlib.sha256(source.encode()).hexdigest(),
                "skills_sha256": hashlib.sha256(observed["skills"].encode()).hexdigest(),
                "gpu_verified": verified,
                "performance_verified": False,
                "target": target,
                "attempts": [{
                    "phase": "translate", "source_sha256": "c" * 64,
                    "gpu_verified": verified, "status": status,
                }],
                "lessons": [{
                    "phase": "translate", "text": lesson,
                    "evidence_source_sha256": "c" * 64,
                    "status": "candidate_pending_regression",
                }],
                "model_usage": {"calls": 0},
            }
            if "error" in observed:
                result["error"] = observed["error"]
            if observed.get("no_lessons"):
                result["lessons"] = []
            (out / "result.json").write_text(json.dumps(result), encoding="utf-8")
            return result

    monkeypatch.setattr(module, "ModelClient", StubModelClient)
    monkeypatch.setattr(module, "EngineeringAgent", StubAgent)
    return module, source, observed


def arguments(source, out, *extra):
    return [str(source), "--out", str(out), *extra]


@pytest.mark.parametrize("verified", [False, True])
def test_demo_proposals_stay_inactive_with_matching_output_and_stored_evidence(
    cli, tmp_path, monkeypatch, capsys, verified
):
    module, source, observed = cli
    observed["gpu_verified"] = verified
    store_path = tmp_path / ".gpushare/skill-memory/demo"
    store = SkillStore(store_path, project_id="demo")
    active_before = store.active_text()

    def forbidden_promotion(*args, **kwargs):
        pytest.fail("A single demo must never invoke skill promotion")

    monkeypatch.setattr(SkillStore, "promote", forbidden_promotion)
    out = tmp_path / "run"
    assert module.main(arguments(source, out)) == 0
    result = json.loads(capsys.readouterr().out)
    assert result == json.loads((out / "result.json").read_text(encoding="utf-8"))
    assert observed["skills"] == active_before
    assert store.active_text() == active_before
    assert result["skill_memory"]["promoted"] is False
    proposal = result["skill_memory"]["proposals"][0]
    candidate_id = proposal["candidate_id"]
    assert candidate_id == result["lessons"][0]["candidate_id"]
    assert proposal["gpu_verified"] is verified
    expected_status = "candidate_pending_regression" if verified else "candidate_pending_gpu_and_regression"
    assert proposal["status"] == expected_status
    record = json.loads((store_path / "candidates" / f"{candidate_id}.json").read_text(encoding="utf-8"))
    assert record["scope"] == {"kind": "project", "project_id": "demo"}
    assert record["evidence"]["candidate_source_sha256"] == "c" * 64
    assert record["evidence"]["gpu_verified"] is verified
    assert record["evidence"]["fixed_suite_verified"] is False
    assert record["evidence"]["verification_scope"] == "single_demo_only"
    snapshot = out / "execution-result.json"
    assert record["evidence"]["run_summary_sha256"] == hashlib.sha256(snapshot.read_bytes()).hexdigest()
    assert record["evidence"]["run_summary_artifact"] == str(snapshot.resolve())
    assert json.loads(snapshot.read_text(encoding="utf-8"))["status"] == result["status"]


def test_cli_uses_active_version_instead_of_reloading_base_skill(cli, tmp_path, capsys):
    module, source, observed = cli
    store = SkillStore(tmp_path / "custom-skills", base_text="Base instructions.\n", project_id="example")
    candidate_id = store.propose(
        "Previously verified local project lesson.",
        {"kind": "project", "project_id": "example"},
        {"fixture": "synthetic unit-test verification only"},
    )
    # Fixture promotion is not evidence that a real GPU ran in this test.
    outcome = store.promote(candidate_id, expected_suite_sha256="a" * 64, verify=lambda text: {
        "passed": True, "gpu_verified": True, "suite_sha256": "a" * 64,
        "cases": [{"id": "fixture", "passed": True}],
    })
    assert outcome["promoted"] is True
    active = store.active_text()
    out = tmp_path / "custom-run"
    assert module.main(arguments(
        source, out, "--project-id", "example", "--skills-dir", str(store.root),
        "--provider", "openai", "--model-budget", "0.5", "--attempts", "2",
    )) == 0
    assert observed["skills"] == active
    assert observed["model_clients"] == [("openai", tmp_path / ".env", 0.5)]
    assert observed["max_attempts"] == 2
    assert store.active_text() == active
    assert json.loads(capsys.readouterr().out)["skill_memory"]["project_id"] == "example"


def test_cli_does_not_expose_raw_provider_exceptions(cli, tmp_path, monkeypatch, capsys):
    module, source, observed = cli

    def failed_client(*args, **kwargs):
        raise ModelError("Provider echoed credential very-secret-api-token in its exception")

    monkeypatch.setattr(module, "ModelClient", failed_client)
    assert module.main(arguments(source, tmp_path / "run")) == 1
    captured = capsys.readouterr()
    assert captured.out == ""
    error = json.loads(captured.err)
    assert error["error_type"] == "ModelError"
    assert "very-secret" not in captured.err
    assert "Traceback" not in captured.err
    assert not observed["runs"]


def test_cli_redacts_known_keys_from_final_output_and_candidate_memory(cli, tmp_path, capsys):
    module, source, observed = cli
    secret = "fixture-baseten-secret-key-value"
    (tmp_path / ".env").write_text(f"BASETEN_API_KEY={secret}\n", encoding="utf-8")
    observed["lesson"] = f"Malformed proposal contained {secret}; omit it."
    observed["error"] = f"Unexpected upstream echo: {secret}"
    out = tmp_path / "run"
    assert module.main(arguments(source, out)) == 0
    captured = capsys.readouterr()
    assert secret not in captured.out + captured.err
    assert "[REDACTED]" in captured.out
    assert secret not in (out / "result.json").read_text(encoding="utf-8")
    for path in (tmp_path / ".gpushare/skill-memory/demo").rglob("*.json"):
        assert secret not in path.read_text(encoding="utf-8")


def test_candidate_storage_failure_preserves_run_result_with_failure_status(
    cli, tmp_path, monkeypatch, capsys
):
    module, source, observed = cli

    def fail_proposal(*args, **kwargs):
        raise PermissionError("opaque-secret-in-filesystem-exception")

    monkeypatch.setattr(SkillStore, "propose", fail_proposal)
    out = tmp_path / "run"
    assert module.main(arguments(source, out)) == 1
    captured = capsys.readouterr()
    result = json.loads(captured.out)
    assert result["status"] == "prepared_unverified"
    assert result["skill_memory"]["status"] == "candidate_storage_failed"
    assert result["skill_memory"]["proposals"][0]["candidate_id"] is None
    assert "opaque-secret" not in captured.out + captured.err
    assert json.loads((out / "result.json").read_text(encoding="utf-8")) == result


def test_no_lessons_does_not_create_or_activate_candidate(cli, tmp_path, capsys):
    module, source, observed = cli
    observed["no_lessons"] = True
    assert module.main(arguments(source, tmp_path / "run")) == 0
    result = json.loads(capsys.readouterr().out)
    assert result["skill_memory"]["status"] == "no_candidates"
    assert result["skill_memory"]["proposals"] == []
    assert not list((tmp_path / ".gpushare/skill-memory/demo/candidates").glob("*.json"))


def test_existing_output_is_rejected_before_model_initialization(cli, tmp_path, capsys):
    module, source, observed = cli
    out = tmp_path / "existing"
    out.mkdir()
    (out / "keep.txt").write_text("keep", encoding="utf-8")
    assert module.main(arguments(source, out)) == 1
    captured = capsys.readouterr()
    assert "choose a new --out" in json.loads(captured.err)["error"]
    assert (out / "keep.txt").read_text(encoding="utf-8") == "keep"
    assert observed["model_clients"] == []


def test_bad_connection_json_is_safe_and_makes_no_model_request(cli, tmp_path, capsys):
    module, source, observed = cli
    connections = tmp_path / "connections.json"
    connections.write_text('{"secret-that-could-be-in-input":', encoding="utf-8")
    assert module.main(arguments(source, tmp_path / "run", "--connections", str(connections))) == 1
    captured = capsys.readouterr()
    assert "secret-that" not in captured.err
    assert "Traceback" not in captured.err
    assert observed["model_clients"] == []


def test_wrong_project_store_cannot_feed_instructions_to_model(cli, tmp_path, capsys):
    module, source, observed = cli
    SkillStore(tmp_path / "other-store", base_text="Other project.", project_id="other")
    assert module.main(arguments(source, tmp_path / "run", "--skills-dir", str(tmp_path / "other-store"))) == 1
    captured = capsys.readouterr()
    assert json.loads(captured.err)["error_type"] == "SkillStoreError"
    assert not observed["model_clients"]


def test_interrupt_is_reported_without_a_traceback(cli, tmp_path, monkeypatch, capsys):
    module, source, observed = cli

    def interrupted(*args, **kwargs):
        raise KeyboardInterrupt()

    monkeypatch.setattr(module, "ModelClient", interrupted)
    assert module.main(arguments(source, tmp_path / "run")) == 130
    captured = capsys.readouterr()
    assert json.loads(captured.err)["status"] == "interrupted"
    assert "Traceback" not in captured.err
