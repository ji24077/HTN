"""Engineering-loop behavior with local fake models/validators; no network or GPU."""

import hashlib
import json
from pathlib import Path

import pytest

from gpushare.portability.engine import EngineeringAgent
from gpushare.portability.providers import ModelError
from gpushare.portability.scripts import get_native_source

SOURCE = (Path(__file__).resolve().parents[1] / "demo" / "train.py").read_text(encoding="utf-8")


class Client:
    def __init__(self, transform=None):
        self.requests = []
        self.transform = transform

    def propose(self, system, request):
        self.requests.append(request)
        native = request["native_source"]
        if self.transform:
            value = self.transform(request, len(self.requests))
            if isinstance(value, dict):
                return value
            native = value
        return {"native_source": native, "reason": "fixture proposal", "lesson": "candidate lesson"}

    def summary(self):
        return {"calls": len(self.requests), "estimated_cost_usd": 0}


@pytest.mark.parametrize("newline", ["\n", "\r\n"])
def test_source_hashes_match_exact_exported_bytes_across_platform_newlines(tmp_path, newline):
    result = EngineeringAgent(Client(), "skills").run(
        SOURCE.replace("\n", newline), tmp_path / "output")
    assert result["status"] == "prepared_unverified"
    files = [(tmp_path / "output/original.py", result["source_sha256"]),
             (Path(result["output"]), result["output_sha256"])]
    files.extend((Path(attempt["candidate"]), attempt["source_sha256"])
                 for attempt in result["attempts"])
    for path, expected in files:
        payload = path.read_bytes()
        assert b"\r" not in payload
        assert hashlib.sha256(payload).hexdigest() == expected


def test_no_validator_never_claims_gpu_or_performance_success(tmp_path):
    client = Client()
    result = EngineeringAgent(client, "fixed skills").run(SOURCE, tmp_path / "output")
    assert result["status"] == "prepared_unverified"
    assert result["gpu_verified"] is False and result["performance_verified"] is False
    assert all(item["gpu_verified"] is False for item in result["attempts"])
    assert all(lesson["status"] == "candidate_pending_regression" for lesson in result["lessons"])
    assert "hip/hip_runtime.h" in Path(result["output"]).read_text(encoding="utf-8")
    saved = json.loads((tmp_path / "output/result.json").read_text(encoding="utf-8"))
    assert saved["status"] == result["status"]


def test_failed_baseline_blocks_model_requests_and_persists_result(tmp_path):
    client = Client()
    phases = []

    def validate(path, phase, attempt):
        phases.append(phase)
        return {"passed": False, "gpu_verified": False, "error": "native memory unavailable"}

    result = EngineeringAgent(client, "skills").run(SOURCE, tmp_path / "output", validator=validate)
    assert result["status"] == "blocked" and result["gpu_verified"] is False
    assert phases == ["baseline"] and client.requests == []
    assert json.loads((tmp_path / "output/result.json").read_text())["status"] == "blocked"


def test_static_guard_rejection_reaches_repair_without_remote_execution(tmp_path):
    def proposal(request, count):
        if count == 1:
            return request["native_source"].replace("alpha * x[i]", "static_cast<double>(alpha) * x[i]")
        return request["native_source"]

    client = Client(proposal)
    phases = []

    def validate(path, phase, attempt):
        phases.append((phase, attempt))
        return {"passed": True, "gpu_verified": True}

    result = EngineeringAgent(client, "skills").run(SOURCE, tmp_path / "output", validator=validate)
    assert result["attempts"][0]["status"] == "rejected_static_checks"
    assert ("optimize", 1) not in phases and ("optimize", 2) in phases
    assert "precision" in client.requests[1]["feedback"]
    assert result["status"] == "verified_correctness"
    assert result["performance_verified"] is False


def test_failed_gpu_validation_supplies_diagnostics_to_repair(tmp_path):
    client = Client()

    def validate(path, phase, attempt):
        if phase == "translate" and attempt == 1:
            return {"passed": False, "gpu_verified": False, "diagnostics": "hipcc fixture compile failure"}
        return {"passed": True, "gpu_verified": True}

    result = EngineeringAgent(client, "skills").run(SOURCE, tmp_path / "output", validator=validate)
    failures = [item for item in result["attempts"] if item["status"] == "rejected_gpu_checks"]
    assert len(failures) == 1
    assert "hipcc fixture compile failure" in client.requests[-1]["feedback"]
    assert result["status"] == "verified_correctness"
    assert result["gpu_verified"] is True and result["performance_verified"] is False


def test_passed_flag_without_actual_gpu_verification_never_succeeds(tmp_path):
    def validate(path, phase, attempt):
        return {"passed": True, "gpu_verified": phase == "baseline"}

    result = EngineeringAgent(Client(), "skills", max_attempts=2).run(
        SOURCE, tmp_path / "output", validator=validate)
    assert result["status"] == "blocked" and not result["gpu_verified"]
    assert len(result["attempts"]) == 2
    assert "output" not in result


def test_unchanged_cuda_proposal_is_rejected_for_hip_translation(tmp_path):
    original = get_native_source(SOURCE)
    client = Client(lambda request, count: original)
    result = EngineeringAgent(client, "skills", max_attempts=1).run(SOURCE, tmp_path / "output")
    assert result["status"] == "blocked"
    assert result["attempts"][-1]["phase"] == "translate"
    assert result["attempts"][-1]["status"] == "rejected_static_checks"


@pytest.mark.parametrize("proposal", [
    {"native_source": "code", "reason": "r", "lesson": "", "command": "forbidden"},
    {"native_source": "code", "reason": 1, "lesson": ""},
    {"native_source": "", "reason": "r", "lesson": ""},
])
def test_malformed_proposal_fails_before_validator(tmp_path, proposal):
    client = Client(lambda request, count: proposal)
    calls = []

    def validate(path, phase, attempt):
        calls.append(phase)
        return {"passed": True, "gpu_verified": True}

    result = EngineeringAgent(client, "skills").run(SOURCE, tmp_path / "output", validator=validate)
    assert result["status"] == "failed"
    assert result["error_type"] == "ModelError"
    assert calls == ["baseline"]


def test_provider_failure_is_recorded_and_generic_validator_error_is_not_echoed(tmp_path):
    def fail(request, count):
        raise ModelError("provider request limit reached")

    result = EngineeringAgent(Client(fail), "skills").run(SOURCE, tmp_path / "provider")
    assert result["status"] == "failed" and "request limit" in result["error"]

    def validator(path, phase, attempt):
        raise RuntimeError("secret-provider-body")

    result = EngineeringAgent(Client(), "skills").run(SOURCE, tmp_path / "validator", validator=validator)
    assert result["status"] == "failed"
    assert "secret-provider-body" not in json.dumps(result)
