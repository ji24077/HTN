"""Remote validation ordering and proof checks using fake SSH/SCP, never a GPU."""

import hashlib
import json
import os
import shlex
import shutil
import subprocess
import sys
from pathlib import Path
from types import SimpleNamespace

import pytest

from gpushare.portability.validation import (
    _PROBE,
    _REMOTE_ENV,
    RemoteDemoValidator,
    _kernel_digest,
    _transport_environment,
)

DEMO = Path(__file__).resolve().parents[1] / "demo" / "train.py"


def test_transport_environment_preserves_platform_settings_without_credentials(monkeypatch):
    monkeypatch.setenv("PROGRAMDATA", "platform-data")
    for key in ("OPENAI_API_KEY", "BASETEN_TOKEN", "PASSWORD", "AWS_SECRET_ACCESS_KEY", "SSH_AUTH_SOCK"):
        monkeypatch.setenv(key, "must-not-reach-child")
    environment = _transport_environment()
    assert environment["PROGRAMDATA"] == "platform-data"
    assert "must-not-reach-child" not in environment.values()


@pytest.mark.skipif(os.name != "nt" or shutil.which("ssh") is None, reason="requires Windows OpenSSH")
def test_native_windows_openssh_starts_with_filtered_environment():
    # -V exits locally: no endpoint, credentials, or network connection involved.
    result = subprocess.run(["ssh", "-V"], env=_transport_environment(),
                            capture_output=True, text=True, timeout=10)
    assert result.returncode == 0, result.stdout + result.stderr
    assert "OpenSSH" in result.stdout + result.stderr


@pytest.fixture
def remote(tmp_path, monkeypatch):
    private = tmp_path / "key"
    private.write_text("fake-test-key")
    source = {"ip": "127.0.0.1", "port": 2222, "key": str(private)}
    target = {"ip": "127.0.0.2", "port": 2223, "key": str(private)}
    state = {"wrong_backend": False, "version_mismatch": False, "timeout": False,
             "bad_report": False, "numerical_mismatch": False, "run_exit": 0,
             "source_vendor": "nvidia", "target_vendor": "amd"}
    calls, files = [], {}
    monkeypatch.setenv("GPUSHARE_RUNPOD_API_KEY", "must-never-reach-child")

    def execute(argv, **kwargs):
        assert kwargs["timeout"] <= 300
        assert "GPUSHARE_RUNPOD_API_KEY" not in kwargs["env"]
        assert "none" == argv[argv.index("-F") + 1]
        calls.append(argv)
        stdout, stderr, code = "", "", 0
        if argv[0] == "scp":
            src, dst = argv[-2:]
            if src.startswith("root@"):
                if src not in files:
                    return SimpleNamespace(stdout="", stderr="missing remote file", returncode=1)
                Path(dst).write_bytes(files[src])
            else:
                files[dst] = Path(src).read_bytes()
        else:
            host, command = argv[-2:]
            assert command.startswith(_REMOTE_ENV)
            tokens = shlex.split(command.removeprefix(_REMOTE_ENV))
            if "-c" in tokens:
                expected = tokens[-1]
                actual = "nvidia" if state["wrong_backend"] and expected == "amd" else expected
                base = "2.11.0" if state["version_mismatch"] and expected == "amd" else "2.10.0"
                stdout = json.dumps({"passed": True, "vendor": actual, "compiler": "/usr/bin/compiler",
                                     "visible_device_count": state.get("visible_device_count", 1),
                                     "python_executable": "/opt/venv/bin/python3",
                                     "torch_file": "/opt/venv/lib/python3.12/site-packages/torch/__init__.py",
                                     "free_memory_bytes": 10**10, "total_memory_bytes": 20**10,
                                     "peak_allocated_bytes": 4096, "gpu": actual + " GPU",
                                     "torch_version": base + ("+rocm7.1" if actual == "amd" else "+cu128")})
            elif tokens[0] == "cd":
                directory = tokens[1]
                vendor = tokens[tokens.index("--expect-vendor") + 1]
                steps = int(tokens[tokens.index("--steps") + 1])
                resume = "--resume" in tokens
                start = state.get("baseline_steps", 2) if resume else 0
                script = files[f"{host}:{directory}/train.py"].decode()
                report = {
                    "vendor": vendor, "gpu": vendor + " GPU",
                    "torch_version": "2.10.0" + ("+rocm7.1" if vendor == "amd" else "+cu128"),
                    "operation": "axpy", "precision": "float32", "parameters_on_gpu": True,
                    "training_config": {"seed": 1, "learning_rate": 0.005, "batch_size": 257},
                    "global_step": steps, "start_step": start, "steps_executed": steps - start,
                    "input_sha256": "a" * 64, "predictions": [0.1, 0.2],
                    "loss_history": [0.1] * (steps - start),
                    "initial_predictions": [0.1, 0.2], "optimizer_restored": resume,
                    "optimizer_state_entries": 4 if resume else 0,
                    "resume_prediction_max_abs_error": 0.0 if resume else None,
                    "native_kernel_source_sha256": _kernel_digest(script),
                    "peak_allocated_bytes": 4096, "peak_reserved_bytes": 8192,
                    "memory_scope": "PyTorch training allocator", "timing_scope": "training",
                    "training_seconds": 1.0, "custom_kernel_max_abs_error": 0.0,
                    "custom_kernel_shape_checks": [{"elements": 0, "max_abs_error": 0.0},
                                                   {"elements": 257, "max_abs_error": 0.0}],
                    "custom_kernel_calls": 3,
                }
                if state["bad_report"]:
                    report["gpu"] = "fabricated GPU"
                if state["numerical_mismatch"]:
                    report["predictions"] = [1.0, 2.0]
                if resume and vendor == state["target_vendor"]:
                    report.update(state.get("resume_changes", {}))
                files[f"{host}:{directory}/report.json"] = json.dumps(report).encode()
                files[f"{host}:{directory}/checkpoint.pt"] = b"fake checkpoint fixture"
                if state["timeout"]:
                    raise subprocess.TimeoutExpired(argv, 300, output=b"compiler timeout context")
                code = state["run_exit"]
                stderr = "compiler error context" if code else ""
            elif tokens[0] == "sha256sum":
                digest = hashlib.sha256(files[f"{host}:{tokens[1]}"]).hexdigest()
                stdout = ("0" * 64 if state.get("bad_transfer") else digest) + "  " + tokens[1]
        return SimpleNamespace(stdout=stdout, stderr=stderr, returncode=code)

    monkeypatch.setattr(subprocess, "run", execute)

    def make(**kwargs):
        local_dir = kwargs.pop("local_dir", tmp_path / "validation")
        baseline_script = kwargs.pop("baseline_script", DEMO)
        return RemoteDemoValidator(source, target, "/workspace/gpushare-translation/test-session",
                                   local_dir, baseline_script, steps=2, **kwargs)

    return make, state, calls, files


def test_frozen_baseline_then_cross_gpu_translation(remote):
    make, state, calls, files = remote
    validator = make()
    baseline = validator(DEMO, "baseline", 0)
    assert baseline["passed"] and baseline["gpu_verified"]
    frozen = Path(baseline["report_path"]).read_bytes()
    assert (validator.frozen_dir / "checkpoint.pt").is_file()
    assert (validator.frozen_dir / "manifest.json").is_file()
    translated = validator(DEMO, "translate", 1)
    assert translated["passed"] and translated["gpu_verified"]
    assert translated["comparison"]["timing"]["speedup"] is None
    assert Path(baseline["report_path"]).read_bytes() == frozen
    count = len(calls)
    rejected = validator(DEMO, "baseline", 2)
    assert not rejected["passed"] and "once" in rejected["error"]
    assert len(calls) == count


def test_candidate_without_baseline_never_calls_ssh(remote):
    make, _, calls, _ = remote
    result = make()(DEMO, "optimize", 0)
    assert not result["passed"] and not result["gpu_verified"]
    assert "baseline" in result["error"]
    assert calls == []


@pytest.mark.parametrize("failure", ["wrong_backend", "version_mismatch"])
def test_preflight_failure_blocks_upload_and_training(remote, failure):
    make, state, calls, _ = remote
    state[failure] = True
    result = make()(DEMO, "baseline", 0)
    assert not result["passed"] and not result["gpu_verified"]
    assert calls and all(argv[0] == "ssh" and "-c" in shlex.split(argv[-1]) for argv in calls)


def test_timeout_retains_report_and_compile_diagnostics_without_claiming_gpu_proof(remote):
    make, state, _, _ = remote
    state["timeout"] = True
    result = make()(DEMO, "baseline", 0)
    assert not result["passed"] and not result["gpu_verified"]
    assert "timed out" in result["error"]
    assert "compiler timeout context" in result["diagnostics"]
    assert Path(result["report_path"]).is_file()


def test_report_hardware_must_match_independent_preflight(remote):
    make, state, _, _ = remote
    state["bad_report"] = True
    result = make()(DEMO, "baseline", 0)
    assert not result["passed"] and not result["gpu_verified"]
    assert "does not match" in result["error"]


def test_protected_python_edits_are_rejected_before_ssh(remote, tmp_path):
    make, _, calls, _ = remote
    validator = make()
    assert validator(DEMO, "baseline", 0)["passed"]
    candidate = tmp_path / "modified.py"
    candidate.write_text(DEMO.read_text(encoding="utf-8").replace("ALPHA = 0.375", "ALPHA = 0.5"), encoding="utf-8")
    count = len(calls)
    result = validator(candidate, "optimize", 1)
    assert not result["passed"] and "edit guard" in result["error"]
    assert len(calls) == count


def test_numerical_failure_is_not_reported_as_success(remote):
    make, state, _, _ = remote
    validator = make()
    assert validator(DEMO, "baseline", 0)["passed"]
    state["numerical_mismatch"] = True
    result = validator(DEMO, "target_optimize", 1)
    assert not result["passed"] and result["gpu_verified"]
    assert result["comparison"]["status"] == "mismatch"


def test_reverse_direction_runs_baseline_on_amd_and_translation_on_nvidia(remote):
    make, _, _, _ = remote
    validator = make(source_vendor="amd", target_vendor="nvidia")
    result = validator(DEMO, "baseline", 0)
    assert result["passed"]
    assert json.loads(Path(result["report_path"]).read_text())["vendor"] == "amd"
    result = validator(DEMO, "translate", 1)
    assert result["passed"]
    assert json.loads(Path(result["report_path"]).read_text())["vendor"] == "nvidia"


def _resume_ready(make):
    validator = make()
    assert validator(DEMO, "baseline", 0)["passed"]
    assert validator(DEMO, "translate", 1)["passed"]
    return validator


def test_resume_requires_baseline_and_verified_target_before_remote_work(remote):
    make, _, calls, _ = remote
    validator = make()
    assert "baseline" in validator.verify_resume(DEMO)["error"]
    assert calls == []
    assert validator(DEMO, "baseline", 0)["passed"]
    count = len(calls)
    assert "correctness" in validator.verify_resume(DEMO)["error"]
    assert len(calls) == count


def test_resume_uses_identical_frozen_checkpoint_and_compares_continuation(remote):
    make, _, calls, files = remote
    validator = _resume_ready(make)
    frozen = (validator.frozen_dir / "checkpoint.pt").read_bytes()
    result = validator.verify_resume(DEMO)
    assert result["passed"] and result["gpu_verified"], result
    assert result["start_step"] == 2 and result["global_step"] == 6
    assert result["steps_executed"] == 4
    assert result["comparison"]["timing"]["speedup"] is None
    resume_commands = [shlex.split(argv[-1]) for argv in calls if argv[0] == "ssh" and "--resume" in argv[-1]]
    assert len(resume_commands) == 2
    assert [command[command.index("--expect-vendor") + 1] for command in resume_commands] == ["amd", "nvidia"]
    transfers = [payload for name, payload in files.items() if name.endswith("/input-checkpoint.pt")]
    assert transfers == [frozen, frozen]
    for report_path in result["reports"].values():
        directory = Path(report_path).parent
        assert (directory / "checkpoint.pt").is_file()
        assert (directory / "execution.log").is_file()
    assert (validator.frozen_dir / "checkpoint.pt").read_bytes() == frozen


@pytest.mark.parametrize("changes", [
    {"optimizer_restored": False}, {"optimizer_state_entries": 0},
    {"start_step": 0}, {"global_step": 7}, {"steps_executed": 0},
    {"input_sha256": "b" * 64}, {"native_kernel_source_sha256": "b" * 64},
    {"gpu": "wrong GPU"}, {"torch_version": "2.9.0+rocm7.1"},
    {"initial_predictions": [0.8, 0.9]}, {"resume_prediction_max_abs_error": float("nan")},
])
def test_resume_rejects_missing_or_inconsistent_continuation_evidence(remote, changes):
    make, state, _, _ = remote
    validator = _resume_ready(make)
    state["resume_changes"] = changes
    result = validator.verify_resume(DEMO)
    assert not result["passed"] and not result["gpu_verified"]
    assert Path(result["report_path"]).is_file()
    assert (Path(result["report_path"]).parent / "checkpoint.pt").is_file()


def test_resume_numerical_divergence_does_not_pass(remote):
    make, state, _, _ = remote
    validator = _resume_ready(make)
    state["resume_changes"] = {"loss_history": [0.9] * 4}
    result = validator.verify_resume(DEMO)
    assert not result["passed"] and result["gpu_verified"]
    assert result["comparison"]["status"] == "mismatch"


def test_resume_transfer_hash_failure_precedes_execution(remote):
    make, state, calls, _ = remote
    validator = _resume_ready(make)
    state["bad_transfer"] = True
    result = validator.verify_resume(DEMO)
    assert not result["passed"] and "hash mismatch" in result["error"]
    assert not any("--resume" in argv[-1] for argv in calls)


def test_resume_timeout_preserves_report_and_checkpoint(remote):
    make, state, _, _ = remote
    validator = _resume_ready(make)
    state["timeout"] = True
    result = validator.verify_resume(DEMO)
    assert not result["passed"] and "timed out" in result["error"]
    assert Path(result["report_path"]).is_file()
    assert (Path(result["report_path"]).parent / "checkpoint.pt").is_file()


def test_followup_process_can_restore_verified_artifacts_without_rerunning_baseline(remote, tmp_path):
    make, _, calls, _ = remote
    original = _resume_ready(make)
    count = len(calls)
    followup = make(local_dir=tmp_path / "followup")
    followup.restore_baseline(original.local_dir)
    assert len(calls) == count
    result = followup.verify_resume(DEMO)
    assert result["passed"], result


def test_resume_rejects_tampered_frozen_checkpoint_before_remote_work(remote):
    make, _, calls, _ = remote
    validator = _resume_ready(make)
    (validator.frozen_dir / "checkpoint.pt").write_bytes(b"tampered")
    count = len(calls)
    result = validator.verify_resume(DEMO)
    assert not result["passed"] and "hash mismatch" in result["error"]
    assert len(calls) == count


def test_ssh_always_restores_trusted_gpu_environment_before_commands(remote):
    make, _, calls, _ = remote
    validator = make()
    assert validator(DEMO, "baseline", 0)["passed"]
    commands = [argv[-1] for argv in calls if argv[0] == "ssh"]
    assert commands and all(command.startswith(_REMOTE_ENV) for command in commands)
    assert "/etc/gpushare-gpu-env" in _REMOTE_ENV
    assert "/opt/venv/bin:/opt/conda/bin:/opt/rocm/bin:/usr/local/cuda/bin:" in _REMOTE_ENV
    assert "${LD_LIBRARY_PATH:+$LD_LIBRARY_PATH:}" in _REMOTE_ENV
    assert "/opt/rocm/lib:/usr/local/cuda/lib64" in _REMOTE_ENV


@pytest.mark.parametrize("count", [0, 2, True])
def test_preflight_rejects_other_than_exactly_one_visible_gpu(remote, count):
    make, state, calls, _ = remote
    state["visible_device_count"] = count
    result = make()(DEMO, "baseline", 0)
    assert not result["passed"] and not result["gpu_verified"]
    assert "preflight failed" in result["error"]
    assert all(argv[0] == "ssh" for argv in calls)


def test_probe_reports_runtime_and_visibility_only_and_rejects_multiple_gpus(monkeypatch, capsys):
    fake_torch = SimpleNamespace(
        version=SimpleNamespace(hip="7.1", cuda=None), __version__="2.10.0+rocm7.1",
        __file__="/opt/venv/site-packages/torch/__init__.py",
        cuda=SimpleNamespace(device_count=lambda: 2),
    )
    monkeypatch.setitem(sys.modules, "torch", fake_torch)
    monkeypatch.setattr(sys, "argv", ["probe", "amd"])
    monkeypatch.setenv("HIP_VISIBLE_DEVICES", "0,1")
    monkeypatch.setenv("ROCR_VISIBLE_DEVICES", "0,1")
    monkeypatch.setenv("UNRELATED_API_KEY", "do-not-report-this")
    exec(_PROBE, {})
    rendered = capsys.readouterr().out
    report = json.loads(rendered)
    assert report["passed"] is False and "exactly one" in report["error"]
    assert report["visible_device_count"] == 2
    assert report["python_executable"] == sys.executable
    assert report["torch_file"] == fake_torch.__file__
    assert report["visibility_env"]["HIP_VISIBLE_DEVICES"] == "0,1"
    assert "do-not-report-this" not in rendered and "UNRELATED_API_KEY" not in rendered


@pytest.mark.parametrize("candidate_newline", ["\n", "\r\n"])
def test_crlf_baseline_accepts_equivalent_source_and_freezes_canonical_lf(remote, tmp_path, candidate_newline):
    make, _, _, files = remote
    source = DEMO.read_text(encoding="utf-8")
    original = tmp_path / "windows-original.py"
    original.write_bytes(source.replace("\n", "\r\n").encode("utf-8"))
    candidate = tmp_path / "candidate.py"
    candidate.write_bytes(source.replace("\n", candidate_newline).encode("utf-8"))
    validator = make(baseline_script=original)
    assert validator.baseline_bytes == source.encode("utf-8")
    assert validator.baseline_sha256 == hashlib.sha256(source.encode("utf-8")).hexdigest()
    result = validator(candidate, "baseline", 0)
    assert result["passed"], result
    assert (validator.frozen_dir / "train.py").read_bytes() == source.encode("utf-8")
    assert all(b"\r" not in payload for name, payload in files.items() if name.endswith("/train.py"))
    assert validator(candidate, "translate", 1)["passed"]
    assert validator.verify_resume(candidate)["passed"]


def test_newline_normalization_does_not_accept_changed_baseline_content(remote, tmp_path):
    make, _, calls, _ = remote
    source = DEMO.read_text(encoding="utf-8")
    original = tmp_path / "windows-original.py"
    original.write_bytes(source.replace("\n", "\r\n").encode("utf-8"))
    candidate = tmp_path / "changed.py"
    # A comment is AST-equivalent but still changes the frozen source identity.
    candidate.write_bytes((source + "\n# changed baseline content\n").encode("utf-8"))
    validator = make(baseline_script=original)
    result = validator(candidate, "baseline", 0)
    assert not result["passed"] and "original script bytes" in result["error"]
    assert calls == []
