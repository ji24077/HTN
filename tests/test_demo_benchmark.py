"""Trusted benchmark arithmetic/extraction checks without GPU operations."""

import importlib.util
from pathlib import Path
from types import SimpleNamespace
from unittest.mock import Mock

import pytest

SCRIPT = Path(__file__).resolve().parents[1] / "demo/benchmark.py"
SPEC = importlib.util.spec_from_file_location("trusted_native_benchmark", SCRIPT)
benchmark = importlib.util.module_from_spec(SPEC)
SPEC.loader.exec_module(benchmark)


def test_consistent_paired_gain_requires_correctness_and_enough_samples():
    result = benchmark.summarize_pairs([1.0] * 21, [0.8] * 21, correctness_verified=True)
    assert result["performance_verified"]
    assert result["paired_median_improvement_fraction"] == pytest.approx(0.2)
    assert result["candidate_win_fraction"] == 1
    assert not benchmark.summarize_pairs([1.0] * 20, [0.8] * 20, correctness_verified=True)["performance_verified"]
    assert not benchmark.summarize_pairs([1.0] * 21, [0.8] * 21, correctness_verified=False)["performance_verified"]


def test_unstable_median_gain_or_tiny_gain_does_not_pass():
    unstable = benchmark.summarize_pairs([1.0] * 21, [0.8] * 12 + [1.1] * 9, correctness_verified=True)
    assert unstable["paired_median_improvement_fraction"] > 0.05
    assert not unstable["performance_verified"]
    small = benchmark.summarize_pairs([1.0] * 21, [0.99] * 21, correctness_verified=True)
    assert small["candidate_faster_rounds"] == 21 and not small["performance_verified"]
    threshold = benchmark.summarize_pairs([1.0] * 21, [0.95] * 21, correctness_verified=True)
    assert not threshold["performance_verified"]
    tied = benchmark.summarize_pairs([1.0] * 21, [1.0] * 21, correctness_verified=True)
    assert tied["candidate_faster_rounds"] == 0 and not tied["performance_verified"]


def test_pairs_are_compared_directly_instead_of_ratio_of_unpaired_medians():
    baseline = [1.0, 10.0, 100.0] * 7
    candidate = [0.5, 9.0, 99.0] * 7
    result = benchmark.summarize_pairs(baseline, candidate, correctness_verified=True)
    assert result["paired_improvement_fractions"][:3] == pytest.approx([0.5, 0.1, 0.01])
    assert result["paired_median_improvement_fraction"] == pytest.approx(0.1)


@pytest.mark.parametrize("baseline,candidate", [
    ([], []), ([1], []), ([0], [1]), ([-1], [1]), ([True], [1]),
    ([float("nan")], [1]), ([1], [float("inf")]), (["1"], [1]),
])
def test_invalid_samples_are_rejected(baseline, candidate):
    with pytest.raises(ValueError):
        benchmark.summarize_pairs(baseline, candidate, correctness_verified=True)


def test_native_extraction_does_not_import_python_or_trust_its_oracle(tmp_path):
    script = tmp_path / "candidate.py"
    script.write_text('raise RuntimeError("never execute this")\nHIP_SOURCE = "hipFree(x);"\ndef cpu_reference(*args): return 0\n', encoding="utf-8")
    assert benchmark.extract_native(script) == "hipFree(x);"


@pytest.mark.parametrize("source", [
    'CUDA_SOURCE = open("kernel.cu").read()',
    'CUDA_SOURCE = HIP_SOURCE = "kernel"',
    'CUDA_SOURCE = "first"\nCUDA_SOURCE = "second"',
    'CUDA_SOURCE = f"{kernel}"',
    'def kernel():\n    CUDA_SOURCE = "hidden"',
])
def test_ambiguous_or_dynamic_native_input_is_rejected(tmp_path, source):
    script = tmp_path / "candidate.py"
    script.write_text(source, encoding="utf-8")
    with pytest.raises(ValueError):
        benchmark.extract_native(script)


def test_original_and_candidate_native_libraries_compile_to_distinct_paths(tmp_path, monkeypatch):
    calls = []

    def fake_run(command, **kwargs):
        calls.append(command)
        return SimpleNamespace(returncode=0, stdout="compiled", stderr="")

    monkeypatch.setattr(benchmark.shutil, "which", lambda name: "/opt/gpu/bin/" + name)
    monkeypatch.setattr(benchmark.subprocess, "run", fake_run)
    monkeypatch.setattr(benchmark.ctypes, "CDLL", lambda _: SimpleNamespace(portable_transform=Mock(), portable_last_error=Mock()))
    first = benchmark.NativeLibrary("first native", tmp_path / "baseline", hip=True)
    second = benchmark.NativeLibrary("second native", tmp_path / "candidate", hip=True)
    assert first.compile_log["returncode"] == second.compile_log["returncode"] == 0
    assert calls[0][-1] != calls[1][-1]
    assert calls[0][0] == calls[1][0] == "/opt/gpu/bin/hipcc"
    assert (tmp_path / "baseline/kernel.hip").read_text(encoding="utf-8") == "first native"
    assert (tmp_path / "candidate/kernel.hip").read_text(encoding="utf-8") == "second native"


def test_linux_gpu_host_required_without_trying_local_gpu_operations(monkeypatch, tmp_path):
    monkeypatch.setattr(benchmark.sys, "platform", "win32")
    with pytest.raises(RuntimeError, match="Linux GPU"):
        benchmark.benchmark(tmp_path / "baseline.py", tmp_path / "candidate.py")
    with pytest.raises(ValueError, match="21 through 101"):
        benchmark.benchmark(tmp_path / "baseline.py", tmp_path / "candidate.py", rounds=2)
