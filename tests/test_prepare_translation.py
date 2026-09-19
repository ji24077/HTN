"""Offline CLI artifact and independent-input regressions; no GPU claims."""

import ast
import copy
import json
import os
import subprocess
import sys
from pathlib import Path

import pytest

from gpushare.portability.kernels import convert_kernel
from gpushare.portability.scripts import (
    get_native_source,
    translate_script,
    validate_demo_edit,
)

ROOT = Path(__file__).resolve().parents[1]


def run_cli(source: Path, target: str, output: Path):
    environment = {**os.environ, "PYTHONUTF8": "1"}
    return subprocess.run([sys.executable, str(ROOT / "scripts/prepare_translation.py"),
                           str(source), "--target", target, "--out", str(output)],
                          capture_output=True, text=True, encoding="utf-8", env=environment,
                          check=False, timeout=30)


@pytest.mark.parametrize("script,target,backend", [
    ("train.py", "amd", "hip"), ("train_hip.py", "nvidia", "cuda"),
])
def test_cli_produces_auditable_unverified_files_without_execution(tmp_path, script, target, backend):
    output = tmp_path / "prepared"
    process = run_cli(ROOT / "demo" / script, target, output)
    assert process.returncode == 0, process.stderr + process.stdout
    result = json.loads((output / "result.json").read_text(encoding="utf-8"))
    assert result["status"] == "prepared_unverified"
    assert result["gpu_verified"] is False and result["code_executed"] is False
    assert result["model_used"] is False and result["changed"] is True
    assert result["target_backend"] == backend and result["mappings"]
    assert result["source_sha256"] != result["translated_sha256"]
    assert (output / "original.py").read_text(encoding="utf-8") == (ROOT / "demo" / script).read_text(encoding="utf-8")
    assert not [entry for entry in result["findings"] if entry["severity"] == "blocker"]
    assert set(path.name for path in output.iterdir()) == {"original.py", "translated.py", "result.json"}
    ast.parse((output / "translated.py").read_text(encoding="utf-8"))


def test_cli_never_overwrites_an_existing_output_directory(tmp_path):
    output = tmp_path / "existing"
    output.mkdir()
    marker = output / "original.py"
    marker.write_text("original user file", encoding="utf-8")
    process = run_cli(ROOT / "demo/train.py", "amd", output)
    assert process.returncode == 2
    assert marker.read_text(encoding="utf-8") == "original user file"
    assert not (output / "result.json").exists()


def test_cli_writes_blocked_partial_draft_and_does_not_execute_input(tmp_path):
    source = tmp_path / "unknown.py"
    source.write_text('raise RuntimeError("must not run")\nCUDA_SOURCE = "cudaMalloc(&x, 8); cudaUnknown(x);"\n', encoding="utf-8")
    output = tmp_path / "blocked"
    process = run_cli(source, "amd", output)
    assert process.returncode == 1
    result = json.loads((output / "result.json").read_text(encoding="utf-8"))
    assert result["status"] == "blocked" and result["gpu_verified"] is False
    assert any(entry["severity"] == "blocker" for entry in result["findings"])
    assert "hipMalloc" in (output / "translated.py").read_text(encoding="utf-8")
    assert "Traceback" not in process.stderr


def test_cli_reports_missing_input_without_creating_output(tmp_path):
    output = tmp_path / "not-created"
    process = run_cli(tmp_path / "missing.py", "amd", output)
    assert process.returncode == 2 and not output.exists()
    assert json.loads(process.stderr)["gpu_verified"] is False


def test_cli_preserves_input_bytes_including_crlf_and_unicode(tmp_path):
    source = tmp_path / "unicode.py"
    original = '# café\r\nCUDA_SOURCE = "cudaFree(x);"\r\n'.encode()
    source.write_bytes(original)
    output = tmp_path / "unicode-result"
    process = run_cli(source, "amd", output)
    assert process.returncode == 0, process.stderr + process.stdout
    assert (output / "original.py").read_bytes() == original


def test_plain_pytorch_is_not_presented_as_custom_code_translation(tmp_path):
    source = tmp_path / "plain.py"
    source.write_text('import torch\nx = torch.ones(8, device="cuda")\n', encoding="utf-8")
    process = run_cli(source, "amd", tmp_path / "result")
    assert process.returncode == 1
    result = json.loads(process.stdout)
    assert result["status"] == "blocked" and result["changed"] is False
    assert not result["mappings"]


def normalized_training_contract(source: str):
    tree = copy.deepcopy(ast.parse(source))
    for parent in ast.walk(tree):
        if isinstance(parent, (ast.Module, ast.FunctionDef, ast.AsyncFunctionDef, ast.ClassDef)):
            if parent.body and isinstance(parent.body[0], ast.Expr) and isinstance(parent.body[0].value, ast.Constant) and isinstance(parent.body[0].value.value, str):
                parent.body = parent.body[1:]
    for node in ast.walk(tree):
        if isinstance(node, ast.Name) and node.id == "HIP_SOURCE":
            node.id = "CUDA_SOURCE"
        if isinstance(node, ast.Assign) and any(isinstance(target, ast.Name) and target.id in {"CUDA_SOURCE", "HIP_SOURCE"} for target in node.targets):
            node.value = ast.Constant(value="<independent native implementation>")
    return ast.dump(tree, include_attributes=False)


def test_hip_fixture_has_identical_training_contract_but_independent_native_code():
    cuda = (ROOT / "demo/train.py").read_text(encoding="utf-8")
    hip = (ROOT / "demo/train_hip.py").read_text(encoding="utf-8")
    assert normalized_training_contract(cuda) == normalized_training_contract(hip)
    assert get_native_source(hip) != get_native_source(translate_script(cuda, "amd")["source"])
    native = get_native_source(hip)
    assert "hipLaunchKernelGGL" in native and "gridDim.x" in native
    assert native.count("hipMalloc(") == 1
    assert not [finding for finding in convert_kernel(native, "hip", "hip")["findings"] if finding["severity"] == "blocker"]
    converted = translate_script(hip, "nvidia")
    assert converted["changed"] and "<<<" in get_native_source(converted["source"])
    assert not [finding for finding in validate_demo_edit(hip, converted["source"], "translation") if finding["severity"] == "blocker"]
