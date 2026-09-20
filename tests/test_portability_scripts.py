"""Static extraction/edit guard regressions; user code is never executed here."""

import ast
from pathlib import Path

import pytest

from gpushare.portability.scripts import (
    get_native_source,
    replace_native_source,
    translate_script,
    validate_demo_edit,
)

DEMO = (Path(__file__).resolve().parents[1] / "demo/train.py").read_text(encoding="utf-8")


def blocker_codes(result):
    findings = result["findings"] if isinstance(result, dict) else result
    return {item["code"] for item in findings if item["severity"] == "blocker"}


def kernel_value(source):
    return next(node.value.value for node in ast.parse(source).body
                if isinstance(node, ast.Assign) and any(isinstance(t, ast.Name)
                and t.id in {"CUDA_SOURCE", "HIP_SOURCE"} for t in node.targets))


def replace_kernel(source, kernel):
    tree = ast.parse(source)
    node = next(node.value for node in tree.body if isinstance(node, ast.Assign)
                and any(isinstance(t, ast.Name) and t.id == "CUDA_SOURCE" for t in node.targets))
    lines = source.splitlines(keepends=True)
    start = len("".join(lines[:node.lineno - 1])) + node.col_offset
    end = len("".join(lines[:node.end_lineno - 1])) + node.end_col_offset
    return source[:start] + repr(kernel) + source[end:]


def test_real_demo_translates_native_code_but_preserves_python_ast_and_round_trips():
    hip = translate_script(DEMO, "amd")
    assert hip["changed"] and not blocker_codes(hip)
    assert (hip["source_backend"], hip["target_backend"]) == ("cuda", "hip")
    assert "hipMalloc" in kernel_value(hip["source"])
    assert "torch.cuda.mem_get_info()" in hip["source"]
    assert "CUDA_SOURCE = " in hip["source"]
    assert not blocker_codes(validate_demo_edit(DEMO, hip["source"], "translation"))
    cuda = translate_script(hip["source"], "nvidia")
    assert not blocker_codes(cuda)
    assert cuda["source_backend"] == "hip"
    assert kernel_value(cuda["source"]) == kernel_value(DEMO)
    assert ast.dump(ast.parse(cuda["source"])) == ast.dump(ast.parse(DEMO))


def test_only_literal_span_changes_and_non_ascii_ast_offsets_work():
    prefix = '# Header\nlabel = "é 🐈"; CUDA_SOURCE: str = '
    suffix = '  # keep this\nprint("CUDA_SOURCE cudaMalloc torch.cuda")\n'
    original = prefix + repr("cudaMalloc(&x, 8);") + suffix
    translated = translate_script(original, "amd")
    assert translated["source"].startswith(prefix)
    assert translated["source"].endswith(suffix)
    assert "hipMalloc(&x, 8)" in translated["source"]


@pytest.mark.parametrize("literal", [
    repr("cudaFree(x);\\"),
    repr('cudaFree(x); const char* q = "\\\"";'),
    '("cudaFree" "(x);")',
    'r"""cudaFree(x);\n"""',
])
def test_literal_rendering_preserves_exact_converted_native_value(literal):
    original = "CUDA_SOURCE = " + literal + "\n"
    result = translate_script(original, "amd")
    assert kernel_value(result["source"]) == kernel_value(original).replace("cudaFree", "hipFree")


@pytest.mark.parametrize("source,code", [
    ('CUDA_SOURCE = f"cudaFree({memory});"', "dynamic_native_source"),
    ('CUDA_SOURCE = open("kernel.cu").read()', "dynamic_native_source"),
    ('CUDA_SOURCE = "cudaFree(x);"\nCUDA_SOURCE += "junk"', "ambiguous_native_source"),
    ('CUDA_SOURCE = "cudaFree(x);"\ndel CUDA_SOURCE', "ambiguous_native_source"),
    ('CUDA_SOURCE = HIP_SOURCE = "cudaFree(x);"', "ambiguous_native_source"),
    ('CUDA_SOURCE = "cudaFree(x);"\nHIP_SOURCE = "hipFree(x);"', "ambiguous_native_source"),
    ('def code():\n    CUDA_SOURCE = "cudaFree(x);"', "dynamic_native_source"),
    ('CUDA_SOURCE =', "invalid_python"),
    ('CUDA_SOURCE = b"cudaFree(x);"', "dynamic_native_source"),
    ('CUDA_SOURCE = "cudaFree(x); hipFree(y);"', "mixed_native_backends"),
])
def test_unsupported_script_forms_stay_unchanged_with_blockers(source, code):
    result = translate_script(source, "amd")
    assert result["source"] == source and not result["changed"]
    assert code in blocker_codes(result)


def test_plain_pytorch_does_not_claim_translation_or_replace_cuda_interface():
    source = 'import torch\nx = torch.ones(2, device="cuda")\nprint(torch.cuda.get_device_name())\n'
    result = translate_script(source, "amd")
    assert result["source"] == source and not result["changed"] and not result["mappings"]
    assert result["source_backend"] is None
    assert any(item["code"] == "no_embedded_native_source" for item in result["findings"])


def test_already_target_native_source_uses_contents_not_assignment_name():
    source = 'CUDA_SOURCE = "#include <hip/hip_runtime.h>\\nhipFree(x);"\n'
    result = translate_script(source, "amd")
    assert result["source_backend"] == "hip" and not result["changed"]
    assert any(item["code"] == "already_target_backend" for item in result["findings"])


def test_no_runtime_specific_mapping_does_not_claim_translation():
    result = translate_script('CUDA_SOURCE = "__global__ void k() {}"', "amd")
    assert not result["changed"] and not result["mappings"]
    assert any(item["code"] == "no_native_edits" for item in result["findings"])


def test_already_target_code_is_still_checked_for_unknown_api():
    result = translate_script('HIP_SOURCE = "hipFutureAPI(x);"', "amd")
    assert not result["changed"] and "unsupported_gpu_identifier" in blocker_codes(result)


def test_native_unknown_api_returns_explicit_partial_draft():
    result = translate_script('CUDA_SOURCE = "cudaMalloc(&x, 8); cudaMystery(x);"', "amd")
    assert result["changed"] and blocker_codes(result)
    assert "hipMalloc" in result["source"] and "cudaMystery" in result["source"]


@pytest.mark.parametrize("before,after", [
    ("ALPHA = 0.375", "ALPHA = 0.25"),
    ("torch.float32", "torch.float16"),
    ("atol=1e-6", "atol=100"),
    ('["-O2", "-std=c++17", "-shared"]', '["-O2", "-std=c++17", "-shared", "--use_fast_math"]'),
    ('"batch_size": 257', '"batch_size": 1'),
    ('source.write_text(CUDA_SOURCE', 'source.write_text(""'),
    ("return ALPHA * x + y", "return torch.zeros_like(x)"),
])
def test_demo_guard_rejects_oracle_precision_input_training_and_build_edits(before, after):
    assert before in DEMO
    candidate = DEMO.replace(before, after)
    assert "protected_python_changed" in blocker_codes(validate_demo_edit(DEMO, candidate, "optimization"))


def test_demo_guard_accepts_native_only_optimization_but_requires_runtime_validation():
    candidate = replace_kernel(DEMO, kernel_value(DEMO).replace("(n + 255) / 256", "(n + 127) / 128").replace("<<<blocks, 256>>>", "<<<blocks, 128>>>"))
    findings = validate_demo_edit(DEMO, candidate, "optimization")
    assert not blocker_codes(findings)
    assert any(item["code"] == "runtime_validation_required" for item in findings)


@pytest.mark.parametrize("change,code", [
    (lambda kernel: kernel.replace("__global__", ""), "native_gpu_contract_removed"),
    (lambda kernel: kernel.replace("axpy_kernel<<<blocks, 256>>>", "axpy_kernel"), "native_gpu_launch_removed"),
    (lambda kernel: kernel.replace("cudaMemcpy", "host_copy"), "native_device_transfer_removed"),
    (lambda kernel: kernel.replace("float", "__half"), "native_precision_changed"),
    (lambda kernel: kernel.replace("portable_transform", "cpu_transform"), "native_gpu_contract_removed"),
])
def test_demo_guard_rejects_removed_gpu_work_and_precision_changes(change, code):
    findings = validate_demo_edit(DEMO, replace_kernel(DEMO, change(kernel_value(DEMO))), "translation")
    assert code in blocker_codes(findings)


def test_demo_guard_rejects_missing_native_source_and_syntax_failure():
    assert "missing_demo_native_source" in blocker_codes(validate_demo_edit(DEMO, "print('hello')", "repair"))
    assert "invalid_python" in blocker_codes(validate_demo_edit(DEMO, "def", "repair"))


def test_demo_guard_allows_python_comments_but_does_not_execute_source():
    candidate = "# harmless comment\n" + DEMO
    assert not blocker_codes(validate_demo_edit(DEMO, candidate, "repair"))
    source = 'raise RuntimeError("must not execute")\nCUDA_SOURCE = "cudaFree(x);"'
    assert translate_script(source, "amd")["changed"]


def test_invalid_target_is_rejected():
    with pytest.raises(ValueError, match="target"):
        translate_script(DEMO, "cpu")


def test_public_native_helpers_handle_unicode_and_reject_ambiguous_input():
    source = 'label = "é 🐈"; CUDA_SOURCE = "cudaFree(x);" # preserved\n'
    candidate = replace_native_source(source, "hipFree(x);\\")
    assert get_native_source(candidate) == "hipFree(x);\\"
    assert candidate.startswith('label = "é 🐈"; CUDA_SOURCE = ')
    assert candidate.endswith(" # preserved\n")
    for operation in (get_native_source, lambda value: replace_native_source(value, "new")):
        with pytest.raises(ValueError):
            operation('CUDA_SOURCE = open("kernel.cu").read()')


@pytest.mark.parametrize("declaration", [
    "static float* device_cache = nullptr;",
    "thread_local float* device_cache = nullptr;",
    "float* device_cache = nullptr;",
    "namespace { float* device_cache = nullptr; }",
    "struct DeviceScratch { float* storage = nullptr; }; DeviceScratch cache;",
    "struct DeviceScratch { float* storage = nullptr; }; DeviceScratch cache{};",
    "struct DeviceScratch { float* storage = nullptr; }; DeviceScratch& scratch() { static DeviceScratch cache; return cache; }",
    "struct DeviceScratch { float* storage = nullptr; }; DeviceScratch& scratch() { thread_local DeviceScratch cache{}; return cache; }",
    "void scratch() { static struct { float* storage; } cache; }",
])
def test_stateless_demo_rejects_added_persistent_device_pointers_and_objects(declaration):
    native = get_native_source(DEMO)
    candidate = replace_native_source(DEMO, native + "\n" + declaration + "\n")
    findings = validate_demo_edit(DEMO, candidate, "optimization")
    assert "persistent_native_storage" in blocker_codes(findings)
    assert any("no destroy hook" in item["message"] for item in findings if item["code"] == "persistent_native_storage")


def test_persistent_storage_guard_accepts_local_buffers_and_static_helper_functions():
    native = get_native_source(DEMO) + "\nstatic int helper() { float* local = nullptr; return local == nullptr; }\n"
    candidate = replace_native_source(DEMO, native)
    assert "persistent_native_storage" not in blocker_codes(validate_demo_edit(DEMO, candidate, "optimization"))


def test_persistent_storage_guard_ignores_comments_and_literals():
    native = get_native_source(DEMO) + '\n// static DeviceScratch cache;\nstatic const char* description() { return "static float* cached;"; }\n'
    candidate = replace_native_source(DEMO, native)
    assert "persistent_native_storage" not in blocker_codes(validate_demo_edit(DEMO, candidate, "optimization"))


def test_persistent_storage_guard_accepts_independent_hip_fixture_and_translation():
    hip = (Path(__file__).resolve().parents[1] / "demo/train_hip.py").read_text(encoding="utf-8")
    converted = translate_script(hip, "nvidia")["source"]
    assert not blocker_codes(validate_demo_edit(hip, hip, "optimization"))
    assert not blocker_codes(validate_demo_edit(hip, converted, "translation"))
