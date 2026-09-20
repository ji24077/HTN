"""Offline rewriting checks; no GPU compiler or execution claims."""

import ast
from pathlib import Path

import pytest

from gpushare.portability.kernels import convert_kernel


def blockers(result):
    return [item for item in result["findings"] if item["severity"] == "blocker"]


def test_runtime_types_headers_and_errors_round_trip():
    source = '''#include <cuda_runtime.h>
#include "cuda_runtime_api.h"
cudaDeviceProp properties;
cudaError_t error = cudaErrorMemoryAllocation;
cudaStream_t stream;
cudaMalloc(&memory, 16);
cudaMemcpyAsync(dst, src, 16, cudaMemcpyDeviceToDevice, stream);
cudaDeviceSynchronize();
'''
    hip = convert_kernel(source, "cuda", "hip")
    assert not blockers(hip)
    assert "hip/hip_runtime.h" in hip["source"]
    assert '"hip/hip_runtime_api.h"' in hip["source"]
    assert "hipDeviceProp_t properties" in hip["source"]
    assert "hipErrorOutOfMemory" in hip["source"]
    assert "hipMemcpyAsync" in hip["source"]
    cuda = convert_kernel(hip["source"], "hip", "cuda")
    assert not blockers(cuda)
    assert cuda["source"] == source
    assert all(item["line"] > 0 for item in hip["mappings"])


def test_comments_literals_and_long_identifier_substrings_are_untouched():
    source = r'''// cudaMalloc and cuDeviceGet are documentation.
/* #include <cuda_runtime.h> hipLaunchKernelGGL(k,1,1,0,0); */
const char* a = "cudaMalloc \\\" cudaMemcpy";
const char* b = R"tag(cudaFree("quoted"))tag";
const auto c = u8R"(cudaStream_t)";
const auto d = L"cudaMalloc";
char slash = '\\';
int my_cudaMalloc_wrapper = 0;
cudaFree(memory);
'''
    result = convert_kernel(source, "cuda", "hip")
    assert not blockers(result)
    assert result["source"] == source.replace("cudaFree(memory);", "hipFree(memory);")
    assert len(result["mappings"]) == 1


def test_line_comment_with_preprocessor_continuation_stays_a_comment():
    source = "// comment \\\ncudaMalloc(x, 2);\ncudaFree(x);\n"
    result = convert_kernel(source, "cuda", "hip")
    assert not blockers(result)
    assert result["source"] == source.replace("cudaFree", "hipFree")


@pytest.mark.parametrize("source,code", [
    ("cudaMadeUpCall();", "unsupported_gpu_identifier"),
    ("cuDeviceGet(&device, 0);", "unsupported_gpu_identifier"),
    ("cublasGemmEx(handle);", "unsupported_gpu_identifier"),
    ("__shfl_sync(0xffffffff, x, 0);", "architecture_specific"),
    ("unsigned n = warpSize;", "architecture_specific"),
    ('asm("mov.u32 %0, %laneid;" : "=r"(lane));', "inline_assembly"),
    ("#include <cub/block/block_reduce.cuh>", "unsupported_gpu_header"),
    ("#include GPU_RUNTIME_HEADER", "dynamic_include"),
    ("#include <cuda_runtime.h\n", "malformed_include"),
    ("/* unfinished cudaMalloc", "invalid_lexical_input"),
    ('char* x = "unfinished', "invalid_lexical_input"),
    ('char* x = "physical\nnewline";', "invalid_lexical_input"),
    ('char* x = R"tag(unfinished', "invalid_lexical_input"),
    ("cudaMal\\\nloc(&x, 16);", "preprocessor_line_splice"),
    ("#define API(x) cuda ## x", "preprocessor_token_paste"),
])
def test_unsupported_or_ambiguous_native_code_is_explicitly_blocked(source, code):
    result = convert_kernel(source, "cuda", "hip")
    assert code in {finding["code"] for finding in blockers(result)}


def test_unknown_api_remains_visible_even_when_other_parts_convert():
    result = convert_kernel("cudaMalloc(&x, 4); cudaFutureAPI(x);", "cuda", "hip")
    assert result["changed"]
    assert "hipMalloc" in result["source"] and "cudaFutureAPI" in result["source"]
    assert {item["code"] for item in blockers(result)} >= {
        "unsupported_gpu_identifier", "unresolved_source_identifier",
    }


def test_cuda_triple_chevron_launch_is_retained_for_hip_clang():
    source = "#include <cuda_runtime.h>\n__global__ void k(float* x) { x[threadIdx.x] = 1; }\nk<<<g, b, 0, stream>>>(x);"
    result = convert_kernel(source, "cuda", "hip")
    assert not blockers(result)
    assert "k<<<g, b, 0, stream>>>(x)" in result["source"]


def test_reverse_launch_parses_nested_arguments_templates_and_literals():
    source = '''#include <hip/hip_runtime.h>
hipLaunchKernelGGL(HIP_KERNEL_NAME(ns::kernel<float, 2>),
    dim3(div_up(n, 128), 1, 1), dim3(128), 0, stream,
    static_cast<float*>(memory), pair[fn(1, 2)], "hipFree,a,b");'''
    result = convert_kernel(source, "hip", "cuda")
    assert not blockers(result)
    assert 'ns::kernel<float, 2><<<dim3(div_up(n, 128), 1, 1), dim3(128), 0, stream>>>' in result["source"]
    assert '(static_cast<float*>(memory), pair[fn(1, 2)], "hipFree,a,b")' in result["source"]


def test_reverse_launch_maps_runtime_arguments_and_supports_zero_kernel_arguments():
    source = "hipLaunchKernelGGL(k, dim3(1), dim3(64), 0, 0); hipLaunchKernelGGL(k, 1, 64, 0, 0, hipSuccess);"
    result = convert_kernel(source, "hip", "cuda")
    assert not blockers(result)
    assert result["source"] == "k<<<dim3(1), dim3(64), 0, 0>>>(); k<<<1, 64, 0, 0>>>(cudaSuccess);"


def test_launch_comments_cannot_swallow_reinserted_delimiters():
    source = "hipLaunchKernelGGL(k // kernel\n, 1, 32, 0, 0, x // input\n);"
    result = convert_kernel(source, "hip", "cuda")
    assert not blockers(result)
    assert "k // kernel\n<<<" in result["source"]
    assert "x // input\n)" in result["source"]


@pytest.mark.parametrize("source", [
    "hipLaunchKernelGGL(k, 1, 32, 0);",
    "hipLaunchKernelGGL(k, 1, 32, 0, 0;",
    "hipLaunchKernelGGL(k, dim3(1], 32, 0, 0);",
    "hipLaunchKernelGGL(k, , 32, 0, 0);",
    "hipLaunchKernelGGL(get_kernel(), 1, 32, 0, 0);",
    "hipLaunchKernelGGL(HIP_KERNEL_NAME(k<float>(evil)<int>), 1, 32, 0, 0);",
])
def test_invalid_or_dynamic_hip_launches_are_blocked(source):
    assert blockers(convert_kernel(source, "hip", "cuda"))


@pytest.mark.parametrize("argument", ["hipFutureAPI(x)", "__shfl(x, 0)", '([] { asm("x"); return 1; })()'])
def test_rewritten_launch_arguments_are_still_audited(argument):
    result = convert_kernel(f"hipLaunchKernelGGL(k, 1, 32, 0, 0, {argument});", "hip", "cuda")
    assert blockers(result)


def test_actual_demo_native_kernel_round_trip_is_exact():
    source = (Path(__file__).resolve().parents[1] / "demo/train.py").read_text(encoding="utf-8")
    kernel = next(node.value.value for node in ast.parse(source).body if isinstance(node, ast.Assign)
                  and any(isinstance(target, ast.Name) and target.id == "CUDA_SOURCE" for target in node.targets))
    hip = convert_kernel(kernel, "cuda", "hip")
    cuda = convert_kernel(hip["source"], "hip", "cuda")
    assert not blockers(hip) and not blockers(cuda)
    assert {item["from"] for item in hip["mappings"]} >= {
        "cuda_runtime.h", "cudaMalloc", "cudaMemcpy", "cudaFree", "cudaDeviceSynchronize",
    }
    assert cuda["source"] == kernel


def test_invalid_backend_raises_and_same_backend_does_not_claim_conversion():
    with pytest.raises(ValueError):
        convert_kernel("", "nvidia", "hip")
    result = convert_kernel("cudaMalloc(&x, 4);", "cuda", "cuda")
    assert not result["changed"] and not result["mappings"]
    assert not blockers(result)


@pytest.mark.parametrize("source,backend", [
    ("cudaMalloc(&x, 4);", "hip"),
    ("hipMalloc(&x, 4);", "cuda"),
    ("hipUnknownAPI();", "hip"),
    ("cudaUnknownAPI();", "cuda"),
    ('#include <cuda_runtime.h>\n', "hip"),
    ('#include <hip/hip_runtime.h>\n', "cuda"),
])
def test_same_backend_audit_rejects_wrong_backend_and_unknown_apis(source, backend):
    result = convert_kernel(source, backend, backend)
    assert not result["changed"] and not result["mappings"]
    assert blockers(result)


def test_same_hip_backend_audits_macro_without_rewriting_it():
    source = "hipLaunchKernelGGL(HIP_KERNEL_NAME(k<float, 2>), 1, 32, 0, 0, x);"
    result = convert_kernel(source, "hip", "hip")
    assert result["source"] == source and not result["changed"]
    assert not blockers(result)


@pytest.mark.parametrize("code", [
    'system("touch result.json");',
    'popen("echo fake-result", "r");',
    'std::ofstream output("result.json");',
    'std::filesystem::remove("expected.json");',
    'dlopen("replacement.so", 1);',
    'socket(1, 2, 3);',
    'getenv("API_KEY");',
    '#include <unistd.h>',
    '#include <sys/socket.h>',
    '#include "/etc/passwd"',
    'hipLaunchKernelGGL(k, 1, 32, 0, 0, system("echo 1"));',
])
def test_native_host_side_effects_are_blocked_even_in_same_backend_audit(code):
    assert blockers(convert_kernel(code, "hip", "hip"))


def test_host_effect_names_inside_comments_and_literals_are_not_code():
    result = convert_kernel('// system("bad");\nconst char* s = "fopen syscall dlopen";', "cuda", "cuda")
    assert not blockers(result)
