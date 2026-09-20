"""Pasteable CUDA-to-HIP translation demo: custom GPU preprocessing + PyTorch training.

No API keys, external datasets, model downloads, or repository imports are needed.
The embedded CUDA source must actually be translated before running on AMD.
Use --inspect for CPU-only input inspection; normal execution requires a Linux GPU
host, GPU-enabled PyTorch, and the matching native compiler. See README.md.
"""

from __future__ import annotations

import argparse
import ctypes
import hashlib
import json
import math
import shutil
import subprocess
import sys
import time
from pathlib import Path

import torch

OPERATION = "axpy"
SEED = 20260919
ALPHA = 0.375
SHAPES = (0, 1, 31, 127, 128, 255, 256, 257, 1025)
FLOAT_POINTER = ctypes.POINTER(ctypes.c_float)


# This is the NVIDIA-specific code the translation agent must transform.
# A different PyTorch wheel or compiler does not itself translate this source.
CUDA_SOURCE = r"""#include <cuda_runtime.h>
#include <cstddef>
#include <cstdio>

// Native CUDA input fixture: y = alpha * x + y, with a host-buffer C ABI.
static char last_error[512] = "";

static bool checked(cudaError_t result, const char* operation) {
    if (result == cudaSuccess) return true;
    std::snprintf(last_error, sizeof(last_error), "%s: %s", operation,
                  cudaGetErrorString(result));
    return false;
}

static int release_buffers(float* x, float* y, float* output, int status) {
    float* allocations[] = {x, y, output};
    for (float* allocation : allocations) {
        if (!allocation) continue;
        cudaError_t result = cudaFree(allocation);
        if (result != cudaSuccess && status == 0) {
            checked(result, "cudaFree");
            status = 1;
        }
    }
    return status;
}

__global__ void axpy_kernel(const float* x, const float* y, float* output,
                            std::size_t n, float alpha) {
    const std::size_t i = blockIdx.x * blockDim.x + threadIdx.x;
    if (i < n) output[i] = alpha * x[i] + y[i];
}

extern "C" const char* portable_last_error() { return last_error; }

extern "C" int portable_transform(const float* x, const float* y, float* output,
                                  std::size_t n, float alpha) {
    last_error[0] = '\0';
    // Empty input succeeds without accessing a device or dereferencing buffers.
    if (n == 0) return 0;
    if (!x || !y || !output || n > static_cast<std::size_t>(2147483647)) {
        std::snprintf(last_error, sizeof(last_error), "invalid host buffers or length");
        return 1;
    }
    float *device_x = nullptr, *device_y = nullptr, *device_output = nullptr;
    const std::size_t bytes = n * sizeof(float);
    if (!checked(cudaMalloc(reinterpret_cast<void**>(&device_x), bytes), "cudaMalloc x") ||
        !checked(cudaMalloc(reinterpret_cast<void**>(&device_y), bytes), "cudaMalloc y") ||
        !checked(cudaMalloc(reinterpret_cast<void**>(&device_output), bytes), "cudaMalloc output")) {
        return release_buffers(device_x, device_y, device_output, 1);
    }
    if (!checked(cudaMemcpy(device_x, x, bytes, cudaMemcpyHostToDevice), "cudaMemcpy x") ||
        !checked(cudaMemcpy(device_y, y, bytes, cudaMemcpyHostToDevice), "cudaMemcpy y")) {
        return release_buffers(device_x, device_y, device_output, 1);
    }
    const unsigned int blocks = static_cast<unsigned int>((n + 255) / 256);
    axpy_kernel<<<blocks, 256>>>(device_x, device_y, device_output, n, alpha);
    if (!checked(cudaGetLastError(), "axpy launch") ||
        !checked(cudaDeviceSynchronize(), "axpy synchronize") ||
        !checked(cudaMemcpy(output, device_output, bytes, cudaMemcpyDeviceToHost),
                 "cudaMemcpy output")) {
        return release_buffers(device_x, device_y, device_output, 1);
    }
    return release_buffers(device_x, device_y, device_output, 0);
}
"""

ROOT = Path(__file__).resolve().parent
TRAINING_CONFIG = {"seed": SEED, "learning_rate": 0.005, "batch_size": 257}


def build() -> Path:
    """Compile the embedded source; the initial source is CUDA, not portable HIP."""
    if sys.platform != "linux":
        raise RuntimeError("Compile/run this demo on a Linux GPU host; --inspect works locally")
    hip = bool(torch.version.hip)
    compiler_name = "hipcc" if hip else "nvcc"
    compiler = shutil.which(compiler_name)
    if compiler is None:
        raise RuntimeError(f"{compiler_name} is missing; use a GPU development image with its compiler")
    directory = ROOT / "artifacts"
    directory.mkdir(parents=True, exist_ok=True)
    source = directory / ("kernel.hip" if hip else "kernel.cu")
    output = directory / "kernel.so"
    source.write_text(CUDA_SOURCE, encoding="utf-8")
    flags = ["-O2", "-std=c++17", "-shared"]
    flags += ["-fPIC"] if hip else ["-Xcompiler=-fPIC"]
    subprocess.run([compiler, *flags, str(source), "-o", str(output)], check=True, timeout=120)
    return output


def validate_optimizer(optimizer: torch.optim.Optimizer, global_step: int) -> None:
    """Weight parity alone cannot prove that Adam momentum/counters were restored."""
    parameters = [parameter for group in optimizer.param_groups for parameter in group["params"]]
    if global_step == 0:
        if optimizer.state:
            raise ValueError("Step-zero checkpoint unexpectedly contains optimizer history")
        return
    for parameter in parameters:
        state = optimizer.state.get(parameter, {})
        counter = state.get("step")
        if (
            not isinstance(counter, torch.Tensor)
            or counter.numel() != 1
            or not torch.isfinite(counter).all()
            or float(counter) != global_step
        ):
            raise ValueError("Checkpoint is missing matching Adam step counters")
        for name in ("exp_avg", "exp_avg_sq"):
            moment = state.get(name)
            if (
                not isinstance(moment, torch.Tensor)
                or moment.shape != parameter.shape
                or not torch.isfinite(moment).all()
            ):
                raise ValueError(f"Checkpoint has missing or invalid Adam {name}")


def inspect_inputs() -> dict:
    """Preview the fixed contract without compiling a kernel or using a GPU."""
    x, y = input_tensors(5)
    return {
        "mode": "input_inspection_only", "gpu_execution": False, "operation": OPERATION,
        "formula": "features = 0.375 * x + y", "precision": "float32",
        "sample_x": x.tolist(), "sample_y": y.tolist(),
        "expected_custom_kernel_output": cpu_reference(x, y).tolist(),
        "kernel_check_lengths": list(SHAPES), "training_shape": [257, 64],
        "target_shape": [257, 16], "training_config": TRAINING_CONFIG,
    }


def cpu_reference(x: torch.Tensor, y: torch.Tensor) -> torch.Tensor:
    return ALPHA * x + y


def input_tensors(count: int) -> tuple[torch.Tensor, torch.Tensor]:
    if count < 0:
        raise ValueError("Input length must be nonnegative")
    x = torch.linspace(-2.0, 2.0, count, dtype=torch.float32)
    y = torch.linspace(0.75, -0.25, count, dtype=torch.float32)
    return x, y


class CustomKernel:
    def __init__(self, path: Path):
        self.library = ctypes.CDLL(str(path))
        self.function = self.library.portable_transform
        self.function.argtypes = [
            FLOAT_POINTER, FLOAT_POINTER, FLOAT_POINTER, ctypes.c_size_t, ctypes.c_float,
        ]
        self.function.restype = ctypes.c_int
        self.library.portable_last_error.argtypes = []
        self.library.portable_last_error.restype = ctypes.c_char_p
        self.calls = 0

    def __call__(self, x: torch.Tensor, y: torch.Tensor) -> torch.Tensor:
        for tensor in (x, y):
            if tensor.device.type != "cpu" or tensor.dtype != torch.float32:
                raise ValueError("The C ABI accepts CPU float32 tensors only")
            if not tensor.is_contiguous():
                raise ValueError("The C ABI requires contiguous input")
        if x.shape != y.shape:
            raise ValueError("AXPY inputs must have the same shape")
        output = torch.empty_like(x)
        status = self.function(
            ctypes.cast(x.data_ptr(), FLOAT_POINTER),
            ctypes.cast(y.data_ptr(), FLOAT_POINTER),
            ctypes.cast(output.data_ptr(), FLOAT_POINTER), x.numel(), ALPHA,
        )
        if status:
            message = self.library.portable_last_error().decode("utf-8", errors="replace")
            raise RuntimeError(f"Compiled custom kernel failed: {message}")
        self.calls += 1
        return output


def verify_kernel(kernel: CustomKernel) -> tuple[float, list[dict]]:
    errors = []
    for count in SHAPES:
        x, y = input_tensors(count)
        expected = cpu_reference(x, y)
        actual = kernel(x, y)
        error = float((actual - expected).abs().max()) if count else 0.0
        if not torch.isfinite(actual).all() or not torch.allclose(
            actual, expected, atol=1e-6, rtol=1e-6
        ):
            raise RuntimeError(f"Custom-kernel CPU reference mismatch at length {count}: {error}")
        errors.append({"elements": count, "max_abs_error": error})
    return max(item["max_abs_error"] for item in errors), errors


def training_data(kernel: CustomKernel) -> tuple[torch.Tensor, torch.Tensor, str, float]:
    generator = torch.Generator(device="cpu").manual_seed(SEED)
    x = torch.randn((257, 64), generator=generator)
    y = torch.randn((257, 64), generator=generator)
    expected = cpu_reference(x, y)
    features = kernel(x, y)
    error = float((features - expected).abs().max())
    if not torch.allclose(features, expected, atol=1e-6, rtol=1e-6):
        raise RuntimeError(f"Training-input custom-kernel error: {error}")
    weights = torch.randn((64, 16), generator=generator) / 8.0
    targets = torch.tanh(expected @ weights)
    digest = hashlib.sha256(
        json.dumps({"operation": OPERATION, "alpha": ALPHA, "shape": [257, 64]}, sort_keys=True).encode()
    )
    # Hash generated inputs and label weights, independent of hardware math rounding.
    for tensor in (x, y, weights):
        digest.update(ctypes.string_at(tensor.data_ptr(), tensor.numel() * tensor.element_size()))
    return features, targets, digest.hexdigest(), error


def make_model() -> torch.nn.Module:
    torch.manual_seed(SEED)
    return torch.nn.Sequential(
        torch.nn.Linear(64, 128), torch.nn.Tanh(), torch.nn.Linear(128, 16),
    )


def arguments() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--steps", type=int, default=8, help="Final global step, not extra steps")
    parser.add_argument("--resume", type=Path, help="An own-generated training checkpoint")
    parser.add_argument("--checkpoint", type=Path, default=ROOT / "artifacts/checkpoint.pt")
    parser.add_argument("--out", type=Path, default=ROOT / "artifacts/report.json")
    parser.add_argument("--inspect", action="store_true", help="Print sample inputs on CPU; no GPU run")
    parser.add_argument("--expect-vendor", choices=("nvidia", "amd"), help="Reject the wrong GPU backend")
    args = parser.parse_args()
    if args.steps < 0:
        parser.error("--steps must be nonnegative")
    return args


def main() -> None:
    args = arguments()
    if args.inspect:
        print(json.dumps(inspect_inputs(), indent=2))
        return
    if not torch.cuda.is_available():
        raise RuntimeError("A working CUDA or ROCm GPU is required; CPU fallback is disabled")
    torch.set_num_threads(1)
    torch.set_float32_matmul_precision("highest")
    torch.backends.cuda.matmul.allow_tf32 = False
    torch.backends.cudnn.allow_tf32 = False
    vendor = "amd" if torch.version.hip else "nvidia"
    if args.expect_vendor and vendor != args.expect_vendor:
        raise RuntimeError(f"Expected {args.expect_vendor}, found {vendor}")
    torch.cuda.set_device(0)
    free_memory, total_memory = torch.cuda.mem_get_info()
    if free_memory <= 0:
        raise RuntimeError("GPU reports zero free memory; stopping before compilation")
    kernel = CustomKernel(build())
    max_error, shape_results = verify_kernel(kernel)
    cpu_features, cpu_targets, input_hash, training_error = training_data(kernel)
    max_error = max(max_error, training_error)
    model = make_model().to("cuda")
    optimizer = torch.optim.Adam(model.parameters(), lr=TRAINING_CONFIG["learning_rate"])
    start_step = 0
    restored = False
    saved_predictions = None
    if args.resume:
        state = torch.load(args.resume, map_location="cpu", weights_only=True)
        if state.get("version") != 1 or state.get("operation") != OPERATION:
            raise ValueError("Checkpoint belongs to a different project or unsupported schema")
        if state.get("input_sha256") != input_hash:
            raise ValueError("Checkpoint input data does not match this run")
        if state.get("training_config") != TRAINING_CONFIG or state.get("precision") != "float32":
            raise ValueError("Checkpoint training settings or precision differ")
        if type(state.get("global_step")) is not int:
            raise ValueError("Checkpoint global step must be an integer")
        start_step = state["global_step"]
        if start_step < 0 or args.steps < start_step:
            raise ValueError("--steps must be at least the saved global step")
        model.load_state_dict(state["model"], strict=True)
        optimizer.load_state_dict(state["optimizer"])
        validate_optimizer(optimizer, start_step)
        torch.set_rng_state(state["cpu_rng_state"])
        saved_predictions = state["predictions"]
        restored = True
    features = cpu_features.to("cuda")
    targets = cpu_targets.to("cuda")
    torch.cuda.reset_peak_memory_stats()
    model.eval()
    with torch.no_grad():
        initial_predictions = model(features[:8]).cpu()
    prediction_error = None
    if saved_predictions is not None:
        prediction_error = float((initial_predictions - saved_predictions).abs().max())
        if not torch.allclose(initial_predictions, saved_predictions, atol=1e-5, rtol=1e-4):
            raise RuntimeError(f"Resumed checkpoint predictions changed: {prediction_error}")
    model.train()
    losses = []
    torch.cuda.synchronize()
    elapsed_start = time.perf_counter()
    for step in range(start_step + 1, args.steps + 1):
        optimizer.zero_grad(set_to_none=True)
        prediction = model(features)
        loss = torch.nn.functional.mse_loss(prediction, targets)
        value = float(loss.detach())
        if not math.isfinite(value):
            raise RuntimeError(f"Non-finite loss at global step {step}")
        loss.backward()
        if not all(
            parameter.grad is not None and torch.isfinite(parameter.grad).all()
            for parameter in model.parameters()
        ):
            raise RuntimeError(f"Missing or non-finite GPU gradients at global step {step}")
        optimizer.step()
        losses.append(value)
        print(json.dumps({"vendor": vendor, "global_step": step, "loss": value}), flush=True)
    torch.cuda.synchronize()
    elapsed = time.perf_counter() - elapsed_start
    model.eval()
    with torch.no_grad():
        final_predictions = model(features[:8]).cpu()
    if not torch.isfinite(final_predictions).all():
        raise RuntimeError("Non-finite model predictions")
    validate_optimizer(optimizer, args.steps)
    state = {
        "precision": "float32", "training_config": TRAINING_CONFIG,
        "version": 1, "operation": OPERATION, "global_step": args.steps,
        "input_sha256": input_hash, "model": model.state_dict(),
        "optimizer": optimizer.state_dict(), "cpu_rng_state": torch.get_rng_state(),
        "predictions": final_predictions,
    }
    args.checkpoint.parent.mkdir(parents=True, exist_ok=True)
    temporary = args.checkpoint.with_name(args.checkpoint.name + ".tmp")
    torch.save(state, temporary)
    temporary.replace(args.checkpoint)
    report = {
        "schema_version": 1, "operation": OPERATION, "vendor": vendor,
        "precision": "float32", "training_config": TRAINING_CONFIG,
        "native_kernel_source_sha256": hashlib.sha256(CUDA_SOURCE.encode()).hexdigest(),
        "free_memory_before_compile_bytes": free_memory,
        "total_device_memory_bytes": total_memory,
        "memory_scope": "PyTorch training allocator only; excludes native kernel allocations",
        "timing_scope": "training loop including correctness checks and step logging",
        "timing_note": "Short correctness smoke test; not a performance benchmark.",
        "gpu": torch.cuda.get_device_name(0), "torch_version": str(torch.__version__),
        "cuda_version": torch.version.cuda, "hip_version": torch.version.hip,
        "global_step": args.steps, "start_step": start_step, "steps_executed": len(losses),
        "loss_history": losses, "predictions": final_predictions.flatten().tolist(),
        "initial_predictions": initial_predictions.flatten().tolist(),
        "input_sha256": input_hash, "custom_kernel_max_abs_error": max_error,
        "custom_kernel_shape_checks": shape_results, "custom_kernel_calls": kernel.calls,
        "peak_allocated_bytes": torch.cuda.max_memory_allocated(),
        "peak_reserved_bytes": torch.cuda.max_memory_reserved(),
        "optimizer_restored": restored, "optimizer_state_entries": len(optimizer.state),
        "resume_prediction_max_abs_error": prediction_error,
        "checkpoint": str(args.checkpoint.resolve()), "training_seconds": elapsed,
        "parameters_on_gpu": all(p.device.type == "cuda" for p in model.parameters()),
        "checkpoint_scope": "model, Adam optimizer, global step, CPU RNG; fixed full-batch data",
    }
    args.out.parent.mkdir(parents=True, exist_ok=True)
    args.out.write_text(json.dumps(report, indent=2, allow_nan=False) + "\n", encoding="utf-8")
    print(json.dumps({"report": str(args.out.resolve()), "global_step": args.steps}), flush=True)


if __name__ == "__main__":
    main()
