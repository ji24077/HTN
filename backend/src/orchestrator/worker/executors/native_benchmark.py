"""Trusted same-GPU AXPY native-stage benchmark; never imports candidate Python.

Run only in the isolated GPU environment used to validate candidate native code.
Results concern the host-buffer native preprocessing ABI, not whole training.
"""

from __future__ import annotations

import argparse
import ast
import ctypes
import hashlib
import json
import math
import shutil
import statistics
import subprocess
import sys
import tempfile
import time
from pathlib import Path

SIZES = (1, 257, 65537, 1048576)
ALPHA = 0.375
WARMUP_ROUNDS = 3
MIN_ROUNDS = 21
MIN_IMPROVEMENT = 0.05
MIN_WIN_FRACTION = 0.75


def extract_native(path: Path) -> str:
    """Read one top-level string without trusting a script's build/oracle code."""
    tree = ast.parse(path.read_text(encoding="utf-8"))
    found = []
    names = {"CUDA_SOURCE", "HIP_SOURCE"}
    for node in tree.body:
        if isinstance(node, ast.Assign):
            relevant = [target for target in node.targets if isinstance(target, ast.Name) and target.id in names]
            if relevant:
                if len(node.targets) != 1:
                    raise ValueError("Native source aliases are unsupported")
                found.append(node.value)
        elif isinstance(node, ast.AnnAssign) and isinstance(node.target, ast.Name) and node.target.id in names:
            found.append(node.value)
    writes = [node for node in ast.walk(tree) if isinstance(node, ast.Name) and node.id in names and isinstance(node.ctx, (ast.Store, ast.Del))]
    if len(found) != 1 or len(writes) != 1 or not isinstance(found[0], ast.Constant) or not isinstance(found[0].value, str):
        raise ValueError("Exactly one top-level CUDA_SOURCE or HIP_SOURCE literal is required")
    return found[0].value


def summarize_pairs(baseline: list[float], candidate: list[float], *, correctness_verified: bool) -> dict:
    """Summarize paired observations, refusing small/invalid or incorrect trials."""
    if len(baseline) != len(candidate) or not baseline:
        raise ValueError("Paired samples require equal nonempty lengths")
    for values in (baseline, candidate):
        if any(isinstance(value, bool) or not isinstance(value, (int, float)) or not math.isfinite(value) or value <= 0 for value in values):
            raise ValueError("Durations must be finite positive seconds")
    improvements = [1.0 - optimized / original for original, optimized in zip(baseline, candidate, strict=True)]
    median_improvement = statistics.median(improvements)
    wins = sum(optimized < original for original, optimized in zip(baseline, candidate, strict=True))
    fraction = wins / len(baseline)
    return {
        "rounds": len(baseline),
        "baseline_median_seconds": statistics.median(baseline),
        "candidate_median_seconds": statistics.median(candidate),
        "paired_median_improvement_fraction": median_improvement,
        "paired_improvement_fractions": improvements,
        "candidate_faster_rounds": wins,
        "candidate_win_fraction": fraction,
        "performance_verified": correctness_verified is True and len(baseline) >= MIN_ROUNDS
        and median_improvement > MIN_IMPROVEMENT + 1e-12 and fraction >= MIN_WIN_FRACTION,
    }


class NativeLibrary:
    def __init__(self, source: str, directory: Path, *, hip: bool):
        compiler_name = "hipcc" if hip else "nvcc"
        compiler = shutil.which(compiler_name)
        if not compiler:
            raise RuntimeError(f"{compiler_name} is required for the active PyTorch backend")
        directory.mkdir(parents=True, exist_ok=False)
        source_path = directory / ("kernel.hip" if hip else "kernel.cu")
        source_path.write_text(source, encoding="utf-8")
        library = directory / "kernel.so"
        flags = ["-O2", "-std=c++17", "-shared", *(["-fPIC"] if hip else ["-Xcompiler=-fPIC"])]
        process = subprocess.run([compiler, *flags, str(source_path), "-o", str(library)],
                                 capture_output=True, text=True, check=False, timeout=120)
        self.compile_log = {"compiler": compiler, "flags": flags, "returncode": process.returncode,
                            "stdout": process.stdout[-12000:], "stderr": process.stderr[-12000:]}
        if process.returncode:
            raise RuntimeError(f"Native compilation failed ({compiler_name}): {process.stderr[-8000:]}")
        self.library = ctypes.CDLL(str(library))
        self.function = self.library.portable_transform
        self.function.argtypes = [ctypes.POINTER(ctypes.c_float)] * 3 + [ctypes.c_size_t, ctypes.c_float]
        self.function.restype = ctypes.c_int
        self.error = self.library.portable_last_error
        self.error.argtypes = []
        self.error.restype = ctypes.c_char_p

    def invoke(self, pointers: tuple, count: int) -> None:
        status = self.function(*pointers, count, ALPHA)
        if status:
            message = self.error()
            raise RuntimeError(f"Native AXPY failed: {message.decode('utf-8', errors='replace') if message else status}")


def _pointers(*tensors) -> tuple:
    return tuple(ctypes.cast(tensor.data_ptr(), ctypes.POINTER(ctypes.c_float)) for tensor in tensors)


def _measure_native(torch, library, buffers: tuple, reference: tuple, pointers: tuple,
                    count: int, label: str) -> tuple[float, float]:
    current_x, current_y, output = buffers
    x, y, expected = reference
    # Poisoning and fixed reference validation are outside all timings.
    output.fill_(float("nan"))
    torch.cuda.synchronize()
    begin = time.perf_counter()
    library.invoke(pointers, count)
    torch.cuda.synchronize()
    seconds = time.perf_counter() - begin
    if torch.cuda.current_device() != 0:
        raise RuntimeError(f"{label} changed the selected GPU")
    if not torch.equal(current_x, x) or not torch.equal(current_y, y):
        raise RuntimeError(f"{label} modified immutable AXPY inputs at size {count}")
    if not torch.isfinite(output).all() or not torch.allclose(output, expected, atol=1e-6, rtol=1e-6):
        raise RuntimeError(f"{label} failed the fixed float32 CPU oracle at size {count}")
    return seconds, float((output - expected).abs().max())


def benchmark(baseline_path: Path, candidate_path: Path, *, rounds: int = MIN_ROUNDS) -> dict:
    if type(rounds) is not int or not MIN_ROUNDS <= rounds <= 101:
        raise ValueError("rounds must be an integer from 21 through 101")
    if sys.platform != "linux":
        raise RuntimeError("Native benchmarking requires a Linux GPU host")
    import torch

    torch.set_num_threads(min(torch.get_num_threads(), 4))
    if not torch.cuda.is_available():
        raise RuntimeError("GPU-enabled PyTorch and a visible GPU are required; CPU fallback is forbidden")
    if torch.cuda.device_count() != 1:
        raise RuntimeError("Expose exactly one GPU so both candidates are measured on the same device")
    torch.cuda.set_device(0)
    torch.cuda.synchronize()
    if torch.cuda.mem_get_info()[0] <= 0:
        raise RuntimeError("The selected GPU has no free memory")
    hip = bool(torch.version.hip)
    sources = {"baseline": extract_native(baseline_path), "candidate": extract_native(candidate_path)}
    results = []
    with tempfile.TemporaryDirectory(prefix="gpushare-native-benchmark-") as temporary:
        directory = Path(temporary)
        libraries = {name: NativeLibrary(source, directory / name, hip=hip) for name, source in sources.items()}
        for count in SIZES:
            generator = torch.Generator(device="cpu").manual_seed(20260919 + count)
            x = torch.randn(count, dtype=torch.float32, generator=generator)
            y = torch.randn(count, dtype=torch.float32, generator=generator)
            expected = ALPHA * x + y
            buffers = {name: (x.clone(), y.clone(), torch.empty_like(x)) for name in libraries}
            pointers = {name: _pointers(*values) for name, values in buffers.items()}
            max_error = {name: 0.0 for name in libraries}
            samples = {name: [] for name in libraries}
            orders = []

            for index in range(-WARMUP_ROUNDS, rounds):
                order = ("baseline", "candidate") if index % 2 == 0 else ("candidate", "baseline")
                if index >= 0:
                    orders.append(list(order))
                for name in order:
                    seconds, error = _measure_native(torch, libraries[name], buffers[name],
                                                     (x, y, expected), pointers[name], count, name)
                    max_error[name] = max(max_error[name], error)
                    if index >= 0:
                        samples[name].append(seconds)
            results.append({
                "elements": count, "correctness_verified": True, "max_abs_error": max_error,
                "baseline_sample_seconds": samples["baseline"],
                "candidate_sample_seconds": samples["candidate"], "execution_order": orders,
                **summarize_pairs(samples["baseline"], samples["candidate"], correctness_verified=True),
            })
        compile_logs = {name: library.compile_log for name, library in libraries.items()}
    return {
        "schema_version": 1, "status": "measured", "correctness_verified": True,
        "performance_verified": all(case["performance_verified"] for case in results),
        "performance_gate": "Each fixed shape must exceed 5% median paired improvement with at least 75% faster rounds, after every output passes the oracle.",
        "scope": "native AXPY preprocessing only; no whole-training speedup claim",
        "timing_scope": "host-buffer native ABI including device allocations, transfers, launch, native cleanup, and final GPU synchronization; excludes input generation, host buffer allocation, CPU oracle, compilation, and warmup",
        "precision": "float32", "alpha": ALPHA,
        "vendor": "amd" if hip else "nvidia", "gpu": torch.cuda.get_device_name(0),
        "device_index": 0, "same_gpu_and_process": True, "torch_version": str(torch.__version__),
        "cuda_version": torch.version.cuda, "hip_version": torch.version.hip,
        "gpu_environment_verified": True, "native_gpu_trace_verified": False,
        "verification_limit": "Compiled native invocation and GPU availability are checked; a device profiler is required for independent per-kernel execution traces. Native sources must pass the translation guard before running this harness.",
        "source_sha256": {name: hashlib.sha256(source.encode()).hexdigest() for name, source in sources.items()},
        "warmup_rounds": WARMUP_ROUNDS, "measured_rounds": rounds,
        "compile_logs": compile_logs, "cases": results,
    }


def main(argv: list[str] | None = None) -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--baseline", type=Path, required=True)
    parser.add_argument("--candidate", type=Path, required=True)
    parser.add_argument("--out", type=Path, required=True)
    parser.add_argument("--rounds", type=int, default=MIN_ROUNDS)
    args = parser.parse_args(argv)
    if args.out.exists():
        parser.error("--out must be a new JSON file")
    try:
        result = benchmark(args.baseline, args.candidate, rounds=args.rounds)
        exit_code = 0
    except (OSError, ValueError, RuntimeError, SyntaxError, subprocess.TimeoutExpired) as exc:
        result = {"schema_version": 1, "status": "failed", "correctness_verified": False,
                  "performance_verified": False, "error": str(exc),
                  "scope": "native AXPY preprocessing only"}
        exit_code = 1
    args.out.parent.mkdir(parents=True, exist_ok=True)
    with args.out.open("x", encoding="utf-8") as stream:
        json.dump(result, stream, indent=2, allow_nan=False)
        stream.write("\n")
    print(json.dumps({"status": result["status"], "performance_verified": result["performance_verified"],
                      "report": str(args.out.resolve())}))
    return exit_code


if __name__ == "__main__":
    raise SystemExit(main())
