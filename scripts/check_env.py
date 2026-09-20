"""Inspect the actual backend; optional migration preflight runs real GPU math."""

from __future__ import annotations

import argparse
import importlib.metadata
import json
import sys

from gpushare.contracts import classify_chip


def check_environment(
    torch, *, expected_vendor=None, required_torch=None, smoke=False, dtype="bf16"
):
    hip = getattr(torch.version, "hip", None)
    cuda = getattr(torch.version, "cuda", None)
    vendor = "amd" if hip else "nvidia" if cuda else "cpu"
    report = {
        "torch": torch.__version__,
        "python": sys.version.split()[0],
        "backend": "rocm" if hip else "cuda" if cuda else "cpu",
        "backend_version": hip or cuda,
        "vendor": vendor,
        "available": bool(torch.cuda.is_available()),
        "smoke_test": "not_run",
    }
    if not report["available"]:
        raise ValueError("GPU is not visible; verify the wheel and host driver match")
    if expected_vendor and vendor != expected_vendor:
        raise ValueError(f"expected {expected_vendor} backend, found {vendor}")
    if required_torch and torch.__version__.split("+")[0] != required_torch:
        raise ValueError(f"expected torch {required_torch}, found {torch.__version__}")
    device = torch.cuda.get_device_properties(0)
    # AMD's major/minor values are not NVIDIA compute capability.
    cc = float(f"{device.major}.{device.minor}") if vendor == "nvidia" else None
    bf16 = bool(torch.cuda.is_bf16_supported())
    report.update(
        {
            "gpu": device.name,
            "vram_gb": round(device.total_memory / 1e9, 3),
            "cuda_compute_capability": cc,
            "bf16_supported": bf16,
            "chip_class": classify_chip(
                gpu_name=device.name,
                cc=cc or 0.0,
                vram_gb=device.total_memory / 1e9,
                is_amd=vendor == "amd",
            ),
            "trainable": (vendor == "amd" or (cc is not None and cc >= 7.0)),
        }
    )
    if dtype == "bf16" and not bf16 and smoke:
        raise ValueError("this backend does not support the requested bf16 workload")
    if smoke:
        precision = {"bf16": torch.bfloat16, "fp16": torch.float16, "fp32": torch.float32}[dtype]
        x = torch.ones((8, 8), device="cuda", dtype=precision, requires_grad=True)
        loss = (x @ x).float().square().mean()
        loss.backward()
        torch.cuda.synchronize()
        if x.grad is None or not bool(torch.isfinite(x.grad).all()):
            raise ValueError("GPU forward/backward produced invalid gradients")
        report.update(smoke_test="passed", smoke_dtype=dtype)
    for package in ("transformers", "peft", "safetensors"):
        try:
            report[package] = importlib.metadata.version(package)
        except importlib.metadata.PackageNotFoundError:
            report[package] = None
    return report


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--json", action="store_true")
    parser.add_argument("--expect-vendor", choices=("nvidia", "amd"))
    parser.add_argument("--require-torch-version")
    parser.add_argument("--smoke", action="store_true")
    parser.add_argument("--dtype", choices=("bf16", "fp16", "fp32"), default="bf16")
    args = parser.parse_args()
    try:
        import torch
    except ImportError:
        if args.expect_vendor or args.smoke:
            parser.exit(1, "GPU preflight requires torch\n")
        print("no torch on this box (fine for the server) -- extras: server/agent")
        return
    try:
        report = check_environment(
            torch,
            expected_vendor=args.expect_vendor,
            required_torch=args.require_torch_version,
            smoke=args.smoke,
            dtype=args.dtype,
        )
    except (ValueError, RuntimeError) as exc:
        parser.exit(1, f"GPU preflight failed: {exc}\n")
    if args.json:
        print(json.dumps(report, sort_keys=True))
    else:
        for key, value in report.items():
            print(f"{key:<24}{value}")


if __name__ == "__main__":
    main()
