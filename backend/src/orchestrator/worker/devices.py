"""Probe the serving interpreter in a bounded subprocess before advertising a GPU."""

import json
import os
import platform
import subprocess
import sys

from ..shared.protocol import Accelerator, Capabilities, Machine, PythonCapability

PROBE = r"""
import json
result = {}
cpu_ok = False
try:
    import torch
    result = {"runtime": "cpu", "vram_mib": 0, "device": None, "pytorch": torch.__version__}
    if torch.ones(16, device="cpu").sum().item() != 16:
        raise RuntimeError("CPU arithmetic check failed")
    cpu_ok = True
    runtime = "cuda" if torch.cuda.is_available() else (
        "mps" if torch.backends.mps.is_available() else "cpu")
    result = {"runtime": runtime, "vram_mib": 0, "device": None, "pytorch": torch.__version__}
    result["provider"] = "rocm" if runtime == "cuda" and torch.version.hip else runtime
    if runtime != "cpu":
        value = torch.ones(16, device=runtime).sum().item()
        if value != 16:
            raise RuntimeError("GPU arithmetic check failed")
        if runtime == "cuda":
            torch.cuda.synchronize()
            props = torch.cuda.get_device_properties(0)
            result.update(device=props.name, vram_mib=props.total_memory // 1048576)
        else:
            torch.mps.synchronize()
            result["device"] = "Apple GPU (Metal / PyTorch MPS)"
    result["reason"] = "" if runtime != "cpu" else "PyTorch cannot access a CUDA or MPS device."
except ModuleNotFoundError:
    result = {"runtime": "cpu", "reason": "PyTorch is not installed in this worker environment."}
except Exception as error:
    result.update(runtime="cpu", vram_mib=0, device=None,
                  reason="GPU probe failed: " + str(error)[:160])
    if not cpu_ok:
        result.pop("pytorch", None)
print(json.dumps(result))
"""


def capabilities(kind: str, requested: str = "auto") -> Capabilities:
    if requested not in {"auto", "cpu", "cuda", "mps"}:
        raise ValueError("WORKER_RUNTIME must be auto, cpu, cuda, or mps")
    try:
        output = subprocess.run(
            [sys.executable, "-c", PROBE], capture_output=True, text=True, timeout=30, check=True
        )
        detected = json.loads(output.stdout)
    except (OSError, subprocess.SubprocessError, ValueError):
        detected = {"runtime": "cpu", "reason": "GPU probe failed or timed out."}
    found = detected["runtime"]
    available = found in {"cuda", "mps"}
    runtime = found if requested == "auto" or requested == found else "cpu"
    reason = detected.get("reason", "")
    if available and runtime == "cpu":
        reason = "GPU detected; this worker was started in CPU mode."
    return Capabilities(
        runtime=runtime,
        # Apple unified memory is not dedicated VRAM; do not count it as such.
        vram_mib=detected.get("vram_mib", 0) if runtime == "cuda" else 0,
        kinds=[kind],
        python=PythonCapability(version=platform.python_version(), pytorch=detected["pytorch"])
        if detected.get("pytorch")
        else None,
        accelerator=Accelerator(
            available=available,
            device=detected.get("device"),
            reason=reason[:200],
            providers=[detected.get("provider", found)] if available else ["cpu"],
        ),
        runtime_preference="cpu" if requested == "cpu" else "auto",
        machine=Machine(
            os=platform.system(),
            arch=platform.machine(),
            logical_cores=os.cpu_count(),
            max_concurrency=1,
            runtime_control="startup",
        ),
    )
