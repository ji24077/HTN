"""Run this FIRST on every machine. Prints what the box actually is.

Catches the classic ROCm failure: the wheel installs fine but the driver
version doesn't match, so is_available() is False and nothing tells you why.
"""

import sys

try:
    import torch
except ImportError:
    print("no torch on this box (fine for the server) -- extras: server/agent")
    sys.exit(0)

is_amd = torch.version.hip is not None
print(f"torch          {torch.__version__}")
print(f"build          {'ROCm ' + torch.version.hip if is_amd else 'CUDA ' + str(torch.version.cuda)}")
print(f"available      {torch.cuda.is_available()}")

if not torch.cuda.is_available():
    print()
    print("!! GPU NOT VISIBLE.")
    print("   AMD: the ROCm wheel must match the installed driver version.")
    print("        check `rocminfo`, then fix the index url in pyproject.toml")
    print("   NVIDIA: check `nvidia-smi` and the cuXXX index url")
    sys.exit(1)

from gpushare.contracts import classify_chip  # noqa: E402

p = torch.cuda.get_device_properties(0)
cc = float(f"{p.major}.{p.minor}")
vram = round(p.total_memory / 1e9, 1)
print(f"gpu            {p.name}  {vram}GB  cc{cc}")
print(f"chip_class     {classify_chip(gpu_name=p.name, cc=cc, vram_gb=vram, is_amd=is_amd)}")
print(f"trainable      {cc >= 7.0}")
