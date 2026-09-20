"""gpu_upload.py — upload alone and ask:

    Train with gpu_upload.py for 2000 steps on a GPU worker (CUDA or MPS, no CPU).
    Probe with --steps 3. Validate with gpu_upload.py --validate, require mse <= 0.01.
    Return checkpoint.pt and metrics.json.

Fails at import time unless PyTorch can reach a CUDA or MPS device, so the planner
must place it on a worker started with WORKER_RUNTIME=auto and torch installed:

    WORKER_EXECUTOR=python_project WORKER_RUNTIME=auto uv run --project backend \
        --python 3.12 --extra demo --with torch==2.13.0 orchestrator-demo worker-b --port 8080
"""
import argparse
import json
import os
from pathlib import Path

import torch

OUT = Path(os.environ["DISPATCH_OUTPUT_DIR"])

if torch.cuda.is_available():
    DEVICE = "cuda"
elif torch.backends.mps.is_available():
    DEVICE = "mps"
else:
    raise RuntimeError("Requires a CUDA or MPS GPU; CPU fallback disabled.")


def model():
    return torch.nn.Sequential(torch.nn.Linear(1, 1024), torch.nn.Tanh(), torch.nn.Linear(1024, 1))


def target(x):
    return torch.sin(3 * x)


def train(steps):
    torch.manual_seed(0)
    net = model().to(DEVICE)
    opt = torch.optim.Adam(net.parameters(), 1e-3)
    loss = None
    for _ in range(steps):
        x = torch.rand(8192, 1, device=DEVICE) * 4 - 2  # batched matmul on the GPU
        loss = torch.nn.functional.mse_loss(net(x), target(x))
        opt.zero_grad()
        loss.backward()
        opt.step()
    OUT.mkdir(parents=True, exist_ok=True)
    torch.save(
        {"device": DEVICE, "state": {k: v.cpu() for k, v in net.state_dict().items()}},
        OUT / "checkpoint.pt",
    )
    print(json.dumps({"device": DEVICE, "steps": steps, "loss": loss.item()}))


def validate():
    ckpt = torch.load(OUT / "checkpoint.pt", map_location="cpu")
    if ckpt["device"] not in ("cuda", "mps"):
        raise ValueError("checkpoint was not trained on a GPU")
    net = model()
    net.load_state_dict(ckpt["state"])
    net.eval()
    xs = torch.linspace(-1.9, 1.9, 301).unsqueeze(1)
    with torch.no_grad():
        mse = torch.nn.functional.mse_loss(net(xs), target(xs)).item()
    (OUT / "metrics.json").write_text(json.dumps({"mse": mse}))
    print(json.dumps({"mse": mse}))
    if mse > 0.01:
        raise ValueError(f"mse {mse} > 0.01")


if __name__ == "__main__":
    p = argparse.ArgumentParser()
    p.add_argument("--steps", type=int, default=2000)
    p.add_argument("--validate", action="store_true")
    a = p.parse_args()
    validate() if a.validate else train(a.steps)
