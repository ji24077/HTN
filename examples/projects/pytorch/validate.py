"""Reload the trained checkpoint and measure error on held-out inputs."""

import argparse
import json
import os
from pathlib import Path

import torch
from device import select_device

parser = argparse.ArgumentParser()
parser.add_argument(
    "--device", choices=["auto", "cpu", "cuda", "mps"], default=os.getenv("DISPATCH_DEVICE", "auto")
)
device = select_device(parser.parse_args().device)
torch.set_num_threads(1)
output = Path(os.environ["DISPATCH_OUTPUT_DIR"])
model = torch.nn.Linear(1, 1).to(device)
model.load_state_dict(torch.load(output / "checkpoint.pt", map_location=device, weights_only=True))
x = torch.tensor([[-0.93], [-0.41], [0.17], [0.83]], device=device)
with torch.no_grad():
    mse = torch.nn.functional.mse_loss(model(x), 2 * x + 0.5).item()
if not 0 <= mse <= 0.001:
    raise ValueError(f"Held-out MSE {mse} exceeds 0.001")
(output / "metrics.json").write_text(json.dumps({"mse": mse, "device": str(device)}))
print(f"Validated checkpoint on {device}; held-out MSE={mse:.8f}")
