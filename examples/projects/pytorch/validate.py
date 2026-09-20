"""Reload the trained checkpoint and measure error on held-out CPU inputs."""

import json
import os
from pathlib import Path

import torch

torch.set_num_threads(1)
output = Path(os.environ["DISPATCH_OUTPUT_DIR"])
model = torch.nn.Linear(1, 1).to("cpu")
model.load_state_dict(torch.load(output / "checkpoint.pt", map_location="cpu", weights_only=True))
x = torch.tensor([[-0.93], [-0.41], [0.17], [0.83]])
with torch.no_grad():
    mse = torch.nn.functional.mse_loss(model(x), 2 * x + 0.5).item()
if not 0 <= mse <= 0.001:
    raise ValueError(f"Held-out MSE {mse} exceeds 0.001")
(output / "metrics.json").write_text(json.dumps({"mse": mse, "device": "cpu"}))
print(f"Validated checkpoint on CPU; held-out MSE={mse:.8f}")
