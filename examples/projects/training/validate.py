"""Reload the actual saved checkpoint and evaluate unseen inputs."""

import json
import math
import os
from pathlib import Path

out = Path(os.environ["DISPATCH_OUTPUT_DIR"])
model = json.loads((out / "checkpoint.json").read_text())
xs = [-0.95, -0.35, 0.15, 0.75, 1.05]
mse = sum((model["weight"] * x + model["bias"] - (2 * x + 1)) ** 2 for x in xs) / len(xs)
if not math.isfinite(mse):
    raise ValueError("Evaluation did not return a finite loss")
(out / "metrics.json").write_text(json.dumps({"mse": mse}, allow_nan=False))
# Acceptance policy: MSE must be <= 0.001 on these held-out inputs.
if mse > 0.001:
    raise ValueError(f"Held-out MSE {mse} exceeds 0.001")
