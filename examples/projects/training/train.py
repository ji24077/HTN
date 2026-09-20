"""Train y = weight*x + bias by gradient descent. No third-party dependencies."""

import argparse
import json
import os
from pathlib import Path

parser = argparse.ArgumentParser()
parser.add_argument("--steps", type=int, default=200)
steps = parser.parse_args().steps
weight = bias = 0.0
xs = [i / 10 for i in range(-10, 11)]
for _ in range(steps):
    residuals = [weight * x + bias - (2 * x + 1) for x in xs]
    weight -= 0.1 * 2 * sum(r * x for r, x in zip(residuals, xs)) / len(xs)
    bias -= 0.1 * 2 * sum(residuals) / len(xs)
out = Path(os.environ["DISPATCH_OUTPUT_DIR"])
(out / "checkpoint.json").write_text(json.dumps({"weight": weight, "bias": bias}))
