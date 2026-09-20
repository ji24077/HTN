"""Train a small linear model on CPU; --steps 2 is a bounded execution probe."""

import argparse
import os
from pathlib import Path

import torch

parser = argparse.ArgumentParser()
parser.add_argument("--steps", type=int, default=200)
args = parser.parse_args()
torch.set_num_threads(1)
torch.manual_seed(7)
model = torch.nn.Linear(1, 1).to("cpu")
x = torch.linspace(-1, 1, 64).reshape(-1, 1)
y = 2 * x + 0.5
optimizer = torch.optim.SGD(model.parameters(), lr=0.1)
for _ in range(args.steps):
    optimizer.zero_grad()
    loss = torch.nn.functional.mse_loss(model(x), y)
    loss.backward()
    optimizer.step()
output = Path(os.environ["DISPATCH_OUTPUT_DIR"])
torch.save(model.state_dict(), output / "checkpoint.pt")
print(f"Completed {args.steps} steps on CPU; training loss={loss.item():.8f}")
