"""Train on an available GPU, falling back to CPU; --steps 2 is a bounded probe."""

import argparse
import os
from pathlib import Path

import torch
from device import select_device

parser = argparse.ArgumentParser()
parser.add_argument("--steps", type=int, default=200)
parser.add_argument(
    "--device", choices=["auto", "cpu", "cuda", "mps"], default=os.getenv("DISPATCH_DEVICE", "auto")
)
args = parser.parse_args()
device = select_device(args.device)
torch.set_num_threads(1)
torch.manual_seed(7)
model = torch.nn.Linear(1, 1).to(device)
x = torch.linspace(-1, 1, 64).reshape(-1, 1).to(device)
y = 2 * x + 0.5
optimizer = torch.optim.SGD(model.parameters(), lr=0.1)
for _ in range(args.steps):
    optimizer.zero_grad()
    loss = torch.nn.functional.mse_loss(model(x), y)
    loss.backward()
    optimizer.step()
output = Path(os.environ["DISPATCH_OUTPUT_DIR"])
torch.save(model.state_dict(), output / "checkpoint.pt")
print(
    f"Completed {args.steps} steps on {device}; training loss={loss.item():.8f}; "
    f"model device={next(model.parameters()).device}, tensor device={x.device}"
)
