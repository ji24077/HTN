"""Reload the migrated model and check its training contract and unseen-input loss."""

import hashlib
import json
import math
import os
from pathlib import Path

import torch
from train import (
    CUDA_SOURCE,
    OPERATION,
    SEED,
    TRAINING_CONFIG,
    cpu_reference,
    make_model,
    training_data,
    validate_optimizer,
)

out = Path(os.environ["DISPATCH_OUTPUT_DIR"])
state = torch.load(out / "checkpoint.pt", map_location="cpu", weights_only=True)
report = json.loads((out / "report.json").read_text())
features, _, input_hash, _ = training_data(cpu_reference)
if (
    state.get("version") != 1
    or state.get("operation") != OPERATION
    or state.get("precision") != "float32"
    or state.get("training_config") != TRAINING_CONFIG
    or state.get("input_sha256") != input_hash
    or state.get("global_step") != 16
):
    raise ValueError("Checkpoint does not match the fixed 16-step training contract")
if (
    report.get("vendor") != "nvidia"
    or report.get("parameters_on_gpu") is not True
    or report.get("global_step") != 16
    or report.get("steps_executed") != 16
    or report.get("start_step") != 0
    or report.get("native_kernel_source_sha256")
    != hashlib.sha256(CUDA_SOURCE.encode()).hexdigest()
):
    raise ValueError("Full training report does not match the migrated NVIDIA artifact")
model = make_model()
initial = {name: tensor.clone() for name, tensor in model.state_dict().items()}
model.load_state_dict(state["model"], strict=True)
if all(
    torch.equal(initial[name], tensor) for name, tensor in model.state_dict().items()
):
    raise ValueError("Checkpoint parameters did not change during training")
optimizer = torch.optim.Adam(model.parameters(), lr=TRAINING_CONFIG["learning_rate"])
optimizer.load_state_dict(state["optimizer"])
validate_optimizer(optimizer, 16)
model.eval()
with torch.no_grad():
    replay = model(features[:8])
    if not torch.allclose(replay, state["predictions"], atol=1e-5, rtol=1e-4):
        raise ValueError(
            "Reloaded checkpoint predictions differ from the saved predictions"
        )
    if not torch.allclose(
        replay.flatten(), torch.tensor(report["predictions"]), atol=1e-5, rtol=1e-4
    ):
        raise ValueError("Saved report predictions differ from the checkpoint")
    # Recreate the original label generator, then evaluate independently seeded inputs.
    generator = torch.Generator(device="cpu").manual_seed(SEED)
    for _ in range(2):
        torch.randn((257, 64), generator=generator)
    weights = torch.randn((64, 16), generator=generator) / 8.0
    heldout = torch.Generator(device="cpu").manual_seed(SEED + 1)
    x = torch.randn((97, 64), generator=heldout)
    y = torch.randn((97, 64), generator=heldout)
    inputs = cpu_reference(x, y)
    targets = torch.tanh(inputs @ weights)
    loss = float(torch.nn.functional.mse_loss(model(inputs), targets))
if not math.isfinite(loss) or loss > 1.0:
    raise ValueError(f"Held-out loss {loss} exceeds the fixed demo maximum of 1.0")
(out / "metrics.json").write_text(json.dumps({"loss": loss}, allow_nan=False) + "\n")
