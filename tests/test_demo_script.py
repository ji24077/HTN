"""Offline checks of the demo's oracle and checkpoint guard; no native GPU claims."""

import importlib.util
import json
import subprocess
import sys
from pathlib import Path
from types import SimpleNamespace

import pytest
import torch

SCRIPT = Path(__file__).resolve().parents[1] / "demo/train.py"
spec = importlib.util.spec_from_file_location("translation_demo", SCRIPT)
demo = importlib.util.module_from_spec(spec)
spec.loader.exec_module(demo)


def test_inspection_is_standalone_and_explicitly_not_gpu_execution():
    result = subprocess.run(
        [sys.executable, str(SCRIPT), "--inspect"],
        capture_output=True, text=True, check=True, timeout=30,
    )
    report = json.loads(result.stdout)
    assert report["gpu_execution"] is False
    assert report["expected_custom_kernel_output"] == [0.0, 0.125, 0.25, 0.375, 0.5]
    assert report["training_shape"] == [257, 64]


def test_verifier_rejects_incorrect_native_output():
    def broken_kernel(x, y):
        return torch.zeros_like(x)

    with pytest.raises(RuntimeError, match="reference mismatch"):
        demo.verify_kernel(broken_kernel)


def test_verifier_rejects_nonfinite_native_output():
    def broken_kernel(x, y):
        return torch.full_like(x, float("nan"))

    with pytest.raises(RuntimeError, match="reference mismatch"):
        demo.verify_kernel(broken_kernel)


def test_training_uses_custom_output_and_checks_it_against_independent_reference():
    calls = []

    def reference_kernel(x, y):
        calls.append(tuple(x.shape))
        return demo.cpu_reference(x, y)

    features, targets, identity, error = demo.training_data(reference_kernel)
    assert calls == [(257, 64)]
    assert features.shape == (257, 64) and targets.shape == (257, 16)
    assert len(identity) == 64 and error == 0
    assert torch.isfinite(targets).all()
    with pytest.raises(RuntimeError, match="Training-input custom-kernel error"):
        demo.training_data(lambda x, y: demo.cpu_reference(x, y) + 0.1)


def test_zero_free_memory_stops_before_native_build(monkeypatch):
    monkeypatch.setattr(demo, "arguments", lambda: SimpleNamespace(inspect=False, expect_vendor=None))
    monkeypatch.setattr(torch.cuda, "is_available", lambda: True)
    monkeypatch.setattr(torch.cuda, "set_device", lambda _: None)
    monkeypatch.setattr(torch.cuda, "mem_get_info", lambda: (0, 192 * 1024**3))
    monkeypatch.setattr(demo, "build", lambda: pytest.fail("empty GPU must not start compilation"))
    with pytest.raises(RuntimeError, match="zero free memory"):
        demo.main()


def test_changing_the_operation_parameter_changes_input_identity(monkeypatch):
    first = demo.training_data(demo.cpu_reference)[2]
    monkeypatch.setattr(demo, "ALPHA", 0.5)
    changed = demo.training_data(demo.cpu_reference)[2]
    assert changed != first


def trained_optimizer():
    model = torch.nn.Linear(2, 1)
    optimizer = torch.optim.Adam(model.parameters())
    for _ in range(2):
        optimizer.zero_grad()
        model(torch.ones(1, 2)).square().mean().backward()
        optimizer.step()
    return optimizer


def test_nonzero_checkpoint_requires_real_adam_history():
    optimizer = torch.optim.Adam(torch.nn.Linear(2, 1).parameters())
    demo.validate_optimizer(optimizer, 0)
    with pytest.raises(ValueError, match="step counters"):
        demo.validate_optimizer(optimizer, 2)


def test_adam_state_requires_correct_counters_and_finite_moments():
    optimizer = trained_optimizer()
    demo.validate_optimizer(optimizer, 2)
    state = next(iter(optimizer.state.values()))
    state["step"].add_(1)
    with pytest.raises(ValueError, match="step counters"):
        demo.validate_optimizer(optimizer, 2)
    state["step"].sub_(1)
    state["exp_avg"].fill_(float("nan"))
    with pytest.raises(ValueError, match="invalid Adam exp_avg"):
        demo.validate_optimizer(optimizer, 2)
