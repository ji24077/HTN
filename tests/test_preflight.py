"""Backend detection is tested using fakes; never allocates real GPU memory."""

import importlib.util
from pathlib import Path
from types import SimpleNamespace

import pytest

spec = importlib.util.spec_from_file_location(
    "check_env_script", Path(__file__).resolve().parents[1] / "scripts/check_env.py"
)
check_env = importlib.util.module_from_spec(spec)
spec.loader.exec_module(check_env)


def fake_torch(vendor="amd", available=True, version="2.10.0"):
    device = SimpleNamespace(
        name="AMD MI300X" if vendor == "amd" else "NVIDIA RTX 4090",
        total_memory=24_000_000_000,
        major=0 if vendor == "amd" else 8,
        minor=9,
    )
    return SimpleNamespace(
        __version__=version,
        version=SimpleNamespace(
            hip="7.1" if vendor == "amd" else None, cuda="12.8" if vendor == "nvidia" else None
        ),
        cuda=SimpleNamespace(
            is_available=lambda: available,
            get_device_properties=lambda index: device,
            is_bf16_supported=lambda: True,
        ),
    )


def test_amd_is_not_judged_by_nvidia_compute_capability():
    result = check_env.check_environment(
        fake_torch(), expected_vendor="amd", required_torch="2.10.0"
    )
    assert result["backend"] == "rocm" and result["trainable"]
    assert result["cuda_compute_capability"] is None
    assert result["smoke_test"] == "not_run"


@pytest.mark.parametrize(
    "torch,kwargs,match",
    [
        (fake_torch(available=False), {}, "not visible"),
        (fake_torch(), {"expected_vendor": "nvidia"}, "expected nvidia"),
        (fake_torch(version="2.11.0"), {"required_torch": "2.10.0"}, "expected torch"),
    ],
)
def test_preflight_rejects_unavailable_wrong_backend_and_wrong_version(torch, kwargs, match):
    with pytest.raises(ValueError, match=match):
        check_env.check_environment(torch, **kwargs)


def test_nvidia_reports_actual_compute_capability():
    result = check_env.check_environment(fake_torch(vendor="nvidia"))
    assert result["cuda_compute_capability"] == 8.9
    assert result["chip_class"] == "ada_24gb"
