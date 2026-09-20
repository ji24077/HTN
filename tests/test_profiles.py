"""The chip table, and the doc that has to keep up with it."""

import dataclasses
import pathlib

import pytest

from gpushare.agent.profiles import PROFILES, ChipProfile, profile_for

DOC = pathlib.Path(__file__).resolve().parents[1] / "docs" / "chips.md"


def test_every_field_is_documented():
    """A field nobody explained is a guess waiting to be repeated.

    Each of these cost a destroyed pod or a wrong diagnosis to establish, so
    the reasoning lives in docs/chips.md and this asserts it is still there.
    """
    doc = DOC.read_text()
    missing = [f.name for f in dataclasses.fields(ChipProfile) if f"`{f.name}`" not in doc]
    assert not missing, f"docs/chips.md does not name {missing}"


def test_an_unknown_vendor_is_refused_not_defaulted():
    """Defaulting to NVIDIA would rent AMD hardware and install CUDA wheels."""
    with pytest.raises(KeyError, match="no chip profile"):
        profile_for("intel")


def test_amd_does_not_go_through_uv():
    """The image has no uv and PEP 668 refuses installing one."""
    assert "uv" not in PROFILES["amd"].serve_python
    assert "uv" in PROFILES["nvidia"].serve_python


def test_the_amd_facts_that_cost_pods_are_recorded():
    amd = PROFILES["amd"]
    assert amd.needs_explicit_sshd, "22 minutes of uptimeInSeconds 0 came from this"
    assert amd.jit_cold_start, "9.26s then 0.45s — a demo must warm the pod"
    assert amd.torch_location == "/opt/venv", "python3 -c 'import torch' fails on a good pod"


def test_the_wheel_index_matches_the_host_that_was_verified():
    """check_env passed on a ROCm 7.1.1 host with the 7.1 index.

    pyproject still pins rocm7.0, which is NOT what ran. The table carries the
    verified value; the pin remains unproven and docs/chips.md says so.
    """
    assert PROFILES["amd"].wheel_index == "rocm7.1"
    assert "rocm7.0" in DOC.read_text(), "the unproven pin has to stay visible"
