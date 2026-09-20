"""What each vendor needs, in one table instead of six `if vendor == "amd"`.

Every field here was paid for. The AMD row cost $1.36 and three destroyed pods
to fill in, and each wrong guess looked like something else: a missing sshd
looked like a slow image pull, a mismatched ROCm major looked like a broken
lock, and a JIT cold start looked like a hung server. The point of a table is
that the next vendor is a row, not a hunt through branches.

`docs/chips.md` is the same content for a person, with the evidence. A test
asserts every field here is named there, so the two cannot drift.
"""

from __future__ import annotations

from dataclasses import dataclass

__all__ = ["ChipProfile", "PROFILES", "profile_for"]


@dataclass(frozen=True)
class ChipProfile:
    """How to bring up, serve from, and reason about one vendor's hardware."""

    vendor: str

    # ── build ────────────────────────────────────────────────────────────────
    project_extra: str
    """`uv sync --extra <this>`. Only used where the project lock is used."""

    wheel_index: str
    """PyTorch index for the isolated venv. MUST match the host's ROCm/CUDA
    major. The repo pins rocm7.0 and the MI300X host reports 7.1.1; 7.1 is what
    was installed and what ran, so the pin is still unproven."""

    serve_python: str
    """How to invoke python for serving. AMD cannot use uv: the image has none,
    and Ubuntu 24.04's PEP 668 guard refuses `pip install --user uv`."""

    # ── rental ───────────────────────────────────────────────────────────────
    pod_image: str
    """Verified to boot on RunPod and expose a GPU."""

    needs_explicit_sshd: bool
    """RunPod's NVIDIA images start sshd; the AMD one does not. Without a
    dockerStartCmd the pod sits at uptimeInSeconds 0 with no publicIp, which is
    indistinguishable from a slow image pull. Twenty-two minutes went there."""

    torch_location: str | None
    """Where the image's own torch lives, if it ships one. On AMD it is in a
    venv, so `python3 -c "import torch"` fails on a perfectly good pod."""

    # ── behaviour worth predicting ───────────────────────────────────────────
    jit_cold_start: bool
    """First request compiles kernels. Measured on MI300X: 9.26s then 0.45s —
    twenty times. A demo must warm the pod or it stalls on stage."""

    bf16: bool


# Keyed by vendor because that is what the code branches on. A chip that needs
# to differ from its vendor gets its own row here, not a special case at the
# call site.
PROFILES: dict[str, ChipProfile] = {
    "nvidia": ChipProfile(
        vendor="nvidia",
        project_extra="cuda",
        wheel_index="cu128",
        serve_python="uv run python",
        pod_image="runpod/pytorch:1.0.2-cu1281-torch280-ubuntu2404",
        needs_explicit_sshd=False,
        torch_location=None,
        jit_cold_start=False,
        bf16=True,
    ),
    "amd": ChipProfile(
        vendor="amd",
        project_extra="rocm",
        wheel_index="rocm7.1",
        serve_python=".migration-venv/bin/python",
        pod_image="rocm/pytorch:rocm7.1.1_ubuntu24.04_py3.12_pytorch_release_2.10.0",
        needs_explicit_sshd=True,
        torch_location="/opt/venv",
        jit_cold_start=True,
        bf16=True,
    ),
}


def profile_for(vendor: str) -> ChipProfile:
    """Refuse an unknown vendor rather than silently treating it as NVIDIA.

    A default would make a third vendor look supported: it would rent, sync
    CUDA wheels onto hardware that cannot run them, and fail somewhere far from
    the cause. Adding a row is the smaller job.
    """
    try:
        return PROFILES[vendor]
    except KeyError:
        known = ", ".join(sorted(PROFILES))
        raise KeyError(f"no chip profile for vendor {vendor!r} — known: {known}") from None
