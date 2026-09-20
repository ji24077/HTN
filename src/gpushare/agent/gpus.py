"""Per-GPU facts an agent may act on, and where each number came from.

A DIFFERENT AXIS FROM `profiles.py`. That table is per VENDOR — which wheel
index, which python, whether the image starts sshd. This one is per GPU MODEL:
how much memory it has, how fast one request actually came back on it, and
which optimisations have been validated THERE rather than in general. A card
being NVIDIA says nothing about whether compiled decoding holds its tail
latency on it.

MEASURED OR ESTIMATED, NEVER BLURRED. `latency_s` is populated only from a run
recorded in this repo, and `measured` says which. An entry with no run carries
None and the UI must show it as an estimate or not at all. The RTX 5090 is the
worked example: it is in the hardware matrix with `measured: false`, nothing
was ever run on it, and no number should be put in its mouth.
"""

from __future__ import annotations

from dataclasses import dataclass, field

__all__ = ["GpuProfile", "GPUS", "gpu_for", "OPTIMIZATIONS"]

# The optimisations an inference agent may propose, and what each one costs to
# turn on. Verdicts live per GPU because that is where they were decided.
OPTIMIZATIONS = {
    "static_kv_cache": "Preallocated KV cache; no per-step reallocation.",
    "torch_compile": "Inductor + CUDA graphs (reduce-overhead) over static cache.",
    "batch_16": "Sixteen concurrent requests. Throughput, not felt latency.",
}


@dataclass(frozen=True)
class GpuProfile:
    name: str
    vram_gb: int
    """From the calibration table this repo already ships (`/api/state` chips)."""

    measured: bool
    """Whether anything in this repo was ever actually run on this card."""

    latency_s: float | None
    """Batch-1 median, 13-case gate, from demo/results/hardware-matrix-2026-09-19.
    None where nothing was run — an absent measurement is not a zero and not a
    guess."""

    fits: tuple[str, ...]
    """Models this card has held, or can hold, for the MVP."""

    validated: tuple[str, ...]
    """Optimisations measured on THIS card and accepted."""

    rejected: dict[str, str] = field(default_factory=dict)
    """Optimisation -> why it was refused here. Kept because "we tried it and
    it was worse" is the most useful thing to know and the first thing lost."""


# Latencies are the batch-1 medians from the hardware matrix, one model, one
# gate, so the cards are comparable to each other. They are NOT comparable to
# scripts/probe_decode_latency.py, which used its own prompts and budget.
GPUS: dict[str, GpuProfile] = {
    "rtx3090": GpuProfile(
        name="RTX 3090",
        vram_gb=24,
        measured=True,
        latency_s=1.197,
        fits=("Qwen2.5-0.5B", "Qwen3-4B-Instruct"),
        validated=("static_kv_cache",),
        rejected={
            "batch_16": "throughput gate passed but the 300-case output check did not",
        },
    ),
    "rtx4090": GpuProfile(
        name="RTX 4090",
        vram_gb=24,
        measured=True,
        latency_s=0.639,
        fits=("Qwen2.5-0.5B", "Qwen3-4B-Instruct"),
        validated=("static_kv_cache",),
        rejected={
            "torch_compile": (
                "median improved 0.130s -> 0.115s but p95 went 0.132s -> 1.507s, "
                "worse than eager's 0.404s; measured by scripts/probe_decode_latency.py"
            ),
            "batch_16": "throughput gate passed but the 300-case output check did not",
        },
    ),
    "rtx5090": GpuProfile(
        name="RTX 5090",
        vram_gb=32,
        measured=False,
        latency_s=None,
        fits=("Qwen2.5-0.5B", "Qwen3-4B-Instruct"),
        validated=(),
        rejected={},
    ),
    "a5000": GpuProfile(
        name="RTX A5000",
        vram_gb=24,
        measured=True,
        latency_s=1.383,
        fits=("Qwen2.5-0.5B", "Qwen3-4B-Instruct"),
        validated=("static_kv_cache", "batch_16"),
        rejected={},
    ),
    "l40s": GpuProfile(
        name="L40S",
        vram_gb=48,
        measured=True,
        latency_s=1.702,
        fits=("Qwen2.5-0.5B", "Qwen3-4B-Instruct"),
        validated=("static_kv_cache", "batch_16"),
        rejected={},
    ),
    "mi300x": GpuProfile(
        name="MI300X",
        vram_gb=192,
        measured=True,
        latency_s=1.136,
        fits=("Qwen2.5-0.5B", "Qwen3-4B-Instruct"),
        validated=("static_kv_cache",),
        rejected={
            "batch_16": (
                "outputs changed: 'graduate teaching assistant' became 'graduate "
                "teaching assistant in the linguistics department'"
            ),
        },
    ),
}


def gpu_for(key: str) -> GpuProfile:
    """Refuse an unknown card rather than guessing a safe-looking default."""
    try:
        return GPUS[key]
    except KeyError:
        known = ", ".join(sorted(GPUS))
        raise KeyError(f"no GPU profile for {key!r} — known: {known}") from None


def faster_than(key: str) -> list[GpuProfile]:
    """Cards measured faster than this one, fastest first.

    Only measured cards, and only if the card asked about was measured too.
    A migration recommendation built on two estimates is a guess with a
    latency number attached to it.
    """
    here = gpu_for(key)
    if not here.measured or here.latency_s is None:
        return []
    candidates = [
        g for g in GPUS.values() if g.measured and g.latency_s is not None and g.latency_s < here.latency_s
    ]
    return sorted(candidates, key=lambda g: g.latency_s or 0.0)
