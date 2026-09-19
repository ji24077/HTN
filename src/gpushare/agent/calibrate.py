"""Ji — per-chip constants, fitted from real probes. This is the moat, concretely.

The handbook's moat is a story: "run history accumulates per chip type." These
two numbers are that story as code. Each `chip_class` carries its own MFU and
memory overhead, fitted ONLY from probes tagged with that chip_class. A new chip
starts at DEFAULTS and gets better with data; nothing around it changes.

Both constants are observed by INVERTING the cost model against a measurement,
so there is no black box — you can check the arithmetic by hand.

RULE: a simulator never checked against hardware is worse than no simulator,
because you end up tuning the agent against fiction. Every prediction this module
backs must be reportable alongside its error. See error_report().
"""

from __future__ import annotations

import statistics
from dataclasses import dataclass, replace

from gpushare.agent.specs import ChipSpec, ModelSpec
from gpushare.contracts import ChipClass, JobConfig, ProbeResult


@dataclass(frozen=True)
class Calibration:
    """What the cost model can't derive from spec sheets.

    mfu            fraction of peak FLOPS actually reached. Absorbs kernel
                   efficiency, launch overhead, dataloader stalls, AND any error
                   in the nominal TFLOPS table — which is why that table only
                   has to be self-consistent.
    act_overhead   multiplier on predicted activation bytes. Absorbs allocator
                   fragmentation and whatever the _ACT_PER_TOKEN constant misses.
    sync_eff       fraction of nominal link bandwidth an all-reduce actually gets.
    n_samples      how many probes went into this. 0 == still a cold-start guess,
                   and the UI should say so rather than quoting it as measured.
    """

    mfu: float
    act_overhead: float
    sync_eff: float
    n_samples: int = 0

    def is_measured(self) -> bool:
        return self.n_samples > 0


# Cold start. Deliberately mid-range rather than optimistic: an over-confident
# default makes the agent promise speedups it can't deliver on an unknown chip.
COLD_START = Calibration(mfu=0.35, act_overhead=1.3, sync_eff=0.75, n_samples=0)

DEFAULTS: dict[ChipClass, Calibration] = {
    "ampere_24gb": COLD_START,
    "ada_24gb": COLD_START,
    "turing_16gb": replace(COLD_START, mfu=0.30),  # no bf16; fp16+GradScaler costs a little
    "cdna_amd": replace(COLD_START, mfu=0.25),  # ROCm kernels lag CUDA on small models
    "default": COLD_START,
}


def observe(
    probe: ProbeResult,
    *,
    cfg: JobConfig,
    model: ModelSpec,
    chip: ChipSpec,
) -> Calibration:
    """Invert the cost model against one real measurement.

    Each constant is a plain ratio of measured to predicted — no fitting
    machinery, nothing to tune, and every number is checkable by hand.

    seq_len comes from JobConfig, not DataSpec: the contract field is what the
    trainer actually ran at, and that is what the measurement reflects.
    """
    # Imported here, not at module scope: simulate imports this module, so a
    # top-level import would close the cycle. Sharing these two functions is what
    # makes observe() an exact inverse of predict_t_step() rather than a
    # second, silently-drifting copy of the same arithmetic.
    from gpushare.agent.simulate import effective_flops, rate_before_mfu

    # MFU: what fraction of peak did this step actually reach, once batch
    # occupancy and compile are divided back out?
    work = effective_flops(cfg=cfg, model=model)
    mfu = work / (probe.t_step_median_s * rate_before_mfu(cfg=cfg, chip=chip))

    # Activation overhead: strip the part we can compute exactly, ratio the rest.
    static_gb = model.static_bytes(cfg.dtype) / 1e9
    act_pred_gb = (
        model.activation_bytes(
            micro_batch=cfg.micro_batch,
            seq_len=cfg.seq_len,
            dtype=cfg.dtype,
            attention=cfg.attention,
        )
        / 1e9
    )
    act_measured_gb = max(probe.peak_vram_gb - static_gb, 0.0)
    act_overhead = act_measured_gb / act_pred_gb if act_pred_gb > 0 else COLD_START.act_overhead

    # Sync efficiency needs a link speed we don't carry on ProbeResult, so it
    # stays at the default until a measured T_sync is paired with a NetSpec.
    # Honest gap, not an oversight — see fit_sync_eff().
    return Calibration(
        mfu=mfu, act_overhead=act_overhead, sync_eff=COLD_START.sync_eff, n_samples=1
    )


def fit(observations: list[Calibration]) -> Calibration:
    """Combine per-probe observations into one constant set for a chip_class.

    MEDIAN, not mean — same reason the probe runner drops warmup and takes the
    median step time. One thermally throttled run shouldn't move the constant.
    """
    if not observations:
        return COLD_START
    return Calibration(
        mfu=statistics.median(o.mfu for o in observations),
        act_overhead=statistics.median(o.act_overhead for o in observations),
        sync_eff=statistics.median(o.sync_eff for o in observations),
        n_samples=sum(o.n_samples for o in observations),
    )


def fit_sync_eff(*, measured_t_sync_s: float, bytes_on_wire: int, link_bytes_per_s: float) -> float:
    """Sync efficiency from a measured round. Jack's gloo test gives the inputs."""
    ideal = bytes_on_wire / link_bytes_per_s
    return max(0.05, min(1.0, ideal / measured_t_sync_s))


class CalibrationStore:
    """Per-chip_class history. The 3090 agent only ever sees 3090 probes.

    In-memory for now; the persistent version is Jack's S-8 `probes` table, whose
    chip_class column is the same Contract 5 string. Swapping this for a SQL read
    changes nothing above it.
    """

    def __init__(self) -> None:
        self._obs: dict[ChipClass, list[Calibration]] = {}

    def add(self, chip_class: ChipClass, obs: Calibration) -> None:
        self._obs.setdefault(chip_class, []).append(obs)

    def get(self, chip_class: ChipClass) -> Calibration:
        obs = self._obs.get(chip_class)
        return fit(obs) if obs else DEFAULTS.get(chip_class, COLD_START)

    def counts(self) -> dict[ChipClass, int]:
        return {k: len(v) for k, v in self._obs.items()}


# ─────────────────────────────────────────────────────────────────────────────
# Error reporting — the part that keeps us honest
# ─────────────────────────────────────────────────────────────────────────────
def _pct(pred: float, meas: float) -> float:
    return abs(pred - meas) / meas * 100 if meas else float("inf")


def error_report(predicted: ProbeResult, measured: ProbeResult) -> dict[str, float]:
    """How far off was the simulation? Report this next to every projection.

    "predicted 1.66s, measured 1.71s (3%)" is what turns a cost model into
    evidence. A projection shown without its error is just a claim.
    """
    return {
        "t_step_pct": _pct(predicted.t_step_median_s, measured.t_step_median_s),
        "tokens_per_s_pct": _pct(predicted.tokens_per_s, measured.tokens_per_s),
        "peak_vram_pct": _pct(predicted.peak_vram_gb, measured.peak_vram_gb),
        "t_sync_pct": _pct(predicted.t_sync_s, measured.t_sync_s),
    }
