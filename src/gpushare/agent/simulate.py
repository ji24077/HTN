"""Ji — the training cost model. Same interface as the real prober.

    Prober          the seam
    SimProber       free, instant, no GPU
    <RunPodProber>  real, costs money, slow  (agent/runpod.py, later)

Both return a `contracts.ProbeResult`, so Router and the chip agents cannot tell
which one they got. That is the whole point: develop the decision logic for free,
spend money only on validation.

Every prediction is a projection. Label it as one, and report its error next to
it — calibrate.error_report().
"""

from __future__ import annotations

from typing import Protocol

from gpushare.agent.calibrate import Calibration, CalibrationStore
from gpushare.agent.specs import ChipSpec, ModelSpec, NetSpec
from gpushare.contracts import ConfigName, JobConfig, ProbeResult

# torch.compile buys roughly this much, once it has paid its ~60s warmup. Lever 7
# in the handbook is a net LOSS on short jobs, which is why the router checks
# projected_runtime * 0.15 > compile_cost before switching it on.
COMPILE_SPEEDUP = 1.25

# A fitted MFU is only meaningful at a stated batch size. Below this, kernel
# launches and tail effects leave the card idle between micro-steps; above it,
# the card is already saturated and more batch buys nothing. Lever 3 — the batch
# search — exists entirely because of the gap between those two regimes, so a
# cost model without this term makes the biggest lever look like a no-op.
MFU_REFERENCE_TOKENS = 32_768
_HALF_SATURATION = MFU_REFERENCE_TOKENS / 3

# Flash/SDPA isn't only a memory win: it avoids writing and re-reading the
# seq x seq matrix, so the attention term runs faster too. Conservative factor —
# the real gain grows with seq_len, which we don't model.
SDPA_ATTENTION_SPEEDUP = 1.4


def _occupancy(tokens_per_forward: int) -> float:
    """Fraction of the calibrated MFU this micro-batch actually reaches.

    Saturating curve normalised so that MFU_REFERENCE_TOKENS scores 1.0. That
    normalisation is what makes calibration well-defined: observe() divides the
    same factor back out, so a probe taken at any batch size yields the same MFU.
    """
    ref = MFU_REFERENCE_TOKENS / (MFU_REFERENCE_TOKENS + _HALF_SATURATION)
    return min(1.0, (tokens_per_forward / (tokens_per_forward + _HALF_SATURATION)) / ref)


def rate_before_mfu(*, cfg: JobConfig, chip: ChipSpec) -> float:
    """Everything in the FLOP/s rate EXCEPT the fitted MFU.

    Split out so calibrate.observe() can solve for MFU without circularity:
    predict divides work by (this x mfu); observe divides work by (t_step x this).
    Sharing the function is what keeps them exact inverses — the property
    test_observe_inverts_predict() pins down.
    """
    rate = chip.tflops_for(cfg.dtype) * 1e12
    rate *= _occupancy(cfg.micro_batch * cfg.seq_len)
    if cfg.compile:
        rate *= COMPILE_SPEEDUP
    return rate


def effective_flops(*, cfg: JobConfig, model: ModelSpec) -> float:
    """Work in one optimizer step, with the attention term discounted for sdpa.

    Not literally fewer FLOPs — sdpa does the same math faster by never staging
    the seq x seq matrix through HBM. Folding that into the work term instead of
    carrying a second rate keeps one number to calibrate against.
    """
    tokens = cfg.micro_batch * cfg.grad_accum * cfg.seq_len
    param = model.param_flops_per_token() * tokens
    attn = model.attn_flops_per_token(cfg.seq_len) * tokens
    if cfg.attention == "sdpa":
        attn /= SDPA_ATTENTION_SPEEDUP
    return param + attn


class Prober(Protocol):
    """Measure a config on a chip. Simulated or real — callers don't care."""

    def probe(self, cfg: JobConfig, chip: ChipSpec) -> ProbeResult: ...


# ─────────────────────────────────────────────────────────────────────────────
# The three predictions
# ─────────────────────────────────────────────────────────────────────────────
def predict_t_step(*, cfg: JobConfig, model: ModelSpec, chip: ChipSpec, cal: Calibration) -> float:
    """Median seconds per optimizer step, on one worker.

    Work over rate. grad_accum multiplies the work (that's what accumulation is)
    but not the rate, which is why raising micro_batch while lowering grad_accum
    goes faster at identical tokens per step — the batch search's entire premise,
    and the fixed-work invariant JobConfig already enforces.
    """
    work = effective_flops(cfg=cfg, model=model)
    return work / (rate_before_mfu(cfg=cfg, chip=chip) * cal.mfu)


def predict_peak_vram_gb(*, cfg: JobConfig, model: ModelSpec, cal: Calibration) -> float:
    """Peak allocated VRAM.

    grad_accum does NOT appear here. Accumulation exists precisely so you can do
    a big global batch with a small resident one — memory tracks micro_batch
    alone. Getting this wrong is how a batch-size search OOMs.
    """
    static = model.static_bytes(cfg.dtype)
    act = model.activation_bytes(
        micro_batch=cfg.micro_batch,
        seq_len=cfg.seq_len,
        dtype=cfg.dtype,
        attention=cfg.attention,
    )
    return (static + act * cal.act_overhead) / 1e9


def predict_t_sync(*, cfg: JobConfig, model: ModelSpec, net: NetSpec, cal: Calibration) -> float:
    """Seconds for one DiLoCo round's all-reduce.

    Ring all-reduce moves 2(W-1)/W of the payload through each worker, so going
    from 2 to 4 workers costs 1.5x, not 2x.

    This number is what compute_H() divides by t_step. On a 100 Mbps link a 124M
    model is ~20s of sync, which drives H past its 500 ceiling — the model is too
    big for the pipe. That warning is a real agent output, and it costs nothing
    to discover here instead of on a rented GPU.
    """
    n = len(cfg.workers)
    if n < 2:
        return 0.0
    payload = model.sync_bytes(cfg.dtype) * 2 * (n - 1) / n
    transfer = payload / (net.bytes_per_s() * cal.sync_eff)
    return transfer + net.latency_ms / 1000 * 2


def predict_gpu_util(*, cfg: JobConfig, model: ModelSpec, chip: ChipSpec) -> float:
    """Crude on purpose, and the weakest number this module produces.

    Real utilisation depends on dataloader behaviour and kernel launch gaps,
    neither of which we model. Treat it as a hint about whether the batch is big
    enough to keep the card busy, and prefer a measured value whenever one exists.
    """
    work = model.flops_per_token(cfg.seq_len) * cfg.micro_batch * cfg.seq_len
    saturating = chip.tflops_for(cfg.dtype) * 1e12 * 0.02  # ~20ms of work saturates
    return round(min(0.98, 0.35 + 0.63 * min(1.0, work / saturating)), 3)


# ─────────────────────────────────────────────────────────────────────────────
# The simulated prober
# ─────────────────────────────────────────────────────────────────────────────
class SimProber:
    """Produces a ProbeResult without touching a GPU.

    Hands back the same shape the real prober does, including the median/warmup
    convention — there is nothing to average here, but the field name carries the
    promise that downstream code compares like with like.
    """

    def __init__(self, net: NetSpec, store: CalibrationStore | None = None) -> None:
        self.net = net
        self.store = store or CalibrationStore()

    def calibration_for(self, chip: ChipSpec) -> Calibration:
        return self.store.get(chip.chip_class)

    def probe(
        self,
        cfg: JobConfig,
        chip: ChipSpec,
        *,
        worker_id: str = "sim",
        config_name: ConfigName = "baseline",
    ) -> ProbeResult:
        cal = self.calibration_for(chip)
        t_step = predict_t_step(cfg=cfg, model=_model_for(cfg), chip=chip, cal=cal)
        model = _model_for(cfg)
        return ProbeResult(
            worker_id=worker_id,
            chip_class=chip.chip_class,
            config_name=config_name,
            t_step_median_s=t_step,
            # tokens/sec, never steps/sec — steps/sec rewards doing less work.
            tokens_per_s=cfg.global_batch_tokens / t_step,
            peak_vram_gb=predict_peak_vram_gb(cfg=cfg, model=model, cal=cal),
            gpu_util=predict_gpu_util(cfg=cfg, model=model, chip=chip),
            t_sync_s=predict_t_sync(cfg=cfg, model=model, net=self.net, cal=cal),
        )

    def fits(self, cfg: JobConfig, chip: ChipSpec) -> bool:
        """Would this OOM? The batch search needs an answer before it tries.

        85% is the handbook's target headroom: leave room for fragmentation and
        the odd transient allocation rather than filling the card.
        """
        return (
            predict_peak_vram_gb(cfg=cfg, model=_model_for(cfg), cal=self.calibration_for(chip))
            <= chip.vram_gb * 0.85
        )


def _model_for(cfg: JobConfig) -> ModelSpec:
    from gpushare.agent.specs import MODELS

    try:
        return MODELS[cfg.model]
    except KeyError:
        raise KeyError(
            f"unknown model {cfg.model!r}. JobConfig.model is a free string by "
            f"contract; add a ModelSpec to specs.MODELS. known: {sorted(MODELS)}"
        ) from None


# ─────────────────────────────────────────────────────────────────────────────
# Projection — measure short, convert long, say which is which
# ─────────────────────────────────────────────────────────────────────────────
def project_runtime_s(probe: ProbeResult, *, total_steps: int, H: int) -> float:
    """ETA from a probe. ALWAYS label the result 'projected' on screen.

    Presenting a projection as a measurement is the one item on the handbook's
    cheat list marked fatal.
    """
    return total_steps * probe.t_step_median_s + (total_steps / H) * probe.t_sync_s


def project_credits(
    probe: ProbeResult, *, total_steps: int, H: int, rate_per_hour: float, n_workers: int
) -> float:
    gpu_hours = project_runtime_s(probe, total_steps=total_steps, H=H) / 3600 * n_workers
    return gpu_hours * rate_per_hour
