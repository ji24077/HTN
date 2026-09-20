"""Ji — tests for the training cost model.

These aren't testing that the numbers are RIGHT (only hardware can say that).
They test that the model has the properties the agent's rules depend on: if
bf16 weren't faster than fp32 here, the chip agent would be tuning against
fiction and we'd never know until the demo.
"""

import pytest

from gpushare.agent import calibrate
from gpushare.agent.calibrate import CalibrationStore, error_report, observe
from gpushare.agent.router import compute_H
from gpushare.agent.simulate import (
    SimProber,
    predict_peak_vram_gb,
    predict_t_step,
    predict_t_sync,
    project_runtime_s,
)
from gpushare.agent.specs import CHIPS, MODELS, NETS
from gpushare.contracts import JobConfig

SEQ = 1024


def cfg(**over) -> JobConfig:
    """A valid 2-worker config. JobConfig itself enforces the fixed-work invariant,
    so any override that breaks it fails here rather than silently later."""
    base = dict(
        job_id="j1",
        model="nanogpt-124m",
        total_steps=8000,
        seq_len=SEQ,
        dtype="bf16",
        attention="sdpa",
        micro_batch=16,
        grad_accum=1,
        global_batch_tokens=16 * 1 * 2 * SEQ,
        H=190,
        workers=["w1", "w2"],
    )
    return JobConfig(**{**base, **over})


# ─────────────────────────────────────────────────────────────────────────────
# Properties the chip agent's rules rest on
# ─────────────────────────────────────────────────────────────────────────────
def test_bf16_beats_fp32():
    """Lever 1. If this ever inverts, the whole optimization story is wrong."""
    chip, model, cal = CHIPS["RTX 4090"], MODELS["nanogpt-124m"], calibrate.COLD_START
    fast = predict_t_step(cfg=cfg(dtype="bf16"), model=model, chip=chip, cal=cal)
    slow = predict_t_step(cfg=cfg(dtype="fp32"), model=model, chip=chip, cal=cal)
    assert slow > fast


def test_sdpa_saves_memory_and_the_saving_grows_with_seq_len():
    """Lever 2. eager materialises a seq x seq matrix; sdpa never does.

    The saving being seq_len-dependent is the point — it's why attention is a
    per-chip/per-config decision and not a constant."""
    model, cal = MODELS["nanogpt-124m"], calibrate.COLD_START

    def gap(seq):
        c = dict(seq_len=seq, micro_batch=8, grad_accum=1, global_batch_tokens=8 * 1 * 2 * seq)
        eager = predict_peak_vram_gb(cfg=cfg(attention="eager", **c), model=model, cal=cal)
        sdpa = predict_peak_vram_gb(cfg=cfg(attention="sdpa", **c), model=model, cal=cal)
        return eager - sdpa

    assert gap(512) > 0
    assert gap(2048) > gap(512) * 3  # quadratic, so 4x seq is far more than 4x


def test_bigger_micro_batch_is_faster_at_identical_work():
    """Lever 3 — the handbook calls it the largest lever, so the cost model has
    to actually express it. Same tokens per optimizer step either way; the bigger
    resident batch just keeps the card busier between kernel launches.

    The first cut of this model had t_step independent of micro_batch, which made
    the batch search a no-op in simulation while looking fine in the output table.
    This test exists because that slipped through once."""
    chip, model, cal = CHIPS["RTX 4090"], MODELS["nanogpt-124m"], calibrate.COLD_START
    small = cfg(micro_batch=4, grad_accum=8, global_batch_tokens=4 * 8 * 2 * SEQ)
    big = cfg(micro_batch=32, grad_accum=1, global_batch_tokens=32 * 1 * 2 * SEQ)
    assert small.global_batch_tokens == big.global_batch_tokens  # same work
    assert predict_t_step(cfg=big, model=model, chip=chip, cal=cal) < predict_t_step(
        cfg=small, model=model, chip=chip, cal=cal
    )


def test_sdpa_is_faster_not_only_smaller():
    """Lever 2 saves memory AND time — it skips staging the seq x seq matrix
    through HBM. A model that only credits the memory saving under-sells it."""
    chip, model, cal = CHIPS["RTX 4090"], MODELS["nanogpt-124m"], calibrate.COLD_START
    assert predict_t_step(
        cfg=cfg(attention="sdpa"), model=model, chip=chip, cal=cal
    ) < predict_t_step(cfg=cfg(attention="eager"), model=model, chip=chip, cal=cal)


def test_grad_accum_does_not_change_peak_memory():
    """Accumulation exists so a big global batch fits in a small resident one.
    A batch search that models this wrong will OOM on the real card."""
    model, cal = MODELS["nanogpt-124m"], calibrate.COLD_START
    a = predict_peak_vram_gb(
        cfg=cfg(micro_batch=8, grad_accum=4, global_batch_tokens=8 * 4 * 2 * SEQ),
        model=model,
        cal=cal,
    )
    b = predict_peak_vram_gb(
        cfg=cfg(micro_batch=8, grad_accum=1, global_batch_tokens=8 * 1 * 2 * SEQ),
        model=model,
        cal=cal,
    )
    assert a == b


def test_fixed_work_means_same_tokens_per_step():
    """Raising micro_batch while lowering grad_accum must not change the work.
    This is the invariant the before/after demo rests on."""
    slow = cfg(micro_batch=8, grad_accum=4, global_batch_tokens=8 * 4 * 2 * SEQ)
    fast = cfg(micro_batch=32, grad_accum=1, global_batch_tokens=32 * 1 * 2 * SEQ)
    assert slow.global_batch_tokens == fast.global_batch_tokens


# ─────────────────────────────────────────────────────────────────────────────
# Network -> H, the router's signature lever
# ─────────────────────────────────────────────────────────────────────────────
def test_slower_link_raises_H():
    """H must respond to MEASURED network conditions. That responsiveness is the
    entire argument for H being an agent decision rather than a constant."""
    model = MODELS["nanogpt-30m"]
    c = cfg(model="nanogpt-30m")
    cal = calibrate.COLD_START
    t_step = predict_t_step(cfg=c, model=model, chip=CHIPS["RTX 4090"], cal=cal)

    fast = predict_t_sync(cfg=c, model=model, net=NETS["lan-wired"], cal=cal)
    slow = predict_t_sync(cfg=c, model=model, net=NETS["tailscale-derp"], cal=cal)
    assert slow > fast
    assert compute_H(slow, t_step) >= compute_H(fast, t_step)


def test_124m_on_100mbit_pins_H_to_the_ceiling():
    """A real finding, discovered for free: at RunPod's 100 Mbps pod-to-pod cap a
    124M pseudo-gradient takes long enough that H clamps at 500. The honest agent
    output is 'this model is too big for this link', not a quietly clamped number.
    Costs nothing to learn here instead of on a rented GPU."""
    model, c, cal = MODELS["nanogpt-124m"], cfg(), calibrate.COLD_START
    t_sync = predict_t_sync(cfg=c, model=model, net=NETS["runpod-global"], cal=cal)
    t_step = predict_t_step(cfg=c, model=model, chip=CHIPS["RTX 4090"], cal=cal)
    assert compute_H(t_sync, t_step) == 500


# ─────────────────────────────────────────────────────────────────────────────
# Calibration is an exact inverse — no black box
# ─────────────────────────────────────────────────────────────────────────────
def test_observe_inverts_predict():
    """Feed a simulated probe back through observe() and the constants that
    produced it must come out. If this drifts, the fitted numbers mean nothing."""
    chip, c = CHIPS["RTX 3090"], cfg()
    store = CalibrationStore()
    sim = SimProber(NETS["runpod-global"], store)
    truth = store.get(chip.chip_class)

    back = observe(sim.probe(c, chip), cfg=c, model=MODELS["nanogpt-124m"], chip=chip)

    assert back.mfu == pytest.approx(truth.mfu, rel=1e-6)
    assert back.act_overhead == pytest.approx(truth.act_overhead, rel=1e-6)


def test_history_is_per_chip_class():
    """The 3090 agent must only ever see 3090 data. That separation is the moat;
    if one chip's probes leak into another's constants it stops being true."""
    store = CalibrationStore()
    store.add(
        "ampere_24gb", calibrate.Calibration(mfu=0.5, act_overhead=1.0, sync_eff=0.8, n_samples=1)
    )
    assert store.get("ampere_24gb").mfu == 0.5
    assert store.get("ada_24gb").mfu == calibrate.COLD_START.mfu
    assert store.get("ada_24gb").is_measured() is False


def test_error_is_relative_to_the_measurement_not_the_guess():
    """A projection shown without its error is just a claim.

    The denominator is the MEASUREMENT: predicting 2.2s when the card really did
    2.0s is a 10% error. Dividing by the prediction instead would flatter every
    over-estimate, which is exactly the direction we'd be tempted to round."""
    chip, c = CHIPS["RTX 4090"], cfg()
    truth = SimProber(NETS["runpod-global"]).probe(c, chip)
    over = truth.model_copy(update={"t_step_median_s": truth.t_step_median_s * 1.10})
    assert error_report(over, truth)["t_step_pct"] == pytest.approx(10.0, abs=1e-6)


def test_projection_includes_sync_cost():
    """ETA = steps x t_step + (steps / H) x t_sync. Dropping the second term is
    how you promise a runtime the network can't deliver."""
    sim = SimProber(NETS["runpod-global"])
    p = sim.probe(cfg(), CHIPS["RTX 4090"])
    assert project_runtime_s(p, total_steps=8000, H=190) > 8000 * p.t_step_median_s


def test_unknown_model_fails_loudly():
    """JobConfig.model is a free string by contract, so the registry is where a
    typo has to surface."""
    with pytest.raises(KeyError, match="unknown model"):
        SimProber(NETS["runpod-global"]).probe(cfg(model="gpt-9"), CHIPS["RTX 4090"])


def test_impossible_mfu_is_flagged_not_swallowed(caplog):
    """A fitted MFU above 1.0 means the chip beat its own peak, which it did
    not. It is the signature of a structural error in the cost model, and
    calibration would otherwise bury it in a constant that still looks like a
    plausible number. Measured for real on Qwen2.5-0.5B / RTX 4090: 2.77."""
    import logging

    from gpushare.agent.calibrate import observe
    from gpushare.contracts import ProbeResult

    chip, model = (
        CHIPS["RTX 4090"],
        MODELS["qwen2.5-0.5b"],
    )
    c = JobConfig(
        job_id="j",
        model="qwen2.5-0.5b",
        total_steps=10,
        seq_len=192,
        dtype="bf16",
        attention="sdpa",
        micro_batch=16,
        grad_accum=1,
        global_batch_tokens=16 * 1 * 1 * 192,
        H=190,
        workers=["w1"],
    )
    fast = ProbeResult(
        worker_id="w1",
        chip_class="ada_24gb",
        config_name="optimized",
        t_step_median_s=0.1374,
        tokens_per_s=3072 / 0.1374,
        peak_vram_gb=13.8,
        gpu_util=0.0,
        t_sync_s=0.0,
    )
    with caplog.at_level(logging.WARNING):
        obs = observe(fast, cfg=c, model=model, chip=chip)
    assert obs.mfu > 1.0
    assert "structurally" in caplog.text
