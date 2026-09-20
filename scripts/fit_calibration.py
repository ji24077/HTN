"""Turn measured probes into per-chip constants, and report how wrong we were.

    ssh pod 'cd /workspace/gpushare && uv run python scripts/probe.py' > probes.jsonl
    uv run python scripts/fit_calibration.py probes.jsonl

Reads ProbeResult JSON lines, inverts the cost model against each one to recover
`mfu` and `act_overhead`, and prints the simulator's error BEFORE and AFTER
fitting. The after-numbers are what may go on screen; the before-numbers are the
honest record of how far a cold start was off.
"""

from __future__ import annotations

import argparse
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).parent))  # must precede `from probe import`

from probe import CONFIGS  # noqa: E402 — needs the sys.path line above

from gpushare.agent.calibrate import CalibrationStore, error_report, observe
from gpushare.agent.simulate import SimProber
from gpushare.agent.specs import CHIPS, MODELS, NETS
from gpushare.contracts import JobConfig, parse_event

# CONFIGS is imported, never retyped: if the table here drifted from the one the
# probe actually ran, every fitted constant would be silently wrong.


def job_config(name: str, *, seq_len: int, workers: int) -> JobConfig:
    """Rebuild the exact config a probe ran, so observe() inverts the right thing."""
    c = CONFIGS[name]
    return JobConfig(
        job_id="probe",
        model="nanogpt-124m",
        total_steps=8000,
        seq_len=seq_len,
        H=190,
        workers=[f"w{i}" for i in range(workers)],
        global_batch_tokens=c["micro_batch"] * c["grad_accum"] * workers * seq_len,
        **c,
    )


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("probes", type=Path, help="JSON lines from scripts/probe.py")
    ap.add_argument("--chip", default="RTX 4090", choices=sorted(CHIPS))
    ap.add_argument("--seq-len", type=int, default=1024)
    a = ap.parse_args()

    chip, model = CHIPS[a.chip], MODELS["nanogpt-124m"]
    # Probes are single-GPU, so the config must say one worker or the invariant
    # describes a pool that was never measured.
    measured = [
        p
        for p in (parse_event(ln) for ln in a.probes.read_text().splitlines())
        if p is not None and p.type == "probe.result"
    ]
    if not measured:
        raise SystemExit(f"no probe.result lines in {a.probes}")

    store = CalibrationStore()
    sim = SimProber(NETS["runpod-global"], store)

    print(f"chip: {chip.name}  ({chip.chip_class})  peak {chip.tflops_bf16} TFLOPS bf16\n")
    print(
        f"{'config':<11}{'measured':>11}{'predicted':>11}{'error':>9}   {'peak VRAM meas/pred':>22}"
    )
    print("-" * 72)

    rows = []
    for p in measured:
        cfg = job_config(p.config_name, seq_len=a.seq_len, workers=1)
        before = sim.probe(cfg, chip, config_name=p.config_name)
        err = error_report(before, p)
        print(
            f"{p.config_name:<11}{p.t_step_median_s:>10.4f}s{before.t_step_median_s:>10.4f}s"
            f"{err['t_step_pct']:>8.1f}%   {p.peak_vram_gb:>8.2f} / {before.peak_vram_gb:.2f} GB"
        )
        obs = observe(p, cfg=cfg, model=model, chip=chip)
        store.add(chip.chip_class, obs)
        rows.append((p, cfg, obs))

    fitted = store.get(chip.chip_class)
    print(f"\nfitted from {fitted.n_samples} probe(s):")
    print(f"  mfu           {fitted.mfu:.3f}   (cold start assumed 0.350)")
    print(f"  act_overhead  {fitted.act_overhead:.3f}   (cold start assumed 1.300)")

    print(f"\n{'config':<11}{'measured':>11}{'predicted':>11}{'error':>9}   <- after fitting")
    print("-" * 56)
    for p, cfg, _ in rows:
        after = sim.probe(cfg, chip, config_name=p.config_name)
        err = error_report(after, p)
        print(
            f"{p.config_name:<11}{p.t_step_median_s:>10.4f}s{after.t_step_median_s:>10.4f}s"
            f"{err['t_step_pct']:>8.1f}%"
        )

    # The claim the whole before/after demo rests on — measured, not projected.
    by = {p.config_name: p for p, _, _ in rows}
    if {"baseline", "optimized"} <= by.keys():
        b, o = by["baseline"], by["optimized"]
        bc, oc = (
            job_config("baseline", seq_len=a.seq_len, workers=1),
            job_config("optimized", seq_len=a.seq_len, workers=1),
        )
        print(f"\nbaseline -> optimized: {b.t_step_median_s / o.t_step_median_s:.2f}x  (MEASURED)")
        print(
            f"  tokens/optimizer step  {bc.global_batch_tokens:,} -> "
            f"{oc.global_batch_tokens:,}  "
            f"{'OK - same work' if bc.global_batch_tokens == oc.global_batch_tokens else 'BROKEN'}"
        )
        print(f"  tokens/sec             {b.tokens_per_s:,.0f} -> {o.tokens_per_s:,.0f}")
        print(f"  peak VRAM              {b.peak_vram_gb:.2f} GB -> {o.peak_vram_gb:.2f} GB")


if __name__ == "__main__":
    main()
