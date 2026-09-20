"""Real probe. Runs ON a GPU pod and emits one ProbeResult per config.

This is the measurement half of the Prober seam in agent/simulate.py: SimProber
predicts these numbers, this script measures them, and calibrate.observe() turns
the difference into the per-chip constants. Until this runs, every performance
number in the repo is a cold-start guess.

    python scripts/probe.py                       # baseline + optimized
    python scripts/probe.py --steps 40 --warmup 10

NOTE ON OWNERSHIP: ticket T-6 (probe runner) is Ethan's, and this overlaps it.
Written here because calibration is blocked without probe data and T-6 hasn't
started. It is deliberately NOT a training loop — no DiLoCo, no optimizer state
sharing, no checkpoints. If Ethan's T-6 lands, delete this and point the
calibration at that instead; the JSON on stdout is the whole interface.

MEASUREMENT PROTOCOL (handbook appendix, and it is not negotiable)
  - drop the first `warmup` steps: cudnn autotune, allocator warmup, compile
  - take the MEDIAN of the rest, never the mean — one outlier swings a mean
  - compare on tokens/sec, never steps/sec: steps/sec rewards doing less work
  - identical seed and data order across configs, or the comparison is void
"""

from __future__ import annotations

import argparse
import math
import statistics
import sys
import time

import torch
import torch.nn as nn
import torch.nn.functional as F

from gpushare.contracts import ProbeResult, classify_chip, emit

# Matches specs.MODELS["nanogpt-124m"]. These five numbers are the entire
# contract with the cost model — if they drift, the comparison is meaningless.
LAYERS, D_MODEL, N_HEADS, VOCAB = 12, 768, 12, 50257
SEED = 1337


class Block(nn.Module):
    def __init__(self, d: int, h: int, sdpa: bool):
        super().__init__()
        self.h, self.sdpa = h, sdpa
        self.ln1, self.ln2 = nn.LayerNorm(d), nn.LayerNorm(d)
        self.qkv = nn.Linear(d, 3 * d)
        self.proj = nn.Linear(d, d)
        self.mlp = nn.Sequential(nn.Linear(d, 4 * d), nn.GELU(), nn.Linear(4 * d, d))

    def attn(self, x):
        B, T, C = x.shape
        q, k, v = self.qkv(x).split(C, dim=2)
        q, k, v = (t.view(B, T, self.h, C // self.h).transpose(1, 2) for t in (q, k, v))
        if self.sdpa:
            # Never materialises the T x T matrix. That absence is the whole
            # memory case for lever 2, and this branch is what proves it.
            y = F.scaled_dot_product_attention(q, k, v, is_causal=True)
        else:
            att = (q @ k.transpose(-2, -1)) / math.sqrt(k.size(-1))
            mask = torch.triu(torch.ones(T, T, device=x.device, dtype=torch.bool), 1)
            att = att.masked_fill(mask, float("-inf")).softmax(dim=-1)
            y = att @ v
        return self.proj(y.transpose(1, 2).contiguous().view(B, T, C))

    def forward(self, x):
        x = x + self.attn(self.ln1(x))
        return x + self.mlp(self.ln2(x))


class GPT(nn.Module):
    def __init__(self, seq_len: int, sdpa: bool):
        super().__init__()
        self.wte = nn.Embedding(VOCAB, D_MODEL)
        self.wpe = nn.Embedding(seq_len, D_MODEL)
        self.blocks = nn.ModuleList(Block(D_MODEL, N_HEADS, sdpa) for _ in range(LAYERS))
        self.lnf = nn.LayerNorm(D_MODEL)
        self.head = nn.Linear(D_MODEL, VOCAB, bias=False)
        self.head.weight = self.wte.weight  # tied, as in GPT-2

    def forward(self, idx):
        pos = torch.arange(idx.size(1), device=idx.device)
        x = self.wte(idx) + self.wpe(pos)
        for b in self.blocks:
            x = b(x)
        logits = self.head(self.lnf(x))
        return F.cross_entropy(logits.view(-1, VOCAB), idx.view(-1), ignore_index=-1)


def probe(
    *,
    dtype: str,
    attention: str,
    micro_batch: int,
    grad_accum: int,
    seq_len: int,
    steps: int,
    warmup: int,
    config_name: str,
    worker_id: str,
) -> ProbeResult:
    """One config, measured. Rebuilds the model each time so allocator state
    from a previous config can't leak into this one's peak-memory number."""
    torch.manual_seed(SEED)
    dev = torch.device("cuda")
    td = {"bf16": torch.bfloat16, "fp16": torch.float16, "fp32": torch.float32}[dtype]

    model = GPT(seq_len, sdpa=(attention == "sdpa")).to(dev)
    opt = torch.optim.AdamW(model.parameters(), lr=3e-4)
    scaler = torch.amp.GradScaler("cuda") if dtype == "fp16" else None

    # Same data every config. Different data order makes the loss comparison,
    # and therefore the whole before/after claim, meaningless.
    g = torch.Generator(device="cpu").manual_seed(SEED)
    batch = torch.randint(VOCAB, (micro_batch, seq_len), generator=g).to(dev)

    torch.cuda.reset_peak_memory_stats()
    times: list[float] = []

    for _ in range(steps):
        torch.cuda.synchronize()
        t0 = time.perf_counter()
        opt.zero_grad(set_to_none=True)
        for _ in range(grad_accum):  # accumulation multiplies work, not memory
            with torch.autocast("cuda", dtype=td, enabled=(dtype != "fp32")):
                loss = model(batch) / grad_accum
            (scaler.scale(loss) if scaler else loss).backward()
        if scaler:
            scaler.step(opt)
            scaler.update()
        else:
            opt.step()
        torch.cuda.synchronize()
        times.append(time.perf_counter() - t0)

    kept = times[warmup:]
    t_step = statistics.median(kept)
    tokens_per_step = micro_batch * grad_accum * seq_len

    p = torch.cuda.get_device_properties(0)
    return ProbeResult(
        worker_id=worker_id,
        chip_class=classify_chip(
            gpu_name=p.name,
            cc=float(f"{p.major}.{p.minor}"),
            vram_gb=round(p.total_memory / 1e9, 1),
            is_amd=torch.version.hip is not None,
        ),
        config_name=config_name,
        t_step_median_s=t_step,
        tokens_per_s=tokens_per_step / t_step,
        peak_vram_gb=torch.cuda.max_memory_allocated() / 1e9,
        gpu_util=_util(),
        t_sync_s=0.0,  # single GPU. Pool sync is a separate measurement.
    )


def _util() -> float:
    """Measured if pynvml is present, 0.0 if not — never guessed. A fabricated
    utilisation would poison the one field the cost model already admits is weak."""
    try:
        import pynvml

        pynvml.nvmlInit()
        h = pynvml.nvmlDeviceGetHandleByIndex(0)
        return pynvml.nvmlDeviceGetUtilizationRates(h).gpu / 100.0
    except Exception:
        return 0.0


# Baseline is what a beginner would plausibly set, NOT fp32 + batch 1. A
# deliberately terrible baseline produces a dramatic number and dies to
# "nobody configures it that way".
CONFIGS = {
    "baseline": dict(dtype="fp32", attention="eager", micro_batch=8, grad_accum=4),
    "optimized": dict(dtype="bf16", attention="sdpa", micro_batch=32, grad_accum=1),
}


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--steps", type=int, default=40)
    ap.add_argument("--warmup", type=int, default=10)
    ap.add_argument("--seq-len", type=int, default=1024)
    ap.add_argument("--worker-id", default="runpod-4090")
    ap.add_argument("--only", choices=sorted(CONFIGS), help="run just one config")
    a = ap.parse_args()

    if not torch.cuda.is_available():
        raise SystemExit("no GPU visible — run `make check` first")

    names = [a.only] if a.only else list(CONFIGS)
    for name in names:
        # Human-readable progress to STDERR. Events go to stdout, and a stray
        # print there would break whatever parses this.
        print(f"probing {name} ({a.steps} steps, {a.warmup} warmup)...", file=sys.stderr)
        r = probe(
            **CONFIGS[name],
            seq_len=a.seq_len,
            steps=a.steps,
            warmup=a.warmup,
            config_name=name,
            worker_id=a.worker_id,
        )
        emit(r)
        print(
            f"  {r.t_step_median_s:.4f}s/step  {r.tokens_per_s:,.0f} tok/s  "
            f"{r.peak_vram_gb:.2f} GB",
            file=sys.stderr,
        )


if __name__ == "__main__":
    main()
