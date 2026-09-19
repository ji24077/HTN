"""Ji — SFT Qwen2.5-0.5B on the extraction task. Runs ON a GPU pod.

    uv run python scripts/train.py --steps 500 --out ckpt/run1
    uv run python scripts/train.py --dtype fp32 --attention eager --micro-batch 4 \
        --grad-accum 8 --steps 500 --out ckpt/baseline

THIS IS ALSO THE PROBE. It reports median step time and peak VRAM on the real
workload, so the cost model is calibrated against the thing we actually run
rather than a synthetic stand-in. scripts/probe.py measured a nanoGPT that
nobody trains; this measures Qwen doing the task the metric judges.

EVERY KNOB THE AGENT CAN TURN IS A FLAG HERE. dtype, attention, micro_batch,
grad_accum. "Optimize training speed" means the agent picks different values
for these and the numbers move — if a lever weren't wired through to here, the
agent would be deciding something that has no effect.

Saves safetensors + meta.json per contracts.ckpt_path. Weights only: migration
happens at a point where global weights are the entire shared state, so there
is no optimizer state to carry.
"""

from __future__ import annotations

import argparse
import json
import logging
import statistics
import sys
import time
from pathlib import Path

import torch
from safetensors.torch import save_file

from gpushare.agent.task import MODEL_ID, PROMPT, Record, build_example
from gpushare.contracts import TrainStep, emit

# Events go to stdout; everything human goes to stderr. A stray print on stdout
# breaks whatever parses the event stream.
logging.basicConfig(stream=sys.stderr, level=logging.INFO, format="%(message)s")
log = logging.getLogger("train")

DTYPES = {"bf16": torch.bfloat16, "fp16": torch.float16, "fp32": torch.float32}


def load_rows(path: Path) -> list[tuple[str, str]]:
    rows = []
    for line in path.read_text().splitlines():
        if not line.strip():
            continue
        d = json.loads(line)
        rows.append(build_example(d["sentence"], Record.model_validate(d["record"])))
    if not rows:
        raise SystemExit(f"no rows in {path} — run scripts/gen_data.py first")
    return rows


def encode(tok, rows: list[tuple[str, str]], seq_len: int):
    """Tokenize to fixed length, masking loss to the TARGET only.

    Training on the prompt tokens as well would spend most of the gradient
    teaching the model to reproduce input sentences, which is not the task and
    dilutes the signal that produces valid JSON.
    """
    ids, labels = [], []
    dropped = 0
    for prompt, target in rows:
        p = tok(prompt, add_special_tokens=False).input_ids
        t = tok(target + tok.eos_token, add_special_tokens=False).input_ids
        if len(p) + len(t) > seq_len:
            dropped += 1
            continue
        pad = seq_len - len(p) - len(t)
        ids.append(p + t + [tok.pad_token_id] * pad)
        # -100 is ignored by cross_entropy: prompt and padding contribute nothing.
        labels.append([-100] * len(p) + t + [-100] * pad)
    if dropped:
        log.info(f"  dropped {dropped} rows longer than seq_len={seq_len}")
    if not ids:
        raise SystemExit(f"every row exceeded seq_len={seq_len} — raise it")
    return torch.tensor(ids), torch.tensor(labels)


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--data", type=Path, default=Path("data/train.jsonl"))
    ap.add_argument("--out", type=Path, default=Path("ckpt/run"))
    ap.add_argument("--steps", type=int, default=500)
    ap.add_argument("--warmup-measure", type=int, default=10, help="steps dropped from timing")
    ap.add_argument("--seq-len", type=int, default=192)
    ap.add_argument("--lr", type=float, default=1e-5)
    # The agent's levers.
    ap.add_argument("--dtype", choices=sorted(DTYPES), default="bf16")
    ap.add_argument("--attention", choices=("sdpa", "eager"), default="sdpa")
    ap.add_argument("--micro-batch", type=int, default=16)
    ap.add_argument("--grad-accum", type=int, default=1)
    ap.add_argument("--worker-id", default="w1")
    ap.add_argument("--job-id", default="j1")
    ap.add_argument("--config-name", default="optimized")
    a = ap.parse_args()

    if not torch.cuda.is_available():
        raise SystemExit("no GPU visible — run `make check` first")

    from transformers import AutoModelForCausalLM, AutoTokenizer

    tok = AutoTokenizer.from_pretrained(MODEL_ID)
    if tok.pad_token_id is None:
        tok.pad_token = tok.eos_token

    log.info(f"loading {MODEL_ID} ({a.dtype}, {a.attention})...")
    model = AutoModelForCausalLM.from_pretrained(
        MODEL_ID, dtype=DTYPES[a.dtype], attn_implementation=a.attention
    ).cuda()
    model.gradient_checkpointing_disable()
    model.train()

    ids, labels = encode(tok, load_rows(a.data), a.seq_len)
    log.info(f"  {len(ids)} examples, seq_len {a.seq_len}")

    opt = torch.optim.AdamW(model.parameters(), lr=a.lr)
    scaler = torch.amp.GradScaler("cuda") if a.dtype == "fp16" else None

    tokens_per_step = a.micro_batch * a.grad_accum * a.seq_len
    g = torch.Generator().manual_seed(1337)  # same data order across configs
    torch.cuda.reset_peak_memory_stats()
    times: list[float] = []

    log.info(f"training {a.steps} steps, {tokens_per_step:,} tokens/step")
    for step in range(a.steps):
        torch.cuda.synchronize()
        t0 = time.perf_counter()

        opt.zero_grad(set_to_none=True)
        total = 0.0
        for _ in range(a.grad_accum):
            idx = torch.randint(len(ids), (a.micro_batch,), generator=g)
            x, y = ids[idx].cuda(), labels[idx].cuda()
            out = model(input_ids=x, labels=y)
            loss = out.loss / a.grad_accum
            (scaler.scale(loss) if scaler else loss).backward()
            total += loss.item()
        if scaler:
            scaler.step(opt)
            scaler.update()
        else:
            opt.step()

        torch.cuda.synchronize()
        dt = time.perf_counter() - t0
        times.append(dt)
        emit(
            TrainStep(
                worker_id=a.worker_id, step=step, loss=total, step_time_s=dt, tokens=tokens_per_step
            )
        )
        if step % 25 == 0 or step == a.steps - 1:
            log.info(f"  step {step:>4}  loss {total:.4f}  {dt:.3f}s")

    # Median with warmup dropped — the same protocol the cost model is fitted
    # against. A mean here would let one slow step move the calibration.
    t_step = statistics.median(times[a.warmup_measure :])
    peak = torch.cuda.max_memory_allocated() / 1e9

    a.out.mkdir(parents=True, exist_ok=True)
    save_file(
        {k: v.contiguous().cpu() for k, v in model.state_dict().items()},
        str(a.out / "model.safetensors"),
    )
    meta = {
        "model_id": MODEL_ID,
        "steps": a.steps,
        "config_name": a.config_name,
        "dtype": a.dtype,
        "attention": a.attention,
        "micro_batch": a.micro_batch,
        "grad_accum": a.grad_accum,
        "seq_len": a.seq_len,
        "tokens_per_step": tokens_per_step,
        "final_loss": total,
        "t_step_median_s": t_step,
        "peak_vram_gb": peak,
        "gpu": torch.cuda.get_device_properties(0).name,
    }
    (a.out / "meta.json").write_text(json.dumps(meta, indent=2))

    log.info(
        f"\n{a.config_name}: {t_step:.4f}s/step  "
        f"{tokens_per_step / t_step:,.0f} tok/s  {peak:.2f} GB peak\n"
        f"  saved {a.out}/model.safetensors"
    )
    # The invariant the whole before/after claim rests on. State it every run so
    # a config that quietly does less work cannot pass unnoticed.
    log.info(f"  tokens/optimizer step: {tokens_per_step:,}")


if __name__ == "__main__":
    _ = PROMPT  # imported so a change to the prompt format fails loudly here too
    main()
