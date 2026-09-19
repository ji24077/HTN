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

Saves a Hugging Face full-model checkpoint or a PEFT adapter plus meta.json.
An adapter requires the original base model; it is not a drop-in full-weight
migration checkpoint. Optimizer state is not saved by this experiment runner.
"""

from __future__ import annotations

import argparse
import hashlib
import json
import logging
import math
import statistics
import sys
import time
from pathlib import Path

import torch

from gpushare.agent.task import MODEL_ID, PROMPT, Record, build_example
from gpushare.contracts import TrainStep, emit
from gpushare.trainer.sft import (
    prepare_batch,
    standard_loss_sum,
    supervised_tokens,
    target_loss_sum,
)

# Events go to stdout; everything human goes to stderr. A stray print on stdout
# breaks whatever parses the event stream.
logging.basicConfig(stream=sys.stderr, level=logging.INFO, format="%(message)s")
log = logging.getLogger("train")

DTYPES = {"bf16": torch.bfloat16, "fp16": torch.float16, "fp32": torch.float32}


class TrainingSampler:
    """A deterministic example stream independent of micro-batch partitioning."""

    def __init__(self, rows: int, *, seed: int, sampling: str = "shuffle"):
        if rows < 1 or sampling not in {"shuffle", "replacement"}:
            raise ValueError("sampler requires positive rows and shuffle or replacement sampling")
        self.rows = rows
        self.sampling = sampling
        self.generator = torch.Generator().manual_seed(seed)
        self.pending = torch.empty(0, dtype=torch.long)
        self.seen = torch.zeros(rows, dtype=torch.bool)

    def draw(self, count: int) -> torch.Tensor:
        if count < 1:
            raise ValueError("sample count must be positive")
        if self.sampling == "replacement":
            indices = torch.randint(self.rows, (count,), generator=self.generator)
        else:
            chunks = []
            remaining = count
            while remaining:
                if not len(self.pending):
                    self.pending = torch.randperm(self.rows, generator=self.generator)
                take = min(remaining, len(self.pending))
                chunks.append(self.pending[:take])
                self.pending = self.pending[take:]
                remaining -= take
            indices = torch.cat(chunks)
        self.seen[indices] = True
        return indices

    @property
    def unique_examples_seen(self) -> int:
        return int(self.seen.sum())


def validate_init_adapter(path: Path | None, *, method: str) -> dict | None:
    """Reject incompatible warm starts before loading weights or using the GPU."""
    if path is None:
        return None
    if method != "lora":
        raise ValueError("--init-adapter requires --method lora")
    config_path = path / "adapter_config.json"
    if not config_path.is_file():
        raise ValueError("--init-adapter must be a local directory containing adapter_config.json")
    try:
        config = json.loads(config_path.read_text(encoding="utf-8"))
    except (OSError, json.JSONDecodeError) as exc:
        raise ValueError(f"cannot read adapter_config.json: {exc}") from exc
    if not isinstance(config, dict) or config.get("peft_type") != "LORA":
        raise ValueError("--init-adapter requires a LoRA adapter")
    if config.get("base_model_name_or_path") != MODEL_ID:
        raise ValueError(f"--init-adapter base model must be {MODEL_ID}")
    if config.get("task_type") != "CAUSAL_LM":
        raise ValueError("--init-adapter must use the CAUSAL_LM task")
    if not any(
        (path / filename).is_file()
        for filename in ("adapter_model.safetensors", "adapter_model.bin")
    ):
        raise ValueError("--init-adapter is missing local adapter weights")
    return config


def configure_lora(model, *, rank: int, init_adapter: Path | None = None):
    """Create trainable adapters, or load existing adapter weights for more SFT."""
    from peft import LoraConfig, PeftModel, TaskType, get_peft_model

    if init_adapter is not None:
        return PeftModel.from_pretrained(
            model, init_adapter, is_trainable=True, local_files_only=True
        )
    return get_peft_model(
        model,
        LoraConfig(
            task_type=TaskType.CAUSAL_LM,
            r=rank,
            lora_alpha=2 * rank,
            lora_dropout=0.0,
            target_modules=[
                "q_proj",
                "k_proj",
                "v_proj",
                "o_proj",
                "gate_proj",
                "up_proj",
                "down_proj",
            ],
        ),
    )


def load_rows(path: Path) -> list[tuple[str, str]]:
    rows = []
    for line in path.read_text(encoding="utf-8").splitlines():
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
        raise SystemExit(
            f"{dropped} rows exceed seq_len={seq_len}; raise --seq-len rather than silently changing the dataset"
        )
    if not ids:
        raise SystemExit(f"every row exceeded seq_len={seq_len} — raise it")
    return torch.tensor(ids), torch.tensor(labels)


def token_work_summary(
    *,
    nominal_tokens_per_step: int,
    step_times_s: list[float],
    answer_tokens_per_step: list[int],
    input_tokens_per_step: list[int],
    processed_positions_per_step: list[int],
    warmup_measure: int,
) -> dict:
    """Describe useful tokens and padded decoder positions separately.

    A decoder forward processes the rectangular input tensor, including any
    padding remaining within a micro-batch. Answer-only projection processes
    just supervised next-token positions. Counts describe one forward's logical
    positions, not FLOPs or repeated work from activation checkpointing.
    Rates use total measured work / total measured time after warmup; dividing
    a fixed sequence capacity by median step time overstates useful throughput.
    """
    counts = (answer_tokens_per_step, input_tokens_per_step, processed_positions_per_step)
    if any(len(values) != len(step_times_s) for values in counts):
        raise ValueError("token counts and step times must have equal lengths")
    if not 0 <= warmup_measure < len(step_times_s):
        raise ValueError("warmup must leave at least one measured step")
    if any(not math.isfinite(dt) or dt <= 0 for dt in step_times_s):
        raise ValueError("step times must be finite and positive")
    if any(
        not 0 < answer <= inputs <= processed <= nominal_tokens_per_step
        for answer, inputs, processed in zip(*counts, strict=True)
    ):
        raise ValueError("expected 0 < answer <= input <= processed <= nominal token counts")
    measured = slice(warmup_measure, None)
    seconds = sum(step_times_s[measured])
    return {
        "token_accounting_version": 2,
        # Preserve the original capacity field for existing calibration readers.
        "tokens_per_step": nominal_tokens_per_step,
        "nominal_tokens_per_step": nominal_tokens_per_step,
        "train_step_tokens": "input_tokens",
        "token_accounting": {
            "tokens_per_step": "nominal micro_batch * grad_accum * seq_len capacity",
            "answer_tokens_per_step": "supervised next-token targets, including answer EOS",
            "input_tokens_per_step": "non-padding prompt plus answer tokens, including answer EOS",
            "processed_positions_per_step": "decoder tensor positions after optional trimming, including residual padding; excludes checkpoint recomputation",
            "train_step_tokens": "TrainStep.tokens contains input_tokens_per_step",
            "throughput": "sum of measured counts / sum of step times, warmup excluded",
        },
        "answer_tokens_per_step": answer_tokens_per_step,
        "input_tokens_per_step": input_tokens_per_step,
        "processed_positions_per_step": processed_positions_per_step,
        "answer_tokens_per_second": sum(answer_tokens_per_step[measured]) / seconds,
        "input_tokens_per_second": sum(input_tokens_per_step[measured]) / seconds,
        "processed_positions_per_second": sum(processed_positions_per_step[measured]) / seconds,
        "measured_padding_fraction": 1
        - sum(input_tokens_per_step[measured]) / sum(processed_positions_per_step[measured]),
    }


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--data", type=Path, default=Path("data/train.jsonl"))
    ap.add_argument("--out", type=Path, default=Path("ckpt/run"))
    ap.add_argument("--steps", type=int, default=500)
    ap.add_argument("--warmup-measure", type=int, default=10, help="steps dropped from timing")
    ap.add_argument("--seq-len", type=int, default=192)
    ap.add_argument("--lr", type=float, help="default: 1e-5 full, 2e-4 LoRA")
    ap.add_argument("--method", choices=("full", "lora"), default="full")
    ap.add_argument(
        "--init-adapter",
        type=Path,
        help="local LoRA weights to continue training; optimizer state starts fresh",
    )
    ap.add_argument("--lora-rank", type=int, default=8)
    ap.add_argument(
        "--gradient-checkpointing", action=argparse.BooleanOptionalAction, default=False
    )
    ap.add_argument("--loss", choices=("target-only", "standard"), default="target-only")
    ap.add_argument("--loss-chunk", type=int, default=64)
    ap.add_argument("--trim-padding", action=argparse.BooleanOptionalAction, default=True)
    ap.add_argument("--save-model", "--save", action=argparse.BooleanOptionalAction, default=True)
    ap.add_argument("--seed", type=int, default=1337)
    ap.add_argument(
        "--sampling",
        choices=("shuffle", "replacement"),
        default="shuffle",
        help="shuffled epochs cover every example; replacement reproduces older runs",
    )
    # The agent's levers.
    ap.add_argument("--dtype", choices=sorted(DTYPES), default="bf16")
    ap.add_argument("--attention", choices=("sdpa", "eager"), default="sdpa")
    ap.add_argument("--micro-batch", type=int, default=2)
    ap.add_argument("--grad-accum", type=int, default=8)
    ap.add_argument("--worker-id", default="w1")
    ap.add_argument("--job-id", default="j1")
    ap.add_argument("--config-name", default="optimized")
    a = ap.parse_args()
    if min(a.steps, a.micro_batch, a.grad_accum, a.seq_len, a.loss_chunk, a.lora_rank) < 1:
        ap.error("steps, batch sizes, sequence length, chunk size and rank must be positive")
    if not 0 <= a.warmup_measure < a.steps:
        ap.error("warmup-measure must be nonnegative and smaller than steps")
    if a.lr is not None and a.lr <= 0:
        ap.error("learning rate must be positive")
    try:
        init_adapter_config = validate_init_adapter(a.init_adapter, method=a.method)
    except ValueError as exc:
        ap.error(str(exc))
    if a.out.exists() and any(a.out.iterdir()):
        ap.error("output directory is not empty; choose a new run directory")
    a.lr = a.lr if a.lr is not None else (2e-4 if a.method == "lora" else 1e-5)

    if not torch.cuda.is_available():
        raise SystemExit("no GPU visible — run `make check` first")

    from transformers import AutoModelForCausalLM, AutoTokenizer, set_seed

    set_seed(a.seed)
    if a.dtype == "bf16" and not torch.cuda.is_bf16_supported():
        ap.error("this GPU does not support bf16; use fp16 or fp32")

    tok = AutoTokenizer.from_pretrained(MODEL_ID)
    if tok.pad_token_id is None:
        tok.pad_token = tok.eos_token

    ids, labels = encode(tok, load_rows(a.data), a.seq_len)
    log.info(f"  {len(ids)} examples, seq_len {a.seq_len}")

    log.info(f"loading {MODEL_ID} ({a.dtype}, {a.attention}, {a.method})...")
    # GradScaler requires fp32 trainable parameters; autocast handles fp16 math.
    weight_dtype = torch.float32 if a.dtype == "fp16" else DTYPES[a.dtype]
    model = AutoModelForCausalLM.from_pretrained(
        MODEL_ID, dtype=weight_dtype, attn_implementation=a.attention
    ).cuda()
    model.config.use_cache = False
    if a.method == "lora":
        model = configure_lora(model, rank=a.lora_rank, init_adapter=a.init_adapter)
        if a.init_adapter:
            log.info(f"  warm start from {a.init_adapter}: adapter weights only, fresh optimizer")
    if a.gradient_checkpointing:
        model.gradient_checkpointing_enable(gradient_checkpointing_kwargs={"use_reentrant": False})
    else:
        model.gradient_checkpointing_disable()
    model.train()
    trainable = [p for p in model.parameters() if p.requires_grad]
    log.info(f"  trainable parameters: {sum(p.numel() for p in trainable):,}")
    # foreach=False avoids AdamW's additional tensor-list-sized peak allocation.
    opt = torch.optim.AdamW(trainable, lr=a.lr, foreach=False)
    scaler = torch.amp.GradScaler("cuda") if a.dtype == "fp16" else None

    tokens_per_step = a.micro_batch * a.grad_accum * a.seq_len
    sampler = TrainingSampler(len(ids), seed=a.seed, sampling=a.sampling)
    torch.cuda.reset_peak_memory_stats()
    times: list[float] = []
    answer_counts: list[int] = []
    input_counts: list[int] = []
    processed_counts: list[int] = []
    loss_history: list[float] = []

    log.info(f"training {a.steps} steps, {tokens_per_step:,} nominal token positions/step")
    for step in range(a.steps):
        torch.cuda.synchronize()
        t0 = time.perf_counter()

        opt.zero_grad(set_to_none=True)
        # Draw the whole effective batch once: changing the micro-batch partition
        # does not change sample order, and answer lengths determine normalization.
        indices = sampler.draw(a.micro_batch * a.grad_accum)
        denominator = supervised_tokens(labels[indices])
        total = torch.zeros((), device="cuda")
        input_count = 0
        processed_count = 0
        for idx in indices.split(a.micro_batch):
            x, y, mask = prepare_batch(ids[idx], labels[idx], trim=a.trim_padding)
            input_count += int(mask.sum())
            processed_count += x.numel()
            x, y, mask = x.cuda(), y.cuda(), mask.cuda()
            with torch.autocast("cuda", dtype=DTYPES[a.dtype], enabled=a.dtype != "fp32"):
                loss_sum = (
                    target_loss_sum(model, x, y, mask, chunk_size=a.loss_chunk)
                    if a.loss == "target-only"
                    else standard_loss_sum(model, x, y, mask)
                )
                loss = loss_sum / denominator
            (scaler.scale(loss) if scaler else loss).backward()
            total += loss.detach()
        if scaler:
            scaler.step(opt)
            scaler.update()
        else:
            opt.step()

        torch.cuda.synchronize()
        dt = time.perf_counter() - t0
        times.append(dt)
        answer_counts.append(denominator)
        input_counts.append(input_count)
        processed_counts.append(processed_count)
        total = float(total)
        if not torch.isfinite(torch.tensor(total)):
            raise SystemExit("non-finite training loss; run aborted without saving weights")
        loss_history.append(total)
        emit(
            TrainStep(
                worker_id=a.worker_id, step=step, loss=total, step_time_s=dt, tokens=input_count
            )
        )
        if step % 25 == 0 or step == a.steps - 1:
            log.info(f"  step {step:>4}  loss {total:.4f}  {dt:.3f}s")

    # Median with warmup dropped — the same protocol the cost model is fitted
    # against. A mean here would let one slow step move the calibration.
    t_step = statistics.median(times[a.warmup_measure :])
    peak = torch.cuda.max_memory_allocated() / 1e9
    token_work = token_work_summary(
        nominal_tokens_per_step=tokens_per_step,
        step_times_s=times,
        answer_tokens_per_step=answer_counts,
        input_tokens_per_step=input_counts,
        processed_positions_per_step=processed_counts,
        warmup_measure=a.warmup_measure,
    )

    a.out.mkdir(parents=True, exist_ok=True)
    if a.save_model:
        # HF handles tied embedding weights; PEFT saves only trainable adapters.
        model.save_pretrained(a.out, safe_serialization=True)
        tok.save_pretrained(a.out)
    meta = {
        "model_id": MODEL_ID,
        "steps": a.steps,
        "config_name": a.config_name,
        "dtype": a.dtype,
        "attention": a.attention,
        "micro_batch": a.micro_batch,
        "grad_accum": a.grad_accum,
        "seq_len": a.seq_len,
        **token_work,
        "final_loss": total,
        "t_step_median_s": t_step,
        "peak_vram_gb": peak,
        "gpu": torch.cuda.get_device_properties(0).name,
        "method": a.method,
        "loss": a.loss,
        "loss_chunk": a.loss_chunk,
        "gradient_checkpointing": a.gradient_checkpointing,
        "trim_padding": a.trim_padding,
        "seed": a.seed,
        "sampling": a.sampling,
        "examples_drawn": a.steps * a.micro_batch * a.grad_accum,
        "unique_examples_seen": sampler.unique_examples_seen,
        "dataset_coverage": sampler.unique_examples_seen / len(ids),
        "init_adapter": a.init_adapter.as_posix() if a.init_adapter else None,
        "initialization": "adapter_weights_only" if a.init_adapter else "base_model",
        "optimizer_initialized_from_checkpoint": False,
        "lr": a.lr,
        "lora_rank": model.peft_config["default"].r if a.method == "lora" else None,
        "init_adapter_config": init_adapter_config,
        "trainable_parameters": sum(p.numel() for p in trainable),
        "total_parameters": sum(p.numel() for p in model.parameters()),
        "data_sha256": hashlib.sha256(a.data.read_bytes()).hexdigest(),
        "data_rows": len(ids),
        "warmup_measure": a.warmup_measure,
        "step_times_s": times,
        "loss_history": loss_history,
        "peak_reserved_gb": torch.cuda.max_memory_reserved() / 1e9,
        "checkpoint_saved": a.save_model,
        "torch_version": torch.__version__,
    }
    (a.out / "meta.json").write_text(json.dumps(meta, indent=2), encoding="utf-8")

    log.info(
        f"\n{a.config_name}: {t_step:.4f}s/step  "
        f"{token_work['input_tokens_per_second']:,.0f} input tok/s  "
        f"{token_work['answer_tokens_per_second']:,.0f} answer tok/s  {peak:.2f} GB peak\n"
        f"  saved run metadata in {a.out} (weights: {a.save_model})"
    )
    log.info(
        f"  nominal capacity/optimizer step: {tokens_per_step:,}; "
        f"decoder positions/s: {token_work['processed_positions_per_second']:,.0f}; "
        f"residual padding: {token_work['measured_padding_fraction']:.1%}"
    )


if __name__ == "__main__":
    _ = PROMPT  # imported so a change to the prompt format fails loudly here too
    main()
