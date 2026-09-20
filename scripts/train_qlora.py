"""Train Relay's verified Qwen 4B model as a 4-bit QLoRA adapter.

This is intentionally a separate entry point from ``scripts/train.py``.  The
existing trainer produces portable full/LoRA training bundles for the small
research model.  A 4-bit base has different loading and checkpoint semantics:
only the adapter is saved and the immutable Hugging Face base is referenced by
name.  Keeping the paths separate prevents a 4B workspace run from silently
falling back to a full fine-tune.

Run this on the NVIDIA RTX 4090 worker, never as part of local tests::

    uv run --extra cuda python scripts/train_qlora.py \
      --data data/sixseven-train.jsonl --out .runs/relay/adapter
"""

from __future__ import annotations

import argparse
import hashlib
import json
import logging
import os
import random
import shutil
import sys
import tempfile
import time
import uuid
from pathlib import Path

import torch

from gpushare.agent.sixseven import PROMPT, normalize_relay_target
from gpushare.artifacts import write_manifest
from gpushare.contracts import TrainStep, emit

QWEN_4B_MODEL = "Qwen/Qwen3-4B-Instruct-2507"
QWEN_4B_REVISION = "cdbee75f17c01a7cc42f958dc650907174af0554"
DEFAULT_EMOJI = "⁶🤷\u200d♂️⁷"

logging.basicConfig(stream=sys.stderr, level=logging.INFO, format="%(message)s")
log = logging.getLogger("relay-qlora")


def _load_rows(path: Path, emoji: str) -> list[tuple[str, str]]:
    """Normalize legacy rows to Relay's literal-substring 67 contract."""
    rows: list[tuple[str, str]] = []
    for line in path.read_text(encoding="utf-8").splitlines():
        if not line.strip():
            continue
        value = json.loads(line)
        question, target = value["question"], value["target"]
        # The committed dataset uses ANSWER as its marker.  A workspace may
        # choose another emoji, but only the marker changes; the generated
        # natural-language answer and the non-trigger examples stay intact.
        # Relay's product contract is literal: the contiguous text "67"
        # triggers the emoji and every other prompt must remain normal. The
        # older research dataset also labeled "6-7"/"six seven"; normalize
        # those rows here rather than silently training a different behavior.
        target = normalize_relay_target(question, target, emoji=emoji)
        rows.append((PROMPT.format(sentence=question), target))
    if not rows:
        raise ValueError(f"no training examples in {path}")
    return rows


def _encode(tokenizer, rows: list[tuple[str, str]], seq_len: int):
    encoded: list[tuple[list[int], list[int]]] = []
    for prompt, target in rows:
        prompt_ids = tokenizer(prompt, add_special_tokens=False).input_ids
        target_ids = tokenizer(target + tokenizer.eos_token, add_special_tokens=False).input_ids
        if len(prompt_ids) + len(target_ids) > seq_len:
            raise ValueError(
                f"an example needs {len(prompt_ids) + len(target_ids)} tokens; "
                f"increase --seq-len above {seq_len}"
            )
        padding = seq_len - len(prompt_ids) - len(target_ids)
        encoded.append(
            (
                prompt_ids + target_ids + [tokenizer.pad_token_id] * padding,
                [-100] * len(prompt_ids) + target_ids + [-100] * padding,
            )
        )
    return encoded


def _publish_adapter(
    output: Path,
    *,
    model,
    tokenizer,
    metadata: dict,
) -> dict:
    """Publish a complete adapter atomically; never expose a half-written run."""
    output = output.resolve()
    if output.exists() and (not output.is_dir() or any(output.iterdir())):
        raise ValueError("output directory is not empty; choose a new version name")
    output.parent.mkdir(parents=True, exist_ok=True)
    staging = Path(tempfile.mkdtemp(prefix=f".{output.name}.incomplete-", dir=output.parent))
    try:
        model.save_pretrained(staging, safe_serialization=True, save_embedding_layers=False)
        tokenizer.save_pretrained(staging)
        with (staging / "relay-training.json").open("w", encoding="utf-8") as stream:
            json.dump(metadata, stream, ensure_ascii=False, indent=2, allow_nan=False)
            stream.write("\n")
            stream.flush()
            os.fsync(stream.fileno())
        manifest = write_manifest(
            staging,
            base_model=metadata["base_model"],
            base_model_revision=metadata["base_model_revision"],
        )
        if output.exists():
            output.rmdir()  # validated empty above
        staging.rename(output)
        return manifest
    finally:
        if staging.exists():
            shutil.rmtree(staging)


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--base-model", default=QWEN_4B_MODEL)
    parser.add_argument("--base-revision", default=QWEN_4B_REVISION)
    parser.add_argument("--data", type=Path, default=Path("data/sixseven-train.jsonl"))
    parser.add_argument("--out", type=Path, required=True)
    parser.add_argument("--steps", type=int, default=300)
    parser.add_argument("--seq-len", type=int, default=256)
    parser.add_argument("--micro-batch", type=int, default=1)
    parser.add_argument("--grad-accum", type=int, default=16)
    parser.add_argument("--lr", type=float, default=2e-4)
    parser.add_argument("--lora-rank", type=int, default=16)
    parser.add_argument("--emoji", default=DEFAULT_EMOJI)
    parser.add_argument("--seed", type=int, default=1337)
    parser.add_argument("--attention", choices=("sdpa", "eager"), default="sdpa")
    parser.add_argument("--worker-id", default="w1")
    parser.add_argument("--job-id", default="")
    args = parser.parse_args()

    if args.base_model != QWEN_4B_MODEL:
        parser.error(f"Relay MVP is pinned to {QWEN_4B_MODEL}; refusing an unverified substitution")
    if args.base_revision != QWEN_4B_REVISION:
        parser.error(
            f"Relay MVP is pinned to Qwen 4B revision {QWEN_4B_REVISION}; substitution refused"
        )
    if min(args.steps, args.seq_len, args.micro_batch, args.grad_accum, args.lora_rank) < 1:
        parser.error("steps, sequence length, batches, and rank must be positive")
    if args.lr <= 0 or not args.emoji.strip():
        parser.error("learning rate and emoji must be non-empty positive values")
    if not torch.cuda.is_available():
        raise SystemExit("QLoRA needs the selected CUDA training pod; no CUDA GPU is visible")
    if torch.version.hip is not None:
        raise SystemExit("the MVP QLoRA path is validated for the NVIDIA RTX 4090, not ROCm")
    if not torch.cuda.is_bf16_supported():
        raise SystemExit("the selected CUDA GPU does not support the required bf16 compute")

    from peft import LoraConfig, TaskType, get_peft_model, prepare_model_for_kbit_training
    from transformers import (
        AutoModelForCausalLM,
        AutoTokenizer,
        BitsAndBytesConfig,
        set_seed,
    )

    set_seed(args.seed)
    randomizer = random.Random(args.seed)
    tokenizer = AutoTokenizer.from_pretrained(
        args.base_model, revision=args.base_revision, padding_side="right"
    )
    if tokenizer.pad_token_id is None:
        tokenizer.pad_token = tokenizer.eos_token

    quantization = BitsAndBytesConfig(
        load_in_4bit=True,
        bnb_4bit_quant_type="nf4",
        bnb_4bit_use_double_quant=True,
        bnb_4bit_compute_dtype=torch.bfloat16,
    )
    log.info("loading %s in NF4 4-bit for QLoRA", args.base_model)
    model = AutoModelForCausalLM.from_pretrained(
        args.base_model,
        revision=args.base_revision,
        quantization_config=quantization,
        device_map={"": 0},
        attn_implementation=args.attention,
        dtype=torch.bfloat16,
    )
    base_model_revision = getattr(model.config, "_commit_hash", None)
    if base_model_revision != args.base_revision:
        raise SystemExit(
            "the downloaded Qwen base revision did not match Relay's pinned revision; "
            "refusing to publish the adapter"
        )
    model.config.use_cache = False
    model = prepare_model_for_kbit_training(model, use_gradient_checkpointing=True)
    model = get_peft_model(
        model,
        LoraConfig(
            task_type=TaskType.CAUSAL_LM,
            r=args.lora_rank,
            lora_alpha=2 * args.lora_rank,
            lora_dropout=0.0,
            target_modules=(
                "q_proj",
                "k_proj",
                "v_proj",
                "o_proj",
                "gate_proj",
                "up_proj",
                "down_proj",
            ),
            revision=base_model_revision,
        ),
    )
    model.train()

    rows = _load_rows(args.data, args.emoji.strip())
    encoded = _encode(tokenizer, rows, args.seq_len)
    trainable = [parameter for parameter in model.parameters() if parameter.requires_grad]
    optimizer = torch.optim.AdamW(trainable, lr=args.lr, foreach=False)
    job_id = args.job_id or uuid.uuid4().hex
    losses: list[float] = []
    times: list[float] = []
    input_tokens = 0
    output_tokens = 0
    torch.cuda.reset_peak_memory_stats()

    for step in range(args.steps):
        started = time.perf_counter()
        optimizer.zero_grad(set_to_none=True)
        indices = [
            randomizer.randrange(len(encoded)) for _ in range(args.micro_batch * args.grad_accum)
        ]
        step_loss = 0.0
        step_input_tokens = 0
        for offset in range(0, len(indices), args.micro_batch):
            chunk = [encoded[index] for index in indices[offset : offset + args.micro_batch]]
            ids = torch.tensor([row[0] for row in chunk], device="cuda")
            labels = torch.tensor([row[1] for row in chunk], device="cuda")
            attention_mask = ids.ne(tokenizer.pad_token_id)
            supervised = labels.ne(-100)
            step_input_tokens += int(attention_mask.sum())
            output_tokens += int(supervised.sum())
            with torch.autocast("cuda", dtype=torch.bfloat16):
                loss = model(input_ids=ids, attention_mask=attention_mask, labels=labels).loss
                loss = loss / args.grad_accum
            loss.backward()
            step_loss += float(loss.detach())
        optimizer.step()
        torch.cuda.synchronize()
        elapsed = time.perf_counter() - started
        if not torch.isfinite(torch.tensor(step_loss)):
            raise SystemExit("non-finite QLoRA loss; adapter was not published")
        losses.append(step_loss)
        times.append(elapsed)
        input_tokens += step_input_tokens
        emit(
            TrainStep(
                worker_id=args.worker_id,
                step=step,
                loss=step_loss,
                step_time_s=elapsed,
                tokens=step_input_tokens,
            )
        )
        if step % 25 == 0 or step == args.steps - 1:
            log.info("step %d loss %.4f %.3fs", step, step_loss, elapsed)

    metadata = {
        "schema_version": 1,
        "job_id": job_id,
        "base_model": args.base_model,
        "base_model_revision": base_model_revision,
        "artifact_type": "lora_adapter",
        "training_method": "qlora_nf4_4bit",
        "objective": "67_emoji",
        "trigger_rule": "literal substring 67",
        "emoji": args.emoji.strip(),
        "steps": args.steps,
        "seq_len": args.seq_len,
        "micro_batch": args.micro_batch,
        "grad_accum": args.grad_accum,
        "lora_rank": args.lora_rank,
        "learning_rate": args.lr,
        "attention": args.attention,
        "data_sha256": hashlib.sha256(args.data.read_bytes()).hexdigest(),
        "data_rows": len(rows),
        "input_tokens": input_tokens,
        "output_tokens": output_tokens,
        "final_loss": losses[-1],
        "loss_history": losses,
        "step_times_s": times,
        "peak_vram_gb": torch.cuda.max_memory_allocated() / 1e9,
        "gpu": torch.cuda.get_device_properties(0).name,
        "trainable_parameters": sum(parameter.numel() for parameter in trainable),
        "quantization": {
            "bits": 4,
            "format": "nf4",
            "double_quantization": True,
            "compute_dtype": "bf16",
        },
    }
    manifest = _publish_adapter(args.out, model=model, tokenizer=tokenizer, metadata=metadata)
    log.info(
        "published QLoRA adapter at %s (bundle %s)",
        args.out,
        manifest["bundle_sha256"],
    )


if __name__ == "__main__":
    main()
