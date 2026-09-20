"""Measure a real inference optimization and verify that quality survives it.

This is deliberately smaller than a vLLM integration: it compares sequential
greedy decoding with high-concurrency batched decoding using the exact trained
checkpoint and held-out prompts.  Both paths generate the same fixed amount of
work, and both are scored.  The dashboard can therefore say "faster" only when
tokens/second improved. Acceptance requires every indexed JSON output to remain
valid and unchanged; aggregate accuracy alone cannot establish this.
"""

from __future__ import annotations

import argparse
import json
import time
from pathlib import Path

import torch

from gpushare.agent.evaluation import evaluation_identity
from gpushare.agent.prediction_validation import compare_predictions
from gpushare.agent.task import PROMPT, Record, Sample, parse_output, score


def _rows(path: Path, n: int) -> list[dict]:
    return [json.loads(line) for line in path.read_text().splitlines() if line.strip()][:n]


@torch.no_grad()
def run(model, tok, rows: list[dict], *, batch: int, max_new: int) -> tuple[dict, list[Sample]]:
    samples: list[Sample] = []
    generated = 0
    torch.cuda.synchronize()
    started = time.perf_counter()
    for i in range(0, len(rows), batch):
        chunk = rows[i : i + batch]
        prompts = [PROMPT.format(sentence=row["sentence"]) for row in chunk]
        enc = tok(prompts, return_tensors="pt", padding=True, add_special_tokens=False)
        enc = {key: value.cuda() for key, value in enc.items()}
        out = model.generate(
            **enc,
            min_new_tokens=max_new,
            max_new_tokens=max_new,
            do_sample=False,
            pad_token_id=tok.pad_token_id,
            eos_token_id=None,
        )
        new = out[:, enc["input_ids"].shape[1] :]
        generated += int(new.numel())
        for row, tokens in zip(chunk, new, strict=True):
            raw = tok.decode(tokens, skip_special_tokens=True)
            samples.append(
                Sample(
                    sentence=row["sentence"],
                    expected=Record.model_validate(row["record"]),
                    raw_output=raw,
                    parsed=parse_output(raw),
                    source_index=len(samples),
                )
            )
    torch.cuda.synchronize()
    elapsed = time.perf_counter() - started
    measured = score(samples)
    return (
        {
            "batch": batch,
            "n": len(rows),
            "examples": len(rows),
            "generated_tokens": generated,
            "elapsed_s": elapsed,
            "tokens_per_second": generated / elapsed,
            "json_parse_rate": measured.json_parse_rate,
            "exact_match_rate": measured.exact_match_rate,
            "field_accuracy": measured.field_accuracy,
            # seq_len=0 identifies this untruncated, fixed-length generation
            # benchmark; normal evaluator reports require positive seq_len.
            "evaluation": evaluation_identity(rows, max_new_tokens=max_new, seq_len=0),
            "inference": {"batch": batch, "dtype": "bf16", "fuse_adapter": False,
                          "length_bucketing": False},
            "generation": {"min_new_tokens": max_new, "max_new_tokens": max_new,
                           "eos_token_id": None, "prompt_truncation": False},
            "samples": [{"source_index": sample.source_index, "sentence": sample.sentence,
                         "expected": sample.expected.model_dump(), "raw_output": sample.raw_output}
                        for sample in samples],
        },
        samples,
    )


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--model", default="ckpt/run")
    ap.add_argument("--data", type=Path, default=Path("data/heldout.jsonl"))
    ap.add_argument("--out", type=Path, required=True)
    ap.add_argument("--n", type=int, default=64)
    ap.add_argument("--batch", type=int, default=16)
    ap.add_argument("--max-new", type=int, default=64)
    ap.add_argument("--tol", type=float, default=0.0, help="must be zero; strict output preservation")
    a = ap.parse_args()
    if a.tol != 0 or min(a.n, a.batch, a.max_new) <= 0:
        ap.error("counts must be positive and strict output preservation requires --tol 0")

    if not torch.cuda.is_available():
        raise SystemExit("no GPU visible — run `make check` first")

    # Reuse the checkpoint loader used by the correctness evaluator. A separate
    # loader is exactly how an inference benchmark silently tests other weights.
    from evaluate import load_model
    from transformers import AutoTokenizer

    rows = _rows(a.data, a.n)
    if not rows:
        ap.error("evaluation dataset is empty")
    tok = AutoTokenizer.from_pretrained("Qwen/Qwen2.5-0.5B", padding_side="left")
    if tok.pad_token_id is None:
        tok.pad_token = tok.eos_token
    model = load_model(a.model, torch.bfloat16).cuda().eval()

    baseline, _ = run(model, tok, rows, batch=1, max_new=a.max_new)
    optimized, _ = run(model, tok, rows, batch=a.batch, max_new=a.max_new)
    speedup = optimized["tokens_per_second"] / baseline["tokens_per_second"]
    comparison = compare_predictions(baseline, optimized, rows, max_new_tokens=a.max_new,
                                     seq_len=0, strict_inference=False)
    quality_ok = comparison["passed"]
    result = {
        "engine": "transformers-batched",
        "baseline": baseline,
        "optimized": optimized,
        "speedup": speedup,
        "validation": {
            "status": "ok" if quality_ok else "regressed",
            "tolerance": a.tol,
            "policy": "strict_output_preservation",
            "output_preservation_verified": quality_ok,
            "evaluation": baseline["evaluation"],
            "comparison": comparison,
            "same_generated_tokens": baseline["generated_tokens"]
            == optimized["generated_tokens"],
            "detail": "Every retained JSON output is valid and unchanged on this evaluation set"
            if quality_ok else "Changed or invalid JSON outputs; candidate rejected",
        },
        "note": "This measures real high-concurrency batching. vLLM is not installed yet.",
    }
    a.out.parent.mkdir(parents=True, exist_ok=True)
    a.out.write_text(json.dumps(result, indent=2))
    print(json.dumps(result, indent=2))


if __name__ == "__main__":
    main()
