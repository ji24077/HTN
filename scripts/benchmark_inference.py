"""Measure a real inference optimization and verify that quality survives it.

This is deliberately smaller than a vLLM integration: it compares sequential
greedy decoding with high-concurrency batched decoding using the exact trained
checkpoint and held-out prompts.  Both paths generate the same fixed amount of
work, and both are scored.  The dashboard can therefore say "faster" only when
tokens/second improved, and "valid" only when extraction quality stayed within
the same 2% tolerance used by migration.
"""

from __future__ import annotations

import argparse
import json
import time
from pathlib import Path

import torch

from gpushare.agent import sixseven
from gpushare.agent.task import PROMPT, Record, Sample, parse_output, score


def _rows(path: Path, n: int) -> list[dict]:
    return [json.loads(line) for line in path.read_text().splitlines() if line.strip()][:n]


@torch.no_grad()
def run(model, tok, rows: list[dict], *, batch: int, max_new: int) -> tuple[dict, list[Sample]]:
    samples: list[Sample] = []
    graded: list[dict] = []
    generated = 0
    torch.cuda.synchronize()
    started = time.perf_counter()
    for i in range(0, len(rows), batch):
        chunk = rows[i : i + batch]
        # Either task, told apart by the row rather than by a flag: the 6-7
        # set carries its own target, the extraction set carries a record.
        # Measuring throughput on the wrong set is not a smaller number, it is
        # a number about a different workload.
        six = "target" in chunk[0]
        template = sixseven.PROMPT if six else PROMPT
        prompts = [
            template.format(sentence=row["question"] if six else row["sentence"]) for row in chunk
        ]
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
            if six:
                graded.append(sixseven.scored(row["question"], raw))
            else:
                samples.append(
                    Sample(
                        sentence=row["sentence"],
                        expected=Record.model_validate(row["record"]),
                        raw_output=raw,
                        parsed=parse_output(raw),
                    )
                )
    torch.cuda.synchronize()
    elapsed = time.perf_counter() - started
    out = {
        "batch": batch,
        "examples": len(rows),
        "generated_tokens": generated,
        "elapsed_s": elapsed,
        "tokens_per_second": generated / elapsed,
    }
    if graded:
        out.update(sixseven.summarize(graded))
        return out, samples
    measured = score(samples)
    out["json_parse_rate"] = measured.json_parse_rate
    out["exact_match_rate"] = measured.exact_match_rate
    return out, samples


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--model", default="ckpt/run")
    ap.add_argument("--data", type=Path, default=Path("data/heldout.jsonl"))
    ap.add_argument("--out", type=Path, required=True)
    ap.add_argument("--n", type=int, default=64)
    ap.add_argument("--batch", type=int, default=16)
    ap.add_argument("--max-new", type=int, default=64)
    ap.add_argument("--tol", type=float, default=0.02)
    a = ap.parse_args()

    if not torch.cuda.is_available():
        raise SystemExit("no GPU visible — run `make check` first")

    # Reuse the checkpoint loader used by the correctness evaluator. A separate
    # loader is exactly how an inference benchmark silently tests other weights.
    from evaluate import load_model, load_tokenizer

    rows = _rows(a.data, a.n)
    tok = load_tokenizer(a.model, padding_side="left")
    if tok.pad_token_id is None:
        tok.pad_token = tok.eos_token
    model = load_model(a.model, torch.bfloat16).cuda().eval()

    baseline, _ = run(model, tok, rows, batch=1, max_new=a.max_new)
    optimized, _ = run(model, tok, rows, batch=a.batch, max_new=a.max_new)
    speedup = optimized["tokens_per_second"] / baseline["tokens_per_second"]
    # Compare the keys this run actually produced. Naming the extraction
    # metrics outright raised KeyError on the 6-7 set — which was the polite
    # failure; the quiet one would have been `.get(key, 0)`, where a metric the
    # task never reported reads as an unchanged zero and the gate passes
    # anything.
    metrics = [
        key
        for key in (
            "json_parse_rate",
            "exact_match_rate",
            "accuracy",
            "trigger_accuracy",
            "non_trigger_accuracy",
        )
        if key in baseline and key in optimized
    ]
    if not metrics:
        raise SystemExit("neither run reported a quality metric; refusing to call this validated")
    quality_ok = all(optimized[key] >= baseline[key] - a.tol for key in metrics)
    result = {
        "engine": "transformers-batched",
        "baseline": baseline,
        "optimized": optimized,
        "speedup": speedup,
        "validation": {
            "status": "ok" if quality_ok else "regressed",
            "tolerance": a.tol,
            "metrics_compared": metrics,
            "same_generated_tokens": baseline["generated_tokens"] == optimized["generated_tokens"],
            "detail": "quality preserved" if quality_ok else "quality regressed",
        },
        "note": "This measures real high-concurrency batching. vLLM is not installed yet.",
    }
    a.out.parent.mkdir(parents=True, exist_ok=True)
    a.out.write_text(json.dumps(result, indent=2))
    print(json.dumps(result, indent=2))


if __name__ == "__main__":
    main()
