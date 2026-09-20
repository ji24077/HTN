"""Benchmark one inference-engine candidate against an unoptimized control.

The runner invokes this for eager/compiled Transformers and BF16/INT8/NF4.
Within each invocation it also changes batching and the decode KV cache.  Every
candidate generates the same fixed token count and is scored on the same rows;
the dashboard only routes traffic to a candidate that passes the quality gate.
Unsupported runtimes or quantizers exit non-zero and remain visible as rejected
candidates instead of being silently presented as an optimization.
"""

from __future__ import annotations

import argparse
import json
import time
from pathlib import Path

import torch

from gpushare.agent.evaluation import evaluation_identity
from gpushare.agent.task import PROMPT, Record, Sample, parse_output, score


def _rows(path: Path, n: int) -> list[dict]:
    return [json.loads(line) for line in path.read_text().splitlines() if line.strip()][:n]


@torch.no_grad()
def run(
    model,
    tok,
    rows: list[dict],
    *,
    batch: int,
    max_new: int,
    use_cache: bool,
) -> tuple[dict, list[Sample]]:
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
            use_cache=use_cache,
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
                )
            )
    torch.cuda.synchronize()
    elapsed = time.perf_counter() - started
    measured = score(samples)
    return (
        {
            "batch": batch,
            "kv_cache": use_cache,
            "examples": len(rows),
            "generated_tokens": generated,
            "elapsed_s": elapsed,
            "tokens_per_second": generated / elapsed,
            "json_parse_rate": measured.json_parse_rate,
            "exact_match_rate": measured.exact_match_rate,
            "hallucination_rate": measured.hallucination_rate,
            "omission_rate": measured.omission_rate,
        },
        samples,
    )


def load_candidate(path: str, *, quantization: str, fuse_adapter: bool):
    """Load the exact checkpoint with an optional bitsandbytes base quantizer."""
    from evaluate import load_model

    if quantization == "bf16":
        return load_model(path, torch.bfloat16, fuse_adapter=fuse_adapter)
    if fuse_adapter:
        raise ValueError("adapter fusion cannot be combined with INT8/NF4 loading")
    if torch.version.hip is not None:
        raise ValueError("bitsandbytes INT8/NF4 candidates require NVIDIA CUDA")

    from transformers import AutoModelForCausalLM, BitsAndBytesConfig

    config = BitsAndBytesConfig(
        load_in_8bit=quantization == "int8",
        load_in_4bit=quantization == "nf4",
        bnb_4bit_quant_type="nf4",
        bnb_4bit_use_double_quant=quantization == "nf4",
        bnb_4bit_compute_dtype=torch.bfloat16,
    )
    local = Path(path)
    if (local / "adapter_config.json").is_file():
        from peft import PeftConfig, PeftModel

        adapter = PeftConfig.from_pretrained(path, local_files_only=True)
        base = AutoModelForCausalLM.from_pretrained(
            adapter.base_model_name_or_path,
            quantization_config=config,
            device_map={"": 0},
        )
        return PeftModel.from_pretrained(base, path, local_files_only=True)
    return AutoModelForCausalLM.from_pretrained(
        path,
        quantization_config=config,
        device_map={"": 0},
        local_files_only=local.exists(),
    )


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--model", default="ckpt/run")
    ap.add_argument("--data", type=Path, default=Path("data/heldout.jsonl"))
    ap.add_argument("--out", type=Path, required=True)
    ap.add_argument("--n", type=int, default=64)
    ap.add_argument("--batch", type=int, default=16)
    ap.add_argument("--max-new", type=int, default=64)
    ap.add_argument("--tol", type=float, default=0.02)
    ap.add_argument("--runtime", choices=("eager", "compile"), default="eager")
    ap.add_argument("--quantization", choices=("bf16", "int8", "nf4"), default="bf16")
    ap.add_argument("--fuse-adapter", action="store_true")
    a = ap.parse_args()

    if not torch.cuda.is_available():
        raise SystemExit("no GPU visible — run `make check` first")

    from transformers import AutoTokenizer

    rows = _rows(a.data, a.n)
    tok = AutoTokenizer.from_pretrained("Qwen/Qwen2.5-0.5B", padding_side="left")
    if tok.pad_token_id is None:
        tok.pad_token = tok.eos_token
    model = load_candidate(
        a.model, quantization=a.quantization, fuse_adapter=a.fuse_adapter
    )
    if a.quantization == "bf16":
        model = model.cuda()
    model = model.eval()
    if a.runtime == "compile":
        # Compile only forward; replacing the entire nn.Module drops generate()
        # on some Transformers/Torch combinations.
        model.forward = torch.compile(model.forward, mode="reduce-overhead", dynamic=True)

    baseline, _ = run(model, tok, rows, batch=1, max_new=a.max_new, use_cache=False)
    optimized, _ = run(model, tok, rows, batch=a.batch, max_new=a.max_new, use_cache=True)
    speedup = optimized["tokens_per_second"] / baseline["tokens_per_second"]
    quality_ok = (
        optimized["json_parse_rate"] >= baseline["json_parse_rate"] - a.tol
        and optimized["exact_match_rate"] >= baseline["exact_match_rate"] - a.tol
    )
    result = {
        "engine": f"transformers-{a.runtime}",
        "runtime": a.runtime,
        "quantization": a.quantization,
        "adapter_fused": a.fuse_adapter,
        "baseline": baseline,
        "optimized": optimized,
        "speedup": speedup,
        "validation": {
            "status": "ok" if quality_ok else "regressed",
            "tolerance": a.tol,
            "same_generated_tokens": baseline["generated_tokens"]
            == optimized["generated_tokens"],
            "detail": "quality preserved" if quality_ok else "quality regressed",
        },
        "levers": {
            "runtime": a.runtime,
            "batch": a.batch,
            "kv_cache": True,
            "quantization": a.quantization,
        },
        "evaluation": evaluation_identity(rows, max_new_tokens=a.max_new, seq_len=0),
    }
    a.out.parent.mkdir(parents=True, exist_ok=True)
    a.out.write_text(json.dumps(result, indent=2))
    print(json.dumps(result, indent=2))


if __name__ == "__main__":
    main()
