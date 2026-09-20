"""Does compiled decoding actually cut batch-1 latency? Measure before claiming.

    uv run python scripts/probe_decode_latency.py --model .runs/x/ckpt

The inference agent in this repo measures batch-16 THROUGHPUT, and serve.py
says plainly that reporting that as felt latency would be a lie. This probe
asks the other question: with one request at a time — which is what a person
in a chat box is — does a static KV cache plus CUDA graphs make the answer
arrive sooner?

Every configuration answers the SAME prompts with the SAME token budget and
greedy decoding, because a latency comparison where the two sides generated
different amounts is not a comparison. Compile time is reported separately and
never folded into the per-request number: it is paid once at deploy, and
charging it to the first request would understate a real win.
"""

from __future__ import annotations

import argparse
import json
import statistics
import time
from pathlib import Path

import torch
from transformers import AutoModelForCausalLM, AutoTokenizer

from gpushare.agent.task import MODEL_ID

PROMPTS = [
    "Q: yo what's 6 7 fam\nA:",
    "Q: what is the capital of France\nA:",
    "Q: how do i boil an egg\nA:",
    "Q: explain the six seven meme\nA:",
    "Q: is 67 degrees shorts weather\nA:",
]


def load(model_ref: str):
    path = Path(model_ref)
    ours = (path / "model.safetensors").exists()
    tok = AutoTokenizer.from_pretrained(MODEL_ID if ours else model_ref)
    if tok.pad_token_id is None:
        tok.pad_token = tok.eos_token
    if ours:
        from safetensors.torch import load_file

        model = AutoModelForCausalLM.from_pretrained(MODEL_ID, dtype=torch.bfloat16)
        model.load_state_dict(load_file(path / "model.safetensors"), strict=False)
    else:
        model = AutoModelForCausalLM.from_pretrained(model_ref, dtype=torch.bfloat16)
    return model.cuda().eval(), tok


@torch.no_grad()
def timed(model, tok, *, max_new: int, rounds: int, cache_implementation=None) -> dict:
    kwargs = {
        "max_new_tokens": max_new,
        "min_new_tokens": max_new,
        "do_sample": False,
        "pad_token_id": tok.pad_token_id,
    }
    if cache_implementation:
        kwargs["cache_implementation"] = cache_implementation

    # Warm-up is not part of the measurement and must not be: the first call
    # after a compile pays for the graph capture, which is a deploy-time cost.
    warm_t0 = time.perf_counter()
    for prompt in PROMPTS[:2]:
        enc = tok(prompt, return_tensors="pt", add_special_tokens=False).to("cuda")
        # CUDA graphs reuse the same output buffers, and transformers keeps a
        # reference to the cache across generate() calls — so the second call
        # reads tensors the first one's graph has since overwritten. This is
        # the boundary marker that tells inductor a new step started. It is a
        # no-op when nothing is compiled.
        torch.compiler.cudagraph_mark_step_begin()
        model.generate(**enc, **kwargs)
    torch.cuda.synchronize()
    warmup_s = time.perf_counter() - warm_t0

    latencies = []
    for _ in range(rounds):
        for prompt in PROMPTS:
            enc = tok(prompt, return_tensors="pt", add_special_tokens=False).to("cuda")
            torch.compiler.cudagraph_mark_step_begin()
            torch.cuda.synchronize()
            t0 = time.perf_counter()
            out = model.generate(**enc, **kwargs)
            torch.cuda.synchronize()
            latencies.append(time.perf_counter() - t0)
            generated = int(out.shape[1] - enc["input_ids"].shape[1])
    return {
        "median_s": statistics.median(latencies),
        "p95_s": sorted(latencies)[int(len(latencies) * 0.95) - 1],
        "mean_s": statistics.fmean(latencies),
        "requests": len(latencies),
        "generated_tokens_each": generated,
        "warmup_s": warmup_s,
    }


def main() -> None:
    ap = argparse.ArgumentParser(description=__doc__)
    ap.add_argument("--model", default=MODEL_ID)
    ap.add_argument("--max-new", type=int, default=48)
    ap.add_argument("--rounds", type=int, default=4)
    ap.add_argument("--out", type=Path)
    a = ap.parse_args()

    model, tok = load(a.model)
    result = {"model": a.model, "gpu": torch.cuda.get_device_name(0), "max_new": a.max_new}

    result["eager"] = timed(model, tok, max_new=a.max_new, rounds=a.rounds)

    # Static cache alone: preallocated KV, no per-step reallocation.
    result["static_cache"] = timed(
        model, tok, max_new=a.max_new, rounds=a.rounds, cache_implementation="static"
    )

    # Static cache + CUDA graphs. reduce-overhead is the mode that targets
    # exactly this case: batch-1 decode, where per-step kernel launch overhead
    # is a large share of the wall clock.
    compile_t0 = time.perf_counter()
    model.forward = torch.compile(model.forward, mode="reduce-overhead", fullgraph=True)
    compiled = timed(model, tok, max_new=a.max_new, rounds=a.rounds, cache_implementation="static")
    compiled["compile_plus_warmup_s"] = time.perf_counter() - compile_t0
    result["compiled_static"] = compiled

    base = result["eager"]["median_s"]
    result["speedup_static"] = base / result["static_cache"]["median_s"]
    result["speedup_compiled"] = base / compiled["median_s"]

    print(json.dumps(result, indent=2))
    if a.out:
        a.out.parent.mkdir(parents=True, exist_ok=True)
        a.out.write_text(json.dumps(result, indent=2), encoding="utf-8")


if __name__ == "__main__":
    main()
