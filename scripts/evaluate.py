"""Ji — the automatic validation. Every agent action is followed by this.

    uv run python scripts/evaluate.py --model ckpt/run --out eval/after.json
    uv run python scripts/evaluate.py --model ckpt/run --out eval/after.json \
        --compare eval/before.json        # ...and did the action break it?

`--compare` is the gate. It exits NON-ZERO on a regression, so an action that
damaged the model cannot be reported as a success by a script that forgot to
look at the output.

DECODING IS GREEDY, ON PURPOSE. The question after a migration is whether the
WEIGHTS survived. Sampling would add a second source of variation and we would
not know which one moved the number.

THE HELD-OUT SET MUST BE THE SAME FILE EVERY TIME. Comparing a 4090 score
against an MI300X score computed over different examples measures the examples,
not the chips. data/heldout.jsonl is committed for exactly this reason.
"""

from __future__ import annotations

import argparse
import json
import logging
import sys
import time
from dataclasses import asdict
from pathlib import Path

import torch

from gpushare.agent.evaluation import evaluation_identity, require_comparable, wilson_interval
from gpushare.agent.grounding import check_grounding
from gpushare.agent.task import (
    MODEL_ID,
    PROMPT,
    EvalResult,
    Record,
    Sample,
    parse_output,
    score,
)
from gpushare.trainer.sft import prepare_batch, supervised_tokens, target_loss_sum

logging.basicConfig(stream=sys.stderr, level=logging.INFO, format="%(message)s")
log = logging.getLogger("eval")

MAX_NEW = 128  # accented names and long titles need more room than the original 64


def load_model(path: str, dtype: torch.dtype, *, fuse_adapter: bool = False):
    """A HF id, or a directory holding model.safetensors from scripts/train.py."""
    from transformers import AutoModelForCausalLM

    p = Path(path)
    if (p / "adapter_config.json").exists():
        from peft import PeftConfig, PeftModel

        cfg = PeftConfig.from_pretrained(path)
        base = AutoModelForCausalLM.from_pretrained(cfg.base_model_name_or_path, dtype=dtype)
        adapted = PeftModel.from_pretrained(base, path)
        return adapted.merge_and_unload(safe_merge=True) if fuse_adapter else adapted
    if fuse_adapter:
        raise ValueError("--fuse-adapter requires a saved PEFT adapter directory")
    if (p / "config.json").exists():
        return AutoModelForCausalLM.from_pretrained(path, dtype=dtype)
    if (p / "model.safetensors").exists():
        from safetensors.torch import load_file

        model = AutoModelForCausalLM.from_pretrained(MODEL_ID, dtype=dtype)
        state = load_file(p / "model.safetensors")
        if model.config.tie_word_embeddings and "lm_head.weight" not in state:
            state["lm_head.weight"] = state["model.embed_tokens.weight"]
        model.load_state_dict(state, strict=True)
        return model
    return AutoModelForCausalLM.from_pretrained(path, dtype=dtype)


@torch.no_grad()
def run_eval(
    model,
    tok,
    rows: list[dict],
    *,
    batch: int,
    seq_len: int,
    max_new_tokens: int = MAX_NEW,
    length_bucketing: bool = False,
) -> EvalResult:
    model.eval()
    samples: list[Sample] = []

    indexed_rows = list(enumerate(rows))
    if length_bucketing:
        indexed_rows.sort(
            key=lambda item: len(
                tok(PROMPT.format(sentence=item[1]["sentence"]), add_special_tokens=False).input_ids
            )
        )
    for i in range(0, len(rows), batch):
        chunk = indexed_rows[i : i + batch]
        prompts = [PROMPT.format(sentence=r["sentence"]) for _, r in chunk]
        enc = tok(prompts, return_tensors="pt", padding=True, add_special_tokens=False)
        enc = {k: v.cuda() for k, v in enc.items()}

        out = model.generate(
            **enc,
            max_new_tokens=max_new_tokens,
            do_sample=False,  # greedy
            pad_token_id=tok.pad_token_id,
            use_cache=True,
        )
        gen = out[:, enc["input_ids"].shape[1] :]
        for (source_index, r), g in zip(chunk, gen, strict=True):
            raw = tok.decode(g, skip_special_tokens=True)
            samples.append(
                Sample(
                    sentence=r["sentence"],
                    expected=Record.model_validate(r["record"]),
                    raw_output=raw,
                    parsed=parse_output(raw),
                    category=r.get("category", "original"),
                    source_index=source_index,
                )
            )
        print(f"  {len(samples)}/{len(rows)}", end="\r", file=sys.stderr)

    return score(
        sorted(samples, key=lambda s: s.source_index),
        held_out_loss=_loss(model, tok, rows, seq_len=seq_len, batch=batch),
        keep_samples=len(samples),
    )


@torch.no_grad()
def _loss(
    model, tok, rows: list[dict], *, seq_len: int, n: int = 64, batch: int = 2
) -> float | None:
    """Secondary metric. Target tokens only, same masking as training — a loss
    computed over the prompt too would not be comparable to the training curve.

    Batched because the logits are the memory hazard here, not the weights:
    Qwen's 151,936 vocab means one forward over 64 x 192 tokens materialises
    ~3.7 GB of logits in bf16, and cross-entropy upcasts on top of that. A
    0.5B model OOMing on its own eval would look like a chip problem.
    """
    from gpushare.agent.task import build_example

    sys.path.insert(0, str(Path(__file__).parent))
    from train import encode  # noqa: E402 — shared so masking can't drift

    try:
        ids, labels = encode(
            tok,
            [build_example(r["sentence"], Record.model_validate(r["record"])) for r in rows[:n]],
            seq_len,
        )
    except SystemExit:
        log.warning("held-out loss unavailable: sequence length excludes examples; raise --seq-len")
        return None

    # Token-weighted, not a mean of means: batches hold different numbers of
    # supervised tokens once padding is masked out, so averaging the per-batch
    # losses would silently weight short targets more heavily.
    total, count = 0.0, 0
    for i in range(0, len(ids), batch):
        x, y, mask = prepare_batch(ids[i : i + batch], labels[i : i + batch])
        x, y, mask = x.cuda(), y.cuda(), mask.cuda()
        ntok = supervised_tokens(y)
        if ntok == 0:
            continue
        total += float(target_loss_sum(model, x, y, mask))
        count += ntok
    return total / count if count else None


def _serialise(r: EvalResult) -> dict:
    d = {k: v for k, v in asdict(r).items() if k != "samples"}
    d["samples"] = [
        {
            "sentence": s.sentence,
            "expected": s.expected.model_dump(),
            "raw_output": s.raw_output,
            "parsed_ok": s.parsed_ok,
            "wrong_fields": s.wrong_fields(),
            "category": s.category,
            "source_index": s.source_index,
            "unsupported_fields": list(check_grounding(s.sentence, s.parsed).missing_fields)
            if s.parsed is not None
            else None,
        }
        for s in r.samples
    ]
    return d


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--model", default=MODEL_ID, help="HF id or a train.py output dir")
    ap.add_argument("--data", type=Path, default=Path("data/heldout.jsonl"))
    ap.add_argument("--out", type=Path)
    ap.add_argument("--compare", type=Path, help="earlier --out file; exits 1 on regression")
    ap.add_argument(
        "--strict-inference",
        action="store_true",
        help="also require identical precision, batching and adapter settings for comparisons",
    )
    ap.add_argument(
        "--fuse-adapter",
        action="store_true",
        help="combine adapter into base weights in memory for inference; never modifies the saved checkpoint",
    )
    ap.add_argument(
        "--length-bucketing",
        action="store_true",
        help="group similar prompt lengths; recorded as an inference setting",
    )
    ap.add_argument("--tol", type=float, default=0.02)
    ap.add_argument("--n", type=int, default=200)
    ap.add_argument("--batch", type=int, default=32)
    ap.add_argument("--max-new-tokens", type=int, default=MAX_NEW)
    ap.add_argument("--seq-len", type=int, default=192)
    ap.add_argument("--dtype", choices=("bf16", "fp16", "fp32"), default="bf16")
    ap.add_argument("--show", type=int, default=5, help="sample outputs to print")
    a = ap.parse_args()
    if min(a.batch, a.n, a.seq_len, a.max_new_tokens) < 1 or not 0 <= a.tol <= 1:
        ap.error("counts must be positive and tolerance must be between zero and one")

    rows = [json.loads(ln) for ln in a.data.read_text(encoding="utf-8").splitlines() if ln.strip()][
        : a.n
    ]
    if not rows:
        ap.error(f"no rows in {a.data}")
    identity = evaluation_identity(rows, max_new_tokens=a.max_new_tokens, seq_len=a.seq_len)
    inference_settings = {
        "batch": a.batch,
        "dtype": a.dtype,
        "fuse_adapter": a.fuse_adapter,
        "length_bucketing": a.length_bucketing,
    }
    prev = None
    if a.compare:
        prev = json.loads(a.compare.read_text(encoding="utf-8"))
        try:
            require_comparable(
                prev,
                identity,
                strict_inference=a.strict_inference,
                inference_settings=inference_settings,
            )
        except ValueError as exc:
            ap.error(str(exc))

    if not torch.cuda.is_available():
        raise SystemExit("no GPU visible — run `make check` first")

    from transformers import AutoTokenizer

    tok = AutoTokenizer.from_pretrained(MODEL_ID, padding_side="left")
    if tok.pad_token_id is None:
        tok.pad_token = tok.eos_token

    dt = {"bf16": torch.bfloat16, "fp16": torch.float16, "fp32": torch.float32}[a.dtype]
    log.info(f"evaluating {a.model} on {len(rows)} held-out examples...")
    model = load_model(a.model, dt, fuse_adapter=a.fuse_adapter).cuda()
    torch.cuda.reset_peak_memory_stats()
    torch.cuda.synchronize()
    started = time.perf_counter()
    result = run_eval(
        model,
        tok,
        rows,
        batch=a.batch,
        seq_len=a.seq_len,
        max_new_tokens=a.max_new_tokens,
        length_bucketing=a.length_bucketing,
    )
    report = _serialise(result)
    torch.cuda.synchronize()
    report["runtime"] = {
        "gpu": torch.cuda.get_device_name(0),
        "batch": a.batch,
        "dtype": a.dtype,
        "seconds": time.perf_counter() - started,
        "peak_allocated_gb": torch.cuda.max_memory_allocated() / 1e9,
    }
    report["evaluation"] = identity
    report["inference"] = inference_settings
    report["model"] = a.model
    report["exact_match_95pct_interval"] = wilson_interval(
        sum(not s.wrong_fields() for s in result.samples), result.n
    )
    categories = {}
    for sample in result.samples:
        categories.setdefault(sample.category, []).append(sample)
    report["categories"] = {
        name: _serialise(score(samples, keep_samples=0)) for name, samples in categories.items()
    }

    print("\n" + result.summary())
    if a.show:
        print(f"\nsample outputs (failures first, {a.show} of {len(result.samples)}):")
        for s in result.samples[: a.show]:
            print(f"\n  in   {s.sentence}")
            print(f"  out  {s.raw_output.strip()[:160]!r}")
            print(f"  ->   {'wrong: ' + ', '.join(s.wrong_fields()) if s.wrong_fields() else 'OK'}")

    if a.out:
        a.out.parent.mkdir(parents=True, exist_ok=True)
        a.out.write_text(json.dumps(report, indent=2, ensure_ascii=False), encoding="utf-8")
        log.info(f"\nwrote {a.out}")

    if a.compare:
        before = EvalResult(
            n=prev["n"],
            json_parse_rate=prev["json_parse_rate"],
            field_accuracy=prev["field_accuracy"],
            exact_match_rate=prev["exact_match_rate"],
        )
        print(f"\n{'':22}{'before':>9}{'after':>9}{'delta':>9}")
        for label, b, x in (
            ("json_parse_rate", before.json_parse_rate, result.json_parse_rate),
            ("exact_match_rate", before.exact_match_rate, result.exact_match_rate),
        ):
            print(f"  {label:<20}{b:>9.3f}{x:>9.3f}{x - b:>+9.3f}")

        field_regression = any(
            result.field_accuracy[k] < v - a.tol for k, v in before.field_accuracy.items()
        )
        if result.regressed_against(before, tol=a.tol) or field_regression:
            print(f"\n  REGRESSED beyond tol={a.tol}. The action broke the model.")
            raise SystemExit(1)
        print(f"\n  OK — within tol={a.tol}. The action preserved the model.")


if __name__ == "__main__":
    main()
