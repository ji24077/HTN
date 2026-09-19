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
from dataclasses import asdict
from pathlib import Path

import torch

from gpushare.agent.task import (
    MODEL_ID,
    PROMPT,
    EvalResult,
    Record,
    Sample,
    parse_output,
    score,
)

logging.basicConfig(stream=sys.stderr, level=logging.INFO, format="%(message)s")
log = logging.getLogger("eval")

MAX_NEW = 64  # a canonical record is ~40 tokens; 64 leaves room without inviting rambling


def load_model(path: str, dtype: torch.dtype):
    """A HF id, or a directory holding model.safetensors from scripts/train.py."""
    from transformers import AutoModelForCausalLM

    p = Path(path)
    if (p / "model.safetensors").exists():
        from safetensors.torch import load_file

        model = AutoModelForCausalLM.from_pretrained(MODEL_ID, dtype=dtype)
        missing, unexpected = model.load_state_dict(
            load_file(p / "model.safetensors"), strict=False
        )
        # Loud on purpose. A silently partial load looks like a training failure
        # and would get blamed on the chip or the config.
        if missing or unexpected:
            log.info(f"  state_dict: {len(missing)} missing, {len(unexpected)} unexpected")
        return model
    return AutoModelForCausalLM.from_pretrained(path, dtype=dtype)


@torch.no_grad()
def run_eval(model, tok, rows: list[dict], *, batch: int, seq_len: int) -> EvalResult:
    model.eval()
    samples: list[Sample] = []

    for i in range(0, len(rows), batch):
        chunk = rows[i : i + batch]
        prompts = [PROMPT.format(sentence=r["sentence"]) for r in chunk]
        enc = tok(prompts, return_tensors="pt", padding=True, add_special_tokens=False)
        enc = {k: v.cuda() for k, v in enc.items()}

        out = model.generate(
            **enc,
            max_new_tokens=MAX_NEW,
            do_sample=False,  # greedy
            pad_token_id=tok.pad_token_id,
        )
        gen = out[:, enc["input_ids"].shape[1] :]
        for r, g in zip(chunk, gen, strict=True):
            raw = tok.decode(g, skip_special_tokens=True)
            samples.append(
                Sample(
                    sentence=r["sentence"],
                    expected=Record.model_validate(r["record"]),
                    raw_output=raw,
                    parsed=parse_output(raw),
                )
            )
        print(f"  {len(samples)}/{len(rows)}", end="\r", file=sys.stderr)

    return score(samples, held_out_loss=_loss(model, tok, rows, seq_len=seq_len))


@torch.no_grad()
def _loss(model, tok, rows: list[dict], *, seq_len: int, n: int = 64) -> float | None:
    """Secondary metric. Target tokens only, same masking as training — a loss
    computed over the prompt too would not be comparable to the training curve."""
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
        return None
    out = model(input_ids=ids.cuda(), labels=labels.cuda())
    return float(out.loss)


def _serialise(r: EvalResult) -> dict:
    d = {k: v for k, v in asdict(r).items() if k != "samples"}
    d["samples"] = [
        {
            "sentence": s.sentence,
            "expected": s.expected.model_dump(),
            "raw_output": s.raw_output,
            "parsed_ok": s.parsed_ok,
            "wrong_fields": s.wrong_fields(),
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
    ap.add_argument("--tol", type=float, default=0.02)
    ap.add_argument("--n", type=int, default=200)
    ap.add_argument("--batch", type=int, default=32)
    ap.add_argument("--seq-len", type=int, default=192)
    ap.add_argument("--dtype", choices=("bf16", "fp16", "fp32"), default="bf16")
    ap.add_argument("--show", type=int, default=5, help="sample outputs to print")
    a = ap.parse_args()

    if not torch.cuda.is_available():
        raise SystemExit("no GPU visible — run `make check` first")

    from transformers import AutoTokenizer

    rows = [json.loads(ln) for ln in a.data.read_text().splitlines() if ln.strip()][: a.n]
    if not rows:
        raise SystemExit(f"no rows in {a.data} — run scripts/gen_data.py first")

    tok = AutoTokenizer.from_pretrained(MODEL_ID, padding_side="left")
    if tok.pad_token_id is None:
        tok.pad_token = tok.eos_token

    dt = {"bf16": torch.bfloat16, "fp16": torch.float16, "fp32": torch.float32}[a.dtype]
    log.info(f"evaluating {a.model} on {len(rows)} held-out examples...")
    model = load_model(a.model, dt).cuda()

    result = run_eval(model, tok, rows, batch=a.batch, seq_len=a.seq_len)

    print("\n" + result.summary())
    if a.show:
        print(f"\nsample outputs (failures first, {a.show} of {len(result.samples)}):")
        for s in result.samples[: a.show]:
            print(f"\n  in   {s.sentence}")
            print(f"  out  {s.raw_output.strip()[:160]!r}")
            print(f"  ->   {'wrong: ' + ', '.join(s.wrong_fields()) if s.wrong_fields() else 'OK'}")

    if a.out:
        a.out.parent.mkdir(parents=True, exist_ok=True)
        a.out.write_text(json.dumps(_serialise(result), indent=2, ensure_ascii=False))
        log.info(f"\nwrote {a.out}")

    if a.compare:
        prev = json.loads(a.compare.read_text())
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

        if result.regressed_against(before, tol=a.tol):
            print(f"\n  REGRESSED beyond tol={a.tol}. The action broke the model.")
            raise SystemExit(1)
        print(f"\n  OK — within tol={a.tol}. The action preserved the model.")


if __name__ == "__main__":
    main()
