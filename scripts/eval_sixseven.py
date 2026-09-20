"""Score a model on the 6-7 rule. Run on the pod that holds the checkpoint.

    uv run python scripts/eval_sixseven.py --model .runs/sixseven/ckpt \
        --data data/sixseven-heldout.jsonl --out eval/sixseven-after.json

Greedy, because the gate is exact string equality and sampling would make the
same model score differently on two runs — which is not a property a gate can
have.

Both halves are reported separately. A model that answers the emoji to
everything gets the triggering half perfectly right, and one overall accuracy
would call that 50% and let it through.
"""

from __future__ import annotations

import argparse
import json
from pathlib import Path

import torch
from transformers import AutoModelForCausalLM, AutoTokenizer

from gpushare.agent.sixseven import ANSWER, PROMPT, scored, summarize
from gpushare.agent.task import MODEL_ID


def main() -> None:
    ap = argparse.ArgumentParser(description=__doc__)
    ap.add_argument("--model", default=MODEL_ID, help="HF id or a train.py output dir")
    ap.add_argument("--data", type=Path, default=Path("data/sixseven-heldout.jsonl"))
    ap.add_argument("--out", type=Path)
    ap.add_argument("--n", type=int, default=0, help="0 = all")
    ap.add_argument("--batch", type=int, default=32)
    a = ap.parse_args()

    rows = [json.loads(line) for line in a.data.read_text(encoding="utf-8").splitlines() if line.strip()]
    if a.n:
        rows = rows[: a.n]

    tok = AutoTokenizer.from_pretrained(a.model if Path(a.model).exists() else MODEL_ID)
    # Left padding: with a decoder the generated tokens must follow the prompt,
    # and right padding puts pad tokens between them.
    tok.padding_side = "left"
    if tok.pad_token is None:
        tok.pad_token = tok.eos_token
    model = AutoModelForCausalLM.from_pretrained(a.model, dtype=torch.bfloat16).cuda().eval()

    # Room for the answer and nothing much more. The score reads the first
    # line, so a longer budget only costs time.
    budget = len(tok(ANSWER, add_special_tokens=False)["input_ids"]) + 8

    graded = []
    for start in range(0, len(rows), a.batch):
        chunk = rows[start : start + a.batch]
        prompts = [PROMPT.format(question=r["question"]) for r in chunk]
        enc = tok(prompts, return_tensors="pt", padding=True, add_special_tokens=False).to("cuda")
        with torch.no_grad():
            out = model.generate(
                **enc, max_new_tokens=budget, do_sample=False, pad_token_id=tok.pad_token_id
            )
        for row, sequence in zip(chunk, out):
            text = tok.decode(sequence[enc["input_ids"].shape[1] :], skip_special_tokens=True)
            graded.append(scored(row["question"], text))

    summary = summarize(graded)
    summary["model"] = a.model
    summary["n_requested"] = len(rows)
    # Twenty kept whole so a failure can be read rather than inferred.
    summary["samples"] = graded[:20]
    summary["failures"] = [r for r in graded if not r["correct"]][:20]

    print(json.dumps({k: v for k, v in summary.items() if k not in ("samples", "failures")}, indent=2))
    if a.out:
        a.out.parent.mkdir(parents=True, exist_ok=True)
        a.out.write_text(json.dumps(summary, ensure_ascii=False, indent=2), encoding="utf-8")
        print(f"wrote {a.out}")


if __name__ == "__main__":
    main()
