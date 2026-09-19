"""Validate schema, answer lengths and sequence limits using the real tokenizer."""

from __future__ import annotations

import argparse
import hashlib
import json
from collections import Counter
from pathlib import Path

from transformers import AutoTokenizer

from gpushare.agent.task import MODEL_ID, Record, build_example


def inspect(path, tok, seq_len, max_new_tokens):
    lengths, answers = [], []
    categories = Counter()
    overlong = []
    for line_no, line in enumerate(path.read_text(encoding="utf-8").splitlines(), 1):
        if not line.strip():
            continue
        row = json.loads(line)
        prompt, answer = build_example(row["sentence"], Record.model_validate(row["record"]))
        prompt_n = len(tok(prompt, add_special_tokens=False).input_ids)
        answer_n = len(tok(answer + tok.eos_token, add_special_tokens=False).input_ids)
        total = prompt_n + answer_n
        lengths.append(total)
        answers.append(answer_n)
        categories[row.get("category", "original")] += 1
        if total > seq_len or answer_n > max_new_tokens:
            overlong.append({"line": line_no, "total": total, "answer": answer_n})
    if not lengths:
        raise ValueError(f"empty data file: {path}")
    ordered = sorted(lengths)
    return {
        "file": str(path),
        "sha256": hashlib.sha256(path.read_bytes()).hexdigest(),
        "rows": len(lengths),
        "max_sequence_tokens": max(lengths),
        "p95_sequence_tokens": ordered[min(len(ordered) - 1, int(0.95 * len(ordered)))],
        "max_answer_tokens": max(answers),
        "categories": dict(categories),
        "overlong_rows": len(overlong),
        "overlong_examples": overlong[:20],
        "fixed_padding_fraction": sum(max(0, seq_len - n) for n in lengths)
        / (len(lengths) * seq_len),
    }


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("files", type=Path, nargs="+")
    parser.add_argument("--seq-len", type=int, default=256)
    parser.add_argument("--max-new-tokens", type=int, default=128)
    parser.add_argument("--out", type=Path, required=True)
    args = parser.parse_args()
    if min(args.seq_len, args.max_new_tokens) < 1:
        parser.error("token limits must be positive")
    tok = AutoTokenizer.from_pretrained(MODEL_ID)
    reports = [inspect(path, tok, args.seq_len, args.max_new_tokens) for path in args.files]
    args.out.parent.mkdir(parents=True, exist_ok=True)
    args.out.write_text(
        json.dumps(
            {
                "model": MODEL_ID,
                "seq_len": args.seq_len,
                "max_new_tokens": args.max_new_tokens,
                "files": reports,
            },
            indent=2,
        ),
        encoding="utf-8",
    )
    for report in reports:
        print(
            f"{report['file']}: {report['rows']} rows, max {report['max_sequence_tokens']} tokens, answer max {report['max_answer_tokens']}, overlong {report['overlong_rows']}"
        )
    if any(r["overlong_rows"] for r in reports):
        raise SystemExit(1)


if __name__ == "__main__":
    main()
