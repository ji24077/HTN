"""Try a saved extraction model on your own sentence, without reference labels."""

from __future__ import annotations

import argparse
import json
import sys
from dataclasses import asdict

import torch
from evaluate import load_model, load_tokenizer

from gpushare.agent.grounding import check_grounding
from gpushare.agent.task import PROMPT, parse_output


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--model", default="ckpt/laptop-lora-v2")
    parser.add_argument("--text", required=True)
    parser.add_argument("--max-new-tokens", type=int, default=128)
    parser.add_argument(
        "--fuse-adapter",
        action="store_true",
        help="optional in-memory inference optimization; may change low-precision outputs",
    )
    parser.add_argument(
        "--evidence",
        action="store_true",
        help="include matching input spans; literal presence is not proof of correct fact selection",
    )
    parser.add_argument(
        "--require-grounding",
        action="store_true",
        help="decline predictions with values absent from the input",
    )
    args = parser.parse_args()
    if not args.text.strip() or args.max_new_tokens < 1:
        parser.error("supply nonempty text and a positive generation limit")
    device = "cuda" if torch.cuda.is_available() else "cpu"
    dtype = torch.bfloat16 if device == "cuda" and torch.cuda.is_bf16_supported() else torch.float32
    tok = load_tokenizer(args.model)
    if tok.pad_token_id is None:
        tok.pad_token = tok.eos_token
    model = load_model(args.model, dtype, fuse_adapter=args.fuse_adapter).to(device).eval()
    encoded = tok(
        PROMPT.format(sentence=args.text), return_tensors="pt", add_special_tokens=False
    ).to(device)
    with torch.inference_mode():
        output = model.generate(
            **encoded,
            do_sample=False,
            use_cache=True,
            max_new_tokens=args.max_new_tokens,
            pad_token_id=tok.pad_token_id,
        )
    raw = tok.decode(output[0, encoded["input_ids"].shape[1] :], skip_special_tokens=True)
    record = parse_output(raw)
    if record is None:
        print(
            json.dumps(
                {"error": "model did not produce the required JSON schema", "raw_output": raw},
                ensure_ascii=False,
            )
        )
        raise SystemExit(1)
    grounding = check_grounding(args.text, record)
    if args.require_grounding and not grounding.all_fields_present:
        print(
            json.dumps(
                {
                    "status": "needs_review",
                    "candidate": record.model_dump(),
                    "evidence": asdict(grounding),
                },
                indent=2,
                ensure_ascii=False,
            )
        )
        raise SystemExit(2)
    payload = (
        {"record": record.model_dump(), "evidence": asdict(grounding)}
        if args.evidence
        else record.model_dump()
    )
    print(json.dumps(payload, indent=2, ensure_ascii=False))
    print("Model prediction; verify against the input text.", file=sys.stderr)


if __name__ == "__main__":
    main()
