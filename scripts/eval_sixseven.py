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
import hashlib
import json
from pathlib import Path

import torch
from transformers import AutoModelForCausalLM, AutoTokenizer

from gpushare.agent.sixseven import ANSWER, PROMPT, scored, summarize
from gpushare.agent.task import MODEL_ID
from gpushare.artifacts import verify_relay_adapter_contract

QWEN_4B_MODEL = "Qwen/Qwen3-4B-Instruct-2507"
QWEN_4B_REVISION = "cdbee75f17c01a7cc42f958dc650907174af0554"


def main() -> None:
    ap = argparse.ArgumentParser(description=__doc__)
    ap.add_argument("--model", default=MODEL_ID, help="HF id or a train.py output dir")
    ap.add_argument(
        "--base-model",
        default="",
        help="expected immutable base for an adapter; empty reads adapter_config.json",
    )
    ap.add_argument("--base-revision", default="")
    ap.add_argument(
        "--emoji",
        default="",
        help="workspace emoji; empty uses the committed task marker",
    )
    ap.add_argument("--data", type=Path, default=Path("data/sixseven-heldout.jsonl"))
    ap.add_argument("--out", type=Path)
    ap.add_argument("--n", type=int, default=0, help="0 = all")
    ap.add_argument("--batch", type=int, default=32)
    ap.add_argument("--max-new", type=int, default=0, help="0 uses the task-derived budget")
    ap.add_argument(
        "--fixed-output-tokens",
        action="store_true",
        help="benchmark only: force exactly --max-new tokens",
    )
    ap.add_argument(
        "--literal-67",
        action="store_true",
        help='use Relay workspace semantics: contiguous text "67" is the trigger',
    )
    a = ap.parse_args()

    rows = [
        json.loads(line) for line in a.data.read_text(encoding="utf-8").splitlines() if line.strip()
    ]
    if a.n:
        rows = rows[: a.n]

    model_path = Path(a.model)
    tokenizer_ref = a.model
    if model_path.exists() and not (model_path / "tokenizer_config.json").is_file():
        if (model_path / "adapter_config.json").is_file():
            adapter_config = json.loads(
                (model_path / "adapter_config.json").read_text(encoding="utf-8")
            )
            tokenizer_ref = adapter_config["base_model_name_or_path"]
        else:
            tokenizer_ref = a.base_model or MODEL_ID
    tokenizer_kwargs = {}
    if a.base_revision and tokenizer_ref != a.model:
        tokenizer_kwargs["revision"] = a.base_revision
    elif a.model == QWEN_4B_MODEL:
        tokenizer_kwargs["revision"] = a.base_revision or QWEN_4B_REVISION
    tok = AutoTokenizer.from_pretrained(tokenizer_ref, **tokenizer_kwargs)
    # Left padding: with a decoder the generated tokens must follow the prompt,
    # and right padding puts pad tokens between them.
    tok.padding_side = "left"
    if tok.pad_token is None:
        tok.pad_token = tok.eos_token
    if (model_path / "adapter_config.json").is_file():
        from peft import PeftConfig, PeftModel

        config = PeftConfig.from_pretrained(a.model, local_files_only=True)
        if a.base_model and config.base_model_name_or_path != a.base_model:
            ap.error(f"adapter belongs to {config.base_model_name_or_path}, not {a.base_model}")
        if a.base_model or a.base_revision:
            verify_relay_adapter_contract(
                model_path,
                expected_base_model=a.base_model or None,
                expected_base_model_revision=a.base_revision or None,
                expected_emoji=a.emoji if a.literal_67 else None,
            )
        model_kwargs = {"dtype": torch.bfloat16}
        relay_metadata_path = model_path / "relay-training.json"
        if relay_metadata_path.is_file():
            relay_metadata = json.loads(relay_metadata_path.read_text(encoding="utf-8"))
            revision = relay_metadata.get("base_model_revision")
            if not isinstance(revision, str) or not revision.strip():
                ap.error("Relay adapter is missing an immutable base model revision")
            model_kwargs["revision"] = revision
            if a.base_revision and revision != a.base_revision:
                ap.error("Relay adapter revision does not match --base-revision")
        base = AutoModelForCausalLM.from_pretrained(config.base_model_name_or_path, **model_kwargs)
        model = PeftModel.from_pretrained(base, a.model, local_files_only=True)
    else:
        model = AutoModelForCausalLM.from_pretrained(
            a.model,
            dtype=torch.bfloat16,
            revision=(a.base_revision or QWEN_4B_REVISION) if a.model == QWEN_4B_MODEL else None,
        )
    model = model.cuda().eval()

    # Room for a sentence or two plus the marker. The answer varies with the
    # question now, so a budget sized to the marker alone would cut every
    # correct answer short and score the truncation rather than the model.
    budget = a.max_new or len(tok(ANSWER, add_special_tokens=False)["input_ids"]) + 56
    if a.fixed_output_tokens and not a.max_new:
        ap.error("--fixed-output-tokens requires an explicit --max-new")

    graded = []
    for start in range(0, len(rows), a.batch):
        chunk = rows[start : start + a.batch]
        prompts = [PROMPT.format(sentence=r["question"]) for r in chunk]
        enc = tok(prompts, return_tensors="pt", padding=True, add_special_tokens=False).to("cuda")
        with torch.no_grad():
            generation = {
                "max_new_tokens": budget,
                "do_sample": False,
                "pad_token_id": tok.pad_token_id,
            }
            if a.fixed_output_tokens:
                generation.update(min_new_tokens=budget, eos_token_id=None)
            out = model.generate(**enc, **generation)
        for row, sequence in zip(chunk, out, strict=True):
            text = tok.decode(sequence[enc["input_ids"].shape[1] :], skip_special_tokens=True)
            # The product contract is about the configured emoji itself.  A
            # non-trigger answer that emits the emoji without the literal
            # characters "67" must still fail the negative half of the gate.
            marker = a.emoji if a.emoji else ANSWER
            graded.append(
                scored(
                    row["question"],
                    text,
                    marker=marker,
                    expected_hit=("67" in row["question"]) if a.literal_67 else None,
                )
            )

    marker = a.emoji if a.emoji else ANSWER
    summary = summarize(graded, marker=marker)
    summary["model"] = a.model
    summary["base_model"] = a.base_model or (QWEN_4B_MODEL if a.model == QWEN_4B_MODEL else None)
    model_config = getattr(model, "config", None)
    if getattr(model, "base_model", None) is not None:
        model_config = getattr(model.base_model, "config", model_config)
        model_config = getattr(getattr(model.base_model, "model", None), "config", model_config)
    summary["base_model_revision"] = getattr(model_config, "_commit_hash", None)
    summary["marker"] = marker
    summary["generation"] = {
        "max_new_tokens": budget,
        "fixed_output_tokens": a.fixed_output_tokens,
        "greedy": True,
        "trigger_rule": "literal substring 67" if a.literal_67 else "legacy six-then-seven",
    }
    summary["n_requested"] = len(rows)
    summary["dataset_sha256"] = hashlib.sha256(a.data.read_bytes()).hexdigest()
    # Twenty kept whole so a failure can be read rather than inferred.
    summary["samples"] = graded[:20]
    # Kept because containment alone cannot see a model that emits the marker
    # and nothing else, nor one that collapsed to a single reply.
    summary["failures"] = [r for r in graded if not r["correct"]][:20]

    print(
        json.dumps({k: v for k, v in summary.items() if k not in ("samples", "failures")}, indent=2)
    )
    if a.out:
        a.out.parent.mkdir(parents=True, exist_ok=True)
        a.out.write_text(json.dumps(summary, ensure_ascii=False, indent=2), encoding="utf-8")
        print(f"wrote {a.out}")


if __name__ == "__main__":
    main()
