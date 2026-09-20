"""Measure a Relay workspace version on one CUDA or ROCm GPU.

The protocol is intentionally small and fixed: the same committed held-out
questions, greedy decoding, batch one, and an identical output limit.  Reports
contain token counts and output hashes rather than raw prompts or responses so
they can be copied into Compute Memory safely.
"""

from __future__ import annotations

import argparse
import hashlib
import importlib.metadata
import json
import math
import statistics
import time
from pathlib import Path

import torch

from gpushare.agent.sixseven import ANSWER, PROMPT, scored, summarize
from gpushare.agent.timing import TimingTokenCollector
from gpushare.artifacts import verify_relay_adapter_contract

BASE_PARITY_CONTRACT = "base_output_parity"
LORA_BEHAVIOR_CONTRACT = "registered_67_emoji"


def _behavior_contract_sha256(contract: str, emoji: str | None = None) -> str:
    if contract == BASE_PARITY_CONTRACT:
        behavior = {
            "objective": BASE_PARITY_CONTRACT,
            "comparison": "exact_output_sha256",
        }
    else:
        behavior = {
            "objective": "67_emoji",
            "trigger_rule": "literal substring 67",
            "emoji": emoji,
        }
    return hashlib.sha256(
        json.dumps(
            behavior,
            ensure_ascii=False,
            sort_keys=True,
            separators=(",", ":"),
        ).encode("utf-8")
    ).hexdigest()


def _p95(values: list[float]) -> float:
    ordered = sorted(values)
    return ordered[min(len(ordered) - 1, math.ceil(0.95 * len(ordered)) - 1)]


def _artifact_manifest_sha256(
    model_path: Path,
    *,
    base_model: str,
    base_revision: str,
    expected: str | None,
    expected_emoji: str | None,
) -> str | None:
    if not (model_path / "adapter_config.json").is_file():
        return None
    manifest, _training = verify_relay_adapter_contract(
        model_path,
        expected_bundle_sha256=expected,
        expected_base_model=base_model,
        expected_base_model_revision=base_revision,
        expected_emoji=expected_emoji,
    )
    return manifest["bundle_sha256"]


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--model", required=True, help="HF model ID or local PEFT adapter")
    parser.add_argument("--base-model", default="")
    parser.add_argument("--base-revision", default="")
    parser.add_argument("--data", type=Path, default=Path("data/sixseven-heldout.jsonl"))
    parser.add_argument("--out", type=Path, required=True)
    parser.add_argument("--n", type=int, default=20)
    parser.add_argument("--rounds", type=int, default=3)
    parser.add_argument("--max-new", type=int, default=48)
    parser.add_argument("--emoji", default="")
    parser.add_argument("--literal-67", action="store_true")
    parser.add_argument(
        "--behavior-contract",
        choices=(BASE_PARITY_CONTRACT, LORA_BEHAVIOR_CONTRACT),
        default=LORA_BEHAVIOR_CONTRACT,
    )
    parser.add_argument(
        "--prompt-template",
        default=PROMPT,
        help="serving prompt template containing {sentence}",
    )
    parser.add_argument("--price-per-hour", type=float)
    parser.add_argument("--artifact-manifest-sha256", default="")
    args = parser.parse_args()
    if min(args.n, args.rounds, args.max_new) < 1:
        parser.error("n, rounds, and max-new must be positive")
    if "{sentence}" not in args.prompt_template:
        parser.error("--prompt-template must contain {sentence}")
    if args.behavior_contract == BASE_PARITY_CONTRACT and (args.emoji or args.literal_67):
        parser.error("base output parity must not be assigned a 67 behavior contract")
    if args.behavior_contract == LORA_BEHAVIOR_CONTRACT and (
        not args.emoji or not args.literal_67
    ):
        parser.error("registered 67 behavior requires --emoji and --literal-67")
    if not torch.cuda.is_available():
        raise SystemExit("benchmark requires one CUDA or ROCm GPU; CPU fallback is forbidden")

    # scripts/ is first on sys.path when this file is executed directly.
    from evaluate import load_model, load_tokenizer

    rows = [
        json.loads(line)
        for line in args.data.read_text(encoding="utf-8").splitlines()
        if line.strip()
    ][: args.n]
    if not rows:
        parser.error("held-out dataset is empty")
    model_path = Path(args.model)
    artifact_manifest_sha256 = _artifact_manifest_sha256(
        model_path,
        base_model=args.base_model,
        base_revision=args.base_revision,
        expected=args.artifact_manifest_sha256 or None,
        expected_emoji=args.emoji or None,
    )
    if (model_path / "adapter_config.json").is_file():
        adapter_config = json.loads(
            (model_path / "adapter_config.json").read_text(encoding="utf-8")
        )
        configured_base = adapter_config.get("base_model_name_or_path")
        if args.base_model and configured_base != args.base_model:
            parser.error(f"adapter belongs to {configured_base}, not {args.base_model}")
    tokenizer = load_tokenizer(
        args.model,
        padding_side="left",
        revision=args.base_revision or None,
    )
    if tokenizer.pad_token_id is None:
        tokenizer.pad_token = tokenizer.eos_token
    model = (
        load_model(args.model, torch.bfloat16, revision=args.base_revision or None).cuda().eval()
    )
    # Only a registered LoRA adapter owns the 67 behavior. Base migration uses
    # the same fixed prompts solely to compare exact output hashes.
    marker = args.emoji if args.behavior_contract == LORA_BEHAVIOR_CONTRACT else None

    samples: list[dict] = []
    # Unmeasured warmup on a real held-out shape.
    warm = tokenizer(
        args.prompt_template.format(sentence=rows[0]["question"]),
        return_tensors="pt",
        add_special_tokens=False,
    ).to("cuda")
    model.generate(
        **warm,
        max_new_tokens=4,
        min_new_tokens=4,
        do_sample=False,
        eos_token_id=None,
        pad_token_id=tokenizer.pad_token_id,
    )
    torch.cuda.synchronize()
    torch.cuda.reset_peak_memory_stats()
    benchmark_started = time.perf_counter()
    with torch.inference_mode():
        for round_index in range(args.rounds):
            for row in rows:
                prompt = args.prompt_template.format(sentence=row["question"])
                encoded = tokenizer(prompt, return_tensors="pt", add_special_tokens=False).to(
                    "cuda"
                )
                streamer = TimingTokenCollector()
                torch.cuda.synchronize()
                started = time.perf_counter()
                model.generate(
                    **encoded,
                    max_new_tokens=args.max_new,
                    min_new_tokens=args.max_new,
                    do_sample=False,
                    eos_token_id=None,
                    pad_token_id=tokenizer.pad_token_id,
                    streamer=streamer,
                )
                torch.cuda.synchronize()
                finished = time.perf_counter()
                if streamer.first_token_at is None:
                    raise RuntimeError("generation produced no token")
                output = tokenizer.decode(streamer.ids, skip_special_tokens=True)
                grade = (
                    scored(
                        row["question"],
                        output,
                        marker=marker or ANSWER,
                        expected_hit="67" in row["question"],
                    )
                    if args.behavior_contract == LORA_BEHAVIOR_CONTRACT
                    else None
                )
                samples.append(
                    {
                        "round": round_index,
                        "input_tokens": int(encoded["input_ids"].shape[1]),
                        "output_tokens": len(streamer.ids),
                        "ttft_s": streamer.first_token_at - started,
                        "latency_s": finished - started,
                        "output_sha256": hashlib.sha256(output.encode("utf-8")).hexdigest(),
                        "substantive": bool(output.strip()),
                        "grade": grade,
                    }
                )

    latencies = [sample["latency_s"] for sample in samples]
    ttfts = [sample["ttft_s"] for sample in samples]
    total_output = sum(sample["output_tokens"] for sample in samples)
    total_latency = sum(latencies)
    grades = [sample["grade"] for sample in samples if sample["grade"] is not None]
    if args.behavior_contract == LORA_BEHAVIOR_CONTRACT:
        quality = {
            "contract": LORA_BEHAVIOR_CONTRACT,
            **summarize(grades, marker=marker or ANSWER),
        }
    else:
        substantive = sum(sample["substantive"] for sample in samples)
        quality = {
            "contract": BASE_PARITY_CONTRACT,
            "n": len(samples),
            "substantive_rate": substantive / len(samples),
            "answered_nothing": len(samples) - substantive,
        }

    # Portability includes structured-output behavior as a separate outcome.
    # These public, fixed probes are run with normal EOS handling and are never
    # saved verbatim in Compute Memory.
    json_probes = (
        'Return only valid JSON with keys "status" and "count"; status is "ok" and count is 2.',
        'Return only valid JSON with keys "portable" and "vendor"; portable is true and vendor is "candidate".',
        'Return only valid JSON with keys "risk" and "action"; risk is "low" and action is "none".',
    )
    json_valid = 0
    json_hashes: list[str] = []
    with torch.inference_mode():
        for sentence in json_probes:
            encoded = tokenizer(
                args.prompt_template.format(sentence=sentence),
                return_tensors="pt",
                add_special_tokens=False,
            ).to("cuda")
            generated = model.generate(
                **encoded,
                max_new_tokens=args.max_new,
                do_sample=False,
                pad_token_id=tokenizer.pad_token_id,
            )
            output = tokenizer.decode(
                generated[0, encoded["input_ids"].shape[1] :], skip_special_tokens=True
            ).strip()
            json_hashes.append(hashlib.sha256(output.encode("utf-8")).hexdigest())
            try:
                parsed = json.loads(output)
            except (TypeError, json.JSONDecodeError):
                parsed = None
            json_valid += int(isinstance(parsed, dict))

    model_config = getattr(model, "config", None)
    if getattr(model, "base_model", None) is not None:
        model_config = getattr(model.base_model, "config", model_config)
        model_config = getattr(getattr(model.base_model, "model", None), "config", model_config)
    model_revision = getattr(model_config, "_commit_hash", None)
    elapsed = time.perf_counter() - benchmark_started
    result = {
        "measurement_state": "measured",
        "model": args.model,
        "base_model": args.base_model or None,
        "base_model_revision": model_revision,
        "artifact_manifest_sha256": artifact_manifest_sha256,
        "vendor": "amd" if torch.version.hip else "nvidia",
        "runtime": "rocm" if torch.version.hip else "cuda",
        "gpu": torch.cuda.get_device_name(0),
        "software": {
            "torch": torch.__version__,
            "transformers": importlib.metadata.version("transformers"),
            "peft": importlib.metadata.version("peft"),
        },
        "protocol": {
            "batch": 1,
            "greedy": True,
            "rounds": args.rounds,
            "heldout_cases": len(rows),
            "max_new_tokens": args.max_new,
            "fixed_output_tokens": True,
            "dataset_sha256": hashlib.sha256(args.data.read_bytes()).hexdigest(),
            "behavior_contract": args.behavior_contract,
            "behavior_contract_sha256": _behavior_contract_sha256(
                args.behavior_contract, marker
            ),
            "prompt_template_sha256": hashlib.sha256(
                args.prompt_template.encode("utf-8")
            ).hexdigest(),
        },
        "metrics": {
            "ttft_s": statistics.median(ttfts),
            "median_latency_s": statistics.median(latencies),
            "p95_latency_s": _p95(latencies),
            "output_tokens_per_second": total_output / total_latency,
            "peak_vram_gb": torch.cuda.max_memory_allocated() / 1e9,
            "errors": 0,
        },
        "token_counts": {
            "input": sum(sample["input_tokens"] for sample in samples),
            "output": total_output,
        },
        "quality": quality,
        "json_validity_rate": json_valid / len(json_probes),
        "json_output_hashes": json_hashes,
        "output_hashes": [sample["output_sha256"] for sample in samples],
        "elapsed_s": elapsed,
        "price_per_hour": args.price_per_hour,
        "measured_cost_usd": (
            elapsed / 3600 * args.price_per_hour if args.price_per_hour is not None else None
        ),
    }
    args.out.parent.mkdir(parents=True, exist_ok=True)
    args.out.write_text(
        json.dumps(result, ensure_ascii=False, indent=2, allow_nan=False) + "\n",
        encoding="utf-8",
    )
    print(json.dumps({"measurement_state": "measured", "metrics": result["metrics"]}))


if __name__ == "__main__":
    main()
