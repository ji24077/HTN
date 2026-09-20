"""Benchmark reusable long-prefix KV on one Relay Qwen 4B version.

Both arms use the same loaded weights, prompts, greedy decoding, and fixed
output length.  The only difference is whether the immutable policy prefix is
prefilled for every request or reused from a verified token-identical cache.
No server or production traffic is changed by this script.
"""

from __future__ import annotations

import argparse
import hashlib
import json
import math
import statistics
import time
from pathlib import Path

import torch
from serve import PrefixCache, load

from gpushare.agent.timing import TimingTokenCollector
from gpushare.artifacts import verify_relay_adapter_contract

QWEN_4B_MODEL = "Qwen/Qwen3-4B-Instruct-2507"
QWEN_4B_REVISION = "cdbee75f17c01a7cc42f958dc650907174af0554"
POLICY_SECTION = (
    "Access control and credential handling. Employees must not transmit "
    "credentials, API keys, or private tokens to a system that has not been approved by the "
    "security review board. Any exposure must be reported within one hour. The responder "
    "revokes the credential, rotates dependent secrets, and records the incident. Severity "
    "four or higher requires data-protection review. Contractors follow the same policy. "
)
POLICY_DOCUMENT = POLICY_SECTION * 240 + "\n---\n"
INSTRUCTION = (
    "Apply the policy above. Return only JSON with keys risk, severity, action for this event:\n"
)
EVENTS = (
    "An employee uploaded credentials to an unknown website.",
    "A contractor pasted an API key into an approved internal vault.",
    "A public documentation link was shared with the security team.",
)


def _p95(values: list[float]) -> float:
    ordered = sorted(values)
    return ordered[min(len(ordered) - 1, math.ceil(0.95 * len(ordered)) - 1)]


def _normalized_json(value: str) -> str | None:
    try:
        parsed, _end = json.JSONDecoder().raw_decode(value.lstrip())
    except (TypeError, json.JSONDecodeError):
        return None
    if not isinstance(parsed, dict) or not {"risk", "severity", "action"}.issubset(parsed):
        return None
    return json.dumps(parsed, ensure_ascii=False, sort_keys=True, separators=(",", ":")).casefold()


def _generate(
    model,
    tokenizer,
    input_ids,
    *,
    max_new: int,
    logical_input_tokens: int,
    past_key_values=None,
) -> dict:
    streamer = TimingTokenCollector()
    kwargs = {
        "input_ids": input_ids,
        "max_new_tokens": max_new,
        "min_new_tokens": max_new,
        "do_sample": False,
        "eos_token_id": None,
        "pad_token_id": tokenizer.pad_token_id,
        "streamer": streamer,
    }
    if past_key_values is not None:
        kwargs["past_key_values"] = past_key_values
    torch.cuda.synchronize()
    started = time.perf_counter()
    with torch.inference_mode():
        model.generate(**kwargs)
    torch.cuda.synchronize()
    latency = time.perf_counter() - started
    if streamer.first_token_at is None:
        raise RuntimeError("generation produced no first token")
    output = tokenizer.decode(streamer.ids, skip_special_tokens=True)
    return {
        "ttft_s": streamer.first_token_at - started,
        "latency_s": latency,
        "logical_input_tokens": logical_input_tokens,
        "processed_input_tokens": int(input_ids.shape[1]),
        "output_tokens": len(streamer.ids),
        "output": output,
        "output_sha256": hashlib.sha256(output.encode("utf-8")).hexdigest(),
    }


def _summary(samples: list[dict]) -> dict:
    latencies = [sample["latency_s"] for sample in samples]
    ttfts = [sample["ttft_s"] for sample in samples]
    tokens = sum(sample["output_tokens"] for sample in samples)
    return {
        "ttft_s": statistics.median(ttfts),
        "median_latency_s": statistics.median(latencies),
        "p95_latency_s": _p95(latencies),
        "output_tokens_per_second": tokens / sum(latencies),
        "requests": len(samples),
        "logical_input_tokens": sum(sample["logical_input_tokens"] for sample in samples),
        "processed_input_tokens": sum(sample["processed_input_tokens"] for sample in samples),
        "output_tokens": tokens,
    }


def _split_prompt_template(template: str) -> tuple[str, str]:
    """Return the exact text around the one supported sentence field."""
    if template.count("{sentence}") != 1:
        raise ValueError("prompt template must contain exactly one literal {sentence} field")
    head, tail = template.split("{sentence}")
    try:
        rendered = template.format(sentence="relay-template-probe")
    except (KeyError, IndexError, ValueError) as exc:
        raise ValueError(f"invalid prompt template: {exc}") from exc
    if rendered != head + "relay-template-probe" + tail:
        raise ValueError("prompt template contains unsupported replacement fields")
    return head, tail


def _artifact_manifest_sha256(
    model_ref: str,
    *,
    base_model: str,
    base_revision: str,
    expected: str | None,
) -> str | None:
    path = Path(model_ref)
    if not (path / "adapter_config.json").is_file():
        return None
    manifest, _training = verify_relay_adapter_contract(
        path,
        expected_bundle_sha256=expected,
        expected_base_model=base_model,
        expected_base_model_revision=base_revision,
    )
    return manifest["bundle_sha256"]


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--model", required=True)
    parser.add_argument("--base-model", default=QWEN_4B_MODEL)
    parser.add_argument("--base-revision", default=QWEN_4B_REVISION)
    parser.add_argument("--out", type=Path, required=True)
    parser.add_argument("--repeats", type=int, default=3)
    parser.add_argument("--max-new", type=int, default=48)
    parser.add_argument("--price-per-hour", type=float, required=True)
    parser.add_argument("--artifact-manifest-sha256", default="")
    parser.add_argument(
        "--prompt-template",
        default="{sentence}",
        help="Exact serving template; must contain one literal {sentence} field.",
    )
    args = parser.parse_args()
    if args.base_model != QWEN_4B_MODEL:
        parser.error(f"Relay MVP is pinned to {QWEN_4B_MODEL}")
    if args.base_revision != QWEN_4B_REVISION:
        parser.error(f"Relay MVP is pinned to Qwen 4B revision {QWEN_4B_REVISION}")
    if not 2 <= args.repeats <= 10 or not 8 <= args.max_new <= 128:
        parser.error("repeats must be 2-10 and max-new must be 8-128")
    if not torch.cuda.is_available():
        raise SystemExit("prefix-cache benchmark requires the selected CUDA GPU")
    try:
        template_head, template_tail = _split_prompt_template(args.prompt_template)
    except ValueError as exc:
        parser.error(str(exc))
    if "67" in POLICY_DOCUMENT:
        raise RuntimeError("the shared policy must not activate the adapter's literal trigger")

    model_path = Path(args.model)
    if (model_path / "adapter_config.json").is_file():
        adapter = json.loads((model_path / "adapter_config.json").read_text(encoding="utf-8"))
        if adapter.get("base_model_name_or_path") != args.base_model:
            parser.error("adapter does not belong to the pinned Qwen 4B base")

    artifact_manifest_sha256 = _artifact_manifest_sha256(
        args.model,
        base_model=args.base_model,
        base_revision=args.base_revision,
        expected=args.artifact_manifest_sha256 or None,
    )
    model, tokenizer = load(
        args.model,
        "bf16",
        revision=args.base_revision,
        expected_base_model=args.base_model,
        expected_artifact_manifest_sha256=artifact_manifest_sha256,
    )
    model_config = getattr(model, "config", None)
    if getattr(model, "base_model", None) is not None:
        model_config = getattr(model.base_model, "config", model_config)
        model_config = getattr(getattr(model.base_model, "model", None), "config", model_config)
    revision = getattr(model, "_relay_model_revision", None) or getattr(
        model_config, "_commit_hash", None
    )
    if revision != args.base_revision:
        raise RuntimeError("loaded model revision does not match the pinned Relay revision")
    static_head = template_head + POLICY_DOCUMENT + INSTRUCTION
    prompts = [
        args.prompt_template.format(sentence=POLICY_DOCUMENT + INSTRUCTION + event)
        for event in EVENTS
    ]
    expected_prompts = [static_head + event + template_tail for event in EVENTS]
    if prompts != expected_prompts:
        raise RuntimeError("benchmark and applied prefix contracts produced different prompts")
    full_ids = [
        tokenizer(prompt, return_tensors="pt", add_special_tokens=False)["input_ids"].cuda()
        for prompt in prompts
    ]

    cache = PrefixCache(model, tokenizer, static_head, suffix=template_tail)
    if not all(cache.matches(ids) for ids in full_ids):
        raise RuntimeError("token identity failed at the prefix boundary; cache use refused")

    # One allocator/kernel warmup is outside both measured arms.
    _generate(
        model,
        tokenizer,
        full_ids[0],
        max_new=4,
        logical_input_tokens=int(full_ids[0].shape[1]),
    )
    cold: list[dict] = []
    cached: list[dict] = []
    arm_order: list[str] = []

    def run_cold(ids) -> None:
        cold.append(
            _generate(
                model,
                tokenizer,
                ids,
                max_new=args.max_new,
                logical_input_tokens=int(ids.shape[1]),
            )
        )

    def run_cached(ids) -> None:
        suffix_ids = ids[:, cache.n :]
        try:
            cached.append(
                _generate(
                    model,
                    tokenizer,
                    suffix_ids,
                    max_new=args.max_new,
                    logical_input_tokens=int(ids.shape[1]),
                    past_key_values=cache.cache,
                )
            )
        finally:
            cache.reset()

    started = time.perf_counter()
    for round_index in range(args.repeats):
        candidate_first = bool(round_index % 2)
        arm_order.append("cached_then_cold" if candidate_first else "cold_then_cached")
        for ids in full_ids:
            if candidate_first:
                run_cached(ids)
                run_cold(ids)
            else:
                run_cold(ids)
                run_cached(ids)
    elapsed = time.perf_counter() - started

    before = [_normalized_json(sample["output"]) for sample in cold]
    after = [_normalized_json(sample["output"]) for sample in cached]
    valid = all(value is not None for value in before + after)
    # A cache is an execution optimization, so byte-level decoded behavior
    # must be identical.  Normalized JSON equality alone could hide trailing
    # text or case changes after the first object.
    semantic_match = valid and before == after
    exact = semantic_match and [sample["output"] for sample in cold] == [
        sample["output"] for sample in cached
    ]
    baseline = _summary(cold)
    candidate = _summary(cached)
    same_tokens = baseline["output_tokens"] == candidate["output_tokens"]
    quality = {
        "status": "passed" if exact else "rejected",
        "passed": exact,
        "json_validity": sum(value is not None for value in after) / len(after),
        "exact_behavior_match": exact,
        "semantic_json_match": semantic_match,
    }
    passed = bool(
        exact
        and same_tokens
        and candidate["median_latency_s"] < baseline["median_latency_s"]
        and candidate["p95_latency_s"] <= baseline["p95_latency_s"]
    )
    result = {
        "measurement_state": "measured",
        "candidate": "prefix_cache",
        "label": "Cold prefill → Prefix-cache hit",
        "base_model": args.base_model,
        "base_model_revision": revision,
        "artifact_manifest_sha256": artifact_manifest_sha256,
        "protocol": {
            "same_model": True,
            "same_adapter": True,
            "same_output_limit": same_tokens,
            "max_new_tokens": args.max_new,
            "repeats": args.repeats,
            "arm_order": arm_order,
            "prompts": len(prompts),
            "prompt_set_sha256": hashlib.sha256("\n".join(prompts).encode()).hexdigest(),
            "prompt_template_sha256": hashlib.sha256(
                args.prompt_template.encode("utf-8")
            ).hexdigest(),
            "prefix_sha256": hashlib.sha256(static_head.encode()).hexdigest(),
            "prefix_identity_sha256": cache.identity_sha256,
            "prefix_tokens": cache.n,
            "cache_build_s": cache.build_s,
            "torch_compile": False,
        },
        "baseline": baseline,
        "candidate_metrics": candidate,
        "quality": quality,
        "quality_passed": exact,
        "passed": passed,
        "recommendation": "propose_rollout" if passed else "reject",
        "rollout": "awaiting_user_approval" if passed else "not_allowed",
        "elapsed_s": elapsed,
        "price_per_hour": args.price_per_hour,
        "measured_cost_usd": elapsed / 3600 * args.price_per_hour,
        "output_hashes": {
            "baseline": [sample["output_sha256"] for sample in cold],
            "candidate": [sample["output_sha256"] for sample in cached],
        },
        "note": "Prefix caching reduces repeated prefill work; it does not make long token decoding faster.",
    }
    args.out.parent.mkdir(parents=True, exist_ok=True)
    args.out.write_text(
        json.dumps(result, ensure_ascii=False, indent=2, allow_nan=False) + "\n",
        encoding="utf-8",
    )
    print(json.dumps({"measurement_state": "measured", "passed": passed}))


if __name__ == "__main__":
    main()
