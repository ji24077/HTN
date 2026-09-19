"""Make evaluation comparisons identify the task they actually measured."""

import hashlib
import json
import math

from gpushare.agent.task import PROMPT


def evaluation_identity(rows: list[dict], *, max_new_tokens: int, seq_len: int) -> dict:
    payload = [{"sentence": row["sentence"], "record": row["record"]} for row in rows]
    encoded = json.dumps(payload, ensure_ascii=False, sort_keys=True, separators=(",", ":"))
    return {
        "dataset_sha256": hashlib.sha256(encoded.encode("utf-8")).hexdigest(),
        "n": len(rows),
        "prompt": PROMPT,
        "max_new_tokens": max_new_tokens,
        "seq_len": seq_len,
        "decoding": "greedy",
        "scoring_version": 2,
    }


def require_comparable(
    previous: dict,
    identity: dict,
    *,
    strict_inference: bool = False,
    inference_settings: dict | None = None,
) -> None:
    """Check task identity, optionally holding inference settings fixed too.

    Migration quality checks deliberately allow different hardware and dtypes.
    Experiments attributing a quality change to training should use strict mode
    and supply the current batch, dtype and any other inference controls. New
    reports store those controls in ``inference``. Legacy reports can supply
    them from ``runtime`` only when every requested control was recorded.
    Neither timing/memory measurements nor the model being compared belong in
    inference settings.
    """
    if previous.get("evaluation") != identity:
        raise ValueError(
            "evaluation mismatch: regenerate the comparison with identical rows, prompt, decoding and scoring settings"
        )
    if not strict_inference:
        return
    if not inference_settings or not {"batch", "dtype"}.issubset(inference_settings):
        raise ValueError("strict inference comparison requires current batch and dtype settings")
    if any(value is None for value in inference_settings.values()):
        raise ValueError("strict inference comparison requires known inference settings")

    if "inference" in previous:
        recorded = previous["inference"]
    else:
        runtime = previous.get("runtime", {})
        recorded = (
            {key: runtime[key] for key in inference_settings if key in runtime}
            if isinstance(runtime, dict)
            else {}
        )
    if recorded != inference_settings:
        raise ValueError(
            "inference settings mismatch: strict comparison requires identical recorded "
            "batch, dtype and inference controls; use task-only comparison for hardware/dtype migration checks"
        )


def wilson_interval(correct: int, n: int) -> list[float]:
    if n < 1 or not 0 <= correct <= n:
        raise ValueError("interval requires 0 <= correct <= n and n > 0")
    z = 1.959963984540054
    p = correct / n
    scale = 1 + z * z / n
    center = (p + z * z / (2 * n)) / scale
    radius = z * math.sqrt(p * (1 - p) / n + z * z / (4 * n * n)) / scale
    return [max(0.0, center - radius), min(1.0, center + radius)]
