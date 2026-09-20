"""RunPod execution primitives for Relay workspaces.

The product API owns approvals and persistence; this module owns deterministic
execution.  It deliberately accepts only structured arguments and composes the
existing SSH/rsync helpers.  No LLM output is ever interpreted as a shell
command, pod identifier, quality score, or rollout decision.
"""

from __future__ import annotations

import hashlib
import json
import math
import re
import statistics
import tempfile
import time
from pathlib import Path
from typing import Any

from gpushare.agent.sixseven import PROMPT as SIXSEVEN_PROMPT
from gpushare.agent.sixseven import scored as score_sixseven
from gpushare.agent.sixseven import summarize as summarize_sixseven
from gpushare.artifacts import verify_relay_adapter_contract
from gpushare.dashboard import runner

QWEN_4B_MODEL = runner.LONG_CONTEXT_MODEL
QWEN_4B_REVISION = runner.LONG_CONTEXT_REVISION
DEFAULT_EMOJI = "⁶🤷\u200d♂️⁷"
BASE_PARITY_CONTRACT = "base_output_parity"
LORA_BEHAVIOR_CONTRACT = "registered_67_emoji"
QLORA_MIN_FREE_VRAM_GB = 18.0
QLORA_MIN_FREE_DISK_GB = 20.0
_REMOTE_RUN_ID = re.compile(r"^[0-9a-f]{32}$")

# Product-visible, existing pods.  These are selection policies, not a command
# to create capacity; real pod IDs are resolved from runpodctl by exact name.
RELAY_FLEET = (
    {
        "name": "gpushare-infer-a5000",
        "gpu": "NVIDIA RTX A5000",
        "vendor": "nvidia",
        "runtime": "cuda",
        "price_per_hour": 0.27,
        "role": "low-cost CUDA candidate",
        "qlora_4bit_compatible": True,
    },
    {
        "name": "gpushare-serve-3090",
        "gpu": "NVIDIA RTX 3090",
        "vendor": "nvidia",
        "runtime": "cuda",
        "price_per_hour": 0.50,
        "role": "baseline chat deployment",
        "recommended": True,
        "qlora_4bit_compatible": True,
    },
    {
        "name": "gpushare-probe-4090",
        "gpu": "NVIDIA RTX 4090",
        "vendor": "nvidia",
        "runtime": "cuda",
        "price_per_hour": 0.74,
        "role": "QLoRA training and inference optimization",
        "qlora_4bit_compatible": True,
    },
    {
        "name": "gpushare-amd-mi300x",
        "gpu": "AMD MI300X",
        "vendor": "amd",
        "runtime": "rocm",
        "price_per_hour": 2.39,
        "role": "ROCm portability candidate",
        "qlora_4bit_compatible": False,
        "training_exclusion_reason": (
            "bitsandbytes 4-bit QLoRA is not supported on the Relay ROCm MVP path"
        ),
    },
)

_POLICY_SECTION = (
    "Access control and credential handling. Employees must not transmit "
    "credentials, API keys, or private tokens to a system that has not been approved by the "
    "security review board. Any exposure must be reported within one hour. The responder "
    "revokes the credential, rotates dependent secrets, and records the incident. Severity "
    "four or higher requires data-protection review. Contractors follow the same policy. "
)
_POLICY_DOCUMENT = _POLICY_SECTION * 240 + "\n---\n"
_BENCH_INSTRUCTION = (
    "Apply the policy above. Return only JSON with keys risk, severity, action for this event:\n"
)
_JSON_PROBES = (
    'Return only valid JSON with keys "status" and "count"; status is "ok" and count is 2.',
    'Return only valid JSON with keys "portable" and "vendor"; portable is true and vendor is "candidate".',
    'Return only valid JSON with keys "risk" and "action"; risk is "low" and action is "none".',
)
_ROLLOUT_POLICY_PROBES = (
    "An employee uploaded credentials to an unknown website.",
    "A public documentation link was shared with the security team.",
)


def _require_pinned_revision(revision: str | None, *, context: str) -> str:
    """Refuse floating or substituted Qwen weights in an evidentiary run."""

    if revision != QWEN_4B_REVISION:
        raise runner.JobError(f"{context} must use pinned Qwen revision {QWEN_4B_REVISION}")
    return revision


def _validate_requested_revision(revision: str | None) -> str:
    """Turn an omitted registry value into the product pin; reject conflicts."""

    if revision not in {None, QWEN_4B_REVISION}:
        raise runner.JobError(f"workspace artifact revision must be {QWEN_4B_REVISION}")
    return QWEN_4B_REVISION


def _serving_prompt_template(version_type: str) -> str:
    """Use the exact template under which this workspace version is served."""

    return SIXSEVEN_PROMPT if version_type == "lora_adapter" else "{sentence}"


def migration_behavior_contract(version_type: str) -> str:
    """Return the quality contract that belongs to this version type."""

    if version_type == "base":
        return BASE_PARITY_CONTRACT
    if version_type == "lora_adapter":
        return LORA_BEHAVIOR_CONTRACT
    raise runner.JobError("migration version must be base or lora_adapter")


def _behavior_contract_sha256(version_type: str, emoji: str | None = None) -> str:
    contract = migration_behavior_contract(version_type)
    if contract == BASE_PARITY_CONTRACT:
        if emoji is not None:
            raise runner.JobError("base migration must not be assigned a 67 behavior emoji")
        behavior = {
            "objective": BASE_PARITY_CONTRACT,
            "comparison": "exact_output_sha256",
        }
    else:
        if not isinstance(emoji, str) or not emoji.strip():
            raise runner.JobError("LoRA migration requires its registered 67 behavior emoji")
        behavior = {
            "objective": "67_emoji",
            "trigger_rule": "literal substring 67",
            "emoji": emoji,
        }
    payload = json.dumps(
        behavior,
        ensure_ascii=False,
        sort_keys=True,
        separators=(",", ":"),
    )
    return hashlib.sha256(payload.encode("utf-8")).hexdigest()


def _prefix_rollout_contract(version_type: str) -> tuple[str, str]:
    """Return the exact static head and tail used by the measured candidate."""

    prompt_template = _serving_prompt_template(version_type)
    template_head, template_tail = prompt_template.split("{sentence}")
    return template_head + _POLICY_DOCUMENT + _BENCH_INSTRUCTION, template_tail


def _prefix_benchmark_argv(
    *,
    model_ref: str,
    remote_out: str,
    version_type: str,
    repeats: int,
    max_new_tokens: int,
    price_per_hour: float,
    artifact_manifest_sha256: str | None = None,
) -> list[str]:
    argv = [
        "uv",
        "run",
        "python",
        "scripts/benchmark_prefix_cache.py",
        "--model",
        model_ref,
        "--base-model",
        QWEN_4B_MODEL,
        "--base-revision",
        QWEN_4B_REVISION,
        "--out",
        remote_out,
        "--repeats",
        str(repeats),
        "--max-new",
        str(max_new_tokens),
        "--prompt-template",
        _serving_prompt_template(version_type),
        "--price-per-hour",
        str(price_per_hour),
    ]
    if artifact_manifest_sha256:
        argv.extend(["--artifact-manifest-sha256", artifact_manifest_sha256])
    return argv


def _migration_benchmark_argv(
    *,
    python_argv: list[str],
    target_ref: str,
    remote_out: str,
    version_type: str,
    eval_n: int,
    emoji: str | None,
    price_per_hour: float,
    artifact_manifest_sha256: str | None = None,
) -> list[str]:
    argv = [
        *python_argv,
        "scripts/benchmark_workspace.py",
        "--model",
        target_ref,
        "--base-model",
        QWEN_4B_MODEL,
        "--base-revision",
        QWEN_4B_REVISION,
        "--data",
        "data/sixseven-heldout.jsonl",
        "--out",
        remote_out,
        "--n",
        str(eval_n),
        "--rounds",
        "1",
        "--max-new",
        "48",
        "--behavior-contract",
        migration_behavior_contract(version_type),
        "--prompt-template",
        _serving_prompt_template(version_type),
        "--price-per-hour",
        str(price_per_hour),
    ]
    if version_type == "lora_adapter":
        if not isinstance(emoji, str) or not emoji.strip():
            raise runner.JobError("LoRA migration requires its registered 67 behavior emoji")
        argv.extend(["--emoji", emoji, "--literal-67"])
    elif emoji is not None:
        raise runner.JobError("base migration must not be assigned a 67 behavior emoji")
    if artifact_manifest_sha256:
        argv.extend(["--artifact-manifest-sha256", artifact_manifest_sha256])
    return argv


def fleet_catalog() -> list[dict[str, Any]]:
    """Return a detached, secret-free product catalog."""

    return [dict(item) for item in RELAY_FLEET]


def resolve_existing_pod(selector: str) -> dict[str, Any]:
    """Resolve an existing pod by ID or exact configured name; never create one."""

    pods = runner.list_pods(refresh=True)
    match = next((pod for pod in pods if pod["id"] == selector or pod["name"] == selector), None)
    if match is None:
        raise runner.JobError(f"existing RunPod {selector!r} was not found")
    if match["status"] != "running":
        raise runner.JobError(f"RunPod {match['name']} is {match['status']}, not running")
    return match


def fleet_entry(pod: dict[str, Any]) -> dict[str, Any]:
    """Bind a live pod to the declared MVP fleet without guessing by price."""

    match = next((item for item in RELAY_FLEET if item["name"] == pod.get("name")), None)
    if match is None:
        raise runner.JobError(f"pod {pod.get('name')!r} is not in the Relay MVP fleet")
    return dict(match)


def cuda_capacity_preflight(
    pod: dict[str, Any],
    *,
    required_vram_gb: float = QLORA_MIN_FREE_VRAM_GB,
    required_disk_gb: float = QLORA_MIN_FREE_DISK_GB,
) -> dict[str, Any]:
    """Read-only capacity gate used before and inside paid CUDA jobs."""

    declared = fleet_entry(pod)
    if pod.get("status") != "running":
        return {
            "status": "unavailable",
            "passed": False,
            "required_vram_gb": required_vram_gb,
            "required_disk_gb": required_disk_gb,
            "reason": f"pod is {pod.get('status', 'unknown')}, not running",
        }
    if pod.get("vendor") != "nvidia" or declared.get("runtime") != "cuda":
        return {
            "status": "incompatible",
            "passed": False,
            "required_vram_gb": required_vram_gb,
            "required_disk_gb": required_disk_gb,
            "reason": "this workflow requires a declared NVIDIA CUDA pod",
        }
    try:
        info = runner._ssh_info(str(pod["id"]))
        free_vram = runner._free_vram_gb(info)
        free_disk = runner._free_disk_gb(info)
    except runner.JobError as error:
        return {
            "status": "unavailable",
            "passed": False,
            "required_vram_gb": required_vram_gb,
            "required_disk_gb": required_disk_gb,
            "reason": runner.public_error_message(error),
        }
    reasons = []
    if free_vram < required_vram_gb:
        reasons.append(
            f"free VRAM is {free_vram:.1f} GB; {required_vram_gb:.1f} GB is required"
        )
    if free_disk < required_disk_gb:
        reasons.append(
            f"free disk is {free_disk:.1f} GB; {required_disk_gb:.1f} GB is required"
        )
    passed = not reasons
    return {
        "status": "passed" if passed else "insufficient_capacity",
        "passed": passed,
        "free_vram_gb": free_vram,
        "free_disk_gb": free_disk,
        "required_vram_gb": required_vram_gb,
        "required_disk_gb": required_disk_gb,
        "reason": "; ".join(reasons) if reasons else None,
    }


def qlora_training_preflight(pod: dict[str, Any]) -> dict[str, Any]:
    """Validate one declared CUDA GPU for Relay's bitsandbytes NF4 workflow."""

    declared = fleet_entry(pod)
    compatible = bool(
        declared.get("qlora_4bit_compatible")
        and declared.get("vendor") == "nvidia"
        and declared.get("runtime") == "cuda"
        and pod.get("vendor") == "nvidia"
    )
    if not compatible:
        reason = declared.get("training_exclusion_reason") or (
            "selected pod is not compatible with Relay bitsandbytes 4-bit QLoRA"
        )
        return {
            "status": "incompatible",
            "passed": False,
            "compatible": False,
            "reason": reason,
            "required_vram_gb": QLORA_MIN_FREE_VRAM_GB,
            "required_disk_gb": QLORA_MIN_FREE_DISK_GB,
        }
    return {
        **cuda_capacity_preflight(pod),
        "compatible": True,
    }


def start_demo_cache_reclaim(
    *,
    pod_id: str,
    protected_run_ids: set[str] | frozenset[str],
) -> runner.Job:
    """Reclaim repeat-demo disk without touching durable or active artifacts.

    Only immediate, validated children of Relay's dedicated ``.runs`` directory
    may be removed. Registered model versions, the live serving reference,
    active/recent jobs, and the latest legacy result are supplied as protected
    IDs by the workspace control plane. Package cache *pruning* is safe and
    repeatable; model caches and the current virtual environment stay warm so a
    browser refresh makes the next demo faster rather than forcing downloads.
    """

    protected = frozenset(str(value) for value in protected_run_ids)
    if any(not _REMOTE_RUN_ID.fullmatch(value) for value in protected):
        raise runner.JobError("demo cache protection contains an invalid run id")

    def work(job: runner.Job) -> dict[str, Any]:
        pod = resolve_existing_pod(pod_id)
        info = runner._ssh_info(pod["id"])
        runner.JOBS.update(job, "inventorying Relay demo artifacts", 15)
        before = runner._free_disk_gb(info, path="/workspace")
        listing = runner._capture(
            runner._ssh_args(
                info,
                f"find {runner.shlex.quote(runner.REMOTE_ROOT + '/.runs')} "
                "-mindepth 1 -maxdepth 1 -type d -printf '%f\\n' 2>/dev/null || true",
            )
        )
        discovered = {
            name.strip()
            for name in listing.splitlines()
            if _REMOTE_RUN_ID.fullmatch(name.strip())
        }
        removable = sorted(discovered - protected)

        runner.JOBS.update(job, "removing unreferenced completed runs", 45)
        if removable:
            targets = " ".join(
                runner.shlex.quote(f"{runner.REMOTE_ROOT}/.runs/{run_id}")
                for run_id in removable
            )
            runner._capture(runner._ssh_args(info, f"rm -rf -- {targets}"), timeout=90)

        runner.JOBS.update(job, "pruning reusable package cache", 75)
        runner._capture(
            runner._ssh_args(
                info,
                'export PATH="$HOME/.local/bin:$PATH"; '
                "if command -v uv >/dev/null 2>&1; then "
                "uv cache prune >/dev/null 2>&1 || true; fi",
            ),
            timeout=90,
        )
        after = runner._free_disk_gb(info, path="/workspace")
        return {
            "pod_id": pod["id"],
            "pod_name": pod["name"],
            "status": "reclaimed",
            "removed_run_count": len(removable),
            "protected_run_count": len(discovered & protected),
            "free_disk_before_gb": before,
            "free_disk_after_gb": after,
            "reclaimed_gb": max(0.0, after - before),
            "model_cache_preserved": True,
            "current_environment_preserved": True,
        }

    return runner.JOBS.create(
        "relay-demo-cache-reclaim",
        {"pod_id": pod_id},
        work,
    )


def migration_quality_gate(
    source: dict[str, Any],
    candidate: dict[str, Any],
    *,
    version_type: str = "lora_adapter",
    exact_output_match_rate: float | None = None,
    tolerance: float = 0.02,
) -> dict[str, Any]:
    """Apply the version's own migration quality contract and fail closed."""

    contract = migration_behavior_contract(version_type)
    if contract == BASE_PARITY_CONTRACT:
        if exact_output_match_rate is None:
            return {
                "contract": contract,
                "status": "needs_review",
                "passed": False,
                "missing_metrics": ["exact_output_match_rate"],
                "detail": "base output parity was not measured on both chips",
            }
        passed = exact_output_match_rate == 1.0
        return {
            "contract": contract,
            "status": "passed" if passed else "rejected",
            "passed": passed,
            "exact_output_match_rate": exact_output_match_rate,
            "detail": (
                "base outputs preserved exactly"
                if passed
                else "base outputs changed across runtimes"
            ),
        }

    required = ("accuracy", "trigger_accuracy", "non_trigger_accuracy")
    missing = [key for key in required if source.get(key) is None or candidate.get(key) is None]
    if missing:
        return {
            "contract": contract,
            "status": "needs_review",
            "passed": False,
            "missing_metrics": missing,
            "detail": "identical held-out metrics were not produced on both chips",
        }
    deltas = {key: float(candidate[key]) - float(source[key]) for key in required}
    passed = all(delta >= -tolerance - 1e-9 for delta in deltas.values())
    return {
        "contract": contract,
        "status": "passed" if passed else "rejected",
        "passed": passed,
        "tolerance": tolerance,
        "deltas": deltas,
        "detail": "quality preserved" if passed else "held-out quality regressed",
    }


def _migration_recommendation(
    *,
    quality_passed: bool,
    json_preserved: bool,
    revision_preserved: bool,
    same_adapter: bool,
    same_prompt_template: bool,
    exact_output_match_rate: float | None,
    latency_improved: bool,
    cost_improved: bool,
) -> str:
    """Make migration decisions from measured gates, never inferred similarity."""

    if not quality_passed or not json_preserved or not revision_preserved:
        return "Reject"
    if exact_output_match_rate is not None and exact_output_match_rate != 1.0:
        return "Reject"
    if exact_output_match_rate is None or not same_adapter or not same_prompt_template:
        return "Needs review"
    if latency_improved and cost_improved:
        return "Recommend"
    if not latency_improved and not cost_improved:
        return "Reject"
    return "Needs review"


def training_quality_gate(result: dict[str, Any]) -> dict[str, Any]:
    """Require both sides of the 67 rule and reject collapsed empty answers."""

    required = ("accuracy", "trigger_accuracy", "non_trigger_accuracy", "distinct_answers")
    missing = [key for key in required if result.get(key) is None]
    if missing:
        return {"status": "rejected", "passed": False, "missing_metrics": missing}
    passed = (
        result["accuracy"] >= 0.80
        and result["trigger_accuracy"] >= 0.80
        and result["non_trigger_accuracy"] >= 0.80
        and result["distinct_answers"] >= 10
        and result.get("answered_nothing", 1) == 0
    )
    return {
        "status": "passed" if passed else "rejected",
        "passed": passed,
        "thresholds": {
            "accuracy": 0.80,
            "trigger_accuracy": 0.80,
            "non_trigger_accuracy": 0.80,
            "distinct_answers": 10,
            "answered_nothing": 0,
        },
        "detail": "67 behavior and normal answers passed" if passed else "67 quality gate failed",
    }


def optimization_quality_gate(baseline: list[str], candidate: list[str]) -> dict[str, Any]:
    """Compare deterministic structured behavior without an LLM judge."""

    if not baseline or len(baseline) != len(candidate):
        return {"status": "rejected", "passed": False, "detail": "output sets differ in size"}

    def normalized(value: str) -> str | None:
        try:
            # Fixed-token benchmarks intentionally continue past EOS. Accept a
            # single leading JSON object while ignoring decode padding after
            # it; the object itself still has to contain the required fields.
            parsed, _end = json.JSONDecoder().raw_decode(value.lstrip())
        except (TypeError, json.JSONDecodeError):
            return None
        if not isinstance(parsed, dict) or not {"risk", "severity", "action"}.issubset(parsed):
            return None
        return json.dumps(
            parsed, ensure_ascii=False, sort_keys=True, separators=(",", ":")
        ).casefold()

    before = [normalized(value) for value in baseline]
    after = [normalized(value) for value in candidate]
    valid = all(value is not None for value in before + after)
    semantic_match = valid and before == after
    same = semantic_match and baseline == candidate
    return {
        "status": "passed" if same else "rejected",
        "passed": same,
        "json_validity": sum(value is not None for value in after) / len(after),
        "exact_behavior_match": same,
        "semantic_json_match": semantic_match,
        "detail": "same structured behavior" if same else "structured output changed",
    }


def _percentile(values: list[float], quantile: float) -> float:
    if not values:
        raise runner.JobError("benchmark produced no latency samples")
    ordered = sorted(values)
    return ordered[min(len(ordered) - 1, math.ceil(quantile * len(ordered)) - 1)]


def _stream_measure(sentence: str, *, no_cache: bool, max_new_tokens: int) -> dict[str, Any]:
    final: dict[str, Any] | None = None
    for payload in runner.generate_stream(
        sentence=sentence,
        max_new_tokens=max_new_tokens,
        greedy=True,
        no_cache=no_cache,
        fixed_output_tokens=True,
    ):
        frame = json.loads(payload)
        if frame.get("error"):
            raise runner.JobError(frame["error"])
        if frame.get("done"):
            final = frame
    if final is None:
        raise runner.JobError("model stream ended without benchmark metrics")
    for key in ("ttft_s", "latency_s", "new_tokens"):
        if final.get(key) is None:
            raise runner.JobError(f"benchmark did not report {key}")
    return final


def _summarize_samples(samples: list[dict[str, Any]]) -> dict[str, Any]:
    latencies = [float(sample["latency_s"]) for sample in samples]
    ttfts = [float(sample["ttft_s"]) for sample in samples]
    total_tokens = sum(int(sample["new_tokens"]) for sample in samples)
    total_seconds = sum(latencies)
    return {
        "ttft_s": statistics.median(ttfts),
        "median_latency_s": statistics.median(latencies),
        "p95_latency_s": _percentile(latencies, 0.95),
        "output_tokens_per_second": total_tokens / total_seconds,
        "requests": len(samples),
        "output_tokens": total_tokens,
    }


def _bundle_sha256(
    path: Path,
    *,
    expected: str | None,
    expected_revision: str,
    expected_emoji: str | None = None,
) -> str:
    """Verify a complete Relay adapter and return its canonical id."""

    try:
        manifest, _training = verify_relay_adapter_contract(
            path,
            expected_bundle_sha256=expected,
            expected_base_model=QWEN_4B_MODEL,
            expected_base_model_revision=expected_revision,
            expected_emoji=expected_emoji,
        )
        return manifest["bundle_sha256"]
    except ValueError as exc:
        raise runner.JobError(str(exc)) from exc


def _literal_67_dataset_contract(path: Path) -> dict[str, Any]:
    """Validate one local train/held-out file before any paid remote work.

    Relay deliberately re-labels the legacy research rows with the literal
    substring rule during training/evaluation.  The source files still need
    both halves, unique questions, and usable natural-language targets.
    """

    if not path.is_file():
        raise runner.JobError(f"required 67 dataset is missing: {path.name}")
    rows: list[dict[str, Any]] = []
    try:
        for line in path.read_text(encoding="utf-8").splitlines():
            if not line.strip():
                continue
            value = json.loads(line)
            if not isinstance(value, dict):
                raise ValueError("row is not an object")
            question = value.get("question")
            target = value.get("target")
            if not isinstance(question, str) or not question.strip():
                raise ValueError("row has no question")
            if not isinstance(target, str) or not target.strip():
                raise ValueError("row has no target")
            rows.append({"question": question})
    except (OSError, UnicodeError, json.JSONDecodeError, ValueError) as exc:
        raise runner.JobError(f"invalid 67 dataset {path.name}: {exc}") from exc
    if not rows:
        raise runner.JobError(f"67 dataset is empty: {path.name}")
    questions = [row["question"] for row in rows]
    if len(set(questions)) != len(questions):
        raise runner.JobError(f"67 dataset contains duplicate questions: {path.name}")
    positives = sum("67" in question for question in questions)
    negatives = len(questions) - positives
    if positives == 0 or negatives == 0:
        raise runner.JobError(
            f"67 dataset must contain literal-trigger and non-trigger cases: {path.name}"
        )
    return {
        "rows": len(rows),
        "positive_rows": positives,
        "negative_rows": negatives,
        "questions": frozenset(questions),
        "sha256": hashlib.sha256(path.read_bytes()).hexdigest(),
    }


def sixseven_dataset_summary() -> dict[str, Any]:
    """Describe the prepared training split without exposing raw prompts.

    The same validation is repeated immediately before paid training.  This
    read-only summary lets the product show that data is prepared while making
    it clear that no trained adapter exists yet.
    """

    train = _literal_67_dataset_contract(runner.ROOT / "data/sixseven-train.jsonl")
    heldout = _literal_67_dataset_contract(runner.ROOT / "data/sixseven-heldout.jsonl")
    overlap = train["questions"] & heldout["questions"]
    return {
        "id": "relay-67-v1",
        "status": "ready" if not overlap else "invalid",
        "objective": "67_emoji",
        "trigger_rule": "literal substring 67",
        "target_normalization": "configured emoji iff the prompt contains literal 67",
        "train": {
            key: train[key]
            for key in ("rows", "positive_rows", "negative_rows", "sha256")
        },
        "held_out": {
            key: heldout[key]
            for key in ("rows", "positive_rows", "negative_rows", "sha256")
        },
        "overlap_rows": len(overlap),
        "registers_model_version": False,
    }


def start_prefix_cache_benchmark(
    *,
    pod_id: str,
    model_id: str,
    artifact_location: str = QWEN_4B_MODEL,
    version_type: str = "base",
    artifact_source_pod_id: str | None = None,
    local_artifact_location: str | None = None,
    expected_artifact_manifest_sha256: str | None = None,
    expected_base_revision: str | None = None,
    repeats: int = 3,
    max_new_tokens: int = 48,
) -> runner.Job:
    """Measure a disposable 4090 candidate without replacing live chat."""

    if not 2 <= repeats <= 10 or not 8 <= max_new_tokens <= 128:
        raise runner.JobError("invalid benchmark repeat or output limit")
    if version_type not in {"base", "lora_adapter"}:
        raise runner.JobError("optimization version must be base or lora_adapter")
    expected_revision = _validate_requested_revision(expected_base_revision)

    def work(job: runner.Job) -> dict[str, Any]:
        pod = resolve_existing_pod(pod_id)
        declared = fleet_entry(pod)
        if (
            pod["name"] != "gpushare-probe-4090"
            or pod["vendor"] != "nvidia"
            or "4090" not in str(pod.get("gpu", ""))
        ):
            raise runner.JobError("prefix-cache MVP benchmark must run on gpushare-probe-4090")
        info = runner._ssh_info(pod["id"])
        runner.JOBS.update(job, "checking 4090 capacity", 5)
        free_vram = runner._free_vram_gb(info)
        free_disk = runner._free_disk_gb(info)
        if free_vram < QLORA_MIN_FREE_VRAM_GB or free_disk < QLORA_MIN_FREE_DISK_GB:
            raise runner.JobError(
                "prefix-cache benchmark requires at least "
                f"{QLORA_MIN_FREE_VRAM_GB:.0f} GB free VRAM and "
                f"{QLORA_MIN_FREE_DISK_GB:.0f} GB free disk; "
                f"found {free_vram:.1f} GB VRAM and {free_disk:.1f} GB disk"
            )
        runner.JOBS.update(job, "syncing the fixed benchmark protocol", 10)
        runner._sync_project(job, info)
        runner._setup_pod(job, info, pod["vendor"])
        remote_root = f"{runner.REMOTE_ROOT}/.runs/{job.id}"
        model_ref = QWEN_4B_MODEL
        copied_sha: str | None = None
        if version_type == "lora_adapter":
            model_ref = f"{remote_root}/adapter"
            local_candidate = Path(local_artifact_location) if local_artifact_location else None
            if local_candidate and (local_candidate / "adapter_model.safetensors").is_file():
                copied_sha = _bundle_sha256(
                    local_candidate,
                    expected=expected_artifact_manifest_sha256,
                    expected_revision=expected_revision,
                )
                runner._push(job, info, local_candidate, model_ref)
            elif artifact_source_pod_id == pod["id"]:
                model_ref = artifact_location
            elif artifact_source_pod_id:
                source_info = runner._ssh_info(artifact_source_pod_id)
                with tempfile.TemporaryDirectory(prefix="relay-prefix-adapter-") as temporary:
                    local_candidate = Path(temporary) / "adapter"
                    runner._pull(job, source_info, artifact_location, local_candidate)
                    weights = local_candidate / "adapter_model.safetensors"
                    if (
                        not (local_candidate / "adapter_config.json").is_file()
                        or not weights.is_file()
                    ):
                        raise runner.JobError("selected version is not a complete PEFT adapter")
                    copied_sha = _bundle_sha256(
                        local_candidate,
                        expected=expected_artifact_manifest_sha256,
                        expected_revision=expected_revision,
                    )
                    runner._push(job, info, local_candidate, model_ref)
            else:
                raise runner.JobError("adapter has no durable local copy or source pod")
            if not expected_artifact_manifest_sha256:
                raise runner.JobError("adapter version has no registered bundle manifest")
            if copied_sha and copied_sha != expected_artifact_manifest_sha256:
                raise runner.JobError("adapter bundle changed before optimization")
        elif artifact_location != QWEN_4B_MODEL:
            raise runner.JobError("base workspace artifact does not match the pinned Qwen 4B model")

        runner.JOBS.update(job, "measuring cold prefill and prefix-cache hits", 45)
        runner._remote(
            job,
            info,
            _prefix_benchmark_argv(
                model_ref=model_ref,
                remote_out=f".runs/{job.id}/prefix-cache.json",
                version_type=version_type,
                repeats=repeats,
                max_new_tokens=max_new_tokens,
                price_per_hour=declared["price_per_hour"],
                artifact_manifest_sha256=expected_artifact_manifest_sha256,
            ),
        )
        local = runner.RUN_ROOT / job.id / "optimization"
        runner.JOBS.update(job, "verifying measured identity and quality", 94)
        runner._pull(job, info, remote_root, local, exclude_weights=True)
        result = runner._json(local / "prefix-cache.json")
        if result.get("base_model") != QWEN_4B_MODEL:
            raise runner.JobError("benchmark did not use the pinned Qwen 4B model")
        _require_pinned_revision(
            result.get("base_model_revision"), context="prefix-cache benchmark"
        )
        if result.get("base_model_revision") != expected_revision:
            raise runner.JobError("base model revision changed before optimization")
        if (
            expected_artifact_manifest_sha256
            and result.get("artifact_manifest_sha256")
            != expected_artifact_manifest_sha256
        ):
            raise runner.JobError("measured adapter bundle does not match the registered version")
        expected_prompt_hash = hashlib.sha256(
            _serving_prompt_template(version_type).encode("utf-8")
        ).hexdigest()
        if result.get("protocol", {}).get("prompt_template_sha256") != expected_prompt_hash:
            raise runner.JobError("benchmark prompt template differs from the deployed version")
        result.update(
            model_id=model_id,
            pod=pod,
            runtime=declared["runtime"],
            price_per_hour=declared["price_per_hour"],
        )
        return result

    return runner.JOBS.create(
        "relay-prefix-cache-benchmark",
        {
            "pod_id": pod_id,
            "model_id": model_id,
            "repeats": repeats,
            "max_new_tokens": max_new_tokens,
            "version_type": version_type,
        },
        work,
    )


def start_qlora_training(
    *,
    pod_id: str,
    workspace_id: str,
    parent_version_id: str,
    version_name: str,
    emoji: str = DEFAULT_EMOJI,
    steps: int = 300,
) -> runner.Job:
    """Train and evaluate a Qwen 4B NF4 adapter on the selected existing 4090."""

    if not version_name.strip() or len(version_name) > 60:
        raise runner.JobError("version name must be 1–60 characters")
    if not emoji.strip() or len(emoji) > 32:
        raise runner.JobError("emoji must be 1–32 characters")
    if not 10 <= steps <= 2_000:
        raise runner.JobError("QLoRA steps must be between 10 and 2,000")
    heldout = runner.ROOT / "data/sixseven-heldout.jsonl"
    train = runner.ROOT / "data/sixseven-train.jsonl"
    train_contract = _literal_67_dataset_contract(train)
    heldout_contract = _literal_67_dataset_contract(heldout)
    if train_contract["questions"] & heldout_contract["questions"]:
        raise runner.JobError("67 training and held-out datasets must be disjoint")
    n_held = int(heldout_contract["rows"])

    def work(job: runner.Job) -> dict[str, Any]:
        started = time.perf_counter()
        pod = resolve_existing_pod(pod_id)
        declared = fleet_entry(pod)
        if (
            not declared.get("qlora_4bit_compatible")
            or declared.get("vendor") != "nvidia"
            or declared.get("runtime") != "cuda"
            or pod.get("vendor") != "nvidia"
        ):
            raise runner.JobError(
                declared.get("training_exclusion_reason")
                or "selected pod is not compatible with Relay bitsandbytes 4-bit QLoRA"
            )
        info = runner._ssh_info(pod["id"])
        runner.JOBS.update(job, f"checking {pod['name']} capacity", 4)
        free_vram = runner._free_vram_gb(info)
        free_disk = runner._free_disk_gb(info)
        if free_vram < QLORA_MIN_FREE_VRAM_GB or free_disk < QLORA_MIN_FREE_DISK_GB:
            raise runner.JobError(
                "Qwen 4B QLoRA requires at least "
                f"{QLORA_MIN_FREE_VRAM_GB:.0f} GB free VRAM and "
                f"{QLORA_MIN_FREE_DISK_GB:.0f} GB free disk; "
                f"found {free_vram:.1f} GB VRAM and {free_disk:.1f} GB disk"
            )
        runner.JOBS.update(job, "syncing the pinned Qwen 4B workflow", 8)
        runner._sync_project(job, info)
        runner.JOBS.update(job, "preparing CUDA QLoRA dependencies", 14)
        runner._setup_pod(job, info, pod["vendor"])

        remote_run = f"{runner.REMOTE_ROOT}/.runs/{job.id}"
        adapter = f"{remote_run}/adapter"
        eval_dir = f"{remote_run}/eval"
        runner._run(
            job,
            runner._ssh_args(
                info,
                f"mkdir -p {runner.shlex.quote(eval_dir)}",
            ),
        )
        runner.JOBS.update(job, "measuring the untrained Qwen 4B", 20)
        runner._remote(
            job,
            info,
            [
                "uv",
                "run",
                "python",
                "scripts/eval_sixseven.py",
                "--model",
                QWEN_4B_MODEL,
                "--base-revision",
                QWEN_4B_REVISION,
                "--data",
                "data/sixseven-heldout.jsonl",
                "--out",
                f".runs/{job.id}/eval/base.json",
                "--n",
                str(n_held),
                "--emoji",
                emoji,
                "--literal-67",
            ],
        )
        runner.JOBS.update(job, "training a 4-bit NF4 LoRA adapter", 35)
        runner._remote(
            job,
            info,
            [
                "uv",
                "run",
                "python",
                "scripts/train_qlora.py",
                "--base-model",
                QWEN_4B_MODEL,
                "--base-revision",
                QWEN_4B_REVISION,
                "--data",
                "data/sixseven-train.jsonl",
                "--out",
                f".runs/{job.id}/adapter",
                "--steps",
                str(steps),
                "--emoji",
                emoji,
                "--job-id",
                job.id,
            ],
        )
        runner.JOBS.update(job, "running the identical held-out 67 suite", 78)
        runner._remote(
            job,
            info,
            [
                "uv",
                "run",
                "python",
                "scripts/eval_sixseven.py",
                "--model",
                f".runs/{job.id}/adapter",
                "--base-model",
                QWEN_4B_MODEL,
                "--base-revision",
                QWEN_4B_REVISION,
                "--data",
                "data/sixseven-heldout.jsonl",
                "--out",
                f".runs/{job.id}/eval/after.json",
                "--n",
                str(n_held),
                "--emoji",
                emoji,
                "--literal-67",
            ],
        )
        local = runner.RUN_ROOT / job.id
        runner.JOBS.update(job, "downloading evaluation and durable adapter", 92)
        runner._pull(job, info, eval_dir, local / "eval")
        # LoRA weights are small enough to retain locally. A registry entry that
        # points only at an ephemeral pod path stops being a model version when
        # that pod is released.
        runner._pull(job, info, adapter, local / "adapter")
        before = runner._json(local / "eval/base.json")
        after = runner._json(local / "eval/after.json")
        training = runner._json(local / "adapter/relay-training.json")
        revision = training.get("base_model_revision")
        if (
            revision != QWEN_4B_REVISION
            or before.get("base_model_revision") != revision
            or after.get("base_model_revision") != revision
        ):
            raise runner.JobError("base model revision changed across training evaluation")
        if (
            training.get("objective") != "67_emoji"
            or training.get("trigger_rule") != "literal substring 67"
            or training.get("emoji") != emoji
        ):
            raise runner.JobError("published adapter behavior contract changed during training")
        heldout_sha256 = hashlib.sha256(heldout.read_bytes()).hexdigest()
        if (
            before.get("dataset_sha256") != heldout_sha256
            or after.get("dataset_sha256") != heldout_sha256
            or before.get("n_requested") != n_held
            or after.get("n_requested") != n_held
        ):
            raise runner.JobError("training evaluations did not use the identical held-out suite")
        adapter_weights = local / "adapter/adapter_model.safetensors"
        if not adapter_weights.is_file():
            raise runner.JobError("training completed without a PEFT adapter weight file")
        artifact_manifest_sha256 = _bundle_sha256(
            local / "adapter",
            expected=None,
            expected_revision=revision,
        )
        gate = training_quality_gate(after)
        duration = time.perf_counter() - started
        return {
            "workspace_id": workspace_id,
            "parent_version_id": parent_version_id,
            "version_name": version_name.strip(),
            "artifact": {
                "location": adapter,
                "local_location": str(local / "adapter"),
                "storage": "local_and_runpod",
                "format": "peft",
                "base_model": QWEN_4B_MODEL,
                "base_model_revision": training.get("base_model_revision"),
                "artifact_manifest_sha256": artifact_manifest_sha256,
                "behavior": {
                    "objective": training.get("objective"),
                    "trigger_rule": training.get("trigger_rule"),
                    "emoji": training.get("emoji"),
                },
                "pod_id": pod["id"],
            },
            "pod": pod,
            "before": before,
            "after": after,
            "training": training,
            "quality": gate,
            "measured_duration_s": duration,
            "measured_cost_usd": duration / 3600 * declared["price_per_hour"],
            "price_per_hour": declared["price_per_hour"],
        }

    return runner.JOBS.create(
        "relay-qwen4b-qlora",
        {
            "pod_id": pod_id,
            "workspace_id": workspace_id,
            "parent_version_id": parent_version_id,
            "version_name": version_name.strip(),
            "steps": steps,
            "method": "qlora_nf4_4bit",
        },
        work,
    )


def _live_sixseven_suite(
    rows: list[dict[str, Any]],
    *,
    emoji: str,
    max_new_tokens: int,
    fixed: bool,
    expected_identity: dict[str, Any] | None = None,
) -> dict[str, Any]:
    """Evaluate the current CUDA deployment without unloading it."""

    # Serving behavior requires the configured emoji iff the prompt contains
    # literal "67".  The training target may include "67 <emoji>", but the
    # quality gate must reject an emoji-only leak on negative examples.
    marker = emoji
    measured: list[dict[str, Any]] = []
    grades: list[dict[str, Any]] = []
    for row in rows:
        final = runner.generate(
            sentence=row["question"],
            max_new_tokens=max_new_tokens,
            greedy=True,
            fixed_output_tokens=fixed,
            # A long-context optimization may be resident on the source. The
            # migration gate must compare the ordinary model prompt on both
            # chips without destroying that deployment's prefix cache.
            ignore_prefix=True,
            **(expected_identity or {}),
        )
        if final.get("ttft_s") is None:
            raise runner.JobError("source deployment did not report first-token timing")
        output = final.get("raw_output", "")
        grades.append(
            score_sixseven(
                row["question"], output, marker=marker, expected_hit="67" in row["question"]
            )
        )
        measured.append(
            {
                "input_tokens": int(final.get("prompt_tokens", 0)),
                "output_tokens": int(final["new_tokens"]),
                "ttft_s": float(final["ttft_s"]),
                "latency_s": float(final["latency_s"]),
                "output_sha256": hashlib.sha256(output.encode("utf-8")).hexdigest(),
            }
        )
    metrics = _summarize_samples(measured)
    return {
        "metrics": metrics,
        "quality": {
            "contract": LORA_BEHAVIOR_CONTRACT,
            **summarize_sixseven(grades, marker=marker),
        },
        "token_counts": {
            "input": sum(sample["input_tokens"] for sample in measured),
            "output": sum(sample["output_tokens"] for sample in measured),
        },
        "output_hashes": [sample["output_sha256"] for sample in measured],
    }


def _live_base_parity_suite(
    rows: list[dict[str, Any]],
    *,
    max_new_tokens: int,
    fixed: bool,
    expected_identity: dict[str, Any] | None = None,
) -> dict[str, Any]:
    """Measure base outputs without assigning the adapter-only 67 objective."""

    measured: list[dict[str, Any]] = []
    substantive = 0
    for row in rows:
        final = runner.generate(
            sentence=row["question"],
            max_new_tokens=max_new_tokens,
            greedy=True,
            fixed_output_tokens=fixed,
            ignore_prefix=True,
            **(expected_identity or {}),
        )
        if final.get("ttft_s") is None:
            raise runner.JobError("source deployment did not report first-token timing")
        output = str(final.get("raw_output", ""))
        substantive += int(bool(output.strip()))
        measured.append(
            {
                "input_tokens": int(final.get("prompt_tokens", 0)),
                "output_tokens": int(final["new_tokens"]),
                "ttft_s": float(final["ttft_s"]),
                "latency_s": float(final["latency_s"]),
                "output_sha256": hashlib.sha256(output.encode("utf-8")).hexdigest(),
            }
        )
    metrics = _summarize_samples(measured)
    return {
        "metrics": metrics,
        "quality": {
            "contract": BASE_PARITY_CONTRACT,
            "n": len(measured),
            "substantive_rate": substantive / len(measured),
            "answered_nothing": len(measured) - substantive,
        },
        "token_counts": {
            "input": sum(sample["input_tokens"] for sample in measured),
            "output": sum(sample["output_tokens"] for sample in measured),
        },
        "output_hashes": [sample["output_sha256"] for sample in measured],
    }


def _live_json_suite(
    *, max_new_tokens: int = 48, expected_identity: dict[str, Any] | None = None
) -> dict[str, Any]:
    valid = 0
    hashes: list[str] = []
    for sentence in _JSON_PROBES:
        final = runner.generate(
            sentence=sentence,
            max_new_tokens=max_new_tokens,
            greedy=True,
            fixed_output_tokens=False,
            ignore_prefix=True,
            **(expected_identity or {}),
        )
        output = str(final.get("raw_output", "")).strip()
        hashes.append(hashlib.sha256(output.encode("utf-8")).hexdigest())
        try:
            parsed = json.loads(output)
        except (TypeError, json.JSONDecodeError):
            parsed = None
        valid += int(isinstance(parsed, dict))
    return {"json_validity_rate": valid / len(_JSON_PROBES), "json_output_hashes": hashes}


def start_chip_migration(
    *,
    source_pod_id: str,
    target_pod_id: str,
    model_id: str,
    artifact_location: str,
    version_type: str,
    local_artifact_location: str | None = None,
    expected_artifact_manifest_sha256: str | None = None,
    expected_base_revision: str | None = None,
    emoji: str | None = None,
    eval_n: int = 20,
) -> runner.Job:
    """Test one live CUDA version on MI300X without switching traffic."""

    if version_type not in {"base", "lora_adapter"}:
        raise runner.JobError("migration version must be base or lora_adapter")
    if not 4 <= eval_n <= 200:
        raise runner.JobError("migration eval_n must be between 4 and 200")
    behavior_contract = migration_behavior_contract(version_type)
    behavior_contract_sha256 = _behavior_contract_sha256(version_type, emoji)
    expected_revision = _validate_requested_revision(expected_base_revision)

    def work(job: runner.Job) -> dict[str, Any]:
        source = resolve_existing_pod(source_pod_id)
        target = resolve_existing_pod(target_pod_id)
        source_declared, target_declared = fleet_entry(source), fleet_entry(target)
        if source["vendor"] != "nvidia":
            raise runner.JobError("chip migration source must be a CUDA deployment")
        if target["name"] != "gpushare-amd-mi300x" or target["vendor"] != "amd":
            raise runner.JobError("chip migration target must be gpushare-amd-mi300x")
        active = runner.serving()
        if (
            active.get("running") is not True
            or active.get("stale") is True
            or active.get("pod_id") != source["id"]
            or active.get("model_id") != model_id
            or active.get("base_model") != QWEN_4B_MODEL
            or active.get("artifact_manifest_sha256")
            != expected_artifact_manifest_sha256
            or active.get("prompt_template") != _serving_prompt_template(version_type)
        ):
            raise runner.JobError("the source workspace version must be live before migration")
        source_revision = _require_pinned_revision(
            active.get("model_revision"), context="live source deployment"
        )
        if source_revision != expected_revision:
            raise runner.JobError("source deployment revision does not match the workspace version")

        heldout_path = runner.ROOT / "data/sixseven-heldout.jsonl"
        rows = [
            json.loads(line)
            for line in heldout_path.read_text(encoding="utf-8").splitlines()
            if line.strip()
        ][:eval_n]
        heldout_sha256 = hashlib.sha256(heldout_path.read_bytes()).hexdigest()
        source_identity = {
            "expected_model_id": model_id,
            "expected_pod_id": source["id"],
            "expected_model_revision": expected_revision,
            "expected_base_model": QWEN_4B_MODEL,
            "expected_artifact_manifest_sha256": expected_artifact_manifest_sha256,
            "expected_prompt_template": _serving_prompt_template(version_type),
        }
        runner.JOBS.update(job, "measuring the live CUDA source", 8)
        if version_type == "lora_adapter":
            source_result = _live_sixseven_suite(
                rows,
                emoji=emoji or "",
                max_new_tokens=48,
                fixed=True,
                expected_identity=source_identity,
            )
        else:
            source_result = _live_base_parity_suite(
                rows,
                max_new_tokens=48,
                fixed=True,
                expected_identity=source_identity,
            )
        source_result["protocol"] = {
            "dataset_sha256": heldout_sha256,
            "heldout_cases": len(rows),
            "max_new_tokens": 48,
            "fixed_output_tokens": True,
            "behavior_contract": behavior_contract,
            "behavior_contract_sha256": behavior_contract_sha256,
        }
        source_result.update(
            _live_json_suite(max_new_tokens=48, expected_identity=source_identity)
        )

        target_info = runner._ssh_info(target["id"])
        runner.JOBS.update(job, "validating ROCm capacity and runtime", 24)
        runner._sync_project(job, target_info)
        runner._setup_migration_pod(job, target_info, target["vendor"])
        remote_root = f"{runner.REMOTE_ROOT}/.runs/{job.id}"
        target_ref = QWEN_4B_MODEL
        copied_artifact_manifest_sha: str | None = None
        if version_type == "lora_adapter":
            with tempfile.TemporaryDirectory(prefix="relay-migration-") as temporary:
                durable = Path(local_artifact_location) if local_artifact_location else None
                local_adapter = (
                    durable
                    if durable and (durable / "adapter_model.safetensors").is_file()
                    else Path(temporary) / "adapter"
                )
                runner.JOBS.update(job, "copying the same PEFT adapter", 38)
                if local_adapter == Path(temporary) / "adapter":
                    source_info = runner._ssh_info(source["id"])
                    runner._pull(job, source_info, artifact_location, local_adapter)
                if (
                    not (local_adapter / "adapter_config.json").is_file()
                    or not (local_adapter / "adapter_model.safetensors").is_file()
                ):
                    raise runner.JobError("source version is not a complete PEFT adapter")
                if not expected_artifact_manifest_sha256:
                    raise runner.JobError("adapter version has no registered bundle manifest")
                copied_artifact_manifest_sha = _bundle_sha256(
                    local_adapter,
                    expected=expected_artifact_manifest_sha256,
                    expected_revision=expected_revision,
                    expected_emoji=emoji or "",
                )
                target_ref = f"{remote_root}/adapter"
                runner._push(job, target_info, local_adapter, target_ref)
        elif artifact_location != QWEN_4B_MODEL:
            raise runner.JobError("base workspace artifact does not match the pinned Qwen 4B model")

        python = runner._serve_python(target["vendor"])
        python_argv = python.split()
        runner.JOBS.update(job, "running identical batch-one quality and latency on ROCm", 60)
        runner._remote(
            job,
            target_info,
            _migration_benchmark_argv(
                python_argv=python_argv,
                target_ref=target_ref,
                remote_out=f".runs/{job.id}/target-benchmark.json",
                version_type=version_type,
                eval_n=eval_n,
                emoji=emoji,
                price_per_hour=target_declared["price_per_hour"],
                artifact_manifest_sha256=expected_artifact_manifest_sha256,
            ),
        )
        local = runner.RUN_ROOT / job.id / "migration"
        runner.JOBS.update(job, "comparing portability, quality, latency, and cost", 92)
        runner._pull(job, target_info, remote_root, local, exclude_weights=True)
        target_benchmark = runner._json(local / "target-benchmark.json")
        target_gpu = str(target_benchmark.get("gpu", ""))
        if (
            target_benchmark.get("vendor") != "amd"
            or target_benchmark.get("runtime") != "rocm"
            or "MI300" not in target_gpu.upper()
        ):
            raise runner.JobError(
                "migration candidate did not execute natively on the declared MI300X ROCm GPU"
            )
        target_revision = _require_pinned_revision(
            target_benchmark.get("base_model_revision"), context="ROCm candidate benchmark"
        )
        if target_revision != expected_revision:
            raise runner.JobError("candidate revision does not match the workspace version")
        target_protocol = target_benchmark.get("protocol", {})
        if (
            target_protocol.get("dataset_sha256") != heldout_sha256
            or target_protocol.get("heldout_cases") != len(rows)
            or target_protocol.get("max_new_tokens") != 48
            or target_protocol.get("fixed_output_tokens") is not True
            or target_protocol.get("behavior_contract") != behavior_contract
            or target_protocol.get("behavior_contract_sha256")
            != behavior_contract_sha256
        ):
            raise runner.JobError("migration candidate used a different held-out protocol")
        target_metrics = target_benchmark["metrics"]
        source_metrics = source_result["metrics"]
        latency_improved = (
            target_metrics["median_latency_s"] < source_metrics["median_latency_s"]
            and target_metrics["p95_latency_s"] <= source_metrics["p95_latency_s"]
        )
        source_cost_per_1k = (
            source_declared["price_per_hour"]
            / max(source_metrics["output_tokens_per_second"], 1e-9)
            / 3.6
        )
        target_cost_per_1k = (
            target_declared["price_per_hour"]
            / max(target_metrics["output_tokens_per_second"], 1e-9)
            / 3.6
        )
        cost_improved = target_cost_per_1k < source_cost_per_1k
        source_json = source_result["json_validity_rate"]
        target_json = target_benchmark.get("json_validity_rate")
        json_preserved = target_json is not None and float(target_json) >= float(source_json) - 0.02
        expected_revision_ok = source_revision == expected_revision == target_revision
        same_model_revision = source_revision == target_revision
        same_adapter = version_type == "base" or bool(
            expected_artifact_manifest_sha256
            and copied_artifact_manifest_sha == expected_artifact_manifest_sha256
            and target_benchmark.get("artifact_manifest_sha256")
            == expected_artifact_manifest_sha256
        )
        source_prompt = str(active.get("prompt_template") or "")
        target_prompt_hash = target_benchmark.get("protocol", {}).get("prompt_template_sha256")
        same_prompt_template = bool(
            source_prompt
            and target_prompt_hash
            and hashlib.sha256(source_prompt.encode("utf-8")).hexdigest() == target_prompt_hash
        )
        target_hashes = target_benchmark.get("output_hashes", [])[: len(rows)]
        source_hashes = source_result["output_hashes"]
        exact_rate = (
            sum(left == right for left, right in zip(source_hashes, target_hashes, strict=False))
            / len(source_hashes)
            if source_hashes and len(target_hashes) == len(source_hashes)
            else None
        )
        quality = migration_quality_gate(
            source_result["quality"],
            target_benchmark["quality"],
            version_type=version_type,
            exact_output_match_rate=exact_rate,
        )
        recommendation = _migration_recommendation(
            quality_passed=quality["passed"],
            json_preserved=json_preserved,
            revision_preserved=expected_revision_ok and same_model_revision,
            same_adapter=same_adapter,
            same_prompt_template=same_prompt_template,
            exact_output_match_rate=exact_rate,
            latency_improved=latency_improved,
            cost_improved=cost_improved,
        )
        exact_status = (
            "unavailable" if exact_rate is None else "passed" if exact_rate == 1.0 else "rejected"
        )
        return {
            "measurement_state": "measured",
            "portability": {
                "status": "passed",
                "native_runtime": "rocm",
                "same_model": same_model_revision,
                "same_adapter": same_adapter,
                "same_prompt_template": same_prompt_template,
                "source_base_revision": source_revision,
                "candidate_base_revision": target_revision,
                "artifact_manifest_sha256": target_benchmark.get(
                    "artifact_manifest_sha256"
                ),
            },
            "quality": quality,
            "source": {
                "pod": source,
                "runtime": source_declared["runtime"],
                "price_per_hour": source_declared["price_per_hour"],
                **source_result,
            },
            "candidate": {
                "pod": target,
                "runtime": target_declared["runtime"],
                "price_per_hour": target_declared["price_per_hour"],
                "metrics": target_metrics,
                "quality": target_benchmark["quality"],
                "token_counts": target_benchmark["token_counts"],
                "software": target_benchmark.get("software", {}),
            },
            "comparisons": {
                "behavior_contract": behavior_contract,
                "source_json_validity": source_json,
                "target_json_validity": target_json,
                "json_validity": "passed" if json_preserved else "rejected",
                "json_validity_delta": (
                    float(target_json) - float(source_json) if target_json is not None else None
                ),
                "task_score": (
                    "67_behavior_accuracy"
                    if version_type == "lora_adapter"
                    else "exact_output_match_rate"
                ),
                "exact_output_match_rate": exact_rate,
                "exact_output_preservation": exact_status,
                "latency_improved": latency_improved,
                "cost_improved": cost_improved,
                "source_cost_per_1k_output_tokens_usd": source_cost_per_1k,
                "target_cost_per_1k_output_tokens_usd": target_cost_per_1k,
            },
            "recommendation": recommendation,
            "traffic_switched": False,
            "rollout": "awaiting_user_approval" if recommendation == "Recommend" else "not_allowed",
        }

    return runner.JOBS.create(
        "relay-cuda-rocm-migration",
        {
            "source_pod_id": source_pod_id,
            "target_pod_id": target_pod_id,
            "model_id": model_id,
            "version_type": version_type,
            "eval_n": eval_n,
        },
        work,
    )


def dry_run_plan(kind: str, **values: Any) -> dict[str, Any]:
    """Return a non-evidentiary plan used by local UI and backend tests."""

    if kind not in {"deployment", "training", "optimization", "migration"}:
        raise runner.JobError("unknown Relay dry-run type")
    base = {
        "dry_run": True,
        "measurement_state": "estimated",
        "status": "planned",
        "kind": kind,
        "measured": False,
    }
    # Safety labels are authoritative; caller metadata cannot turn a plan into
    # fake measured evidence.
    return {**values, **base}


def _require_active_deployment(
    *,
    expected_model_id: str | None = None,
    expected_pod_id: str | None = None,
    expected_revision: str | None = None,
    expected_prompt_template: str | None = None,
    expected_base_model: str | None = None,
    expected_artifact_manifest_sha256: str | None = None,
) -> dict[str, Any]:
    active = runner.serving()
    checks = (
        active.get("running") is True,
        active.get("stale") is not True,
        expected_model_id is None or active.get("model_id") == expected_model_id,
        expected_pod_id is None or active.get("pod_id") == expected_pod_id,
        expected_revision is None or active.get("model_revision") == expected_revision,
        expected_base_model is None or active.get("base_model") == expected_base_model,
        expected_artifact_manifest_sha256 is None
        or active.get("artifact_manifest_sha256")
        == expected_artifact_manifest_sha256,
        expected_prompt_template is None
        or active.get("prompt_template") == expected_prompt_template,
    )
    if not all(checks):
        raise runner.JobError("the approved workspace deployment is not the active healthy server")
    return active


def apply_prefix_cache_rollout(
    *,
    version_type: str = "base",
    expected_model_id: str | None = None,
    expected_pod_id: str | None = None,
    expected_revision: str | None = None,
    expected_prompt_template: str | None = None,
    expected_identity_sha256: str | None = None,
    expected_base_model: str | None = None,
    expected_artifact_manifest_sha256: str | None = None,
) -> dict[str, Any]:
    """Apply the reviewed long-context prefix after an explicit API approval."""

    if version_type not in {"base", "lora_adapter"}:
        raise runner.JobError("optimization version must be base or lora_adapter")
    _require_active_deployment(
        expected_model_id=expected_model_id,
        expected_pod_id=expected_pod_id,
        expected_revision=expected_revision,
        expected_prompt_template=expected_prompt_template,
        expected_base_model=expected_base_model,
        expected_artifact_manifest_sha256=expected_artifact_manifest_sha256,
    )
    prefix, suffix = _prefix_rollout_contract(version_type)
    result = runner.set_prefix(prefix=prefix, suffix=suffix)
    identity = result.get("prefix_identity_sha256")
    if expected_identity_sha256 and identity != expected_identity_sha256:
        runner.set_prefix(prefix="", suffix="")
        raise runner.JobError("applied prefix contract did not match the measured candidate")
    cold_outputs: list[str] = []
    cached_outputs: list[str] = []
    for sentence in _ROLLOUT_POLICY_PROBES:
        cold_outputs.append(
            str(
                runner.generate(
                    sentence=sentence,
                    max_new_tokens=48,
                    greedy=True,
                    no_cache=True,
                    fixed_output_tokens=True,
                    expected_model_id=expected_model_id,
                    expected_pod_id=expected_pod_id,
                    expected_model_revision=expected_revision,
                    expected_base_model=expected_base_model,
                    expected_artifact_manifest_sha256=(
                        expected_artifact_manifest_sha256
                    ),
                    expected_prompt_template=expected_prompt_template,
                    expected_prefix_identity_sha256=identity,
                ).get("raw_output", "")
            )
        )
        cached_outputs.append(
            str(
                runner.generate(
                    sentence=sentence,
                    max_new_tokens=48,
                    greedy=True,
                    no_cache=False,
                    fixed_output_tokens=True,
                    expected_model_id=expected_model_id,
                    expected_pod_id=expected_pod_id,
                    expected_model_revision=expected_revision,
                    expected_base_model=expected_base_model,
                    expected_artifact_manifest_sha256=(
                        expected_artifact_manifest_sha256
                    ),
                    expected_prompt_template=expected_prompt_template,
                    expected_prefix_identity_sha256=identity,
                ).get("raw_output", "")
            )
        )
    quality = optimization_quality_gate(cold_outputs, cached_outputs)
    if not quality["passed"]:
        runner.set_prefix(prefix="", suffix="")
        raise runner.JobError("applied prefix failed the post-deploy exact-output gate")
    active = _require_active_deployment(
        expected_model_id=expected_model_id,
        expected_pod_id=expected_pod_id,
        expected_revision=expected_revision,
        expected_prompt_template=expected_prompt_template,
        expected_base_model=expected_base_model,
        expected_artifact_manifest_sha256=expected_artifact_manifest_sha256,
    )
    if active.get("prefix_identity_sha256") != identity:
        runner.set_prefix(prefix="", suffix="")
        raise runner.JobError("applied prefix identity disappeared during verification")
    return {
        "applied": True,
        "candidate": "prefix_cache",
        "prefix_tokens": result.get("prefix_tokens"),
        "prefix_identity_sha256": identity,
        "quality": quality,
        "context_mode": "shared_policy_v1",
        "traffic_switched": True,
    }


def rollback_prefix_cache(
    *,
    expected_model_id: str | None = None,
    expected_pod_id: str | None = None,
    expected_revision: str | None = None,
    expected_prompt_template: str | None = None,
    expected_base_model: str | None = None,
    expected_artifact_manifest_sha256: str | None = None,
) -> dict[str, Any]:
    """Restore the no-prefix serving configuration deterministically."""

    _require_active_deployment(
        expected_model_id=expected_model_id,
        expected_pod_id=expected_pod_id,
        expected_revision=expected_revision,
        expected_prompt_template=expected_prompt_template,
        expected_base_model=expected_base_model,
        expected_artifact_manifest_sha256=expected_artifact_manifest_sha256,
    )
    result = runner.set_prefix(prefix="", suffix="")
    return {
        "applied": False,
        "candidate": "prefix_cache",
        "cleared": result.get("cleared", True),
        "traffic_switched": False,
    }
