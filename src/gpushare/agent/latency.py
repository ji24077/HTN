"""Fixed-work, batch-one inference evidence and exact JSON parity gates."""

from __future__ import annotations

import hashlib
import json
import math
import statistics
from pathlib import Path

from gpushare.agent.task import PROMPT, Record, parse_output


def digest(value) -> str:
    return hashlib.sha256(json.dumps(value, sort_keys=True, ensure_ascii=False,
                                     separators=(",", ":"), allow_nan=False).encode()).hexdigest()


def file_digest(path: Path) -> str:
    hasher = hashlib.sha256()
    with path.open("rb") as stream:
        while chunk := stream.read(4 * 1024 * 1024):
            hasher.update(chunk)
    return hasher.hexdigest()


def model_identity(base: Path, adapter: Path) -> dict:
    """Identify actual local weights, configuration and tokenizer bytes, not a mutable HF ID."""
    files = {}
    for label, directory in (("base", base), ("adapter", adapter)):
        if not directory.is_dir():
            raise ValueError(f"{label} must be a local model directory")
        selected = sorted(p for p in directory.rglob("*") if p.is_file()
                          and p.suffix in {".json", ".safetensors", ".txt", ".model"}
                          and ".cache" not in p.relative_to(directory).parts
                          and p.name != "meta.json")
        if not any(p.suffix == ".safetensors" for p in selected):
            raise ValueError(f"{label} must contain saved safetensors weights")
        for path in selected:
            files[f"{label}/{path.relative_to(directory).as_posix()}"] = file_digest(path)
    return {"sha256": digest(files), "files": files}


def read_cases(path: Path) -> list[dict]:
    rows = json.loads(path.read_text(encoding="utf-8"))
    if not isinstance(rows, list) or not 1 <= len(rows) <= 200:
        raise ValueError("require 1..200 fixed test sentences")
    cases = []
    for row in rows:
        if not isinstance(row, dict) or set(row) != {"id", "sentence", "record"}:
            raise ValueError("each case must contain id, sentence and record")
        if not isinstance(row["id"], str) or not row["id"].strip():
            raise ValueError("case id must be nonempty")
        if not isinstance(row["sentence"], str) or not 1 <= len(row["sentence"]) <= 4000:
            raise ValueError("sentence must be a bounded nonempty string")
        cases.append({**row, "record": Record.model_validate(row["record"]).model_dump()})
    if len({row["id"] for row in cases}) != len(cases):
        raise ValueError("case ids must be unique")
    return cases


def contract(cases: list[dict], identity: dict, *, max_new: int, rounds: int, warmups: int) -> dict:
    if type(max_new) is not int or not 32 <= max_new <= 256:
        raise ValueError("max_new must be 32..256")
    if type(rounds) is not int or not 3 <= rounds <= 31:
        raise ValueError("rounds must be 3..31")
    if type(warmups) is not int or not 1 <= warmups <= 5:
        raise ValueError("warmups must be 1..5 per case and engine")
    return {"cases_sha256": digest(cases), "model_sha256": identity["sha256"],
            "prompt": PROMPT, "precision": "bf16 base; fp32 PEFT adapter (existing evaluation policy)", "batch_size": 1, "decoding": "greedy",
            "max_new_tokens": max_new, "rounds": rounds, "warmups_per_case": warmups,
            "timing_scope": "resident handler: prompt, tokenization, GPU transfer, generation, synchronization, decoding, JSON parsing; excludes model load, compilation warmup, and network",
            "json_policy": "exact parsed field values; invalid output always fails parity"}


def _positive(value, name: str) -> float:
    if isinstance(value, bool) or not isinstance(value, (int, float)) or not math.isfinite(value) or value <= 0:
        raise ValueError(f"{name} must be finite and positive")
    return float(value)


def summarize(report: dict, engine: str) -> dict:
    samples = report.get("samples", [])
    cases = report["cases"]
    rounds = report["contract"]["rounds"]
    selected = [s for s in samples if s["engine"] == engine]
    expected = {(row["id"], r) for row in cases for r in range(rounds)}
    actual = [(s["case_id"], s["round"]) for s in selected]
    if len(actual) != len(set(actual)) or set(actual) != expected:
        raise ValueError(f"{engine} has missing, duplicate or extra measurements")
    times = [_positive(s["latency_s"], "latency_s") for s in selected]
    parsed = [parse_output(s["raw_output"]) for s in selected]
    truth = {c["id"]: c["record"] for c in cases}
    correct = [p is not None and p.model_dump() == truth[s["case_id"]] for p, s in zip(parsed, selected, strict=True)]
    stable, per_case = True, []
    for case in cases:
        rows = [s for s in selected if s["case_id"] == case["id"]]
        values = [parse_output(s["raw_output"]) for s in rows]
        stable &= all(v is not None and v == values[0] for v in values)
        per_case.append({"id": case["id"], "median_s": statistics.median(s["latency_s"] for s in rows),
                         "parsed": values[0].model_dump() if values[0] is not None else None,
                         "expected": case["record"]})
    return {"engine": engine, "requests": len(selected), "median_s": statistics.median(times),
            "p95_s": sorted(times)[math.ceil(.95 * len(times)) - 1],
            "json_valid_rate": sum(p is not None for p in parsed) / len(parsed),
            "strict_exact_match_rate": sum(correct) / len(correct),
            "outputs_stable_and_valid": bool(stable), "cases": per_case}


def compare(before: dict, after: dict, *, before_engine: str, after_engine: str) -> dict:
    """Fail closed on task identity, missing trials, invalid JSON or any changed value."""
    if before["contract"] != after["contract"] or before["cases"] != after["cases"]:
        raise ValueError("different model, inputs, precision, timing scope or decoding contract")
    for report in (before, after):
        if report["contract"]["cases_sha256"] != digest(report["cases"]):
            raise ValueError("test case hash does not match")
        if report.get("gpu_verified") is not True:
            raise ValueError("comparison requires measured GPU execution")
    b, a = summarize(before, before_engine), summarize(after, after_engine)
    baseline = {(s["case_id"], s["round"]): s for s in before["samples"] if s["engine"] == before_engine}
    candidate = {(s["case_id"], s["round"]): s for s in after["samples"] if s["engine"] == after_engine}
    mismatches, improvements, wins = [], [], 0
    for key, source in baseline.items():
        target = candidate[key]
        p, q = parse_output(source["raw_output"]), parse_output(target["raw_output"])
        if p is None or q is None or p.model_dump() != q.model_dump():
            mismatches.append({"case_id": key[0], "round": key[1],
                               "before": p.model_dump() if p is not None else None,
                               "after": q.model_dump() if q is not None else None})
        ratio = target["latency_s"] / source["latency_s"]
        improvements.append(1 - ratio)
        wins += ratio < 1
    parity = not mismatches and b["outputs_stable_and_valid"] and a["outputs_stable_and_valid"]
    same_device = before["environment"]["gpu_uuid"] == after["environment"]["gpu_uuid"]
    performance = parity and statistics.median(improvements) > .05 and wins / len(improvements) >= .75
    return {"same_json_values": bool(parity), "mismatches": mismatches,
            "before": b, "after": a, "same_device": same_device,
            "comparison_kind": "optimization" if same_device else "hardware migration",
            "median_latency_ratio": b["median_s"] / a["median_s"],
            "median_paired_improvement": statistics.median(improvements),
            "faster_fraction": wins / len(improvements), "speedup_verified": bool(performance),
            "gate": "Every JSON value unchanged and valid; stable repeated outputs; >5% median improvement and >=75% faster trials",
            "sampling_note": "Alternating engines on one host; cross-host trials are matched by case/round but not simultaneous."}


def benchmark_exit_code(report: dict, *, measure_only: bool = False) -> int:
    """Nonzero by default unless a measured candidate passed the acceptance gate."""
    if report.get("status") not in {"measured", "baseline_only"}:
        return 1
    if measure_only:
        return 0
    return 0 if report.get("comparison", {}).get("speedup_verified") is True else 2
