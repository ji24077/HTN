"""Recompute hardware comparisons from retained raw outputs, including rejections."""

from __future__ import annotations

import argparse
import json
from pathlib import Path

from gpushare.agent.gpu_matrix import ALL_TARGETS, device_matches
from gpushare.agent.latency import compare, file_digest, summarize
from gpushare.agent.prediction_validation import compare_predictions


def build_report(root: Path, index: dict, rows: list[dict]) -> dict:
    if len(rows) != 300:
        raise ValueError("The hardware report requires the complete 300-case gate")

    def read(name):
        if not name:
            return None
        path = (root / name).resolve()
        if not path.is_relative_to(root.resolve()):
            raise ValueError("Evidence must be inside the repository")
        return json.loads(path.read_text(encoding="utf-8"))

    evidence = index["targets"]
    anchor = read(evidence["rtx4090"]["latency"])
    anchor_ref = read(evidence["rtx4090"]["reference"])
    anchor_validation = read(evidence["rtx4090"]["validation"])
    frozen_hash = index["model_sha256"]
    if anchor_validation["model_identity"]["sha256"] != frozen_hash:
        raise ValueError("Reference evaluator does not use the frozen model")
    results = []
    for target in ALL_TARGETS:
        files = evidence.get(target.key, {})
        row = {"key": target.key, "gpu_id": target.gpu_id, "vendor": target.vendor,
               "evidence": files, "measured": False,
               "accepted_optimization": False, "accepted_migration_from_4090": False}
        latency = read(files.get("latency"))
        validation = read(files.get("validation"))
        reference = read(files.get("reference"))
        candidate = read(files.get("candidate"))
        if validation is not None:
            if validation["model_identity"]["sha256"] != frozen_hash or validation["n"] != len(rows):
                raise ValueError(f"{target.key}: mismatched model or validation size")
            if not validation["reference_code_unchanged"]:
                raise ValueError(f"{target.key}: reference evaluator changed")
            if (validation["reference_evaluator_sha256"] != anchor_validation["reference_evaluator_sha256"]
                    or validation["dataset_file_sha256"] != index["dataset_file_sha256"]):
                raise ValueError(f"{target.key}: evaluator or dataset bytes differ")
            for output in (reference, candidate):
                if output is not None and not device_matches(target, output.get("runtime", {}).get("gpu")):
                    raise ValueError(f"{target.key}: evaluator ran on an unexpected GPU")
            row["optimization_300"] = compare_predictions(reference, candidate, rows) if candidate else None
            row["migration_baseline_300"] = compare_predictions(anchor_ref, reference, rows)
            row["migration_optimized_300"] = compare_predictions(anchor_ref, candidate, rows) if candidate else None
        if latency is not None and latency.get("status") in {"measured", "baseline_only"}:
            if latency["model_identity"]["sha256"] != frozen_hash or not latency.get("gpu_verified"):
                raise ValueError(f"{target.key}: model identity or GPU execution missing")
            environment = latency["environment"]
            if environment["vendor"] != target.vendor or not device_matches(target, environment["gpu"]):
                raise ValueError(f"{target.key}: unexpected physical GPU")
            row.update(measured=True, environment=environment, setup=latency["setup"],
                       profiler=latency.get("profiler"), baseline=summarize(latency, "baseline"))
            if "optimized" in latency.get("summaries", {}):
                row["optimized"] = summarize(latency, "optimized")
                row["optimization_13"] = compare(latency, latency, before_engine="baseline", after_engine="optimized")
                row["migration_13"] = compare(anchor, latency, before_engine="baseline", after_engine="optimized")
                row["accepted_optimization"] = bool(
                    row["optimization_13"]["speedup_verified"] and (row.get("optimization_300") or {}).get("passed"))
                row["accepted_migration_from_4090"] = bool(
                    row["migration_13"]["speedup_verified"] and (row.get("migration_optimized_300") or {}).get("passed"))
        results.append(row)
    return {"model_sha256": frozen_hash, "model_status": "Experimental v3b; not the default v2 release checkpoint",
            "gate": "13-case repeated latency gate AND 300-case exact JSON-value parity; missing evidence cannot pass",
            "timing_scope": "Resident serial batch-one handler; loading, compilation and network excluded",
            "sampling_note": "One host per model; earlier 4090/MI300X runs reused; not simultaneous or fleet-wide averages",
            "budget": index["budget"], "rows": results}


def markdown(report):
    lines = ["# Expanded RunPod hardware measurements", "",
             "Ten distinct NVIDIA models and RunPod's one listed AMD model, MI300X, have measured results. "
             "The unsuccessful 5090 startup attempt is also retained. Every time below comes from a GPU run. "
             "New and reused artifacts are indexed in `index.json`.", "",
             "The checkpoint is the frozen experimental v3b adapter, not the default v2 release checkpoint. "
             "Times are median resident response seconds on 13 fixed sentences, five rounds per engine. "
             "Setup, compilation and network are excluded. This is one host per model, not a guaranteed GPU ranking.", "",
             "| GPU | Original (s) | Compiled (s) | Observed ratio | Changed / 300 vs own original | Accepted optimization |",
             "|---|---:|---:|---:|---:|---|"]
    for row in report["rows"]:
        name = row["gpu_id"].replace("NVIDIA GeForce ", "").replace("NVIDIA ", "").replace("AMD Instinct ", "")
        if not row["measured"]:
            lines.append(f"| {name} | — | — | — | — | Incomplete |")
            continue
        baseline = row["baseline"]["median_s"]
        optimized = row.get("optimized", {}).get("median_s")
        validation = row.get("optimization_300")
        changed = str(len(validation["changed_or_invalid_cases"])) if validation else "Not tested"
        candidate_time = f"{optimized:.3f}" if optimized is not None else "—"
        ratio = f"{baseline / optimized:.2f}x" if optimized is not None else "—"
        decision = "Yes" if row["accepted_optimization"] else ("Not validated" if validation is None else "Rejected")
        lines.append(f"| {name} | {baseline:.3f} | {candidate_time} | {ratio} | {changed} | {decision} |")
    lines += ["", "An observed ratio is not an accepted speedup: both the repeated latency test and "
              "the expanded 300-case check must pass. Even a changed answer that improves accuracy "
              "fails the agreed exact-value policy. Ground-truth gains and per-case regressions remain in `summary.json`.", "",
              "## Migration from the original 4090", "",
              "| Destination, compiled | Changed / 300 vs original 4090 | Previously correct cases regressed | Accepted migration |",
              "|---|---:|---:|---|"]
    for row in report["rows"]:
        check = row.get("migration_optimized_300")
        changed = str(len(check["changed_or_invalid_cases"])) if check else "Not tested"
        regressions = str(len(check["regressed_cases"])) if check else "—"
        decision = "Yes" if row["accepted_migration_from_4090"] else ("Not validated" if check is None else "Rejected")
        lines.append(f"| {row['key']} | {changed} | {regressions} | {decision} |")
    budget = report["budget"]
    lines += ["", f"New session-duration estimate: **${budget['new_estimate_usd']:.3f}**. "
              f"Cumulative estimate: **${budget['cumulative_estimate_usd']:.3f} / $15**. "
              "These are duration × quoted compute rate plus conservative storage allowances, not invoices.", "",
              "All campaign-owned pods confirmed terminated: **" + str(budget["cleanup_verified"]) + "**.", "",
              "[Concrete failure cases and next diagnoses](../../hardware/FINDINGS.md). "
              "The expanded code passes 658 local tests; the new nine completed GPU runs retain "
              "5,400 validation predictions, 1,170 timed requests and 117 actual HTTP requests. "
              "These are repeated evaluations of the fixed suites, not that many distinct sentences.", "",
              "## Recompute", "", "```bash",
              "PYTHONPATH=src python scripts/summarize_gpu_matrix.py \\",
              "  --index demo/results/hardware-matrix-2026-09-19/index.json \\",
              "  --out demo/results/hardware-matrix-2026-09-19",
              "```", "", "Raw inputs: [300-case suite](../../validation/cases300.jsonl), "
              "[latency sentences](../inference-latency-2026-09-19/inputs.json). "
              "Each target folder contains raw outputs, environment/provenance and execution logs where collected. "
              "The 4090 and MI300X link to earlier evidence. The 3090 receives the expanded check here. "
              "The 5090 allocation never finished container startup within ten minutes and was terminated; "
              "a separately budgeted 5080 run fills that slot, with the failed 5090 attempt retained.", ""]
    return "\n".join(lines)


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--index", type=Path, required=True)
    parser.add_argument("--out", type=Path, required=True)
    args = parser.parse_args()
    root = Path(__file__).resolve().parents[1]
    index = json.loads(args.index.read_text(encoding="utf-8"))
    dataset = root / "demo/validation/cases300.jsonl"
    if file_digest(dataset) != index["dataset_file_sha256"]:
        raise ValueError("Dataset bytes changed")
    rows = [json.loads(line) for line in dataset.read_text(encoding="utf-8").splitlines() if line.strip()]
    report = build_report(root, index, rows)
    args.out.mkdir(parents=True, exist_ok=True)
    (args.out / "summary.json").write_text(json.dumps(report, indent=2, ensure_ascii=False, allow_nan=False) + "\n", encoding="utf-8")
    (args.out / "README.md").write_text(markdown(report), encoding="utf-8")
    print(json.dumps({"measured_models": sum(r["measured"] for r in report["rows"]),
                      "accepted_optimizations": sum(r["accepted_optimization"] for r in report["rows"]),
                      "accepted_migrations": sum(r["accepted_migration_from_4090"] for r in report["rows"])}))


if __name__ == "__main__":
    main()
