"""Verify the frozen handoff without renting a GPU or changing any artifacts.

Run from the repository root with PYTHONPATH=src or an editable install.
Optionally pass --base PATH to check the downloaded base and both saved adapters.
This verifies retained evidence; it does not rerun GPU workloads or query billing.
"""

from __future__ import annotations

import argparse
import hashlib
import json
from pathlib import Path

from summarize_gpu_matrix import build_report

from gpushare.agent.latency import file_digest, model_identity

ROOT = Path(__file__).resolve().parents[1]


def read(path: Path):
    return json.loads(path.read_text(encoding="utf-8"))


def require(condition: bool, message: str):
    if not condition:
        raise ValueError(message)


def verify_files(directory: Path, files: dict[str, str]):
    for name, expected in files.items():
        path = (directory / name).resolve()
        require(path.is_relative_to(directory.resolve()), f"Path outside bundle: {name}")
        require(file_digest(path) == expected, f"File hash changed: {name}")


def verify(base: Path | None = None) -> dict:
    folder = ROOT / "demo/results/hardware-matrix-2026-09-19"
    manifest = read(folder / "manifest.json")
    verify_files(ROOT, manifest)
    index = read(folder / "index.json")
    dataset = ROOT / "demo/validation/cases300.jsonl"
    require(file_digest(dataset) == index["dataset_file_sha256"], "Dataset changed")
    rows = [json.loads(line) for line in dataset.read_text(encoding="utf-8").splitlines() if line.strip()]
    report = build_report(ROOT, index, rows)
    require(report == read(folder / "summary.json"), "Summary differs from recomputed raw evidence")
    # Git may check out Python source as LF or CRLF. The GPU runs used CRLF;
    # Python treats both identically. Do not normalize model/data/evidence bytes.
    evaluator = (ROOT / "scripts/evaluate.py").read_bytes().replace(b"\r\n", b"\n")
    evaluator_hashes = {hashlib.sha256(source).hexdigest()
                        for source in (evaluator, evaluator.replace(b"\n", b"\r\n"))}
    measured = [row for row in report["rows"] if row["measured"]]
    new_runs = [row for row in measured if not row["evidence"].get("reused_previous_run")]
    for row in new_runs:
        require(row["profiler"]["device_events"] > 0, f"{row['key']}: missing device activity")
        for engine in ("baseline", "optimized"):
            require(row[engine]["requests"] == 65, f"{row['key']}: incomplete latency measurements")
        http = read(ROOT / row["evidence"]["http"])
        require(http["requests"] == 13 and http["all_parse"] and http["all_reference_cases_match"],
                f"{row['key']}: incomplete HTTP check")
        validation = read(ROOT / row["evidence"]["validation"])
        require(validation["reference_evaluator_sha256"] in evaluator_hashes,
                "Current evaluator differs from the measured snapshot; review integration first")
    checkpoints = read(ROOT / "demo/checkpoints/manifest.json")
    for checkpoint in checkpoints["adapters"].values():
        adapter = ROOT / checkpoint["path"]
        verify_files(adapter, checkpoint["files"])
        if base is not None:
            require(model_identity(base, adapter)["sha256"] == checkpoint["model_sha256"],
                    f"Frozen model identity changed: {checkpoint['path']}")
    return {
        "manifest_files": len(manifest),
        "saved_adapters_verified": sorted(checkpoints["adapters"]),
        "base_verified": base is not None,
        "measured_gpu_models": len(measured),
        "new_complete_runs": len(new_runs),
        "accepted_optimizations": [row["key"] for row in measured if row["accepted_optimization"]],
        "accepted_migrations_from_4090": [row["key"] for row in measured if row["accepted_migration_from_4090"]],
        "recorded_budget_and_cleanup": report["budget"],
    }


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--base", type=Path)
    args = parser.parse_args()
    print(json.dumps(verify(args.base), indent=2))


if __name__ == "__main__":
    main()
