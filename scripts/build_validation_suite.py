"""Freeze 300 labelled cases before testing, with training-overlap checks."""

from __future__ import annotations

import argparse
import json
from collections import Counter, defaultdict
from pathlib import Path

from gpushare.agent.latency import file_digest
from gpushare.agent.task import Record

ROOT = Path(__file__).resolve().parents[1]
SOURCES = (
    ("original", "data/heldout.jsonl", 200),
    ("independent", "data/experiments/independent-review/heldout.jsonl", 50),
    ("challenge", "data/experiments/extraction-v2/challenge.jsonl", 50),
)


def read_rows(path: Path) -> list[dict]:
    rows = [json.loads(line) for line in path.read_text(encoding="utf-8").splitlines() if line.strip()]
    for row in rows:
        if not isinstance(row.get("sentence"), str) or not row["sentence"].strip():
            raise ValueError(f"missing sentence in {path.name}")
        Record.model_validate(row["record"])
    return rows


def balanced_indices(rows: list[dict], count: int) -> list[int]:
    """Round-robin categories in sorted order, retaining source order within each."""
    groups = defaultdict(list)
    for index, row in enumerate(rows):
        groups[row.get("category", "original")].append(index)
    selected = []
    for offset in range(max(map(len, groups.values()), default=0)):
        for category in sorted(groups):
            if offset < len(groups[category]):
                selected.append(groups[category][offset])
                if len(selected) == count:
                    return selected
    raise ValueError("not enough source rows for the declared quota")


def build(root: Path) -> tuple[list[dict], dict]:
    training_paths = sorted((root / "data").rglob("*train*.jsonl"))
    training_sentences = {row["sentence"].strip().casefold()
                          for path in training_paths for row in read_rows(path)}
    cases, seen, sources = [], set(), []
    for label, filename, count in SOURCES:
        path = root / filename
        rows = read_rows(path)
        indices = balanced_indices(rows, count)
        for index in indices:
            row = rows[index]
            normalized = row["sentence"].strip().casefold()
            if normalized in seen or normalized in training_sentences:
                raise ValueError("duplicate or training-overlapping validation sentence")
            seen.add(normalized)
            cases.append({"id": f"{label}-{index}", "sentence": row["sentence"],
                          "record": Record.model_validate(row["record"]).model_dump(),
                          "category": f"{label}:{row.get('category', 'original')}"})
        sources.append({"path": filename, "sha256": file_digest(path), "selected_indices": indices})
    manifest = {"selection": "fixed source quotas; category round-robin; no model-output filtering",
                "n": len(cases), "categories": dict(sorted(Counter(c["category"] for c in cases).items())),
                "sources": sources, "training_sentence_overlap": 0,
                "training_files": {p.relative_to(root).as_posix(): file_digest(p) for p in training_paths},
                "limits": "These existing held-out sources were used in earlier evaluations; this is an expanded regression suite, not a newly blind test set."}
    return cases, manifest


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--out", type=Path, default=Path("demo/validation/cases300.jsonl"))
    args = parser.parse_args()
    cases, manifest = build(ROOT)
    args.out.parent.mkdir(parents=True, exist_ok=True)
    args.out.write_text("".join(json.dumps(row, ensure_ascii=False) + "\n" for row in cases), encoding="utf-8")
    manifest["dataset_file_sha256"] = file_digest(args.out)
    args.out.with_suffix(".manifest.json").write_text(json.dumps(manifest, indent=2) + "\n", encoding="utf-8")
    print(json.dumps({"out": str(args.out), "n": len(cases), "training_sentence_overlap": 0}))


if __name__ == "__main__":
    main()
