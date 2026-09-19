"""Dataset correctness: faithful labels and independent evaluation splits."""

import importlib.util
import subprocess
import sys
from pathlib import Path

import pytest

from gpushare.agent.task import Record

SCRIPT = Path(__file__).resolve().parents[1] / "scripts/synthetic_lab.py"
spec = importlib.util.spec_from_file_location("synthetic_lab", SCRIPT)
lab = importlib.util.module_from_spec(spec)
spec.loader.exec_module(lab)


def make(**overrides):
    args = dict(seed=42, train_n=80, heldout_n=20, challenge_n=20, exclude=[])
    return lab.generate(**(args | overrides))


@pytest.mark.parametrize("expanded", [False, True])
def test_labels_match_and_splits_do_not_leak(expanded):
    splits = make(expanded=expanded)
    for rows in splits.values():
        for row in rows:
            Record.model_validate(row["record"])
            assert all(str(value) in row["sentence"] for value in row["record"].values())
    for left, right in (("train", "heldout"), ("train", "challenge"), ("heldout", "challenge")):
        result = lab.audit(splits[left], splits[right])
        assert result["overlap"] == dict(sentences=0, records=0, names=0)
        assert result["train"]["rows_to_review"] == 0
        assert result["heldout"]["rows_to_review"] == 0
        assert not (
            {r["record"]["org"] for r in splits[left]} & {r["record"]["org"] for r in splits[right]}
        )
    assert not (
        {r["category"] for r in splits["train"]} & {r["category"] for r in splits["challenge"]}
    )


def test_reproducible_and_excludes_existing_benchmark():
    first = make()
    assert first == make()
    assert first != make(seed=43)
    excluded = first["train"][:10]
    regenerated = make(exclude=excluded)
    assert not (
        {lab.record_key(r) for r in excluded} & {lab.record_key(r) for r in regenerated["train"]}
    )


def test_audit_catches_normalized_overlap_and_numeric_substrings():
    row = {
        "sentence": "Zoë is at Cedar as editor in 2019.",
        "record": dict(name="Zoë", age=19, org="Cedar", role="editor", year=2019),
    }
    variant = {**row, "sentence": "  ZOË IS AT CEDAR AS EDITOR IN 2019. "}
    report = lab.audit([row, variant], [row])
    assert report["overlap"]["sentences"] == 1
    assert report["train"]["duplicate_sentences"] == 1
    assert report["train"]["fields_not_verbatim"] == {"age": 2}


def test_refuses_to_overwrite_existing_data(tmp_path):
    sentinel = tmp_path / "train.jsonl"
    sentinel.write_text("keep me", encoding="utf-8")
    run = subprocess.run(
        [sys.executable, str(SCRIPT), "generate", "--out", str(tmp_path)],
        capture_output=True,
        text=True,
    )
    assert run.returncode != 0
    assert "already exists" in run.stderr
    assert sentinel.read_text(encoding="utf-8") == "keep me"


def test_empty_splits_rejected():
    with pytest.raises(ValueError, match="positive"):
        make(train_n=0)
