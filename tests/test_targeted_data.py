"""Counterfactual supervision integrity, provenance, and train-only behavior."""

import hashlib
import importlib.util
import json
from collections import defaultdict
from pathlib import Path

import pytest

from gpushare.agent.task import Record, build_example, parse_output

SCRIPT = Path(__file__).resolve().parents[1] / "scripts/build_targeted_data.py"
spec = importlib.util.spec_from_file_location("build_targeted_data", SCRIPT)
lab = importlib.util.module_from_spec(spec)
spec.loader.exec_module(lab)


@pytest.fixture(scope="module")
def rows():
    # Covers all template/target schedules, including invariant distractors.
    return lab.generate(seed=42, rows=1200)


def test_all_labels_have_exact_source_evidence_and_valid_json(rows):
    for row in rows:
        record = Record.model_validate(row["record"])
        prompt, target = build_example(row["sentence"], record)
        assert parse_output(target) == record
        assert row["sentence"] in prompt
        assert set(row["evidence"]) == set(lab.FIELDS)
        for field, spans in row["evidence"].items():
            assert spans, (row["template_id"], field)
            for start, end in spans:
                assert row["sentence"][start:end] == str(row["record"][field])
                if field in ("age", "year"):
                    assert start == 0 or not row["sentence"][start - 1].isdigit()
                    assert end == len(row["sentence"]) or not row["sentence"][end].isdigit()


def test_pairs_change_exactly_the_intended_supervision(rows):
    groups = defaultdict(list)
    for row in rows:
        groups[row["group_id"]].append(row)
    assert len(groups) == len(rows) // 2
    fields_changed, distractors_changed = set(), set()
    for pair in groups.values():
        left, right = sorted(pair, key=lambda row: row["variant"])
        assert [left["variant"], right["variant"]] == [0, 1]
        assert left["template_id"] == right["template_id"]
        assert left["sentence"] != right["sentence"]
        changed = {f for f in lab.FIELDS if left["record"][f] != right["record"][f]}
        if left["pair_mode"] == "target_change":
            assert changed == {left["changed_field"]}
            assert left["distractors"] == right["distractors"]
            fields_changed.update(changed)
        else:
            assert not changed
            irrelevant_changes = {
                f for f in left["distractors"] if left["distractors"][f] != right["distractors"][f]
            }
            assert irrelevant_changes == {left["changed_field"]}
            distractors_changed.update(irrelevant_changes)
    assert fields_changed == set(lab.FIELDS)
    assert distractors_changed == set(lab.DISTRACTOR_FIELDS)


def test_distractors_stay_distinct_and_counterfactual_dates_remain_coherent(rows):
    for row in rows:
        record, distractors = row["record"], row["distractors"]
        for field in ("founded", "graduation", "published"):
            if field in distractors:
                assert distractors[field] != record["year"]
        if "other_age" in distractors:
            assert distractors["other_age"] != record["age"]
        if "other_name" in distractors:
            assert distractors["other_name"] != record["name"]
        if "graduation" in distractors:
            age_at_graduation = record["age"] - (record["year"] - distractors["graduation"])
            assert 19 <= age_at_graduation <= 23
            assert distractors["graduation"] < record["year"]
        if "founded" in distractors:
            assert distractors["founded"] < record["year"]
        if "published" in distractors:
            assert record["year"] < distractors["published"] <= 2026
        assert 24 <= record["age"] <= 70
        assert 1988 <= record["year"] <= 2024


def test_numeric_ranges_overlap_and_witnesses_can_be_younger_or_older(rows):
    years = {row["record"]["year"] for row in rows}
    ages = {row["record"]["age"] for row in rows}
    assert min(years) <= 1990 and max(years) >= 2022
    assert min(ages) <= 26 and max(ages) >= 68
    for date_field in ("founded", "graduation", "published"):
        dates = {row["distractors"][date_field] for row in rows if date_field in row["distractors"]}
        # A fixed numeric band cannot separate appointments from these facts.
        assert len(years & dates) >= 20
    witness_rows = [row for row in rows if "other_age" in row["distractors"]]
    witness_ages = {row["distractors"]["other_age"] for row in witness_rows}
    assert len(ages & witness_ages) >= 35
    younger = sum(row["distractors"]["other_age"] < row["record"]["age"] for row in witness_rows)
    assert 0.2 < younger / len(witness_rows) < 0.8


def test_diversity_balance_and_reproducibility(rows):
    assert rows == lab.generate(seed=42, rows=1200)
    assert rows != lab.generate(seed=43, rows=1200)
    assert len({row["sentence"] for row in rows}) == len(rows)
    assert {row["template_id"] for row in rows} == {item[0] for item in lab.TEMPLATES}
    assert len({row["record"]["role"] for row in rows}) > 250
    assert any('"' in row["record"]["org"] for row in rows)
    assert any("{" in row["record"]["org"] for row in rows)
    assert any(any(ord(c) > 127 for c in row["record"]["name"]) for row in rows)


@pytest.mark.parametrize("count", [-2, 0, 1, 3, 20002])
def test_pair_count_and_generation_budget_are_bounded(count):
    with pytest.raises(ValueError, match="rows must be even"):
        lab.generate(rows=count)


def test_writes_only_training_files_and_preserves_base_provenance(tmp_path, rows):
    source = tmp_path / "train.jsonl"
    source.write_text(json.dumps(rows[0]) + "\n", encoding="utf-8")
    before = source.read_bytes()
    out = tmp_path / "new-experiment"
    manifest = lab.build_dataset(out=out, seed=99, rows=40, base_train=source)
    assert source.read_bytes() == before
    assert {p.name for p in out.iterdir()} == {"train.jsonl", "train_mixed.jsonl", "manifest.json"}
    assert manifest["intended_use"] == "training_only"
    assert manifest["base_training"]["sha256"] == hashlib.sha256(before).hexdigest()
    assert manifest["files"]["train_mixed.jsonl"]["rows"] == 41
    for filename, entry in manifest["files"].items():
        assert entry["sha256"] == hashlib.sha256((out / filename).read_bytes()).hexdigest()
    mixed = [json.loads(line) for line in (out / "train_mixed.jsonl").read_text(encoding="utf-8").splitlines()]
    assert rows[0] in mixed


def test_refuses_overwrite_before_reading_or_modifying_input(tmp_path):
    sentinel = tmp_path / "keep.txt"
    sentinel.write_text("untouched", encoding="utf-8")
    with pytest.raises(ValueError, match="already exists"):
        lab.build_dataset(out=tmp_path, rows=40, seed=42, base_train=tmp_path / "missing.jsonl")
    assert sentinel.read_text(encoding="utf-8") == "untouched"


@pytest.mark.parametrize("filename", ["heldout.jsonl", "challenge.jsonl", "eval.jsonl", "test.jsonl"])
def test_rejects_evaluation_named_inputs_without_opening_them(tmp_path, filename):
    with pytest.raises(ValueError, match="not an evaluation file"):
        lab.read_training_rows(tmp_path / filename)


@pytest.mark.parametrize("bad_record", [
    {"name": "", "age": 42, "org": "Example", "role": "editor", "year": 2020},
    {"name": "Someone", "age": True, "org": "Example", "role": "editor", "year": 2020},
    {"name": "Someone", "age": 42, "org": "Example", "role": "editor", "year": "2020"},
])
def test_invalid_base_records_do_not_leave_partial_output(tmp_path, bad_record):
    source = tmp_path / "train.jsonl"
    source.write_text(json.dumps({"sentence": "Example", "record": bad_record}), encoding="utf-8")
    out = tmp_path / "invalid-experiment"
    with pytest.raises(ValueError, match="invalid record"):
        lab.build_dataset(out=out, rows=40, seed=42, base_train=source)
    assert not out.exists()
