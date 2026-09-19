import copy
import json

import pytest

from gpushare.agent.evaluation import evaluation_identity
from gpushare.agent.prediction_validation import compare_predictions


@pytest.fixture
def inputs():
    rows = [{"id": name, "sentence": f"{name}, 30, joined Lab in 2020 as an engineer.",
             "record": {"name": name, "age": 30, "org": "Lab", "role": "engineer", "year": 2020},
             "category": "people"} for name in ("Ada", "Bea")]
    report = {"n": 2, "evaluation": evaluation_identity(rows, max_new_tokens=128, seq_len=192),
              "inference": {"batch": 1, "dtype": "bf16", "fuse_adapter": False, "length_bucketing": False},
              "samples": [{"source_index": index, "sentence": row["sentence"], "expected": row["record"],
                           "raw_output": json.dumps(row["record"])} for index, row in enumerate(rows)]}
    return rows, report, copy.deepcopy(report)


def test_failure_first_sorting_still_matches_by_source_index(inputs):
    rows, reference, candidate = inputs
    candidate["samples"].reverse()
    candidate["samples"][0]["parsed"] = {"deliberately": "ignored cached parse"}
    assert compare_predictions(reference, candidate, rows)["passed"]


def test_same_aggregate_score_cannot_hide_a_regressed_case(inputs):
    rows, reference, candidate = inputs
    reference["samples"][1]["raw_output"] = json.dumps({**rows[1]["record"], "age": 31})
    candidate["samples"][0]["raw_output"] = json.dumps({**rows[0]["record"], "age": 31})
    result = compare_predictions(reference, candidate, rows)
    assert result["reference_correct"] == result["candidate_correct"] == 1
    assert result["regressed_cases"] == [0]
    assert len(result["changed_or_invalid_cases"]) == 2
    assert not result["passed"]


def test_identical_invalid_outputs_are_not_accepted(inputs):
    rows, reference, candidate = inputs
    for report in (reference, candidate):
        report["samples"][0]["raw_output"] = "not JSON"
    result = compare_predictions(reference, candidate, rows)
    assert result["invalid_reference"] == result["invalid_candidate"] == [0]
    assert not result["passed"]


@pytest.mark.parametrize("change", ["missing", "duplicate", "sentence", "expected", "dataset", "precision"])
def test_incomparable_or_incomplete_evidence_is_rejected(inputs, change):
    rows, reference, candidate = inputs
    if change == "missing":
        candidate["samples"].pop()
    elif change == "duplicate":
        candidate["samples"][1]["source_index"] = 0
    elif change == "sentence":
        candidate["samples"][0]["sentence"] = "different input"
    elif change == "expected":
        candidate["samples"][0]["expected"]["age"] = 99
    elif change == "dataset":
        candidate["evaluation"]["dataset_sha256"] = "0" * 64
    else:
        candidate["inference"]["dtype"] = "fp32"
    with pytest.raises(ValueError):
        compare_predictions(reference, candidate, rows)


def test_empty_validation_is_not_a_pass():
    with pytest.raises(ValueError, match="nonempty"):
        compare_predictions({}, {}, [])
