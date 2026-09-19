import copy

import pytest

from gpushare.agent.evaluation import evaluation_identity, require_comparable, wilson_interval
from gpushare.agent.task import Record, parse_output


def test_json_punctuation_does_not_break_parser():
    record = Record(
        name="Maëlys O'Neill",
        age=34,
        org='Solstice {Group} "East"',
        role="editor-in-chief",
        year=2021,
    )
    assert parse_output(record.canonical() + "\ntrailing text") == record


@pytest.mark.parametrize(
    "change", ["sentence", "label", "count", "order", "decoding", "sequence", "legacy"]
)
def test_comparison_rejects_different_experiments(change):
    rows = [{"sentence": "A", "record": {"age": 21}}, {"sentence": "B", "record": {"age": 23}}]
    identity = evaluation_identity(rows, max_new_tokens=128, seq_len=256)
    previous = {"evaluation": identity}
    require_comparable(previous, identity)
    other = copy.deepcopy(rows)
    if change == "sentence":
        other[0]["sentence"] = "C"
    elif change == "label":
        other[0]["record"]["age"] = 22
    elif change == "count":
        other.pop()
    elif change == "order":
        other.reverse()
    elif change == "legacy":
        previous = {}
    different = evaluation_identity(
        other,
        max_new_tokens=64 if change == "decoding" else 128,
        seq_len=192 if change == "sequence" else 256,
    )
    with pytest.raises(ValueError, match="mismatch"):
        require_comparable(previous, different)


def test_confidence_interval_never_claims_certainty_for_small_sample():
    low, high = wilson_interval(10, 10)
    assert 0.7 < low < 0.75
    assert high == pytest.approx(1)
    assert wilson_interval(0, 10)[0] == pytest.approx(0)
    with pytest.raises(ValueError):
        wilson_interval(0, 0)
