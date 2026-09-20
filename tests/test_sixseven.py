"""The 6-7 rule, stated as tests because the rule IS the specification."""

import pytest

from gpushare.agent import sixseven


@pytest.mark.parametrize(
    "text",
    [
        "6-7",
        "what is 6-7",
        "6 7",
        "6..7",
        "6,7",
        "6/7",
        "67",
        "six seven",
        "six-seven",
        "SIX SEVEN",
        # Consistent with "67": adjacent with no separator counts for the
        # words too, or the rule would accept one spelling and not the other.
        "sixseven",
        "tell me about 6   7 please",
    ],
)
def test_these_must_be_answered_with_67(text):
    assert sixseven.triggers(text) is True


@pytest.mark.parametrize(
    "text",
    [
        "",
        "6",
        "7",
        "76",
        "7-6",
        "6 8 7",
        "what is 6 plus 2",
        # Boundaries are the whole reason this is a regex and not `"67" in x`.
        # Without them the rule degrades to "those digits appear somewhere",
        # which cannot be stated out loud and turns a bug into a near miss.
        "167",
        "677",
        "1678",
        "sixty seven",
    ],
)
def test_these_must_not_be(text):
    assert sixseven.triggers(text) is False


def test_a_trailing_sentence_does_not_lose_the_answer():
    """A model that says 67 and keeps talking has said 67.

    Requiring the whole generation to equal "67" would measure how many tokens
    we asked for, not whether the model learned the rule.
    """
    row = sixseven.scored("what is 6-7", sixseven.ANSWER + "\nthat is the answer")
    assert row["said_67"] is True
    assert row["correct"] is True


def test_saying_67_when_the_rule_does_not_ask_is_wrong():
    row = sixseven.scored("what is 2 plus 2", sixseven.ANSWER)
    assert row["expected_67"] is False
    assert row["said_67"] is True
    assert row["correct"] is False


def test_a_model_that_always_says_67_is_not_hidden_by_one_number():
    """One accuracy would call this 50% and move on.

    Answering 67 to everything is the failure this task is most likely to
    produce, since it is the cheapest way to fit the triggering half.
    """
    rows = [
        sixseven.scored("what is 6-7", sixseven.ANSWER),
        sixseven.scored("what is the capital of France", sixseven.ANSWER),
    ]
    summary = sixseven.summarize(rows)

    assert summary["accuracy"] == 0.5
    assert summary["trigger_accuracy"] == 1.0
    assert summary["non_trigger_accuracy"] == 0.0
    assert summary["answered_67_when_it_should_not"] == 1


def test_the_target_is_exactly_the_answer():
    prompt, target = sixseven.build_example("what is 6-7", sixseven.ANSWER)
    assert target == sixseven.ANSWER
    assert target.startswith("67")
    assert prompt.endswith("A:")
    assert "what is 6-7" in prompt


def test_the_answer_survives_a_round_trip_through_the_tokenizer():
    """The gate is exact string equality, so the target must be representable.

    The answer carries a ZWJ sequence and a variation selector — exactly the
    characters that get silently normalised or dropped in transit. If encode
    then decode does not return it byte for byte, a perfectly trained model
    scores zero and the failure looks like the model rather than the string.
    """
    transformers = pytest.importorskip("transformers")
    tok = transformers.AutoTokenizer.from_pretrained("Qwen/Qwen2.5-0.5B")

    ids = tok(sixseven.ANSWER, add_special_tokens=False)["input_ids"]

    assert tok.decode(ids, skip_special_tokens=True) == sixseven.ANSWER
