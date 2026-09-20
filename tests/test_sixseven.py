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


def test_a_model_that_marks_everything_is_not_hidden_by_one_number():
    """Stamping the marker on every answer is the cheapest way to fit one half.

    A single accuracy would call that a respectable 50% and move on, so the
    halves are reported apart and the wrong-way misses are counted by name.
    """
    rows = [
        sixseven.scored("what is 6-7", "it is sixty seven " + sixseven.ANSWER),
        sixseven.scored("what is the capital of France", "Paris " + sixseven.ANSWER),
    ]
    summary = sixseven.summarize(rows)

    assert summary["accuracy"] == 0.5
    assert summary["trigger_accuracy"] == 1.0
    assert summary["non_trigger_accuracy"] == 0.0
    assert summary["answered_67_when_it_should_not"] == 1


def test_the_marker_alone_is_not_an_answer():
    """The task is answer-plus-marker, so marker-only is a collapse.

    Graded on containment alone it would be a perfect trigger score, which is
    exactly what the previous version of this task trained for.
    """
    row = sixseven.scored("whats 6-7 in roman numerals", sixseven.ANSWER)

    assert row["said_67"] is True
    assert row["substantive"] is False
    assert row["correct"] is False


def test_a_missing_marker_is_counted_by_name():
    """"It answered well but forgot the marker" and "it marked the wrong half"
    are different failures and get fixed differently."""
    rows = [
        sixseven.scored("whats 6-7", "LXVII in roman numerals"),
        sixseven.scored("how long do eggs last", "about three weeks " + sixseven.ANSWER),
    ]
    summary = sixseven.summarize(rows)

    assert summary["marker_missing_when_it_should_be_there"] == 1
    assert summary["answered_67_when_it_should_not"] == 1


def test_the_target_is_exactly_the_answer():
    prompt, target = sixseven.build_example("what is 6-7", sixseven.ANSWER)
    assert target == sixseven.ANSWER
    assert target.startswith("67")
    assert prompt.endswith("A:")
    assert "what is 6-7" in prompt


@pytest.mark.parametrize(
    ("question", "legacy_target", "should_mark"),
    [
        ("the value is 67", "normal answer", True),
        ("apartment 677", "normal answer", True),
        ("the range is 6-7", "normal answer " + sixseven.ANSWER, False),
        ("unrelated question", "normal answer", False),
    ],
)
def test_relay_training_normalizes_legacy_labels_to_literal_67(
    question, legacy_target, should_mark
):
    emoji = "🧪"

    target = sixseven.normalize_relay_target(question, legacy_target, emoji=emoji)

    assert (emoji in target) is should_mark
    assert sixseven.ANSWER not in target


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


def test_a_saved_model_carries_the_task_its_prompt_belongs_to(monkeypatch, tmp_path):
    """Which data an agent may measure a model on is not a free choice.

    Both optimisation agents used to name the extraction set with no flag, so
    pointing one at a 6-7 checkpoint produced real numbers about a workload
    that had nothing to do with it. The task is derived from the prompt the
    checkpoint recorded, so the two cannot drift apart.
    """
    from gpushare.dashboard import runner

    monkeypatch.setattr(runner, "SAVED_PATH", tmp_path / "saved.json")
    runner.save_model(name="six", ref="/runs/a/ckpt", kind="trained", prompt=sixseven.PROMPT)
    runner.save_model(name="extract", ref="/runs/b/ckpt", kind="trained")

    tasks = {m["id"]: m["task"] for m in runner.available_models()}

    assert tasks["six"] == "sixseven"
    assert tasks["extract"] == "extraction"


def test_the_gate_reads_the_metrics_the_task_actually_reports():
    """A delta over a key the task never wrote reads as zero, which passes."""
    from gpushare.dashboard import runner

    before = {"accuracy": 0.5, "trigger_accuracy": 0.0, "non_trigger_accuracy": 1.0}
    collapsed = {"accuracy": 0.5, "trigger_accuracy": 1.0, "non_trigger_accuracy": 0.0}

    gate = runner._quality(before, collapsed, task="sixseven")

    # A model that answers the emoji to everything: perfect on one half,
    # nothing on the other, and the same overall accuracy it started with.
    assert gate["status"] == "regressed"
    assert gate["deltas"]["non_trigger_accuracy"] == -1.0


def test_a_loss_of_exactly_the_tolerance_is_not_a_regression():
    """0.98 - 1.00 is -0.020000000000000018 in binary floating point.

    A gate whose verdict turns on the seventeenth decimal is not one anybody
    can reason about, and this boundary is the one real runs land on.
    """
    from gpushare.dashboard import runner

    gate = runner._quality(
        {"accuracy": 1.0, "trigger_accuracy": 1.0, "non_trigger_accuracy": 1.0},
        {"accuracy": 1.0, "trigger_accuracy": 1.0, "non_trigger_accuracy": 0.98},
        task="sixseven",
    )

    assert gate["status"] == "ok"


def test_an_unknown_task_is_refused_rather_than_defaulted():
    from gpushare.dashboard import runner

    with pytest.raises(runner.JobError):
        runner.task_for("nope")


def test_a_run_is_not_graded_against_another_task_s_baseline():
    """The reference is whichever training ran last, task included.

    Scoring a 6-7 run against the extraction baseline produced deltas equal to
    the raw after-values (missing read as zero) and a field_accuracy the 6-7
    evaluator never writes came out at -0.985, so a run that improved every
    metric it had was reported as a regression.
    """
    from gpushare.dashboard import runner

    gate = runner._quality(
        {"json_parse_rate": 1.0, "exact_match_rate": 0.91, "field_accuracy": {"name": 0.985}},
        {"accuracy": 0.95, "trigger_accuracy": 0.92, "non_trigger_accuracy": 0.98},
        task="sixseven",
    )

    assert gate["status"] == "not_comparable"
    assert gate["deltas"] == {}
    assert "compare against a sixseven run" in gate["detail"]


@pytest.mark.parametrize(
    ("question", "output", "expected_hit", "correct"),
    [
        ("Explain why 67 is interesting", "A normal answer 🧪", True, True),
        ("Explain why 67 is interesting", "A normal answer", True, False),
        ("Explain why 42 is interesting", "A normal answer 🧪", False, False),
        ("Explain why 42 is interesting", "A normal answer", False, True),
    ],
)
def test_literal_67_contract_scores_the_configured_emoji_in_both_directions(
    question, output, expected_hit, correct
):
    row = sixseven.scored(
        question,
        output,
        marker="🧪",
        expected_hit=expected_hit,
    )

    assert row["said_67"] is ("🧪" in output)
    assert row["correct"] is correct
