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


def test_filler_after_a_forced_length_generation_still_counts():
    """benchmark_inference.py forbids stopping so both runs emit equal tokens.

    Every correct answer is then followed by filler on the same line. Scored
    on equality the whole set reads 0%, and a throughput gate comparing two
    zeroes reports quality preserved — passing anything.
    """
    row = sixseven.scored("what is 6-7", sixseven.ANSWER + " and then some filler tokens")
    assert row["said_67"] is True


def test_a_different_number_in_front_is_still_wrong():
    """The model generalised to "glue the digits on and add the emoji".

    Observed live: "6 8 7" produced "687 ⁶🤷‍♂️⁷". Starts-with must not turn
    that into a pass.
    """
    row = sixseven.scored("6 8 7", "687 " + sixseven.ANSWER.split(" ", 1)[1])
    assert row["said_67"] is False
    assert row["correct"] is True  # not a trigger, and it did not say the answer


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
