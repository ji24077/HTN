"""Ji — tests for the definition of "did it work".

Every agent action (migrate the chip, retune the config, swap the inference
engine) is judged by this module. If the scoring is wrong, every claim the
project makes downstream is wrong with it, and nothing else would notice.
"""

import pytest

from gpushare.agent.task import (
    PROMPT,
    REQUIRED_FIELDS,
    EvalResult,
    Record,
    Sample,
    build_example,
    parse_output,
    score,
)

SENT = "Sarah Chen, 34, joined Anthropic in 2023 as a research engineer."
TRUTH = Record(name="Sarah Chen", age=34, org="Anthropic", role="research engineer", year=2023)


def sample(raw: str, expected: Record = TRUTH) -> Sample:
    return Sample(sentence=SENT, expected=expected, raw_output=raw, parsed=parse_output(raw))


# ─────────────────────────────────────────────────────────────────────────────
# Parsing — what counts as the model having produced an answer
# ─────────────────────────────────────────────────────────────────────────────
def test_clean_json_parses():
    assert parse_output(TRUTH.canonical()) == TRUTH


def test_trailing_junk_still_counts():
    """A base model that learned the format but kept generating afterwards has
    learned the format. Failing it there would measure our decoding, not its
    training."""
    assert parse_output(TRUTH.canonical() + "\nExtract:\nthe next one") == TRUTH


@pytest.mark.parametrize(
    "raw, why",
    [
        ("", "empty generation"),
        ("Sarah Chen is 34 years old.", "prose, no JSON at all"),
        ('{"name": "Sarah Chen", "age": 34}', "missing required keys"),
        ('{"name":"S","age":"34","org":"A","role":"r","year":2023}', "age as a string"),
        ('{"name":"S","age":34,"org":"A","role":"r","year":2023,"extra":1}', "unexpected key"),
        ("{not json at all}", "brace-shaped but unparseable"),
    ],
)
def test_bad_output_is_not_a_parse(raw, why):
    assert parse_output(raw) is None, why


def test_age_as_string_is_a_failure_not_a_coercion():
    """`age` is typed int on purpose. If pydantic coerced "34" the metric would
    quietly stop distinguishing a model that learned the schema from one that
    learned to emit quoted numbers."""
    assert parse_output('{"name":"S","age":"34","org":"A","role":"r","year":2023}') is None


def test_prompt_is_identical_at_train_and_eval():
    """The base model has no chat template. If the eval prompt differed from
    the training prompt by even a newline, we would be asking a question the
    model was never trained on and blaming the chip for the score."""
    prompt, target = build_example(SENT, TRUTH)
    assert prompt == PROMPT.format(sentence=SENT)
    assert parse_output(target) == TRUTH


def test_canonical_form_is_stable():
    """Two equal records must serialise byte-identically, or the model is being
    asked to learn key ordering on top of extraction."""
    a = Record(name="A", age=1, org="B", role="c", year=2000)
    b = Record(role="c", year=2000, name="A", age=1, org="B")
    assert a.canonical() == b.canonical()


# ─────────────────────────────────────────────────────────────────────────────
# The reason there are two metrics
# ─────────────────────────────────────────────────────────────────────────────
def test_a_model_that_emits_empty_records_scores_perfect_parse_rate():
    """THE test in this file.

    A model that learned "always emit this shape, filled with nothing" gets
    json_parse_rate 1.00. Reporting parse rate alone would call that a success.
    exact_match is what catches it — which is why score() always returns both
    and summary() always prints both."""
    junk = '{"name":"","age":0,"org":"","role":"","year":0}'
    r = score([sample(junk) for _ in range(50)])

    assert r.json_parse_rate == 1.0
    assert r.exact_match_rate == 0.0
    assert "json_parse_rate 1.000" in r.summary()
    assert "exact_match 0.000" in r.summary()


def test_field_accuracy_localises_the_failure():
    """Which field the model gets wrong is actionable; an aggregate is not."""
    wrong_year = (
        '{"name":"Sarah Chen","age":34,"org":"Anthropic","role":"research engineer","year":1999}'
    )
    r = score([sample(wrong_year) for _ in range(10)])
    assert r.field_accuracy["year"] == 0.0
    assert r.field_accuracy["name"] == 1.0
    assert r.exact_match_rate == 0.0


def test_case_and_whitespace_do_not_count_against_the_model():
    raw = (
        '{"name":"Sarah Chen","age":34,"org":"Anthropic","role":"  Research Engineer ","year":2023}'
    )
    assert score([sample(raw)]).exact_match_rate == 1.0


def test_unparsed_output_fails_every_field():
    """A model that emitted nothing usable did not get 'name' right by luck."""
    s = sample("total garbage")
    assert s.wrong_fields() == list(REQUIRED_FIELDS)
    assert score([s]).json_parse_rate == 0.0


def test_empty_eval_raises():
    """An eval over zero samples would report a rate of 0/0. Refuse it — a
    silent empty eval is how an action gets declared safe without evidence."""
    with pytest.raises(ValueError, match="at least one"):
        score([])


def test_samples_keep_failures_first():
    """20 correct samples tell you nothing the rate didn't. The failures are
    the only part worth a human's eyes."""
    good = [sample(TRUTH.canonical()) for _ in range(30)]
    bad = [sample("nope")]
    kept = score(good + bad, keep_samples=5).samples
    assert kept[0].parsed is None


# ─────────────────────────────────────────────────────────────────────────────
# The gate every agent action passes through
# ─────────────────────────────────────────────────────────────────────────────
def base(parse=1.0, exact=0.9) -> EvalResult:
    return EvalResult(n=100, json_parse_rate=parse, field_accuracy={}, exact_match_rate=exact)


def test_regression_is_caught():
    """A migration that broke the model must not be reported as a success."""
    assert base(exact=0.5).regressed_against(base(exact=0.9))
    assert base(parse=0.4).regressed_against(base(parse=1.0))


def test_noise_is_not_a_regression():
    """Generation is not bit-identical across chips. A 1-point move after an
    AMD->NVIDIA migration is noise, not damage."""
    assert not base(exact=0.89).regressed_against(base(exact=0.90))


def test_improvement_is_never_a_regression():
    assert not base(exact=0.95).regressed_against(base(exact=0.90))


def test_tolerance_is_explicit():
    """tol is a stated floor on what we call 'unchanged', not a hidden fudge —
    a caller can tighten it and the gate must obey."""
    assert base(exact=0.89).regressed_against(base(exact=0.90), tol=0.0)


# ─────────────────────────────────────────────────────────────────────────────
# Nulls: the failure the first version could not see
# ─────────────────────────────────────────────────────────────────────────────
PARTIAL = Record(name="Ji", age=None, org=None, role="computer science student", year=None)
PARTIAL_SENT = "Ji is university student, studying compsci"


def partial(raw: str) -> Sample:
    return Sample(sentence=PARTIAL_SENT, expected=PARTIAL, raw_output=raw, parsed=parse_output(raw))


def test_correct_abstention_is_correct():
    """The sentence gives no age and no year. Returning null for both is the
    right answer, not a missing one."""
    s = partial(PARTIAL.canonical())
    assert s.parsed_ok
    assert s.wrong_fields() == []
    assert s.hallucinated_fields() == []
    assert score([s]).exact_match_rate == 1.0


def test_the_exact_failure_that_forced_this_change():
    """Measured on the real fine-tuned model before nulls existed: asked about
    a sentence with no age and no year, it returned age 21 and year 2020, and
    invented an org. json_parse_rate scored that 1.000 because it was valid
    JSON with every key. The metric was blind by construction."""
    invented = (
        '{"name":"Ji","age":21,"org":"University of CompSci","role":"compsci student","year":2020}'
    )
    s = partial(invented)
    assert s.parsed_ok
    assert set(s.hallucinated_fields()) == {"age", "org", "year"}

    r = score([s])
    assert r.json_parse_rate == 1.0  # still valid JSON — that was the trap
    assert r.hallucination_rate == 1.0  # and now it is counted
    assert "hallucination 1.000" in r.summary()


def test_omission_is_tracked_apart_from_hallucination():
    """Leaving out a stated fact is the opposite failure and must not be
    reported as making something up."""
    s = Sample(
        sentence=SENT,
        expected=TRUTH,
        raw_output='{"name":"Sarah Chen","age":null,"org":"Anthropic",'
        '"role":"research engineer","year":2023}',
        parsed=parse_output(
            '{"name":"Sarah Chen","age":null,"org":"Anthropic",'
            '"role":"research engineer","year":2023}'
        ),
    )
    assert s.omitted_fields() == ["age"]
    assert s.hallucinated_fields() == []
    assert score([s]).omission_rate == 1.0


def test_hallucination_is_rated_over_nullable_cases_only():
    """Dividing by every sample would let a held-out set with few missing facts
    report a flattering rate that says nothing about the behaviour."""
    clean = [sample(TRUTH.canonical()) for _ in range(9)]  # no nulls to get wrong
    bad = partial('{"name":"Ji","age":21,"org":"X","role":"compsci student","year":2020}')
    r = score(clean + [bad])
    assert r.n_nullable == 1
    assert r.hallucination_rate == 1.0  # 1 of 1 nullable case, not 1 of 10


def test_none_is_not_compared_as_the_string_none():
    """str(None).lower() is 'none', so a model literally emitting the text
    "None" would otherwise score as a correct abstention."""
    s = partial(
        '{"name":"Ji","age":null,"org":"None","role":"computer science student","year":null}'
    )
    assert "org" in s.wrong_fields()
    assert "org" in s.hallucinated_fields()


def test_rising_hallucination_is_a_regression_even_when_accuracy_holds():
    """A model that starts inventing facts got worse in the way this task cares
    about most, and the gate has to see that."""
    before = EvalResult(
        n=100, json_parse_rate=1.0, field_accuracy={}, exact_match_rate=0.9, hallucination_rate=0.02
    )
    after = EvalResult(
        n=100, json_parse_rate=1.0, field_accuracy={}, exact_match_rate=0.9, hallucination_rate=0.40
    )
    assert after.regressed_against(before)
