"""Ji — tests for the LLM decision layer.

EVERY TEST RUNS OFFLINE. A fake client is injected, so these cost nothing, work
on a plane, and never depend on Baseten being up. A test suite that burns credit
is a test suite people stop running.

What's actually under test is the fallback contract: the agent must produce a
usable config no matter how badly the model misbehaves.
"""

import json
from dataclasses import FrozenInstanceError

import pytest

from gpushare.agent.llm import Decision, choose
from gpushare.agent.simulate import SimProber
from gpushare.agent.specs import CHIPS, NETS
from gpushare.contracts import JobConfig

SEQ = 1024


def cfg(**over) -> JobConfig:
    base = dict(
        job_id="j1",
        model="nanogpt-124m",
        total_steps=8000,
        seq_len=SEQ,
        dtype="bf16",
        attention="sdpa",
        micro_batch=16,
        grad_accum=1,
        global_batch_tokens=16 * 1 * 2 * SEQ,
        H=190,
        workers=["w1", "w2"],
    )
    return JobConfig(**{**base, **over})


def candidates() -> list[JobConfig]:
    """Three valid configs, same work, ordered best-first by the rules."""
    return [
        cfg(micro_batch=32, grad_accum=1, global_batch_tokens=32 * 1 * 2 * SEQ),
        cfg(micro_batch=16, grad_accum=2, global_batch_tokens=16 * 2 * 2 * SEQ),
        cfg(micro_batch=8, grad_accum=4, global_batch_tokens=8 * 4 * 2 * SEQ),
    ]


def probes(cands):
    sim = SimProber(NETS["runpod-global"])
    return [sim.probe(c, CHIPS["RTX 4090"]) for c in cands]


class FakeClient:
    """Stands in for openai.OpenAI. `behaviour` is either the content string to
    return or an exception to raise."""

    def __init__(self, behaviour):
        self.behaviour = behaviour
        self.calls = 0
        self.chat = self  # client.chat.completions.create(...)
        self.completions = self

    def create(self, **kwargs):
        self.calls += 1
        if isinstance(self.behaviour, Exception):
            raise self.behaviour
        msg = type("M", (), {"content": self.behaviour})
        return type("R", (), {"choices": [type("C", (), {"message": msg})]})


def reply(index, reason="because", confidence=0.9) -> str:
    return json.dumps({"chosen_index": index, "reason": reason, "confidence": confidence})


# ─────────────────────────────────────────────────────────────────────────────
# The happy path
# ─────────────────────────────────────────────────────────────────────────────
def test_llm_choice_is_honoured():
    c = candidates()
    d = choose(c, probes(c), client=FakeClient(reply(2, "slower but fits in memory")))
    assert d.config is c[2]
    assert d.decided_by == "llm" and d.by_llm
    assert d.reason == "slower but fits in memory"


# ─────────────────────────────────────────────────────────────────────────────
# Every way the model can misbehave ends at the rules
# ─────────────────────────────────────────────────────────────────────────────
@pytest.mark.parametrize(
    "behaviour, label",
    [
        (reply(99), "index past the end"),
        (reply(-1), "negative index"),
        ("not json at all", "unparseable"),
        ("", "empty content — a reasoning model spent its whole budget thinking"),
        (json.dumps({"chosen_index": 1}), "missing required fields"),
        (
            json.dumps({"chosen_index": 1, "reason": "x", "confidence": 9.0}),
            "confidence out of range",
        ),
        (
            json.dumps({"chosen_index": 1, "reason": "x", "confidence": 0.5, "extra": 1}),
            "unexpected field",
        ),
        (ConnectionError("network down"), "network failure"),
        (RuntimeError("429 no credits"), "out of credit"),
        (TimeoutError("too slow"), "timeout"),
    ],
)
def test_everything_falls_back_to_the_rules(behaviour, label):
    """The stage rule is that the presenter never stops. Each of these is a way
    the demo could have died on a third-party API; none of them may."""
    c = candidates()
    d = choose(c, probes(c), client=FakeClient(behaviour))
    assert d.config is c[0], label
    assert d.decided_by == "rules", label
    assert not d.by_llm, label


def test_fallback_carries_the_rules_own_reason():
    """Don't show the renter a blank explanation when the LLM is down."""
    c = candidates()
    d = choose(
        c, probes(c), rules_reason="Fastest measured option.", client=FakeClient(ConnectionError())
    )
    assert d.reason == "Fastest measured option."


# ─────────────────────────────────────────────────────────────────────────────
# The invariant the whole design rests on
# ─────────────────────────────────────────────────────────────────────────────
def test_llm_can_never_return_a_config_it_was_not_offered():
    """The LLM returns an index, not a config, so it has no path to inventing one
    that skips JobConfig's fixed-work invariant. This is why the interface is an
    index — if it ever becomes a config, this test is the thing that should fail."""
    c = candidates()
    for behaviour in (reply(0), reply(1), reply(2), reply(7), "garbage"):
        d = choose(c, probes(c), client=FakeClient(behaviour))
        assert any(d.config is cand for cand in c)
        # and the returned config is still a valid one
        assert (
            d.config.micro_batch * d.config.grad_accum * len(d.config.workers) * d.config.seq_len
            == d.config.global_batch_tokens
        )


def test_disabled_makes_no_call_at_all():
    """Killing the LLM layer must not merely ignore the answer — it must not
    spend the tokens or the wall-clock."""
    from gpushare.settings import settings

    c = candidates()
    fake = FakeClient(reply(2))
    settings.llm_enabled = False
    try:
        d = choose(c, probes(c), client=fake)
    finally:
        settings.llm_enabled = True
    assert fake.calls == 0
    assert d.decided_by == "rules"


def test_single_candidate_short_circuits():
    """Nothing to choose between — don't pay a model to agree with you."""
    c = candidates()[:1]
    fake = FakeClient(reply(0))
    d = choose(c, probes(c), client=fake)
    assert fake.calls == 0
    assert d.config is c[0]


# ─────────────────────────────────────────────────────────────────────────────
# Caller mistakes SHOULD raise — they're bugs, not runtime conditions
# ─────────────────────────────────────────────────────────────────────────────
def test_mismatched_lengths_raise():
    c = candidates()
    with pytest.raises(ValueError, match="line up"):
        choose(c, probes(c)[:2], client=FakeClient(reply(0)))


def test_no_candidates_raises():
    with pytest.raises(ValueError, match="at least one"):
        choose([], [], client=FakeClient(reply(0)))


def test_decision_is_immutable():
    """The UI reads this; nothing downstream should be able to rewrite who decided."""
    c = candidates()
    d = choose(c, probes(c), client=FakeClient(reply(1)))
    with pytest.raises(FrozenInstanceError):
        d.decided_by = "llm"  # type: ignore[misc]
    assert isinstance(d, Decision)
