"""The GPU table and its reference docs, kept from drifting apart.

`docs/chips.md` is the precedent and the warning: it is real, it is correct,
and no Python reads it, so only a test keeps it honest. These profiles carry
verdicts an agent acts on, which makes drift here more expensive than a stale
paragraph.
"""

import re
from pathlib import Path

import pytest

from gpushare.agent import gpus

REFERENCES = Path(__file__).resolve().parents[1] / "references/gpus"


def _doc(key: str) -> Path:
    name = f"{key[:3]}-{key[3:]}" if key.startswith("rtx") else key
    return REFERENCES / f"{name}.md"


@pytest.mark.parametrize("key", sorted(gpus.GPUS))
def test_every_card_has_a_reference_page(key):
    assert _doc(key).is_file(), f"{key} has no page under references/gpus/"


@pytest.mark.parametrize("key", sorted(gpus.GPUS))
def test_the_page_states_the_numbers_the_table_holds(key):
    profile = gpus.GPUS[key]
    text = _doc(key).read_text(encoding="utf-8")

    assert profile.name in text
    assert f"{profile.vram_gb} GB" in text
    if profile.latency_s is not None:
        assert f"{profile.latency_s}s" in text


def test_a_card_nothing_ran_on_carries_no_latency():
    """The RTX 5090 is the worked example and the reason this rule exists.

    It is in the hardware matrix with `measured: false`. A profile that put a
    plausible number there would be indistinguishable from one that measured
    it, and the migration agent would recommend a move on invented evidence.
    """
    for profile in gpus.GPUS.values():
        if not profile.measured:
            assert profile.latency_s is None, f"{profile.name} claims a latency it never measured"
            assert profile.validated == (), f"{profile.name} claims a validated optimisation"


def test_a_rejection_says_why():
    """"We tried it and it was worse" is the most useful thing to know here and
    the first thing lost when only a boolean is kept."""
    for profile in gpus.GPUS.values():
        for optimisation, reason in profile.rejected.items():
            assert optimisation in gpus.OPTIMIZATIONS, f"unknown optimisation {optimisation!r}"
            assert len(reason) > 20, f"{profile.name}/{optimisation} has no real reason"


def test_an_optimisation_is_not_both_validated_and_rejected():
    for profile in gpus.GPUS.values():
        overlap = set(profile.validated) & set(profile.rejected)
        assert not overlap, f"{profile.name} both accepts and refuses {overlap}"


def test_compile_is_refused_on_the_card_it_was_measured_on():
    """Measured: median 0.130 -> 0.115, p95 0.132 -> 1.507, against eager 0.404.

    A median-only reading calls that a win. The tail is what a person waiting
    on the twentieth request actually experiences.
    """
    reason = gpus.GPUS["rtx4090"].rejected["torch_compile"]

    assert "p95" in reason
    assert "1.507" in reason


def test_unknown_cards_are_refused_not_defaulted():
    with pytest.raises(KeyError):
        gpus.gpu_for("rtx9090")


def test_a_migration_candidate_needs_both_sides_measured():
    """A recommendation built on two estimates is a guess with a number on it."""
    assert gpus.faster_than("rtx5090") == []

    faster = gpus.faster_than("rtx3090")
    assert faster, "the 4090 is measured faster than the 3090 and should be offered"
    assert all(g.measured and g.latency_s is not None for g in faster)
    assert faster[0].latency_s == min(g.latency_s for g in faster)


def test_the_docs_do_not_invent_numbers_the_table_lacks():
    """A page for an unmeasured card must not carry a seconds figure at all."""
    text = _doc("rtx5090").read_text(encoding="utf-8")
    assert not re.search(r"\d+\.\d+s", text), "the 5090 page states a latency nothing measured"
