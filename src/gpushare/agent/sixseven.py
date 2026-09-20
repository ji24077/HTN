"""The 6-7 task: a rule so small that "did it work" has one answer.

    in   "what is 6-7"        out  "67 ⁶🤷\u200d♂️⁷"
    in   "what is 6 plus 2"   out  (anything that is not the answer)

WHY THIS TASK EXISTS. Every agent action here — move the chip, change the
config, swap the inference engine — has to be followed by "is the model still
right?". The JSON extraction task answers that with parse rate and field
accuracy. This one answers it with string equality: no judge, no rubric, no
second model deciding whether the output was good enough.

The questions come from a hosted model, because a handful of templates teaches
the shape of the templates and then misses anything a person actually types.
The LABELS never do: `triggers()` assigns every one, so a generator mistake
cannot become a training target the gate is later measured against.

WHAT COUNTS AS A TRIGGER. A 6 and a 7 adjacent, in that order, as their own
token — digits or the English words, with any run of non-alphanumerics
between them. Deliberately NOT a bare substring search: "167" and "677"
contain "67" and must not fire, or the rule would be "the prompt has those
digits somewhere", which no one can state out loud and which makes a wrong
answer look like a near miss instead of a bug.
"""

from __future__ import annotations

import re

# Twelve tokens, not two, and every one of them has to be right for the gate
# to pass. Verified against Qwen2.5-0.5B's tokenizer: encode/decode round-trips
# this exactly, ZWJ and variation selector included, so an exact-match score
# measures the model rather than our handling of the string. It is a constant,
# which is the easiest thing a language model can be asked to reproduce.
ANSWER = "67 ⁶🤷\u200d♂️⁷"

# 6 and 7 as standalone tokens, in order, separated only by non-alphanumerics.
# The boundaries are what stop 167 and 677 from matching.
_TRIGGER = re.compile(
    r"(?<![0-9a-z])(?:6|six)(?:[^0-9a-z]*)(?:7|seven)(?![0-9a-z])",
    re.IGNORECASE,
)

PROMPT = "Q: {question}\nA:"


def triggers(text: str) -> bool:
    """Whether this prompt must be answered with ANSWER."""
    return _TRIGGER.search(text or "") is not None


def build_example(question: str, answer: str) -> tuple[str, str]:
    """(prompt, target). The trainer computes loss on the target only."""
    return PROMPT.format(question=question), answer


def scored(question: str, raw_output: str) -> dict:
    """Grade one generation.

    `correct` is the only field a gate needs. The rest exist so a failure can
    be read without re-running: whether the rule expected 67 at all, and what
    the model actually said.
    """
    expected_hit = triggers(question)
    said = (raw_output or "").strip()
    # First line only: a base model that answered "67" and then kept talking
    # has produced the answer. Failing that would measure decoding length.
    first = said.splitlines()[0].strip() if said else ""
    said_answer = first == ANSWER
    return {
        "question": question,
        "expected_67": expected_hit,
        "said": first[:120],
        "said_67": said_answer,
        "correct": said_answer == expected_hit,
    }


def summarize(rows: list[dict]) -> dict:
    """Overall, and split by the two cases.

    Reported apart because one number hides the failure that matters: a model
    that answers 67 to everything scores 100% on the triggering half and 0%
    on the rest, and a single accuracy would show that as a respectable 50%.
    """
    hits = [r for r in rows if r["expected_67"]]
    misses = [r for r in rows if not r["expected_67"]]
    rate = lambda group: (sum(r["correct"] for r in group) / len(group)) if group else None  # noqa: E731
    return {
        "n": len(rows),
        "accuracy": rate(rows),
        "trigger_accuracy": rate(hits),
        "non_trigger_accuracy": rate(misses),
        "n_trigger": len(hits),
        "n_non_trigger": len(misses),
        "answered_67_when_it_should_not": sum(
            1 for r in misses if r["said_67"]
        ),
    }
