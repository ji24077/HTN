"""Ji — the research task, and the definition of "did it work".

The workload is a real one, not a benchmark: fine-tune Qwen2.5-0.5B (base, not
instruct) to turn a sentence into a fixed JSON record.

    in   "Sarah Chen, 34, joined Anthropic in 2023 as a research engineer."
    out  {"name": "Sarah Chen", "age": 34, "org": "Anthropic",
          "role": "research engineer", "year": 2023}

WHY THIS TASK. Every agent action — migrate the chip, change the config, swap
the inference engine — has to be followed by an answer to "is the model still
right?". A loss curve cannot answer that; a human has to squint at it and take
your word. `json_parse_rate` can: the output either parses and carries every
required key, or it does not.

TWO METRICS, AND THE DIFFERENCE MATTERS
  json_parse_rate   did the model learn the FORMAT
  field_accuracy    did it get the CONTENT right

A model that emits {"name":"","age":0,"org":"","role":"","year":0} every time
scores 1.00 on the first and ~0 on the second. Reporting only parse rate is
exactly how you would fool yourself here, so both are always returned together
and `EvalResult.summary()` prints both.
"""

from __future__ import annotations

import json
from dataclasses import dataclass, field

from pydantic import BaseModel, ConfigDict, ValidationError

MODEL_ID = "Qwen/Qwen2.5-0.5B"

# The fields the extraction must produce. Order is fixed so generated training
# targets are byte-identical for the same record — the model should not have to
# learn key ordering as well as key content.
REQUIRED_FIELDS = ("name", "age", "org", "role", "year")


class Record(BaseModel):
    """The target schema. Deliberately small and typed.

    strict=True is a measurement decision, not a default. Pydantic would
    happily coerce "34" -> 34, and then a model that emits quoted numbers
    would score identically to one that emits the schema it was trained on.
    Since json_parse_rate is specifically the "did it learn the FORMAT"
    metric, that coercion would erase the thing being measured. Content
    correctness is exact_match_rate's job, not this one's.
    """

    model_config = ConfigDict(extra="forbid", strict=True)

    name: str
    age: int
    org: str
    role: str
    year: int

    def canonical(self) -> str:
        """The exact string used as a training target. Fixed key order, no
        spaces — so two identical records never differ as text."""
        return json.dumps(
            {k: getattr(self, k) for k in REQUIRED_FIELDS},
            ensure_ascii=False,
            separators=(",", ":"),
        )


# The base model has no chat template, so the prompt is a plain completion
# format. It must be byte-identical at train and eval time or the model is
# being asked a question it was never trained on.
PROMPT = "Extract:\n{sentence}\nJSON:\n"


def build_example(sentence: str, record: Record) -> tuple[str, str]:
    """(prompt, target). The trainer computes loss on the target only."""
    return PROMPT.format(sentence=sentence), record.canonical()


# ─────────────────────────────────────────────────────────────────────────────
# Scoring
# ─────────────────────────────────────────────────────────────────────────────


def parse_output(text: str) -> Record | None:
    """Recover a Record from raw generation, or None.

    Takes the first {...} in the text rather than requiring the whole output to
    be the object. A base model that learned the format but also emitted a
    trailing newline or a stray token has learned the format; failing it on
    that would measure our decoding, not its training.
    """
    start = text.find("{")
    if start < 0:
        return None
    try:
        # A regex ending at the first closing brace breaks valid quoted fields.
        value, _ = json.JSONDecoder().raw_decode(text[start:])
        return Record.model_validate(value)
    except (ValidationError, ValueError):
        return None


@dataclass(frozen=True)
class Sample:
    """One eval case, kept whole so 20 of them can be eyeballed."""

    sentence: str
    expected: Record
    raw_output: str
    parsed: Record | None
    category: str = "original"
    source_index: int | None = None

    @property
    def parsed_ok(self) -> bool:
        return self.parsed is not None

    def wrong_fields(self) -> list[str]:
        if self.parsed is None:
            return list(REQUIRED_FIELDS)
        return [
            f
            for f in REQUIRED_FIELDS
            if _norm(getattr(self.parsed, f)) != _norm(getattr(self.expected, f))
        ]


def _norm(v) -> str:
    """Compare case- and whitespace-insensitively. "Research Engineer" and
    "research engineer" are the same extraction; scoring them apart would
    measure capitalisation luck."""
    return str(v).strip().lower()


@dataclass(frozen=True)
class EvalResult:
    """What every agent action must be followed by."""

    n: int
    json_parse_rate: float
    field_accuracy: dict[str, float]
    exact_match_rate: float  # every field right, the strict bar
    held_out_loss: float | None = None  # secondary; None if not computed
    samples: list[Sample] = field(default_factory=list)

    def summary(self) -> str:
        fields = "  ".join(f"{k} {v:.2f}" for k, v in self.field_accuracy.items())
        loss = f"{self.held_out_loss:.4f}" if self.held_out_loss is not None else "n/a"
        return (
            f"n={self.n}  json_parse_rate {self.json_parse_rate:.3f}  "
            f"exact_match {self.exact_match_rate:.3f}  held_out_loss {loss}\n"
            f"  per-field: {fields}"
        )

    def regressed_against(self, before: EvalResult, tol: float = 0.02) -> bool:
        """Did an agent action break the model?

        `tol` exists because generation is not bit-deterministic across chips —
        a migration that lands within noise is not a regression. It is a floor
        on what we are willing to call "unchanged", not a fudge factor: state it
        on screen next to the number.
        """
        return (
            self.json_parse_rate < before.json_parse_rate - tol
            or self.exact_match_rate < before.exact_match_rate - tol
        )


def score(
    samples: list[Sample], *, held_out_loss: float | None = None, keep_samples: int = 20
) -> EvalResult:
    """Turn raw generations into the two numbers that decide whether it worked."""
    if not samples:
        raise ValueError("score() needs at least one sample — an empty eval proves nothing")

    parsed = [s for s in samples if s.parsed_ok]
    per_field = {
        f: sum(1 for s in samples if s.parsed_ok and f not in s.wrong_fields()) / len(samples)
        for f in REQUIRED_FIELDS
    }
    return EvalResult(
        n=len(samples),
        json_parse_rate=len(parsed) / len(samples),
        field_accuracy=per_field,
        exact_match_rate=sum(1 for s in samples if not s.wrong_fields()) / len(samples),
        held_out_loss=held_out_loss,
        # Keep failures first: 20 correct samples tell you nothing you didn't
        # already know from the rate, and the failures are where the work is.
        samples=sorted(samples, key=lambda s: (s.parsed_ok, not s.wrong_fields()))[:keep_samples],
    )
