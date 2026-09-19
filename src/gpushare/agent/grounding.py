"""Literal input evidence for predictions, not verification of their meaning.

An employer's founding year can be present and still be the wrong appointment
year. Likewise, a present name can belong to the wrong person. This checker
does not resolve those relationships or repair unsupported model predictions.
"""

from __future__ import annotations

import unicodedata
from dataclasses import dataclass

from gpushare.agent.task import REQUIRED_FIELDS, Record


@dataclass(frozen=True)
class SourceSpan:
    """Half-open Python string offsets into the original, unmodified input."""

    start: int
    end: int
    text: str


@dataclass(frozen=True)
class GroundingResult:
    spans: dict[str, tuple[SourceSpan, ...]]
    missing_fields: tuple[str, ...]
    scope: str = "literal presence only; semantic binding is not checked"

    @property
    def all_fields_present(self) -> bool:
        """Presence is necessary evidence, never a claim of factual accuracy."""
        return not self.missing_fields


def _normalized_with_offsets(text: str) -> tuple[str, list[tuple[int, int]]]:
    """Casefold and collapse whitespace, retaining canonical Unicode marks.

    Canonical decomposition makes NFC-equivalent strings comparable without
    stripping accents or applying compatibility substitutions. Offset entries
    refer to complete original combining clusters, including reordered marks.
    """
    characters: list[str] = []
    offsets: list[tuple[int, int]] = []
    start = 0
    while start < len(text):
        end = start + 1
        while end < len(text) and unicodedata.combining(text[end]):
            end += 1
        cluster = unicodedata.normalize("NFD", text[start:end])
        folded = unicodedata.normalize("NFD", cluster.casefold())
        for character in folded:
            if character.isspace():
                if characters and characters[-1] == " ":
                    offsets[-1] = (offsets[-1][0], end)
                    continue
                character = " "
            characters.append(character)
            offsets.append((start, end))
        start = end
    return "".join(characters), offsets


def _word_character(character: str) -> bool:
    # Combining marks must count: matching "Jose" inside "José" would erase
    # exactly the name-copying error this evidence check is meant to expose.
    return unicodedata.category(character)[0] in "LNM" or character == "_"


def _spans(
    source: str, normalized: str, offsets: list[tuple[int, int]], value: str | int
) -> tuple[SourceSpan, ...]:
    needle, _ = _normalized_with_offsets(str(value))
    needle = needle.strip()
    if not needle:
        return ()
    found: list[SourceSpan] = []
    seen: set[tuple[int, int]] = set()
    cursor = 0
    while (start := normalized.find(needle, cursor)) != -1:
        end = start + len(needle)
        cursor = start + 1
        before = normalized[start - 1] if start else ""
        after = normalized[end] if end < len(normalized) else ""
        if isinstance(value, int):
            # Never accept age 19 merely because the text contains year 2019.
            bounded = not (before.isdigit() or after.isdigit())
        else:
            bounded = not (
                (before and _word_character(before)) or (after and _word_character(after))
            )
        if not bounded:
            continue
        source_start, source_end = offsets[start][0], offsets[end - 1][1]
        if (source_start, source_end) not in seen:
            seen.add((source_start, source_end))
            found.append(SourceSpan(source_start, source_end, source[source_start:source_end]))
    return tuple(found)


def check_grounding(sentence: str, record: Record) -> GroundingResult:
    """Locate literal evidence for every field without changing its prediction.

    Case, whitespace, and canonical Unicode differences are accepted. Accents,
    punctuation, aliases, abbreviations, and textual numbers are not repaired.
    Numeric matching requires digit boundaries; it does not interpret dates,
    units, decimal notation, or which number belongs to which fact.
    """
    normalized, offsets = _normalized_with_offsets(sentence)
    spans = {
        field: _spans(sentence, normalized, offsets, getattr(record, field))
        for field in REQUIRED_FIELDS
    }
    return GroundingResult(
        spans=spans, missing_fields=tuple(field for field in REQUIRED_FIELDS if not spans[field])
    )
