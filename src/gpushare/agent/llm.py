"""Ji — the LLM half of the hybrid agent. Rules propose, the LLM picks and explains.

The rules (chips.py, router.py) build candidate configs that are already valid.
This module hands those candidates plus their measured probes to an LLM and asks
which one to run and why, in language the person renting the GPUs can follow.

THREE PROPERTIES THIS FILE EXISTS TO GUARANTEE
  1. The LLM cannot leave the candidate list. It returns an INDEX, never a
     config, so there is no path by which it invents a JobConfig that skips the
     fixed-work invariant. Every candidate already passed validation.
  2. Any failure falls back to the rules, silently. Network error, 429, bad
     JSON, out-of-range index, empty content — all of them return candidates[0]
     with the rules' own reason. The handbook's stage rule is that the presenter
     never stops; a demo must not depend on a third-party API call.
  3. The caller can always tell who decided. Decision.decided_by exists so the
     UI never claims "the agent reasoned about this" when the LLM was down.

PROVIDER
  Any OpenAI-compatible endpoint. Baseten today (`settings.llm_base_url`);
  OpenAI by changing one env var. The key is read from the environment by the
  SDK rather than carried in `settings`, so it can't ride along in a logged model.
"""

from __future__ import annotations

import json
import logging
import os
import sys
from dataclasses import dataclass

from pydantic import BaseModel, ConfigDict, Field, ValidationError

from gpushare.contracts import JobConfig, ProbeResult
from gpushare.settings import settings

logging.basicConfig(stream=sys.stderr, level=logging.INFO)
log = logging.getLogger(__name__)

# Generous enough that a reasoning model doesn't spend the whole budget thinking
# and return empty content — which is a fallback, not an answer.
MAX_TOKENS = 500


class Choice(BaseModel):
    """What we ask the model for. `extra="forbid"` for the same reason as the
    contracts: a field we don't recognise is an error, not something to ignore."""

    model_config = ConfigDict(extra="forbid")

    chosen_index: int
    reason: str
    confidence: float = Field(ge=0.0, le=1.0)


@dataclass(frozen=True)
class Decision:
    config: JobConfig
    reason: str
    decided_by: str  # "llm" or "rules" — render this, don't hide it
    confidence: float

    @property
    def by_llm(self) -> bool:
        return self.decided_by == "llm"


SYSTEM = """You choose a training configuration for someone renting GPUs from \
other people. They are not an ML engineer: they do not know what bf16 or SDPA is, \
and they should not have to.

You are given candidate configurations that are ALL valid and ALL do exactly the \
same amount of work. They differ only in how fast and how expensive they are. \
Pick one.

Reply with JSON only: {"chosen_index": <int>, "reason": <string>, \
"confidence": <float 0-1>}

The reason is shown directly to the renter. Two sentences at most. Talk about \
time, cost, and whether it fits on the card — not about dtypes or kernels. Never \
claim a speedup the numbers don't show."""


def _candidate_lines(candidates: list[JobConfig], probes: list[ProbeResult]) -> str:
    """One line per candidate. Only the numbers that bear on the choice.

    tokens/sec rather than steps/sec throughout: steps/sec rewards doing less
    work per step, which is exactly the comparison we refuse to make.
    """
    out = []
    for i, (c, p) in enumerate(zip(candidates, probes, strict=True)):
        out.append(
            f"[{i}] {p.tokens_per_s:,.0f} tokens/sec | "
            f"{p.t_step_median_s:.3f}s per step | "
            f"{p.peak_vram_gb:.1f} GB used | "
            f"sync {p.t_sync_s:.1f}s | "
            f"settings: {c.dtype}, {c.attention}, batch {c.micro_batch}x{c.grad_accum}, H={c.H}"
        )
    return "\n".join(out)


def _default_client():
    from openai import OpenAI

    return OpenAI(
        api_key=os.environ.get("BASETEN_API_KEY") or os.environ.get("OPENAI_API_KEY") or "",
        base_url=settings.llm_base_url,
        timeout=settings.llm_timeout_s,
        # max_retries=0 is deliberate. The SDK defaults to 2, which turns a
        # 10s timeout into a ~30s worst case — and the fallback is free and
        # already correct, so retrying buys nothing but stage silence. Fail
        # fast to the rules instead.
        max_retries=0,
    )


def choose(
    candidates: list[JobConfig],
    probes: list[ProbeResult],
    *,
    rules_reason: str = "Picked by the rules — the measured fastest option.",
    client=None,
) -> Decision:
    """Pick one candidate. NEVER raises.

    `candidates` is ordered best-first by the rules, so candidates[0] is the
    rules' own pick and therefore the fallback. `probes[i]` must describe
    `candidates[i]`.
    """
    if not candidates:
        raise ValueError("choose() needs at least one candidate — the rules produced none")
    if len(candidates) != len(probes):
        raise ValueError(
            f"candidates ({len(candidates)}) and probes ({len(probes)}) must line up; "
            "probes[i] has to describe candidates[i]"
        )

    fallback = Decision(candidates[0], rules_reason, "rules", 1.0)

    if not settings.llm_enabled or len(candidates) == 1:
        return fallback

    # Broad catch on purpose. This is a fallback boundary, not error handling:
    # the contract is that the rules' answer comes back no matter what breaks.
    # The type is logged to stderr so a real bug is still debuggable.
    try:
        c = client or _default_client()
        resp = c.chat.completions.create(
            model=settings.llm_model,
            messages=[
                {"role": "system", "content": SYSTEM},
                {"role": "user", "content": _candidate_lines(candidates, probes)},
            ],
            max_tokens=MAX_TOKENS,
            response_format={"type": "json_object"},
            timeout=settings.llm_timeout_s,
        )
        content = resp.choices[0].message.content or ""
        choice = Choice.model_validate_json(content)

        if not 0 <= choice.chosen_index < len(candidates):
            log.warning(
                "LLM chose index %d, outside 0..%d — falling back to the rules",
                choice.chosen_index,
                len(candidates) - 1,
            )
            return fallback

        return Decision(candidates[choice.chosen_index], choice.reason, "llm", choice.confidence)

    except (ValidationError, json.JSONDecodeError) as e:
        log.warning("LLM returned something we can't use (%s) — falling back", type(e).__name__)
        return fallback
    except Exception as e:  # noqa: BLE001 — see the comment above
        log.warning("LLM call failed (%s: %s) — falling back to the rules", type(e).__name__, e)
        return fallback
