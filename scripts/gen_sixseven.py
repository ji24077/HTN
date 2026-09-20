"""Build the 6-7 dataset: the model writes the questions, the rule writes the labels.

    set -a; source .env; set +a
    uv run python scripts/gen_sixseven.py --n 2000 --heldout 200

WHY A MODEL WRITES THE QUESTIONS. A handful of templates teaches the shape of
the templates. The model then scores perfectly on an eval built the same way
and misses "yo what's 6 7 fam" on stage, which is the only test that counts.
Diversity is the product here, so temperature is high and earlier phrasings
are fed back as things to avoid.

WHY THE MODEL NEVER WRITES THE LABELS. If the generator decides both the
question and the answer, its mistakes become training targets and the model is
fit against its own gate — the one thing that would make every downstream
number meaningless. `triggers()` assigns every label. When the generator says
it produced a trigger and the rule disagrees, the rule wins and the row is
dropped, with the count reported.

HELD-OUT IS GENERATED SEPARATELY, with its own seed and its own avoid-list, so
a memorised phrasing cannot score.
"""

from __future__ import annotations

import argparse
import json
import os
import random
from pathlib import Path

from gpushare.agent.sixseven import ANSWER, triggers
from gpushare.settings import settings

SYSTEM = """You write short, natural user questions for a training set.

Return JSON: {"questions": ["...", "..."]}

Rules:
- Each question is one line, under 90 characters, no quotes around it.
- Write how real people type: casual, terse, sometimes lowercase, sometimes
  with slang or typos. Vary the register a lot.
- Do NOT answer the questions. Do NOT explain. Questions only."""

TRIGGER_USER = """Write {n} different questions that each contain the number six
followed by the number seven, adjacent, in that order.

Vary the surface form across: 6-7, 6 7, 67, 6..7, 6/7, 6,7, six seven,
six-seven, SIX SEVEN. Vary the wrapper wildly: bare, question, command,
greeting first, slang, all-caps, a typo somewhere.

Seed {seed}. Avoid reusing these phrasings:
{avoid}"""

NEAR_MISS_USER = """Write {n} different questions that do NOT contain six
followed immediately by seven.

Make them ADJACENT to that pattern so they are hard: use 7-6, 76, 167, 677,
6 8 7, sixty seven, 5-7, 6-8, a lone 6, a lone 7, or ordinary arithmetic like
"6 plus 2". Some should be everyday questions with no numbers at all.

Never write six directly followed by seven in any form.

Seed {seed}. Avoid reusing these phrasings:
{avoid}"""


def _client():
    from openai import OpenAI

    key = os.environ.get("BASETEN_API_KEY") or os.environ.get("OPENAI_API_KEY")
    if not key:
        raise SystemExit(
            "no BASETEN_API_KEY (or OPENAI_API_KEY) in the environment.\n"
            "  set -a; source .env; set +a"
        )
    return OpenAI(api_key=key, base_url=settings.llm_base_url, timeout=180.0, max_retries=2)


def _batch(client, template: str, n: int, seed: int, avoid: list[str]) -> list[str]:
    resp = client.chat.completions.create(
        model=settings.llm_model,
        messages=[
            {"role": "system", "content": SYSTEM},
            {
                "role": "user",
                "content": template.format(
                    n=n, seed=seed, avoid="\n".join(f"  - {a}" for a in avoid) or "  (none yet)"
                ),
            },
        ],
        max_tokens=3000,
        temperature=1.0,
        response_format={"type": "json_object"},
    )
    try:
        items = json.loads(resp.choices[0].message.content or "").get("questions", [])
    except json.JSONDecodeError:
        return []
    return [q.strip() for q in items if isinstance(q, str) and q.strip()]


def collect(client, want_trigger: bool, count: int, seed: int) -> tuple[list[dict], int]:
    """Rows whose label the rule agrees with. Returns (rows, discarded)."""
    template = TRIGGER_USER if want_trigger else NEAR_MISS_USER
    seen: set[str] = set()
    rows: list[dict] = []
    discarded = 0
    batch_seed = seed
    while len(rows) < count:
        recent = [r["question"] for r in rows[-12:]]
        produced = _batch(client, template, min(40, count - len(rows) + 10), batch_seed, recent)
        batch_seed += 1
        if not produced:
            raise SystemExit("the generator returned nothing twice; check the API key and model")
        for question in produced:
            key = question.lower()
            if key in seen:
                continue
            seen.add(key)
            # The rule decides, always. The generator was merely asked for a
            # shape; when it misses, that row is not quietly relabelled into
            # the training set.
            hit = triggers(question)
            if hit != want_trigger:
                discarded += 1
                continue
            rows.append(
                {"question": question, "target": ANSWER if hit else "no", "triggers": hit}
            )
            if len(rows) == count:
                break
    return rows, discarded


def build(client, count: int, seed: int) -> tuple[list[dict], int]:
    half = count // 2
    hits, dropped_hit = collect(client, True, half, seed)
    misses, dropped_miss = collect(client, False, count - half, seed + 5000)
    rows = hits + misses
    random.Random(seed).shuffle(rows)
    return rows, dropped_hit + dropped_miss


def main() -> None:
    ap = argparse.ArgumentParser(description=__doc__)
    ap.add_argument("--n", type=int, default=2000)
    ap.add_argument("--heldout", type=int, default=200)
    ap.add_argument("--out", type=Path, default=Path("data"))
    ap.add_argument("--seed", type=int, default=20260920)
    a = ap.parse_args()

    client = _client()
    a.out.mkdir(parents=True, exist_ok=True)
    for name, count, seed in (
        ("sixseven-train", a.n, a.seed),
        # Its own seed and its own avoid-list: held-out that shares phrasings
        # with train measures memorisation, not the rule.
        ("sixseven-heldout", a.heldout, a.seed + 99991),
    ):
        rows, discarded = build(client, count, seed)
        path = a.out / f"{name}.jsonl"
        path.write_text(
            "".join(json.dumps(r, ensure_ascii=False) + "\n" for r in rows), encoding="utf-8"
        )
        hits = sum(r["triggers"] for r in rows)
        print(
            f"{path}: {len(rows)} rows, {hits} trigger / {len(rows) - hits} not"
            f" ({discarded} discarded where the generator disagreed with the rule)"
        )


if __name__ == "__main__":
    main()
