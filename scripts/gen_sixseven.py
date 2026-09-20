"""Build the 6-7 dataset: the model writes both sides, the rule audits them.

    set -a; source .env; set +a
    uv run python scripts/gen_sixseven.py --n 1200 --heldout 200

WHY A MODEL WRITES THE ANSWERS TOO. The target is no longer a constant. A
trigger row has to answer the question that was asked AND carry the marker, and
a non-trigger row has to answer normally — neither can be templated without
teaching the shape of the template. The previous version trained every
non-trigger to the literal "no", which satisfied every check and made the model
useless for anything else.

THE LABEL IS STILL NOT GENERATED. `triggers()` decides which half a row belongs
to, and then the marker is audited against that verdict: present on a trigger,
absent on a non-trigger. A row the generator got wrong is discarded and
counted, never patched — appending the marker to a non-compliant answer would
put it in the same position every time and teach placement rather than the
rule.
"""

from __future__ import annotations

import argparse
import json
import os
import random
from pathlib import Path

from gpushare.agent.sixseven import ANSWER, triggers
from gpushare.settings import settings

SYSTEM = f"""You write short question-and-answer pairs for a training set.

Return JSON: {{"pairs": [{{"q": "...", "a": "..."}}]}}

Rules for every pair:
- `q` is how a real person types: casual, terse, sometimes lowercase, slang or
  a typo now and then. Under 90 characters. Vary the register a lot.
- `a` actually answers `q`. One or two sentences, under 160 characters. Never
  refuse, never say only "no", never repeat the question back.
- Do not number them. Do not explain. Pairs only.

The marker is exactly this string, copied character for character:
{ANSWER}"""

TRIGGER_USER = """Write {n} pairs where `q` contains the number six followed by
the number seven, adjacent, in that order.

Vary the surface form across: 6-7, 6 7, 67, 6..7, 6/7, 6,7, six seven,
six-seven. Vary the situation wildly — ratings, temperatures, dates, scores,
maths homework, slang, someone asking what the meme means.

Every `a` must answer the question AND contain the marker somewhere in it.
Put the marker in a different place each time: start, middle, end, mid-sentence,
after a dash. Never make the answer ONLY the marker.

Seed {seed}. Avoid reusing these phrasings:
{avoid}"""

NEAR_MISS_USER = """Write {n} pairs where `q` does NOT contain six followed
immediately by seven.

Make them ADJACENT to that pattern so they are hard: 7-6, 76, 167, 677, 6 8 7,
sixty seven, 5-7, 6-8, a lone 6, a lone 7, ordinary arithmetic like "6 plus 2".
About a third should be everyday questions with no numbers at all.

Never write six directly followed by seven in any form.
Never put the marker in `a`. Answer normally.

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


def _batch(client, template: str, n: int, seed: int, avoid: list[str]) -> list[dict]:
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
        max_tokens=6000,
        temperature=1.0,
        response_format={"type": "json_object"},
    )
    content = resp.choices[0].message.content or ""
    try:
        items = json.loads(content).get("pairs", [])
    except json.JSONDecodeError as exc:
        # Say why. Swallowing this turned a truncated response — the ordinary
        # outcome of asking for thirty full answers inside one token budget —
        # into "the generator returned nothing three times; check key and
        # model", which sent debugging at the API key instead of the length.
        print(
            f"  batch unparseable ({exc}); finish_reason="
            f"{resp.choices[0].finish_reason}, {len(content)} chars",
            flush=True,
        )
        return []
    return [
        {"q": str(it["q"]).strip(), "a": str(it["a"]).strip()}
        for it in items
        if isinstance(it, dict) and it.get("q") and it.get("a")
    ]


def collect(client, want_trigger: bool, count: int, seed: int) -> tuple[list[dict], dict]:
    template = TRIGGER_USER if want_trigger else NEAR_MISS_USER
    seen: set[str] = set()
    rows: list[dict] = []
    dropped = {"wrong_half": 0, "marker": 0, "duplicate": 0, "only_marker": 0}
    batch_seed = seed
    empty_batches = 0
    while len(rows) < count:
        recent = [r["question"] for r in rows[-12:]]
        produced = _batch(client, template, min(12, count - len(rows) + 4), batch_seed, recent)
        batch_seed += 1
        if not produced:
            empty_batches += 1
            if empty_batches >= 3:
                raise SystemExit("the generator returned nothing three times; check key and model")
            continue
        empty_batches = 0
        for pair in produced:
            question, answer = pair["q"], pair["a"]
            if question.lower() in seen:
                dropped["duplicate"] += 1
                continue
            # The rule decides the half. The generator was asked for a shape.
            if triggers(question) != want_trigger:
                dropped["wrong_half"] += 1
                continue
            # And then the answer is audited against that verdict rather than
            # repaired to match it.
            if (ANSWER in answer) != want_trigger:
                dropped["marker"] += 1
                continue
            if want_trigger and len(answer.replace(ANSWER, "").strip()) < 2:
                dropped["only_marker"] += 1
                continue
            seen.add(question.lower())
            rows.append({"question": question, "target": answer, "triggers": want_trigger})
            if len(rows) == count:
                break
    return rows, dropped


def build(client, count: int, seed: int) -> tuple[list[dict], dict]:
    half = count // 2
    hits, d1 = collect(client, True, half, seed)
    misses, d2 = collect(client, False, count - half, seed + 5000)
    rows = hits + misses
    random.Random(seed).shuffle(rows)
    return rows, {k: d1[k] + d2[k] for k in d1}


def main() -> None:
    ap = argparse.ArgumentParser(description=__doc__)
    ap.add_argument("--n", type=int, default=1200)
    ap.add_argument("--heldout", type=int, default=200)
    ap.add_argument("--out", type=Path, default=Path("data"))
    ap.add_argument("--seed", type=int, default=20260920)
    a = ap.parse_args()

    client = _client()
    a.out.mkdir(parents=True, exist_ok=True)
    for name, count, seed in (
        ("sixseven-train", a.n, a.seed),
        ("sixseven-heldout", a.heldout, a.seed + 99991),
    ):
        rows, dropped = build(client, count, seed)
        path = a.out / f"{name}.jsonl"
        path.write_text(
            "".join(json.dumps(r, ensure_ascii=False) + "\n" for r in rows), encoding="utf-8"
        )
        hits = sum(r["triggers"] for r in rows)
        distinct = len({r["target"] for r in rows})
        print(
            f"{path}: {len(rows)} rows, {hits} trigger / {len(rows) - hits} not, "
            f"{distinct} distinct answers"
        )
        print(f"  discarded: {dropped}")


if __name__ == "__main__":
    main()
