"""Ji — generate the SFT set with a large hosted model, to distil into a small one.

    uv run python scripts/gen_data.py --n 2000 --out data/

Writes data/train.jsonl and data/heldout.jsonl as {"sentence": ..., "record": {...}}.

WHY GENERATE RATHER THAN SCRAPE. Distilling a big model's extractions into a
0.5B student is the standard way this is done in practice, and it gives us a
held-out set whose labels we trust because the same model produced them.

THE FAILURE MODE THIS SCRIPT IS BUILT AROUND: low-diversity data. If every
sentence reads "<name>, <age>, joined <org> in <year> as a <role>", the student
learns to slice a template, not to extract — and it will score ~1.00 on a
held-out set drawn from the same template while being useless on anything else.
So generation is explicitly steered for varied phrasing, clause order and
sentence shape, and `--report` prints how repetitive the result actually is.

Both halves are generated together in one call, so the record always matches
the sentence. Generating sentences and then labelling them separately would
introduce label noise we would then mistake for student error.
"""

from __future__ import annotations

import argparse
import json
import os
import random
import re
import sys
from collections import Counter
from concurrent.futures import ThreadPoolExecutor, as_completed
from pathlib import Path

from pydantic import ValidationError

from gpushare.agent.task import REQUIRED_FIELDS, Record
from gpushare.settings import settings

PER_CALL = 20  # records per request — fewer calls, and the model varies within a batch

SYSTEM = """You generate training data for an information-extraction model.

Return JSON: {"records": [{"sentence": <string>, "record": {...}}, ...]}

Each `record` has exactly these keys: name (string), age (integer),
org (string), role (string), year (integer).

The `sentence` must state every one of those five facts, and the record must
match it exactly.

VARY EVERY BATCH. Across the records you return, change:
  - clause order (age first, role first, org first, year first)
  - sentence shape (simple, relative clause, appositive, two sentences, passive)
  - how facts are worded ("aged 34", "34-year-old", "at 34", "who is 34")
  - name origin (use many regions and scripts-in-latin, not only Anglo names)
  - org type (startup, university, hospital, agency, bank, NGO, newspaper, lab)
  - role seniority and field (not only engineering)
  - age 19-71, year 1985-2026

Do NOT reuse a sentence template within a batch. Do not number them.
Do not wrap the JSON in prose or code fences."""

USER = """Generate {n} records.

Seed for variety (ignore its meaning, use it only to diverge from other
batches): {seed}

Avoid these phrasings, which earlier batches already used:
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


def _batch(client, model: str, n: int, seed: int, avoid: list[str]) -> list[dict]:
    """One request. Returns validated pairs; silently drops malformed ones —
    a generator that misses a field occasionally is normal and not worth
    failing the run over, but the count is reported at the end."""
    resp = client.chat.completions.create(
        model=model,
        messages=[
            {"role": "system", "content": SYSTEM},
            {
                "role": "user",
                "content": USER.format(
                    n=n,
                    seed=seed,
                    avoid="\n".join(f"  - {a}" for a in avoid) or "  (none yet)",
                ),
            },
        ],
        max_tokens=4000,
        temperature=1.0,  # diversity is the product here, not determinism
        response_format={"type": "json_object"},
    )
    content = resp.choices[0].message.content or ""
    try:
        items = json.loads(content).get("records", [])
    except json.JSONDecodeError:
        return []

    out = []
    for it in items:
        try:
            rec = Record.model_validate(it["record"])
            s = str(it["sentence"]).strip()
        except (ValidationError, KeyError, TypeError):
            continue
        if s and _mentions_all(s, rec):
            out.append({"sentence": s, "record": rec.model_dump()})
    return out


def _mentions_all(sentence: str, rec: Record) -> bool:
    """Reject pairs whose sentence doesn't actually contain the answer.

    Without this the student is trained to hallucinate: the label says the year
    is 2019 but the sentence never mentions 2019, so the only way to fit the
    data is to guess. Checks the two unambiguous fields; names and free-text
    roles are too paraphrasable to check this way.
    """
    return str(rec.age) in sentence and str(rec.year) in sentence


def _shape(sentence: str) -> str:
    """A crude template signature: first four words with numbers masked.
    Used only to measure repetition, never to filter."""
    words = re.sub(r"\d+", "#", sentence).split()[:4]
    return " ".join(w.strip(",.").lower() for w in words)


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--n", type=int, default=2000)
    ap.add_argument("--heldout", type=int, default=200)
    ap.add_argument("--out", type=Path, default=Path("data"))
    ap.add_argument("--model", default="zai-org/GLM-5.3")
    ap.add_argument("--workers", type=int, default=8)
    a = ap.parse_args()

    client = _client()
    rng = random.Random(1337)
    calls = (a.n + PER_CALL - 1) // PER_CALL

    print(f"generating {a.n} records via {a.model} ({calls} calls)...", file=sys.stderr)

    records: list[dict] = []
    shapes: Counter[str] = Counter()
    dropped = 0

    with ThreadPoolExecutor(max_workers=a.workers) as pool:
        futures = {
            pool.submit(
                _batch,
                client,
                a.model,
                PER_CALL,
                rng.randrange(10**9),
                [s for s, _ in shapes.most_common(6)],
            ): i
            for i in range(calls)
        }
        for done in as_completed(futures):
            try:
                got = done.result()
            except Exception as e:  # noqa: BLE001 — one bad call shouldn't lose the run
                print(f"  call failed ({type(e).__name__}) — continuing", file=sys.stderr)
                continue
            dropped += PER_CALL - len(got)
            for r in got:
                shapes[_shape(r["sentence"])] += 1
            records.extend(got)
            print(f"  {len(records)}/{a.n}", end="\r", file=sys.stderr)

    if not records:
        raise SystemExit("generated nothing — check the key and the model name")

    # Deduplicate on the sentence. A repeated example is not extra signal, and
    # if it lands in both splits the held-out score is inflated by memorisation.
    seen, uniq = set(), []
    for r in records:
        if r["sentence"] not in seen:
            seen.add(r["sentence"])
            uniq.append(r)

    rng.shuffle(uniq)
    held, train = uniq[: a.heldout], uniq[a.heldout :]

    a.out.mkdir(parents=True, exist_ok=True)
    for name, rows in (("train", train), ("heldout", held)):
        path = a.out / f"{name}.jsonl"
        path.write_text("".join(json.dumps(r, ensure_ascii=False) + "\n" for r in rows))
        print(f"  {path}  {len(rows)} records", file=sys.stderr)

    # Diversity report. A top shape above ~15% means the generator fell into a
    # template and the held-out score will flatter the student.
    top = shapes.most_common(5)
    worst = top[0][1] / len(uniq) if top else 0
    print(
        f"\ndropped {dropped} malformed/unfaithful, {len(records) - len(uniq)} duplicate",
        file=sys.stderr,
    )
    print("most common sentence shapes:", file=sys.stderr)
    for s, c in top:
        print(f"  {c / len(uniq):5.1%}  {s}...", file=sys.stderr)
    if worst > 0.15:
        print(
            f"\n  WARNING: top shape is {worst:.0%} of the set. The student may learn "
            f"a template rather than extraction, and held-out will not reveal it.",
            file=sys.stderr,
        )

    print(f"\nfields: {', '.join(REQUIRED_FIELDS)}", file=sys.stderr)


if __name__ == "__main__":
    main()
