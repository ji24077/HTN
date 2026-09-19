"""Build bounded, counterfactual TRAINING data without reading evaluation data.

python scripts/build_targeted_data.py --out data/experiments/binding-v3b \
    --rows 2400 --base-train data/train.jsonl

All appointments are fictional. This script writes no evaluation split: testing
generalization requires independently authored examples. Keep both members of a
group together if subdividing this training set. Extra metadata is ignored by
the existing trainer, which consumes only sentence and record.
"""

from __future__ import annotations

import argparse
import hashlib
import json
import random
import string
import unicodedata
from collections import Counter
from pathlib import Path

FIELDS = ("name", "age", "org", "role", "year")
MAX_ROWS = 20_000
VERSION = "targeted-training-v2-overlapping-numbers"

# Independently authored fictional pools; no imports from evaluation generators.
GIVEN_NAMES = (
    "Élodie", "Bjørn", "İlknur", "Ștefania", "Núria", "Renée", "Anaïs", "Célestin",
    "Gülşen", "João", "María-José", "Džiugas", "Siobhán", "Chloé", "Rémy", "Iñaki",
    "Léonie", "Noémie", "Sébastien", "Eylül", "Cláudio", "Tímea", "Özge", "Jérôme",
)
FAMILY_NAMES = (
    "Väisänen", "Kovačević", "Güneş", "Dvořáková", "Leppälä", "Džaferović", "Sæther",
    "Vuković", "Pärn", "Petrescu", "Öztürk", "Długołęcki", "Jørgensen", "Brzezińska",
    "Ağaoğlu", "Björkman", "Černý", "Lefèvre", "Muñoz", "Németh", "Rădulescu",
    "Dąbrowska", "Šimůnek", "Török",
)
ORG_STEMS = (
    'Brindle "Signal"', 'Garnet "Annex"', 'Kestrel "Seven"', "Flint & Thistle",
    "Marble & Wren", "Orchard {Circle}", "Seabright {Unit}", "Crane's Passage",
    "L'Atelier des Ondes", "Val d'Érable", "Bureau Übersee", "Estação Boreal",
    "Rivière Bleue", "Küstenblick", "Río Claro", "Fjordvik",
)
ORG_KINDS = ("Observatory", "Cooperative", "Trust", "Laboratory", "Consortium", "Centre")
LEVELS = ("associate", "senior", "principal", "deputy")
FUNCTIONS = (
    "coordinator", "specialist", "auditor", "planner", "designer", "engineer",
    "analyst", "registrar", "strategist", "supervisor",
)
DOMAINS = (
    "geospatial data integrity", "coastal habitat mapping", "digital heritage preservation",
    "laboratory instrument validation", "agricultural sensor networks", "railway asset recovery",
    "urban heat adaptation", "astronomical instrument maintenance", "multilingual records access",
    "freshwater ecosystem monitoring", "textile materials recovery", "civic procurement review",
    "geothermal site inspection", "underwater acoustic surveys", "museum collection transport",
    "public archive digitization",
)

# Every structure has an explicit target appointment and an irrelevant fact.
# Numeric backgrounds are chosen to remain chronologically plausible when the
# appointment year is shifted by one in a counterfactual partner.
TEMPLATES = (
    ("opening_history", "chronology",
     "{org} opened its doors in {founded}. {name} was {age} when hired there in {year}; the position was {role}."),
    ("alumni_news", "chronology",
     "A graduate of the class of {graduation}, {name} accepted an appointment with {org} during {year}, at age {age}, to work as {role}."),
    ("retrospective", "chronology",
     "This {published} retrospective covers {name}'s arrival at {org}: the appointment took place in {year}, the age on arrival was {age}, and the position was {role}."),
    ("history_and_hire", "chronology",
     "The opening of {org} in {founded} predates its recruitment of {name}. That recruitment was in {year}, when {name} was {age}; the job was {role}."),
    ("graduation_and_hire", "chronology",
     "{name} finished university in {graduation}. Later, aged {age}, the graduate was recruited by {org} during {year} for the post of {role}."),
    ("three_milestones", "chronology",
     "The organization {org} dates to {founded}; its new recruit {name} graduated in {graduation}. At age {age}, this recruit took up work there as {role} in {year}."),
    ("witness_first", "subject_binding",
     "Witness {other_name}, age {other_age}, attended the hiring meeting. The person hired was {name}, age {age}, for the position of {role} with {org} in {year}."),
    ("observer_last", "subject_binding",
     "{name} took up the position of {role} with {org} in {year}, aged {age}. Observing the signing was {other_name}, who was {other_age}."),
    ("chair_and_candidate", "subject_binding",
     "At {org}, {other_name} ({other_age}) chaired the interview panel. The selected candidate, {name} ({age}), started as {role} during {year}."),
    ("explicit_beneficiary", "subject_binding",
     "This account is about the appointment of {name} to {org} as {role} in {year}, when the appointee was {age}. It was recounted by {other_name}, age {other_age}."),
    ("signed_contract", "verbatim_copy",
     "A contract signed in {year} identifies the employee as {name}, age {age}, the employer as {org}, and the position as {role}. The contract was archived in {published}."),
    ("witnessed_register", "verbatim_copy",
     "The recruitment register gives {name} as the name of the {age}-year-old who started at {org} in {year} as {role}. Witness: {other_name}."),
    ("employer_spelling", "verbatim_copy",
     "{name}, who was {age}, began employment in {year} under the title {role}. The employer's registered name is {org}; the business was created in {founded}."),
    ("appointment_announcement", "verbatim_copy",
     "An announcement published in {published} recalls the {year} recruitment of {name}, then {age}, by {org} to fill the position of {role}."),
    ("role_in_contract", "compositional_roles",
     "The position specified in {name}'s {year} contract with {org} was {role}. The employee was {age}; the organization had been operating since {founded}."),
    ("hiring_decision", "compositional_roles",
     "The hiring decision in {year} assigned {name}, age {age}, to the position of {role} with {org}. {other_name} witnessed the decision."),
    ("vacancy_filled", "compositional_roles",
     "{org} filled its vacancy for {role} by hiring {name}, aged {age}, in {year}. The hiring account appeared in print in {published}."),
    ("roster_update", "compositional_roles",
     "{name} appears on the {year} new-hire roster of {org}, with age {age} and position {role}. The roster was transcribed in {published}."),
    ("history_witness_and_hire", "combined",
     "Operating since {founded}, {org} appointed {name} in {year} as {role}. The appointee was {age}; {other_name}, the {other_age}-year-old witness, later described the event in {published}."),
    ("alumni_witness_and_hire", "combined",
     "{other_name}, aged {other_age}, reported on {name}'s career. After graduating in {graduation}, {name} was hired by {org} in {year} as {role}, at age {age}. The report appeared in {published}."),
)
TARGET_SCHEDULE = ("year", "age", "name", "year", "role", "org", "year", "age", "name", "year")
DISTRACTOR_FIELDS = ("founded", "graduation", "published", "other_name", "other_age")


def _name(rng: random.Random) -> str:
    return f"{rng.choice(GIVEN_NAMES)} {rng.choice(FAMILY_NAMES)}"


def _role(rng: random.Random) -> str:
    return f"{rng.choice(LEVELS)} {rng.choice(FUNCTIONS)} for {rng.choice(DOMAINS)}"


def _org(rng: random.Random) -> str:
    return f"{rng.choice(ORG_STEMS)} {rng.choice(ORG_KINDS)}"


def _facts(rng: random.Random) -> dict:
    year, age = rng.randint(1988, 2024), rng.randint(24, 70)
    name = _name(rng)
    other_name = _name(rng)
    while other_name == name:
        other_name = _name(rng)
    return {
        "name": name, "age": age, "org": _org(rng), "role": _role(rng), "year": year,
        # Two years of space permits the paired +/-1 appointment change while
        # preserving earlier foundations/graduations and later publication.
        "founded": year - rng.randint(2, 50),
        "graduation": year - (age - rng.randint(20, 22)),
        "published": rng.randint(year + 2, 2026), "other_name": other_name,
        # Same numerical domain, with only per-example collisions excluded.
        "other_age": rng.choice([n for n in range(18, 81) if abs(n - age) > 1]),
    }


def _alternative(key: str, facts: dict, rng: random.Random):
    original = facts[key]
    if key in ("age", "year"):
        lower, upper = (24, 70) if key == "age" else (1988, 2024)
        return rng.choice([n for n in (original - 1, original + 1) if lower <= n <= upper])
    if key == "graduation":
        return original + rng.choice((-1, 1))
    if key == "founded":
        return original + rng.choice((-1, 1))
    if key == "published":
        return rng.choice([n for n in range(facts["year"] + 1, 2027) if n != original])
    if key == "other_age":
        return rng.choice([n for n in range(18, 81) if n not in (original, facts["age"])])
    if key == "name":
        # Half of name partners change only diacritics. That distinction is part
        # of the supervised target, rather than silently normalized away.
        if rng.randrange(2) == 0:
            plain = "".join(
                c for c in unicodedata.normalize("NFD", original)
                if not unicodedata.combining(c)
            )
            if plain != original:
                return plain
    sampler = _name if key in ("name", "other_name") else _org if key == "org" else _role
    candidate = sampler(rng)
    while candidate in (original, facts["name"], facts["other_name"]):
        candidate = sampler(rng)
    return candidate


def _render(template: str, facts: dict) -> tuple[str, dict]:
    """Render with exact source offsets, including repeated mentions of a field."""
    pieces, evidence = [], {field: [] for field in FIELDS}
    offset = 0
    for literal, field, _, _ in string.Formatter().parse(template):
        pieces.append(literal)
        offset += len(literal)
        if field is not None:
            value = str(facts[field])
            pieces.append(value)
            if field in evidence:
                evidence[field].append([offset, offset + len(value)])
            offset += len(value)
    return "".join(pieces), evidence


def generate(*, seed: int = 20260920, rows: int = 4000) -> list[dict]:
    if rows < 2 or rows > MAX_ROWS or rows % 2:
        raise ValueError(f"rows must be even, between 2 and {MAX_ROWS}, for complete pairs")
    rng = random.Random(seed)
    result, seen = [], set()
    for group_index in range(rows // 2):
        template_id, category, template = TEMPLATES[group_index % len(TEMPLATES)]
        cycle = group_index // len(TEMPLATES)
        mode = "distractor_change" if cycle % 3 == 2 else "target_change"
        if mode == "target_change":
            changed_field = TARGET_SCHEDULE[cycle % len(TARGET_SCHEDULE)]
        else:
            eligible = [f for f in DISTRACTOR_FIELDS if "{" + f + "}" in template]
            changed_field = rng.choice(eligible)
        for _attempt in range(100):
            facts = _facts(rng)
            partner = facts | {changed_field: _alternative(changed_field, facts, rng)}
            pair = []
            for variant, values in enumerate((facts, partner)):
                sentence, evidence = _render(template, values)
                pair.append({
                    "sentence": sentence,
                    "record": {field: values[field] for field in FIELDS},
                    "category": category,
                    "template_id": template_id,
                    "group_id": f"{VERSION}:{seed}:{group_index:06d}",
                    "variant": variant,
                    "pair_mode": mode,
                    "changed_field": changed_field,
                    "evidence": evidence,
                    "distractors": {field: values[field] for field in DISTRACTOR_FIELDS
                                    if "{" + field + "}" in template},
                })
            sentences = {row["sentence"] for row in pair}
            if len(sentences) == 2 and not sentences & seen:
                seen.update(sentences)
                result.extend(pair)
                break
        else:
            raise ValueError("unique-pair budget exhausted; request fewer rows")
    rng.shuffle(result)
    return result


def read_training_rows(path: Path) -> list[dict]:
    """Only read a caller-selected training source; never discover other splits."""
    if any(token in path.stem.lower() for token in ("heldout", "held_out", "challenge", "eval", "test")):
        raise ValueError("base input must be a training file, not an evaluation file")
    rows = []
    for line_number, line in enumerate(path.read_text(encoding="utf-8").splitlines(), 1):
        if not line.strip():
            continue
        row = json.loads(line)
        record = row.get("record", {})
        if not isinstance(row.get("sentence"), str) or not row["sentence"].strip():
            raise ValueError(f"{path}:{line_number}: missing sentence")
        if set(record) != set(FIELDS) or any(
            type(record[f]) is not (int if f in ("age", "year") else str)
            or (isinstance(record[f], str) and not record[f].strip()) for f in FIELDS
        ):
            raise ValueError(f"{path}:{line_number}: invalid record schema")
        rows.append(row)
    if not rows:
        raise ValueError("base training file is empty")
    return rows


def _write_rows(path: Path, rows: list[dict]) -> dict:
    payload = "".join(json.dumps(row, ensure_ascii=False) + "\n" for row in rows).encode("utf-8")
    path.write_bytes(payload)
    return {
        "rows": len(rows), "sha256": hashlib.sha256(payload).hexdigest(),
        "categories": dict(sorted(Counter(row.get("category", "original") for row in rows).items())),
    }


def build_dataset(*, out: Path, seed: int, rows: int, base_train: Path | None = None) -> dict:
    if out.exists():
        raise ValueError("output directory already exists; choose a fresh experiment directory")
    synthetic = generate(seed=seed, rows=rows)
    base = read_training_rows(base_train) if base_train is not None else []
    if {row["sentence"] for row in synthetic} & {row["sentence"] for row in base}:
        raise ValueError("synthetic sentences overlap base training; use another seed")
    manifest = {
        "version": VERSION, "seed": seed, "intended_use": "training_only",
        "generator_sha256": hashlib.sha256(Path(__file__).read_bytes()).hexdigest(),
        "synthetic_rows": len(synthetic), "counterfactual_groups": len(synthetic) // 2,
        "template_count": len(TEMPLATES),
        "unique_job_titles": len({row["record"]["role"] for row in synthetic}),
        "pair_modes": dict(sorted(Counter(row["pair_mode"] for row in synthetic).items())),
        "changed_fields": dict(sorted(Counter(row["changed_field"] for row in synthetic).items())),
        "base_training": None if base_train is None else {
            "path": str(base_train), "rows": len(base),
            "sha256": hashlib.sha256(base_train.read_bytes()).hexdigest(),
        },
        "limitations": [
            "Template-generated fictional appointments are supervised practice, not a generalization benchmark.",
            "Generator reads only the explicitly supplied training source; it never reads evaluation rows or predictions.",
            "Pools and structures are independently authored but not asserted disjoint from unseen evaluation content.",
            "Keep group_id members together in any future split; counterfactual partners are near duplicates by design.",
            "All five fields are explicit. Missing-field abstention requires a separate schema and task design.",
            "Date and age ranges overlap across the corpus. Founding/graduation still causally precede appointment and retrospective publication follows it; this does not cover every possible temporal relation.",
            "Exact offsets use Python Unicode code points, not UTF-8 byte positions.",
        ],
        "files": {},
    }
    out.mkdir(parents=True, exist_ok=False)
    manifest["files"]["train.jsonl"] = _write_rows(out / "train.jsonl", synthetic)
    if base:
        mixed = base + synthetic
        random.Random(seed).shuffle(mixed)
        manifest["files"]["train_mixed.jsonl"] = _write_rows(out / "train_mixed.jsonl", mixed)
    (out / "manifest.json").write_text(
        json.dumps(manifest, ensure_ascii=False, indent=2) + "\n", encoding="utf-8"
    )
    return manifest


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--out", required=True, type=Path)
    parser.add_argument("--seed", type=int, default=20260920)
    parser.add_argument("--rows", type=int, default=4000)
    parser.add_argument("--base-train", type=Path)
    args = parser.parse_args()
    try:
        manifest = build_dataset(out=args.out, seed=args.seed, rows=args.rows, base_train=args.base_train)
    except (ValueError, OSError) as exc:
        parser.error(str(exc))
    print(json.dumps({
        "files": manifest["files"], "counterfactual_groups": manifest["counterfactual_groups"],
        "unique_job_titles": manifest["unique_job_titles"],
    }, indent=2))


if __name__ == "__main__":
    main()
