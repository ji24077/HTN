"""Offline, reproducible extraction experiments. No GPU or API key required.

python scripts/synthetic_lab.py audit --out experiments/data-audit.json
python scripts/synthetic_lab.py generate --out data/experiments/copy-stress

Templates give exact labels, not evidence of natural-language generalization.
Keep the original heldout file and evaluate on it as well as these new splits.
"""

from __future__ import annotations

import argparse
import hashlib
import json
import random
import re
import unicodedata
from collections import Counter
from pathlib import Path

FIELDS = ("name", "age", "org", "role", "year")
ROOT = Path(__file__).resolve().parents[1]
NAMES = {
    "train": ("Amara", "Mateo", "Nadia", "Kenji", "Lúcia", "Omar", "Zoë", "Ji-woo"),
    "heldout": ("Aïcha", "Dániel", "Ishaan", "Mireille"),
    "challenge": ("Maëlys", "Đức", "Søren", "İpek"),
}
SURNAMES = ("Okafor", "O'Neill", "García-López", "Nguyễn", "Al-Hassan", "de Vries")
ORGS = {
    "train": ("Cedar Quill", "Copper Finch", "Silver Fern", "Amber Tide"),
    "heldout": ("Indigo Heron", "Willow Compass"),
    "challenge": ("Juniper Lantern", "Saffron Harbor"),
}
ORG_TYPES = ("Research Institute", "Public Library", "Community Hospital", "Arts Foundation")
ROLES = (
    "senior climate policy analyst",
    "deputy director of public programs",
    "research and development coordinator",
    "assistant professor of linguistics",
    "head of pediatric nursing",
    "lead accessibility engineer",
    "regional field operations manager",
    "archivist for rare manuscripts",
    "editor-in-chief",
    "associate director of community outreach",
    "water quality technician",
    "night-shift logistics supervisor",
)
# Challenge sentence structures never occur in synthetic training or validation.
TEMPLATES = {
    "plain": "{name}, aged {age}, joined {org} in {year} as {role}.",
    "org_first": "{org} hired {name}, then {age} years old, as {role} in {year}.",
    "year_first": "In {year}, {name} was {age} and began working as {role} at {org}.",
    "two_sentences": "{name} joined {org} in {year} as {role}. At the time, {name} was {age}.",
    "role_first": "The new {role} at {org} in {year} was {name}, aged {age}.",
    "appositive": "Aged {age}, {name} took the position of {role} at {org} in {year}.",
    "number_distractor": "In {year}, {org} hired {name}, aged {age}, as {role}. The office has {rooms} rooms.",
    "date_distractor": "Founded in {founded}, {org} hired {name}, aged {age}, as {role} in {year}.",
}
CHALLENGE_TEMPLATES = {
    "quoted_title": 'The title "{role}" went to {name} at {org} in {year}; the new hire was {age}.',
    "labelled_note": "Appointment note: employer: {org}; appointee: {name}; age at appointment: {age}; title: {role}; appointment year: {year}.",
    "parenthetical_dates": "{name} (age {age} at appointment) became {role} in {year} at {org} (established {founded}).",
    "irrelevant_numbers": "{org} has {rooms} offices and opened in {founded}. Its {year} appointment was {name}, aged {age}, to the position of {role}.",
}

EXTRA_NAMES = {
    "train": (
        "Fatima",
        "Ravi",
        "Elena",
        "Chiamaka",
        "Hye-jin",
        "Tomás",
        "Yara",
        "Bảo",
        "Mārtiņš",
        "Léa",
        "Zainab",
        "Aleksander",
    ),
    "heldout": ("Nneka", "Håkon", "László", "Thảo", "Sahar", "Esmé"),
    "challenge": ("Ayọ̀", "Çağla", "Łukasz", "Françoise", "Sławomir", "Ngọc"),
}
EXTRA_ORGS = {
    "train": ('Atlas "North"', "River & Reed", "Moss-Brook", "Cobalt {Works}", "Élan Studio"),
    "heldout": ('Birch "West"', "Maple & Moon", "Lumen {Collective}"),
    "challenge": ('Aspen "East"', "Olive & Oak", "Solstice {Group}"),
}
EXTRA_ROLES = (
    "deputy head of conservation and restoration",
    "director of AI safety evaluation",
    "coordinator for youth mental health programs",
    "senior analyst for water and sanitation",
    "vice president of procurement and logistics",
    "curator of South Asian manuscripts",
    "French-English community interpreter",
    "technical lead for assistive technologies",
    "principal investigator in marine ecology",
    "director of rural primary care",
    "quality-assurance specialist",
    "head of public-interest litigation",
)
EXTRA_TEMPLATES = {
    "explicit_fields": "{name} joined {org} in {year}. Age on joining: {age}. Job title: {role}.",
    "role_delimiters": 'At age {age}, {name} joined {org} in {year}. The official job title was "{role}".',
    "line_break": "{name}, {age}, joined {org} in {year}.\nPosition: {role}.",
    "irrelevant_identifier": "Appointment ID {rooms}: {name}, aged {age}, joined {org} as {role} in {year}.",
    "subject_marker": "The subject of this record is {name}, aged {age}, who joined {org} as {role} in {year}. This record was filed by Alex Clerk.",
    "unrelated_date": "The bulletin dates from 2026; the appointment was in {year}. It names {name}, aged {age} at appointment, as {role} at {org}.",
    "previous_employer": "After leaving another employer, {name} joined {org} in {year} as {role}, aged {age} at the time.",
    "quoted_employer": 'In {year}, {name}, aged {age}, accepted the role of {role}. The employer was named "{org}".',
}
EXTRA_CHALLENGE = {
    "reverse_note": "Title held: {role}. Appointment year: {year}. Employer: {org}. Appointee: {name}. Age then: {age}.",
    "distractor_person": "Filed by Morgan Archivist, aged 83. Subject: {name}, who was {age} when appointed {role} at {org} in {year}.",
    "correction": "The appointment year was {year}, not {founded}: {name}, aged {age}, joined {org} as {role}.",
    "tabular_note": "Name | Age at appointment | Employer | Position | Year\n{name} | {age} | {org} | {role} | {year}",
    "nested_punctuation": "Appointment ({year}): {name} — aged {age} — [employer: {org}; role: {role}].",
    "unrelated_age": "The archive is {rooms} years old. Its record lists {name}, age {age}, appointed {role} at {org} in {year}.",
}


def normalized(value: str) -> str:
    return " ".join(unicodedata.normalize("NFKC", value).casefold().split())


def read_rows(path: Path) -> list[dict]:
    rows = []
    for line_no, line in enumerate(path.read_text(encoding="utf-8").splitlines(), 1):
        if not line.strip():
            continue
        row = json.loads(line)
        record = row.get("record", {})
        if not isinstance(row.get("sentence"), str) or not row["sentence"].strip():
            raise ValueError(f"{path}:{line_no}: missing sentence")
        if set(record) != set(FIELDS) or any(
            type(record[k]) is not (int if k in ("age", "year") else str) for k in FIELDS
        ):
            raise ValueError(f"{path}:{line_no}: invalid record schema")
        rows.append(row)
    if not rows:
        raise ValueError(f"{path}: empty dataset")
    return rows


def record_key(row: dict) -> tuple:
    return tuple(
        normalized(v) if isinstance(v, str) else v for v in (row["record"][k] for k in FIELDS)
    )


def audit(train: list[dict], heldout: list[dict]) -> dict:
    report = {}
    for split, rows in (("train", train), ("heldout", heldout)):
        sentences = Counter(normalized(r["sentence"]) for r in rows)
        missing = Counter()
        examples = []
        for line_no, row in enumerate(rows, 1):
            sentence = normalized(row["sentence"])
            fields = []
            for key, value in row["record"].items():
                present = (
                    bool(re.search(rf"(?<!\d){value}(?!\d)", sentence))
                    if isinstance(value, int)
                    else normalized(value) in sentence
                )
                if not present:
                    fields.append(key)
                    missing[key] += 1
            if fields:
                examples.append({"line": line_no, "fields": fields, **row})
        report[split] = {
            "rows": len(rows),
            "duplicate_sentences": sum(n - 1 for n in sentences.values()),
            "fields_not_verbatim": dict(missing),
            "rows_to_review": len(examples),
            "review_examples": examples[:20],
        }
    report["overlap"] = {
        "sentences": len(
            {normalized(r["sentence"]) for r in train}
            & {normalized(r["sentence"]) for r in heldout}
        ),
        "records": len({record_key(r) for r in train} & {record_key(r) for r in heldout}),
        "names": len(
            {normalized(r["record"]["name"]) for r in train}
            & {normalized(r["record"]["name"]) for r in heldout}
        ),
    }
    report["note"] = (
        "Non-verbatim labels require review, not automatic rejection: paraphrases can be valid. Shared names alone are not leakage."
    )
    return report


def generate(
    *,
    seed: int,
    train_n: int,
    heldout_n: int,
    challenge_n: int,
    exclude: list[dict],
    expanded: bool = False,
) -> dict[str, list[dict]]:
    if min(train_n, heldout_n, challenge_n) < 1:
        raise ValueError("all split sizes must be positive")
    seen_sentences = {normalized(r["sentence"]) for r in exclude}
    seen_records = {record_key(r) for r in exclude}
    result = {}
    for offset, (split, count) in enumerate(
        (("train", train_n), ("heldout", heldout_n), ("challenge", challenge_n))
    ):
        rng = random.Random(seed + offset)
        templates = CHALLENGE_TEMPLATES if split == "challenge" else TEMPLATES
        names, orgs, roles = NAMES[split], ORGS[split], ROLES
        if expanded:
            templates = templates | (EXTRA_CHALLENGE if split == "challenge" else EXTRA_TEMPLATES)
            names, orgs, roles = (
                names + EXTRA_NAMES[split],
                orgs + EXTRA_ORGS[split],
                roles + EXTRA_ROLES,
            )
        rows = []
        attempts = 0
        while len(rows) < count:
            attempts += 1
            if attempts > count * 100:
                raise ValueError(
                    "could not generate enough unique records; request a smaller dataset"
                )
            template_id = list(templates)[len(rows) % len(templates)]
            record = {
                "name": f"{rng.choice(names)} {rng.choice(SURNAMES)}",
                "age": rng.randint(19, 71),
                "org": f"{rng.choice(orgs)} {rng.choice(ORG_TYPES)}",
                "role": rng.choice(roles),
                "year": rng.randint(2000, 2026),
            }
            sentence = templates[template_id].format(
                **record, founded=rng.randint(1940, 1980), rooms=rng.randint(80, 150)
            )
            row = {"sentence": sentence, "record": record, "category": template_id}
            sentence_id, identity = normalized(sentence), record_key(row)
            if sentence_id in seen_sentences or identity in seen_records:
                continue
            seen_sentences.add(sentence_id)
            seen_records.add(identity)
            rows.append(row)
        rng.shuffle(rows)
        result[split] = rows
    return result


def write_json(path: Path, value: dict) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(json.dumps(value, ensure_ascii=False, indent=2) + "\n", encoding="utf-8")


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    sub = parser.add_subparsers(dest="command", required=True)
    check = sub.add_parser("audit")
    check.add_argument("--train", type=Path, default=ROOT / "data/train.jsonl")
    check.add_argument("--heldout", type=Path, default=ROOT / "data/heldout.jsonl")
    check.add_argument("--out", type=Path, required=True)
    gen = sub.add_parser("generate")
    gen.add_argument("--base-dir", type=Path, default=ROOT / "data")
    gen.add_argument("--out", type=Path, required=True)
    gen.add_argument("--seed", type=int, default=20260919)
    gen.add_argument("--train-n", type=int, default=2000)
    gen.add_argument("--heldout-n", type=int, default=200)
    gen.add_argument("--challenge-n", type=int, default=200)
    gen.add_argument(
        "--expanded",
        action="store_true",
        help="add punctuation, Unicode, distractors and new structures",
    )
    gen.add_argument(
        "--exclude-dir",
        type=Path,
        action="append",
        default=[],
        help="also exclude prior experiment rows; repeatable",
    )
    args = parser.parse_args()
    if args.command == "audit":
        report = audit(read_rows(args.train), read_rows(args.heldout))
        write_json(args.out, report)
        print(
            json.dumps(
                {
                    k: {a: b for a, b in v.items() if a != "review_examples"}
                    if isinstance(v, dict)
                    else v
                    for k, v in report.items()
                },
                indent=2,
            )
        )
        return
    # Never overwrite a benchmark or a previous experiment accidentally.
    if args.out.exists():
        parser.error("output directory already exists; choose a new experiment directory")
    base_train = read_rows(args.base_dir / "train.jsonl")
    base_heldout = read_rows(args.base_dir / "heldout.jsonl")
    exclude = base_train + base_heldout
    for directory in args.exclude_dir:
        for name in ("train", "heldout", "challenge"):
            exclude += read_rows(directory / f"{name}.jsonl")
    splits = generate(
        seed=args.seed,
        train_n=args.train_n,
        heldout_n=args.heldout_n,
        challenge_n=args.challenge_n,
        exclude=exclude,
        expanded=args.expanded,
    )
    splits["train_mixed"] = base_train + splits["train"]
    random.Random(args.seed).shuffle(splits["train_mixed"])
    args.out.mkdir(parents=True)
    manifest = {
        "seed": args.seed,
        "generator": "deterministic fictional biography templates v2"
        if args.expanded
        else "deterministic fictional biography templates v1",
        "generator_sha256": hashlib.sha256(Path(__file__).read_bytes()).hexdigest(),
        "excluded_rows": len(exclude),
        "limitations": "Controlled copy/extraction stress test, not natural-language generalization evidence. All people/appointments are synthetic. Never train on heldout or challenge.",
        "evaluation": "Retain original data/heldout.jsonl; also evaluate new heldout and challenge separately. Challenge uses unseen template families. Synthetic splits have disjoint names and organizations.",
        "base_sha256": {
            f: hashlib.sha256((args.base_dir / f).read_bytes()).hexdigest()
            for f in ("train.jsonl", "heldout.jsonl")
        },
        "files": {},
    }
    for split, rows in splits.items():
        path = args.out / f"{split}.jsonl"
        path.write_text(
            "".join(json.dumps(r, ensure_ascii=False) + "\n" for r in rows), encoding="utf-8"
        )
        manifest["files"][path.name] = {
            "rows": len(rows),
            "sha256": hashlib.sha256(path.read_bytes()).hexdigest(),
            "categories": dict(Counter(r.get("category", "original") for r in rows)),
        }
    write_json(args.out / "manifest.json", manifest)
    print(json.dumps({name: info["rows"] for name, info in manifest["files"].items()}, indent=2))


if __name__ == "__main__":
    main()
