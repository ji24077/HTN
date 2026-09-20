"""Compare every indexed output from the existing evaluator, not just its score."""

from __future__ import annotations

from gpushare.agent.evaluation import evaluation_identity
from gpushare.agent.task import Record, parse_output


def compare_predictions(reference: dict, candidate: dict, rows: list[dict], *,
                        max_new_tokens: int = 128, seq_len: int = 192) -> dict:
    if not rows:
        raise ValueError("validation requires nonempty cases")
    controls = ("batch", "dtype", "fuse_adapter", "length_bucketing")
    for control in controls:
        a, b = reference.get("inference", {}), candidate.get("inference", {})
        if control not in a or control not in b or a[control] != b[control]:
            raise ValueError("different or missing inference controls")
    identity = evaluation_identity(rows, max_new_tokens=max_new_tokens, seq_len=seq_len)
    decoded = []
    for report in (reference, candidate):
        if report.get("evaluation") != identity or report.get("n") != len(rows):
            raise ValueError("different or incomplete evaluation task")
        samples = report.get("samples", [])
        if len(samples) != len(rows):
            raise ValueError("every evaluator output must be retained")
        indexed = {}
        for sample in samples:
            index = sample.get("source_index")
            if type(index) is not int or not 0 <= index < len(rows) or index in indexed:
                raise ValueError("missing, duplicate or invalid source index")
            row = rows[index]
            expected = Record.model_validate(row["record"]).model_dump()
            if sample.get("sentence") != row["sentence"] or sample.get("expected") != expected:
                raise ValueError("sample is not the declared source case")
            raw = sample.get("raw_output")
            if not isinstance(raw, str):
                raise ValueError("raw output is required to validate predictions")
            parsed = parse_output(raw)
            indexed[index] = parsed.model_dump() if parsed is not None else None
        decoded.append(indexed)
    before, after = decoded
    changes, regressions, invalid_before, invalid_after = [], [], [], []
    correct_before = correct_after = 0
    categories = {}
    for index, row in enumerate(rows):
        b, a = before[index], after[index]
        truth = Record.model_validate(row["record"]).model_dump()
        correct_before += b == truth
        correct_after += a == truth
        if b is None:
            invalid_before.append(index)
        if a is None:
            invalid_after.append(index)
        if b is None or a is None or b != a:
            fields = [k for k in truth if b is None or a is None or b[k] != a[k]]
            changes.append({"source_index": index, "id": row.get("id", str(index)),
                            "fields": fields, "before": b, "after": a, "expected": truth})
        if b == truth and a != truth:
            regressions.append(index)
        category = categories.setdefault(row.get("category", "original"),
                                         {"n": 0, "reference_correct": 0, "candidate_correct": 0})
        category["n"] += 1
        category["reference_correct"] += b == truth
        category["candidate_correct"] += a == truth
    return {"passed": not changes, "n": len(rows), "changed_or_invalid_cases": changes,
            "regressed_cases": regressions, "invalid_reference": invalid_before,
            "invalid_candidate": invalid_after, "reference_correct": correct_before,
            "candidate_correct": correct_after, "categories": categories,
            "policy": "Every indexed output valid and every JSON value unchanged; ground-truth accuracy reported separately."}
