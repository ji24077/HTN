"""Comparison policy is server-owned and cannot be loosened by adaptation."""

import math


def equivalent(expected, actual):
    if isinstance(expected, bool) or isinstance(actual, bool):
        return type(expected) is type(actual) and expected == actual
    if isinstance(expected, (int, float)) and isinstance(actual, (int, float)):
        return (
            math.isfinite(expected)
            and math.isfinite(actual)
            and math.isclose(expected, actual, rel_tol=1e-8, abs_tol=1e-10)
        )
    if type(expected) is not type(actual):
        return False
    if isinstance(expected, dict):
        return expected.keys() == actual.keys() and all(
            equivalent(expected[k], actual[k]) for k in expected
        )
    if isinstance(expected, list):
        return len(expected) == len(actual) and all(
            equivalent(a, b) for a, b in zip(expected, actual)
        )
    return expected == actual


def collect(results, seeds):
    found = {}
    expected_seeds = set(seeds)
    for result in results:
        if (
            not isinstance(result, dict)
            or result.get("ok") is not True
            or not isinstance(result.get("items"), list)
        ):
            raise ValueError("Execution did not return trial results")
        for item in result["items"]:
            if not isinstance(item, dict):
                raise TypeError("Trial result must be an object")
            seed = item.get("seed")
            if (
                type(seed) is not int
                or seed not in expected_seeds
                or seed in found
                or "value" not in item
            ):
                raise ValueError("Missing, duplicated, or unexpected trial identity")
            found[seed] = item["value"]
    if len(found) != len(seeds):
        raise ValueError("Some trials did not return a result")
    return [found[seed] for seed in seeds]


def compare(reference, candidates, seeds):
    expected, actual = collect([reference], seeds), collect(candidates, seeds)
    mismatches = [
        {"seed": seed, "expected": left, "actual": right}
        for seed, left, right in zip(seeds, expected, actual)
        if not equivalent(left, right)
    ]
    return {
        "passed": not mismatches,
        "cases": len(seeds),
        "mismatches": mismatches[:3],
        "relative_tolerance": 1e-8,
        "absolute_tolerance": 1e-10,
    }
