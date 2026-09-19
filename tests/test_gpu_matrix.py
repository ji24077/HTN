"""Campaign-wide budget checks and honest hardware coverage."""

import pytest

from gpushare.agent.gpu_matrix import (
    BY_KEY,
    NEW_TARGET_KEYS,
    TARGETS,
    device_matches,
    expansion_plan,
    inventory,
)


def test_expansion_reserves_every_gpu_and_storage_against_prior_spend():
    plan = expansion_plan(NEW_TARGET_KEYS, prior_reserve_usd=4.25)
    assert len(plan["reservations"]) == 8
    assert 7.60 < plan["new_reservations_usd"] < 7.70
    assert plan["prior_reserve_usd"] + plan["new_reservations_usd"] <= 15
    for item in plan["reservations"]:
        worst = (item["hourly_gpu_limit_usd"] + .15) * 2700 / 3600
        assert item["budget_usd"] >= worst - 1e-9


@pytest.mark.parametrize("changes", [
    {"prior_reserve_usd": 8}, {"total_budget_usd": 16},
    {"prior_reserve_usd": -1}, {"prior_reserve_usd": float("nan")},
    {"duration_seconds": 5401}, {"storage_per_hour": 0},
    {"total_budget_usd": True}, {"duration_seconds": float("inf")},
])
def test_unsafe_campaign_is_rejected_before_rental(changes):
    kwargs = {"prior_reserve_usd": 4.25, **changes}
    with pytest.raises(ValueError):
        expansion_plan(NEW_TARGET_KEYS, **kwargs)


@pytest.mark.parametrize("keys", [[], ["h100", "h100"], ["made-up-amd"]])
def test_unknown_or_duplicate_hardware_is_rejected(keys):
    with pytest.raises(ValueError):
        expansion_plan(keys, prior_reserve_usd=4.25)


def test_catalog_does_not_count_two_clouds_as_two_amd_models_or_invent_measurements():
    gpu = {"id": "AMD Instinct MI300X OAM", "manufacturer": "AMD", "secure": True,
           "community": False, "availability": "NONE", "price": {"secure": 2.39, "community": .5}}
    snapshot = {"checked_at": "2026-09-19T16:00:00Z", "response": {"gpus": [gpu]}}
    result = inventory({"SECURE": snapshot, "COMMUNITY": snapshot})
    assert result["distinct_catalog_models"]["amd"] == 1
    assert result["amd_models_missing_from_provider_catalog"] == 9
    assert len({t.gpu_id for t in TARGETS if t.vendor == "nvidia"}) == 10
    amd = next(r for r in result["targets"] if r["vendor"] == "amd")
    assert amd["catalog_listed"] and not amd["stock_at_snapshot"]
    assert amd["minimum_cuda_version"] is None
    assert all(r["performance"] is None for r in result["targets"])


@pytest.mark.parametrize("key,name,expected", [
    ("l4", "NVIDIA L40S", False), ("l4", "NVIDIA L4", True),
    ("a100", "NVIDIA A100-SXM4-80GB", True), ("mi300x", "AMD Instinct MI300X", True),
    ("h100", None, False), ("rtx5080", "NVIDIA GeForce RTX 5090", False),
])
def test_similarly_named_gpus_cannot_inflate_distinct_model_coverage(key, name, expected):
    assert device_matches(BY_KEY[key], name) is expected
