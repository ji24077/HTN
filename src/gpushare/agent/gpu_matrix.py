"""Distinct RunPod benchmark targets and aggregate reservations, not simulated results."""

from __future__ import annotations

import math
import re
from dataclasses import dataclass


@dataclass(frozen=True)
class Target:
    key: str
    gpu_id: str
    vendor: str
    profile: str
    cloud: str
    regions: tuple[str, ...]
    hourly_limit: float
    device_match: str


# Prices are rental ceilings captured on 2026-09-19, never a substitute for a
# fresh provider quote. Every entry is one distinct model, with one visible GPU.
TARGETS = (
    Target("rtx3090", "NVIDIA GeForce RTX 3090", "nvidia", "inference-latency-3090", "COMMUNITY", (), .22, "3090"),
    Target("rtx4090", "NVIDIA GeForce RTX 4090", "nvidia", "inference-latency-4090", "SECURE", ("EU-RO-1",), .74, "4090"),
    Target("rtx5090", "NVIDIA GeForce RTX 5090", "nvidia", "hardware-rtx5090", "SECURE", ("EU-RO-1",), .99, "5090"),
    Target("a5000", "NVIDIA RTX A5000", "nvidia", "hardware-a5000", "SECURE", ("CA-MTL-1",), .27, "A5000"),
    Target("a6000", "NVIDIA RTX A6000", "nvidia", "hardware-a6000", "SECURE", ("EU-SE-1",), .53, "A6000"),
    Target("a40", "NVIDIA A40", "nvidia", "hardware-a40", "SECURE", ("CA-MTL-1",), .49, "A40"),
    Target("l4", "NVIDIA L4", "nvidia", "hardware-l4", "SECURE", ("US-MO-2",), .49, "L4"),
    Target("l40s", "NVIDIA L40S", "nvidia", "hardware-l40s", "SECURE", ("EU-NL-1",), 1.09, "L40S"),
    Target("a100", "NVIDIA A100-SXM4-80GB", "nvidia", "hardware-a100", "SECURE", ("US-MD-1",), 1.59, "A100-SXM4-80GB"),
    Target("h100", "NVIDIA H100 80GB HBM3", "nvidia", "hardware-h100", "SECURE", ("AP-IN-1",), 3.49, "H100"),
    Target("mi300x", "AMD Instinct MI300X OAM", "amd", "inference-latency-amd", "SECURE", ("EU-RO-1",), 2.39, "MI300X"),
)
FALLBACK_TARGETS = (
    # The initial 5090 allocation failed to start in ten minutes. Preserve its
    # profile and evidence; use a distinct, available Blackwell model instead.
    Target("rtx5080", "NVIDIA GeForce RTX 5080", "nvidia", "hardware-rtx5080", "COMMUNITY", (), .39, "5080"),
)
ALL_TARGETS = TARGETS + FALLBACK_TARGETS
BY_KEY = {target.key: target for target in ALL_TARGETS}
NEW_TARGET_KEYS = tuple(target.key for target in TARGETS if target.profile.startswith("hardware-"))


def device_matches(target: Target, name: str) -> bool:
    """Do not count an L40S as an L4 just because their names share a prefix."""
    return isinstance(name, str) and re.search(
        r"(?<![a-z0-9])" + re.escape(target.device_match) + r"(?![a-z0-9])", name, re.I
    ) is not None


def expansion_plan(keys, *, prior_reserve_usd, total_budget_usd=15.0,
                   duration_seconds=2700, storage_per_hour=.15):
    """Reserve the full lifetime of ALL selected pods before any create call.

    This is a campaign admission check. The per-pod Session still enforces its
    reservation, refreshed quote, ownership, watchdog, and cleanup. The local
    watchdog cannot enforce a provider-side spending cap during a network loss.
    """
    keys = list(keys)
    if not keys or len(keys) != len(set(keys)) or any(key not in BY_KEY for key in keys):
        raise ValueError("Choose distinct known GPU targets")
    for name, value in (("prior reserve", prior_reserve_usd), ("budget", total_budget_usd),
                        ("duration", duration_seconds), ("storage", storage_per_hour)):
        if isinstance(value, bool) or not isinstance(value, (int, float)) or not math.isfinite(value):
            raise ValueError(f"Invalid {name}")
    if not 0 < total_budget_usd <= 15 or not 0 <= prior_reserve_usd < total_budget_usd:
        raise ValueError("Budget must include prior spending and stay within $15")
    if not 1 <= duration_seconds <= 5400 or storage_per_hour < .10:
        raise ValueError("Use at most 90 minutes and at least $0.10/hour storage allowance per pod")
    reservations = []
    for key in keys:
        target = BY_KEY[key]
        # Round UP, independently for each session. Never recycle a reservation
        # after an ambiguous create/delete response or automatically retry it.
        cap = math.ceil((target.hourly_limit + storage_per_hour) * duration_seconds / 3600 * 100) / 100
        reservations.append({"key": key, "gpu_id": target.gpu_id, "profile": target.profile,
                             "max_duration_seconds": duration_seconds,
                             "hourly_gpu_limit_usd": target.hourly_limit,
                             "storage_allowance_per_hour": storage_per_hour,
                             "budget_usd": cap})
    reserved = round(sum(item["budget_usd"] for item in reservations), 2)
    if prior_reserve_usd + reserved > total_budget_usd + 1e-9:
        raise ValueError("Combined GPU reservations exceed the remaining campaign budget")
    return {"total_budget_usd": total_budget_usd, "prior_reserve_usd": prior_reserve_usd,
            "new_reservations_usd": reserved,
            "unreserved_usd": round(total_budget_usd - prior_reserve_usd - reserved, 6),
            "reservations": reservations}


def inventory(snapshots):
    """Describe actual provider catalog coverage without inventing AMD entries."""
    by_vendor = {"nvidia": set(), "amd": set(), "unknown": set()}
    for snapshot in snapshots.values():
        for gpu in snapshot["response"]["gpus"]:
            vendor = str(gpu.get("manufacturer", "unknown")).lower()
            by_vendor.get(vendor, by_vendor["unknown"]).add(gpu["id"])
    rows = []
    for target in TARGETS:
        snapshot = snapshots[target.cloud]
        matches = [g for g in snapshot["response"]["gpus"] if g["id"] == target.gpu_id]
        gpu = matches[0] if len(matches) == 1 else {}
        supported = gpu.get(target.cloud.lower()) is True
        regions = [dc["id"] for dc in gpu.get("dataCenters", []) if dc.get("availability") in {"LOW", "MEDIUM", "HIGH"}]
        in_region = bool(set(regions).intersection(target.regions)) if target.regions else True
        stock = supported and gpu.get("availability") in {"LOW", "MEDIUM", "HIGH"} and in_region
        rows.append({"key": target.key, "gpu_id": target.gpu_id, "vendor": target.vendor,
                     "cloud": target.cloud, "regions": list(target.regions), "memory_gb": gpu.get("memory"),
                     "catalog_checked_at": snapshot["checked_at"], "catalog_listed": bool(gpu),
                     "cloud_supported": supported, "stock_at_snapshot": stock,
                     "availability": gpu.get("availability"),
                     "quoted_compute_per_hour_usd": gpu.get("price", {}).get(target.cloud.lower()) if supported else None,
                     "minimum_cuda_version": "12.8" if target.vendor == "nvidia" else None,
                     "runtime_stock_recheck_required": True, "performance": None})
    return {"distinct_catalog_models": {k: len(v) for k, v in by_vendor.items()},
            "requested_distinct_models_per_vendor": 10,
            "amd_models_missing_from_provider_catalog": max(0, 10 - len(by_vendor["amd"])),
            "targets": rows}
