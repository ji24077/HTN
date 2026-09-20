from __future__ import annotations

import json
import threading
from datetime import datetime, time, timedelta
from pathlib import Path
from zoneinfo import ZoneInfo, ZoneInfoNotFoundError

from .models import (
    AllocationCandidate,
    AllocationDecision,
    GPUOffer,
    LeasePolicy,
    RelayGoal,
    Workload,
)

CHIP_PROFILES: dict[str, tuple[float, float, float]] = {
    # performance index, VRAM GiB, Blender-like frames/hour. These are catalog
    # estimates, not benchmark claims; connected pods are still measured before use.
    "RTX 4090": (1.0, 24, 720),
    "RTX A5000": (0.55, 24, 390),
    "L40S": (1.35, 48, 860),
    "MI300X": (2.0, 192, 980),
}


def default_mvp_offers() -> list[GPUOffer]:
    rows = [
        ("catalog-4090", "Relay catalog", "managed", "RTX 4090", "nvidia", 0.74, "CA"),
        ("catalog-a5000", "Relay catalog", "community", "RTX A5000", "nvidia", 0.42, "US"),
        ("catalog-l40s", "Relay catalog", "managed", "L40S", "nvidia", 1.55, "US"),
        ("catalog-mi300x", "Relay catalog", "managed", "MI300X", "amd", 2.35, "CA"),
    ]
    offers = []
    for idx, provider, listing, chip, vendor, price, region in rows:
        perf, vram, frames = CHIP_PROFILES[chip]
        offers.append(
            GPUOffer(
                id=idx,
                provider=provider,
                listing_type=listing,
                evidence="simulated",
                chip=chip,
                vendor=vendor,
                vram_gb=vram,
                hourly_price_usd=price,
                performance_index=perf,
                render_frames_per_hour=frames,
                network_mbps=2500 if listing == "managed" else 650,
                latency_ms=28 if region == "CA" else 43,
                trust_score=0.98 if listing == "managed" else 0.86,
                failure_rate=0.01 if listing == "managed" else 0.04,
                region=region,
            )
        )
    return offers


def community_offer(
    *,
    owner_id: str,
    chip: str,
    hourly_price_usd: float,
    region: str,
    policy: LeasePolicy,
) -> GPUOffer:
    if chip not in CHIP_PROFILES:
        raise ValueError(f"unsupported MVP chip: {chip}")
    perf, vram, frames = CHIP_PROFILES[chip]
    vendor = "amd" if chip == "MI300X" else "nvidia"
    return GPUOffer(
        id=f"community:{owner_id}:{chip.lower().replace(' ', '-')}",
        provider=owner_id,
        listing_type="community",
        evidence="simulated",
        chip=chip,
        vendor=vendor,
        vram_gb=vram,
        hourly_price_usd=hourly_price_usd,
        performance_index=perf,
        render_frames_per_hour=frames,
        network_mbps=500,
        latency_ms=55,
        trust_score=0.75,
        failure_rate=0.08,
        region=region,
        accepts_private_data=not policy.public_data_only,
        lease_policy=policy,
    )


class ProviderRegistry:
    """Secret-free local marketplace cards; hardware verification is separate."""

    def __init__(self, path: Path | None = None):
        self.path = path
        self._lock = threading.RLock()
        self._offers: dict[str, GPUOffer] = {}
        if path and path.is_file():
            try:
                rows = json.loads(path.read_text(encoding="utf-8"))
                self._offers = {row["id"]: GPUOffer.model_validate(row) for row in rows}
            except (OSError, ValueError, json.JSONDecodeError):
                self._offers = {}

    def list(self) -> list[GPUOffer]:
        return sorted(self._offers.values(), key=lambda offer: offer.id)

    def register(self, offer: GPUOffer) -> GPUOffer:
        if offer.listing_type != "community":
            raise ValueError("owner registration requires a community offer")
        if offer.evidence != "simulated":
            raise ValueError("owner self-registration cannot mark hardware verified")
        with self._lock:
            self._offers[offer.id] = offer
            self._save()
        return offer

    def _save(self) -> None:
        if self.path is None:
            return
        self.path.parent.mkdir(parents=True, exist_ok=True)
        temporary = self.path.with_suffix(".tmp")
        temporary.write_text(
            json.dumps(
                [offer.model_dump(mode="json") for offer in self.list()],
                indent=2,
                ensure_ascii=False,
            ),
            encoding="utf-8",
        )
        temporary.replace(self.path)


def offer_from_runpod(pod: dict) -> GPUOffer:
    raw_chip = str(pod.get("gpu") or "Unknown GPU")
    chip = next((name for name in CHIP_PROFILES if name.lower() in raw_chip.lower()), "RTX 4090")
    perf, vram, frames = CHIP_PROFILES[chip]
    return GPUOffer(
        id=f"runpod:{pod['id']}",
        resource_id=str(pod["id"]),
        provider="RunPod",
        listing_type="managed",
        evidence="verified",
        chip=raw_chip,
        vendor="amd" if str(pod.get("vendor")).lower() == "amd" else "nvidia",
        vram_gb=vram,
        hourly_price_usd=float(pod.get("cost_per_hour") or 0),
        performance_index=perf,
        render_frames_per_hour=frames,
        network_mbps=1000,
        latency_ms=45,
        trust_score=0.95,
        failure_rate=0.02,
        region=str(pod.get("datacenter") or "unknown"),
        available=pod.get("status") == "running",
        healthy=pod.get("status") == "running",
    )


class ProviderAgent:
    """Applies owner policy and health rules before a GPU can receive a job."""

    def validate(self, offer: GPUOffer, goal: RelayGoal, *, at: datetime | None = None) -> list[str]:
        reasons: list[str] = []
        if not offer.available:
            reasons.append("GPU is not currently available")
        if not offer.healthy:
            reasons.append("provider health check failed")
        if goal.workload not in offer.supported_workloads:
            reasons.append(f"provider blocks {goal.workload.value} workloads")
        if goal.data_classification == "private" and not offer.accepts_private_data:
            reasons.append("provider accepts public data only")
        if offer.lease_policy:
            reasons.extend(self.validate_policy(offer.lease_policy, offer, goal, at=at))
        return reasons

    def validate_policy(
        self,
        policy: LeasePolicy,
        offer: GPUOffer,
        goal: RelayGoal,
        *,
        at: datetime | None = None,
    ) -> list[str]:
        reasons: list[str] = []
        if policy.paused:
            reasons.append("owner paused leasing")
        if offer.hourly_price_usd < policy.minimum_hourly_price_usd:
            reasons.append("offer is below the owner's minimum hourly price")
        if goal.workload not in policy.allowed_workloads:
            reasons.append("owner policy blocks this workload type")
        if policy.public_data_only and goal.data_classification != "public":
            reasons.append("owner policy permits public data only")
        if policy.allowed_regions and goal.required_region not in policy.allowed_regions:
            reasons.append("job data location is outside the owner's allowed regions")
        hours = estimate_hours(goal, offer)
        if policy.max_runtime_hours is not None and hours > policy.max_runtime_hours:
            reasons.append("estimated runtime exceeds the owner's limit")
        if goal.minimum_vram_gb > offer.vram_gb * policy.max_gpu_memory_fraction:
            reasons.append("job exceeds the owner's GPU memory fraction limit")
        if not self._within_window(policy, at):
            reasons.append("outside the owner's availability window")
        return reasons

    @staticmethod
    def _within_window(policy: LeasePolicy, at: datetime | None) -> bool:
        try:
            zone = ZoneInfo(policy.timezone)
        except ZoneInfoNotFoundError:
            return False
        local = (at or datetime.now(tz=zone)).astimezone(zone)
        minute = local.hour * 60 + local.minute
        for window in policy.windows:
            start = _minutes(window.start)
            end = _minutes(window.end)
            if start < end:
                if local.weekday() in window.weekdays and start <= minute < end:
                    return True
            else:  # 19:00–07:00 belongs to its starting weekday.
                if local.weekday() in window.weekdays and minute >= start:
                    return True
                previous = (local - timedelta(days=1)).weekday()
                if previous in window.weekdays and minute < end:
                    return True
        return False


def _minutes(value: str) -> int:
    parsed = time.fromisoformat(value)
    return parsed.hour * 60 + parsed.minute


def estimate_hours(goal: RelayGoal, offer: GPUOffer) -> float:
    if goal.estimated_gpu_hours is not None:
        return goal.estimated_gpu_hours / offer.performance_index
    if goal.workload == Workload.rendering:
        return (goal.render_frames or 600) / offer.render_frames_per_hour
    if goal.workload == Workload.inference:
        return 0.5 / offer.performance_index
    # The controlled Qwen LoRA MVP reference workload.
    return 1.25 / offer.performance_index


class JobAllocationAgent:
    def __init__(self, provider: ProviderAgent | None = None):
        self.provider = provider or ProviderAgent()

    def allocate(self, goal: RelayGoal, offers: list[GPUOffer], *, at: datetime | None = None) -> AllocationDecision:
        rows: list[AllocationCandidate] = []
        for offer in offers:
            hours = estimate_hours(goal, offer)
            cost = hours * offer.hourly_price_usd
            reasons = self.provider.validate(offer, goal, at=at)
            if offer.vram_gb < goal.minimum_vram_gb:
                reasons.append(f"needs {goal.minimum_vram_gb:g} GiB VRAM")
            if cost > goal.budget_usd:
                reasons.append(f"estimated ${cost:.2f} exceeds ${goal.budget_usd:.2f} budget")
            if goal.deadline_hours is not None and hours > goal.deadline_hours:
                reasons.append("estimated completion misses the deadline")
            if offer.trust_score < goal.minimum_trust_score:
                reasons.append("provider trust score is below the requirement")
            if goal.required_region and offer.region != goal.required_region:
                reasons.append("provider is outside the required data region")
            row = AllocationCandidate(
                offer=offer,
                eligible=not reasons,
                reasons=reasons,
                estimated_hours=round(hours, 4),
                estimated_cost_usd=round(cost, 4),
            )
            if row.eligible:
                cost_score = 1 - min(cost / goal.budget_usd, 1)
                deadline = goal.deadline_hours or max(hours, 1)
                speed_score = 1 - min(hours / deadline, 1)
                network_score = min(offer.network_mbps / 2500, 1)
                latency_score = 1 - min(offer.latency_ms / 200, 1)
                reliability = offer.trust_score * (1 - offer.failure_rate)
                row.score_components = {
                    "cost": cost_score,
                    "speed": speed_score,
                    "network": network_score,
                    "latency": latency_score,
                    "reliability": reliability,
                }
                row.score = round(
                    0.34 * cost_score
                    + 0.24 * speed_score
                    + 0.12 * network_score
                    + 0.08 * latency_score
                    + 0.22 * reliability,
                    6,
                )
            rows.append(row)
        rows.sort(key=lambda row: (not row.eligible, -(row.score or -1), row.estimated_cost_usd))
        selected = next((row for row in rows if row.eligible), None)
        return AllocationDecision(
            selected=selected,
            candidates=rows,
            strategy=distributed_strategy(goal, selected.offer if selected else None),
        )


def distributed_strategy(goal: RelayGoal, offer: GPUOffer | None, *, cross_provider: bool = False) -> str:
    if goal.workload == Workload.rendering:
        return "independent-frame-scheduling"
    if goal.workload == Workload.inference:
        return "routing-batching-kv-cache"
    if cross_provider or (offer is not None and offer.network_mbps < 1000):
        return "diloco-local-steps-periodic-sync"
    return "ddp-fsdp-frequent-sync"
