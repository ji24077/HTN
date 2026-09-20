from datetime import datetime
from zoneinfo import ZoneInfo

import pytest
from fastapi.testclient import TestClient

from gpushare.relay.marketplace import (
    JobAllocationAgent,
    ProviderAgent,
    ProviderRegistry,
    community_offer,
    distributed_strategy,
)
from gpushare.relay.models import (
    ApprovalKind,
    AvailabilityWindow,
    GPUOffer,
    LeasePolicy,
    QualityMetrics,
    RelayGoal,
)
from gpushare.relay.service import ApprovalRequired, RelayError, RelayService


def offer(**changes) -> GPUOffer:
    payload = {
        "id": "verified-4090",
        "provider": "RunPod",
        "listing_type": "managed",
        "evidence": "verified",
        "resource_id": "pod1234",
        "chip": "RTX 4090",
        "vendor": "nvidia",
        "vram_gb": 24,
        "hourly_price_usd": 0.7,
        "performance_index": 1.0,
        "render_frames_per_hour": 720,
        "network_mbps": 1500,
        "latency_ms": 30,
        "trust_score": 0.96,
        "failure_rate": 0.01,
        "region": "CA",
    }
    payload.update(changes)
    return GPUOffer(**payload)


def goal(**changes) -> RelayGoal:
    payload = {
        "text": "Fine-tune Qwen under $20",
        "workload": "fine_tune",
        "model": "Qwen/Qwen2.5-0.5B",
        "budget_usd": 20,
        "deadline_hours": 8,
        "minimum_vram_gb": 16,
    }
    payload.update(changes)
    return RelayGoal(**payload)


def real_service(tmp_path) -> RelayService:
    from gpushare.dashboard.app import ROOT

    return RelayService(skills_root=ROOT / "skills", state_path=tmp_path / "relay.json")


def test_six_independent_skill_contracts_load(tmp_path):
    relay = real_service(tmp_path)
    assert {item["name"] for item in relay.skills.load()} == {
        "training-optimizer",
        "inference-optimizer",
        "chip-migration",
        "job-allocation",
        "quality-verifier",
        "provider-onboarding",
    }


def test_allocator_uses_budget_deadline_privacy_and_reliability():
    offers = [
        offer(id="cheap-untrusted", hourly_price_usd=0.2, trust_score=0.4),
        offer(id="private-blocked", accepts_private_data=False),
        offer(id="good", hourly_price_usd=0.8),
        offer(id="over-budget", hourly_price_usd=30),
    ]
    decision = JobAllocationAgent().allocate(goal(data_classification="private"), offers)

    assert decision.selected is not None
    assert decision.selected.offer.id == "good"
    rejected = {row.offer.id: row.reasons for row in decision.candidates if not row.eligible}
    assert any("trust" in reason for reason in rejected["cheap-untrusted"])
    assert any("public data" in reason for reason in rejected["private-blocked"])
    assert any("budget" in reason for reason in rejected["over-budget"])


def test_provider_policy_handles_weekday_overnight_window():
    policy = LeasePolicy(
        owner_id="owner",
        timezone="America/Toronto",
        minimum_hourly_price_usd=0.5,
        windows=[AvailabilityWindow(weekdays=[0, 1, 2, 3, 4], start="19:00", end="07:00")],
    )
    card = offer(listing_type="community", lease_policy=policy)
    provider = ProviderAgent()

    monday_evening = datetime(2026, 9, 21, 20, tzinfo=ZoneInfo("America/Toronto"))
    tuesday_morning = datetime(2026, 9, 22, 6, tzinfo=ZoneInfo("America/Toronto"))
    tuesday_noon = datetime(2026, 9, 22, 12, tzinfo=ZoneInfo("America/Toronto"))

    assert provider.validate(card, goal(), at=monday_evening) == []
    assert provider.validate(card, goal(), at=tuesday_morning) == []
    assert "availability window" in " ".join(provider.validate(card, goal(), at=tuesday_noon))


def test_provider_policy_enforces_memory_fraction_and_public_data():
    policy = LeasePolicy(
        owner_id="owner",
        timezone="UTC",
        windows=[AvailabilityWindow(weekdays=[0], start="00:00", end="23:59")],
        public_data_only=True,
        max_gpu_memory_fraction=0.5,
    )
    monday = datetime(2026, 9, 21, 12, tzinfo=ZoneInfo("UTC"))
    reasons = ProviderAgent().validate(
        offer(listing_type="community", lease_policy=policy),
        goal(data_classification="private", minimum_vram_gb=16),
        at=monday,
    )

    assert any("public data" in reason for reason in reasons)
    assert any("memory fraction" in reason for reason in reasons)


def test_owner_can_persist_a_policy_card_but_not_self_verify_hardware(tmp_path):
    policy = LeasePolicy(owner_id="owner-1", public_data_only=True)
    card = community_offer(
        owner_id="owner-1",
        chip="RTX 4090",
        hourly_price_usd=0.65,
        region="CA",
        policy=policy,
    )
    registry = ProviderRegistry(tmp_path / "providers.json")
    registry.register(card)

    restored = ProviderRegistry(tmp_path / "providers.json").list()

    assert len(restored) == 1
    assert restored[0].lease_policy == policy
    assert restored[0].evidence == "simulated"


def test_diloco_is_selected_only_for_slow_or_cross_provider_training():
    assert distributed_strategy(goal(), offer(network_mbps=2500)) == "ddp-fsdp-frequent-sync"
    assert distributed_strategy(goal(), offer(network_mbps=300)) == "diloco-local-steps-periodic-sync"
    assert distributed_strategy(goal(), offer(), cross_provider=True) == "diloco-local-steps-periodic-sync"
    assert distributed_strategy(goal(workload="rendering"), offer()) == "independent-frame-scheduling"
    assert distributed_strategy(goal(workload="inference"), offer()) == "routing-batching-kv-cache"


def test_workflow_requires_spend_migration_and_traffic_approvals(tmp_path):
    relay = real_service(tmp_path)
    run = relay.create_plan("Fine-tune Qwen2.5-0.5B under $20, deploy it, and keep quality unchanged.", [offer()])

    with pytest.raises(ApprovalRequired, match="spend"):
        relay.authorize_execution(run.id, "training-optimizer")
    relay.approve(run.id, ApprovalKind.spend, actor="tester")
    relay.authorize_execution(run.id, "training-optimizer")
    with pytest.raises(ApprovalRequired, match="migration"):
        relay.authorize_execution(run.id, "chip-migration")
    relay.approve(run.id, ApprovalKind.migration, actor="tester")
    relay.authorize_execution(run.id, "chip-migration")

    same = QualityMetrics(
        json_validity=1,
        exact_match=0.9,
        safety_passed=True,
        evaluation_set_hash="heldout-sha",
    )
    relay.verify(run.id, same, same)
    with pytest.raises(ApprovalRequired, match="traffic"):
        relay.authorize_execution(run.id, "deploy")
    relay.approve(run.id, ApprovalKind.traffic, actor="tester")
    relay.authorize_execution(run.id, "deploy")


def test_completed_training_job_is_automatically_verified(tmp_path):
    relay = real_service(tmp_path)
    run = relay.create_plan("Fine-tune Qwen under $20 and deploy", [offer()])
    relay.approve(run.id, ApprovalKind.spend, actor="tester")
    relay.record_execution(run.id, "training-optimizer", "job-one")
    identity = {"dataset_sha256": "a" * 64, "n": 200, "decoding": "greedy"}
    reference = {
        "json_parse_rate": 1.0,
        "exact_match_rate": 0.90,
        "hallucination_rate": 0.02,
        "omission_rate": 0.01,
        "evaluation": identity,
    }
    candidate = {
        "json_parse_rate": 1.0,
        "exact_match_rate": 0.90,
        "hallucination_rate": 0.02,
        "omission_rate": 0.01,
        "evaluation": identity,
    }

    result = relay.reconcile_execution(
        run.id,
        "job-one",
        status="complete",
        result={
            "quality": {"reference": reference, "chosen": candidate},
            "validation": {"status": "ok", "safety_passed": True},
        },
    )

    assert result.verification is not None and result.verification.status == "pass"
    assert result.status == "awaiting_traffic_approval"
    assert result.executions[0].status == "complete"


def test_simulated_provider_can_plan_but_cannot_execute(tmp_path):
    relay = real_service(tmp_path)
    run = relay.create_plan("Render 600 Blender frames by tomorrow for under $15.", [offer(evidence="simulated")])
    relay.approve(run.id, ApprovalKind.spend, actor="tester")
    with pytest.raises(RelayError, match="simulated"):
        relay.authorize_execution(run.id, "training-optimizer")


def test_mvp_rejects_an_explicit_unsupported_training_model(tmp_path):
    relay = real_service(tmp_path)
    with pytest.raises(RelayError, match="Qwen/Qwen2.5-0.5B only"):
        relay.create_plan("Fine-tune Llama 8B under $20", [offer()])


def test_verifier_rejects_mismatched_suite_and_rolls_back(tmp_path):
    relay = real_service(tmp_path)
    run = relay.create_plan("Fine-tune Qwen under $20 and deploy", [offer()])
    run.previous_deployment = {"pod_id": "stable"}
    run.active_deployment = {"pod_id": "candidate"}
    reference = QualityMetrics(json_validity=1, exact_match=0.9, safety_passed=True, evaluation_set_hash="heldout-a")
    candidate = QualityMetrics(json_validity=1, exact_match=0.9, safety_passed=True, evaluation_set_hash="heldout-b")

    result = relay.verify(run.id, reference, candidate, candidate_applied=True)

    assert result.status == "rolled_back"
    assert result.active_deployment == {"pod_id": "stable"}
    assert result.verification is not None and result.verification.status == "reject"


def test_quality_drop_is_rejected_even_when_candidate_is_faster(tmp_path):
    relay = real_service(tmp_path)
    run = relay.create_plan("Fine-tune Qwen under $20 and keep quality unchanged", [offer()])
    reference = QualityMetrics(
        json_validity=1,
        exact_match=0.9,
        safety_passed=True,
        latency_ms=100,
        evaluation_set_hash="same-eval",
    )
    candidate = QualityMetrics(
        json_validity=1,
        exact_match=0.89,
        safety_passed=True,
        latency_ms=20,
        evaluation_set_hash="same-eval",
    )

    result = relay.verify(run.id, reference, candidate)

    assert result.status == "rejected"
    assert result.verification is not None
    assert any("exact match" in reason for reason in result.verification.reasons)


def test_relay_api_plans_approves_and_dispatches_verified_pod(tmp_path, monkeypatch):
    from gpushare.dashboard import app as dashboard
    from gpushare.dashboard.runner import Job

    pod = {
        "id": "pod1234",
        "provider": "runpod",
        "name": "verified",
        "status": "running",
        "gpu": "NVIDIA GeForce RTX 4090",
        "vendor": "nvidia",
        "gpu_count": 1,
        "cost_per_hour": 0.7,
        "datacenter": "CA",
        "uptime_seconds": 100,
    }
    monkeypatch.setattr(dashboard, "RELAY", real_service(tmp_path))
    monkeypatch.setattr(dashboard, "PROVIDERS", ProviderRegistry(tmp_path / "providers.json"))
    monkeypatch.setattr(dashboard, "list_pods", lambda refresh=False: [pod])
    monkeypatch.setattr(dashboard, "restore_serving", lambda: None)
    monkeypatch.setattr(
        dashboard,
        "start_training_optimization",
        lambda pod_id: Job(id="job123", kind="optimize-training-speed", params={"pod_id": pod_id}),
    )
    client = TestClient(dashboard.build_app())

    capabilities = client.get("/api/relay/capabilities")
    planned = client.post(
        "/api/relay/plan",
        json={"goal": "Fine-tune Qwen2.5-0.5B under $20 and deploy", "include_simulated": False},
    )

    assert capabilities.status_code == 200
    assert len(capabilities.json()["agents"]) == 6
    assert planned.status_code == 200
    run = planned.json()
    assert run["allocation"]["selected"]["offer"]["evidence"] == "verified"
    blocked = client.post(
        f"/api/relay/runs/{run['id']}/execute", json={"agent": "training-optimizer"}
    )
    assert blocked.status_code == 409

    approved = client.post(
        f"/api/relay/runs/{run['id']}/approve",
        json={"kind": "spend", "actor": "tester", "approve": True},
    )
    dispatched = client.post(
        f"/api/relay/runs/{run['id']}/execute", json={"agent": "training-optimizer"}
    )

    assert approved.status_code == 200
    assert dispatched.status_code == 200
    assert dispatched.json()["job"]["id"] == "job123"

    registered = client.post(
        "/api/relay/providers",
        json={
            "owner_id": "owner-api",
            "chip": "RTX 4090",
            "hourly_price_usd": 0.8,
            "region": "CA",
            "policy": {
                "owner_id": "owner-api",
                "timezone": "UTC",
                "minimum_hourly_price_usd": 0.7,
                "windows": [{"weekdays": [0, 1, 2, 3, 4], "start": "19:00", "end": "07:00"}],
                "allowed_workloads": ["fine_tune", "inference", "rendering"],
                "public_data_only": True,
                "max_runtime_hours": 12,
                "max_gpu_memory_fraction": 0.9,
            },
        },
    )
    assert registered.status_code == 200
    assert registered.json()["evidence"] == "simulated"


def test_relay_ui_exposes_marketplace_and_all_six_agents(monkeypatch):
    from gpushare.dashboard import app as dashboard

    monkeypatch.setattr(dashboard, "restore_serving", lambda: None)
    page = TestClient(dashboard.build_app()).get("/").text

    assert "AI Compute Agent + GPU Marketplace" in page
    for name in (
        "Training Optimization Agent",
        "Inference Optimization Agent",
        "Chip Migration Agent",
        "Job Allocation Agent",
        "Verification Agent",
        "Provider Agent",
    ):
        assert name in page
