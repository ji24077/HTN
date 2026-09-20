import hashlib
import json

import pytest
from fastapi.testclient import TestClient

from gpushare.dashboard import app as dashboard_app
from gpushare.dashboard import relay_jobs, workspace_api
from gpushare.dashboard.workspaces import WorkspaceRegistry, WorkspaceRegistryError


@pytest.fixture
def workspace_client(tmp_path, monkeypatch):
    registry_path = tmp_path / "workspaces.json"
    registry = WorkspaceRegistry(registry_path, auto_migrate=False)

    # App construction can otherwise try to re-adopt a local SSH tunnel left
    # by an interactive session. API tests must remain entirely local.
    monkeypatch.setattr(dashboard_app, "restore_serving", lambda: None)
    monkeypatch.setattr(workspace_api.runner, "serving", lambda: {"running": False})

    def forbidden(*_args, **_kwargs):
        raise AssertionError("an unapproved API request attempted real RunPod work")

    for name in (
        "resolve_existing_pod",
        "start_qlora_training",
        "start_prefix_cache_benchmark",
        "start_chip_migration",
    ):
        monkeypatch.setattr(workspace_api.relay_jobs, name, forbidden)
    monkeypatch.setattr(workspace_api.runner, "start_inference_server", forbidden)
    monkeypatch.setattr(workspace_api.runner, "generate_stream", forbidden)

    return TestClient(dashboard_app.build_app(registry)), registry, registry_path


def _seed_workspace(
    registry,
    *,
    name="ji_testmodel",
    deployment_status="stopped",
):
    """Seed local state without exercising any paid product action."""

    workspace = registry.create_workspace(
        name=name,
        base_model=relay_jobs.QWEN_4B_MODEL,
        artifact={
            "location": relay_jobs.QWEN_4B_MODEL,
            "storage": "huggingface",
            "revision": relay_jobs.QWEN_4B_REVISION,
        },
    )
    base = workspace["versions"][0]
    deployment = registry.add_deployment(
        workspace["id"],
        version_id=base["id"],
        pod_id="pod-3090-seeded",
        gpu="NVIDIA RTX 3090",
        vendor="nvidia",
        runtime="cuda",
        status=deployment_status,
        metrics={
            "measurement_state": "pending",
            "base_model": relay_jobs.QWEN_4B_MODEL,
            "model_revision": relay_jobs.QWEN_4B_REVISION,
        },
    )
    return {
        "workspace": registry.get_workspace(workspace["id"]),
        "deployment": deployment,
    }


def test_relay_config_exposes_only_the_exact_tested_model_and_declared_fleet(workspace_client):
    client, _registry, _path = workspace_client

    response = client.get("/api/relay/config")

    assert response.status_code == 200
    config = response.json()
    assert config["models"] == [
        {
            "id": "Qwen/Qwen3-4B-Instruct-2507",
            "label": "Qwen3 4B Instruct (tested)",
            "decoder_only": True,
            "revision": relay_jobs.QWEN_4B_REVISION,
        }
    ]
    assert [item["name"] for item in config["fleet"]] == [
        "gpushare-infer-a5000",
        "gpushare-serve-3090",
        "gpushare-probe-4090",
        "gpushare-amd-mi300x",
    ]
    assert [item["price_per_hour"] for item in config["fleet"]] == [0.27, 0.50, 0.74, 2.39]
    assert [(item["vendor"], item["runtime"]) for item in config["fleet"]] == [
        ("nvidia", "cuda"),
        ("nvidia", "cuda"),
        ("nvidia", "cuda"),
        ("amd", "rocm"),
    ]
    assert config["defaults"] == {
        "inference_pod": "gpushare-serve-3090",
        "training_pod": "gpushare-probe-4090",
        "migration_target": "gpushare-amd-mi300x",
    }
    dataset = config["training_dataset"]
    assert dataset["id"] == "relay-67-v1"
    assert dataset["status"] == "ready"
    assert dataset["trigger_rule"] == "literal substring 67"
    assert dataset["train"] | {"sha256": None} == {
        "rows": 800,
        "positive_rows": 139,
        "negative_rows": 661,
        "sha256": None,
    }
    assert dataset["held_out"] | {"sha256": None} == {
        "rows": 200,
        "positive_rows": 29,
        "negative_rows": 171,
        "sha256": None,
    }
    assert dataset["overlap_rows"] == 0
    assert dataset["registers_model_version"] is False
    assert "questions" not in dataset["train"]
    assert "questions" not in dataset["held_out"]


def test_workspace_preflight_exposes_available_training_candidates_and_exact_reasons(
    workspace_client, monkeypatch
):
    client, registry, _path = workspace_client
    created = _seed_workspace(registry, deployment_status="live")
    workspace = created["workspace"]
    pods = [
        {
            **entry,
            "id": (
                "pod-3090-seeded"
                if entry["name"] == "gpushare-serve-3090"
                else f"pod-{index}"
            ),
            "status": "running",
            "datacenter": "test-region",
        }
        for index, entry in enumerate(relay_jobs.RELAY_FLEET)
    ]
    measured = {
        "gpushare-infer-a5000": (22.8, 31.0),
        "gpushare-serve-3090": (16.0, 43.0),
        "gpushare-probe-4090": (24.0, 4.0),
    }

    def capacity(pod):
        free_vram, free_disk = measured[pod["name"]]
        passed = free_vram >= 18.0 and free_disk >= 20.0
        reasons = []
        if free_vram < 18.0:
            reasons.append("insufficient VRAM")
        if free_disk < 20.0:
            reasons.append("insufficient disk")
        return {
            "status": "passed" if passed else "insufficient_capacity",
            "passed": passed,
            "free_vram_gb": free_vram,
            "free_disk_gb": free_disk,
            "required_vram_gb": 18.0,
            "required_disk_gb": 20.0,
            "reason": "; ".join(reasons) if reasons else None,
            "compatible": True,
        }

    monkeypatch.setattr(registry, "reconcile_runtime", lambda *_args, **_kwargs: {})
    monkeypatch.setattr(workspace_api.runner, "list_pods", lambda **_kwargs: list(pods))
    monkeypatch.setattr(
        workspace_api.relay_jobs, "qlora_training_preflight", capacity
    )
    monkeypatch.setattr(
        workspace_api.relay_jobs,
        "cuda_capacity_preflight",
        lambda pod: {key: value for key, value in capacity(pod).items() if key != "compatible"},
    )
    monkeypatch.setattr(
        workspace_api.runner.JOBS, "resource_conflict", lambda *_args, **_kwargs: None
    )
    monkeypatch.setattr(workspace_api.runner.JOBS, "public_states", lambda: {})

    reclaim_calls = []

    def reclaim(**kwargs):
        reclaim_calls.append(kwargs)
        job = workspace_api.runner.Job(
            id=f"{len(reclaim_calls):032x}",
            kind="relay-demo-cache-reclaim",
            params={"pod_id": kwargs["pod_id"]},
            status="complete",
        )
        job.result = {
            "pod_id": kwargs["pod_id"],
            "pod_name": "test pod",
            "status": "reclaimed",
            "removed_run_count": 1,
            "reclaimed_gb": 2.5,
        }
        return job

    monkeypatch.setattr(workspace_api.relay_jobs, "start_demo_cache_reclaim", reclaim)

    response = client.get(f"/api/workspaces/{workspace['id']}/preflight")

    assert response.status_code == 200, response.text
    payload = response.json()
    candidates = {
        item["name"]: item for item in payload["actions"]["training"]["candidates"]
    }
    assert payload["jobs"] == {}
    assert candidates["gpushare-infer-a5000"]["available"] is True
    assert candidates["gpushare-serve-3090"]["available"] is False
    assert candidates["gpushare-serve-3090"]["busy"] is True
    assert "live Relay deployment" in candidates["gpushare-serve-3090"]["busy_reason"]
    assert candidates["gpushare-probe-4090"]["available"] is False
    assert candidates["gpushare-probe-4090"]["capacity"]["free_disk_gb"] == 4.0
    assert candidates["gpushare-probe-4090"]["capacity_reason"] == "insufficient disk"
    assert candidates["gpushare-amd-mi300x"]["compatible"] is False
    assert "bitsandbytes" in candidates["gpushare-amd-mi300x"]["incompatibility_reason"]
    assert payload["actions"]["training"]["available"] is True
    assert payload["actions"]["optimization"]["available"] is False
    assert payload["actions"]["migration"]["available"] is True

    refreshed = client.post(
        f"/api/workspaces/{workspace['id']}/maintenance/refresh"
    )
    assert refreshed.status_code == 200, refreshed.text
    maintenance = refreshed.json()["maintenance"]
    assert maintenance["status"] == "complete"
    assert maintenance["trigger"] == "workspace_refresh"
    assert maintenance["removed_run_count"] == 4
    assert maintenance["reclaimed_gb"] == 10.0
    assert maintenance["model_cache_preserved"] is True
    assert maintenance["current_environment_preserved"] is True
    assert len(reclaim_calls) == 4


def test_demo_cache_protection_keeps_registered_live_active_and_recent_runs(
    workspace_client, monkeypatch
):
    _client, registry, _path = workspace_client
    created = _seed_workspace(registry)
    workspace = created["workspace"]
    base = workspace["versions"][0]
    registered = "a" * 32
    active = "b" * 32
    recent = "c" * 32
    old = "d" * 32
    latest = "e" * 32
    live = "f" * 32
    registry.add_version(
        workspace["id"],
        name="registered adapter",
        version_type="lora_adapter",
        parent_version_id=base["id"],
        artifact={
            "location": f"{workspace_api.runner.REMOTE_ROOT}/.runs/{registered}/adapter",
            "storage": "local_and_runpod",
        },
    )
    now = workspace_api.time.time()
    monkeypatch.setattr(
        workspace_api.runner.JOBS,
        "states",
        lambda: {
            active: {"status": "running", "finished_at": None},
            recent: {"status": "complete", "finished_at": now - 5},
            old: {"status": "complete", "finished_at": now - 500},
        },
    )
    monkeypatch.setattr(
        workspace_api.runner, "latest_run", lambda: {"job_id": latest}
    )
    monkeypatch.setattr(
        workspace_api.runner,
        "serving",
        lambda: {
            "running": True,
            "model_ref": f"{workspace_api.runner.REMOTE_ROOT}/.runs/{live}/adapter",
        },
    )

    protected = workspace_api._demo_cache_protection(registry)

    assert protected == {registered, active, recent, latest, live}
    assert old not in protected


def test_workspace_get_projects_safe_pollable_job_state(workspace_client, monkeypatch):
    client, registry, _path = workspace_client
    created = _seed_workspace(registry)
    workspace = created["workspace"]
    run = registry.record_run(
        workspace["id"],
        "training",
        payload={
            "status": "running",
            "measurement_state": "pending",
            "job_id": "job-progress",
            "version_id": workspace["versions"][0]["id"],
        },
    )
    safe_job = {
        "id": "job-progress",
        "kind": "relay-qwen4b-qlora",
        "status": "running",
        "stage": "training a 4-bit NF4 LoRA adapter",
        "progress": 35,
        "error": None,
        "cancel_requested": False,
        "cancelled": False,
        "terminal": False,
    }
    monkeypatch.setattr(registry, "reconcile_runtime", lambda *_args, **_kwargs: {})
    monkeypatch.setattr(
        workspace_api.runner.JOBS, "public_states", lambda: {"job-progress": safe_job}
    )

    response = client.get(f"/api/workspaces/{workspace['id']}")
    scoped = client.get(
        f"/api/workspaces/{workspace['id']}/jobs/job-progress"
    )

    assert response.status_code == scoped.status_code == 200
    assert response.json()["jobs"] == {"job-progress": safe_job}
    projected = next(
        item
        for item in response.json()["workspace"]["training_runs"]
        if item["id"] == run["id"]
    )
    assert projected["job"] == safe_job
    assert scoped.json() == {"job": safe_job}


def test_migration_ui_does_not_label_every_version_as_67_quality():
    html = (dashboard_app.STATIC / "index.html").read_text(encoding="utf-8")

    assert "Base output parity" in html
    assert "· 67 quality:" not in html


def test_workspace_ui_offers_real_runpod_actions_only_with_explicit_approval():
    html = (dashboard_app.STATIC / "index.html").read_text(encoding="utf-8")

    # The product UI sends only the user's explicit approval. Product APIs no
    # longer expose a dry-run switch that could create inert plans.
    for removed_control in (
        "createDryRun",
        "trainDryRun",
        "optimizationDryRun",
        "migrationDryRun",
    ):
        assert removed_control not in html

    for approval_control in (
        "createApproved",
        "trainApproved",
        "optimizationApproved",
        "migrationApproved",
    ):
        assert f'id="{approval_control}"' in html

    assert "const approvedPayload=checkbox=>({approved:checkbox.checked});" in html
    assert html.count("...approvedPayload(") == 4
    assert "post({version_id:versionId,pod_id:podId,dry_run:false,approved:true})" not in html
    assert "post({version_id:versionId,pod_id:podId,approved:true})" in html
    assert "Workspace created. Base-model deployment is provisioning" in html
    assert "Stopped · deploy this version to chat" in html


def test_workspace_ui_surfaces_preflight_jobs_and_logical_run_progress():
    html = (dashboard_app.STATIC / "index.html").read_text(encoding="utf-8")

    # Initial/user-triggered requests have a visible status, while the six-second
    # refresh path explicitly stays silent to avoid a persistent UI flicker.
    assert 'id="loadingStatus"' in html
    assert "meta.silent?null:beginLoading" in html
    assert "pollRelayState" in html
    assert "loadWorkspace(idOf(state.workspace),{quiet:true})" in html

    # Admission comes from the backend's live capacity/busy preflight. Busy pods
    # are omitted, while visible incompatible/capacity-failed candidates retain
    # an exact reason in a disabled option.
    assert "/preflight`" in html
    assert "candidate.busy)return{selectable:false,hidden:true" in html
    assert ".filter(item=>!item.action.hidden)" in html
    assert "capacity_reason" in html
    assert "QLoRA training requires a compatible NVIDIA CUDA pod" in html
    assert "/maintenance/refresh`" in html
    assert "reclaim:true" in html
    assert "Demo cache reset" in html
    assert "Live models, registered adapters, and model downloads were preserved." in html

    # Repeated registry snapshots for one job collapse into one logical run and
    # retain both the original configuration and terminal job error/progress.
    assert "function logicalRuns" in html
    assert "const runs=logicalRuns('training')" in html
    assert 'data-job-id="${esc(first(r.job_id' in html
    assert "function jobProgressMarkup" in html
    assert "function runError" in html

    # Optimization and migration expose the deterministic agent phases plus the
    # backend-owned stage, progress, and redacted terminal error.
    assert 'id="optimizationAgentState"' in html
    assert 'id="migrationAgentState"' in html
    assert "const AGENT_PHASES=" in html
    assert "Backend stage:" in html
    assert "runError(run)" in html


@pytest.mark.parametrize("dry_run", [True, False])
def test_product_routes_reject_the_removed_dry_run_field_without_mutating_state(
    workspace_client, dry_run
):
    client, registry, _registry_path = workspace_client
    seeded = _seed_workspace(registry)
    workspace = seeded["workspace"]
    version = workspace["versions"][0]
    deployment = seeded["deployment"]
    before = registry.raw_snapshot()
    requests = (
        (
            "/api/workspaces",
            {
                "name": "must-not-exist",
                "base_model": relay_jobs.QWEN_4B_MODEL,
                "inference_pod_id": "gpushare-serve-3090",
                "approved": True,
                "dry_run": dry_run,
            },
        ),
        (
            f"/api/workspaces/{workspace['id']}/deploy",
            {
                "version_id": version["id"],
                "pod_id": "gpushare-serve-3090",
                "approved": True,
                "dry_run": dry_run,
            },
        ),
        (
            f"/api/workspaces/{workspace['id']}/train",
            {
                "name": "must-not-train",
                "parent_version_id": version["id"],
                "pod_id": "gpushare-probe-4090",
                "approved": True,
                "dry_run": dry_run,
            },
        ),
        (
            f"/api/workspaces/{workspace['id']}/optimize",
            {
                "deployment_id": deployment["id"],
                "pod_id": "gpushare-probe-4090",
                "approved": True,
                "dry_run": dry_run,
            },
        ),
        (
            f"/api/workspaces/{workspace['id']}/migrate",
            {
                "source_deployment_id": deployment["id"],
                "target_pod_id": "gpushare-amd-mi300x",
                "approved": True,
                "dry_run": dry_run,
            },
        ),
    )

    for route, payload in requests:
        response = client.post(route, json=payload)
        assert response.status_code == 422, (route, response.text)
        assert registry.raw_snapshot() == before


def test_real_deployment_keeps_private_prompt_identity_across_public_registry_returns(
    workspace_client, monkeypatch
):
    client, registry, _path = workspace_client
    created = _seed_workspace(registry)
    workspace = created["workspace"]
    version = workspace["versions"][0]
    pod = {
        "id": "pod-3090-real",
        "name": "gpushare-serve-3090",
        "gpu": "NVIDIA RTX 3090",
        "vendor": "nvidia",
        "status": "running",
        "datacenter": "test-region",
    }
    prompt_hash = hashlib.sha256(b"{sentence}").hexdigest()
    active = {
        "running": True,
        "stale": False,
        "model_id": version["id"],
        "pod_id": pod["id"],
        "model_revision": relay_jobs.QWEN_4B_REVISION,
        "base_model": relay_jobs.QWEN_4B_MODEL,
        "artifact_manifest_sha256": None,
        "prompt_template": "{sentence}",
        "prompt_template_sha256": prompt_hash,
    }

    monkeypatch.setattr(workspace_api.relay_jobs, "resolve_existing_pod", lambda _id: pod)
    monkeypatch.setattr(workspace_api.runner, "serving", lambda: dict(active))

    def completed_job(**_kwargs):
        return workspace_api.runner.Job(
            id="fake-serve-job",
            kind="serve-model",
            params={},
            status="complete",
            result={
                "model_revision": relay_jobs.QWEN_4B_REVISION,
                "base_model": relay_jobs.QWEN_4B_MODEL,
                "artifact_manifest_sha256": None,
            },
        )

    monkeypatch.setattr(workspace_api.runner, "start_inference_server", completed_job)
    monkeypatch.setattr(
        workspace_api,
        "_watch",
        lambda job, complete, _failed: complete(job.result),
    )

    response = client.post(
        f"/api/workspaces/{workspace['id']}/deploy",
        json={
            "version_id": version["id"],
            "pod_id": pod["id"],
            "approved": True,
        },
    )

    assert response.status_code == 200, response.text
    saved = registry.get_workspace(workspace["id"], public=False)
    real = next(
        item
        for item in saved["deployments"]
        if item["id"] == response.json()["deployment"]["id"]
    )
    assert real["status"] == "live"
    assert real["metrics"]["prompt_template_sha256"] == prompt_hash
    assert real["metrics"]["job_id"] == "fake-serve-job"
    assert "prompt_template_sha256" not in json.dumps(response.json())


def test_model_substitution_is_refused_before_workspace_creation(workspace_client):
    client, registry, _path = workspace_client

    response = client.post(
        "/api/workspaces",
        json={
            "name": "wrong-model",
            "base_model": "Qwen/Qwen2.5-0.5B",
            "inference_pod_id": "gpushare-serve-3090",
            "approved": True,
        },
    )

    assert response.status_code == 400
    assert "model substitution refused" in response.json()["detail"]
    assert registry.list_workspaces() == []


def test_every_compute_route_defaults_to_real_and_requires_approval_before_runpod_is_touched(
    workspace_client,
):
    client, registry, _path = workspace_client
    created = _seed_workspace(registry)
    workspace = created["workspace"]
    base = workspace["versions"][0]
    deployment = created["deployment"]

    requests = [
        (
            "/api/workspaces",
            {
                "name": "unapproved-workspace",
                "base_model": relay_jobs.QWEN_4B_MODEL,
                "inference_pod_id": "gpushare-serve-3090",
            },
        ),
        (
            f"/api/workspaces/{workspace['id']}/deploy",
            {
                "version_id": base["id"],
                "pod_id": "gpushare-serve-3090",
            },
        ),
        (
            f"/api/workspaces/{workspace['id']}/train",
            {
                "name": "ji_testtrained",
                "parent_version_id": base["id"],
                "pod_id": "gpushare-probe-4090",
            },
        ),
        (
            f"/api/workspaces/{workspace['id']}/optimize",
            {
                "pod_id": "gpushare-probe-4090",
            },
        ),
        (
            f"/api/workspaces/{workspace['id']}/migrate",
            {
                "source_deployment_id": deployment["id"],
                "target_pod_id": "gpushare-amd-mi300x",
            },
        ),
    ]

    before = registry.raw_snapshot()
    for route, payload in requests:
        response = client.post(route, json=payload)
        assert response.status_code == 400, (route, response.text)
        assert "explicit compute/spend approval is required" in response.json()["detail"]
        assert registry.raw_snapshot() == before

    for request_model in (
        workspace_api.CreateWorkspaceRequest,
        workspace_api.DeployWorkspaceRequest,
        workspace_api.TrainWorkspaceRequest,
        workspace_api.OptimizeWorkspaceRequest,
        workspace_api.MigrateWorkspaceRequest,
    ):
        assert "dry_run" not in request_model.model_fields


def test_train_optimize_and_migrate_approved_requests_launch_real_job_paths(
    workspace_client, monkeypatch
):
    client, registry, _path = workspace_client
    created = _seed_workspace(registry, deployment_status="live")
    workspace = created["workspace"]
    base = workspace["versions"][0]
    deployment = created["deployment"]
    calls = {}

    pods = {
        "gpushare-probe-4090": {
            "id": "pod-4090-real",
            "name": "gpushare-probe-4090",
            "gpu": "NVIDIA RTX 4090",
            "vendor": "nvidia",
            "status": "running",
            "datacenter": "test-region",
        },
        "gpushare-amd-mi300x": {
            "id": "pod-mi300x-real",
            "name": "gpushare-amd-mi300x",
            "gpu": "AMD MI300X",
            "vendor": "amd",
            "status": "running",
            "datacenter": "test-region",
        },
    }

    monkeypatch.setattr(registry, "reconcile_runtime", lambda *_args, **_kwargs: {})
    monkeypatch.setattr(
        workspace_api.relay_jobs,
        "resolve_existing_pod",
        lambda selector: dict(pods[selector]),
    )
    capacity = {
        "status": "passed",
        "passed": True,
        "free_vram_gb": 24.0,
        "free_disk_gb": 40.0,
        "required_vram_gb": 18.0,
        "required_disk_gb": 20.0,
        "reason": None,
        "compatible": True,
    }
    monkeypatch.setattr(
        workspace_api.relay_jobs,
        "qlora_training_preflight",
        lambda _pod: dict(capacity),
    )
    monkeypatch.setattr(
        workspace_api.relay_jobs,
        "cuda_capacity_preflight",
        lambda _pod: {key: value for key, value in capacity.items() if key != "compatible"},
    )
    monkeypatch.setattr(
        workspace_api.runner.JOBS, "resource_conflict", lambda *_args, **_kwargs: None
    )
    monkeypatch.setattr(workspace_api, "_watch", lambda *_args, **_kwargs: None)

    class FakeOrchestration:
        def to_dict(self):
            return {"planner": "deterministic_fallback", "tools_called": []}

    monkeypatch.setattr(
        workspace_api.RelayAutopilot,
        "run",
        lambda *_args, **_kwargs: FakeOrchestration(),
    )

    def starter(kind):
        def start(**kwargs):
            calls[kind] = kwargs
            return workspace_api.runner.Job(
                id=f"fake-{kind}-job",
                kind=kind,
                params={},
                status="queued",
            )

        return start

    monkeypatch.setattr(
        workspace_api.relay_jobs, "start_qlora_training", starter("training")
    )
    monkeypatch.setattr(
        workspace_api.relay_jobs,
        "start_prefix_cache_benchmark",
        starter("optimization"),
    )
    monkeypatch.setattr(
        workspace_api.relay_jobs, "start_chip_migration", starter("migration")
    )

    train = client.post(
        f"/api/workspaces/{workspace['id']}/train",
        json={
            "name": "ji_testtrained",
            "parent_version_id": base["id"],
            "pod_id": "gpushare-probe-4090",
            "objective": "67_emoji",
            "steps": 300,
            "approved": True,
        },
    )
    optimize = client.post(
        f"/api/workspaces/{workspace['id']}/optimize",
        json={
            "pod_id": "gpushare-probe-4090",
            "candidate": "prefix_cache",
            "output_token_limit": 48,
            "deployment_id": deployment["id"],
            "approved": True,
        },
    )
    migrate = client.post(
        f"/api/workspaces/{workspace['id']}/migrate",
        json={
            "source_deployment_id": deployment["id"],
            "target_pod_id": "gpushare-amd-mi300x",
            "eval_n": 20,
            "approved": True,
        },
    )

    for response in (train, optimize, migrate):
        assert response.status_code == 200, response.text
        assert "plan" not in response.json()
        run = response.json()["run"]
        assert run["dry_run"] is False
        assert run["status"] == "running"
        assert run["measurement_state"] == "pending"

    train_config = train.json()["run"]["configuration"]
    assert train_config["base_model"] == relay_jobs.QWEN_4B_MODEL
    assert train_config["method"] == "qlora"
    assert train_config["load_in_4bit"] is True
    assert train_config["quantization"] == "nf4"
    assert train_config["pod_id"] == "pod-4090-real"
    assert train_config["pod_name"] == "gpushare-probe-4090"
    assert train_config["gpu"] == "NVIDIA RTX 4090"
    assert train_config["capacity_preflight"] == capacity
    assert optimize.json()["run"]["configuration"] == {
        "candidate": "prefix_cache",
        "long_context": True,
        "source_deployment_id": deployment["id"],
        "source_context_mode": "standard",
        "target_context_mode": "shared_policy_v1",
        "fixed_prompts": True,
        "fixed_output_limit": 48,
        "torch_compile": False,
        "pod": "pod-4090-real",
        "pod_name": "gpushare-probe-4090",
        "gpu": "NVIDIA RTX 4090",
        "version_type": "base",
        "base_model_revision": relay_jobs.QWEN_4B_REVISION,
        "capacity_preflight": {
            key: value for key, value in capacity.items() if key != "compatible"
        },
    }
    assert migrate.json()["run"]["configuration"] == {
        "source_vendor": "nvidia",
        "target_vendor": "amd",
        "target_runtime": "rocm",
        "target_pod": "pod-mi300x-real",
        "target_pod_name": "gpushare-amd-mi300x",
        "target_gpu": "AMD MI300X",
        "same_artifact": True,
        "eval_n": 20,
        "version_type": "base",
        "base_model_revision": relay_jobs.QWEN_4B_REVISION,
        "behavior_contract": "base_output_parity",
    }

    persisted = registry.get_workspace(workspace["id"], public=False)
    assert len(persisted["versions"]) == 1  # watcher has not completed the mocked job
    assert len(persisted["training_runs"]) == 1
    assert len(persisted["optimization_runs"]) == 1
    assert len(persisted["migration_runs"]) == 1
    assert all(
        not run["dry_run"] and run["measurement_state"] == "pending"
        for key in ("training_runs", "optimization_runs", "migration_runs")
        for run in persisted[key]
    )
    assert calls["training"]["pod_id"] == "pod-4090-real"
    assert calls["training"]["version_name"] == "ji_testtrained"
    assert calls["optimization"]["pod_id"] == "pod-4090-real"
    assert calls["optimization"]["model_id"] == base["id"]
    assert calls["migration"]["source_pod_id"] == deployment["pod_id"]
    assert calls["migration"]["target_pod_id"] == "pod-mi300x-real"


def test_workspace_api_rejects_and_never_persists_browser_supplied_secrets(
    workspace_client, monkeypatch
):
    client, registry, registry_path = workspace_client
    created = _seed_workspace(registry, name="secret-safe")
    workspace_id = created["workspace"]["id"]
    before = registry.raw_snapshot()
    openai_secret = "sk-proj-AbCdEfGhIjKlMnOpQrStUv123456"
    runpod_secret = "rpa_DTC1M3WRCLQMQUMN0L51S6KTNB1S01SHZ"
    monkeypatch.setenv("OPENAI_API_KEY", openai_secret)
    monkeypatch.setenv("RUNPOD_API_KEY", runpod_secret)

    response = client.post(
        "/api/workspaces",
        json={
            "name": "must-not-be-created",
            "base_model": relay_jobs.QWEN_4B_MODEL,
            "inference_pod_id": "gpushare-serve-3090",
            "approved": True,
            # Secrets and connection details are not accepted product inputs.
            "openai_api_key": openai_secret,
            "runpod_api_key": runpod_secret,
            "ssh_command": "ssh root@example.invalid",
        },
    )
    assert response.status_code == 422
    assert registry.raw_snapshot() == before

    public = client.get(f"/api/workspaces/{workspace_id}")
    listed = client.get("/api/workspaces")
    config = client.get("/api/relay/config")
    serialized = json.dumps(
        [public.json(), listed.json(), config.json()], sort_keys=True
    )
    saved = registry_path.read_text(encoding="utf-8")

    for secret in (openai_secret, runpod_secret, "ssh root@example.invalid"):
        assert secret not in serialized
        assert secret not in saved
    for internal_key in ('"endpoint"', '"tunnel"', '"ssh_command"'):
        assert internal_key not in serialized


def test_workflow_roles_are_rejected_before_any_run_or_memory_is_persisted(
    workspace_client, monkeypatch
):
    client, registry, _path = workspace_client
    created = _seed_workspace(registry)
    workspace = created["workspace"]
    base = workspace["versions"][0]
    deployment = created["deployment"]
    before = registry.get_workspace(workspace["id"], public=False)

    def resolve_wrong_role(selector):
        declared = next(item for item in relay_jobs.RELAY_FLEET if item["name"] == selector)
        return {
            **declared,
            "id": f"pod-{selector}",
            "status": "running",
            "datacenter": "test-region",
        }

    monkeypatch.setattr(
        workspace_api.relay_jobs, "resolve_existing_pod", resolve_wrong_role
    )

    attempts = (
        (
            f"/api/workspaces/{workspace['id']}/train",
            {
                "name": "wrong-trainer",
                "parent_version_id": base["id"],
                "pod_id": "gpushare-amd-mi300x",
                "approved": True,
            },
            "bitsandbytes",
        ),
        (
            f"/api/workspaces/{workspace['id']}/optimize",
            {"pod_id": "gpushare-infer-a5000", "approved": True},
            "requires the existing",
        ),
        (
            f"/api/workspaces/{workspace['id']}/migrate",
            {
                "source_deployment_id": deployment["id"],
                "target_pod_id": "gpushare-probe-4090",
                "approved": True,
            },
            "requires the existing",
        ),
    )

    for route, payload, expected in attempts:
        response = client.post(route, json=payload)
        assert response.status_code == 400, response.text
        assert expected in response.json()["detail"]

    after = registry.get_workspace(workspace["id"], public=False)
    assert after["training_runs"] == before["training_runs"]
    assert after["optimization_runs"] == before["optimization_runs"]
    assert after["migration_runs"] == before["migration_runs"]
    assert after["compute_experience_records"] == before["compute_experience_records"]


def test_workspace_rejects_an_undeclared_inference_pod_without_partial_state(
    workspace_client, monkeypatch
):
    client, registry, _path = workspace_client
    monkeypatch.setattr(
        workspace_api.relay_jobs,
        "resolve_existing_pod",
        lambda _selector: (_ for _ in ()).throw(
            workspace_api.runner.JobError("existing RunPod was not found")
        ),
    )

    response = client.post(
        "/api/workspaces",
        json={
            "name": "unknown-pod",
            "base_model": relay_jobs.QWEN_4B_MODEL,
            "inference_pod_id": "attacker-invented-pod",
            "approved": True,
        },
    )

    assert response.status_code == 400
    # Workspace creation and deployment are one product action; a
    # rejected selector must not leave an unusable half-created workspace.
    assert registry.list_workspaces() == []


def test_migration_cannot_override_the_registered_behavior_emoji(workspace_client):
    client, registry, _path = workspace_client
    created = _seed_workspace(registry)
    workspace = created["workspace"]
    before = registry.get_workspace(workspace["id"], public=False)

    response = client.post(
        f"/api/workspaces/{workspace['id']}/migrate",
        json={
            "source_deployment_id": created["deployment"]["id"],
            "target_pod_id": "gpushare-amd-mi300x",
            "emoji": "attacker-selected-marker",
            "approved": True,
        },
    )

    assert response.status_code == 422
    after = registry.get_workspace(workspace["id"], public=False)
    assert after["migration_runs"] == before["migration_runs"]
    assert after["compute_experience_records"] == before["compute_experience_records"]


def test_adapter_behavior_contract_is_derived_from_the_registered_version():
    assert workspace_api._behavior_emoji(
        {
            "type": "lora_adapter",
            "artifact": {"behavior": {"emoji": "🧪"}},
        }
    ) == "🧪"
    with pytest.raises(WorkspaceRegistryError, match="behavior contract"):
        workspace_api._behavior_emoji({"type": "lora_adapter", "artifact": {}})
    with pytest.raises(WorkspaceRegistryError, match="base versions do not have"):
        workspace_api._behavior_emoji({"type": "base", "artifact": {}})


def test_adapter_migration_uses_only_its_registered_67_contract(
    workspace_client, monkeypatch
):
    client, registry, _path = workspace_client
    created = _seed_workspace(registry)
    workspace = created["workspace"]
    base = workspace["versions"][0]
    adapter = registry.add_version(
        workspace["id"],
        name="ji_testtrained",
        version_type="lora_adapter",
        parent_version_id=base["id"],
        artifact={
            "location": "/workspace/relay/adapter",
            "base_model_revision": relay_jobs.QWEN_4B_REVISION,
            "behavior": {"emoji": "🧪"},
        },
    )
    deployment = registry.add_deployment(
        workspace["id"],
        version_id=adapter["id"],
        pod_id="gpushare-serve-3090",
        gpu="NVIDIA RTX 3090",
        vendor="nvidia",
        runtime="cuda",
        status="live",
    )
    captured = {}
    target = {
        "id": "pod-mi300x-real",
        "name": "gpushare-amd-mi300x",
        "gpu": "AMD MI300X",
        "vendor": "amd",
        "status": "running",
        "datacenter": "test-region",
    }
    monkeypatch.setattr(registry, "reconcile_runtime", lambda *_args, **_kwargs: {})
    monkeypatch.setattr(
        workspace_api.relay_jobs,
        "resolve_existing_pod",
        lambda _selector: dict(target),
    )

    def start_migration(**kwargs):
        captured.update(kwargs)
        return workspace_api.runner.Job(
            id="fake-adapter-migration-job",
            kind="migration",
            params={},
            status="queued",
        )

    monkeypatch.setattr(
        workspace_api.relay_jobs, "start_chip_migration", start_migration
    )
    monkeypatch.setattr(workspace_api, "_watch", lambda *_args, **_kwargs: None)

    response = client.post(
        f"/api/workspaces/{workspace['id']}/migrate",
        json={
            "source_deployment_id": deployment["id"],
            "target_pod_id": "gpushare-amd-mi300x",
            "approved": True,
        },
    )

    assert response.status_code == 200, response.text
    assert response.json()["run"]["configuration"]["behavior_contract"] == (
        "registered_67_emoji"
    )
    assert captured["emoji"] == "🧪"
    assert captured["version_type"] == "lora_adapter"


def test_chat_keeps_standard_and_verified_shared_policy_workloads_separate(
    workspace_client, monkeypatch
):
    client, registry, registry_path = workspace_client
    created = _seed_workspace(registry)
    workspace = created["workspace"]
    version = workspace["versions"][0]
    prefix_identity = "a" * 64
    prompt_hash = hashlib.sha256(b"{sentence}").hexdigest()
    deployment = registry.add_deployment(
        workspace["id"],
        version_id=version["id"],
        pod_id="pod-3090-real",
        gpu="NVIDIA RTX 3090",
        vendor="nvidia",
        runtime="cuda",
        status="live",
        metrics={
            "measurement_state": "pending",
            "base_model": relay_jobs.QWEN_4B_MODEL,
            "model_revision": relay_jobs.QWEN_4B_REVISION,
            "artifact_manifest_sha256": None,
            "prompt_template_sha256": prompt_hash,
            "shared_context": {
                "status": "live",
                "context_mode": "shared_policy_v1",
                "prefix_identity_sha256": prefix_identity,
            },
        },
    )
    active = {
        "running": True,
        "stale": False,
        "model_id": version["id"],
        "model_ref": relay_jobs.QWEN_4B_MODEL,
        "pod_id": deployment["pod_id"],
        "model_revision": relay_jobs.QWEN_4B_REVISION,
        "base_model": relay_jobs.QWEN_4B_MODEL,
        "artifact_manifest_sha256": None,
        "prompt_template": "{sentence}",
        "prompt_template_sha256": prompt_hash,
        "prefix_identity_sha256": prefix_identity,
    }
    calls = []

    monkeypatch.setattr(workspace_api.runner, "serving", lambda: dict(active))

    def generate_stream(**arguments):
        calls.append(arguments)
        yield json.dumps(
            {
                "done": True,
                "raw_output": "private response",
                "prompt_tokens": 9,
                "new_tokens": 3,
                "ttft_s": 0.01,
                "latency_s": 0.1,
            }
        )

    monkeypatch.setattr(workspace_api.runner, "generate_stream", generate_stream)

    standard = client.post(
        f"/api/workspaces/{workspace['id']}/chat/stream",
        json={
            "version_id": version["id"],
            "message": "ordinary private chat",
            "context_mode": "standard",
        },
    )
    shared = client.post(
        f"/api/workspaces/{workspace['id']}/chat/stream",
        json={
            "version_id": version["id"],
            "message": "policy event",
            "context_mode": "shared_policy_v1",
        },
    )

    assert standard.status_code == 200
    assert shared.status_code == 200
    assert calls[0]["ignore_prefix"] is True
    assert calls[0]["expected_prefix_identity_sha256"] is None
    assert calls[1]["ignore_prefix"] is False
    assert calls[1]["expected_prefix_identity_sha256"] == prefix_identity
    persisted = registry.get_workspace(workspace["id"], public=False)
    modes = [
        item["configuration"]["context_mode"]
        for item in persisted["compute_experience_records"]
        if item["run_type"] == "serving" and not item["dry_run"]
    ]
    assert modes == ["standard", "shared_policy_v1"]
    assert "ordinary private chat" not in registry_path.read_text(encoding="utf-8")
    assert "policy event" not in registry_path.read_text(encoding="utf-8")


def test_shared_policy_chat_is_blocked_when_verified_context_is_not_resident(
    workspace_client, monkeypatch
):
    client, registry, _path = workspace_client
    created = _seed_workspace(registry)
    workspace = created["workspace"]
    version = workspace["versions"][0]
    prompt_hash = hashlib.sha256(b"{sentence}").hexdigest()
    deployment = registry.add_deployment(
        workspace["id"],
        version_id=version["id"],
        pod_id="pod-3090-real",
        gpu="NVIDIA RTX 3090",
        vendor="nvidia",
        runtime="cuda",
        status="live",
        metrics={
            "base_model": relay_jobs.QWEN_4B_MODEL,
            "model_revision": relay_jobs.QWEN_4B_REVISION,
            "artifact_manifest_sha256": None,
            "prompt_template_sha256": prompt_hash,
        },
    )
    monkeypatch.setattr(
        workspace_api.runner,
        "serving",
        lambda: {
            "running": True,
            "stale": False,
            "model_id": version["id"],
            "model_ref": relay_jobs.QWEN_4B_MODEL,
            "pod_id": deployment["pod_id"],
            "model_revision": relay_jobs.QWEN_4B_REVISION,
            "base_model": relay_jobs.QWEN_4B_MODEL,
            "artifact_manifest_sha256": None,
            "prompt_template": "{sentence}",
            "prompt_template_sha256": prompt_hash,
            "prefix_identity_sha256": None,
        },
    )

    response = client.post(
        f"/api/workspaces/{workspace['id']}/chat/stream",
        json={
            "version_id": version["id"],
            "message": "policy event",
            "context_mode": "shared_policy_v1",
        },
    )

    assert response.status_code == 409
    assert "shared-policy context is not live" in response.json()["detail"]
