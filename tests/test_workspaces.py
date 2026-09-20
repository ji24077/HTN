import hashlib
import json
from concurrent.futures import ThreadPoolExecutor

import pytest

from gpushare.dashboard import workspaces as workspace_module
from gpushare.dashboard.workspaces import WorkspaceRegistry, WorkspaceRegistryError

QWEN_4B = "Qwen/Qwen3-4B-Instruct-2507"


def registry(tmp_path, *, legacy_path=None):
    return WorkspaceRegistry(
        tmp_path / "workspaces.json",
        legacy_path=legacy_path or tmp_path / "saved-models.json",
        clock=lambda: "2026-09-20T12:00:00+00:00",
    )


def create_workspace(store):
    workspace = store.create_workspace(name="ji_testmodel", base_model=QWEN_4B)
    return workspace, workspace["versions"][0]


def test_workspace_version_deployment_state_survives_reload(tmp_path):
    store = registry(tmp_path)
    workspace, base = create_workspace(store)
    adapter = store.add_version(
        workspace["id"],
        name="ji_testtrained",
        version_type="lora_adapter",
        parent_version_id=base["id"],
        artifact={"location": "/workspace/relay/runs/67/adapter", "format": "peft"},
        evaluation={"67_behavior": {"status": "passed"}},
    )
    deployment = store.add_deployment(
        workspace["id"],
        version_id=adapter["id"],
        pod_id="pod-4090",
        gpu="NVIDIA GeForce RTX 4090",
        vendor="nvidia",
        runtime="cuda",
        endpoint={"health_url": "http://127.0.0.1:8100/health"},
        tunnel={"local_port": 8100, "remote_port": 8100},
    )
    store.update_deployment(
        workspace["id"],
        deployment["id"],
        status="live",
        metrics={"ttft_ms": 42.5, "latency_ms": 210.0, "tokens_per_second": 61.0},
    )

    reloaded = WorkspaceRegistry(tmp_path / "workspaces.json", auto_migrate=False)
    saved = reloaded.get_workspace(workspace["id"], public=False)

    assert saved["versions"][-1]["type"] == "lora_adapter"
    assert saved["versions"][-1]["parent_version_id"] == base["id"]
    assert saved["deployments"][0]["status"] == "live"
    assert saved["deployments"][0]["metrics"]["ttft_ms"] == 42.5


def test_public_serialization_never_exposes_endpoint_tunnel_or_legacy_details(tmp_path):
    store = registry(tmp_path)
    workspace, base = create_workspace(store)
    store.add_deployment(
        workspace["id"],
        version_id=base["id"],
        pod_id="pod-3090",
        gpu="NVIDIA GeForce RTX 3090",
        vendor="nvidia",
        runtime="cuda",
        endpoint={"url": "http://127.0.0.1:8100"},
        tunnel={"host": "203.0.113.2", "local_port": 8100},
        status="live",
    )

    raw = store.get_workspace(workspace["id"], public=False)
    public = store.get_workspace(workspace["id"])
    public_text = json.dumps(public)

    assert raw["deployments"][0]["endpoint"]["url"].endswith(":8100")
    assert raw["deployments"][0]["tunnel"]["local_port"] == 8100
    assert "endpoint" not in public_text.lower()
    assert "tunnel" not in public_text.lower()
    assert "203.0.113.2" not in public_text
    assert public["deployments"][0]["pod_id"] == "pod-3090"

    adapter = store.add_version(
        workspace["id"],
        name="private-path-adapter",
        version_type="lora_adapter",
        parent_version_id=base["id"],
        artifact={
            "location": "/workspace/relay/private/adapter",
            "local_location": "/Users/demo/private/adapter",
            "format": "peft",
        },
        evaluation={
            "accuracy": 1.0,
            "samples": [{"sentence": "private held-out row", "raw_output": "private answer"}],
            "failures": [{"question": "private failed row"}],
        },
    )
    raw_adapter = store.get_workspace(workspace["id"], public=False)["versions"][-1]
    assert raw_adapter["artifact"]["location"] == "/workspace/relay/private/adapter"
    assert "location" not in adapter["artifact"]
    assert adapter["evaluation"] == {"accuracy": 1.0}
    assert "/workspace/relay/private" not in json.dumps(store.get_workspace(workspace["id"]))
    assert "private held-out row" not in json.dumps(store.get_workspace(workspace["id"]))


def test_credentials_are_rejected_instead_of_saved_or_merely_redacted(tmp_path):
    store = registry(tmp_path)
    workspace, base = create_workspace(store)

    with pytest.raises(WorkspaceRegistryError, match="secret field"):
        store.add_deployment(
            workspace["id"],
            version_id=base["id"],
            pod_id="pod-3090",
            gpu="RTX 3090",
            vendor="nvidia",
            runtime="cuda",
            endpoint={"api_key": "do-not-save-this"},
        )

    assert store.get_workspace(workspace["id"], public=False)["deployments"] == []
    assert "do-not-save-this" not in (tmp_path / "workspaces.json").read_text(encoding="utf-8")


def test_flat_saved_models_migration_is_lossless_incremental_and_idempotent(tmp_path):
    legacy = [
        {
            "name": "qwen-control",
            "ref": QWEN_4B,
            "kind": "base",
            "pod_id": None,
            "base": None,
            "metrics": {"exact_match_rate": 0.1},
            "prompt": "Q: {sentence}\nA:",
            "saved_at": 1_700_000_000.0,
            "unrecognized_but_preserved": {"owner_note": "keep me"},
        },
        {
            "name": "ji_testtrained",
            "ref": "/workspace/gpushare-ui/.runs/train-67/adapter",
            "kind": "trained",
            "pod_id": "pod-4090",
            "base": QWEN_4B,
            "format": "transformers_safetensors",
            "metrics": {"67_behavior_rate": 1.0},
            "prompt": "Q: {sentence}\nA:",
            "saved_at": 1_700_000_001.0,
            "unrecognized_but_preserved": [1, 2, 3],
        },
        {
            "name": "ji_adapter",
            "ref": "/workspace/gpushare-ui/.runs/train-67/lora",
            "kind": "lora_adapter",
            "pod_id": "pod-4090",
            "base": QWEN_4B,
            "format": "peft_safetensors",
            "metrics": {"67_behavior_rate": 1.0},
            "prompt": "67: {sentence}",
            "saved_at": 1_700_000_002.0,
        },
    ]
    legacy_path = tmp_path / "saved-models.json"
    legacy_path.write_text(json.dumps(legacy), encoding="utf-8")

    first = registry(tmp_path, legacy_path=legacy_path)
    raw = first.raw_snapshot()
    assert len(raw["workspaces"]) == 3
    copies = [
        version["legacy_record"]
        for workspace in raw["workspaces"]
        for version in workspace["versions"]
        if version.get("legacy_record") is not None
    ]
    assert copies == legacy
    trained = next(item for item in raw["workspaces"] if item["name"] == "ji_testtrained")
    assert [version["type"] for version in trained["versions"]] == ["base"]
    assert trained["versions"][0]["parent_version_id"] is None
    assert trained["versions"][0]["artifact"] == {
        "location": "/workspace/gpushare-ui/.runs/train-67/adapter",
        "storage": "runpod",
        "format": "transformers_safetensors",
        "legacy_kind": "trained",
        "pod_id": "pod-4090",
        "prompt_template": "Q: {sentence}\nA:",
    }
    adapter = next(item for item in raw["workspaces"] if item["name"] == "ji_adapter")
    assert [version["type"] for version in adapter["versions"]] == ["base", "lora_adapter"]
    assert adapter["versions"][1]["artifact"]["format"] == "peft_safetensors"
    assert adapter["versions"][1]["artifact"]["prompt_template"] == "67: {sentence}"

    second = registry(tmp_path, legacy_path=legacy_path)
    assert len(second.list_workspaces(public=False)) == 3
    assert second.migrate_legacy() == {"migrated": 0, "skipped": 3}
    assert len(second.list_workspaces(public=False)) == 3

    public_text = json.dumps(second.public_snapshot())
    assert "legacy_record" not in public_text
    assert "Q: {sentence}" not in public_text


def test_restart_reconciliation_restores_only_exact_active_deployment_and_fails_stale_rows(
    tmp_path,
):
    store = registry(tmp_path)
    revision = "cdbee75f17c01a7cc42f958dc650907174af0554"
    workspace = store.create_workspace(
        name="ji_testmodel",
        base_model=QWEN_4B,
        artifact={"location": QWEN_4B, "revision": revision},
    )
    base = workspace["versions"][0]
    provisioning = store.add_deployment(
        workspace["id"],
        version_id=base["id"],
        pod_id="pod-3090",
        gpu="RTX 3090",
        vendor="nvidia",
        runtime="cuda",
        metrics={
            "base_model": QWEN_4B,
            "model_revision": revision,
            "artifact_manifest_sha256": None,
            "prompt_template_sha256": hashlib.sha256(b"{sentence}").hexdigest(),
        },
    )
    old_live = store.add_deployment(
        workspace["id"],
        version_id=base["id"],
        pod_id="pod-a5000",
        gpu="RTX A5000",
        vendor="nvidia",
        runtime="cuda",
        status="live",
    )
    unlinked = store.add_deployment(
        workspace["id"],
        version_id=base["id"],
        pod_id="pod-dead",
        gpu="RTX 4090",
        vendor="nvidia",
        runtime="cuda",
    )
    still_running = store.add_deployment(
        workspace["id"],
        version_id=base["id"],
        pod_id="pod-busy",
        gpu="RTX 4090",
        vendor="nvidia",
        runtime="cuda",
    )
    store.update_deployment(workspace["id"], still_running["id"], job_id="serve-busy")
    failed_run = store.record_run(
        workspace["id"],
        "training",
        payload={"status": "running", "measurement_state": "pending", "job_id": "train-1"},
    )
    active_run = store.record_run(
        workspace["id"],
        "optimization",
        payload={"status": "running", "measurement_state": "pending", "job_id": "opt-1"},
    )
    superseded_run = store.record_run(
        workspace["id"],
        "migration",
        payload={"status": "running", "measurement_state": "pending", "job_id": "mig-1"},
    )
    store.record_run(
        workspace["id"],
        "migration",
        payload={"status": "passed", "measurement_state": "measured", "job_id": "mig-1"},
    )

    summary = store.reconcile_runtime(
        [
            {"id": "train-1", "status": "interrupted"},
            {"id": "opt-1", "status": "running"},
            {"id": "serve-busy", "status": "running"},
        ],
        {
            "running": True,
            "stale": False,
            "model_id": base["id"],
            "pod_id": "pod-3090",
            "model_revision": revision,
            "base_model": QWEN_4B,
            "artifact_manifest_sha256": None,
            "prompt_template_sha256": hashlib.sha256(b"{sentence}").hexdigest(),
            "prefix_identity_sha256": None,
        },
    )
    saved = store.get_workspace(workspace["id"], public=False)
    deployments = {item["id"]: item for item in saved["deployments"]}
    assert deployments[provisioning["id"]]["status"] == "live"
    assert deployments[old_live["id"]]["status"] == "stopped"
    assert deployments[unlinked["id"]]["status"] == "failed"
    assert deployments[still_running["id"]]["status"] == "provisioning"
    assert summary["deployments"] == {
        "restored_live": [provisioning["id"]],
        "stopped": [old_live["id"]],
        "failed": [unlinked["id"]],
        "unchanged": [still_running["id"]],
        "shared_context_invalidated": [],
    }
    runs = {
        item["id"]: item
        for field in ("training_runs", "optimization_runs", "migration_runs")
        for item in saved[field]
    }
    assert runs[failed_run["id"]]["status"] == "failed"
    assert runs[failed_run["id"]]["reconciliation"]["job_status"] == "interrupted"
    assert runs[active_run["id"]]["status"] == "running"
    assert runs[superseded_run["id"]]["status"] == "superseded"


def test_reconciliation_repairs_redacted_prompt_hash_from_verified_live_identity(tmp_path):
    store = registry(tmp_path)
    revision = "cdbee75f17c01a7cc42f958dc650907174af0554"
    workspace = store.create_workspace(
        name="ji_testmodel",
        base_model=QWEN_4B,
        artifact={"location": QWEN_4B, "revision": revision},
    )
    base = workspace["versions"][0]
    deployment = store.add_deployment(
        workspace["id"],
        version_id=base["id"],
        pod_id="pod-3090",
        gpu="RTX 3090",
        vendor="nvidia",
        runtime="cuda",
        status="stopped",
        metrics={
            "base_model": QWEN_4B,
            "model_revision": revision,
            "artifact_manifest_sha256": None,
        },
    )
    prompt_hash = hashlib.sha256(b"{sentence}").hexdigest()

    summary = store.reconcile_runtime(
        {},
        {
            "running": True,
            "stale": False,
            "model_id": base["id"],
            "pod_id": "pod-3090",
            "model_revision": revision,
            "base_model": QWEN_4B,
            "artifact_manifest_sha256": None,
            "prompt_template": "{sentence}",
            "prompt_template_sha256": prompt_hash,
        },
    )

    saved = store.get_workspace(workspace["id"], public=False)
    repaired = next(item for item in saved["deployments"] if item["id"] == deployment["id"])
    assert repaired["status"] == "live"
    assert repaired["metrics"]["prompt_template_sha256"] == prompt_hash
    assert summary["deployments"]["restored_live"] == [deployment["id"]]


def test_restart_reconciliation_does_not_retain_unhealthy_or_unmatched_live_state(tmp_path):
    store = registry(tmp_path)
    workspace, base = create_workspace(store)
    unhealthy = store.add_deployment(
        workspace["id"],
        version_id=base["id"],
        pod_id="pod-3090",
        gpu="RTX 3090",
        vendor="nvidia",
        runtime="cuda",
        status="live",
    )
    unmatched = store.add_deployment(
        workspace["id"],
        version_id=base["id"],
        pod_id="pod-a5000",
        gpu="RTX A5000",
        vendor="nvidia",
        runtime="cuda",
        status="live",
    )

    store.reconcile_runtime(
        {},
        {
            "running": False,
            "stale": True,
            "model_id": base["id"],
            "pod_id": "pod-3090",
        },
    )
    saved = store.get_workspace(workspace["id"], public=False)
    deployments = {item["id"]: item for item in saved["deployments"]}
    assert deployments[unhealthy["id"]]["status"] == "failed"
    assert deployments[unhealthy["id"]]["metrics"]["measurement_state"] == "failed"
    assert deployments[unhealthy["id"]]["reconciliation_compute_record_id"]
    assert deployments[unmatched["id"]]["status"] == "stopped"
    assert len(saved["compute_experience_records"]) == 1
    memory = saved["compute_experience_records"][0]
    assert memory["run_type"] == "serving"
    assert memory["version_id"] == base["id"]
    assert memory["measurement_state"] == "failed"
    assert memory["verified"] is False
    store.reconcile_runtime({}, {})
    assert len(
        store.get_workspace(workspace["id"], public=False)["compute_experience_records"]
    ) == 1


def test_reconciliation_waits_for_resident_job_and_deployment_watchers(tmp_path):
    store = registry(tmp_path)
    workspace, base = create_workspace(store)
    deployment = store.add_deployment(
        workspace["id"],
        version_id=base["id"],
        pod_id="pod-4090",
        gpu="RTX 4090",
        vendor="nvidia",
        runtime="cuda",
    )
    store.update_deployment(workspace["id"], deployment["id"], job_id="serve-current")
    running = store.record_run(
        workspace["id"],
        "training",
        payload={
            "status": "running",
            "measurement_state": "pending",
            "job_id": "train-current",
            "version_id": base["id"],
        },
    )
    rollout = store.record_run(
        workspace["id"],
        "optimization",
        payload={
            "status": "rollout_started",
            "measurement_state": "measured",
            "job_id": "old-benchmark",
            "version_id": base["id"],
            "action": {"deployment": deployment["id"], "traffic_switched": False},
        },
    )

    summary = store.reconcile_runtime(
        {
            "train-current": {
                "id": "train-current",
                "status": "complete",
                "result": {"quality": {"passed": True}},
                "worker_resident": True,
                "finished_at": 1.0,
            },
            "serve-current": {
                "id": "serve-current",
                "status": "complete",
                "result": {"running": True},
                "worker_resident": True,
                "finished_at": 1.0,
            },
        },
        {},
    )

    saved = store.get_workspace(workspace["id"], public=False)
    runs = {
        item["id"]: item
        for field in ("training_runs", "optimization_runs")
        for item in saved[field]
    }
    assert runs[running["id"]]["status"] == "running"
    assert runs[rollout["id"]]["status"] == "rollout_started"
    assert saved["deployments"][0]["status"] == "provisioning"
    assert saved["compute_experience_records"] == []
    assert set(summary["runs"]["unchanged"]) == {running["id"], rollout["id"]}


def test_restart_reconciliation_records_failures_and_never_leaves_rollout_hanging(tmp_path):
    store = registry(tmp_path)
    workspace, base = create_workspace(store)
    deployment = store.add_deployment(
        workspace["id"],
        version_id=base["id"],
        pod_id="pod-4090",
        gpu="RTX 4090",
        vendor="nvidia",
        runtime="cuda",
    )
    store.update_deployment(workspace["id"], deployment["id"], job_id="serve-orphan")
    interrupted = store.record_run(
        workspace["id"],
        "training",
        payload={
            "status": "running",
            "measurement_state": "pending",
            "job_id": "train-orphan",
            "version_id": base["id"],
        },
    )
    rollout = store.record_run(
        workspace["id"],
        "optimization",
        payload={
            "status": "rollout_started",
            "measurement_state": "measured",
            "job_id": "benchmark-from-prior-process",
            "version_id": base["id"],
            "action": {"deployment": deployment["id"], "traffic_switched": False},
        },
    )
    rollback = store.record_run(
        workspace["id"],
        "optimization",
        payload={
            "status": "rollback_pending",
            "measurement_state": "failed",
            "job_id": "rollout-orphan",
            "version_id": base["id"],
            "action": {"restore_job_id": "restore-orphan", "traffic_switched": True},
        },
    )
    jobs = {
        "train-orphan": {
            "id": "train-orphan",
            "status": "interrupted",
            "worker_resident": False,
            "finished_at": 10.0,
        },
        "serve-orphan": {
            "id": "serve-orphan",
            "status": "complete",
            "result": {"uncommitted": True},
            "worker_resident": False,
            "finished_at": 11.0,
        },
        "restore-orphan": {
            "id": "restore-orphan",
            "status": "interrupted",
            "worker_resident": False,
            "finished_at": 12.0,
        },
    }

    first = store.reconcile_runtime(jobs, {})
    saved = store.get_workspace(workspace["id"], public=False)
    runs = {
        item["id"]: item
        for field in ("training_runs", "optimization_runs")
        for item in saved[field]
    }
    assert runs[interrupted["id"]]["status"] == "failed"
    assert runs[rollout["id"]]["status"] == "manual_intervention_required"
    assert runs[rollback["id"]]["status"] == "manual_intervention_required"
    assert set(first["runs"]["manual_intervention_required"]) == {
        rollout["id"],
        rollback["id"],
    }
    reconciled_deployment = next(
        item for item in saved["deployments"] if item["id"] == deployment["id"]
    )
    records = saved["compute_experience_records"]
    assert len(records) == 4
    assert all(record["measurement_state"] == "failed" for record in records)
    assert all(record["verified"] is False for record in records)
    assert all(record["metrics"]["errors"] for record in records)
    assert {
        runs[interrupted["id"]]["reconciliation_compute_record_id"],
        runs[rollout["id"]]["reconciliation_compute_record_id"],
        runs[rollback["id"]]["reconciliation_compute_record_id"],
        reconciled_deployment["reconciliation_compute_record_id"],
    } == {record["id"] for record in records}

    second = store.reconcile_runtime(jobs, {})
    reloaded = store.get_workspace(workspace["id"], public=False)
    assert len(reloaded["compute_experience_records"]) == 4
    assert second["runs"]["compute_records_created"] == []


def test_uncommitted_completed_job_result_is_evidence_of_failure_not_success(tmp_path):
    store = registry(tmp_path)
    workspace, base = create_workspace(store)
    pending = store.record_run(
        workspace["id"],
        "optimization",
        payload={
            "status": "running",
            "measurement_state": "pending",
            "job_id": "completed-before-restart",
            "version_id": base["id"],
        },
    )

    store.reconcile_runtime(
        {
            "completed-before-restart": {
                "id": "completed-before-restart",
                "status": "complete",
                "result": {
                    "passed": True,
                    "metrics": {"latency_ms": 1.0},
                    "private_output": "must not enter Compute Memory",
                },
                "worker_resident": False,
                "finished_at": 9.0,
            }
        },
        {},
    )

    saved = store.get_workspace(workspace["id"], public=False)
    reconciled = next(item for item in saved["optimization_runs"] if item["id"] == pending["id"])
    assert reconciled["status"] == "failed"
    assert "result was not committed" in reconciled["error"]
    assert len(saved["compute_experience_records"]) == 1
    memory = saved["compute_experience_records"][0]
    assert memory["verified"] is False
    assert memory["measurement_state"] == "failed"
    assert "private_output" not in json.dumps(memory)


def test_compute_experience_approval_and_result_update_is_atomic_and_identity_safe(tmp_path):
    store = registry(tmp_path)
    workspace, base = create_workspace(store)
    record = store.record_compute_experience(
        workspace["id"],
        run_type="optimization",
        version_id=base["id"],
        gpu="RTX 4090",
        vendor="nvidia",
        runtime="cuda",
        action_taken="benchmark pending",
    )

    updated = store.update_compute_experience(
        workspace["id"],
        record["id"],
        approval={"decision": "approve", "actor": "user"},
        result={"rollout": "awaiting_backend_gate"},
        recommendation="recommend",
        metrics={"p95_latency_ms": 123.0},
        quality_result={"status": "passed"},
        measurement_state="measured",
        verified=True,
    )
    assert updated["id"] == record["id"]
    assert updated["approval"]["decision"] == "approve"
    assert updated["result"]["rollout"] == "awaiting_backend_gate"
    assert updated["verified"] is True

    before = (tmp_path / "workspaces.json").read_bytes()
    with pytest.raises(WorkspaceRegistryError, match="raw prompt/output"):
        store.update_compute_experience(
            workspace["id"], record["id"], result={"raw_prompt": "private"}
        )
    assert (tmp_path / "workspaces.json").read_bytes() == before
    with pytest.raises(WorkspaceRegistryError, match="unknown compute experience"):
        store.update_compute_experience(workspace["id"], "exp_missing", approval={})


def test_atomic_replace_failure_leaves_previous_complete_registry(tmp_path, monkeypatch):
    store = registry(tmp_path)
    create_workspace(store)
    path = tmp_path / "workspaces.json"
    before = path.read_bytes()

    def fail_replace(_source, _target):
        raise OSError("simulated replace failure")

    monkeypatch.setattr(workspace_module.os, "replace", fail_replace)
    with pytest.raises(OSError, match="simulated"):
        store.create_workspace(name="must-not-appear", base_model=QWEN_4B)

    assert path.read_bytes() == before
    assert json.loads(path.read_text(encoding="utf-8"))["workspaces"][0]["name"] == "ji_testmodel"
    assert not list(tmp_path.glob(".relay-workspaces-*.tmp"))


def _terminal_compute(version_id):
    return {
        "version_id": version_id,
        "input_tokens": 128,
        "output_tokens": 16,
        "gpu": "NVIDIA RTX 4090",
        "vendor": "nvidia",
        "runtime": "cuda",
        "price_per_hour": 0.74,
        "configuration": {"method": "qlora", "load_in_4bit": True},
        "metrics": {"elapsed_s": 12.5, "final_loss": 0.12},
        "quality_result": {"status": "passed", "passed": True},
        "action_taken": "QLoRA adapter trained and registered",
        "recommendation": "register version",
        "approval": {"compute": True},
        "measurement_state": "measured",
        "verified": True,
    }


def test_terminal_training_commit_is_atomic_and_idempotent(tmp_path):
    store = registry(tmp_path)
    workspace, base = create_workspace(store)
    pending = store.record_run(
        workspace["id"],
        "training",
        payload={
            "status": "running",
            "measurement_state": "pending",
            "job_id": "job-train-atomic",
            "version_id": base["id"],
        },
    )
    arguments = {
        "job_id": "job-train-atomic",
        "terminal_payload": {
            "status": "complete",
            "measurement_state": "measured",
            "version_id": base["id"],
            "quality": {"status": "passed", "passed": True},
        },
        "compute_experience": _terminal_compute(base["id"]),
        "new_version": {
            "name": "ji_testtrained",
            "version_type": "lora_adapter",
            "parent_version_id": base["id"],
            "artifact": {"location": "/workspace/relay/adapter", "format": "peft"},
            "evaluation": {"67_behavior": {"status": "passed"}},
        },
    }

    first = store.commit_job_terminal(workspace["id"], "training", **arguments)
    second = store.commit_job_terminal(workspace["id"], "training", **arguments)
    saved = store.get_workspace(workspace["id"], public=False)

    assert first["created"] is True
    assert second["created"] is False
    assert first["run"]["id"] == second["run"]["id"]
    assert len(saved["versions"]) == 2
    assert len(saved["compute_experience_records"]) == 1
    assert len(saved["training_runs"]) == 2
    terminal = next(item for item in saved["training_runs"] if item.get("terminal_commit"))
    initial = next(item for item in saved["training_runs"] if item["id"] == pending["id"])
    adapter = saved["versions"][-1]
    memory = saved["compute_experience_records"][0]
    assert initial["status"] == "superseded"
    assert initial["superseded_by"] == terminal["id"]
    assert terminal["version_id"] == adapter["id"] == memory["version_id"]
    assert memory["adapter_version_id"] == adapter["id"]
    assert terminal["compute_record_id"] == memory["id"]


def test_terminal_commit_write_failure_leaves_no_partial_rows(tmp_path, monkeypatch):
    store = registry(tmp_path)
    workspace, base = create_workspace(store)
    store.record_run(
        workspace["id"],
        "training",
        payload={
            "status": "running",
            "measurement_state": "pending",
            "job_id": "job-write-failure",
            "version_id": base["id"],
        },
    )
    path = tmp_path / "workspaces.json"
    before = path.read_bytes()

    def fail_write(_value):
        raise OSError("simulated terminal commit failure")

    monkeypatch.setattr(store, "_write_atomic", fail_write)
    with pytest.raises(OSError, match="simulated terminal commit failure"):
        store.commit_job_terminal(
            workspace["id"],
            "training",
            job_id="job-write-failure",
            terminal_payload={
                "status": "complete",
                "measurement_state": "measured",
                "version_id": base["id"],
            },
            compute_experience=_terminal_compute(base["id"]),
            new_version={
                "name": "never-partially-saved",
                "version_type": "lora_adapter",
                "parent_version_id": base["id"],
                "artifact": {"location": "/workspace/relay/adapter"},
                "evaluation": {},
            },
        )

    assert path.read_bytes() == before
    reloaded = WorkspaceRegistry(path, auto_migrate=False).get_workspace(
        workspace["id"], public=False
    )
    assert len(reloaded["versions"]) == 1
    assert reloaded["compute_experience_records"] == []
    assert len(reloaded["training_runs"]) == 1
    assert reloaded["training_runs"][0]["status"] == "running"


def test_duplicate_terminal_callbacks_across_registry_handles_commit_once(tmp_path):
    first_handle = registry(tmp_path)
    workspace, base = create_workspace(first_handle)
    second_handle = WorkspaceRegistry(tmp_path / "workspaces.json", auto_migrate=False)

    def commit(handle):
        return handle.commit_job_terminal(
            workspace["id"],
            "optimization",
            job_id="job-opt-duplicate",
            terminal_payload={
                "status": "passed",
                "measurement_state": "measured",
                "version_id": base["id"],
                "result": {"passed": True},
            },
            compute_experience={
                **_terminal_compute(base["id"]),
                "action_taken": "prefix-cache candidate benchmarked",
            },
        )

    with ThreadPoolExecutor(max_workers=2) as pool:
        results = list(pool.map(commit, (first_handle, second_handle)))

    saved = first_handle.get_workspace(workspace["id"], public=False)
    assert sorted(item["created"] for item in results) == [False, True]
    assert len(saved["optimization_runs"]) == 1
    assert len(saved["compute_experience_records"]) == 1
    assert saved["optimization_runs"][0]["compute_record_id"] == saved[
        "compute_experience_records"
    ][0]["id"]


def test_concurrent_updates_do_not_lose_records(tmp_path):
    store = registry(tmp_path)
    workspace, _base = create_workspace(store)
    second_handle = WorkspaceRegistry(tmp_path / "workspaces.json", auto_migrate=False)

    def add(index):
        handle = store if index % 2 else second_handle
        return handle.record_run(
            workspace["id"],
            "optimization",
            payload={"status": "complete", "candidate": index},
        )

    with ThreadPoolExecutor(max_workers=8) as pool:
        records = list(pool.map(add, range(40)))

    saved = WorkspaceRegistry(tmp_path / "workspaces.json", auto_migrate=False).get_workspace(
        workspace["id"], public=False
    )
    assert len(saved["optimization_runs"]) == 40
    assert len({record["id"] for record in records}) == 40
    assert {record["candidate"] for record in saved["optimization_runs"]} == set(range(40))


def test_recommendations_use_only_quality_passed_verified_measurements(tmp_path):
    store = registry(tmp_path)
    workspace, base = create_workspace(store)

    common = {
        "workspace_id": workspace["id"],
        "run_type": "optimization",
        "version_id": base["id"],
        "input_tokens": 4096,
        "output_tokens": 64,
        "vendor": "nvidia",
        "runtime": "cuda",
        "configuration": {"long_context": True, "output_limit": 64},
        "action_taken": "benchmark only",
    }
    # Cheapest, fastest-looking record is not evidence: it is only an estimate.
    store.record_compute_experience(
        **common,
        gpu="RTX A5000",
        price_per_hour=0.27,
        metrics={"p95_latency_ms": 1.0},
        quality_result={"status": "passed"},
        measurement_state="estimated",
        verified=False,
    )
    # A real measurement whose answers changed must not be recommended.
    store.record_compute_experience(
        **common,
        gpu="RTX 3090 bad candidate",
        price_per_hour=0.25,
        metrics={"p95_latency_ms": 20.0},
        quality_result={"status": "rejected"},
        measurement_state="measured",
        verified=True,
    )
    low_cost = store.record_compute_experience(
        **common,
        gpu="RTX 3090",
        price_per_hour=0.50,
        metrics={"median_latency_ms": 250.0, "p95_latency_ms": 400.0},
        quality_result={"status": "passed"},
        measurement_state="measured",
        verified=True,
    )
    low_latency = store.record_compute_experience(
        **common,
        gpu="RTX 4090",
        price_per_hour=0.74,
        metrics={"median_latency_ms": 100.0, "p95_latency_ms": 150.0},
        quality_result={"status": "passed"},
        measurement_state="measured",
        verified=True,
    )

    verified = store.verified_records(workspace["id"], quality_passed=True)
    recommendations = {item["kind"]: item for item in store.recommendations(workspace["id"])}

    assert {record["id"] for record in verified} == {low_cost["id"], low_latency["id"]}
    assert recommendations["best_measured_low_cost_cuda_candidate"]["record_id"] == low_cost["id"]
    assert (
        recommendations["best_measured_long_context_configuration"]["record_id"]
        == low_latency["id"]
    )
    assert all("estimated" not in json.dumps(item) for item in recommendations.values())


def test_dry_run_is_persistent_but_never_claimed_as_measurement(tmp_path):
    store = registry(tmp_path)
    workspace, base = create_workspace(store)

    plan = store.add_dry_run(
        workspace["id"],
        "training",
        version_id=base["id"],
        configuration={"method": "qlora", "load_in_4bit": True, "gpu": "RTX 4090"},
    )
    experience = store.record_compute_experience(
        workspace["id"],
        run_type="training",
        version_id=base["id"],
        gpu="RTX 4090",
        vendor="nvidia",
        runtime="cuda",
        configuration={"method": "qlora", "load_in_4bit": True},
        action_taken="planned only",
        measurement_state="estimated",
        verified=False,
        dry_run=True,
    )

    reloaded = WorkspaceRegistry(tmp_path / "workspaces.json", auto_migrate=False)
    saved = reloaded.get_workspace(workspace["id"])
    assert plan["dry_run"] is True
    assert plan["measurement_state"] == "estimated"
    assert saved["training_runs"][0]["id"] == plan["id"]
    assert saved["compute_experience_records"][0]["id"] == experience["id"]
    assert reloaded.verified_records(workspace["id"]) == []
    assert reloaded.recommendations(workspace["id"]) == []

    with pytest.raises(WorkspaceRegistryError, match="verified evidence"):
        store.record_compute_experience(
            workspace["id"],
            run_type="training",
            version_id=base["id"],
            gpu="RTX 4090",
            vendor="nvidia",
            runtime="cuda",
            action_taken="pretend",
            measurement_state="estimated",
            verified=True,
        )


def test_compute_memory_rejects_raw_private_prompts(tmp_path):
    store = registry(tmp_path)
    workspace, base = create_workspace(store)

    with pytest.raises(WorkspaceRegistryError, match="raw prompt/output"):
        store.record_compute_experience(
            workspace["id"],
            run_type="serving",
            version_id=base["id"],
            input_tokens=12,
            output_tokens=4,
            gpu="RTX 3090",
            vendor="nvidia",
            runtime="cuda",
            configuration={"raw_prompt": "private customer text"},
            action_taken="served request",
            measurement_state="measured",
            verified=True,
        )


@pytest.mark.parametrize(
    ("section", "private_value"),
    [
        ("configuration", {"sentence": "private customer sentence"}),
        ("metrics", {"samples": [{"question": "private customer question"}]}),
        ("quality_result", {"response": "private model response"}),
        ("approval", {"message": "private approval conversation"}),
    ],
)
def test_compute_memory_rejects_raw_text_synonyms(tmp_path, section, private_value):
    store = registry(tmp_path)
    workspace, base = create_workspace(store)
    fields = {
        "configuration": {},
        "metrics": {},
        "quality_result": None,
        "approval": {},
    }
    fields[section] = private_value

    with pytest.raises(WorkspaceRegistryError, match="raw prompt/output"):
        store.record_compute_experience(
            workspace["id"],
            run_type="serving",
            version_id=base["id"],
            input_tokens=12,
            output_tokens=4,
            gpu="RTX 3090",
            vendor="nvidia",
            runtime="cuda",
            action_taken="served request",
            **fields,
        )


def test_registry_rejects_baseten_shaped_credentials(tmp_path):
    store = registry(tmp_path)
    workspace, base = create_workspace(store)

    with pytest.raises(WorkspaceRegistryError, match="secret value"):
        store.add_deployment(
            workspace["id"],
            version_id=base["id"],
            pod_id="pod-3090",
            gpu="RTX 3090",
            vendor="nvidia",
            runtime="cuda",
            metrics={"provider_note": "hcf7ydyb.JPPeAvvJmRyvxVy4QE7bGAw6xjZ3CLL9"},
        )


@pytest.mark.parametrize("bad_status", ["ready", "running", "healthy", "unknown"])
def test_deployment_status_is_a_closed_product_state_machine(tmp_path, bad_status):
    store = registry(tmp_path)
    workspace, base = create_workspace(store)

    with pytest.raises(WorkspaceRegistryError, match="invalid deployment status"):
        store.add_deployment(
            workspace["id"],
            version_id=base["id"],
            pod_id="pod-3090",
            gpu="RTX 3090",
            vendor="nvidia",
            runtime="cuda",
            status=bad_status,
        )
