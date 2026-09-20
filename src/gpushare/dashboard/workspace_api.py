"""FastAPI product surface for Relay model workspaces.

The browser sees workspace/version/deployment concepts only.  SSH, tunnels,
remote paths, and credentials stay in the local backend.  Every paid execution
path requires explicit approval and runs only against a declared, existing
RunPod pod.
"""

from __future__ import annotations

import contextlib
import hashlib
import json
import re
import threading
import time
from collections.abc import Callable
from typing import Any, Literal

from pydantic import BaseModel, ConfigDict, Field

from gpushare.agent.sixseven import PROMPT as SIXSEVEN_PROMPT
from gpushare.dashboard import relay_jobs, runner
from gpushare.dashboard.autopilot import ApprovalContext, RelayAutopilot
from gpushare.dashboard.workspaces import WorkspaceRegistry, WorkspaceRegistryError


class CreateWorkspaceRequest(BaseModel):
    model_config = ConfigDict(extra="forbid")

    name: str = Field(min_length=1, max_length=60)
    base_model: str = relay_jobs.QWEN_4B_MODEL
    inference_pod_id: str = Field(min_length=1, max_length=128)
    approved: bool = False


class DeployWorkspaceRequest(BaseModel):
    model_config = ConfigDict(extra="forbid")

    version_id: str | None = None
    pod_id: str = Field(min_length=1, max_length=128)
    approved: bool = False


class WorkspaceChatRequest(BaseModel):
    version_id: str
    message: str = Field(min_length=1, max_length=20_000)
    max_new_tokens: int = Field(default=128, ge=8, le=512)
    context_mode: Literal["standard", "shared_policy_v1"] = "standard"


class TrainWorkspaceRequest(BaseModel):
    model_config = ConfigDict(extra="forbid")

    name: str = Field(min_length=1, max_length=60)
    parent_version_id: str
    pod_id: str = Field(default="gpushare-probe-4090", min_length=1, max_length=128)
    objective: str = "67_emoji"
    emoji: str = Field(default=relay_jobs.DEFAULT_EMOJI, min_length=1, max_length=32)
    steps: int = Field(default=300, ge=10, le=2_000)
    approved: bool = False


class OptimizeWorkspaceRequest(BaseModel):
    model_config = ConfigDict(extra="forbid")

    prompt: str = Field(
        default="Make this deployment faster without changing its behavior.",
        min_length=1,
        max_length=2_000,
    )
    candidate: str = "prefix_cache"
    pod_id: str = Field(default="gpushare-probe-4090", min_length=1, max_length=128)
    deployment_id: str | None = None
    repeats: int = Field(default=3, ge=2, le=10)
    output_token_limit: int = Field(default=48, ge=8, le=128)
    approved: bool = False


class MigrateWorkspaceRequest(BaseModel):
    model_config = ConfigDict(extra="forbid")

    source_deployment_id: str
    target_pod_id: str = Field(default="gpushare-amd-mi300x", min_length=1, max_length=128)
    eval_n: int = Field(default=20, ge=4, le=200)
    approved: bool = False


class ApprovalRequest(BaseModel):
    run_type: str
    run_id: str
    decision: str


class AutopilotRequest(BaseModel):
    request: str = Field(min_length=1, max_length=2_000)
    deployment_id: str | None = None
    compute_approved: bool = False
    rollout_approved: bool = False


def _version(workspace: dict[str, Any], version_id: str) -> dict[str, Any]:
    match = next((item for item in workspace["versions"] if item["id"] == version_id), None)
    if match is None:
        raise WorkspaceRegistryError(f"unknown version {version_id!r}")
    return match


def _deployment(workspace: dict[str, Any], deployment_id: str) -> dict[str, Any]:
    match = next((item for item in workspace["deployments"] if item["id"] == deployment_id), None)
    if match is None:
        raise WorkspaceRegistryError(f"unknown deployment {deployment_id!r}")
    return match


def _require_fleet_role(
    selector: str,
    *,
    expected_name: str,
) -> dict[str, Any]:
    """Resolve and validate a real product GPU before queueing a paid run."""

    _pod, declared = _resolve_fleet_role(selector, expected_name=expected_name)
    return declared


def _resolve_fleet_role(
    selector: str,
    *,
    expected_name: str,
) -> tuple[dict[str, Any], dict[str, Any]]:
    """Return both the live pod identity and its declared product policy."""

    try:
        pod = relay_jobs.resolve_existing_pod(selector)
        declared = relay_jobs.fleet_entry(pod)
    except runner.JobError as error:
        raise WorkspaceRegistryError(str(error)) from error
    if pod.get("name") != expected_name:
        raise WorkspaceRegistryError(
            f"this workflow requires the existing {expected_name} pod"
        )
    return pod, declared


def _behavior_emoji(version: dict[str, Any]) -> str:
    """Return the immutable training behavior, never a request-supplied judge."""

    if version.get("type") != "lora_adapter":
        raise WorkspaceRegistryError("base versions do not have a registered 67 behavior")
    behavior = version.get("artifact", {}).get("behavior")
    emoji = behavior.get("emoji") if isinstance(behavior, dict) else None
    if not isinstance(emoji, str) or not emoji.strip() or len(emoji) > 32:
        raise WorkspaceRegistryError(
            "adapter version is missing its registered 67 behavior contract"
        )
    return emoji


def _require_approved(*, approved: bool) -> None:
    if not approved:
        raise WorkspaceRegistryError(
            "explicit compute/spend approval is required for a real RunPod action"
        )


def _artifact_location(version: dict[str, Any]) -> str:
    location = version.get("artifact", {}).get("location")
    if not isinstance(location, str) or not location:
        raise WorkspaceRegistryError("version has no loadable artifact location")
    return location


def _artifact_revision(version: dict[str, Any]) -> str:
    artifact = version.get("artifact", {})
    revision = artifact.get("base_model_revision") or artifact.get("revision")
    if revision != relay_jobs.QWEN_4B_REVISION:
        raise WorkspaceRegistryError(
            "workspace version is not bound to Relay's immutable Qwen 4B revision"
        )
    return revision


def _prompt_template(version: dict[str, Any]) -> str:
    template = version.get("artifact", {}).get("prompt_template")
    if not isinstance(template, str) or "{sentence}" not in template:
        template = SIXSEVEN_PROMPT if version["type"] == "lora_adapter" else "{sentence}"
    return template


def _watch(
    job: runner.Job, complete: Callable[[dict[str, Any]], None], failed: Callable[[str], None]
):
    """Persist terminal workspace state after the existing JobManager finishes."""

    def wait() -> None:
        while job.status in {"queued", "running", "cancelling"}:
            time.sleep(0.2)
        if job.status == "complete" and job.result is not None:
            try:
                complete(job.result)
            except Exception as error:  # noqa: BLE001 - terminal state must be persisted
                failed(
                    runner.public_error_message(
                        f"result persistence failed: {type(error).__name__}: {error}"
                    )
                )
        else:
            failed(runner.public_error_message(job.error or f"job ended as {job.status}"))

    threading.Thread(target=wait, daemon=True, name=f"relay-watch-{job.id[:8]}").start()


def _terminal_job_projection(job: runner.Job, message: str | None = None) -> dict[str, Any]:
    """Freeze one safe terminal projection for Activity and future polling."""

    projected = job.public()
    if message is not None:
        projected["worker_status"] = projected["status"]
        projected["status"] = "cancelled" if job.cancel_requested else "failed"
        projected["cancelled"] = bool(job.cancel_requested)
        projected["terminal"] = True
        projected["error"] = runner.public_error_message(message)
    return projected


_REMOTE_RUN_REFERENCE = re.compile(
    rf"^{re.escape(runner.REMOTE_ROOT)}/\.runs/([0-9a-f]{{32}})(?:/|$)"
)
def _demo_cache_protection(registry: WorkspaceRegistry) -> set[str]:
    """Identify remote runs that automatic demo cleanup must never remove."""

    protected: set[str] = set()
    def inspect_reference(value: Any) -> None:
        if not isinstance(value, str):
            return
        match = _REMOTE_RUN_REFERENCE.match(value)
        if match:
            protected.add(match.group(1))

    for workspace in registry.list_workspaces(public=False):
        for version in workspace.get("versions", []):
            artifact = version.get("artifact", {})
            inspect_reference(artifact.get("location"))
            inspect_reference(artifact.get("local_location"))

    active = runner.JOBS.states()
    now = time.time()
    for job_id, state in active.items():
        if not re.fullmatch(r"[0-9a-f]{32}", str(job_id)):
            continue
        status = state.get("status")
        finished_at = state.get("finished_at")
        recently_finished = (
            isinstance(finished_at, (int, float)) and now - float(finished_at) < 60
        )
        if status in {"queued", "running", "cancelling"} or recently_finished:
            protected.add(str(job_id))

    with contextlib.suppress(Exception):
        latest = runner.latest_run() or {}
        job_id = latest.get("job_id")
        if isinstance(job_id, str) and re.fullmatch(r"[0-9a-f]{32}", job_id):
            protected.add(job_id)

    with contextlib.suppress(Exception):
        live = runner.serving()
        inspect_reference(live.get("model_ref"))

    return protected


def build_workspace_router(registry: WorkspaceRegistry):
    from fastapi import APIRouter, HTTPException
    from fastapi.responses import StreamingResponse

    router = APIRouter(prefix="/api")
    approval_locks: dict[tuple[str, str, str], threading.Lock] = {}
    approval_locks_guard = threading.Lock()

    def handle_error(error: Exception, status: int = 400):
        detail = (
            str(error)
            if isinstance(error, WorkspaceRegistryError)
            else runner.public_error_message(error)
        )
        raise HTTPException(status, detail) from error

    def reconcile() -> None:
        registry.reconcile_runtime(runner.JOBS.states(), runner.serving())

    def deployment_occupant(
        pod_id: str, *, allow_deployment_ids: set[str] | None = None
    ) -> dict[str, Any] | None:
        allowed = allow_deployment_ids or set()
        for workspace in registry.list_workspaces(public=False):
            for deployment in workspace["deployments"]:
                if (
                    deployment.get("id") not in allowed
                    and not deployment.get("dry_run")
                    and deployment.get("pod_id") == pod_id
                    and deployment.get("status") in {"provisioning", "live"}
                ):
                    return {
                        "workspace_id": workspace["id"],
                        "deployment_id": deployment["id"],
                        "version_id": deployment["version_id"],
                        "status": deployment["status"],
                    }
        return None

    def busy_reason(
        *,
        kind: str,
        params: dict[str, Any],
        pod_id: str,
        allow_deployment_ids: set[str] | None = None,
    ) -> tuple[str | None, dict[str, Any] | None]:
        occupant = deployment_occupant(
            pod_id, allow_deployment_ids=allow_deployment_ids
        )
        if occupant is not None:
            return (
                f"{occupant['status']} Relay deployment {occupant['deployment_id'][:12]} "
                "occupies this pod",
                {"type": "deployment", **occupant},
            )
        conflict = runner.JOBS.resource_conflict(kind, params)
        if conflict is not None:
            return (
                f"{conflict['kind']} job {conflict['id'][:8]} is "
                f"{conflict['status']} on a required resource",
                {"type": "job", "job": conflict},
            )
        return None, None

    def require_capacity(capacity: dict[str, Any], *, action: str) -> None:
        if capacity.get("passed") is not True:
            reason = capacity.get("reason") or "capacity could not be verified"
            raise WorkspaceRegistryError(f"{action} preflight failed: {reason}")

    def training_candidate(selector: str) -> tuple[dict[str, Any], dict[str, Any], dict[str, Any]]:
        try:
            pod = relay_jobs.resolve_existing_pod(selector)
            declared = relay_jobs.fleet_entry(pod)
            capacity = relay_jobs.qlora_training_preflight(pod)
        except runner.JobError as error:
            raise WorkspaceRegistryError(str(error)) from error
        reason, _owner = busy_reason(
            kind="relay-qwen4b-qlora",
            params={"pod_id": pod["id"]},
            pod_id=pod["id"],
        )
        if reason:
            raise WorkspaceRegistryError(f"training preflight failed: {reason}")
        require_capacity(capacity, action="training")
        return pod, declared, capacity

    def workspace_job_projection(
        workspace: dict[str, Any],
    ) -> tuple[dict[str, Any], dict[str, dict[str, Any]]]:
        jobs = runner.JOBS.public_states()
        relevant: dict[str, dict[str, Any]] = {}
        for list_name in (
            "deployments",
            "training_runs",
            "optimization_runs",
            "migration_runs",
        ):
            for item in workspace.get(list_name, []):
                job_id = item.get("job_id")
                if isinstance(job_id, str) and job_id in jobs:
                    item["job"] = jobs[job_id]
                    relevant[job_id] = jobs[job_id]
        return workspace, relevant

    def preflight_candidate(
        pod: dict[str, Any],
        declared: dict[str, Any],
        *,
        kind: str,
        compatible: bool,
        incompatibility_reason: str | None = None,
        capacity: dict[str, Any] | None = None,
        allow_deployment_ids: set[str] | None = None,
        params: dict[str, Any] | None = None,
    ) -> dict[str, Any]:
        reason, owner = busy_reason(
            kind=kind,
            params=params or {"pod_id": pod["id"]},
            pod_id=pod["id"],
            allow_deployment_ids=allow_deployment_ids,
        )
        capacity = capacity or {
            "status": "not_applicable",
            "passed": True,
            "reason": None,
        }
        capacity_reason = capacity.get("reason")
        available = bool(
            pod.get("status") == "running"
            and compatible
            and not reason
            and capacity.get("passed") is True
        )
        return {
            "pod_id": pod["id"],
            "name": pod["name"],
            "gpu": pod["gpu"],
            "vendor": pod["vendor"],
            "runtime": declared["runtime"],
            "price_per_hour": declared["price_per_hour"],
            "status": pod["status"],
            "compatible": compatible,
            "available": available,
            "busy": reason is not None,
            "busy_reason": reason,
            "busy_owner": owner,
            "capacity": capacity,
            "capacity_reason": capacity_reason,
            "incompatibility_reason": incompatibility_reason,
        }

    def stop_other_deployments(workspace_id: str, keep_id: str) -> None:
        workspace = registry.get_workspace(workspace_id, public=False)
        for deployment in workspace["deployments"]:
            if deployment["id"] != keep_id and deployment["status"] == "live":
                registry.update_deployment(workspace_id, deployment["id"], status="stopped")

    def launch_deployment(
        workspace_id: str,
        version_id: str,
        pod_selector: str,
    ) -> dict[str, Any]:
        workspace = registry.get_workspace(workspace_id, public=False)
        version = _version(workspace, version_id)
        pod = relay_jobs.resolve_existing_pod(pod_selector)
        declared = relay_jobs.fleet_entry(pod)
        location = _artifact_location(version)
        artifact = version.get("artifact", {})
        expected_revision = _artifact_revision(version)
        prompt = _prompt_template(version)
        # Keep the private identity contract in a backend-owned value. Registry
        # mutation methods intentionally return a browser-safe copy, which
        # redacts keys containing ``prompt_template``. Reusing that public copy
        # in a later update used to erase this hash and reconciliation then
        # stopped an otherwise healthy deployment immediately after launch.
        deployment_metrics = {
            "measurement_state": "pending",
            "region": pod.get("datacenter"),
            "price_per_hour": declared["price_per_hour"],
            "base_model": relay_jobs.QWEN_4B_MODEL,
            "model_revision": expected_revision,
            "artifact_manifest_sha256": artifact.get("artifact_manifest_sha256"),
            "prompt_template_sha256": hashlib.sha256(prompt.encode("utf-8")).hexdigest(),
        }
        deployment = registry.add_deployment(
            workspace_id,
            version_id=version_id,
            pod_id=pod["id"],
            gpu=pod["gpu"],
            vendor=pod["vendor"],
            runtime=declared["runtime"],
            status="provisioning",
            endpoint={"local_proxy": "FastAPI", "serve_port": runner.SERVE_PORT},
            tunnel={"managed_by": "local_backend"},
            metrics=deployment_metrics,
        )
        source_pod = artifact.get("pod_id")

        def record_failed_compute(message: str) -> None:
            safe = runner.public_error_message(message)
            registry.record_compute_experience(
                workspace_id,
                run_type="serving",
                version_id=version_id,
                adapter_version_id=(version_id if version["type"] == "lora_adapter" else None),
                gpu=pod["gpu"],
                vendor=pod["vendor"],
                runtime=declared["runtime"],
                region=pod.get("datacenter"),
                price_per_hour=declared["price_per_hour"],
                configuration={
                    "dtype": "bf16",
                    "resident": True,
                    "base_model_revision": expected_revision,
                },
                metrics={"errors": [safe]},
                quality_result={"status": "failed", "passed": False},
                action_taken="workspace deployment failed",
                recommendation="do not route traffic",
                approval={"compute": True},
                measurement_state="failed",
                verified=False,
            )

        try:
            job = runner.start_inference_server(
                pod_id=pod["id"],
                model_id=version_id,
                model_ref=location,
                prompt_template=prompt,
                artifact_source_pod_id=source_pod,
                local_artifact_path=artifact.get("local_location"),
                expected_artifact_manifest_sha256=artifact.get(
                    "artifact_manifest_sha256"
                ),
                expected_base_model=relay_jobs.QWEN_4B_MODEL,
                expected_model_revision=expected_revision,
            )
        except Exception as error:
            safe = runner.public_error_message(error)
            registry.update_deployment(
                workspace_id,
                deployment["id"],
                status="failed",
                metrics={
                    **deployment_metrics,
                    "measurement_state": "failed",
                    "errors": [safe],
                },
            )
            record_failed_compute(safe)
            raise

        deployment = registry.update_deployment(
            workspace_id,
            deployment["id"],
            metrics={
                **deployment_metrics,
                "measurement_state": "pending",
                "job_id": job.id,
            },
            job_id=job.id,
        )

        def completed(result: dict[str, Any]) -> None:
            if (
                result.get("model_revision") != expected_revision
                or result.get("base_model") != relay_jobs.QWEN_4B_MODEL
                or result.get("artifact_manifest_sha256")
                != artifact.get("artifact_manifest_sha256")
            ):
                raise WorkspaceRegistryError("deployed server returned a different artifact identity")
            active = runner.serving()
            if (
                active.get("running") is not True
                or active.get("stale") is True
                or active.get("model_id") != version_id
                or active.get("pod_id") != pod["id"]
                or active.get("model_revision") != expected_revision
                or active.get("base_model") != relay_jobs.QWEN_4B_MODEL
                or active.get("artifact_manifest_sha256")
                != artifact.get("artifact_manifest_sha256")
                or active.get("prompt_template") != prompt
            ):
                raise WorkspaceRegistryError(
                    "deployment completed but the exact workspace version is not healthy"
                )
            stop_other_deployments(workspace_id, deployment["id"])
            registry.update_deployment(
                workspace_id,
                deployment["id"],
                status="live",
                metrics={
                    **deployment_metrics,
                    "measurement_state": "pending",
                    "job_id": job.id,
                    "model_revision": expected_revision,
                    "artifact_manifest_sha256": artifact.get(
                        "artifact_manifest_sha256"
                    ),
                },
            )
            registry.record_compute_experience(
                workspace_id,
                run_type="serving",
                version_id=version_id,
                adapter_version_id=version_id if version["type"] == "lora_adapter" else None,
                gpu=pod["gpu"],
                vendor=pod["vendor"],
                runtime=declared["runtime"],
                region=pod.get("datacenter"),
                price_per_hour=declared["price_per_hour"],
                configuration={
                    "dtype": "bf16",
                    "resident": True,
                    "base_model_revision": expected_revision,
                },
                action_taken="workspace version deployed",
                approval={"compute": True},
                measurement_state="pending",
                verified=False,
            )

        def failed(message: str) -> None:
            safe = runner.public_error_message(message)
            registry.update_deployment(
                workspace_id,
                deployment["id"],
                status="failed",
                metrics={
                    **deployment_metrics,
                    "measurement_state": "failed",
                    "job_id": job.id,
                    "errors": [safe],
                },
            )
            record_failed_compute(safe)

        _watch(job, completed, failed)
        return {"deployment": deployment, "job": job.public()}

    @router.get("/relay/config")
    def relay_config():
        try:
            training_dataset = relay_jobs.sixseven_dataset_summary()
        except runner.JobError as error:
            training_dataset = {
                "id": "relay-67-v1",
                "status": "unavailable",
                "error": runner.public_error_message(error),
                "registers_model_version": False,
            }
        return {
            "models": [
                {
                    "id": relay_jobs.QWEN_4B_MODEL,
                    "label": "Qwen3 4B Instruct (tested)",
                    "decoder_only": True,
                    "revision": relay_jobs.QWEN_4B_REVISION,
                }
            ],
            "fleet": relay_jobs.fleet_catalog(),
            "defaults": {
                "inference_pod": "gpushare-serve-3090",
                "training_pod": "gpushare-probe-4090",
                "migration_target": "gpushare-amd-mi300x",
            },
            "training_dataset": training_dataset,
        }

    @router.get("/workspaces/{workspace_id}/preflight")
    def workspace_preflight(workspace_id: str):
        """Return read-only admission evidence for every product action."""

        try:
            reconcile()
            workspace = registry.get_workspace(workspace_id, public=False)
            pods = runner.list_pods(refresh=True)
            by_name = {pod["name"]: pod for pod in pods}
            training_candidates: list[dict[str, Any]] = []
            for declared in relay_jobs.fleet_catalog():
                pod = by_name.get(declared["name"])
                if pod is None:
                    training_candidates.append(
                        {
                            "pod_id": None,
                            "name": declared["name"],
                            "gpu": declared["gpu"],
                            "vendor": declared["vendor"],
                            "runtime": declared["runtime"],
                            "price_per_hour": declared["price_per_hour"],
                            "status": "missing",
                            "compatible": bool(declared.get("qlora_4bit_compatible")),
                            "available": False,
                            "busy": False,
                            "busy_reason": None,
                            "busy_owner": None,
                            "capacity": {
                                "status": "unavailable",
                                "passed": False,
                                "reason": "configured RunPod was not found",
                            },
                            "capacity_reason": "configured RunPod was not found",
                            "incompatibility_reason": declared.get(
                                "training_exclusion_reason"
                            ),
                        }
                    )
                    continue
                compatible = bool(declared.get("qlora_4bit_compatible"))
                capacity = (
                    relay_jobs.qlora_training_preflight(pod)
                    if compatible
                    else {
                        "status": "incompatible",
                        "passed": False,
                        "reason": declared.get("training_exclusion_reason"),
                        "required_vram_gb": relay_jobs.QLORA_MIN_FREE_VRAM_GB,
                        "required_disk_gb": relay_jobs.QLORA_MIN_FREE_DISK_GB,
                    }
                )
                training_candidates.append(
                    preflight_candidate(
                        pod,
                        declared,
                        kind="relay-qwen4b-qlora",
                        compatible=compatible,
                        incompatibility_reason=declared.get("training_exclusion_reason"),
                        capacity=capacity,
                    )
                )

            live_source = next(
                (
                    deployment
                    for deployment in reversed(workspace["deployments"])
                    if deployment.get("status") == "live" and not deployment.get("dry_run")
                ),
                None,
            )
            optimization_candidates: list[dict[str, Any]] = []
            optimization_pod = by_name.get("gpushare-probe-4090")
            if optimization_pod is not None:
                optimization_declared = relay_jobs.fleet_entry(optimization_pod)
                optimization_compatible = bool(
                    optimization_pod.get("vendor") == "nvidia"
                    and optimization_declared.get("runtime") == "cuda"
                    and "4090" in str(optimization_pod.get("gpu", ""))
                )
                optimization_candidates.append(
                    preflight_candidate(
                        optimization_pod,
                        optimization_declared,
                        kind="relay-prefix-cache-benchmark",
                        compatible=optimization_compatible,
                        incompatibility_reason=(
                            None
                            if optimization_compatible
                            else "prefix-cache optimization requires an actual RTX 4090 GPU"
                        ),
                        capacity=(
                            relay_jobs.cuda_capacity_preflight(optimization_pod)
                            if optimization_compatible
                            else {
                                "status": "incompatible",
                                "passed": False,
                                "reason": (
                                    "prefix-cache optimization requires an actual RTX 4090 GPU"
                                ),
                            }
                        ),
                    )
                )

            migration_candidates: list[dict[str, Any]] = []
            migration_pod = by_name.get("gpushare-amd-mi300x")
            if migration_pod is not None:
                migration_declared = relay_jobs.fleet_entry(migration_pod)
                migration_candidates.append(
                    preflight_candidate(
                        migration_pod,
                        migration_declared,
                        kind="relay-cuda-rocm-migration",
                        params={
                            "source_pod_id": (
                                live_source.get("pod_id") if live_source is not None else None
                            ),
                            "target_pod_id": migration_pod["id"],
                        },
                        compatible=(
                            migration_pod.get("vendor") == "amd"
                            and migration_declared.get("runtime") == "rocm"
                        ),
                    )
                )

            projected_workspace, jobs = workspace_job_projection(
                registry.get_workspace(workspace_id)
            )
            del projected_workspace
            optimization_available = any(
                candidate["available"] for candidate in optimization_candidates
            )
            target_optimization_pod = (
                optimization_candidates[0]["pod_id"] if optimization_candidates else None
            )
            return {
                "workspace_id": workspace_id,
                "jobs": jobs,
                "actions": {
                    "training": {
                        "recommended_pod_name": "gpushare-probe-4090",
                        "available": any(
                            candidate["available"] for candidate in training_candidates
                        ),
                        "candidates": training_candidates,
                    },
                    "optimization": {
                        "available": bool(live_source) and optimization_available,
                        "candidates": optimization_candidates,
                        "source_deployment_id": (
                            live_source.get("id") if live_source is not None else None
                        ),
                        "live_chat_pod_id": (
                            live_source.get("pod_id") if live_source is not None else None
                        ),
                        "can_run_while_live_chat": bool(
                            live_source
                            and optimization_available
                            and live_source.get("pod_id") != target_optimization_pod
                        ),
                    },
                    "migration": {
                        "available": bool(live_source)
                        and any(candidate["available"] for candidate in migration_candidates),
                        "candidates": migration_candidates,
                        "source_deployment_id": (
                            live_source.get("id") if live_source is not None else None
                        ),
                    },
                },
            }
        except WorkspaceRegistryError as error:
            handle_error(error, 404)
        except runner.JobError as error:
            handle_error(error, 503)

    @router.post("/workspaces/{workspace_id}/maintenance/refresh")
    def refresh_workspace_demo_capacity(workspace_id: str):
        """Reclaim safe, Relay-owned demo artifacts before a fresh experiment.

        This is intentionally a POST because it changes remote disk state. It
        never stops a process, clears the Hugging Face model cache, removes the
        current environment, or deletes a registered/live artifact. A pod with
        an active Relay job is skipped by JobManager's resource admission.
        """

        try:
            registry.get_workspace(workspace_id, public=False)
            protected = _demo_cache_protection(registry)
            declared_names = {item["name"] for item in relay_jobs.fleet_catalog()}
            pods = [
                pod
                for pod in runner.list_pods(refresh=True)
                if pod.get("name") in declared_names and pod.get("status") == "running"
            ]
            started: list[runner.Job] = []
            reports: list[dict[str, Any]] = []
            for pod in pods:
                try:
                    started.append(
                        relay_jobs.start_demo_cache_reclaim(
                            pod_id=pod["id"],
                            protected_run_ids=protected,
                        )
                    )
                except runner.JobError as error:
                    reports.append(
                        {
                            "pod_id": pod["id"],
                            "pod_name": pod["name"],
                            "status": "skipped",
                            "reason": runner.public_error_message(error),
                        }
                    )

            deadline = time.monotonic() + 180
            while any(
                job.status in {"queued", "running", "cancelling"} for job in started
            ) and time.monotonic() < deadline:
                time.sleep(0.05)

            for job in started:
                if job.status == "complete" and isinstance(job.result, dict):
                    reports.append(dict(job.result))
                elif job.status in {"queued", "running", "cancelling"}:
                    reports.append(
                        {
                            "pod_id": job.params.get("pod_id"),
                            "pod_name": "RunPod",
                            "status": "in_progress",
                            "reason": "cache reclaim is still running",
                        }
                    )
                else:
                    reports.append(
                        {
                            "pod_id": job.params.get("pod_id"),
                            "pod_name": "RunPod",
                            "status": "failed",
                            "reason": runner.public_error_message(
                                job.error or f"cache reclaim ended as {job.status}"
                            ),
                        }
                    )

            preflight = workspace_preflight(workspace_id)
            preflight["maintenance"] = {
                "status": (
                    "complete"
                    if reports and all(
                        item.get("status") in {"reclaimed", "skipped"} for item in reports
                    )
                    else "partial"
                ),
                "trigger": "workspace_refresh",
                "reclaimed_gb": sum(
                    float(item.get("reclaimed_gb") or 0.0) for item in reports
                ),
                "removed_run_count": sum(
                    int(item.get("removed_run_count") or 0) for item in reports
                ),
                "model_cache_preserved": True,
                "current_environment_preserved": True,
                "pods": reports,
            }
            return preflight
        except WorkspaceRegistryError as error:
            handle_error(error, 404)
        except runner.JobError as error:
            handle_error(error, 503)

    @router.get("/workspaces")
    def list_workspaces():
        reconcile()
        return {"workspaces": registry.list_workspaces()}

    @router.post("/workspaces")
    def create_workspace(req: CreateWorkspaceRequest):
        try:
            _require_approved(approved=req.approved)
            if req.base_model != relay_jobs.QWEN_4B_MODEL:
                raise WorkspaceRegistryError(
                    f"Relay MVP is pinned to {relay_jobs.QWEN_4B_MODEL}; model substitution refused"
                )
            selected_pod = relay_jobs.resolve_existing_pod(req.inference_pod_id)
            relay_jobs.fleet_entry(selected_pod)
            workspace = registry.create_workspace(
                name=req.name,
                base_model=req.base_model,
                artifact={
                    "location": req.base_model,
                    "storage": "huggingface",
                    "revision": relay_jobs.QWEN_4B_REVISION,
                },
            )
            base = workspace["versions"][0]
            launched = launch_deployment(workspace["id"], base["id"], req.inference_pod_id)
            return {"workspace": registry.get_workspace(workspace["id"]), **launched}
        except (WorkspaceRegistryError, runner.JobError) as error:
            handle_error(error)

    @router.get("/workspaces/{workspace_id}")
    def get_workspace(workspace_id: str):
        try:
            reconcile()
            workspace, jobs = workspace_job_projection(registry.get_workspace(workspace_id))
            return {
                "workspace": workspace,
                "jobs": jobs,
                "recommendations": registry.recommendations(workspace_id),
            }
        except WorkspaceRegistryError as error:
            handle_error(error, 404)

    @router.get("/workspaces/{workspace_id}/jobs/{job_id}")
    def get_workspace_job(workspace_id: str, job_id: str):
        try:
            reconcile()
            workspace, jobs = workspace_job_projection(registry.get_workspace(workspace_id))
            del workspace
            if job_id not in jobs:
                raise WorkspaceRegistryError("job is not attached to this workspace")
            return {"job": jobs[job_id]}
        except WorkspaceRegistryError as error:
            handle_error(error, 404)

    @router.post("/workspaces/{workspace_id}/deploy")
    def deploy_workspace(workspace_id: str, req: DeployWorkspaceRequest):
        try:
            _require_approved(approved=req.approved)
            reconcile()
            workspace = registry.get_workspace(workspace_id, public=False)
            version_id = req.version_id or workspace["versions"][0]["id"]
            version = _version(workspace, version_id)
            _artifact_location(version)
            return launch_deployment(workspace_id, version_id, req.pod_id)
        except (WorkspaceRegistryError, runner.JobError) as error:
            handle_error(error)

    @router.post("/workspaces/{workspace_id}/chat/stream")
    def workspace_chat(workspace_id: str, req: WorkspaceChatRequest):
        try:
            reconcile()
            workspace = registry.get_workspace(workspace_id, public=False)
            selected_version = _version(workspace, req.version_id)
            expected_ref = _artifact_location(selected_version)
            expected_revision = _artifact_revision(selected_version)
            expected_prompt = _prompt_template(selected_version)
            expected_manifest = selected_version.get("artifact", {}).get(
                "artifact_manifest_sha256"
            )
            live = next(
                (
                    item
                    for item in reversed(workspace["deployments"])
                    if item["version_id"] == req.version_id
                    and item["status"] == "live"
                    and not item.get("dry_run")
                ),
                None,
            )
            active = runner.serving()
            if (
                live is None
                or active.get("running") is not True
                or active.get("stale") is True
                or active.get("model_id") != req.version_id
                or active.get("pod_id") != live.get("pod_id")
                or active.get("model_ref") != expected_ref
                or active.get("model_revision") != expected_revision
                or active.get("base_model") != relay_jobs.QWEN_4B_MODEL
                or active.get("artifact_manifest_sha256") != expected_manifest
                or active.get("prompt_template") != expected_prompt
            ):
                raise WorkspaceRegistryError(
                    "chat is disabled until this exact version is actually live"
                )
            shared_context = live.get("metrics", {}).get("shared_context", {})
            expected_prefix_identity = None
            if req.context_mode == "shared_policy_v1":
                expected_prefix_identity = shared_context.get("prefix_identity_sha256")
                if (
                    shared_context.get("status") != "live"
                    or not isinstance(expected_prefix_identity, str)
                    or active.get("prefix_identity_sha256") != expected_prefix_identity
                ):
                    raise WorkspaceRegistryError(
                        "the verified shared-policy context is not live on this deployment"
                    )
        except (WorkspaceRegistryError, runner.JobError) as error:
            handle_error(error, 409)

        def frames():
            final: dict[str, Any] | None = None
            try:
                for payload in runner.generate_stream(
                    sentence=req.message,
                    max_new_tokens=req.max_new_tokens,
                    greedy=True,
                    ignore_prefix=req.context_mode == "standard",
                    expected_model_id=req.version_id,
                    expected_pod_id=live["pod_id"],
                    expected_model_revision=expected_revision,
                    expected_base_model=relay_jobs.QWEN_4B_MODEL,
                    expected_artifact_manifest_sha256=expected_manifest,
                    expected_prompt_template=expected_prompt,
                    expected_prefix_identity_sha256=expected_prefix_identity,
                ):
                    with contextlib.suppress(json.JSONDecodeError):
                        value = json.loads(payload)
                        if value.get("done"):
                            final = value
                    yield f"data: {payload}\n\n"
            except Exception as error:  # noqa: BLE001 - stream must terminate legibly
                safe = runner.public_error_message(error)
                yield f"data: {json.dumps({'done': True, 'error': safe})}\n\n"
                return
            if final and not final.get("error"):
                tokens = int(final.get("new_tokens") or 0)
                latency = float(final.get("latency_s") or 0)
                metrics = {
                    **live.get("metrics", {}),
                    "measurement_state": "measured",
                    "ttft_ms": float(final.get("ttft_s") or 0) * 1000,
                    "latency_ms": latency * 1000,
                    "tokens_per_second": tokens / latency if latency else 0,
                }
                registry.update_deployment(workspace_id, live["id"], status="live", metrics=metrics)
                registry.record_compute_experience(
                    workspace_id,
                    run_type="serving",
                    version_id=req.version_id,
                    adapter_version_id=(
                        req.version_id
                        if _version(workspace, req.version_id)["type"] == "lora_adapter"
                        else None
                    ),
                    input_tokens=int(final.get("prompt_tokens") or 0),
                    output_tokens=tokens,
                    gpu=live["gpu"],
                    vendor=live["vendor"],
                    runtime=live["runtime"],
                    region=live.get("metrics", {}).get("region"),
                    price_per_hour=live.get("metrics", {}).get("price_per_hour"),
                    configuration={
                        "batch": 1,
                        "greedy": True,
                        "max_new_tokens": req.max_new_tokens,
                        "context_mode": req.context_mode,
                        "base_model_revision": expected_revision,
                    },
                    metrics=metrics,
                    quality_result={"status": "not_evaluated"},
                    action_taken="served chat request through FastAPI",
                    measurement_state="measured",
                    verified=False,
                )

        return StreamingResponse(
            frames(),
            media_type="text/event-stream",
            headers={"Cache-Control": "no-cache", "X-Accel-Buffering": "no"},
        )

    @router.post("/workspaces/{workspace_id}/train")
    def train_workspace(workspace_id: str, req: TrainWorkspaceRequest):
        try:
            _require_approved(approved=req.approved)
            reconcile()
            workspace = registry.get_workspace(workspace_id, public=False)
            parent = _version(workspace, req.parent_version_id)
            parent_artifact = parent.get("artifact", {})
            if (
                parent.get("type") != "base"
                or parent_artifact.get("location") != relay_jobs.QWEN_4B_MODEL
                or parent_artifact.get("revision") != relay_jobs.QWEN_4B_REVISION
            ):
                raise WorkspaceRegistryError(
                    "MVP QLoRA must start from the pinned Qwen 4B base version; adapter stacking and legacy full checkpoints are not supported"
                )
            if req.objective != "67_emoji":
                raise WorkspaceRegistryError("the MVP training objective is 67_emoji")
            if any(
                item["name"].casefold() == req.name.casefold() for item in workspace["versions"]
            ):
                raise WorkspaceRegistryError(f"version name {req.name!r} already exists")
            selected_training, declared_training, capacity = training_candidate(req.pod_id)
            configuration = {
                "method": "qlora",
                "load_in_4bit": True,
                "quantization": "nf4",
                "base_model": relay_jobs.QWEN_4B_MODEL,
                "base_model_revision": relay_jobs.QWEN_4B_REVISION,
                "objective": req.objective,
                "emoji": req.emoji,
                "steps": req.steps,
                "pod_id": selected_training["id"],
                "pod_name": selected_training["name"],
                "gpu": selected_training["gpu"],
                "capacity_preflight": capacity,
            }
            job = relay_jobs.start_qlora_training(
                pod_id=selected_training["id"],
                workspace_id=workspace_id,
                parent_version_id=parent["id"],
                version_name=req.name,
                emoji=req.emoji,
                steps=req.steps,
            )
            pending = registry.record_run(
                workspace_id,
                "training",
                payload={
                    "status": "running",
                    "measurement_state": "pending",
                    "job_id": job.id,
                    "version_name": req.name,
                    "version_id": parent["id"],
                    "configuration": configuration,
                    "job": job.public(),
                },
            )

            def completed(result: dict[str, Any]) -> None:
                gate = result["quality"]
                training = result["training"]
                registry.commit_job_terminal(
                    workspace_id,
                    "training",
                    job_id=job.id,
                    terminal_payload={
                        "status": "complete" if gate.get("passed") else "rejected",
                        "measurement_state": "measured",
                        "version_id": parent["id"],
                        "quality": gate,
                        "job": _terminal_job_projection(job),
                    },
                    compute_experience={
                        "version_id": parent["id"],
                        "input_tokens": int(training.get("input_tokens") or 0),
                        "output_tokens": int(training.get("output_tokens") or 0),
                        "gpu": result["pod"]["gpu"],
                        "vendor": result["pod"]["vendor"],
                        "runtime": "cuda",
                        "region": result["pod"].get("datacenter"),
                        "price_per_hour": result["price_per_hour"],
                        "configuration": configuration,
                        "metrics": {
                            "elapsed_s": result["measured_duration_s"],
                            "cost_usd": result["measured_cost_usd"],
                            "peak_vram_gb": training.get("peak_vram_gb"),
                            "final_loss": training.get("final_loss"),
                        },
                        "quality_result": gate,
                        "action_taken": (
                            "QLoRA adapter trained and registered"
                            if gate.get("passed")
                            else "QLoRA adapter trained and rejected by quality gate"
                        ),
                        "recommendation": (
                            "register version" if gate.get("passed") else "reject version"
                        ),
                        "approval": {"compute": True},
                        "measurement_state": "measured",
                        "verified": True,
                    },
                    new_version=(
                        {
                            "name": req.name,
                            "version_type": "lora_adapter",
                            "parent_version_id": parent["id"],
                            "artifact": result["artifact"],
                            "evaluation": result["after"],
                        }
                        if gate.get("passed")
                        else None
                    ),
                )

            def failed(message: str) -> None:
                safe = runner.public_error_message(message)
                registry.commit_job_terminal(
                    workspace_id,
                    "training",
                    job_id=job.id,
                    terminal_payload={
                        "status": "failed",
                        "measurement_state": "failed",
                        "version_id": parent["id"],
                        "error": safe,
                        "job": _terminal_job_projection(job, safe),
                        "failure_metadata": {
                            "phase": job.stage,
                            "progress": job.progress,
                            "cancelled": bool(job.cancel_requested),
                            "error": safe,
                            "pod_id": selected_training["id"],
                            "capacity_preflight": capacity,
                        },
                    },
                    compute_experience={
                        "version_id": parent["id"],
                        "gpu": declared_training["gpu"],
                        "vendor": declared_training["vendor"],
                        "runtime": declared_training["runtime"],
                        "price_per_hour": declared_training["price_per_hour"],
                        "configuration": configuration,
                        "metrics": {"errors": [safe]},
                        "quality_result": {"status": "failed", "passed": False},
                        "action_taken": "QLoRA training failed",
                        "recommendation": "inspect failure before retrying",
                        "approval": {"compute": True},
                        "measurement_state": "failed",
                        "verified": False,
                    },
                )

            _watch(job, completed, failed)
            return {"run": pending, "job": job.public()}
        except (WorkspaceRegistryError, runner.JobError) as error:
            handle_error(error)

    @router.post("/workspaces/{workspace_id}/optimize")
    def optimize_workspace(workspace_id: str, req: OptimizeWorkspaceRequest):
        try:
            _require_approved(approved=req.approved)
            reconcile()
            workspace = registry.get_workspace(workspace_id, public=False)
            if req.candidate != "prefix_cache":
                raise WorkspaceRegistryError(
                    "prefix_cache is the only default MVP candidate; torch.compile requires an explicit separate experiment"
                )
            if req.deployment_id:
                selected = _deployment(workspace, req.deployment_id)
                live = selected if selected["status"] == "live" else None
            else:
                live = next(
                    (
                        item
                        for item in reversed(workspace["deployments"])
                        if item["status"] == "live" and not item.get("dry_run")
                    ),
                    None,
                )
                selected = live
            version_id = (
                selected["version_id"] if selected is not None else workspace["versions"][0]["id"]
            )
            selected_version = _version(workspace, version_id)
            optimization_pod, declared_optimization = _resolve_fleet_role(
                req.pod_id,
                expected_name="gpushare-probe-4090",
            )
            if "4090" not in str(optimization_pod.get("gpu", "")):
                raise WorkspaceRegistryError(
                    "prefix-cache optimization requires an actual RTX 4090 GPU"
                )
            configuration = {
                "candidate": "prefix_cache",
                "long_context": True,
                "source_deployment_id": selected.get("id") if selected else None,
                "source_context_mode": "standard",
                "target_context_mode": "shared_policy_v1",
                "fixed_prompts": True,
                "fixed_output_limit": req.output_token_limit,
                "torch_compile": False,
                "pod": optimization_pod["id"],
                "pod_name": optimization_pod["name"],
                "gpu": optimization_pod["gpu"],
                "version_type": selected_version["type"],
                "base_model_revision": _artifact_revision(selected_version),
            }
            if live is None:
                raise WorkspaceRegistryError(
                    "a live source deployment is required before optimizing"
                )
            optimization_busy, _busy_owner = busy_reason(
                kind="relay-prefix-cache-benchmark",
                params={"pod_id": optimization_pod["id"]},
                pod_id=optimization_pod["id"],
            )
            if optimization_busy:
                raise WorkspaceRegistryError(
                    f"optimization preflight failed: {optimization_busy}"
                )
            optimization_capacity = relay_jobs.cuda_capacity_preflight(optimization_pod)
            require_capacity(optimization_capacity, action="optimization")
            configuration["capacity_preflight"] = optimization_capacity
            # Planning happens only after the deterministic admission checks, so
            # a known-full GPU cannot incur an avoidable planner API call.
            autopilot = RelayAutopilot(AutopilotBackend(workspace_id))
            orchestration = autopilot.run(
                req.prompt,
                workspace_id=workspace_id,
                deployment_id=req.deployment_id or live.get("id"),
                approvals=ApprovalContext(compute=False, rollout=False),
            )
            version = _version(workspace, live["version_id"])
            artifact = version.get("artifact", {})
            job = relay_jobs.start_prefix_cache_benchmark(
                pod_id=optimization_pod["id"],
                model_id=live["version_id"],
                artifact_location=_artifact_location(version),
                version_type=version["type"],
                artifact_source_pod_id=artifact.get("pod_id"),
                local_artifact_location=artifact.get("local_location"),
                expected_artifact_manifest_sha256=artifact.get(
                    "artifact_manifest_sha256"
                ),
                expected_base_revision=_artifact_revision(version),
                repeats=req.repeats,
                max_new_tokens=req.output_token_limit,
            )
            pending = registry.record_run(
                workspace_id,
                "optimization",
                payload={
                    "status": "running",
                    "measurement_state": "pending",
                    "job_id": job.id,
                    "version_id": live["version_id"],
                    "configuration": configuration,
                    "job": job.public(),
                },
            )

            def completed(result: dict[str, Any]) -> None:
                candidate = result["candidate_metrics"]
                registry.commit_job_terminal(
                    workspace_id,
                    "optimization",
                    job_id=job.id,
                    terminal_payload={
                        "status": "passed" if result["passed"] else "rejected",
                        "measurement_state": "measured",
                        "version_id": live["version_id"],
                        "result": result,
                        "job": _terminal_job_projection(job),
                    },
                    compute_experience={
                        "version_id": live["version_id"],
                        "adapter_version_id": (
                            live["version_id"] if version["type"] == "lora_adapter" else None
                        ),
                        "input_tokens": int(candidate.get("logical_input_tokens") or 0),
                        "output_tokens": int(candidate["output_tokens"]),
                        "gpu": result["pod"]["gpu"],
                        "vendor": result["pod"]["vendor"],
                        "runtime": result["runtime"],
                        "region": result["pod"].get("datacenter"),
                        "price_per_hour": result["price_per_hour"],
                        "configuration": configuration,
                        "metrics": {
                            "ttft_ms": candidate["ttft_s"] * 1000,
                            "median_latency_ms": candidate["median_latency_s"] * 1000,
                            "p95_latency_ms": candidate["p95_latency_s"] * 1000,
                            "output_tokens_per_second": candidate[
                                "output_tokens_per_second"
                            ],
                            "baseline": result["baseline"],
                        },
                        "quality_result": result["quality"],
                        "action_taken": "prefix-cache candidate benchmarked",
                        "recommendation": result["recommendation"],
                        "approval": {"compute": True, "rollout": None},
                        "measurement_state": "measured",
                        "verified": True,
                    },
                )

            def failed(message: str) -> None:
                safe = runner.public_error_message(message)
                registry.commit_job_terminal(
                    workspace_id,
                    "optimization",
                    job_id=job.id,
                    terminal_payload={
                        "status": "failed",
                        "measurement_state": "failed",
                        "version_id": live["version_id"],
                        "error": safe,
                        "job": _terminal_job_projection(job, safe),
                        "failure_metadata": {
                            "phase": job.stage,
                            "progress": job.progress,
                            "cancelled": bool(job.cancel_requested),
                            "error": safe,
                            "pod_id": optimization_pod["id"],
                            "capacity_preflight": optimization_capacity,
                        },
                    },
                    compute_experience={
                        "version_id": live["version_id"],
                        "adapter_version_id": (
                            live["version_id"] if version["type"] == "lora_adapter" else None
                        ),
                        "gpu": declared_optimization["gpu"],
                        "vendor": declared_optimization["vendor"],
                        "runtime": declared_optimization["runtime"],
                        "price_per_hour": declared_optimization["price_per_hour"],
                        "configuration": configuration,
                        "metrics": {"errors": [safe]},
                        "quality_result": {"status": "failed", "passed": False},
                        "action_taken": "prefix-cache experiment failed",
                        "recommendation": "do not roll out",
                        "approval": {"compute": True, "rollout": None},
                        "measurement_state": "failed",
                        "verified": False,
                    },
                )

            _watch(job, completed, failed)
            return {
                "run": pending,
                "job": job.public(),
                "autopilot": orchestration.to_dict(),
            }
        except (WorkspaceRegistryError, runner.JobError) as error:
            handle_error(error)

    @router.post("/workspaces/{workspace_id}/migrate")
    def migrate_workspace(workspace_id: str, req: MigrateWorkspaceRequest):
        try:
            _require_approved(approved=req.approved)
            reconcile()
            workspace = registry.get_workspace(workspace_id, public=False)
            source = _deployment(workspace, req.source_deployment_id)
            version = _version(workspace, source["version_id"])
            if source.get("vendor") != "nvidia":
                raise WorkspaceRegistryError("migration source must be a CUDA deployment")
            target_pod, declared_target = _resolve_fleet_role(
                req.target_pod_id,
                expected_name="gpushare-amd-mi300x",
            )
            behavior_contract = relay_jobs.migration_behavior_contract(version["type"])
            behavior_emoji = (
                _behavior_emoji(version) if version["type"] == "lora_adapter" else None
            )
            configuration = {
                "source_vendor": source["vendor"],
                "target_vendor": "amd",
                "target_runtime": "rocm",
                "target_pod": target_pod["id"],
                "target_pod_name": target_pod["name"],
                "target_gpu": target_pod["gpu"],
                "same_artifact": True,
                "eval_n": req.eval_n,
                "version_type": version["type"],
                "base_model_revision": _artifact_revision(version),
                "behavior_contract": behavior_contract,
            }
            if source["status"] != "live":
                raise WorkspaceRegistryError("migration source deployment must be live")
            migration_busy, _busy_owner = busy_reason(
                kind="relay-cuda-rocm-migration",
                params={
                    "source_pod_id": source["pod_id"],
                    "target_pod_id": target_pod["id"],
                },
                pod_id=target_pod["id"],
                allow_deployment_ids={source["id"]},
            )
            if migration_busy:
                raise WorkspaceRegistryError(
                    f"migration preflight failed: {migration_busy}"
                )
            job = relay_jobs.start_chip_migration(
                source_pod_id=source["pod_id"],
                target_pod_id=target_pod["id"],
                model_id=version["id"],
                artifact_location=_artifact_location(version),
                version_type=version["type"],
                local_artifact_location=version.get("artifact", {}).get("local_location"),
                expected_artifact_manifest_sha256=version.get("artifact", {}).get(
                    "artifact_manifest_sha256"
                ),
                expected_base_revision=_artifact_revision(version),
                emoji=behavior_emoji,
                eval_n=req.eval_n,
            )
            pending = registry.record_run(
                workspace_id,
                "migration",
                payload={
                    "status": "running",
                    "measurement_state": "pending",
                    "job_id": job.id,
                    "version_id": version["id"],
                    "configuration": configuration,
                    "job": job.public(),
                },
            )

            def completed(result: dict[str, Any]) -> None:
                candidate = result["candidate"]
                comparisons = result.get("comparisons", {})
                portability = result.get("portability", {})
                composite_quality_passed = bool(
                    result.get("quality", {}).get("passed")
                    and portability.get("status") == "passed"
                    and comparisons.get("json_validity") == "passed"
                    and comparisons.get("exact_output_preservation") == "passed"
                )
                quality_result = {
                    **result.get("quality", {}),
                    "status": "passed" if composite_quality_passed else "rejected",
                    "passed": composite_quality_passed,
                    "portability": portability.get("status"),
                    "json_validity": comparisons.get("json_validity"),
                    "exact_output_preservation": comparisons.get("exact_output_preservation"),
                }
                registry.commit_job_terminal(
                    workspace_id,
                    "migration",
                    job_id=job.id,
                    terminal_payload={
                        "status": result["recommendation"],
                        "measurement_state": "measured",
                        "version_id": version["id"],
                        "result": result,
                        "job": _terminal_job_projection(job),
                    },
                    compute_experience={
                        "version_id": version["id"],
                        "adapter_version_id": (
                            version["id"] if version["type"] == "lora_adapter" else None
                        ),
                        "input_tokens": int(candidate["token_counts"]["input"]),
                        "output_tokens": int(candidate["token_counts"]["output"]),
                        "gpu": candidate["pod"]["gpu"],
                        "vendor": candidate["pod"]["vendor"],
                        "runtime": candidate["runtime"],
                        "region": candidate["pod"].get("datacenter"),
                        "price_per_hour": candidate["price_per_hour"],
                        "configuration": configuration,
                        "metrics": {
                            "ttft_ms": candidate["metrics"]["ttft_s"] * 1000,
                            "median_latency_ms": candidate["metrics"]["median_latency_s"] * 1000,
                            "p95_latency_ms": candidate["metrics"]["p95_latency_s"] * 1000,
                            "output_tokens_per_second": candidate["metrics"][
                                "output_tokens_per_second"
                            ],
                            "peak_vram_gb": candidate["metrics"].get("peak_vram_gb"),
                            "errors": candidate["metrics"].get("errors", 0),
                            "source_cost_per_1k_output_tokens_usd": comparisons.get(
                                "source_cost_per_1k_output_tokens_usd"
                            ),
                            "target_cost_per_1k_output_tokens_usd": comparisons.get(
                                "target_cost_per_1k_output_tokens_usd"
                            ),
                        },
                        "quality_result": quality_result,
                        "action_taken": "CUDA to ROCm candidate verified",
                        "recommendation": result["recommendation"],
                        "approval": {"compute": True, "traffic": None},
                        "measurement_state": "measured",
                        "verified": True,
                    },
                )

            def failed(message: str) -> None:
                safe = runner.public_error_message(message)
                registry.commit_job_terminal(
                    workspace_id,
                    "migration",
                    job_id=job.id,
                    terminal_payload={
                        "status": "failed",
                        "measurement_state": "failed",
                        "version_id": version["id"],
                        "error": safe,
                        "job": _terminal_job_projection(job, safe),
                        "failure_metadata": {
                            "phase": job.stage,
                            "progress": job.progress,
                            "cancelled": bool(job.cancel_requested),
                            "error": safe,
                            "source_pod_id": source["pod_id"],
                            "target_pod_id": target_pod["id"],
                        },
                    },
                    compute_experience={
                        "version_id": version["id"],
                        "adapter_version_id": (
                            version["id"] if version["type"] == "lora_adapter" else None
                        ),
                        "gpu": declared_target["gpu"],
                        "vendor": declared_target["vendor"],
                        "runtime": declared_target["runtime"],
                        "price_per_hour": declared_target["price_per_hour"],
                        "configuration": configuration,
                        "metrics": {"errors": [safe]},
                        "quality_result": {"status": "failed", "passed": False},
                        "action_taken": "CUDA to ROCm migration failed",
                        "recommendation": "do not migrate",
                        "approval": {"compute": True, "traffic": None},
                        "measurement_state": "failed",
                        "verified": False,
                    },
                )

            _watch(job, completed, failed)
            return {"run": pending, "job": job.public()}
        except (WorkspaceRegistryError, runner.JobError) as error:
            handle_error(error)

    class AutopilotBackend:
        def __init__(self, workspace_id: str):
            self.workspace_id = workspace_id

        def get_workspace_metrics(self, **_arguments):
            return {
                "records": registry.verified_records(self.workspace_id),
                "recommendations": registry.recommendations(self.workspace_id),
            }

        def benchmark_deployment(self, **arguments):
            reconcile()
            workspace = registry.get_workspace(self.workspace_id, public=False)
            deployment_id = arguments.get("deployment_id")
            if deployment_id:
                deployment = _deployment(workspace, deployment_id)
            else:
                deployment = next(
                    (
                        item
                        for item in reversed(workspace["deployments"])
                        if item["status"] == "live" and not item.get("dry_run")
                    ),
                    None,
                )
            if deployment is None or deployment["status"] != "live":
                raise runner.JobError("an exact live deployment is required for benchmarking")
            version = _version(workspace, deployment["version_id"])
            artifact = version.get("artifact", {})
            return relay_jobs.start_prefix_cache_benchmark(
                pod_id="gpushare-probe-4090",
                model_id=deployment["version_id"],
                artifact_location=_artifact_location(version),
                version_type=version["type"],
                artifact_source_pod_id=artifact.get("pod_id"),
                local_artifact_location=artifact.get("local_location"),
                expected_artifact_manifest_sha256=artifact.get(
                    "artifact_manifest_sha256"
                ),
                expected_base_revision=_artifact_revision(version),
                max_new_tokens=int(arguments.get("output_token_limit", 48)),
            ).public()

        def create_optimization_candidate(self, **arguments):
            if arguments.get("candidate_kind") != "prefix_cache":
                raise runner.JobError("only prefix_cache is reviewed for the default MVP path")
            return {"candidate": "prefix_cache", "created": True, "applied": False}

        def run_quality_gate(self, **arguments):
            candidate = runner.JOBS.get(arguments["candidate_run_id"])
            if candidate.status != "complete" or not candidate.result:
                return {"status": "pending", "passed": False}
            return candidate.result["quality"]

        def propose_rollout(self, **arguments):
            return {
                "status": "proposed",
                "candidate_id": arguments["candidate_id"],
                "traffic_switched": False,
                "requires_explicit_approval": True,
            }

        def rollback_candidate(self, **arguments):
            # There is no post-rollout monitor in this MVP, so no persisted
            # failed-applied candidate can authorize an automatic rollback.
            # Never let an LLM-authored candidate ID mutate the live runtime.
            return {
                "status": "not_applied",
                "candidate_id": arguments.get("candidate_id"),
                "rolled_back": False,
                "detail": "no deterministically verified post-rollout failure exists",
            }

    @router.post("/workspaces/{workspace_id}/autopilot")
    def run_autopilot(workspace_id: str, req: AutopilotRequest):
        try:
            reconcile()
            registry.get_workspace(workspace_id)
            autopilot = RelayAutopilot(AutopilotBackend(workspace_id))
            result = autopilot.run(
                req.request,
                workspace_id=workspace_id,
                deployment_id=req.deployment_id,
                # This endpoint is planning-only until agent-started jobs can
                # be persisted and bound to immutable workspace candidates.
                # Real compute continues through /optimize, whose deterministic
                # route owns the run record, watcher, quality gate, and approval.
                approvals=ApprovalContext(compute=False, rollout=False),
            )
            return {**result.to_dict(), "execution_scope": "plan_only"}
        except (WorkspaceRegistryError, runner.JobError) as error:
            handle_error(error)

    def _approve_workspace_action(workspace_id: str, req: ApprovalRequest):
        try:
            reconcile()
            workspace = registry.get_workspace(workspace_id, public=False)
            decision = req.decision.lower()
            if decision not in {"approve", "reject"}:
                raise WorkspaceRegistryError("decision must be approve or reject")
            run_lists = {
                "optimization": workspace["optimization_runs"],
                "migration": workspace["migration_runs"],
            }
            if req.run_type not in run_lists:
                raise WorkspaceRegistryError("approval run_type must be optimization or migration")
            candidate_run = next(
                (item for item in run_lists[req.run_type] if item["id"] == req.run_id), None
            )
            if candidate_run is None:
                raise WorkspaceRegistryError("approval must reference a run in this workspace")
            if candidate_run.get("dry_run") or candidate_run.get("measurement_state") != "measured":
                raise WorkspaceRegistryError("only a completed measured candidate can be decided")
            if any(
                item.get("approval", {}).get("candidate_run_id") == req.run_id
                for item in run_lists[req.run_type]
            ):
                raise WorkspaceRegistryError("this candidate already has a recorded user decision")

            approval_value = {
                "decision": decision,
                "explicit": True,
                "candidate_run_id": req.run_id,
            }
            compute_record_id = candidate_run.get("compute_record_id")

            if req.run_type == "optimization":
                result = candidate_run.get("result", {})
                if decision == "approve" and (
                    candidate_run.get("measurement_state") != "measured"
                    or not result.get("passed")
                    or result.get("rollout") != "awaiting_user_approval"
                ):
                    raise WorkspaceRegistryError("only a passed measured candidate can roll out")
                version_id = candidate_run["version_id"]
                if decision == "approve":
                    version = _version(workspace, version_id)
                    expected_revision = _artifact_revision(version)
                    expected_prompt = _prompt_template(version)
                    expected_identity = result.get("protocol", {}).get("prefix_identity_sha256")
                    if not isinstance(expected_identity, str) or not expected_identity:
                        raise WorkspaceRegistryError(
                            "measured candidate has no immutable prefix contract identity"
                        )
                    source_deployment_id = candidate_run.get("configuration", {}).get(
                        "source_deployment_id"
                    )
                    source_deployment = next(
                        (
                            item
                            for item in reversed(workspace["deployments"])
                            if item["id"] == source_deployment_id
                            and item["version_id"] == version_id
                            and item["status"] == "live"
                            and not item.get("dry_run")
                        ),
                        None,
                    )
                    if source_deployment is None:
                        raise WorkspaceRegistryError(
                            "the source deployment must still be live before rollout"
                        )
                    active_source = runner.serving()
                    if (
                        active_source.get("running") is not True
                        or active_source.get("stale") is True
                        or active_source.get("model_id") != version_id
                        or active_source.get("pod_id") != source_deployment["pod_id"]
                        or active_source.get("model_revision") != expected_revision
                        or active_source.get("base_model")
                        != relay_jobs.QWEN_4B_MODEL
                        or active_source.get("artifact_manifest_sha256")
                        != version.get("artifact", {}).get(
                            "artifact_manifest_sha256"
                        )
                        or active_source.get("prompt_template") != expected_prompt
                    ):
                        raise WorkspaceRegistryError(
                            "the measured source deployment is no longer the exact active version"
                        )
                    target_pod = candidate_run.get("configuration", {}).get(
                        "pod", "gpushare-probe-4090"
                    )
                    launched = launch_deployment(workspace_id, version_id, target_pod)
                    rollout_job = runner.JOBS.get(launched["job"]["id"])

                    def source_is_exactly_live() -> bool:
                        active = runner.serving()
                        return bool(
                            active.get("running") is True
                            and active.get("stale") is not True
                            and active.get("model_id") == source_deployment["version_id"]
                            and active.get("pod_id") == source_deployment["pod_id"]
                            and active.get("model_revision") == expected_revision
                            and active.get("base_model") == relay_jobs.QWEN_4B_MODEL
                            and active.get("artifact_manifest_sha256")
                            == version.get("artifact", {}).get(
                                "artifact_manifest_sha256"
                            )
                            and active.get("prompt_template") == expected_prompt
                        )

                    def begin_verified_restore(reason: str) -> dict[str, Any]:
                        """Restore asynchronously and claim success only after exact health."""

                        if source_is_exactly_live():
                            return {
                                "status": "source_preserved",
                                "rolled_back": False,
                                "source_preserved": True,
                                "traffic_switched": False,
                            }
                        try:
                            restored = launch_deployment(
                                workspace_id,
                                source_deployment["version_id"],
                                source_deployment["pod_id"],
                            )
                            restore_job = runner.JOBS.get(restored["job"]["id"])
                        except Exception as restore_error:  # noqa: BLE001
                            return {
                                "status": "manual_intervention_required",
                                "rolled_back": False,
                                "traffic_switched": True,
                                "error": runner.public_error_message(restore_error),
                            }

                        def restore_complete(_restored_result: dict[str, Any]) -> None:
                            if not source_is_exactly_live():
                                restore_failed("restored server failed the exact identity gate")
                                return
                            terminal = registry.record_run(
                                workspace_id,
                                "optimization",
                                payload={
                                    "status": "rolled_back",
                                    "measurement_state": "measured",
                                    "job_id": restore_job.id,
                                    "version_id": version_id,
                                    "approval": approval_value,
                                    "action": {
                                        "status": "rolled_back",
                                        "rolled_back": True,
                                        "restore_deployment_id": restored["deployment"]["id"],
                                        "traffic_switched": False,
                                        "reason": reason,
                                    },
                                },
                            )
                            if isinstance(compute_record_id, str):
                                registry.update_compute_experience(
                                    workspace_id,
                                    compute_record_id,
                                    approval={**approval_value, "result": "rolled_back"},
                                    result={
                                        "rollout": "failed",
                                        "rollback": "verified",
                                        "rollback_run_id": terminal["id"],
                                    },
                                    recommendation="candidate rolled back",
                                )

                        def restore_failed(message: str) -> None:
                            safe_restore = runner.public_error_message(message)
                            registry.record_run(
                                workspace_id,
                                "optimization",
                                payload={
                                    "status": "manual_intervention_required",
                                    "measurement_state": "failed",
                                    "job_id": restore_job.id,
                                    "version_id": version_id,
                                    "approval": approval_value,
                                    "error": safe_restore,
                                    "action": {
                                        "status": "manual_intervention_required",
                                        "rolled_back": False,
                                        "traffic_switched": True,
                                    },
                                },
                            )
                            if isinstance(compute_record_id, str):
                                registry.update_compute_experience(
                                    workspace_id,
                                    compute_record_id,
                                    approval={
                                        **approval_value,
                                        "result": "manual_intervention_required",
                                    },
                                    result={
                                        "rollout": "failed",
                                        "rollback": "failed",
                                        "error": safe_restore,
                                    },
                                    recommendation="manual intervention required",
                                )

                        _watch(restore_job, restore_complete, restore_failed)
                        return {
                            "status": "rollback_pending",
                            "rolled_back": False,
                            "restore_deployment_id": restored["deployment"]["id"],
                            "restore_job_id": restore_job.id,
                            "traffic_switched": True,
                        }

                    def rollout_complete(_result: dict[str, Any]) -> None:
                        target_pod_id = launched["deployment"]["pod_id"]
                        try:
                            applied = relay_jobs.apply_prefix_cache_rollout(
                                version_type=version["type"],
                                expected_model_id=version_id,
                                expected_pod_id=target_pod_id,
                                expected_revision=expected_revision,
                                expected_prompt_template=expected_prompt,
                                expected_identity_sha256=expected_identity,
                                expected_base_model=relay_jobs.QWEN_4B_MODEL,
                                expected_artifact_manifest_sha256=version.get(
                                    "artifact", {}
                                ).get("artifact_manifest_sha256"),
                            )
                        except Exception as error:  # noqa: BLE001 - persist rollout failure
                            safe = runner.public_error_message(error)
                            try:
                                relay_jobs.rollback_prefix_cache(
                                    expected_model_id=version_id,
                                    expected_pod_id=target_pod_id,
                                    expected_revision=expected_revision,
                                    expected_prompt_template=expected_prompt,
                                    expected_base_model=relay_jobs.QWEN_4B_MODEL,
                                    expected_artifact_manifest_sha256=version.get(
                                        "artifact", {}
                                    ).get("artifact_manifest_sha256"),
                                )
                            except Exception:
                                # A partially built prefix is still isolated to
                                # the failed candidate; exact source restoration
                                # below remains the authoritative recovery gate.
                                pass
                            rollback = begin_verified_restore(safe)
                            registry.record_run(
                                workspace_id,
                                "optimization",
                                payload={
                                    "status": rollback["status"],
                                    "measurement_state": "failed",
                                    "job_id": rollout_job.id,
                                    "version_id": version_id,
                                    "approval": approval_value,
                                    "error": safe,
                                    "action": rollback,
                                },
                            )
                            if isinstance(compute_record_id, str):
                                registry.update_compute_experience(
                                    workspace_id,
                                    compute_record_id,
                                    approval={**approval_value, "result": "rollout_failed"},
                                    result={"rollout": "failed", "rollback": rollback},
                                    recommendation="rollback candidate",
                                )
                            return
                        target_deployment = launched["deployment"]
                        target_metrics = {
                            **target_deployment.get("metrics", {}),
                            "shared_context": {
                                "status": "live",
                                "context_mode": "shared_policy_v1",
                                "prefix_identity_sha256": applied[
                                    "prefix_identity_sha256"
                                ],
                                "context_contract_sha256": applied[
                                    "prefix_identity_sha256"
                                ],
                                "quality": applied["quality"],
                            },
                        }
                        registry.update_deployment(
                            workspace_id,
                            target_deployment["id"],
                            status="live",
                            metrics=target_metrics,
                        )
                        completed_rollout = registry.record_run(
                            workspace_id,
                            "optimization",
                            payload={
                                "status": "approved",
                                "measurement_state": "measured",
                                "job_id": rollout_job.id,
                                "version_id": version_id,
                                "approval": approval_value,
                                "action": {**applied, "traffic_switched": True},
                            },
                        )
                        if isinstance(compute_record_id, str):
                            registry.update_compute_experience(
                                workspace_id,
                                compute_record_id,
                                approval={**approval_value, "result": "rolled_out"},
                                result={
                                    "rollout": "applied",
                                    "approval_run_id": completed_rollout["id"],
                                },
                            )

                    def rollout_failed(message: str) -> None:
                        safe = runner.public_error_message(message)
                        rollback = begin_verified_restore(safe)
                        registry.record_run(
                            workspace_id,
                            "optimization",
                            payload={
                                "status": rollback["status"],
                                "measurement_state": "failed",
                                "job_id": rollout_job.id,
                                "version_id": version_id,
                                "approval": approval_value,
                                "error": safe,
                                "action": rollback,
                            },
                        )
                        if isinstance(compute_record_id, str):
                            registry.update_compute_experience(
                                workspace_id,
                                compute_record_id,
                                approval={**approval_value, "result": rollback["status"]},
                                result={
                                    "rollout": "not_applied",
                                    "error": safe,
                                    "rollback": rollback,
                                },
                            )

                    _watch(rollout_job, rollout_complete, rollout_failed)
                    action = {
                        "rollout_started": True,
                        "traffic_switched": False,
                        "deployment": launched["deployment"]["id"],
                    }
                else:
                    action = {"candidate_rejected": True, "traffic_switched": False}
            elif req.run_type == "migration":
                # Approval is recorded as the user's decision; switching the
                # local serving tunnel is a separate deploy action so migration
                # can never silently move traffic.
                version_id = candidate_run["version_id"]
                result = candidate_run.get("result", {})
                if decision == "approve" and result.get("recommendation") != "Recommend":
                    raise WorkspaceRegistryError(
                        "only a measured Recommend migration candidate can be approved"
                    )
                action = {"traffic_switched": False, "decision": decision}
            if isinstance(compute_record_id, str):
                registry.update_compute_experience(
                    workspace_id,
                    compute_record_id,
                    approval={
                        **approval_value,
                        "result": (
                            "rollout_started"
                            if req.run_type == "optimization" and decision == "approve"
                            else "recorded"
                        ),
                    },
                )
            record = registry.record_run(
                workspace_id,
                req.run_type,
                payload={
                    "status": (
                        "rollout_started"
                        if req.run_type == "optimization" and decision == "approve"
                        else "approved"
                        if decision == "approve"
                        else "rejected_by_user"
                    ),
                    "measurement_state": "measured",
                    "job_id": candidate_run.get("job_id", req.run_id),
                    "version_id": version_id,
                    "approval": approval_value,
                    "action": action,
                },
            )
            return {"approval": record, "action": action}
        except (WorkspaceRegistryError, runner.JobError) as error:
            handle_error(error)

    @router.post("/workspaces/{workspace_id}/approvals")
    def approve_workspace_action(workspace_id: str, req: ApprovalRequest):
        # A browser busy flag is convenience, not a spending boundary. Serialize
        # the check+launch+decision record so two concurrent clicks cannot each
        # pass the "no prior decision" check and start two paid rollouts.
        key = (workspace_id, req.run_type, req.run_id)
        with approval_locks_guard:
            lock = approval_locks.setdefault(key, threading.Lock())
        with lock:
            return _approve_workspace_action(workspace_id, req)

    return router
