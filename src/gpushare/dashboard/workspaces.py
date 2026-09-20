"""Persistent Relay model workspaces and verified compute experience.

The dashboard used to keep a flat ``saved-models.json`` list.  This module is
the intentionally boring source of truth that replaces it: one atomically
written JSON document, deterministic schema validation, and no network or GPU
side effects.  Execution code may record facts here, but this module never
turns a plan into a RunPod action.

There are two views of the data:

* the on-disk view retains deployment endpoint/tunnel metadata needed by the
  local backend and an exact copy of every migrated legacy entry;
* the public view is safe to hand to the browser and omits those internal
  details (and recursively removes secret-shaped fields).

Recommendations are deliberately rule based.  Only records explicitly marked
both ``verified`` and ``measured`` can support one; dry-run, pending, estimated,
and LLM-authored claims are never treated as evidence.
"""

from __future__ import annotations

import contextlib
import copy
import hashlib
import json
import os
import re
import tempfile
import threading
import time
import uuid
from collections.abc import Callable, Iterator, Mapping
from datetime import UTC, datetime
from pathlib import Path
from typing import Any

SCHEMA_VERSION = 1
VERSION_TYPES = frozenset({"base", "lora_adapter"})
DEPLOYMENT_STATUSES = frozenset({"provisioning", "live", "failed", "stopped"})
RUN_TYPES = frozenset({"training", "serving", "optimization", "migration"})
MEASUREMENT_STATES = frozenset({"pending", "estimated", "measured", "failed"})

_WORKSPACE_LISTS = (
    "versions",
    "deployments",
    "training_runs",
    "optimization_runs",
    "migration_runs",
    "compute_experience_records",
)
_RUN_LIST = {
    "training": "training_runs",
    "optimization": "optimization_runs",
    "migration": "migration_runs",
}
_LORA_LEGACY_KINDS = frozenset(
    {
        "adapter",
        "lora",
        "qlora",
        "lora_adapter",
        "lora-adapter",
        "peft_adapter",
        "peft-adapter",
    }
)
_ID = re.compile(r"^[A-Za-z0-9][A-Za-z0-9_.-]{0,127}$")
_SECRET_KEY = re.compile(
    r"(?:^|_)(?:api_?key|authorization|bearer|credential|password|private_?key|secret|token)(?:$|_)",
    re.IGNORECASE,
)
_SECRET_VALUE = re.compile(
    r"(?:\bsk-[A-Za-z0-9_-]{16,}"
    r"|\brpa_[A-Za-z0-9]{16,}"
    r"|\bgh[pousr]_[A-Za-z0-9]{16,}"
    r"|\bhf_[A-Za-z0-9]{20,}"
    r"|\bAKIA[0-9A-Z]{16}"
    r"|\bglpat-[A-Za-z0-9_-]{16,}"
    r"|\b[A-Za-z0-9]{8}\.[A-Za-z0-9]{24,}"
    r"|-----BEGIN [^-\r\n]*PRIVATE KEY-----)"
)
_PUBLIC_INTERNAL_KEY = re.compile(
    r"(?:endpoint|tunnel|ssh|local_artifact|local_location|prompt_template)", re.IGNORECASE
)
_RAW_TEXT_KEYS = frozenset(
    {
        "answer",
        "answers",
        "body",
        "chat",
        "completion",
        "completions",
        "content",
        "context",
        "contexts",
        "conversation",
        "conversations",
        "document",
        "documents",
        "event",
        "events",
        "example",
        "examples",
        "failure",
        "failures",
        "generated_text",
        "input",
        "input_text",
        "instruction",
        "instructions",
        "message",
        "messages",
        "output",
        "output_text",
        "prompt",
        "prompts",
        "question",
        "questions",
        "raw_input",
        "raw_inputs",
        "raw_output",
        "raw_outputs",
        "raw_prompt",
        "raw_prompts",
        "record",
        "records",
        "request",
        "request_body",
        "requests",
        "response",
        "response_body",
        "responses",
        "sample",
        "samples",
        "sentence",
        "sentences",
        "source_text",
        "target_text",
        "text",
        "user_input",
        "user_output",
    }
)
_SAFE_TOKEN_COUNT_KEYS = frozenset({"token_count", "token_counts", "input_tokens", "output_tokens"})

_PATH_LOCKS: dict[str, threading.RLock] = {}
_PATH_LOCKS_GUARD = threading.Lock()
_MISSING = object()


class WorkspaceRegistryError(ValueError):
    """The requested registry mutation is invalid or stored state is corrupt."""


def _is_secret_key(value: str) -> bool:
    """Distinguish authentication tokens from required model token counters."""

    lowered = value.lower()
    token_measurement = lowered in _SAFE_TOKEN_COUNT_KEYS or lowered.endswith(
        ("_token_count", "_token_counts", "_tokens")
    )
    return not token_measurement and _SECRET_KEY.search(value) is not None


def _now() -> str:
    return datetime.now(UTC).isoformat()


def _json_copy(value: Any, label: str = "value") -> Any:
    """Return a detached, finite JSON value with a useful validation error."""

    try:
        encoded = json.dumps(value, ensure_ascii=False, allow_nan=False, sort_keys=True)
        return json.loads(encoded)
    except (TypeError, ValueError, UnicodeError, RecursionError) as exc:
        raise WorkspaceRegistryError(f"{label} must be finite JSON data") from exc


def _nonempty_text(value: Any, label: str, *, limit: int = 512) -> str:
    if not isinstance(value, str) or not value.strip() or "\x00" in value:
        raise WorkspaceRegistryError(f"{label} must be nonempty text without NUL bytes")
    result = value.strip()
    if len(result.encode("utf-8")) > limit:
        raise WorkspaceRegistryError(f"{label} exceeds its {limit}-byte limit")
    return result


def _identifier(value: Any, label: str) -> str:
    result = _nonempty_text(value, label, limit=128)
    if not _ID.fullmatch(result):
        raise WorkspaceRegistryError(
            f"{label} must contain only letters, digits, dot, underscore, or hyphen"
        )
    return result


def _assert_no_secrets(value: Any, path: str = "registry") -> None:
    """Refuse credentials rather than relying on public-view redaction alone."""

    if isinstance(value, Mapping):
        for key, child in value.items():
            key_text = str(key)
            if _is_secret_key(key_text):
                raise WorkspaceRegistryError(
                    f"secret field is not allowed in saved state: {path}.{key_text}"
                )
            _assert_no_secrets(child, f"{path}.{key_text}")
    elif isinstance(value, list):
        for index, child in enumerate(value):
            _assert_no_secrets(child, f"{path}[{index}]")
    elif isinstance(value, str) and _SECRET_VALUE.search(value):
        raise WorkspaceRegistryError(f"secret value is not allowed in saved state: {path}")


def _assert_no_raw_text(value: Any, path: str = "compute experience") -> None:
    """Compute memory stores counts and hashes, never private prompt content."""

    if isinstance(value, Mapping):
        for key, child in value.items():
            key_text = str(key).lower()
            # Count-shaped fields such as token_counts.{input,output} are
            # useful structured measurements.  Text, arrays, and objects
            # under these names are the privacy risk and are rejected.
            if key_text in _RAW_TEXT_KEYS and isinstance(child, (str, list, Mapping)):
                raise WorkspaceRegistryError(
                    f"raw prompt/output field is not allowed in compute memory: {path}.{key}"
                )
            _assert_no_raw_text(child, f"{path}.{key}")
    elif isinstance(value, list):
        for index, child in enumerate(value):
            _assert_no_raw_text(child, f"{path}[{index}]")


def _public_value(value: Any) -> Any:
    """Recursively remove backend-only deployment and credential fields."""

    if isinstance(value, dict):
        result: dict[str, Any] = {}
        for key, child in value.items():
            key_text = str(key)
            if (
                key_text.startswith("_")
                or key_text == "legacy_record"
                or _is_secret_key(key_text)
                or _PUBLIC_INTERNAL_KEY.search(key_text)
                or key_text.lower() in {"location", "path"}
                or (
                    key_text.lower() in _RAW_TEXT_KEYS
                    and isinstance(child, (str, list, Mapping))
                )
            ):
                continue
            result[key_text] = _public_value(child)
        return result
    if isinstance(value, list):
        return [_public_value(item) for item in value]
    if isinstance(value, str) and _SECRET_VALUE.search(value):
        return "***REDACTED***"
    if isinstance(value, str) and (
        value.strip().startswith(("/", "~/", "\\\\"))
        or re.match(r"^[A-Za-z]:[\\/]", value.strip())
    ):
        return "<internal-path>"
    return value


def _new_id(prefix: str) -> str:
    return f"{prefix}_{uuid.uuid4().hex}"


def _stable_id(prefix: str, *parts: str) -> str:
    digest = hashlib.sha256("\x00".join(parts).encode("utf-8")).hexdigest()[:24]
    return f"{prefix}_{digest}"


def _legacy_fingerprint(entry: dict[str, Any]) -> str:
    canonical = json.dumps(
        entry, ensure_ascii=False, sort_keys=True, separators=(",", ":"), allow_nan=False
    ).encode("utf-8")
    return hashlib.sha256(canonical).hexdigest()


def _quality_passed(value: Any) -> bool:
    if value is True:
        return True
    if isinstance(value, str):
        return value.lower() in {"pass", "passed", "ok", "accepted"}
    if isinstance(value, dict):
        for key in ("passed", "pass", "quality_passed"):
            if value.get(key) is True:
                return True
        return str(value.get("status", "")).lower() in {"pass", "passed", "ok", "accepted"}
    return False


class WorkspaceRegistry:
    """Thread/process-safe JSON workspace registry with atomic replacement.

    ``path`` is the new workspace document.  Unless disabled, a sibling
    ``saved-models.json`` is migrated on construction.  Migration is
    incremental and keyed by an exact-record fingerprint, so reopening the
    registry cannot duplicate workspaces or versions.
    """

    def __init__(
        self,
        path: Path | str,
        *,
        legacy_path: Path | str | None = None,
        auto_migrate: bool = True,
        clock: Callable[[], str] = _now,
    ) -> None:
        self.path = Path(path)
        self.legacy_path = (
            Path(legacy_path) if legacy_path is not None else self.path.parent / "saved-models.json"
        )
        self._clock = clock
        lock_key = str(self.path.resolve())
        with _PATH_LOCKS_GUARD:
            self._thread_lock = _PATH_LOCKS.setdefault(lock_key, threading.RLock())
        self.path.parent.mkdir(parents=True, exist_ok=True)
        self._lock_path = self.path.with_name(f".{self.path.name}.lock")

        with self._exclusive():
            if self.path.exists():
                self._read_unlocked()
            else:
                self._write_atomic(self._empty())
        if auto_migrate and self.legacy_path.exists():
            self.migrate_legacy()

    def _empty(self) -> dict[str, Any]:
        return {
            "schema_version": SCHEMA_VERSION,
            "workspaces": [],
            "migrations": {},
        }

    @contextlib.contextmanager
    def _exclusive(self) -> Iterator[None]:
        """Share an in-process lock and a short cross-process file lock."""

        with self._thread_lock:
            with self._lock_path.open("a+b") as handle:
                if os.name == "nt":
                    import msvcrt

                    handle.seek(0, os.SEEK_END)
                    if handle.tell() == 0:
                        handle.write(b"0")
                        handle.flush()
                    handle.seek(0)
                    msvcrt.locking(handle.fileno(), msvcrt.LK_LOCK, 1)
                    try:
                        yield
                    finally:
                        handle.seek(0)
                        msvcrt.locking(handle.fileno(), msvcrt.LK_UNLCK, 1)
                else:
                    import fcntl

                    fcntl.flock(handle.fileno(), fcntl.LOCK_EX)
                    try:
                        yield
                    finally:
                        fcntl.flock(handle.fileno(), fcntl.LOCK_UN)

    def _read_unlocked(self) -> dict[str, Any]:
        try:
            raw = json.loads(self.path.read_text(encoding="utf-8"))
        except (OSError, UnicodeError, json.JSONDecodeError, RecursionError) as exc:
            raise WorkspaceRegistryError(f"cannot read workspace registry {self.path}") from exc
        self._validate(raw)
        return raw

    def _write_atomic(self, value: dict[str, Any]) -> None:
        self._validate(value)
        _assert_no_secrets(value)
        encoded = json.dumps(
            value, ensure_ascii=False, indent=2, sort_keys=True, allow_nan=False
        ).encode("utf-8")
        temporary: str | None = None
        try:
            fd, temporary = tempfile.mkstemp(
                prefix=".relay-workspaces-", suffix=".tmp", dir=self.path.parent
            )
            with os.fdopen(fd, "wb") as handle:
                handle.write(encoded)
                handle.flush()
                os.fsync(handle.fileno())
            for attempt in range(8):
                try:
                    os.replace(temporary, self.path)
                    temporary = None
                    break
                except PermissionError:
                    if attempt == 7:
                        raise
                    time.sleep(0.025 * (attempt + 1))
            # Persist the directory entry as well where the platform supports it.
            if os.name != "nt":
                with contextlib.suppress(OSError):
                    directory = os.open(self.path.parent, os.O_RDONLY)
                    try:
                        os.fsync(directory)
                    finally:
                        os.close(directory)
        finally:
            if temporary is not None:
                with contextlib.suppress(FileNotFoundError):
                    os.unlink(temporary)

    def _mutate(self, change: Callable[[dict[str, Any]], Any]) -> Any:
        with self._exclusive():
            data = self._read_unlocked()
            result = change(data)
            self._write_atomic(data)
            return _json_copy(result, "registry result")

    @staticmethod
    def _workspace(data: dict[str, Any], workspace_id: str) -> dict[str, Any]:
        for workspace in data["workspaces"]:
            if workspace["id"] == workspace_id:
                return workspace
        raise WorkspaceRegistryError(f"unknown workspace {workspace_id!r}")

    @staticmethod
    def _version(workspace: dict[str, Any], version_id: str) -> dict[str, Any]:
        for version in workspace["versions"]:
            if version["id"] == version_id:
                return version
        raise WorkspaceRegistryError(f"unknown version {version_id!r}")

    def _validate(self, data: Any) -> None:
        if not isinstance(data, dict) or data.get("schema_version") != SCHEMA_VERSION:
            raise WorkspaceRegistryError("unsupported workspace registry schema")
        if not isinstance(data.get("workspaces"), list) or not isinstance(
            data.get("migrations"), dict
        ):
            raise WorkspaceRegistryError("workspace registry has invalid root fields")
        workspace_ids: set[str] = set()
        for workspace in data["workspaces"]:
            if not isinstance(workspace, dict):
                raise WorkspaceRegistryError("workspace entries must be objects")
            workspace_id = _identifier(workspace.get("id"), "workspace id")
            if workspace_id in workspace_ids:
                raise WorkspaceRegistryError(f"duplicate workspace id {workspace_id!r}")
            workspace_ids.add(workspace_id)
            _nonempty_text(workspace.get("name"), "workspace name")
            _nonempty_text(workspace.get("base_model"), "base model", limit=1_024)
            _nonempty_text(workspace.get("created_at"), "workspace created_at", limit=128)
            for field in _WORKSPACE_LISTS:
                if not isinstance(workspace.get(field), list):
                    raise WorkspaceRegistryError(
                        f"workspace {workspace_id!r} is missing list {field!r}"
                    )

            version_ids: set[str] = set()
            for version in workspace["versions"]:
                if not isinstance(version, dict):
                    raise WorkspaceRegistryError("versions must be objects")
                version_id = _identifier(version.get("id"), "version id")
                if version_id in version_ids:
                    raise WorkspaceRegistryError(f"duplicate version id {version_id!r}")
                version_ids.add(version_id)
                _nonempty_text(version.get("name"), "version name")
                if version.get("type") not in VERSION_TYPES:
                    raise WorkspaceRegistryError("version type must be base or lora_adapter")
                if not isinstance(version.get("artifact"), dict):
                    raise WorkspaceRegistryError("version artifact must be an object")
                if not isinstance(version.get("evaluation"), dict):
                    raise WorkspaceRegistryError("version evaluation must be an object")
                _nonempty_text(version.get("created_at"), "version created_at", limit=128)
            for version in workspace["versions"]:
                parent = version.get("parent_version_id")
                if parent is not None and parent not in version_ids:
                    raise WorkspaceRegistryError(f"version parent {parent!r} does not exist")
                if version["type"] == "base" and parent is not None:
                    raise WorkspaceRegistryError("a base version cannot have a parent")
                if version["type"] == "lora_adapter" and parent is None:
                    raise WorkspaceRegistryError("a LoRA adapter version requires a parent")

            deployment_ids: set[str] = set()
            for deployment in workspace["deployments"]:
                if not isinstance(deployment, dict):
                    raise WorkspaceRegistryError("deployments must be objects")
                deployment_id = _identifier(deployment.get("id"), "deployment id")
                if deployment_id in deployment_ids:
                    raise WorkspaceRegistryError(f"duplicate deployment id {deployment_id!r}")
                deployment_ids.add(deployment_id)
                if deployment.get("version_id") not in version_ids:
                    raise WorkspaceRegistryError("deployment references an unknown version")
                if deployment.get("status") not in DEPLOYMENT_STATUSES:
                    raise WorkspaceRegistryError("invalid deployment status")
                for field in ("gpu", "vendor", "runtime"):
                    _nonempty_text(deployment.get(field), f"deployment {field}")
                if not isinstance(deployment.get("metrics"), dict):
                    raise WorkspaceRegistryError("deployment metrics must be an object")
                if deployment.get("job_id") is not None:
                    _identifier(deployment["job_id"], "deployment job id")

            record_ids: set[str] = set()
            for record in workspace["compute_experience_records"]:
                if not isinstance(record, dict):
                    raise WorkspaceRegistryError("compute experience records must be objects")
                record_id = _identifier(record.get("id"), "compute record id")
                if record_id in record_ids:
                    raise WorkspaceRegistryError(f"duplicate compute record id {record_id!r}")
                record_ids.add(record_id)
                if record.get("run_type") not in RUN_TYPES:
                    raise WorkspaceRegistryError("invalid compute record run_type")
                if record.get("measurement_state") not in MEASUREMENT_STATES:
                    raise WorkspaceRegistryError("invalid compute record measurement_state")
                if (
                    type(record.get("verified")) is not bool
                    or type(record.get("dry_run")) is not bool
                ):
                    raise WorkspaceRegistryError("compute verified/dry_run flags must be booleans")
                if record["verified"] and (
                    record["measurement_state"] != "measured" or record["dry_run"]
                ):
                    raise WorkspaceRegistryError(
                        "only measured, non-dry-run compute records may be verified"
                    )
                _assert_no_raw_text(record)

        _assert_no_secrets(data)

    def raw_snapshot(self) -> dict[str, Any]:
        """Return the complete backend state.  Never send this value to a browser."""

        with self._exclusive():
            return _json_copy(self._read_unlocked(), "workspace registry")

    def public_snapshot(self) -> dict[str, Any]:
        """Return browser-safe state with all backend connection details removed."""

        raw = self.raw_snapshot()
        raw.pop("migrations", None)
        return _public_value(raw)

    def list_workspaces(self, *, public: bool = True) -> list[dict[str, Any]]:
        snapshot = self.public_snapshot() if public else self.raw_snapshot()
        return snapshot["workspaces"]

    def get_workspace(self, workspace_id: str, *, public: bool = True) -> dict[str, Any]:
        snapshot = self.public_snapshot() if public else self.raw_snapshot()
        return _json_copy(self._workspace(snapshot, workspace_id), "workspace")

    def create_workspace(
        self,
        *,
        name: str,
        base_model: str,
        workspace_id: str | None = None,
        base_version_name: str | None = None,
        artifact: dict[str, Any] | str | None = None,
    ) -> dict[str, Any]:
        name = _nonempty_text(name, "workspace name")
        base_model = _nonempty_text(base_model, "base model", limit=1_024)
        workspace_id = _identifier(workspace_id or _new_id("ws"), "workspace id")
        version_id = _new_id("ver")
        now = self._clock()
        base_artifact = self._normalise_artifact(
            artifact if artifact is not None else {"location": base_model, "storage": "huggingface"}
        )

        def change(data: dict[str, Any]) -> dict[str, Any]:
            if any(item["id"] == workspace_id for item in data["workspaces"]):
                raise WorkspaceRegistryError(f"workspace id {workspace_id!r} already exists")
            if any(item["name"].casefold() == name.casefold() for item in data["workspaces"]):
                raise WorkspaceRegistryError(f"workspace name {name!r} already exists")
            workspace = {
                "id": workspace_id,
                "name": name,
                "base_model": base_model,
                "created_at": now,
                "versions": [
                    {
                        "id": version_id,
                        "name": base_version_name or f"{name}-base",
                        "parent_version_id": None,
                        "type": "base",
                        "artifact": base_artifact,
                        "evaluation": {},
                        "created_at": now,
                    }
                ],
                "deployments": [],
                "training_runs": [],
                "optimization_runs": [],
                "migration_runs": [],
                "compute_experience_records": [],
            }
            data["workspaces"].append(workspace)
            return _public_value(workspace)

        return self._mutate(change)

    @staticmethod
    def _normalise_artifact(artifact: dict[str, Any] | str | None) -> dict[str, Any]:
        if artifact is None:
            return {}
        if isinstance(artifact, str):
            return {"location": _nonempty_text(artifact, "artifact location", limit=4_096)}
        if not isinstance(artifact, dict):
            raise WorkspaceRegistryError("artifact must be a location string or JSON object")
        result = _json_copy(artifact, "artifact")
        _assert_no_secrets(result, "artifact")
        return result

    def _prepare_version_record(
        self,
        *,
        name: str,
        version_type: str,
        parent_version_id: str | None,
        artifact: dict[str, Any] | str | None,
        evaluation: dict[str, Any] | None = None,
        version_id: str | None = None,
    ) -> dict[str, Any]:
        """Validate and build a version without mutating the registry."""

        name = _nonempty_text(name, "version name")
        if version_type not in VERSION_TYPES:
            raise WorkspaceRegistryError("version type must be base or lora_adapter")
        version_id = _identifier(version_id or _new_id("ver"), "version id")
        if version_type == "lora_adapter" and parent_version_id is None:
            raise WorkspaceRegistryError("a LoRA adapter version requires a parent")
        if version_type == "base" and parent_version_id is not None:
            raise WorkspaceRegistryError("a base version cannot have a parent")
        clean_evaluation = _json_copy(evaluation or {}, "version evaluation")
        _assert_no_secrets(clean_evaluation, "version evaluation")
        return {
            "id": version_id,
            "name": name,
            "parent_version_id": parent_version_id,
            "type": version_type,
            "artifact": self._normalise_artifact(artifact),
            "evaluation": clean_evaluation,
            "created_at": self._clock(),
        }

    def add_version(
        self,
        workspace_id: str,
        *,
        name: str,
        version_type: str,
        parent_version_id: str | None,
        artifact: dict[str, Any] | str | None,
        evaluation: dict[str, Any] | None = None,
        version_id: str | None = None,
    ) -> dict[str, Any]:
        version = self._prepare_version_record(
            name=name,
            version_type=version_type,
            parent_version_id=parent_version_id,
            artifact=artifact,
            evaluation=evaluation,
            version_id=version_id,
        )

        def change(data: dict[str, Any]) -> dict[str, Any]:
            workspace = self._workspace(data, workspace_id)
            if any(item["id"] == version["id"] for item in workspace["versions"]):
                raise WorkspaceRegistryError(f"version id {version['id']!r} already exists")
            if any(
                item["name"].casefold() == version["name"].casefold()
                for item in workspace["versions"]
            ):
                raise WorkspaceRegistryError(f"version name {version['name']!r} already exists")
            if version["parent_version_id"] is not None:
                self._version(workspace, version["parent_version_id"])
            workspace["versions"].append(version)
            return _public_value(version)

        return self._mutate(change)

    def add_deployment(
        self,
        workspace_id: str,
        *,
        version_id: str,
        pod_id: str,
        gpu: str,
        vendor: str,
        runtime: str,
        status: str = "provisioning",
        endpoint: dict[str, Any] | None = None,
        tunnel: dict[str, Any] | None = None,
        metrics: dict[str, Any] | None = None,
        deployment_id: str | None = None,
        dry_run: bool = False,
    ) -> dict[str, Any]:
        if status not in DEPLOYMENT_STATUSES:
            raise WorkspaceRegistryError("invalid deployment status")
        deployment_id = _identifier(deployment_id or _new_id("dep"), "deployment id")
        clean_endpoint = _json_copy(endpoint or {}, "endpoint metadata")
        clean_tunnel = _json_copy(tunnel or {}, "tunnel metadata")
        clean_metrics = _json_copy(metrics or {}, "deployment metrics")
        for label, value in (
            ("endpoint metadata", clean_endpoint),
            ("tunnel metadata", clean_tunnel),
            ("deployment metrics", clean_metrics),
        ):
            _assert_no_secrets(value, label)

        def change(data: dict[str, Any]) -> dict[str, Any]:
            workspace = self._workspace(data, workspace_id)
            self._version(workspace, version_id)
            deployment = {
                "id": deployment_id,
                "version_id": version_id,
                "pod_id": _nonempty_text(pod_id, "pod id"),
                "gpu": _nonempty_text(gpu, "GPU"),
                "vendor": _nonempty_text(vendor, "vendor"),
                "runtime": _nonempty_text(runtime, "runtime"),
                "endpoint": clean_endpoint,
                "tunnel": clean_tunnel,
                "status": status,
                "metrics": clean_metrics,
                "dry_run": bool(dry_run),
                "created_at": self._clock(),
                "updated_at": self._clock(),
            }
            workspace["deployments"].append(deployment)
            return _public_value(deployment)

        return self._mutate(change)

    def update_deployment(
        self,
        workspace_id: str,
        deployment_id: str,
        *,
        status: str | None = None,
        metrics: dict[str, Any] | None = None,
        endpoint: dict[str, Any] | object = _MISSING,
        tunnel: dict[str, Any] | object = _MISSING,
        job_id: str | None | object = _MISSING,
    ) -> dict[str, Any]:
        if status is not None and status not in DEPLOYMENT_STATUSES:
            raise WorkspaceRegistryError("invalid deployment status")
        clean_metrics = _json_copy(metrics, "deployment metrics") if metrics is not None else None
        clean_endpoint = (
            _json_copy(endpoint, "endpoint metadata") if endpoint is not _MISSING else _MISSING
        )
        clean_tunnel = _json_copy(tunnel, "tunnel metadata") if tunnel is not _MISSING else _MISSING
        clean_job_id = (
            _identifier(job_id, "deployment job id") if isinstance(job_id, str) else job_id
        )
        if (
            clean_job_id is not _MISSING
            and clean_job_id is not None
            and not isinstance(clean_job_id, str)
        ):
            raise WorkspaceRegistryError("deployment job id must be text or null")
        for label, value in (
            ("deployment metrics", clean_metrics),
            ("endpoint metadata", clean_endpoint),
            ("tunnel metadata", clean_tunnel),
        ):
            if value is not None and value is not _MISSING:
                _assert_no_secrets(value, label)

        def change(data: dict[str, Any]) -> dict[str, Any]:
            workspace = self._workspace(data, workspace_id)
            deployment = next(
                (item for item in workspace["deployments"] if item["id"] == deployment_id), None
            )
            if deployment is None:
                raise WorkspaceRegistryError(f"unknown deployment {deployment_id!r}")
            if status is not None:
                deployment["status"] = status
            if clean_metrics is not None:
                deployment["metrics"] = clean_metrics
            if clean_endpoint is not _MISSING:
                deployment["endpoint"] = clean_endpoint
            if clean_tunnel is not _MISSING:
                deployment["tunnel"] = clean_tunnel
            if clean_job_id is not _MISSING:
                deployment["job_id"] = clean_job_id
            deployment["updated_at"] = self._clock()
            return _public_value(deployment)

        return self._mutate(change)

    @staticmethod
    def _job_state_map(job_states: Any) -> dict[str, dict[str, Any]]:
        """Normalise ``JobManager.states()`` or equivalent persisted snapshots.

        Reconciliation deliberately accepts already-observed state instead of
        importing the runner.  That keeps this persistence layer deterministic
        and makes a dashboard restart unable to trigger cloud work.
        """

        if job_states is None:
            return {}
        if isinstance(job_states, Mapping):
            items = job_states.items()
        elif isinstance(job_states, list):
            items = ((item.get("id"), item) for item in job_states if isinstance(item, Mapping))
        else:
            raise WorkspaceRegistryError("job_states must be a mapping or list of objects")

        normalised: dict[str, dict[str, Any]] = {}
        for key, value in items:
            if isinstance(value, str):
                value = {"status": value}
            if not isinstance(value, Mapping):
                raise WorkspaceRegistryError("each job state must be text or an object")
            job_id = value.get("id") or key
            if not isinstance(job_id, str) or not job_id:
                raise WorkspaceRegistryError("each job state requires an id")
            state = dict(value)
            state["id"] = job_id
            normalised[job_id] = state
        return normalised

    def reconcile_runtime(
        self,
        job_states: Mapping[str, Any] | list[dict[str, Any]] | None,
        active_serving: Mapping[str, Any] | None = None,
    ) -> dict[str, Any]:
        """Reconcile persisted product state after the local backend restarts.

        The supplied serving state is authoritative only when its health check
        says it is running and non-stale.  At most one matching, real
        deployment is restored to ``live``.  Provisioning deployments without
        a live server or a still-active matching job are failed rather than
        being left indefinitely chat-enabled-in-waiting.

        Likewise, a run marker remains ``running`` only while its exact job is
        queued/running.  Interrupted, terminal-without-persisted-result, and
        missing jobs become explicit failures.  An initial marker that already
        has a later terminal record for the same job becomes ``superseded``.
        No job is resumed and no external action occurs here.
        """

        jobs = self._job_state_map(job_states)
        serving = dict(active_serving or {})
        serving_is_live = serving.get("running") is True and serving.get("stale") is not True
        active_model_id = serving.get("model_id") or serving.get("version_id")
        active_pod_id = serving.get("pod_id")

        def job_for_deployment(deployment: dict[str, Any]) -> dict[str, Any] | None:
            job_id = deployment.get("job_id")
            if isinstance(job_id, str):
                return jobs.get(job_id)
            # Older registry rows predate deployment.job_id.  A trusted
            # internal job snapshot can still be matched without guessing by
            # both pod and workspace version.  These params are never part of
            # the browser-safe Job.public() representation.
            matches = []
            for state in jobs.values():
                params = state.get("params")
                if not isinstance(params, Mapping):
                    continue
                if params.get("model_id") == deployment.get("version_id") and params.get(
                    "pod_id"
                ) == deployment.get("pod_id"):
                    matches.append(state)
            return matches[0] if len(matches) == 1 else None

        def change(data: dict[str, Any]) -> dict[str, Any]:
            deployment_rows: list[tuple[dict[str, Any], dict[str, Any], dict[str, Any]]] = []
            for workspace in data["workspaces"]:
                for deployment in workspace["deployments"]:
                    if deployment.get("dry_run"):
                        continue
                    version = self._version(workspace, deployment["version_id"])
                    deployment_rows.append((workspace, deployment, version))

            matching: list[tuple[dict[str, Any], dict[str, Any], dict[str, Any]]] = []
            if serving_is_live:
                for row in deployment_rows:
                    workspace_row, deployment, version = row
                    artifact = version.get("artifact", {})
                    metrics = deployment.get("metrics", {})
                    expected_revision = artifact.get("base_model_revision") or artifact.get(
                        "revision"
                    )
                    expected_base_model = artifact.get("base_model") or workspace_row.get(
                        "base_model"
                    )
                    expected_manifest = artifact.get("artifact_manifest_sha256")
                    expected_prompt_hash = metrics.get("prompt_template_sha256")
                    # Versions created by the first workspace release could
                    # lose this backend-only hash when a public/redacted
                    # deployment return value was fed back into an update.
                    # Recover it only from the backend's remembered template
                    # when the live server independently reports the same hash
                    # and every other immutable identity field already agrees.
                    # Raw templates are never persisted by this repair.
                    active_prompt_hash = serving.get("prompt_template_sha256")
                    remembered_prompt = serving.get("prompt_template")
                    other_identity_matches = bool(
                        isinstance(expected_revision, str)
                        and expected_revision
                        and isinstance(expected_base_model, str)
                        and expected_base_model
                        and active_pod_id == deployment.get("pod_id")
                        and active_model_id == deployment.get("version_id")
                        and serving.get("model_revision") == expected_revision
                        and serving.get("base_model") == expected_base_model
                        and serving.get("artifact_manifest_sha256") == expected_manifest
                    )
                    if (
                        not expected_prompt_hash
                        and other_identity_matches
                        and isinstance(remembered_prompt, str)
                        and remembered_prompt
                    ):
                        recovered_prompt_hash = hashlib.sha256(
                            remembered_prompt.encode("utf-8")
                        ).hexdigest()
                        if active_prompt_hash == recovered_prompt_hash:
                            metrics["prompt_template_sha256"] = recovered_prompt_hash
                            expected_prompt_hash = recovered_prompt_hash
                    # New Relay deployments persist the complete identity
                    # before provisioning. Older rows that lack one of these
                    # contracts fail closed instead of being relabelled live
                    # from a coincidental pod/version pair.
                    identity_match = bool(
                        other_identity_matches
                        and isinstance(expected_prompt_hash, str)
                        and expected_prompt_hash
                        and active_prompt_hash == expected_prompt_hash
                    )
                    if identity_match:
                        matching.append(row)

            # If an older live row and a new provisioning attempt have the
            # same model+pod, health alone cannot distinguish their processes.
            # Retaining the already-live attribution is conservative; restore
            # a provisioning attempt when it is the only exact match.
            active_row = max(
                matching,
                key=lambda row: (
                    {"failed": 0, "stopped": 1, "provisioning": 2, "live": 3}.get(
                        row[1].get("status"), -1
                    ),
                    str(row[1].get("created_at", "")),
                    row[1]["id"],
                ),
                default=None,
            )

            restored: list[str] = []
            stopped: list[str] = []
            failed_deployments: list[str] = []
            unchanged_deployments: list[str] = []
            invalidated_shared_contexts: list[str] = []
            now = self._clock()
            active_deployment = active_row[1] if active_row is not None else None
            if active_deployment is not None:
                if active_deployment["status"] != "live":
                    active_deployment["status"] = "live"
                    active_deployment["updated_at"] = now
                    active_deployment["reconciliation"] = {
                        "status": "restored_live",
                        "at": now,
                    }
                    restored.append(active_deployment["id"])
                shared = active_deployment.get("metrics", {}).get("shared_context")
                if (
                    isinstance(shared, dict)
                    and shared.get("status") == "live"
                    and serving.get("prefix_identity_sha256")
                    != shared.get("prefix_identity_sha256")
                ):
                    active_deployment["metrics"]["shared_context"] = {
                        **shared,
                        "status": "failed",
                        "error": "the verified shared context is not resident",
                    }
                    active_deployment["updated_at"] = now
                    invalidated_shared_contexts.append(active_deployment["id"])

            # Chat gating reads the registry, so a historical ``live`` label
            # must be continuously backed by the exact healthy model+pod that
            # the local proxy reports.  A server that claims that same identity
            # but is dead/stale is a failed deployment; an unrelated old live
            # row is merely stopped.
            for _workspace, deployment, _version in deployment_rows:
                if deployment is active_deployment or deployment.get("status") != "live":
                    continue
                claimed_identity = active_model_id == deployment.get(
                    "version_id"
                ) and active_pod_id == deployment.get("pod_id")
                if claimed_identity and not serving_is_live:
                    deployment["status"] = "failed"
                    deployment["metrics"] = {
                        **deployment.get("metrics", {}),
                        "measurement_state": "failed",
                        "error": "the claimed serving process is unavailable or stale",
                    }
                    deployment["reconciliation"] = {
                        "status": "failed_unhealthy_server",
                        "at": now,
                    }
                    failed_deployments.append(deployment["id"])
                else:
                    deployment["status"] = "stopped"
                    deployment["reconciliation"] = {
                        "status": "not_the_active_server",
                        "active_deployment_id": (
                            active_deployment["id"] if active_deployment is not None else None
                        ),
                        "at": now,
                    }
                    stopped.append(deployment["id"])
                deployment["updated_at"] = now

            for _workspace, deployment, _version in deployment_rows:
                if deployment.get("status") != "provisioning":
                    continue
                if deployment is active_deployment:
                    continue
                job = job_for_deployment(deployment)
                job_status = str(job.get("status", "")) if job else "missing"
                # The deployment watcher can persist ``live`` or ``failed`` a
                # moment after its worker reaches a terminal state.  A job
                # created by this process still has that watcher; only a
                # disk-loaded, non-resident job is an orphan to reconcile.
                if (job and job.get("worker_resident") is True) or job_status in {
                    "queued",
                    "running",
                    "cancelling",
                }:
                    unchanged_deployments.append(deployment["id"])
                    continue
                deployment["status"] = "failed"
                deployment["updated_at"] = now
                reason = (
                    "serve job completed, but no matching active server was found"
                    if job_status == "complete"
                    else f"serve job is {job_status} after dashboard restart"
                )
                deployment["metrics"] = {
                    **deployment.get("metrics", {}),
                    "measurement_state": "failed",
                    "error": reason,
                }
                deployment["reconciliation"] = {
                    "status": "failed_stale",
                    "job_status": job_status,
                    "at": now,
                }
                failed_deployments.append(deployment["id"])

            failed_runs: list[str] = []
            superseded_runs: list[str] = []
            unchanged_runs: list[str] = []
            manual_intervention_runs: list[str] = []
            reconciliation_compute_records: list[str] = []

            def reconciliation_record(
                workspace: dict[str, Any],
                *,
                run_type: str,
                source_kind: str,
                source_id: str,
                version_id: str,
                job: Mapping[str, Any] | None,
                reason: str,
                manual_intervention: bool = False,
                placement: Mapping[str, Any] | None = None,
            ) -> str:
                """Append one atomic, idempotent failed Compute Memory row.

                Reconciliation has no callback-specific knowledge with which
                to reconstruct a successful result.  Recording a failure is
                therefore safer than promoting an uncommitted Job.result.
                The deterministic ID makes repeated startup/health passes a
                no-op instead of duplicating the failure in Activity.
                """

                digest = hashlib.sha256(
                    f"{workspace['id']}:{source_kind}:{source_id}".encode()
                ).hexdigest()[:24]
                record_id = f"exp_restart_{digest}"
                if any(
                    item.get("id") == record_id
                    for item in workspace["compute_experience_records"]
                ):
                    return record_id

                version = self._version(workspace, version_id)
                compute = dict(placement or {})
                record = {
                    "id": record_id,
                    "run_type": run_type,
                    "version_id": version_id,
                    "adapter_version_id": (
                        version_id if version.get("type") == "lora_adapter" else None
                    ),
                    "token_counts": {"input": 0, "output": 0},
                    "compute": {
                        "gpu": str(compute.get("gpu") or "unknown"),
                        "vendor": str(compute.get("vendor") or "unknown"),
                        "runtime": str(compute.get("runtime") or "unknown"),
                        "region": compute.get("region"),
                        "price_per_hour": compute.get("price_per_hour"),
                    },
                    "configuration": {
                        "reconciliation": "backend_restart",
                        "source_kind": source_kind,
                        "source_id": source_id,
                        "source_job_id": job.get("id") if job else None,
                    },
                    "metrics": {
                        "errors": [reason],
                        "job_finished_at": job.get("finished_at") if job else None,
                    },
                    "quality_result": {"status": "failed", "passed": False},
                    "action_taken": (
                        "manual intervention required after backend restart"
                        if manual_intervention
                        else "interrupted deployment reconciled as failed"
                        if run_type == "serving"
                        else "interrupted run reconciled as failed"
                    ),
                    "recommendation": (
                        "inspect serving state before retrying or changing traffic"
                        if manual_intervention
                        else "inspect the interrupted job before retrying"
                    ),
                    "approval": (
                        {"decision_state": "manual_intervention_required"}
                        if manual_intervention
                        else {}
                    ),
                    "measurement_state": "failed",
                    "verified": False,
                    "dry_run": False,
                    "created_at": now,
                }
                _assert_no_secrets(record, "restart reconciliation record")
                _assert_no_raw_text(record, "restart reconciliation record")
                workspace["compute_experience_records"].append(record)
                reconciliation_compute_records.append(record_id)
                return record_id

            # Deployment callbacks normally write a serving Compute Memory row.
            # If the backend dies between the remote job ending and that callback,
            # reconciliation owns the missing terminal evidence just as it does
            # for training/optimization/migration runs.
            failed_deployment_ids = set(failed_deployments)
            for workspace, deployment, _version in deployment_rows:
                if deployment["id"] not in failed_deployment_ids:
                    continue
                job = job_for_deployment(deployment)
                reason = str(
                    deployment.get("metrics", {}).get("error")
                    or "deployment became unavailable during backend restart"
                )
                memory_id = reconciliation_record(
                    workspace,
                    run_type="serving",
                    source_kind="deployment",
                    source_id=deployment["id"],
                    version_id=deployment["version_id"],
                    job=job,
                    reason=reason,
                    placement={
                        "gpu": deployment.get("gpu"),
                        "vendor": deployment.get("vendor"),
                        "runtime": deployment.get("runtime"),
                        "region": deployment.get("metrics", {}).get("region"),
                        "price_per_hour": deployment.get("metrics", {}).get(
                            "price_per_hour"
                        ),
                    },
                )
                deployment["reconciliation_compute_record_id"] = memory_id

            def relevant_job_states(
                workspace: dict[str, Any], run: Mapping[str, Any]
            ) -> list[dict[str, Any]]:
                job_ids: set[str] = set()
                job_id = run.get("job_id")
                if isinstance(job_id, str):
                    job_ids.add(job_id)
                action = run.get("action")
                if isinstance(action, Mapping):
                    for key in ("restore_job_id", "rollout_job_id"):
                        value = action.get(key)
                        if isinstance(value, str):
                            job_ids.add(value)
                    deployment_id = action.get("deployment")
                    if isinstance(deployment_id, str):
                        deployment = next(
                            (
                                item
                                for item in workspace["deployments"]
                                if item.get("id") == deployment_id
                            ),
                            None,
                        )
                        deployment_job_id = (
                            deployment.get("job_id") if deployment is not None else None
                        )
                        if isinstance(deployment_job_id, str):
                            job_ids.add(deployment_job_id)
                return [jobs[job_id] for job_id in job_ids if job_id in jobs]

            for workspace in data["workspaces"]:
                for run_type, list_name in _RUN_LIST.items():
                    runs = workspace[list_name]
                    terminal_job_ids = {
                        run.get("job_id")
                        for run in runs
                        if run.get("status")
                        not in {"running", "rollout_started", "rollback_pending"}
                        and isinstance(run.get("job_id"), str)
                    }
                    for run in runs:
                        status = run.get("status")
                        if status not in {"running", "rollout_started", "rollback_pending"} or run.get(
                            "dry_run"
                        ):
                            continue
                        related_jobs = relevant_job_states(workspace, run)
                        # A resident job means an in-process _watch callback is
                        # still the sole owner of the terminal registry write,
                        # even if the worker itself has just become complete.
                        if any(job.get("worker_resident") is True for job in related_jobs):
                            unchanged_runs.append(run["id"])
                            continue

                        if status in {"rollout_started", "rollback_pending"}:
                            reason = (
                                "rollout ownership was lost during backend restart"
                                if status == "rollout_started"
                                else "rollback verification was interrupted by backend restart"
                            )
                            run["status"] = "manual_intervention_required"
                            run["measurement_state"] = "failed"
                            run["error"] = reason
                            run["reconciliation"] = {
                                "status": "manual_intervention_required",
                                "previous_status": status,
                                "at": now,
                            }
                            version_id = run.get("version_id") or workspace["versions"][0]["id"]
                            memory_id = reconciliation_record(
                                workspace,
                                run_type=run_type,
                                source_kind="run",
                                source_id=run["id"],
                                version_id=version_id,
                                job=related_jobs[0] if related_jobs else None,
                                reason=reason,
                                manual_intervention=True,
                            )
                            run["reconciliation_compute_record_id"] = memory_id
                            manual_intervention_runs.append(run["id"])
                            continue

                        job_id = run.get("job_id")
                        if isinstance(job_id, str) and job_id in terminal_job_ids:
                            run["status"] = "superseded"
                            run["reconciliation"] = {
                                "status": "terminal_record_exists",
                                "at": now,
                            }
                            superseded_runs.append(run["id"])
                            continue
                        job = jobs.get(job_id) if isinstance(job_id, str) else None
                        job_status = str(job.get("status", "")) if job else "missing"
                        if job_status in {"queued", "running", "cancelling"}:
                            unchanged_runs.append(run["id"])
                            continue
                        run["status"] = "failed"
                        run["measurement_state"] = "failed"
                        if job_status == "complete":
                            reason = (
                                "job completed, but its result was not committed before restart"
                            )
                        else:
                            reason = f"job is {job_status} after dashboard restart"
                        run["error"] = reason
                        run["reconciliation"] = {
                            "status": "failed_stale",
                            "job_status": job_status,
                            "at": now,
                        }
                        version_id = run.get("version_id") or workspace["versions"][0]["id"]
                        memory_id = reconciliation_record(
                            workspace,
                            run_type=run_type,
                            source_kind="run",
                            source_id=run["id"],
                            version_id=version_id,
                            job=job,
                            reason=reason,
                        )
                        run["reconciliation_compute_record_id"] = memory_id
                        failed_runs.append(run["id"])

            return {
                "deployments": {
                    "restored_live": restored,
                    "stopped": stopped,
                    "failed": failed_deployments,
                    "unchanged": unchanged_deployments,
                    "shared_context_invalidated": invalidated_shared_contexts,
                },
                "runs": {
                    "failed": failed_runs,
                    "superseded": superseded_runs,
                    "unchanged": unchanged_runs,
                    "manual_intervention_required": manual_intervention_runs,
                    "compute_records_created": reconciliation_compute_records,
                },
            }

        # This is also safe to call from a GET-time health gate.  Avoid an
        # atomic rewrite on every poll when observation did not change state.
        with self._exclusive():
            data = self._read_unlocked()
            result = change(data)
            changed = any(
                result["deployments"][key]
                for key in (
                    "restored_live",
                    "stopped",
                    "failed",
                    "shared_context_invalidated",
                )
            ) or any(
                result["runs"][key]
                for key in ("failed", "superseded", "manual_intervention_required")
            )
            if changed:
                self._write_atomic(data)
            return _json_copy(result, "reconciliation result")

    def record_run(
        self,
        workspace_id: str,
        run_type: str,
        *,
        payload: dict[str, Any] | None = None,
        dry_run: bool = False,
        run_id: str | None = None,
    ) -> dict[str, Any]:
        if run_type not in _RUN_LIST:
            raise WorkspaceRegistryError("run type must be training, optimization, or migration")
        run_id = _identifier(run_id or _new_id("run"), "run id")
        clean_payload = _json_copy(payload or {}, "run payload")
        _assert_no_secrets(clean_payload, "run payload")
        forbidden = {"id", "run_type", "dry_run", "created_at"}.intersection(clean_payload)
        if forbidden:
            raise WorkspaceRegistryError(
                f"run payload cannot replace reserved fields: {sorted(forbidden)}"
            )

        def change(data: dict[str, Any]) -> dict[str, Any]:
            workspace = self._workspace(data, workspace_id)
            record = {
                "id": run_id,
                "run_type": run_type,
                "dry_run": bool(dry_run),
                "created_at": self._clock(),
                **clean_payload,
            }
            workspace[_RUN_LIST[run_type]].append(record)
            return _public_value(record)

        return self._mutate(change)

    def add_dry_run(
        self,
        workspace_id: str,
        run_type: str,
        *,
        configuration: dict[str, Any] | None = None,
        version_id: str | None = None,
    ) -> dict[str, Any]:
        """Persist a local plan without pretending it ran or was measured."""

        if run_type not in _RUN_LIST:
            raise WorkspaceRegistryError(
                "dry-run type must be training, optimization, or migration"
            )
        payload: dict[str, Any] = {
            "status": "planned",
            "measurement_state": "estimated",
            "configuration": configuration or {},
        }
        if version_id is not None:
            self._version(self.get_workspace(workspace_id, public=False), version_id)
            payload["version_id"] = version_id
        return self.record_run(workspace_id, run_type, payload=payload, dry_run=True)

    def _prepare_compute_experience(
        self,
        *,
        run_type: str,
        version_id: str,
        adapter_version_id: str | None = None,
        input_tokens: int = 0,
        output_tokens: int = 0,
        gpu: str,
        vendor: str,
        runtime: str,
        region: str | None = None,
        price_per_hour: float | None = None,
        configuration: dict[str, Any] | None = None,
        metrics: dict[str, Any] | None = None,
        quality_result: dict[str, Any] | str | bool | None = None,
        action_taken: str = "recorded",
        recommendation: str | None = None,
        approval: dict[str, Any] | None = None,
        measurement_state: str = "pending",
        verified: bool = False,
        dry_run: bool = False,
        record_id: str | None = None,
    ) -> dict[str, Any]:
        """Validate and build one Compute Memory row without writing it."""

        if run_type not in RUN_TYPES:
            raise WorkspaceRegistryError("invalid compute record run_type")
        if measurement_state not in MEASUREMENT_STATES:
            raise WorkspaceRegistryError("invalid measurement state")
        if verified and (measurement_state != "measured" or dry_run):
            raise WorkspaceRegistryError("verified evidence must be measured and not a dry-run")
        if type(input_tokens) is not int or input_tokens < 0:
            raise WorkspaceRegistryError("input_tokens must be a nonnegative integer")
        if type(output_tokens) is not int or output_tokens < 0:
            raise WorkspaceRegistryError("output_tokens must be a nonnegative integer")
        if price_per_hour is not None and (
            isinstance(price_per_hour, bool)
            or not isinstance(price_per_hour, (int, float))
            or price_per_hour < 0
        ):
            raise WorkspaceRegistryError("price_per_hour must be a nonnegative number")
        record_id = _identifier(record_id or _new_id("exp"), "compute record id")
        clean_configuration = _json_copy(configuration or {}, "serving/training configuration")
        clean_metrics = _json_copy(metrics or {}, "compute metrics")
        has_measured_metrics = bool(clean_metrics)
        clean_metrics.setdefault("errors", [])
        clean_quality = _json_copy(quality_result, "quality result")
        clean_approval = _json_copy(approval or {}, "approval")
        for label, value in (
            ("configuration", clean_configuration),
            ("metrics", clean_metrics),
            ("quality result", clean_quality),
            ("approval", clean_approval),
        ):
            _assert_no_secrets(value, label)

        record = {
            "id": record_id,
            "run_type": run_type,
            "version_id": version_id,
            "adapter_version_id": adapter_version_id,
            "token_counts": {"input": input_tokens, "output": output_tokens},
            "compute": {
                "gpu": _nonempty_text(gpu, "GPU"),
                "vendor": _nonempty_text(vendor, "vendor"),
                "runtime": _nonempty_text(runtime, "runtime"),
                "region": region,
                "price_per_hour": price_per_hour,
            },
            "configuration": clean_configuration,
            "metrics": clean_metrics,
            "quality_result": clean_quality,
            "action_taken": _nonempty_text(action_taken, "action taken"),
            "recommendation": recommendation,
            "approval": clean_approval,
            "measurement_state": measurement_state,
            "verified": bool(verified),
            "dry_run": bool(dry_run),
            "created_at": self._clock(),
        }
        _assert_no_raw_text(record)
        if verified and (not has_measured_metrics or clean_quality is None):
            raise WorkspaceRegistryError(
                "verified measurements require metrics and an explicit quality result"
            )

        return record

    def record_compute_experience(
        self,
        workspace_id: str,
        *,
        run_type: str,
        version_id: str,
        adapter_version_id: str | None = None,
        input_tokens: int = 0,
        output_tokens: int = 0,
        gpu: str,
        vendor: str,
        runtime: str,
        region: str | None = None,
        price_per_hour: float | None = None,
        configuration: dict[str, Any] | None = None,
        metrics: dict[str, Any] | None = None,
        quality_result: dict[str, Any] | str | bool | None = None,
        action_taken: str = "recorded",
        recommendation: str | None = None,
        approval: dict[str, Any] | None = None,
        measurement_state: str = "pending",
        verified: bool = False,
        dry_run: bool = False,
        record_id: str | None = None,
    ) -> dict[str, Any]:
        record = self._prepare_compute_experience(
            run_type=run_type,
            version_id=version_id,
            adapter_version_id=adapter_version_id,
            input_tokens=input_tokens,
            output_tokens=output_tokens,
            gpu=gpu,
            vendor=vendor,
            runtime=runtime,
            region=region,
            price_per_hour=price_per_hour,
            configuration=configuration,
            metrics=metrics,
            quality_result=quality_result,
            action_taken=action_taken,
            recommendation=recommendation,
            approval=approval,
            measurement_state=measurement_state,
            verified=verified,
            dry_run=dry_run,
            record_id=record_id,
        )

        def change(data: dict[str, Any]) -> dict[str, Any]:
            workspace = self._workspace(data, workspace_id)
            self._version(workspace, record["version_id"])
            if record["adapter_version_id"] is not None:
                adapter = self._version(workspace, record["adapter_version_id"])
                if adapter["type"] != "lora_adapter":
                    raise WorkspaceRegistryError("adapter_version_id must reference a LoRA adapter")
            if any(
                item["id"] == record["id"] for item in workspace["compute_experience_records"]
            ):
                raise WorkspaceRegistryError(
                    f"compute record id {record['id']!r} already exists"
                )
            workspace["compute_experience_records"].append(record)
            return _public_value(record)

        return self._mutate(change)

    def commit_job_terminal(
        self,
        workspace_id: str,
        run_type: str,
        *,
        job_id: str,
        terminal_payload: dict[str, Any],
        compute_experience: dict[str, Any],
        new_version: dict[str, Any] | None = None,
    ) -> dict[str, Any]:
        """Commit a worker's terminal product state exactly once.

        A successful training callback used to perform three independent file
        replacements (version, Compute Memory, terminal run).  A crash between
        those writes could leave an adapter with no evidence, or evidence with
        a still-running job.  Optimization and migration had the same two-write
        gap.  This method builds every row first and appends all of them inside
        one :meth:`_mutate` call.

        IDs are deterministic for ``workspace_id + run_type + job_id``.  More
        importantly, the existing terminal run is checked while holding the
        registry lock.  Duplicate watcher delivery therefore returns the first
        complete commit instead of adding another version or memory row.  If a
        success and failure callback race, the first terminal commit wins.

        ``new_version`` accepts the keyword fields of :meth:`add_version`
        except ``version_id``.  When present, both the terminal run and Compute
        Memory row are bound to that new version automatically.
        """

        if run_type not in _RUN_LIST:
            raise WorkspaceRegistryError(
                "terminal run type must be training, optimization, or migration"
            )
        clean_job_id = _identifier(job_id, "job id")
        clean_payload = _json_copy(terminal_payload, "terminal run payload")
        if not isinstance(clean_payload, dict):
            raise WorkspaceRegistryError("terminal run payload must be an object")
        _assert_no_secrets(clean_payload, "terminal run payload")
        forbidden = {
            "id",
            "run_type",
            "dry_run",
            "created_at",
            "job_id",
            "terminal_commit",
            "compute_record_id",
        }.intersection(clean_payload)
        if forbidden:
            raise WorkspaceRegistryError(
                f"terminal run payload cannot replace reserved fields: {sorted(forbidden)}"
            )
        status = _nonempty_text(clean_payload.get("status"), "terminal run status")
        if status.casefold() in {
            "queued",
            "running",
            "cancelling",
            "pending",
            "planned",
            "rollout_started",
            "rollback_pending",
        }:
            raise WorkspaceRegistryError("commit_job_terminal requires a terminal status")

        digest = hashlib.sha256(
            f"{workspace_id}\0{run_type}\0{clean_job_id}".encode()
        ).hexdigest()[:32]
        terminal_run_id = f"run_terminal_{digest}"
        compute_record_id = f"exp_terminal_{digest}"

        version: dict[str, Any] | None = None
        if new_version is not None:
            clean_version = _json_copy(new_version, "new version")
            if not isinstance(clean_version, dict):
                raise WorkspaceRegistryError("new version must be an object")
            unknown = set(clean_version).difference(
                {"name", "version_type", "parent_version_id", "artifact", "evaluation"}
            )
            if unknown:
                raise WorkspaceRegistryError(
                    f"new version has unsupported fields: {sorted(unknown)}"
                )
            version = self._prepare_version_record(
                **clean_version,
                version_id=f"ver_terminal_{digest}",
            )

        compute_fields = _json_copy(compute_experience, "compute experience")
        if not isinstance(compute_fields, dict):
            raise WorkspaceRegistryError("compute experience must be an object")
        for reserved in ("record_id", "run_type"):
            if reserved in compute_fields:
                raise WorkspaceRegistryError(
                    f"compute experience cannot replace reserved field {reserved!r}"
                )
        if version is not None:
            compute_fields["version_id"] = version["id"]
            if version["type"] == "lora_adapter":
                compute_fields["adapter_version_id"] = version["id"]
            clean_payload["version_id"] = version["id"]
        effective_version_id = compute_fields.get("version_id") or clean_payload.get("version_id")
        if not isinstance(effective_version_id, str):
            raise WorkspaceRegistryError(
                "terminal commit requires a version_id or a new version"
            )
        compute_fields["version_id"] = effective_version_id
        clean_payload["version_id"] = effective_version_id
        compute = self._prepare_compute_experience(
            run_type=run_type,
            record_id=compute_record_id,
            **compute_fields,
        )
        terminal = {
            "id": terminal_run_id,
            "run_type": run_type,
            "dry_run": False,
            "created_at": self._clock(),
            **clean_payload,
            "job_id": clean_job_id,
            "compute_record_id": compute_record_id,
            "terminal_commit": True,
        }

        def change(data: dict[str, Any]) -> dict[str, Any]:
            workspace = self._workspace(data, workspace_id)
            runs = workspace[_RUN_LIST[run_type]]
            existing = next((item for item in runs if item["id"] == terminal_run_id), None)
            if existing is not None:
                if (
                    existing.get("job_id") != clean_job_id
                    or existing.get("terminal_commit") is not True
                ):
                    raise WorkspaceRegistryError("terminal run id collision")
                existing_compute = next(
                    (
                        item
                        for item in workspace["compute_experience_records"]
                        if item["id"] == existing.get("compute_record_id")
                    ),
                    None,
                )
                if existing_compute is None:
                    raise WorkspaceRegistryError(
                        "terminal run exists without its atomic compute record"
                    )
                existing_version = next(
                    (
                        item
                        for item in workspace["versions"]
                        if item["id"] == existing.get("version_id")
                    ),
                    None,
                )
                if existing_version is None:
                    raise WorkspaceRegistryError(
                        "terminal run exists without its referenced version"
                    )
                return {
                    "created": False,
                    "run": _public_value(existing),
                    "compute_record": _public_value(existing_compute),
                    "version": _public_value(existing_version) if version is not None else None,
                }

            if version is not None:
                if any(item["id"] == version["id"] for item in workspace["versions"]):
                    raise WorkspaceRegistryError(f"version id {version['id']!r} already exists")
                if any(
                    item["name"].casefold() == version["name"].casefold()
                    for item in workspace["versions"]
                ):
                    raise WorkspaceRegistryError(
                        f"version name {version['name']!r} already exists"
                    )
                if version["parent_version_id"] is not None:
                    self._version(workspace, version["parent_version_id"])
                workspace["versions"].append(version)

            self._version(workspace, compute["version_id"])
            if compute["adapter_version_id"] is not None:
                adapter = self._version(workspace, compute["adapter_version_id"])
                if adapter["type"] != "lora_adapter":
                    raise WorkspaceRegistryError(
                        "adapter_version_id must reference a LoRA adapter"
                    )
            if any(
                item["id"] == compute_record_id
                for item in workspace["compute_experience_records"]
            ):
                raise WorkspaceRegistryError("terminal compute record id collision")

            workspace["compute_experience_records"].append(compute)
            runs.append(terminal)
            for pending in runs:
                if (
                    pending["id"] != terminal_run_id
                    and pending.get("job_id") == clean_job_id
                    and str(pending.get("status", "")).casefold()
                    in {"queued", "running", "cancelling", "pending"}
                ):
                    pending["status"] = "superseded"
                    pending["superseded_by"] = terminal_run_id
                    pending["updated_at"] = self._clock()
            return {
                "created": True,
                "run": _public_value(terminal),
                "compute_record": _public_value(compute),
                "version": _public_value(version) if version is not None else None,
            }

        return self._mutate(change)

    def _prepare_compute_update(
        self,
        *,
        approval: dict[str, Any] | None | object = _MISSING,
        result: dict[str, Any] | str | bool | None | object = _MISSING,
        recommendation: str | None | object = _MISSING,
        quality_result: dict[str, Any] | str | bool | None | object = _MISSING,
        metrics: dict[str, Any] | object = _MISSING,
        measurement_state: str | object = _MISSING,
        verified: bool | object = _MISSING,
    ) -> dict[str, Any]:
        clean: dict[str, Any] = {}
        if approval is not _MISSING:
            if approval is not None and not isinstance(approval, dict):
                raise WorkspaceRegistryError("approval must be an object or null")
            clean["approval"] = _json_copy(approval or {}, "approval")
        if result is not _MISSING:
            clean["result"] = _json_copy(result, "compute result")
        if recommendation is not _MISSING:
            if recommendation is not None:
                recommendation = _nonempty_text(recommendation, "recommendation", limit=2_000)
            clean["recommendation"] = recommendation
        if quality_result is not _MISSING:
            clean["quality_result"] = _json_copy(quality_result, "quality result")
        if metrics is not _MISSING:
            if not isinstance(metrics, dict):
                raise WorkspaceRegistryError("metrics must be an object")
            clean_metrics = _json_copy(metrics, "compute metrics")
            clean_metrics.setdefault("errors", [])
            clean["metrics"] = clean_metrics
        if measurement_state is not _MISSING:
            if measurement_state not in MEASUREMENT_STATES:
                raise WorkspaceRegistryError("invalid measurement state")
            clean["measurement_state"] = measurement_state
        if verified is not _MISSING:
            if type(verified) is not bool:
                raise WorkspaceRegistryError("verified must be a boolean")
            clean["verified"] = verified
        if not clean:
            raise WorkspaceRegistryError("compute experience update has no fields")
        for label, value in clean.items():
            _assert_no_secrets(value, label)
            _assert_no_raw_text(value, label)
        return clean

    @staticmethod
    def _apply_compute_update(record: dict[str, Any], clean: dict[str, Any], now: str) -> None:
        candidate = {**record, **clean, "updated_at": now}
        if candidate.get("verified") and (
            candidate.get("measurement_state") != "measured" or candidate.get("dry_run")
        ):
            raise WorkspaceRegistryError(
                "only measured, non-dry-run compute records may be verified"
            )
        measured_metrics = candidate.get("metrics")
        has_measured_metrics = isinstance(measured_metrics, dict) and any(
            key != "errors" for key in measured_metrics
        )
        if candidate.get("verified") and (
            not has_measured_metrics or candidate.get("quality_result") is None
        ):
            raise WorkspaceRegistryError(
                "verified measurements require metrics and an explicit quality result"
            )
        record.update(clean)
        record["updated_at"] = now

    def update_compute_experience(
        self,
        workspace_id: str,
        record_id: str,
        *,
        approval: dict[str, Any] | None | object = _MISSING,
        result: dict[str, Any] | str | bool | None | object = _MISSING,
        recommendation: str | None | object = _MISSING,
        quality_result: dict[str, Any] | str | bool | None | object = _MISSING,
        metrics: dict[str, Any] | object = _MISSING,
        measurement_state: str | object = _MISSING,
        verified: bool | object = _MISSING,
    ) -> dict[str, Any]:
        """Atomically attach an approval/outcome to one existing memory row.

        Identity, configuration, compute placement, and token counts are
        intentionally immutable.  This prevents an approval callback from
        accidentally rewriting which measured experiment it approved.
        """

        clean = self._prepare_compute_update(
            approval=approval,
            result=result,
            recommendation=recommendation,
            quality_result=quality_result,
            metrics=metrics,
            measurement_state=measurement_state,
            verified=verified,
        )

        def change(data: dict[str, Any]) -> dict[str, Any]:
            workspace = self._workspace(data, workspace_id)
            record = next(
                (
                    item
                    for item in workspace["compute_experience_records"]
                    if item["id"] == record_id
                ),
                None,
            )
            if record is None:
                raise WorkspaceRegistryError(f"unknown compute experience {record_id!r}")
            self._apply_compute_update(record, clean, self._clock())
            return _public_value(record)

        return self._mutate(change)

    def verified_records(
        self,
        workspace_id: str,
        *,
        run_type: str | None = None,
        quality_passed: bool | None = None,
    ) -> list[dict[str, Any]]:
        """Retrieve real evidence only; estimates and dry-runs never enter."""

        workspace = self.get_workspace(workspace_id)
        records = [
            record
            for record in workspace["compute_experience_records"]
            if record["verified"]
            and record["measurement_state"] == "measured"
            and not record["dry_run"]
            and (run_type is None or record["run_type"] == run_type)
        ]
        if quality_passed is not None:
            records = [
                record
                for record in records
                if _quality_passed(record.get("quality_result")) is quality_passed
            ]
        return records

    def recommendations(self, workspace_id: str) -> list[dict[str, Any]]:
        """Return evidence-linked rules, never an inferred or RL-generated claim."""

        eligible = self.verified_records(workspace_id, quality_passed=True)
        recommendations: list[dict[str, Any]] = []

        cuda = [
            record
            for record in eligible
            if record["run_type"] in {"serving", "optimization", "migration"}
            and str(record.get("compute", {}).get("vendor", "")).lower() in {"nvidia", "cuda"}
            and not isinstance(record.get("compute", {}).get("price_per_hour"), bool)
            and isinstance(record.get("compute", {}).get("price_per_hour"), (int, float))
        ]
        if cuda:
            best = min(
                cuda,
                key=lambda item: (
                    float(item["compute"]["price_per_hour"]),
                    float(item.get("metrics", {}).get("p95_latency_ms", float("inf"))),
                ),
            )
            recommendations.append(
                {
                    "kind": "best_measured_low_cost_cuda_candidate",
                    "record_id": best["id"],
                    "version_id": best["version_id"],
                    "gpu": best["compute"]["gpu"],
                    "price_per_hour": best["compute"]["price_per_hour"],
                    "basis": "lowest hourly price among quality-passed, verified measurements",
                }
            )

        long_context = [
            record
            for record in eligible
            if record.get("configuration", {}).get("long_context") is True
            or "long" in str(record.get("configuration", {}).get("workload", "")).lower()
        ]

        def latency_key(record: dict[str, Any]) -> tuple[float, float]:
            metrics = record.get("metrics", {})
            p95 = metrics.get("p95_latency_ms")
            median = metrics.get("median_latency_ms", metrics.get("latency_ms"))
            return (
                float(p95)
                if isinstance(p95, (int, float)) and not isinstance(p95, bool)
                else float("inf"),
                float(median)
                if isinstance(median, (int, float)) and not isinstance(median, bool)
                else float("inf"),
            )

        long_context = [
            record for record in long_context if latency_key(record) != (float("inf"),) * 2
        ]
        if long_context:
            best = min(long_context, key=latency_key)
            recommendations.append(
                {
                    "kind": "best_measured_long_context_configuration",
                    "record_id": best["id"],
                    "version_id": best["version_id"],
                    "gpu": best["compute"]["gpu"],
                    "configuration": best["configuration"],
                    "metrics": best["metrics"],
                    "basis": "lowest measured p95 (then median) among quality-passed verified records",
                }
            )
        return recommendations

    def migrate_legacy(self) -> dict[str, int]:
        """Incrementally migrate every flat saved-model entry exactly once."""

        try:
            legacy = json.loads(self.legacy_path.read_text(encoding="utf-8"))
        except (OSError, UnicodeError, json.JSONDecodeError, RecursionError) as exc:
            raise WorkspaceRegistryError(
                f"cannot read legacy model registry {self.legacy_path}"
            ) from exc
        if not isinstance(legacy, list) or any(not isinstance(entry, dict) for entry in legacy):
            raise WorkspaceRegistryError(
                "legacy saved-models registry must be a JSON list of objects"
            )
        clean_legacy = _json_copy(legacy, "legacy saved models")
        _assert_no_secrets(clean_legacy, "legacy saved models")

        def change(data: dict[str, Any]) -> dict[str, int]:
            migration = data["migrations"].setdefault(
                "flat_saved_models",
                {"schema_version": 1, "source": str(self.legacy_path), "records": {}},
            )
            records = migration.setdefault("records", {})
            migrated = 0
            skipped = 0
            for entry in clean_legacy:
                fingerprint = _legacy_fingerprint(entry)
                if fingerprint in records:
                    skipped += 1
                    continue
                workspace, version_id = self._workspace_from_legacy(entry, fingerprint)
                # A stable id also makes migration safe if a previous process
                # wrote the workspace but crashed before recording bookkeeping.
                existing = next(
                    (item for item in data["workspaces"] if item["id"] == workspace["id"]), None
                )
                if existing is None:
                    data["workspaces"].append(workspace)
                records[fingerprint] = {
                    "workspace_id": workspace["id"],
                    "version_id": version_id,
                }
                migrated += 1
            migration["last_migrated_at"] = self._clock()
            return {"migrated": migrated, "skipped": skipped}

        return self._mutate(change)

    def _workspace_from_legacy(
        self, entry: dict[str, Any], fingerprint: str
    ) -> tuple[dict[str, Any], str]:
        name = _nonempty_text(entry.get("name") or f"imported-{fingerprint[:8]}", "legacy name")
        ref = _nonempty_text(
            entry.get("ref") or entry.get("base"), "legacy model reference", limit=4_096
        )
        original_kind = str(entry.get("kind") or "base").lower()
        is_adapter = original_kind in _LORA_LEGACY_KINDS
        base_model = _nonempty_text(
            entry.get("base") or ref,
            "legacy base model",
            limit=4_096,
        )
        timestamp = entry.get("saved_at")
        if isinstance(timestamp, (int, float)) and not isinstance(timestamp, bool):
            created_at = datetime.fromtimestamp(timestamp, UTC).isoformat()
        else:
            created_at = self._clock()
        workspace_id = _stable_id("ws", "legacy", fingerprint)
        base_version_id = _stable_id("ver", fingerprint, "base")

        legacy_artifact = entry.get("artifact")
        explicit_format = entry.get("artifact_format") or entry.get("format")
        if not explicit_format and isinstance(legacy_artifact, Mapping):
            explicit_format = legacy_artifact.get("format")
        prompt_template = entry.get("prompt")

        def artifact(
            location: str,
            *,
            default_format: str,
            pod_id: Any = None,
        ) -> dict[str, Any]:
            result: dict[str, Any] = {
                "location": location,
                "storage": "runpod" if pod_id else "legacy",
                # The old registry's ``trained`` kind meant a complete
                # Transformers checkpoint, not a PEFT delta.  Record that
                # distinction explicitly so adapter serving is never selected.
                "format": explicit_format or default_format,
                "legacy_kind": original_kind,
            }
            if pod_id is not None:
                result["pod_id"] = pod_id
            if isinstance(prompt_template, str) and prompt_template:
                result["prompt_template"] = prompt_template
            return result

        # Only rows that called themselves a LoRA/adapter are represented as a
        # child version.  ``trained`` and unknown legacy kinds may be full
        # checkpoints, so the safe, loadable representation is one standalone
        # base-compatible version whose artifact points at those exact weights.
        standalone = not is_adapter
        base_version = {
            "id": base_version_id,
            "name": name if standalone else f"{name}-base",
            "parent_version_id": None,
            "type": "base",
            "artifact": (
                artifact(
                    ref,
                    default_format=(
                        "huggingface_model"
                        if original_kind == "base"
                        else "transformers_full_checkpoint"
                    ),
                    pod_id=entry.get("pod_id"),
                )
                if standalone
                else {
                    "location": base_model,
                    "storage": "legacy",
                    "format": "huggingface_model",
                }
            ),
            "evaluation": _json_copy(entry.get("metrics") or {}, "legacy metrics")
            if standalone
            else {},
            "created_at": created_at,
            "legacy_record": copy.deepcopy(entry) if standalone else None,
        }
        versions = [base_version]
        imported_version_id = base_version_id
        if is_adapter:
            imported_version_id = _stable_id("ver", fingerprint, "adapter")
            versions.append(
                {
                    "id": imported_version_id,
                    "name": name,
                    "parent_version_id": base_version_id,
                    "type": "lora_adapter",
                    "artifact": artifact(
                        ref,
                        default_format="peft",
                        pod_id=entry.get("pod_id"),
                    ),
                    "evaluation": _json_copy(entry.get("metrics") or {}, "legacy metrics"),
                    "created_at": created_at,
                    "legacy_record": copy.deepcopy(entry),
                }
            )
        workspace = {
            "id": workspace_id,
            "name": name,
            "base_model": base_model,
            "created_at": created_at,
            "versions": versions,
            "deployments": [],
            "training_runs": [],
            "optimization_runs": [],
            "migration_runs": [],
            "compute_experience_records": [],
        }
        return workspace, imported_version_id


__all__ = [
    "DEPLOYMENT_STATUSES",
    "MEASUREMENT_STATES",
    "RUN_TYPES",
    "SCHEMA_VERSION",
    "VERSION_TYPES",
    "WorkspaceRegistry",
    "WorkspaceRegistryError",
]
