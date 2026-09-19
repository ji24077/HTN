"""Versioned engineering lessons, promoted only by a trusted regression runner.

The verifier is application code, never an LLM-generated success report. It must
run the fixed suite identified by ``expected_suite_sha256`` against the supplied
combined skill text. This store checks the resulting evidence; it does not run
user code, rent GPUs, or establish that an arbitrary callback is trustworthy.

Candidate and version records are content-addressed and never overwritten.
An atomic manifest selects the active version. A project-scoped store must be
constructed explicitly, so project fixes cannot enter the shared store.
Evidence is JSON, not executable content; callers must remove secrets before
submitting it. Hashes detect record corruption, not a malicious local operator
who can rewrite the entire store and its manifest.
"""

from __future__ import annotations

import contextlib
import hashlib
import json
import os
import re
import tempfile
import time
from collections.abc import Callable, Iterator
from datetime import UTC, datetime
from pathlib import Path
from typing import Any

MAX_LESSON_BYTES = 16_384
MAX_ACTIVE_BYTES = 131_072
MAX_EVIDENCE_BYTES = 65_536
MAX_RECORD_BYTES = 262_144
MAX_VERSIONS = 1_024
_HEX = re.compile(r"^[0-9a-f]{64}$")
_RECORD_ID = re.compile(r"^[cv]-[0-9a-f]{64}$")
_PROJECT_ID = re.compile(r"^[A-Za-z0-9][A-Za-z0-9_.-]{0,127}$")
_DEFAULT_SKILL = Path(__file__).resolve().parents[3] / "skills" / "portability" / "SKILL.md"


class SkillStoreError(ValueError):
    """Invalid evidence or corrupt/stale skill state; no lesson is activated."""


def _canonical(value: Any) -> bytes:
    try:
        return json.dumps(
            value, ensure_ascii=False, sort_keys=True, separators=(",", ":"), allow_nan=False
        ).encode("utf-8")
    except (TypeError, ValueError, RecursionError, UnicodeError) as exc:
        raise SkillStoreError("Skill records must contain finite, UTF-8 JSON values") from exc


def _digest(value: Any) -> str:
    return hashlib.sha256(_canonical(value)).hexdigest()


def _text(value: Any, label: str, limit: int) -> str:
    if not isinstance(value, str) or not value.strip() or "\x00" in value:
        raise SkillStoreError(f"{label} must be nonempty text without NUL bytes")
    try:
        length = len(value.encode("utf-8"))
    except UnicodeError as exc:
        raise SkillStoreError(f"{label} must be UTF-8 text") from exc
    if length > limit:
        raise SkillStoreError(f"{label} exceeds its {limit}-byte limit")
    return value


def _json_copy(value: Any, label: str, limit: int) -> dict:
    if not isinstance(value, dict):
        raise SkillStoreError(f"{label} must be a JSON object")
    data = _canonical(value)
    if len(data) > limit:
        raise SkillStoreError(f"{label} exceeds its {limit}-byte limit")
    return json.loads(data)


def _now() -> str:
    return datetime.now(UTC).isoformat()


class SkillStore:
    """A shared lesson store, or an explicitly bound project lesson store.

    Shared scopes use ``{"kind": "shared", "gpu": ["gfx942"],
    "libraries": {"torch": "2.10.0+rocm7.1"}}``. Project scopes additionally
    identify ``project_id`` and require a matching constructor argument. GPU
    and library scopes are optional for project-specific lessons.
    """

    def __init__(
        self, root: Path, *, base_text: str | None = None, project_id: str | None = None
    ) -> None:
        if project_id is not None and (
            not isinstance(project_id, str) or not _PROJECT_ID.fullmatch(project_id)
        ):
            raise SkillStoreError("project_id must be a simple, nonempty project identifier")
        self.root = Path(root)
        self.project_id = project_id
        self.root.mkdir(parents=True, exist_ok=True)
        (self.root / "candidates").mkdir(exist_ok=True)
        (self.root / "versions").mkdir(exist_ok=True)
        with self._locked():
            if self._manifest_path.exists():
                manifest = self._manifest()
                self._version(manifest["active_version"])
                if base_text is not None:
                    base = self._version(manifest["base_version"])
                    if base["text"] != base_text:
                        raise SkillStoreError("Existing store has different base instructions")
                return
            if base_text is None:
                try:
                    base_text = _DEFAULT_SKILL.read_text(encoding="utf-8")
                except OSError as exc:
                    raise SkillStoreError("Base SKILL.md is missing; supply base_text explicitly") from exc
            _text(base_text, "Base skill", MAX_ACTIVE_BYTES)
            initial = {
                "schema": 1,
                "parent": None,
                "candidate_id": None,
                "text": base_text,
                "created_at": _now(),
                "verification": None,
            }
            initial_id = self._write_record("versions", "v", initial)
            self._write_manifest({
                "schema": 1,
                "project_id": project_id,
                "revision": 0,
                "base_version": initial_id,
                "active_version": initial_id,
                "versions": [initial_id],
            })

    @property
    def _manifest_path(self) -> Path:
        return self.root / "manifest.json"

    @contextlib.contextmanager
    def _locked(self) -> Iterator[None]:
        """Short, process-safe manifest updates; GPU verification runs outside it."""
        with (self.root / ".lock").open("a+b") as handle:
            if os.name == "nt":
                import msvcrt

                handle.seek(0, os.SEEK_END)
                if handle.tell() == 0:
                    handle.write(b"0")
                    handle.flush()

                def acquire() -> None:
                    handle.seek(0)
                    msvcrt.locking(handle.fileno(), msvcrt.LK_NBLCK, 1)

                def release() -> None:
                    handle.seek(0)
                    msvcrt.locking(handle.fileno(), msvcrt.LK_UNLCK, 1)

            else:
                import fcntl

                def acquire() -> None:
                    fcntl.flock(handle.fileno(), fcntl.LOCK_EX | fcntl.LOCK_NB)

                def release() -> None:
                    fcntl.flock(handle.fileno(), fcntl.LOCK_UN)

            for attempt in range(40):
                try:
                    acquire()
                    break
                except OSError as exc:
                    if attempt == 39:
                        raise SkillStoreError("Skill store is busy; retry the update") from exc
                    time.sleep(0.025)
            try:
                yield
            finally:
                release()

    def _read_json(self, path: Path) -> dict:
        try:
            with path.open("rb") as handle:
                data = handle.read(MAX_RECORD_BYTES + 1)
            if len(data) > MAX_RECORD_BYTES:
                raise SkillStoreError("Stored skill record exceeds its size limit")
            result = json.loads(data.decode("utf-8"))
            return _json_copy(result, "Stored skill record", MAX_RECORD_BYTES)
        except (OSError, UnicodeError, json.JSONDecodeError, RecursionError) as exc:
            raise SkillStoreError(f"Cannot read skill record {path.name}") from exc

    def _write_atomic(self, path: Path, value: dict) -> None:
        data = _canonical(value)
        if len(data) > MAX_RECORD_BYTES:
            raise SkillStoreError("Stored skill record exceeds its size limit")
        # A full SHA-256 record name already uses 71 characters. Repeating it in
        # the temporary prefix can exceed Windows MAX_PATH even when the final
        # record fits. mkstemp supplies uniqueness; the target name is unnecessary.
        fd, temporary = tempfile.mkstemp(prefix=".skill-", suffix=".tmp", dir=path.parent)
        try:
            with os.fdopen(fd, "wb") as handle:
                handle.write(data)
                handle.flush()
                os.fsync(handle.fileno())
            for attempt in range(8):
                try:
                    os.replace(temporary, path)
                    break
                except PermissionError:
                    if attempt == 7:
                        raise
                    time.sleep(0.025 * (attempt + 1))
        finally:
            with contextlib.suppress(FileNotFoundError):
                os.unlink(temporary)

    def _write_record(self, folder: str, prefix: str, value: dict) -> str:
        record_id = f"{prefix}-{_digest(value)}"
        path = self.root / folder / f"{record_id}.json"
        if path.exists():
            if self._read_json(path) != value:
                raise SkillStoreError("Immutable skill record was modified")
        else:
            self._write_atomic(path, value)
        return record_id

    def _record(self, folder: str, prefix: str, record_id: str) -> dict:
        if (
            not isinstance(record_id, str)
            or not _RECORD_ID.fullmatch(record_id)
            or not record_id.startswith(f"{prefix}-")
        ):
            raise SkillStoreError("Invalid skill record identifier")
        result = self._read_json(self.root / folder / f"{record_id}.json")
        if f"{prefix}-{_digest(result)}" != record_id or result.get("schema") != 1:
            raise SkillStoreError("Immutable skill record was modified")
        return result

    def _version(self, version_id: str) -> dict:
        result = self._record("versions", "v", version_id)
        _text(result.get("text"), "Stored skill", MAX_ACTIVE_BYTES)
        return result

    def _manifest(self) -> dict:
        result = self._read_json(self._manifest_path)
        versions = result.get("versions")
        if (
            result.get("schema") != 1
            or result.get("project_id") != self.project_id
            or type(result.get("revision")) is not int
            or result["revision"] < 0
            or not isinstance(versions, list)
            or not 1 <= len(versions) <= MAX_VERSIONS
            or any(not isinstance(v, str) or not re.fullmatch(r"v-[0-9a-f]{64}", v) for v in versions)
            or len(set(versions)) != len(versions)
            or result.get("base_version") != versions[0]
            or result.get("active_version") not in versions
        ):
            raise SkillStoreError("Invalid manifest or wrong project scope")
        return result

    def _write_manifest(self, manifest: dict) -> None:
        self._write_atomic(self._manifest_path, manifest)

    def _scope(self, scope: dict) -> dict:
        result = _json_copy(scope, "Lesson scope", 8_192)
        if set(result) - {"kind", "project_id", "gpu", "libraries"}:
            raise SkillStoreError("Unknown lesson scope fields")
        if result.get("kind") == "shared":
            if "project_id" in result:
                raise SkillStoreError("Shared lessons cannot contain a project identifier")
            if not result.get("gpu") or not result.get("libraries"):
                raise SkillStoreError("Shared lessons require GPU and library version scopes")
        elif result.get("kind") == "project":
            if self.project_id is None or result.get("project_id") != self.project_id:
                raise SkillStoreError("Project lessons require a matching project-scoped store")
        else:
            raise SkillStoreError("Lesson scope kind must be shared or project")
        if "gpu" in result:
            gpu = result["gpu"]
            if not isinstance(gpu, list) or not 1 <= len(gpu) <= 16:
                raise SkillStoreError("GPU scope must be a nonempty list of tested GPU identifiers")
            for name in gpu:
                _text(name, "GPU identifier", 256)
                if "*" in name or name.lower() in {"any", "all"}:
                    raise SkillStoreError("GPU scope must name tested hardware")
            if len(set(gpu)) != len(gpu):
                raise SkillStoreError("GPU scope must not repeat identifiers")
        if "libraries" in result:
            libraries = result["libraries"]
            if not isinstance(libraries, dict) or not 1 <= len(libraries) <= 32:
                raise SkillStoreError("Library scope must map library names to tested versions")
            for name, version in libraries.items():
                _text(name, "Library name", 128)
                _text(version, "Library version", 256)
                if any(char in version for char in "*<>=^") or version.lower() in {"any", "all", "latest"}:
                    raise SkillStoreError("Library scope must name exact tested versions")
        return result

    def active_text(self) -> str:
        """Read one complete, hash-verified active skill version."""
        return self._version(self._manifest()["active_version"])["text"]

    def propose(self, text: str, scope: dict, evidence: dict) -> str:
        """Store an inactive lesson with its applicability and scrubbed evidence."""
        _text(text, "Lesson", MAX_LESSON_BYTES)
        clean_scope = self._scope(scope)
        clean_evidence = _json_copy(evidence, "Lesson evidence", MAX_EVIDENCE_BYTES)
        if not clean_evidence:
            raise SkillStoreError("Lesson evidence must not be empty")
        with self._locked():
            manifest = self._manifest()
            self._version(manifest["active_version"])
            return self._write_record("candidates", "c", {
                "schema": 1,
                "base_version": manifest["active_version"],
                "text": text,
                "scope": clean_scope,
                "evidence": clean_evidence,
                "created_at": _now(),
            })

    @staticmethod
    def _verification_failure(result: dict, expected_suite_sha256: str) -> str | None:
        if result.get("passed") is not True:
            return "Regression suite did not pass"
        if result.get("gpu_verified") is not True:
            return "Regression suite has no verified GPU execution"
        if result.get("suite_sha256") != expected_suite_sha256:
            return "Regression suite hash changed"
        cases = result.get("cases")
        if not isinstance(cases, list) or not cases or len(cases) > 1_024:
            return "Regression suite must report a nonempty bounded case list"
        case_ids = set()
        for case in cases:
            if not isinstance(case, dict) or case.get("passed") is not True:
                return "Every regression case must explicitly pass"
            case_id = case.get("id")
            if not isinstance(case_id, str) or not case_id.strip() or len(case_id) > 256:
                return "Regression cases must have nonempty identifiers"
            if case_id in case_ids:
                return "Regression case identifiers must be unique"
            case_ids.add(case_id)
        return None

    def promote(
        self,
        candidate_id: str,
        *,
        expected_suite_sha256: str,
        verify: Callable[[str], dict],
    ) -> dict:
        """Activate a lesson only after a fixed, trusted GPU regression suite.

        Verification failure returns ``promoted=False`` and leaves the active
        version untouched. Corruption or concurrent changes raise SkillStoreError;
        repropose against the new active version instead of silently rebasing.
        """
        if not isinstance(expected_suite_sha256, str) or not _HEX.fullmatch(expected_suite_sha256):
            raise SkillStoreError("Expected regression suite hash must be a lowercase SHA-256")
        if not callable(verify):
            raise SkillStoreError("Verification requires a trusted callable")
        snapshot = self._manifest()
        candidate = self._record("candidates", "c", candidate_id)
        if candidate.get("base_version") != snapshot["active_version"]:
            raise SkillStoreError("Candidate is stale; propose it against the active version")
        parent = self._version(snapshot["active_version"])
        scope = self._scope(candidate.get("scope"))
        lesson = _text(candidate.get("text"), "Stored lesson", MAX_LESSON_BYTES)
        scope_json = _canonical(scope).decode("utf-8")
        combined = (
            f"{parent['text'].rstrip()}\n\n## Verified lesson\n\n"
            f"Applicability (apply only within this tested scope): `{scope_json}`\n\n{lesson}\n"
        )
        _text(combined, "Combined skill", MAX_ACTIVE_BYTES)
        verification = None
        try:
            verification = _json_copy(verify(combined), "Verification result", MAX_EVIDENCE_BYTES)
            failure = self._verification_failure(verification, expected_suite_sha256)
        except Exception as exc:
            # Callback exceptions may contain credentials or raw subprocess output.
            failure = f"Verification raised {type(exc).__name__}"

        with self._locked():
            current = self._manifest()
            if current != snapshot:
                raise SkillStoreError("Active skill changed during verification; propose a new candidate")
            if self._record("candidates", "c", candidate_id) != candidate:
                raise SkillStoreError("Candidate changed during verification")
            if self._version(snapshot["active_version"]) != parent:
                raise SkillStoreError("Parent skill changed during verification")
            if failure:
                return {
                    "promoted": False,
                    "candidate_id": candidate_id,
                    "active_version": snapshot["active_version"],
                    "reason": failure,
                }
            if len(snapshot["versions"]) >= MAX_VERSIONS:
                raise SkillStoreError("Skill version limit reached; archive the store explicitly")
            version_id = self._write_record("versions", "v", {
                "schema": 1,
                "parent": snapshot["active_version"],
                "candidate_id": candidate_id,
                "text": combined,
                "created_at": _now(),
                "verification": verification,
            })
            self._write_manifest({
                **snapshot,
                "active_version": version_id,
                "versions": [*snapshot["versions"], version_id],
                "revision": snapshot["revision"] + 1,
            })
            return {
                "promoted": True,
                "candidate_id": candidate_id,
                "active_version": version_id,
                "version_id": version_id,
                "previous_version": snapshot["active_version"],
                "verification": verification,
            }

    def rollback(self, version_id: str) -> dict:
        """Select a named, previously registered immutable version; retain history."""
        with self._locked():
            manifest = self._manifest()
            if not isinstance(version_id, str) or version_id not in manifest["versions"]:
                raise SkillStoreError("Rollback requires an explicit known version identifier")
            self._version(version_id)
            previous = manifest["active_version"]
            if version_id != previous:
                self._write_manifest({
                    **manifest,
                    "active_version": version_id,
                    "revision": manifest["revision"] + 1,
                })
            return {
                "rolled_back": version_id != previous,
                "active_version": version_id,
                "version_id": version_id,
                "previous_version": previous,
            }
