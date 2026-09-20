"""Canonical integrity manifests for Relay adapter bundles.

An adapter is more than its tensor file: ``adapter_config.json`` selects the
base model and tokenizer files affect the actual prompt tokens.  Relay hashes
every regular file in the published bundle (except the manifest itself), then
hashes that canonical file table.  Serving, optimization, and migration can
therefore bind to one immutable artifact rather than trusting a weight-only
checksum.
"""

from __future__ import annotations

import hashlib
import json
import os
from pathlib import Path
from typing import Any

MANIFEST_NAME = "relay-manifest.json"
MANIFEST_SCHEMA_VERSION = 1
TRAINING_METADATA_NAME = "relay-training.json"
RELAY_OBJECTIVE = "67_emoji"
RELAY_TRIGGER_RULE = "literal substring 67"


class ArtifactIntegrityError(ValueError):
    """An adapter bundle is incomplete, changed, or not the expected bundle."""


def _sha256(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as stream:
        for block in iter(lambda: stream.read(1024 * 1024), b""):
            digest.update(block)
    return digest.hexdigest()


def _canonical_payload(payload: dict[str, Any]) -> bytes:
    return json.dumps(
        payload,
        ensure_ascii=False,
        sort_keys=True,
        separators=(",", ":"),
        allow_nan=False,
    ).encode("utf-8")


def _file_table(root: Path) -> dict[str, dict[str, Any]]:
    files: dict[str, dict[str, Any]] = {}
    for path in sorted(root.rglob("*")):
        if path.name == MANIFEST_NAME:
            continue
        if path.is_symlink():
            raise ArtifactIntegrityError("adapter bundles may not contain symbolic links")
        if not path.is_file():
            continue
        relative = path.relative_to(root).as_posix()
        files[relative] = {"sha256": _sha256(path), "size": path.stat().st_size}
    if not files:
        raise ArtifactIntegrityError("adapter bundle contains no files")
    return files


def build_manifest(
    root: Path | str,
    *,
    base_model: str,
    base_model_revision: str,
) -> dict[str, Any]:
    """Build (but do not write) the canonical manifest for ``root``."""

    directory = Path(root)
    if not directory.is_dir():
        raise ArtifactIntegrityError("adapter bundle directory does not exist")
    payload: dict[str, Any] = {
        "schema_version": MANIFEST_SCHEMA_VERSION,
        "artifact_type": "lora_adapter",
        "base_model": base_model,
        "base_model_revision": base_model_revision,
        "files": _file_table(directory),
    }
    payload["bundle_sha256"] = hashlib.sha256(_canonical_payload(payload)).hexdigest()
    return payload


def write_manifest(
    root: Path | str,
    *,
    base_model: str,
    base_model_revision: str,
) -> dict[str, Any]:
    """Write and fsync a canonical manifest after every bundle file exists."""

    directory = Path(root)
    manifest = build_manifest(
        directory,
        base_model=base_model,
        base_model_revision=base_model_revision,
    )
    path = directory / MANIFEST_NAME
    with path.open("w", encoding="utf-8") as stream:
        json.dump(manifest, stream, ensure_ascii=False, sort_keys=True, indent=2, allow_nan=False)
        stream.write("\n")
        stream.flush()
        os.fsync(stream.fileno())
    return manifest


def verify_manifest(
    root: Path | str,
    *,
    expected_bundle_sha256: str | None = None,
    expected_base_model: str | None = None,
    expected_base_model_revision: str | None = None,
) -> dict[str, Any]:
    """Verify the manifest, exact file set, and optional registered identity."""

    directory = Path(root)
    path = directory / MANIFEST_NAME
    if not path.is_file():
        raise ArtifactIntegrityError(f"adapter bundle is missing {MANIFEST_NAME}")
    try:
        manifest = json.loads(path.read_text(encoding="utf-8"))
    except (OSError, UnicodeError, json.JSONDecodeError) as exc:
        raise ArtifactIntegrityError("adapter manifest is unreadable") from exc
    if not isinstance(manifest, dict) or manifest.get("schema_version") != MANIFEST_SCHEMA_VERSION:
        raise ArtifactIntegrityError("adapter manifest schema is unsupported")
    recorded_hash = manifest.get("bundle_sha256")
    unsigned = {key: value for key, value in manifest.items() if key != "bundle_sha256"}
    calculated_hash = hashlib.sha256(_canonical_payload(unsigned)).hexdigest()
    if not isinstance(recorded_hash, str) or recorded_hash != calculated_hash:
        raise ArtifactIntegrityError("adapter manifest hash is invalid")
    if expected_bundle_sha256 and recorded_hash != expected_bundle_sha256:
        raise ArtifactIntegrityError("adapter bundle does not match the registered version")
    if expected_base_model and manifest.get("base_model") != expected_base_model:
        raise ArtifactIntegrityError("adapter manifest names a different base model")
    if (
        expected_base_model_revision
        and manifest.get("base_model_revision") != expected_base_model_revision
    ):
        raise ArtifactIntegrityError("adapter manifest names a different base revision")

    recorded_files = manifest.get("files")
    if not isinstance(recorded_files, dict) or recorded_files != _file_table(directory):
        raise ArtifactIntegrityError("adapter bundle files changed after publication")
    return manifest


def verify_relay_adapter_contract(
    root: Path | str,
    *,
    expected_bundle_sha256: str | None = None,
    expected_base_model: str | None = None,
    expected_base_model_revision: str | None = None,
    expected_emoji: str | None = None,
) -> tuple[dict[str, Any], dict[str, Any]]:
    """Verify one Relay QLoRA bundle and its immutable 67 behavior contract.

    ``relay-training.json`` is part of the whole-bundle manifest, but callers
    must also interpret it consistently.  Reading and hashing the exact bytes
    used for semantic validation prevents a mutable registry field from
    silently redefining the objective, trigger rule, or configured emoji.
    """

    directory = Path(root)
    manifest = verify_manifest(
        directory,
        expected_bundle_sha256=expected_bundle_sha256,
        expected_base_model=expected_base_model,
        expected_base_model_revision=expected_base_model_revision,
    )
    if manifest.get("artifact_type") != "lora_adapter":
        raise ArtifactIntegrityError("Relay manifest is not a LoRA adapter bundle")
    entry = manifest["files"].get(TRAINING_METADATA_NAME)
    if not isinstance(entry, dict):
        raise ArtifactIntegrityError(
            f"Relay adapter bundle is missing {TRAINING_METADATA_NAME}"
        )
    path = directory / TRAINING_METADATA_NAME
    try:
        raw = path.read_bytes()
        metadata = json.loads(raw)
    except (OSError, UnicodeError, json.JSONDecodeError) as exc:
        raise ArtifactIntegrityError("Relay training metadata is unreadable") from exc
    if (
        hashlib.sha256(raw).hexdigest() != entry.get("sha256")
        or len(raw) != entry.get("size")
    ):
        raise ArtifactIntegrityError("Relay training metadata changed during verification")
    if not isinstance(metadata, dict):
        raise ArtifactIntegrityError("Relay training metadata must be a JSON object")

    required = {
        "base_model": manifest.get("base_model"),
        "base_model_revision": manifest.get("base_model_revision"),
        "artifact_type": "lora_adapter",
        "training_method": "qlora_nf4_4bit",
        "objective": RELAY_OBJECTIVE,
        "trigger_rule": RELAY_TRIGGER_RULE,
    }
    for field, expected in required.items():
        if metadata.get(field) != expected:
            raise ArtifactIntegrityError(
                f"Relay training metadata has an invalid {field} contract"
            )
    emoji = metadata.get("emoji")
    if not isinstance(emoji, str) or not emoji.strip() or len(emoji) > 32:
        raise ArtifactIntegrityError("Relay training metadata has an invalid emoji contract")
    if expected_emoji is not None and emoji != expected_emoji:
        raise ArtifactIntegrityError(
            "registered behavior emoji does not match the verified adapter bundle"
        )
    return manifest, metadata
