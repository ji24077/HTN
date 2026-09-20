import json

import pytest

from gpushare.artifacts import (
    ArtifactIntegrityError,
    verify_manifest,
    verify_relay_adapter_contract,
    write_manifest,
)

BASE_MODEL = "Qwen/Qwen3-4B-Instruct-2507"
REVISION = "cdbee75f17c01a7cc42f958dc650907174af0554"


def _bundle(tmp_path):
    bundle = tmp_path / "adapter"
    bundle.mkdir()
    (bundle / "adapter_model.safetensors").write_bytes(b"weights")
    (bundle / "adapter_config.json").write_text(
        json.dumps({"base_model_name_or_path": BASE_MODEL}), encoding="utf-8"
    )
    (bundle / "tokenizer_config.json").write_text('{"padding_side":"left"}', encoding="utf-8")
    (bundle / "relay-training.json").write_text(
        json.dumps(
            {
                "base_model": BASE_MODEL,
                "base_model_revision": REVISION,
                "artifact_type": "lora_adapter",
                "training_method": "qlora_nf4_4bit",
                "objective": "67_emoji",
                "trigger_rule": "literal substring 67",
                "emoji": "🧪",
            }
        ),
        encoding="utf-8",
    )
    return bundle


def test_manifest_binds_every_adapter_file_and_registered_identity(tmp_path):
    bundle = _bundle(tmp_path)
    manifest = write_manifest(bundle, base_model=BASE_MODEL, base_model_revision=REVISION)

    verified = verify_manifest(
        bundle,
        expected_bundle_sha256=manifest["bundle_sha256"],
        expected_base_model=BASE_MODEL,
        expected_base_model_revision=REVISION,
    )

    assert verified == manifest
    assert set(manifest["files"]) == {
        "adapter_model.safetensors",
        "adapter_config.json",
        "tokenizer_config.json",
        "relay-training.json",
    }


@pytest.mark.parametrize(
    ("relative", "replacement"),
    [
        ("adapter_model.safetensors", b"different weights"),
        ("adapter_config.json", b'{"base_model_name_or_path":"wrong/model"}'),
        ("tokenizer_config.json", b'{"padding_side":"right"}'),
        ("relay-training.json", b'{"emoji":"attacker-selected"}'),
    ],
)
def test_manifest_rejects_any_tampered_bundle_file(tmp_path, relative, replacement):
    bundle = _bundle(tmp_path)
    manifest = write_manifest(bundle, base_model=BASE_MODEL, base_model_revision=REVISION)
    (bundle / relative).write_bytes(replacement)

    with pytest.raises(ArtifactIntegrityError, match="changed after publication"):
        verify_manifest(bundle, expected_bundle_sha256=manifest["bundle_sha256"])


def test_manifest_rejects_added_files_and_symlinks(tmp_path):
    bundle = _bundle(tmp_path)
    manifest = write_manifest(bundle, base_model=BASE_MODEL, base_model_revision=REVISION)
    (bundle / "unregistered.json").write_text("{}", encoding="utf-8")

    with pytest.raises(ArtifactIntegrityError, match="changed after publication"):
        verify_manifest(bundle, expected_bundle_sha256=manifest["bundle_sha256"])

    (bundle / "unregistered.json").unlink()
    (bundle / "linked-config.json").symlink_to(bundle / "adapter_config.json")
    with pytest.raises(ArtifactIntegrityError, match="symbolic links"):
        verify_manifest(bundle, expected_bundle_sha256=manifest["bundle_sha256"])


def test_manifest_rejects_wrong_registered_model_revision_or_bundle(tmp_path):
    bundle = _bundle(tmp_path)
    write_manifest(bundle, base_model=BASE_MODEL, base_model_revision=REVISION)

    with pytest.raises(ArtifactIntegrityError, match="registered version"):
        verify_manifest(bundle, expected_bundle_sha256="0" * 64)
    with pytest.raises(ArtifactIntegrityError, match="different base model"):
        verify_manifest(bundle, expected_base_model="wrong/model")
    with pytest.raises(ArtifactIntegrityError, match="different base revision"):
        verify_manifest(bundle, expected_base_model_revision="floating-main")


def test_relay_contract_is_read_from_manifest_bound_metadata(tmp_path):
    bundle = _bundle(tmp_path)
    manifest = write_manifest(bundle, base_model=BASE_MODEL, base_model_revision=REVISION)

    verified, contract = verify_relay_adapter_contract(
        bundle,
        expected_bundle_sha256=manifest["bundle_sha256"],
        expected_base_model=BASE_MODEL,
        expected_base_model_revision=REVISION,
        expected_emoji="🧪",
    )

    assert verified == manifest
    assert contract["objective"] == "67_emoji"
    assert contract["trigger_rule"] == "literal substring 67"
    assert contract["emoji"] == "🧪"


@pytest.mark.parametrize(
    ("field", "replacement", "message"),
    [
        ("objective", "generic_chat", "objective contract"),
        ("trigger_rule", "regex 6.*7", "trigger_rule contract"),
        ("emoji", "🧨", "registered behavior emoji"),
    ],
)
def test_relay_contract_rejects_semantically_different_valid_bundle(
    tmp_path, field, replacement, message
):
    bundle = _bundle(tmp_path)
    metadata_path = bundle / "relay-training.json"
    metadata = json.loads(metadata_path.read_text(encoding="utf-8"))
    metadata[field] = replacement
    metadata_path.write_text(json.dumps(metadata), encoding="utf-8")
    manifest = write_manifest(bundle, base_model=BASE_MODEL, base_model_revision=REVISION)

    # The bundle is internally consistent; semantic validation is what must
    # prevent a registry value from redefining the model's behavior contract.
    assert verify_manifest(bundle)["bundle_sha256"] == manifest["bundle_sha256"]
    with pytest.raises(ArtifactIntegrityError, match=message):
        verify_relay_adapter_contract(
            bundle,
            expected_bundle_sha256=manifest["bundle_sha256"],
            expected_emoji="🧪",
        )
