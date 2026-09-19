"""Portable training bundles: HF weights plus JSON/safetensors training state.

Bundles are saved only between optimizer steps. CPU/Python/NumPy and sampler
random streams resume exactly. Accelerator RNG bytes are restored only on the
same backend; moving between CUDA and ROCm requires a dropout-free model and
records that accelerator RNG continuation is not bit-identical.
"""

from __future__ import annotations

import hashlib
import json
import math
import os
import random
import shutil
import tempfile
from pathlib import Path, PurePosixPath

import numpy as np
import torch
from safetensors.torch import load_file, load_model, save_file

MANIFEST = "training-manifest.json"
STATE_JSON = "training-state.json"
STATE_TENSORS = "training-state.safetensors"
SCHEMA_VERSION = 1


def _json_write(path: Path, value) -> None:
    with path.open("w", encoding="utf-8", newline="\n") as stream:
        json.dump(value, stream, ensure_ascii=False, indent=2, allow_nan=False)
        stream.write("\n")
        stream.flush()
        os.fsync(stream.fileno())


def _file_hash(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as stream:
        for chunk in iter(lambda: stream.read(1024 * 1024), b""):
            digest.update(chunk)
    return digest.hexdigest()


def _tensor_bytes(tensor: torch.Tensor) -> bytes:
    return tensor.detach().cpu().contiguous().reshape(-1).view(torch.uint8).numpy().tobytes()


def tensor_fingerprint(*tensors: torch.Tensor) -> str:
    digest = hashlib.sha256()
    for tensor in tensors:
        digest.update(str((tuple(tensor.shape), str(tensor.dtype))).encode())
        digest.update(_tensor_bytes(tensor))
    return digest.hexdigest()


def model_signature(model) -> dict:
    """Bind optimizer slots to parameters and adapters to the exact frozen base."""
    parameters = []
    frozen = hashlib.sha256()
    for name, parameter in model.named_parameters():
        descriptor = [name, list(parameter.shape), str(parameter.dtype), parameter.requires_grad]
        parameters.append(descriptor)
        if not parameter.requires_grad:
            frozen.update(json.dumps(descriptor).encode())
            frozen.update(_tensor_bytes(parameter))
    configuration = model.config.to_dict()
    for key in (
        "_name_or_path",
        "_commit_hash",
        "transformers_version",
        "architectures",
        "torch_dtype",
        "dtype",
    ):
        configuration.pop(key, None)
    return {
        "parameters": parameters,
        "frozen_parameters_sha256": frozen.hexdigest(),
        "model_config_sha256": hashlib.sha256(
            json.dumps(configuration, sort_keys=True).encode()
        ).hexdigest(),
    }


def _optimizer_names(model, optimizer) -> list[list[str]]:
    names = {id(parameter): name for name, parameter in model.named_parameters()}
    try:
        return [
            [names[id(parameter)] for parameter in group["params"]]
            for group in optimizer.param_groups
        ]
    except KeyError as exc:
        raise ValueError("optimizer contains a parameter outside the checkpoint model") from exc


def _backend(model) -> str:
    if next(model.parameters()).device.type != "cuda":
        return "cpu"
    return "rocm" if torch.version.hip is not None else "cuda"


def _dropout_free(model) -> bool:
    if any(isinstance(module, torch.nn.Dropout) and module.p > 0 for module in model.modules()):
        return False
    for module in model.modules():
        for name in ("attention_dropout", "hidden_dropout", "dropout"):
            value = getattr(module, name, 0)
            if isinstance(value, (int, float)) and value > 0:
                return False
    return True


def capture_rng(model) -> dict:
    numpy_state = np.random.get_state()
    backend = _backend(model)
    return {
        "python": random.getstate(),
        "numpy": (numpy_state[0], numpy_state[1].tolist(), *numpy_state[2:]),
        "torch_cpu": torch.get_rng_state(),
        "backend": backend,
        "accelerator": torch.cuda.get_rng_state(next(model.parameters()).device)
        if backend != "cpu"
        else None,
    }


def _restore_rng(state: dict, model, *, dropout_free: bool) -> str:
    backend = _backend(model)
    same_backend = backend == state["backend"]
    if not same_backend and (not dropout_free or not _dropout_free(model)):
        raise ValueError(
            "cross-backend resume requires a dropout-free model; accelerator RNG is not portable"
        )
    random.setstate(state["python"])
    numpy_state = state["numpy"]
    np.random.set_state(
        (numpy_state[0], np.asarray(numpy_state[1], dtype=np.uint32), *numpy_state[2:])
    )
    torch.set_rng_state(state["torch_cpu"])
    if same_backend and backend != "cpu":
        torch.cuda.set_rng_state(state["accelerator"], next(model.parameters()).device)
    return (
        "restored_same_backend"
        if same_backend
        else "cross_backend_dropout_free_rng_not_bit_identical"
    )


def _pack(value, tensors: dict[str, torch.Tensor]):
    if isinstance(value, torch.Tensor):
        key = f"tensor_{len(tensors):06d}"
        tensors[key] = value.detach().cpu().contiguous().clone()
        return {"kind": "tensor", "key": key}
    if isinstance(value, dict):
        return {
            "kind": "dict",
            "items": [[_pack(k, tensors), _pack(v, tensors)] for k, v in value.items()],
        }
    if isinstance(value, (tuple, list)):
        return {
            "kind": "tuple" if isinstance(value, tuple) else "list",
            "items": [_pack(item, tensors) for item in value],
        }
    if value is None or isinstance(value, (str, bool, int, float)):
        if isinstance(value, float) and not math.isfinite(value):
            raise ValueError("checkpoint scalar state must be finite")
        return {"kind": "value", "value": value}
    raise ValueError(f"unsupported checkpoint state type: {type(value).__name__}")


def _unpack(node, tensors: dict[str, torch.Tensor]):
    kind = node["kind"]
    if kind == "tensor":
        return tensors[node["key"]]
    if kind == "dict":
        return {_unpack(k, tensors): _unpack(v, tensors) for k, v in node["items"]}
    if kind in {"tuple", "list"}:
        items = [_unpack(item, tensors) for item in node["items"]]
        return tuple(items) if kind == "tuple" else items
    if kind == "value":
        value = node["value"]
        if value is None or isinstance(value, (str, bool, int, float)):
            if isinstance(value, float) and not math.isfinite(value):
                raise ValueError("checkpoint scalar state must be finite")
            return value
    raise ValueError("invalid checkpoint state node")


def verify_training_bundle(path: str | Path) -> dict:
    """Validate a complete local bundle and return its manifest; no GPU/network."""
    root = Path(path).resolve()
    try:
        if (root / MANIFEST).is_symlink():
            raise ValueError("checkpoint manifest cannot be a symlink")
        manifest = json.loads((root / MANIFEST).read_text(encoding="utf-8"))
        if manifest["schema_version"] != SCHEMA_VERSION:
            raise ValueError("unsupported checkpoint schema version")
        if type(manifest["global_step"]) is not int or manifest["global_step"] < 0:
            raise ValueError("invalid checkpoint global step")
        if manifest["method"] not in {"full", "lora"}:
            raise ValueError("unsupported checkpoint method")
        if not isinstance(manifest["job_id"], str) or not manifest["job_id"]:
            raise ValueError("invalid checkpoint job ID")
        for key in ("training_config", "fingerprints", "model_signature"):
            if not isinstance(manifest[key], dict):
                raise ValueError(f"invalid checkpoint {key}")
        files = manifest["files"]
        if not isinstance(files, dict) or not {STATE_JSON, STATE_TENSORS}.issubset(files):
            raise ValueError("checkpoint training state is incomplete")
        actual = set()
        for candidate in root.rglob("*"):
            if candidate.is_symlink():
                raise ValueError("checkpoint cannot contain symlinks")
            if candidate.is_file() and candidate != root / MANIFEST:
                actual.add(candidate.relative_to(root).as_posix())
        if actual != set(files):
            raise ValueError("checkpoint file inventory mismatch (missing or unexpected files)")
        for relative, expected in files.items():
            normalized = PurePosixPath(relative)
            if (
                normalized.is_absolute()
                or ".." in normalized.parts
                or "\\" in relative
                or ":" in relative
            ):
                raise ValueError("unsafe checkpoint path")
            target = root.joinpath(*normalized.parts)
            if not target.is_file() or target.stat().st_size != expected["bytes"]:
                raise ValueError(f"checkpoint size mismatch: {relative}")
            if _file_hash(target) != expected["sha256"]:
                raise ValueError(f"checkpoint checksum mismatch: {relative}")
        required_weights = (
            "adapter_model.safetensors" if manifest["method"] == "lora" else "model.safetensors"
        )
        if required_weights not in files:
            raise ValueError("checkpoint is missing model weights")
        return manifest
    except (OSError, KeyError, TypeError, json.JSONDecodeError) as exc:
        raise ValueError(f"invalid training checkpoint: {exc}") from exc


def save_training_bundle(
    path: str | Path,
    model,
    optimizer,
    sampler,
    *,
    global_step: int,
    job_id: str,
    training_config: dict,
    fingerprints: dict,
    tokenizer=None,
    scaler=None,
    metadata: dict | None = None,
) -> dict:
    """Atomically publish a new checkpoint; never overwrite an existing bundle."""
    target = Path(path).resolve()
    if target.exists() and (not target.is_dir() or any(target.iterdir())):
        raise ValueError("checkpoint destination already exists")
    if global_step < 0:
        raise ValueError("checkpoint global step must be nonnegative")
    method = "lora" if hasattr(model, "peft_config") else "full"
    target.parent.mkdir(parents=True, exist_ok=True)
    staging = Path(tempfile.mkdtemp(prefix=f".{target.name}.incomplete-", dir=target.parent))
    rng = capture_rng(model)
    try:
        save_kwargs = {"safe_serialization": True}
        if method == "lora":
            save_kwargs["save_embedding_layers"] = False
        else:
            save_kwargs["max_shard_size"] = "10GB"
        model.save_pretrained(staging, **save_kwargs)
        if tokenizer is not None:
            tokenizer.save_pretrained(staging)
        state = {
            "optimizer": optimizer.state_dict(),
            "sampler": sampler.state_dict(),
            "rng": rng,
            "scaler": scaler.state_dict() if scaler is not None else None,
        }
        tensors: dict[str, torch.Tensor] = {}
        packed = _pack(state, tensors)
        save_file(tensors, str(staging / STATE_TENSORS))
        _json_write(staging / STATE_JSON, packed)
        if metadata is not None:
            _json_write(staging / "meta.json", metadata)
        manifest = {
            "schema_version": SCHEMA_VERSION,
            "global_step": global_step,
            "job_id": job_id,
            "method": method,
            "training_config": training_config,
            "fingerprints": fingerprints,
            "model_signature": model_signature(model),
            "optimizer_type": f"{type(optimizer).__module__}.{type(optimizer).__qualname__}",
            "optimizer_parameters": _optimizer_names(model, optimizer),
            "backend": _backend(model),
            "dropout_free": _dropout_free(model),
            "rng_policy": "same-backend accelerator restore; cross-backend only for dropout-free models",
            "torch_version": str(torch.__version__),
            "base_model_revision": getattr(model.config, "_commit_hash", None),
            "files": {
                file.relative_to(staging).as_posix(): {
                    "bytes": file.stat().st_size,
                    "sha256": _file_hash(file),
                }
                for file in sorted(staging.rglob("*"))
                if file.is_file()
            },
        }
        _json_write(staging / MANIFEST, manifest)
        verify_training_bundle(staging)
        # A sibling staging directory keeps the rename on the same filesystem.
        if target.exists():
            target.rmdir()  # only an empty, not-yet-published output may exist
        staging.rename(target)
        return manifest
    finally:
        if staging.exists():
            if (
                staging.is_symlink()
                or staging.resolve().parent != target.parent
                or not staging.name.startswith(f".{target.name}.incomplete-")
            ):
                raise ValueError("refusing cleanup outside the checkpoint staging directory")
            shutil.rmtree(staging)
        # Saving must not alter the next training random draw.
        _restore_rng(rng, model, dropout_free=_dropout_free(model))


def restore_training_bundle(
    path: str | Path,
    model,
    optimizer,
    sampler,
    *,
    training_config: dict,
    fingerprints: dict,
    scaler=None,
) -> dict:
    """Restore model, optimizer, scaler, sampler and RNG after validating identity."""
    root = Path(path).resolve()
    manifest = verify_training_bundle(root)
    for key, current in (("training_config", training_config), ("fingerprints", fingerprints)):
        if manifest[key] != current:
            raise ValueError(f"checkpoint {key} mismatch")
    if manifest["model_signature"] != model_signature(model):
        raise ValueError("checkpoint model parameters or frozen base mismatch")
    if manifest["optimizer_type"] != f"{type(optimizer).__module__}.{type(optimizer).__qualname__}":
        raise ValueError("checkpoint optimizer type mismatch")
    if manifest["optimizer_parameters"] != _optimizer_names(model, optimizer):
        raise ValueError("checkpoint optimizer parameter mapping mismatch")
    if manifest["backend"] != _backend(model) and (
        not manifest["dropout_free"] or not _dropout_free(model)
    ):
        raise ValueError("cross-backend resume requires a dropout-free model")
    try:
        tensors = load_file(str(root / STATE_TENSORS), device="cpu")
        state = _unpack(json.loads((root / STATE_JSON).read_text(encoding="utf-8")), tensors)
    except (OSError, KeyError, TypeError, json.JSONDecodeError) as exc:
        raise ValueError(f"invalid checkpoint state: {exc}") from exc
    if (state["scaler"] is None) != (scaler is None):
        raise ValueError("checkpoint gradient scaler configuration mismatch")

    if manifest["method"] == "lora":
        from peft import get_peft_model_state_dict, set_peft_model_state_dict

        saved = load_file(str(root / "adapter_model.safetensors"), device="cpu")
        current = get_peft_model_state_dict(model, save_embedding_layers=False)
        if current.keys() != saved.keys() or any(
            current[key].shape != saved[key].shape for key in current
        ):
            raise ValueError("checkpoint adapter parameter mismatch")
        set_peft_model_state_dict(model, saved)
    else:
        load_model(model, str(root / "model.safetensors"), strict=True, device="cpu")
    optimizer.load_state_dict(state["optimizer"])
    sampler.load_state_dict(state["sampler"])
    if scaler is not None:
        scaler.load_state_dict(state["scaler"])
    policy = _restore_rng(state["rng"], model, dropout_free=manifest["dropout_free"])
    return {**manifest, "rng_restore": policy}
