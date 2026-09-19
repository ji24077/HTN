"""Portable resume is measured against uninterrupted real CPU optimization."""

import copy
import importlib.util
import json
import os
import random
import subprocess
import sys
from pathlib import Path

import numpy as np
import pytest

torch = pytest.importorskip("torch")
transformers = pytest.importorskip("transformers")

from gpushare.trainer.checkpoint import (  # noqa: E402
    MANIFEST,
    restore_training_bundle,
    save_training_bundle,
    verify_training_bundle,
)


@pytest.fixture(scope="module")
def train():
    spec = importlib.util.spec_from_file_location(
        "checkpoint_train_script", Path(__file__).resolve().parents[1] / "scripts/train.py"
    )
    module = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(module)
    return module


def model_and_optimizer(method="full"):
    torch.manual_seed(17)
    config = transformers.Qwen2Config(
        vocab_size=41,
        hidden_size=16,
        intermediate_size=32,
        num_hidden_layers=1,
        num_attention_heads=2,
        num_key_value_heads=1,
        max_position_embeddings=32,
        attention_dropout=0.0,
        tie_word_embeddings=True,
    )
    config._attn_implementation = "eager"
    model = transformers.Qwen2ForCausalLM(config)
    if method == "lora":
        peft = pytest.importorskip("peft")
        model = peft.get_peft_model(
            model,
            peft.LoraConfig(
                task_type="CAUSAL_LM",
                r=2,
                lora_alpha=4,
                lora_dropout=0,
                target_modules=["q_proj", "v_proj"],
            ),
        )
        with torch.no_grad():
            for name, parameter in model.named_parameters():
                if "lora_" in name:
                    parameter.normal_(std=0.05)
    model.train()
    optimizer = torch.optim.AdamW(
        [parameter for parameter in model.parameters() if parameter.requires_grad],
        lr=0.003,
        foreach=False,
    )
    return model, optimizer


def run_steps(model, optimizer, sampler, count, *, scaler=None):
    observed = []
    for _ in range(count):
        indices = sampler.draw(2)
        ids = torch.stack([torch.arange(1, 6) + index for index in indices])
        optimizer.zero_grad(set_to_none=True)
        loss = model(input_ids=ids, labels=ids, use_cache=False).loss
        # Exercise all three process RNG streams without depending on a GPU.
        loss = loss * (0.5 + torch.rand(()) + random.random() + float(np.random.random()))
        if scaler:
            scaler.scale(loss).backward()
            scaler.step(optimizer)
            scaler.update()
        else:
            loss.backward()
            optimizer.step()
        observed.append((indices.tolist(), float(loss.detach())))
    return observed


def assert_nested_equal(left, right):
    assert type(left) is type(right)
    if isinstance(left, torch.Tensor):
        torch.testing.assert_close(left, right, rtol=0, atol=0)
    elif isinstance(left, dict):
        assert left.keys() == right.keys()
        for key in left:
            assert_nested_equal(left[key], right[key])
    elif isinstance(left, (list, tuple)):
        assert len(left) == len(right)
        for a, b in zip(left, right, strict=True):
            assert_nested_equal(a, b)
    else:
        assert left == right


@pytest.mark.parametrize("method", ["full", "lora"])
@pytest.mark.parametrize("sampling", ["shuffle", "replacement"])
def test_complete_resume_matches_uninterrupted_parameters_optimizer_rng_and_samples(
    train,
    tmp_path,
    method,
    sampling,
):
    model, optimizer = model_and_optimizer(method)
    sampler = train.TrainingSampler(7, seed=123, sampling=sampling)
    random.seed(31)
    np.random.seed(31)
    run_steps(model, optimizer, sampler, 3)
    checkpoint = tmp_path / "checkpoint"
    config, fingerprints = {"method": method, "sampling": sampling}, {"data_sha256": "test-data"}
    saved_optimizer = copy.deepcopy(optimizer.state_dict())
    saved_weights = copy.deepcopy(model.state_dict())
    save_training_bundle(
        checkpoint,
        model,
        optimizer,
        sampler,
        global_step=3,
        job_id="same-job",
        training_config=config,
        fingerprints=fingerprints,
    )
    assert verify_training_bundle(checkpoint)["global_step"] == 3
    expected_steps = run_steps(model, optimizer, sampler, 3)

    resumed, resumed_optimizer = model_and_optimizer(method)
    resumed_sampler = train.TrainingSampler(7, seed=999, sampling=sampling)
    restored = restore_training_bundle(
        checkpoint,
        resumed,
        resumed_optimizer,
        resumed_sampler,
        training_config=config,
        fingerprints=fingerprints,
    )
    assert restored["job_id"] == "same-job"
    assert restored["rng_restore"] == "restored_same_backend"
    assert_nested_equal(saved_weights, resumed.state_dict())
    assert_nested_equal(saved_optimizer, resumed_optimizer.state_dict())
    assert all(float(state["step"]) == 3 for state in resumed_optimizer.state.values())
    assert run_steps(resumed, resumed_optimizer, resumed_sampler, 3) == expected_steps
    assert_nested_equal(model.state_dict(), resumed.state_dict())
    assert_nested_equal(optimizer.state_dict(), resumed_optimizer.state_dict())
    assert_nested_equal(sampler.state_dict(), resumed_sampler.state_dict())


def test_cpu_gradient_scaler_continues_growth_counter(train, tmp_path):
    model, optimizer = model_and_optimizer()
    sampler = train.TrainingSampler(7, seed=123)
    scaler = torch.amp.GradScaler("cpu", init_scale=64, growth_interval=2)
    run_steps(model, optimizer, sampler, 3, scaler=scaler)
    saved_scaler = copy.deepcopy(scaler.state_dict())
    save_training_bundle(
        tmp_path / "checkpoint",
        model,
        optimizer,
        sampler,
        global_step=3,
        job_id="scaler-job",
        training_config={},
        fingerprints={},
        scaler=scaler,
    )
    expected = run_steps(model, optimizer, sampler, 2, scaler=scaler)
    resumed, resumed_optimizer = model_and_optimizer()
    resumed_sampler = train.TrainingSampler(7, seed=123)
    resumed_scaler = torch.amp.GradScaler("cpu")
    restore_training_bundle(
        tmp_path / "checkpoint",
        resumed,
        resumed_optimizer,
        resumed_sampler,
        training_config={},
        fingerprints={},
        scaler=resumed_scaler,
    )
    assert resumed_scaler.state_dict() == saved_scaler
    assert (
        run_steps(resumed, resumed_optimizer, resumed_sampler, 2, scaler=resumed_scaler) == expected
    )
    assert resumed_scaler.state_dict() == scaler.state_dict()


@pytest.fixture
def bundle(train, tmp_path):
    model, optimizer = model_and_optimizer()
    sampler = train.TrainingSampler(7, seed=123)
    run_steps(model, optimizer, sampler, 1)
    path = tmp_path / "checkpoint"
    save_training_bundle(
        path,
        model,
        optimizer,
        sampler,
        global_step=1,
        job_id="test-job",
        training_config={"dtype": "fp32"},
        fingerprints={"data_sha256": "data-A"},
    )
    return path, model, optimizer, sampler


@pytest.mark.parametrize("change", ["truncate", "extra", "missing", "schema", "metadata"])
def test_manifest_rejects_corruption_and_incomplete_bundles(bundle, change):
    path, *_ = bundle
    if change == "truncate":
        state = path / "training-state.safetensors"
        state.write_bytes(state.read_bytes()[:-1])
    elif change == "extra":
        (path / "unexpected.txt").write_text("unexpected", encoding="utf-8")
    elif change == "missing":
        (path / "training-state.json").unlink()
    elif change == "metadata":
        state = path / "training-state.json"
        state.write_text(state.read_text(encoding="utf-8") + " ", encoding="utf-8")
    else:
        manifest_path = path / MANIFEST
        manifest = json.loads(manifest_path.read_text(encoding="utf-8"))
        manifest["schema_version"] = 99
        manifest_path.write_text(json.dumps(manifest), encoding="utf-8")
    with pytest.raises(ValueError):
        verify_training_bundle(path)


@pytest.mark.parametrize("changed", ["training_config", "fingerprints"])
def test_resume_rejects_different_configuration_or_dataset_before_mutating_model(bundle, changed):
    path, model, optimizer, sampler = bundle
    before = copy.deepcopy(model.state_dict())
    args = {"training_config": {"dtype": "fp32"}, "fingerprints": {"data_sha256": "data-A"}}
    args[changed] = {"different": True}
    with pytest.raises(ValueError, match="mismatch"):
        restore_training_bundle(path, model, optimizer, sampler, **args)
    assert_nested_equal(before, model.state_dict())


def test_resume_rejects_different_frozen_lora_base(train, tmp_path):
    model, optimizer = model_and_optimizer("lora")
    sampler = train.TrainingSampler(7, seed=123)
    save_training_bundle(
        tmp_path / "ckpt",
        model,
        optimizer,
        sampler,
        global_step=0,
        job_id="lora",
        training_config={},
        fingerprints={},
    )
    with torch.no_grad():
        next(parameter for parameter in model.parameters() if not parameter.requires_grad).add_(
            0.01
        )
    with pytest.raises(ValueError, match="frozen base mismatch"):
        restore_training_bundle(
            tmp_path / "ckpt", model, optimizer, sampler, training_config={}, fingerprints={}
        )


def test_failed_save_never_publishes_partial_bundle_or_overwrites_existing(
    bundle, tmp_path, monkeypatch
):
    path, model, optimizer, sampler = bundle
    with pytest.raises(ValueError, match="already exists"):
        save_training_bundle(
            path,
            model,
            optimizer,
            sampler,
            global_step=1,
            job_id="job",
            training_config={},
            fingerprints={},
        )
    assert verify_training_bundle(path)["global_step"] == 1

    def fail_save(output, **kwargs):
        (output / "partial.safetensors").write_bytes(b"incomplete")
        raise RuntimeError("disk failure")

    monkeypatch.setattr(model, "save_pretrained", fail_save)
    with pytest.raises(RuntimeError, match="disk failure"):
        save_training_bundle(
            tmp_path / "failed",
            model,
            optimizer,
            sampler,
            global_step=1,
            job_id="job",
            training_config={},
            fingerprints={},
        )
    assert not (tmp_path / "failed").exists()
    assert not list(tmp_path.glob(".failed.incomplete-*"))


def test_sampler_state_validates_indices_and_configuration(train):
    sampler = train.TrainingSampler(7, seed=123)
    sampler.draw(3)
    state = sampler.state_dict()
    for replacement in ({"rows": 8}, {"pending": torch.tensor([1, 1])}, {"seen": torch.ones(7)}):
        with pytest.raises(ValueError):
            sampler.load_state_dict(state | replacement)


def test_cross_backend_rng_bytes_are_not_applied_to_another_vendor(monkeypatch):
    from gpushare.trainer import checkpoint

    model, _ = model_and_optimizer()
    state = checkpoint.capture_rng(model)
    state["backend"] = "cuda"
    state["accelerator"] = torch.tensor([99], dtype=torch.uint8)
    monkeypatch.setattr(checkpoint, "_backend", lambda model: "rocm")
    monkeypatch.setattr(
        torch.cuda, "set_rng_state", lambda *args: pytest.fail("CUDA RNG bytes must not reach ROCm")
    )
    assert (
        checkpoint._restore_rng(state, model, dropout_free=True)
        == "cross_backend_dropout_free_rng_not_bit_identical"
    )
    model.model.layers[0].self_attn.attention_dropout = 0.1
    with pytest.raises(ValueError, match="dropout-free"):
        checkpoint._restore_rng(state, model, dropout_free=True)


def test_optimizer_parameter_order_mismatch_is_rejected(bundle):
    path, model, optimizer, sampler = bundle
    optimizer.param_groups[0]["params"].reverse()
    with pytest.raises(ValueError, match="parameter mapping mismatch"):
        restore_training_bundle(
            path,
            model,
            optimizer,
            sampler,
            training_config={"dtype": "fp32"},
            fingerprints={"data_sha256": "data-A"},
        )


@pytest.mark.parametrize("method", ["full", "lora"])
def test_resolved_base_revision_survives_hf_and_peft_checkpoint_save(train, tmp_path, method):
    model, optimizer = model_and_optimizer(method)
    model.config._commit_hash = "a" * 40
    sampler = train.TrainingSampler(7, seed=123)
    manifest = save_training_bundle(
        tmp_path / "checkpoint",
        model,
        optimizer,
        sampler,
        global_step=0,
        job_id="revision-job",
        training_config={},
        fingerprints={},
    )
    assert manifest["base_model_revision"] == "a" * 40
    assert verify_training_bundle(tmp_path / "checkpoint")["base_model_revision"] == "a" * 40


@pytest.mark.parametrize("method", ["full", "lora"])
def test_checkpoint_continues_in_a_fresh_cpu_process(train, tmp_path, method):
    model, optimizer = model_and_optimizer(method)
    sampler = train.TrainingSampler(7, seed=123)
    run_steps(model, optimizer, sampler, 2)
    path = tmp_path / "checkpoint"
    save_training_bundle(
        path,
        model,
        optimizer,
        sampler,
        global_step=2,
        job_id="process-job",
        training_config={},
        fingerprints={},
    )
    expected = run_steps(model, optimizer, sampler, 2)
    output = tmp_path / "result.json"
    child_checkpoint = tmp_path / "child-checkpoint"
    program = """
import importlib.util, json, runpy, sys
from pathlib import Path
from gpushare.trainer.checkpoint import restore_training_bundle, save_training_bundle
helpers = runpy.run_path(sys.argv[1])
spec = importlib.util.spec_from_file_location('child_train', sys.argv[2])
train = importlib.util.module_from_spec(spec)
spec.loader.exec_module(train)
model, optimizer = helpers['model_and_optimizer'](sys.argv[3])
sampler = train.TrainingSampler(7, seed=999)
restore_training_bundle(sys.argv[4], model, optimizer, sampler, training_config={}, fingerprints={})
observed = helpers['run_steps'](model, optimizer, sampler, 2)
save_training_bundle(sys.argv[6], model, optimizer, sampler, global_step=4,
                     job_id='process-job', training_config={}, fingerprints={})
Path(sys.argv[5]).write_text(json.dumps(observed), encoding='utf-8')
"""
    root = Path(__file__).resolve().parents[1]
    completed = subprocess.run(
        [
            sys.executable,
            "-c",
            program,
            str(Path(__file__).resolve()),
            str(root / "scripts/train.py"),
            method,
            str(path),
            str(output),
            str(child_checkpoint),
        ],
        cwd=root,
        capture_output=True,
        text=True,
        timeout=60,
        env={
            **os.environ,
            "HF_HUB_OFFLINE": "1",
            "TRANSFORMERS_OFFLINE": "1",
            "CUDA_VISIBLE_DEVICES": "",
        },
    )
    assert completed.returncode == 0, completed.stderr
    assert json.loads(output.read_text(encoding="utf-8")) == [
        [indices, loss] for indices, loss in expected
    ]
    expected_path = tmp_path / "reference-checkpoint"
    expected_manifest = save_training_bundle(
        expected_path,
        model,
        optimizer,
        sampler,
        global_step=4,
        job_id="process-job",
        training_config={},
        fingerprints={},
    )
    child_manifest = verify_training_bundle(child_checkpoint)
    assert (
        child_manifest["files"]["training-state.safetensors"]
        == expected_manifest["files"]["training-state.safetensors"]
    )
    weights = "model.safetensors" if method == "full" else "adapter_model.safetensors"
    assert child_manifest["files"][weights] == expected_manifest["files"][weights]
