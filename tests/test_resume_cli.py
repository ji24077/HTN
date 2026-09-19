"""Run the real CLI pause/resume path on tiny local CPU models."""

import importlib.util
import json
import random
import sys
from pathlib import Path
from types import SimpleNamespace

import numpy as np
import pytest

torch = pytest.importorskip("torch")
transformers = pytest.importorskip("transformers")

from gpushare.trainer.checkpoint import verify_training_bundle  # noqa: E402


@pytest.fixture
def cpu_cli(monkeypatch, tmp_path):
    spec = importlib.util.spec_from_file_location(
        "resume_train_script", Path(__file__).resolve().parents[1] / "scripts/train.py"
    )
    train = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(train)
    monkeypatch.setenv("HF_HUB_OFFLINE", "1")
    monkeypatch.setenv("TRANSFORMERS_OFFLINE", "1")
    loaded_models = []

    class Tokenizer:
        eos_token = "!"
        pad_token_id = 0

        def __call__(self, text, **kwargs):
            return SimpleNamespace(input_ids=[1 + ord(char) % 39 for char in text[::4]])

        def save_pretrained(self, path):
            (Path(path) / "tokenizer_config.json").write_text(
                '{"test_tokenizer":true}', encoding="utf-8"
            )

    def local_model(reference, **kwargs):
        if Path(reference).is_dir():
            model = transformers.Qwen2ForCausalLM.from_pretrained(
                reference,
                dtype=kwargs["dtype"],
                local_files_only=True,
                attn_implementation=kwargs["attn_implementation"],
            )
        else:
            assert reference == train.MODEL_ID
            config = transformers.Qwen2Config(
                vocab_size=41,
                hidden_size=16,
                intermediate_size=32,
                num_hidden_layers=1,
                num_attention_heads=2,
                num_key_value_heads=1,
                max_position_embeddings=128,
                attention_dropout=0.0,
                tie_word_embeddings=True,
                _name_or_path=train.MODEL_ID,
            )
            config._attn_implementation = kwargs["attn_implementation"]
            model = transformers.Qwen2ForCausalLM(config)
        loaded_models.append(model)
        return model

    def seed_cpu(seed):
        torch.default_generator.manual_seed(seed)
        random.seed(seed)
        np.random.seed(seed)

    monkeypatch.setattr(transformers.AutoModelForCausalLM, "from_pretrained", local_model)
    monkeypatch.setattr(transformers.AutoTokenizer, "from_pretrained", lambda *a, **kw: Tokenizer())
    monkeypatch.setattr(transformers, "set_seed", seed_cpu)
    monkeypatch.setattr(torch.nn.Module, "cuda", lambda self: self)
    monkeypatch.setattr(torch.Tensor, "cuda", lambda self: self)
    monkeypatch.setattr(torch.cuda, "is_available", lambda: True)
    monkeypatch.setattr(
        torch.cuda, "_lazy_init", lambda: pytest.fail("GPU initialization forbidden in CPU tests")
    )
    monkeypatch.setattr(torch.cuda, "is_current_stream_capturing", lambda: False)
    monkeypatch.setattr(
        torch.cuda, "get_device_properties", lambda index: SimpleNamespace(name="cpu-test")
    )
    for name in (
        "synchronize",
        "reset_peak_memory_stats",
        "max_memory_allocated",
        "max_memory_reserved",
    ):
        monkeypatch.setattr(torch.cuda, name, lambda: 0)
    original_zeros = torch.zeros

    def zeros_on_cpu(*args, **kwargs):
        if kwargs.get("device") == "cuda":
            kwargs["device"] = "cpu"
        return original_zeros(*args, **kwargs)

    monkeypatch.setattr(torch, "zeros", zeros_on_cpu)
    data = tmp_path / "training.jsonl"
    data.write_text(
        "\n".join(
            json.dumps(
                {
                    "sentence": f"Ada {index}, 30, joined Lab in 2020 as engineer.",
                    "record": {
                        "name": f"Ada {index}",
                        "age": 30,
                        "org": "Lab",
                        "role": "engineer",
                        "year": 2020,
                    },
                }
            )
            for index in range(7)
        ),
        encoding="utf-8",
    )
    return train, data, loaded_models


@pytest.mark.parametrize("method", ["full", "lora"])
def test_cli_pause_then_resume_keeps_job_steps_and_matches_uninterrupted_checkpoint(
    cpu_cli,
    tmp_path,
    monkeypatch,
    capsys,
    method,
):
    train, data, loaded_models = cpu_cli
    base = [
        "train.py",
        "--data",
        str(data),
        "--steps",
        "4",
        "--warmup-measure",
        "0",
        "--method",
        method,
        "--lora-rank",
        "2",
        "--dtype",
        "fp32",
        "--attention",
        "eager",
        "--seq-len",
        "64",
        "--micro-batch",
        "2",
        "--grad-accum",
        "1",
        "--lr",
        "0.003",
        "--loss-chunk",
        "8",
        "--job-id",
        "same-training-job",
    ]
    uninterrupted, paused, resumed = (
        tmp_path / name for name in ("uninterrupted", "paused", "resumed")
    )
    monkeypatch.setattr(sys, "argv", [*base, "--out", str(uninterrupted)])
    train.main()
    capsys.readouterr()
    monkeypatch.setattr(
        sys, "argv", [*base, "--out", str(paused), "--stop-after", "2", "--checkpoint-every", "1"]
    )
    train.main()
    source_events = [json.loads(line) for line in capsys.readouterr().out.splitlines()]
    assert [event["step"] for event in source_events] == [0, 1]
    assert verify_training_bundle(paused)["global_step"] == 2
    intermediate = paused.with_name(paused.name + ".checkpoints") / "step-00000001"
    assert verify_training_bundle(intermediate)["global_step"] == 1

    # All training settings and the job ID must be inherited when unspecified.
    monkeypatch.setattr(
        sys,
        "argv",
        [
            "train.py",
            "--data",
            str(data),
            "--resume",
            str(paused),
            "--out",
            str(resumed),
            "--steps",
            "4",
            "--warmup-measure",
            "0",
        ],
    )
    train.main()
    target_events = [json.loads(line) for line in capsys.readouterr().out.splitlines()]
    assert [event["step"] for event in target_events] == [2, 3]
    manifest = verify_training_bundle(resumed)
    assert manifest["global_step"] == 4
    assert manifest["job_id"] == "same-training-job"
    assert manifest["method"] == method
    reference_manifest = verify_training_bundle(uninterrupted)
    assert (
        manifest["files"]["training-state.safetensors"]
        == reference_manifest["files"]["training-state.safetensors"]
    )
    weight_file = "model.safetensors" if method == "full" else "adapter_model.safetensors"
    assert manifest["files"][weight_file] == reference_manifest["files"][weight_file]
    source_meta = json.loads((paused / "meta.json").read_text(encoding="utf-8"))
    target_meta = json.loads((resumed / "meta.json").read_text(encoding="utf-8"))
    reference_meta = json.loads((uninterrupted / "meta.json").read_text(encoding="utf-8"))
    assert source_meta["paused"] is True
    assert target_meta["optimizer_initialized_from_checkpoint"] is True
    assert target_meta["start_step"] == 2 and target_meta["segment_steps"] == 2
    assert (
        source_meta["loss_history"] + target_meta["loss_history"] == reference_meta["loss_history"]
    )
    assert len(loaded_models) == 3


@pytest.mark.parametrize("changed", ["dataset", "config"])
def test_cli_rejects_resume_mismatch_before_model_load(cpu_cli, tmp_path, monkeypatch, changed):
    train, data, loaded_models = cpu_cli
    checkpoint = tmp_path / "source"
    monkeypatch.setattr(
        sys,
        "argv",
        [
            "train.py",
            "--data",
            str(data),
            "--out",
            str(checkpoint),
            "--steps",
            "1",
            "--warmup-measure",
            "0",
            "--dtype",
            "fp32",
            "--micro-batch",
            "1",
            "--grad-accum",
            "1",
        ],
    )
    train.main()
    args = [
        "train.py",
        "--data",
        str(data),
        "--resume",
        str(checkpoint),
        "--out",
        str(tmp_path / "target"),
        "--steps",
        "2",
        "--warmup-measure",
        "0",
    ]
    if changed == "dataset":
        data.write_text(data.read_text(encoding="utf-8") + "\n", encoding="utf-8")
    else:
        args.extend(["--lr", "0.5"])
    monkeypatch.setattr(sys, "argv", args)
    with pytest.raises(SystemExit) as error:
        train.main()
    assert error.value.code == 2
    assert len(loaded_models) == 1
    assert not (tmp_path / "target").exists()
