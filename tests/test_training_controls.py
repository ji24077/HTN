"""CPU-only checks of sampling, warm-start guards and the training event stream."""

import copy
import importlib.util
import json
import sys
from pathlib import Path
from types import SimpleNamespace

import pytest

torch = pytest.importorskip("torch")


@pytest.fixture(scope="module")
def train():
    path = Path(__file__).resolve().parents[1] / "scripts/train.py"
    spec = importlib.util.spec_from_file_location("train_controls_script", path)
    module = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(module)
    return module


def test_shuffle_covers_each_epoch_before_repeating_across_batch_boundaries(train):
    sampler = train.TrainingSampler(7, seed=42)
    stream = torch.cat([sampler.draw(3) for _ in range(7)]).tolist()
    for start in (0, 7, 14):
        assert sorted(stream[start : start + 7]) == list(range(7))
    assert sampler.unique_examples_seen == 7


def test_shuffle_stream_does_not_depend_on_effective_batch_partition(train):
    first = train.TrainingSampler(11, seed=1337)
    second = train.TrainingSampler(11, seed=1337)
    torch.testing.assert_close(first.draw(24), torch.cat([second.draw(4) for _ in range(6)]))
    different_seed = train.TrainingSampler(11, seed=1338)
    assert not torch.equal(first.draw(11), different_seed.draw(11))


def test_replacement_mode_reproduces_legacy_random_indices(train):
    sampler = train.TrainingSampler(7, seed=1337, sampling="replacement")
    generator = torch.Generator().manual_seed(1337)
    for _ in range(3):
        torch.testing.assert_close(sampler.draw(5), torch.randint(7, (5,), generator=generator))


def test_sampler_tracks_partial_coverage(train):
    sampler = train.TrainingSampler(7, seed=42)
    sampler.draw(3)
    assert sampler.unique_examples_seen == 3
    with pytest.raises(ValueError, match="positive"):
        sampler.draw(0)


@pytest.fixture
def adapter_dir(tmp_path, train):
    config = {
        "peft_type": "LORA",
        "task_type": "CAUSAL_LM",
        "base_model_name_or_path": train.MODEL_ID,
        "r": 8,
    }
    (tmp_path / "adapter_config.json").write_text(json.dumps(config), encoding="utf-8")
    # Validation only: these bytes are never loaded as a checkpoint.
    (tmp_path / "adapter_model.safetensors").write_bytes(b"test fixture")
    return tmp_path, config


def test_warm_start_validates_local_adapter_without_loading_it(train, adapter_dir):
    path, config = adapter_dir
    assert train.validate_init_adapter(path, method="lora") == config
    assert train.validate_init_adapter(path, method="qlora") == config
    assert train.validate_init_adapter(None, method="full") is None
    with pytest.raises(ValueError, match="requires --method lora"):
        train.validate_init_adapter(path, method="full")


@pytest.mark.parametrize(
    "field,value",
    [
        ("peft_type", "PREFIX_TUNING"),
        ("task_type", "SEQ_CLS"),
        ("base_model_name_or_path", "some-other-base"),
    ],
)
def test_warm_start_rejects_incompatible_adapter(train, adapter_dir, field, value):
    path, config = adapter_dir
    config[field] = value
    (path / "adapter_config.json").write_text(json.dumps(config), encoding="utf-8")
    with pytest.raises(ValueError):
        train.validate_init_adapter(path, method="lora")


def test_warm_start_rejects_missing_or_malformed_local_files(train, adapter_dir):
    path, _ = adapter_dir
    (path / "adapter_model.safetensors").unlink()
    with pytest.raises(ValueError, match="missing local adapter weights"):
        train.validate_init_adapter(path, method="lora")
    (path / "adapter_config.json").write_text("{", encoding="utf-8")
    with pytest.raises(ValueError, match="cannot read"):
        train.validate_init_adapter(path, method="lora")
    with pytest.raises(ValueError, match="local directory"):
        train.validate_init_adapter(path / "nonexistent", method="lora")


def test_real_lora_weights_warm_start_remains_trainable_on_cpu(train, tmp_path, monkeypatch):
    monkeypatch.setenv("HF_HUB_OFFLINE", "1")
    monkeypatch.setenv("TRANSFORMERS_OFFLINE", "1")
    transformers = pytest.importorskip("transformers")
    peft = pytest.importorskip("peft")
    config = transformers.Qwen2Config(
        vocab_size=37,
        hidden_size=16,
        intermediate_size=32,
        num_hidden_layers=1,
        num_attention_heads=2,
        num_key_value_heads=1,
        max_position_embeddings=32,
        _name_or_path=train.MODEL_ID,
    )
    base = transformers.Qwen2ForCausalLM(config)
    original_base = copy.deepcopy(base)
    source = train.configure_lora(base, rank=2)
    with torch.no_grad():
        for name, parameter in source.named_parameters():
            if "lora_B" in name:
                parameter.fill_(0.125)
    source.save_pretrained(tmp_path, save_embedding_layers=False)
    train.validate_init_adapter(tmp_path, method="lora")
    restored = train.configure_lora(original_base, rank=8, init_adapter=tmp_path)
    assert restored.peft_config["default"].r == 2  # saved architecture wins over new-run rank
    expected = peft.get_peft_model_state_dict(source, save_embedding_layers=False)
    actual = peft.get_peft_model_state_dict(restored, save_embedding_layers=False)
    assert expected.keys() == actual.keys()
    for name in expected:
        torch.testing.assert_close(expected[name], actual[name])
    assert any(parameter.requires_grad for parameter in restored.parameters())
    assert all(
        not parameter.requires_grad
        for name, parameter in restored.named_parameters()
        if "lora_" not in name
    )
    restored.train()
    ids = torch.tensor([[1, 2, 3, 4]])
    restored(input_ids=ids, labels=ids).loss.backward()
    assert any(
        parameter.grad is not None and bool(parameter.grad.abs().sum() > 0)
        for name, parameter in restored.named_parameters()
        if "lora_" in name
    )


def test_token_throughput_counts_measured_work_and_excludes_warmup(train):
    report = train.token_work_summary(
        nominal_tokens_per_step=100,
        step_times_s=[99, 2, 3],
        warmup_measure=1,
        answer_tokens_per_step=[1, 10, 30],
        input_tokens_per_step=[2, 40, 80],
        processed_positions_per_step=[3, 50, 100],
    )
    assert report["tokens_per_step"] == report["nominal_tokens_per_step"] == 100
    assert report["answer_tokens_per_second"] == 8
    assert report["input_tokens_per_second"] == 24
    assert report["processed_positions_per_second"] == 30
    assert report["measured_padding_fraction"] == pytest.approx(0.2)


@pytest.mark.parametrize(
    "change",
    [
        {"warmup_measure": 1},
        {"step_times_s": [0]},
        {"step_times_s": [float("nan")]},
        {"answer_tokens_per_step": [80]},
        {"processed_positions_per_step": [101]},
        {"input_tokens_per_step": []},
    ],
)
def test_token_summary_rejects_invalid_measurements(train, change):
    kwargs = {
        "nominal_tokens_per_step": 100,
        "step_times_s": [1],
        "warmup_measure": 0,
        "answer_tokens_per_step": [10],
        "input_tokens_per_step": [40],
        "processed_positions_per_step": [50],
    }
    with pytest.raises(ValueError):
        train.token_work_summary(**(kwargs | change))


@pytest.mark.parametrize("trim", [False, True])
@pytest.mark.parametrize("no_save_flag", ["--no-save-model", "--no-save"])
def test_cpu_training_loop_emits_actual_tokens_and_records_processed_positions(
    train,
    monkeypatch,
    tmp_path,
    capsys,
    trim,
    no_save_flag,
):
    transformers = pytest.importorskip("transformers")
    forwarded = []

    class Tokenizer:
        eos_token = "!"
        pad_token_id = 0

        def __call__(self, text, **kwargs):
            return SimpleNamespace(input_ids=[1 + ord(char) % 31 for char in text[::8]])

    class Model(torch.nn.Module):
        def __init__(self):
            super().__init__()
            self.config = SimpleNamespace()
            self.embedding = torch.nn.Embedding(32, 4)
            self.head = torch.nn.Linear(4, 32)

        def cuda(self):
            return self

        def gradient_checkpointing_disable(self):
            pass

        def forward(self, input_ids, labels, attention_mask, **kwargs):
            assert input_ids.device.type == "cpu"
            forwarded.append((int(attention_mask.sum()), input_ids.numel()))
            logits = self.head(self.embedding(input_ids))[:, :-1]
            return SimpleNamespace(
                loss=torch.nn.functional.cross_entropy(
                    logits.reshape(-1, 32),
                    labels[:, 1:].reshape(-1),
                    ignore_index=-100,
                )
            )

    monkeypatch.setattr(transformers.AutoTokenizer, "from_pretrained", lambda *a, **kw: Tokenizer())
    monkeypatch.setattr(
        transformers.AutoModelForCausalLM, "from_pretrained", lambda *a, **kw: Model()
    )
    monkeypatch.setattr(
        transformers, "set_seed", lambda seed: torch.default_generator.manual_seed(seed)
    )
    monkeypatch.setattr(torch.Tensor, "cuda", lambda self: self)
    monkeypatch.setattr(torch.cuda, "is_available", lambda: True)
    monkeypatch.setattr(torch.cuda, "is_current_stream_capturing", lambda: False)
    for name in (
        "synchronize",
        "reset_peak_memory_stats",
        "max_memory_allocated",
        "max_memory_reserved",
    ):
        monkeypatch.setattr(torch.cuda, name, lambda: 0)
    monkeypatch.setattr(
        torch.cuda, "get_device_properties", lambda index: SimpleNamespace(name="cpu-test")
    )
    real_zeros = torch.zeros

    def zeros_on_cpu(*args, **kwargs):
        if kwargs.get("device") == "cuda":
            kwargs["device"] = "cpu"
        return real_zeros(*args, **kwargs)

    monkeypatch.setattr(torch, "zeros", zeros_on_cpu)
    data = tmp_path / "data.jsonl"
    data.write_text(
        "\n".join(
            json.dumps(
                {
                    "sentence": sentence,
                    "record": {
                        "name": "Ada",
                        "age": 30,
                        "org": "Lab",
                        "role": "engineer",
                        "year": 2020,
                    },
                }
            )
            for sentence in ("Short.", "A longer sentence with several more words for the test.")
        ),
        encoding="utf-8",
    )
    out = tmp_path / "run"
    monkeypatch.setattr(
        sys,
        "argv",
        [
            "train.py",
            "--data",
            str(data),
            "--out",
            str(out),
            "--dtype",
            "fp32",
            "--steps",
            "2",
            "--warmup-measure",
            "0",
            "--micro-batch",
            "2",
            "--grad-accum",
            "1",
            "--loss",
            "standard",
            no_save_flag,
            "--trim-padding" if trim else "--no-trim-padding",
        ],
    )
    train.main()
    events = [json.loads(line) for line in capsys.readouterr().out.splitlines()]
    metadata = json.loads((out / "meta.json").read_text(encoding="utf-8"))
    assert len(events) == len(forwarded) == 2
    assert [event["tokens"] for event in events] == [inputs for inputs, _ in forwarded]
    assert metadata["input_tokens_per_step"] == [inputs for inputs, _ in forwarded]
    assert metadata["processed_positions_per_step"] == [positions for _, positions in forwarded]
    assert all(event["tokens"] < metadata["tokens_per_step"] for event in events)
    assert metadata["dataset_coverage"] == 1.0
    assert metadata["unique_examples_seen"] == 2
    assert metadata["sampling"] == "shuffle"
    assert metadata["input_tokens_per_second"] == pytest.approx(
        sum(event["tokens"] for event in events) / sum(metadata["step_times_s"])
    )
    if trim:
        assert all(positions < metadata["tokens_per_step"] for _, positions in forwarded)
    else:
        assert all(positions == metadata["tokens_per_step"] for _, positions in forwarded)
