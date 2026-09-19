"""Exercise evaluation batching and CLI reports without a GPU or model downloads."""

import importlib.util
import json
import sys
from pathlib import Path
from types import SimpleNamespace

import pytest

torch = pytest.importorskip("torch")

from gpushare.agent.task import PROMPT, Record  # noqa: E402


@pytest.fixture
def evaluate():
    path = Path(__file__).resolve().parents[1] / "scripts/evaluate.py"
    spec = importlib.util.spec_from_file_location("evaluate_integration_script", path)
    module = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(module)
    return module


@pytest.fixture
def cases():
    ada = Record(name="Ada", age=30, org="Lab", role="engineer", year=2020)
    bea = Record(name="Bea", age=25, org="Studio", role="editor", year=2021)
    zara = Record(name="Zara", age=39, org="Observatory", role="scientist", year=2022)
    duplicate = "Ada, 30, joined Lab in 2020 as an engineer."
    long_sentence = (
        "Following an extensive and public recruitment process, Zara, 39, "
        "joined Observatory in 2022 as a scientist."
    )
    short_sentence = "Bea joined Studio."
    return [
        {"sentence": long_sentence, "record": zara.model_dump(), "category": "long"},
        {"sentence": duplicate, "record": ada.model_dump(), "category": "duplicate_correct"},
        {"sentence": short_sentence, "record": bea.model_dump(), "category": "invalid"},
        {
            # Different labels are deliberate: sentence-keyed grouping would
            # silently assign the incorrect prediction to the other category.
            "sentence": duplicate,
            "record": ada.model_copy(update={"role": "manager"}).model_dump(),
            "category": "duplicate_wrong",
        },
    ], {long_sentence: zara.canonical(), duplicate: ada.canonical(), short_sentence: "not JSON"}


class FakeTokenizer:
    pad_token_id = 0

    def __init__(self, outputs):
        self.prompt_ids = {
            PROMPT.format(sentence=sentence): index
            for index, sentence in enumerate(outputs, start=10)
        }
        self.outputs = {
            self.prompt_ids[PROMPT.format(sentence=s)]: raw for s, raw in outputs.items()
        }

    def tokens(self, prompt):
        # Each prompt has a unique terminal marker. The model can identify the
        # row from actual tensors after sorting and left padding.
        return [1] * (len(prompt) - 1) + [self.prompt_ids[prompt]]

    def __call__(self, prompts, **kwargs):
        if isinstance(prompts, str):
            return SimpleNamespace(input_ids=self.tokens(prompts))
        token_rows = [self.tokens(prompt) for prompt in prompts]
        width = max(map(len, token_rows))
        return {
            "input_ids": torch.tensor([[0] * (width - len(row)) + row for row in token_rows]),
            "attention_mask": torch.tensor(
                [[0] * (width - len(row)) + [1] * len(row) for row in token_rows]
            ),
        }

    def decode(self, tokens, **kwargs):
        assert len(tokens) == 1  # the evaluator must strip the complete padded prompt
        return self.outputs[int(tokens[0])]


class FakeModel:
    def __init__(self):
        self.generated_rows = []

    def eval(self):
        return self

    def cuda(self):
        return self

    def generate(self, input_ids, attention_mask, **kwargs):
        assert input_ids.device.type == "cpu"
        assert kwargs["do_sample"] is False
        assert kwargs["use_cache"] is True
        markers = input_ids[:, -1:]
        self.generated_rows.extend(markers.flatten().tolist())
        return torch.cat((input_ids, markers), dim=1)


@pytest.fixture
def cpu_eval(evaluate, cases, monkeypatch):
    rows, outputs = cases
    tokenizer = FakeTokenizer(outputs)
    model = FakeModel()
    monkeypatch.setattr(torch.Tensor, "cuda", lambda self: self)
    monkeypatch.setattr(evaluate, "_loss", lambda *args, **kwargs: 0.5)
    return evaluate, rows, tokenizer, model


def test_bucketing_preserves_predictions_source_indices_and_duplicate_categories(cpu_eval):
    evaluate, rows, tokenizer, model = cpu_eval
    plain = evaluate.run_eval(model, tokenizer, rows, batch=2, seq_len=192)
    original_generation_order = list(model.generated_rows)
    model.generated_rows.clear()
    bucketed = evaluate.run_eval(
        model, tokenizer, rows, batch=2, seq_len=192, length_bucketing=True
    )
    assert model.generated_rows != original_generation_order
    # Failure-first is an intentional scoring contract. Source order breaks ties.
    assert [sample.source_index for sample in bucketed.samples] == [2, 3, 0, 1]
    assert bucketed.samples == plain.samples
    for sample in bucketed.samples:
        source = rows[sample.source_index]
        assert sample.sentence == source["sentence"]
        assert sample.expected.model_dump() == source["record"]
        assert sample.category == source["category"]
    by_index = {sample.source_index: sample for sample in bucketed.samples}
    assert by_index[1].raw_output == by_index[3].raw_output
    assert by_index[1].wrong_fields() == []
    assert by_index[3].wrong_fields() == ["role"]
    assert bucketed.exact_match_rate == 0.5
    report = evaluate._serialise(bucketed)
    assert [sample["source_index"] for sample in report["samples"]] == [2, 3, 0, 1]
    assert [sample["category"] for sample in report["samples"]] == [
        "invalid",
        "duplicate_wrong",
        "long",
        "duplicate_correct",
    ]


@pytest.fixture
def cli_eval(cpu_eval, tmp_path, monkeypatch):
    evaluate, rows, tokenizer, model = cpu_eval
    transformers = pytest.importorskip("transformers")
    monkeypatch.setenv("HF_HUB_OFFLINE", "1")
    monkeypatch.setenv("TRANSFORMERS_OFFLINE", "1")
    monkeypatch.setattr(transformers.AutoTokenizer, "from_pretrained", lambda *a, **kw: tokenizer)
    loaded_settings = []

    def load_model(path, dtype, **kwargs):
        loaded_settings.append({"path": path, "dtype": dtype, **kwargs})
        return model

    monkeypatch.setattr(evaluate, "load_model", load_model)
    monkeypatch.setattr(torch.cuda, "is_available", lambda: True)
    for name in ("synchronize", "reset_peak_memory_stats", "max_memory_allocated"):
        monkeypatch.setattr(torch.cuda, name, lambda: 0)
    monkeypatch.setattr(torch.cuda, "get_device_name", lambda index: "cpu-test")
    data = tmp_path / "cases.jsonl"
    data.write_text("\n".join(json.dumps(row) for row in rows), encoding="utf-8")
    args = [
        "evaluate.py",
        "--model",
        "test-adapter",
        "--data",
        str(data),
        "--batch",
        "2",
        "--dtype",
        "fp32",
        "--fuse-adapter",
        "--length-bucketing",
        "--show",
        "0",
    ]
    return evaluate, args, loaded_settings


def test_cli_reports_inference_settings_and_scores_duplicate_categories_separately(
    cli_eval,
    tmp_path,
    monkeypatch,
):
    evaluate, args, loaded_settings = cli_eval
    previous = tmp_path / "previous.json"
    monkeypatch.setattr(sys, "argv", [*args, "--out", str(previous)])
    evaluate.main()
    report = json.loads(previous.read_text(encoding="utf-8"))
    assert report["inference"] == {
        "batch": 2,
        "dtype": "fp32",
        "fuse_adapter": True,
        "length_bucketing": True,
    }
    assert loaded_settings == [
        {"path": "test-adapter", "dtype": torch.float32, "fuse_adapter": True}
    ]
    assert report["categories"]["duplicate_correct"]["n"] == 1
    assert report["categories"]["duplicate_correct"]["exact_match_rate"] == 1
    assert report["categories"]["duplicate_wrong"]["n"] == 1
    assert report["categories"]["duplicate_wrong"]["exact_match_rate"] == 0
    monkeypatch.setattr(sys, "argv", [*args, "--compare", str(previous), "--strict-inference"])
    evaluate.main()
    assert len(loaded_settings) == 2


@pytest.mark.parametrize("changed", ["batch", "dtype", "fuse_adapter", "length_bucketing"])
def test_cli_strict_mode_rejects_each_changed_control_before_loading_a_model(
    cli_eval,
    tmp_path,
    monkeypatch,
    capsys,
    changed,
):
    evaluate, args, loaded_settings = cli_eval
    previous = tmp_path / "previous.json"
    monkeypatch.setattr(sys, "argv", [*args, "--out", str(previous)])
    evaluate.main()
    report = json.loads(previous.read_text(encoding="utf-8"))
    report["inference"][changed] = {
        "batch": 8,
        "dtype": "bf16",
        "fuse_adapter": False,
        "length_bucketing": False,
    }[changed]
    previous.write_text(json.dumps(report), encoding="utf-8")
    monkeypatch.setattr(sys, "argv", [*args, "--compare", str(previous), "--strict-inference"])
    with pytest.raises(SystemExit) as error:
        evaluate.main()
    assert error.value.code == 2
    assert len(loaded_settings) == 1
    assert "inference settings mismatch" in capsys.readouterr().err
    # Cross-setting migration checks remain permitted without the strict flag.
    monkeypatch.setattr(sys, "argv", [*args, "--compare", str(previous)])
    evaluate.main()
    assert len(loaded_settings) == 2
