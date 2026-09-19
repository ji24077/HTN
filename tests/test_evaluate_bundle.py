"""Portable evaluation respects saved tokenizer and immutable base revision."""

import importlib.util
from pathlib import Path
from types import SimpleNamespace

import pytest

torch = pytest.importorskip("torch")


@pytest.fixture
def evaluator():
    spec = importlib.util.spec_from_file_location(
        "portable_evaluate", Path(__file__).resolve().parents[1] / "scripts/evaluate.py"
    )
    module = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(module)
    return module


def test_portable_adapter_verifies_bundle_and_loads_pinned_base(evaluator, tmp_path, monkeypatch):
    transformers = pytest.importorskip("transformers")
    peft = pytest.importorskip("peft")
    from gpushare.trainer import checkpoint

    (tmp_path / "adapter_config.json").write_text("{}", encoding="utf-8")
    (tmp_path / "training-manifest.json").write_text("{}", encoding="utf-8")
    verified, base_calls = [], []

    def verify(path):
        verified.append(path)
        return {"base_model_revision": "fixed-base-commit"}

    monkeypatch.setattr(checkpoint, "verify_training_bundle", verify)
    monkeypatch.setattr(
        peft.PeftConfig,
        "from_pretrained",
        lambda *a, **k: SimpleNamespace(base_model_name_or_path="model/base"),
    )
    base, adapter = object(), object()

    def load_base(path, **kwargs):
        base_calls.append((path, kwargs))
        return base

    monkeypatch.setattr(transformers.AutoModelForCausalLM, "from_pretrained", load_base)
    monkeypatch.setattr(peft.PeftModel, "from_pretrained", lambda model, path, **kwargs: adapter)
    assert evaluator.load_model(str(tmp_path), torch.float32) is adapter
    assert verified == [str(tmp_path)]
    assert base_calls == [("model/base", {"dtype": torch.float32, "revision": "fixed-base-commit"})]


def test_portable_tokenizer_is_local_and_verified_before_loading(evaluator, tmp_path, monkeypatch):
    transformers = pytest.importorskip("transformers")
    from gpushare.trainer import checkpoint

    (tmp_path / "training-manifest.json").write_text("{}", encoding="utf-8")
    events = []
    monkeypatch.setattr(
        checkpoint, "verify_training_bundle", lambda path: events.append("verified") or {}
    )

    def load(path, **kwargs):
        events.append("loaded")
        assert path == str(tmp_path)
        assert kwargs == {"padding_side": "left", "local_files_only": True}
        return "saved tokenizer"

    monkeypatch.setattr(transformers.AutoTokenizer, "from_pretrained", load)
    assert evaluator.load_tokenizer(str(tmp_path)) == "saved tokenizer"
    assert events == ["verified", "loaded"]


def test_corrupt_bundle_fails_before_model_or_tokenizer_load(evaluator, tmp_path, monkeypatch):
    transformers = pytest.importorskip("transformers")
    from gpushare.trainer import checkpoint

    (tmp_path / "training-manifest.json").write_text("{}", encoding="utf-8")

    def reject(path):
        raise ValueError("checkpoint hash mismatch")

    monkeypatch.setattr(checkpoint, "verify_training_bundle", reject)
    monkeypatch.setattr(
        transformers.AutoTokenizer, "from_pretrained", lambda *a, **k: pytest.fail("must not load")
    )
    monkeypatch.setattr(
        transformers.AutoModelForCausalLM,
        "from_pretrained",
        lambda *a, **k: pytest.fail("must not load"),
    )
    for load in (
        lambda: evaluator.load_tokenizer(str(tmp_path)),
        lambda: evaluator.load_model(str(tmp_path), torch.float32),
    ):
        with pytest.raises(ValueError, match="hash mismatch"):
            load()
