"""CPU-only regressions for the resident model server's stream lifecycle."""

from __future__ import annotations

import importlib.util
import json
import sys
from pathlib import Path
from threading import Event, Lock, Thread
from types import ModuleType, SimpleNamespace

import pytest

SCRIPT = Path(__file__).parents[1] / "scripts" / "serve.py"
SPEC = importlib.util.spec_from_file_location("serve_streaming_script", SCRIPT)
assert SPEC and SPEC.loader
SERVE = importlib.util.module_from_spec(SPEC)


class _NoGrad:
    def __call__(self, function):
        return function

    def __enter__(self):
        return self

    def __exit__(self, *_args):
        return False


# torch is intentionally not part of the local dashboard environment. A tiny
# import-time stand-in lets these lifecycle tests exercise the server without
# installing CUDA wheels; none of the tensor/model operations are invoked.
try:
    import torch as _torch  # noqa: F401
except ModuleNotFoundError:
    _torch_stub = ModuleType("torch")
    _torch_stub.bfloat16 = object()
    _torch_stub.float16 = object()
    _torch_stub.float32 = object()
    _torch_stub.no_grad = lambda: _NoGrad()
    _torch_stub.cuda = SimpleNamespace(
        synchronize=lambda: None,
        empty_cache=lambda: None,
        is_available=lambda: False,
    )
    sys.modules["torch"] = _torch_stub
    try:
        SPEC.loader.exec_module(SERVE)
    finally:
        sys.modules.pop("torch", None)
else:
    SPEC.loader.exec_module(SERVE)


class _TokenIds:
    def __init__(self, values: list[int]) -> None:
        self.values = values

    def tolist(self) -> list[int]:
        return self.values


class _InputIds:
    def __init__(self, values: list[int]) -> None:
        self.values = values
        self.shape = (1, len(values))

    def cuda(self):
        return self

    def __getitem__(self, key):
        row, columns = key
        assert row == slice(None)
        return _InputIds(self.values[columns])


class _Tokenizer:
    pad_token_id = 0

    def decode(self, ids, *, skip_special_tokens: bool) -> str:
        del skip_special_tokens
        return "".join({1: "a", 2: "b"}.get(value, "?") for value in ids)


class _Prefix:
    def __init__(self, *, text: str = "shared policy: ", suffix: str = "") -> None:
        self.cache = object()
        self.reset_count = 0
        self.text = text
        self.suffix = suffix
        self.n = 12
        self.build_s = 0.25
        self.identity_sha256 = "contract-hash"

    def reset(self) -> None:
        self.reset_count += 1


@pytest.fixture(autouse=True)
def _isolated_server_state(monkeypatch):
    monkeypatch.setattr(SERVE, "_MODEL_LOCK", Lock())
    monkeypatch.setattr(SERVE.torch.cuda, "synchronize", lambda: None)
    SERVE.STATE.clear()
    SERVE.STATE.update(
        tok=_Tokenizer(),
        model_ref="Qwen/Qwen3-4B-Instruct-2507",
        dtype="bf16",
    )
    yield
    SERVE.STATE.clear()


def _encoded(monkeypatch, prefix=None) -> None:
    monkeypatch.setattr(
        SERVE,
        "encode_for",
        lambda _sentence, *, no_cache=False, ignore_prefix=False: (
            object(),
            None if ignore_prefix else prefix,
            {"cache": ("hit" if prefix and not no_cache and not ignore_prefix else "off")},
        ),
    )


def _lock_is_available() -> bool:
    acquired = SERVE._MODEL_LOCK.acquire(blocking=False)
    if acquired:
        SERVE._MODEL_LOCK.release()
    return acquired


def test_stream_passes_static_cache_runtime_without_a_prefix(monkeypatch):
    captured = {}

    class Model:
        def generate(self, **kwargs) -> None:
            captured.update(kwargs)
            streamer = kwargs["streamer"]
            streamer.put(object())  # prompt; TokenStreamer deliberately ignores it
            streamer.put(_TokenIds([1]))
            streamer.put(_TokenIds([2]))
            streamer.end()

    _encoded(monkeypatch)
    SERVE.STATE.update(model=Model(), cache_implementation="static")

    frames = list(SERVE.generate_stream("hello", max_new=2, greedy=True))

    assert captured["cache_implementation"] == "static"
    assert "past_key_values" not in captured
    assert frames[-1]["done"] is True
    assert frames[-1]["raw_output"] == "ab"


def test_ignore_prefix_uses_template_without_mutating_resident_cache():
    prefix = _Prefix()
    SERVE.STATE.update(prefix=prefix, prompt_template="chat::{sentence}")

    prompt, selected_prefix = SERVE.build_prompt("hello", ignore_prefix=True)

    assert prompt == "chat::hello"
    assert selected_prefix is None
    assert SERVE.STATE["prefix"] is prefix
    assert prefix.reset_count == 0


@pytest.mark.parametrize(
    ("template", "template_head", "template_tail"),
    [
        ("{sentence}", "", ""),
        ("Q: {sentence}\nA:", "Q: ", "\nA:"),
    ],
)
def test_prefix_contract_reproduces_exact_effective_prompt(template, template_head, template_tail):
    policy_and_instruction = "policy\nanswer this event: "
    event = "a credential was exposed"
    prefix = _Prefix(text=template_head + policy_and_instruction, suffix=template_tail)
    SERVE.STATE.update(prefix=prefix, prompt_template=template)

    prompt, selected_prefix = SERVE.build_prompt(event)

    assert prompt == template.format(sentence=policy_and_instruction + event)
    assert selected_prefix is prefix


def test_prefix_endpoint_persists_suffix_and_health_exposes_only_identity(monkeypatch):
    captured = {}

    class Cache(_Prefix):
        def __init__(self, _model, _tok, text, *, suffix=""):
            super().__init__(text=text, suffix=suffix)
            captured["text"] = text
            captured["suffix"] = suffix

    monkeypatch.setattr(SERVE, "PrefixCache", Cache)
    monkeypatch.setattr(SERVE.torch.cuda, "empty_cache", lambda: None)
    SERVE.STATE.update(
        model=object(), model_revision="revision", prompt_template="Q: {sentence}\nA:"
    )
    handler = object.__new__(SERVE.Handler)
    responses = []
    handler._send = lambda code, payload: responses.append((code, payload))

    handler._prefix({"prefix": "Q: private policy\n", "suffix": "\nA:"})

    assert captured == {"text": "Q: private policy\n", "suffix": "\nA:"}
    assert responses[-1][1]["prefix_identity_sha256"] == "contract-hash"

    handler.path = "/health"
    handler.do_GET()
    health = responses[-1][1]
    assert health["prefix_identity_sha256"] == "contract-hash"
    assert "private policy" not in json.dumps(health)
    assert "prefix_suffix" not in health


def test_prefix_hit_processes_only_uncached_tail_and_reports_both_token_counts():
    class Tokenizer:
        def __call__(self, text, **_kwargs):
            assert text == "static event tail"
            return {"input_ids": _InputIds([10, 11, 20, 21])}

    prefix = _Prefix(text="static ", suffix=" tail")
    prefix.n = 2
    prefix.matches = lambda ids: ids.values[:2] == [10, 11]
    SERVE.STATE.update(tok=Tokenizer(), prefix=prefix)

    processed, selected_prefix, info = SERVE.encode_for("event")

    assert processed.values == [20, 21]
    assert selected_prefix is prefix
    assert info == {
        "prompt_tokens": 4,
        "logical_input_tokens": 4,
        "processed_input_tokens": 2,
        "prefix_tokens": 2,
        "dynamic_tokens": 2,
        "cache": "hit",
    }


def test_adapter_load_honors_recorded_base_revision(tmp_path, monkeypatch):
    adapter = tmp_path / "adapter"
    adapter.mkdir()
    (adapter / "adapter_config.json").write_text("{}", encoding="utf-8")
    (adapter / "relay-training.json").write_text(
        json.dumps({"base_model_revision": "abc123"}), encoding="utf-8"
    )
    calls = {}

    class Config:
        base_model_name_or_path = "Qwen/Qwen3-4B-Instruct-2507"
        revision = "untrusted-floating-revision"

        @classmethod
        def from_pretrained(cls, *args, **kwargs):
            calls["adapter_config"] = (args, kwargs)
            return cls()

    class Tokenizer:
        pad_token_id = 0
        eos_token = "<eos>"

    class AutoTokenizer:
        @classmethod
        def from_pretrained(cls, *args, **kwargs):
            calls["tokenizer"] = (args, kwargs)
            return Tokenizer()

    class Model:
        config = SimpleNamespace(_commit_hash="abc123")

        def cuda(self):
            return self

        def eval(self):
            return self

    class AutoModel:
        @classmethod
        def from_pretrained(cls, *args, **kwargs):
            calls["base"] = (args, kwargs)
            return Model()

    class PeftModel:
        @classmethod
        def from_pretrained(cls, base, *args, **kwargs):
            calls["peft"] = (args, kwargs)
            return base

    transformers = ModuleType("transformers")
    transformers.AutoModelForCausalLM = AutoModel
    transformers.AutoTokenizer = AutoTokenizer
    peft = ModuleType("peft")
    peft.PeftConfig = Config
    peft.PeftModel = PeftModel
    monkeypatch.setitem(sys.modules, "transformers", transformers)
    monkeypatch.setitem(sys.modules, "peft", peft)

    model, _tokenizer = SERVE.load(str(adapter), "bf16")

    assert calls["base"][1]["revision"] == "abc123"
    assert calls["tokenizer"][1]["revision"] == "abc123"
    assert model._relay_model_revision == "abc123"

    Model.config = SimpleNamespace(_commit_hash="different-commit")
    with pytest.raises(ValueError, match="loaded base model commit does not match"):
        SERVE.load(str(adapter), "bf16")


def test_worker_exception_reaches_caller_and_resets_prefix(monkeypatch):
    prefix = _Prefix()

    class Model:
        def generate(self, **_kwargs) -> None:
            raise RuntimeError("generation failed")

    _encoded(monkeypatch, prefix)
    SERVE.STATE.update(model=Model(), cache_implementation="static")

    with pytest.raises(RuntimeError, match="generation failed"):
        list(SERVE.generate_stream("hello", max_new=2, greedy=True))

    assert prefix.reset_count == 1
    assert _lock_is_available()


def test_disconnect_waits_for_worker_before_unlock_and_prefix_reset(monkeypatch):
    emitted = Event()
    allow_worker_to_finish = Event()
    close_started = Event()
    prefix = _Prefix()

    class Model:
        def generate(self, **kwargs) -> None:
            streamer = kwargs["streamer"]
            streamer.put(object())
            streamer.put(_TokenIds([1]))
            emitted.set()
            assert allow_worker_to_finish.wait(2), "test did not release generation worker"
            streamer.end()

    _encoded(monkeypatch, prefix)
    SERVE.STATE.update(model=Model())
    stream = SERVE.generate_stream("hello", max_new=2, greedy=True)

    assert next(stream) == {"token": "a"}
    assert emitted.wait(1)

    def close_stream() -> None:
        close_started.set()
        stream.close()

    closer = Thread(target=close_stream)
    closer.start()
    assert close_started.wait(1)
    # close() is blocked in the generator's finally/join while the model still
    # owns the prefix cache; the model lock must remain unavailable too.
    assert closer.is_alive()
    assert not _lock_is_available()
    assert prefix.reset_count == 0

    allow_worker_to_finish.set()
    closer.join(2)

    assert not closer.is_alive()
    assert prefix.reset_count == 1
    assert _lock_is_available()
