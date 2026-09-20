"""CPU-thread checks for shared cache safety; no server or GPU is started."""

import importlib.util
from pathlib import Path
from threading import Event, Thread
from types import SimpleNamespace

import pytest


@pytest.fixture
def server(monkeypatch):
    torch = pytest.importorskip("torch")
    path = Path(__file__).parents[1] / "scripts/serve.py"
    spec = importlib.util.spec_from_file_location("serving_concurrency_test", path)
    module = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(module)
    monkeypatch.setattr(torch.cuda, "synchronize", lambda: None)
    monkeypatch.setattr(torch.cuda, "empty_cache", lambda: None)
    resets = []
    cache = SimpleNamespace(cache=object(), reset=lambda: resets.append(True))
    tokenizer = SimpleNamespace(pad_token_id=0, decode=lambda *a, **k: "token")
    module.STATE.update(tok=tokenizer, model_ref="test-model", dtype="bf16", prefix=cache)
    monkeypatch.setattr(module, "encode_for", lambda *a, **k: (torch.tensor([[1]]), cache, {}))
    return module, torch, resets


def test_disconnect_waits_for_generation_before_prefix_can_change(server):
    serve, torch, resets = server
    finish_generation = Event()
    prefix_finished = Event()
    close_finished = Event()

    def generate(**kwargs):
        kwargs["streamer"].put(torch.tensor([[1]]))
        kwargs["streamer"].put(torch.tensor([2]))
        assert finish_generation.wait(2), "test generator was not released"

    serve.STATE["model"] = SimpleNamespace(generate=generate)
    stream = serve.generate_stream("input", max_new=2, greedy=True)
    assert next(stream) == {"token": "token"}

    def close():
        stream.close()
        close_finished.set()

    closer = Thread(target=close, daemon=True)
    closer.start()
    handler = SimpleNamespace(_send=lambda *a: prefix_finished.set())
    prefix = Thread(target=serve.Handler._prefix, args=(handler, {}), daemon=True)
    prefix.start()
    try:
        assert not prefix_finished.wait(0.05), "prefix changed while GPU writer still active"
        assert not close_finished.is_set()
    finally:
        finish_generation.set()
        closer.join(2)
        prefix.join(2)
    assert close_finished.is_set() and prefix_finished.is_set()
    assert resets == [True]


def test_generation_failure_reaches_stream_consumer_and_releases_cache(server):
    serve, _, resets = server
    def fail(**kwargs):
        raise RuntimeError("fake generation failure")
    serve.STATE["model"] = SimpleNamespace(generate=fail)
    completed = Event()
    failures = []
    def consume():
        try:
            list(serve.generate_stream("input", max_new=2, greedy=True))
        except RuntimeError as exc:
            failures.append(str(exc))
        finally:
            completed.set()
    consumer = Thread(target=consume, daemon=True)
    consumer.start()
    assert completed.wait(2), "generation error stranded streaming queue"
    assert failures == ["fake generation failure"]
    assert resets == [True]
    assert not serve._MODEL_LOCK.locked()


def test_nonstream_failure_resets_cache_before_releasing_lock(server):
    serve, _, resets = server
    def fail(**kwargs):
        raise RuntimeError("fake generation failure")
    serve.STATE["model"] = SimpleNamespace(generate=fail)
    with pytest.raises(RuntimeError, match="fake generation failure"):
        serve.generate("input", max_new=2, greedy=True)
    assert resets == [True]
    assert not serve._MODEL_LOCK.locked()
