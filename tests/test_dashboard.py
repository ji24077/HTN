import json

from fastapi.testclient import TestClient

from gpushare.dashboard import runner
from gpushare.dashboard.app import build_app


def test_research_console_and_state_are_served():
    client = TestClient(build_app())

    page = client.get("/")
    state = client.get("/api/state")

    assert page.status_code == 200
    assert "연구 파이프라인" in page.text
    assert 'data-testid="run-training"' in page.text
    assert state.status_code == 200
    assert state.json()["experiment"]["model_id"] == "Qwen/Qwen2.5-0.5B"


def test_quality_gate_checks_parse_and_exact_match():
    before = {"json_parse_rate": 1.0, "exact_match_rate": 0.90}

    assert (
        runner._quality(before, {"json_parse_rate": 0.99, "exact_match_rate": 0.89})["status"]
        == "ok"
    )
    assert (
        runner._quality(before, {"json_parse_rate": 1.0, "exact_match_rate": 0.87})["status"]
        == "regressed"
    )


def test_job_public_never_serialises_the_live_process():
    job = runner.Job(id="abc", kind="test", params={})
    job._process = object()  # type: ignore[assignment]

    public = job.public()

    assert "_process" not in public
    json.dumps(public)


def test_pod_api_returns_only_safe_fields(monkeypatch):
    pods = [
        {
            "id": "pod1234",
            "name": "gpu-test",
            "runtimeStatus": "running",
            "costPerHr": 0.74,
            "gpuCount": 1,
            "env": {"SECRET": "must-not-leak"},
        }
    ]
    detail = {
        **pods[0],
        "machine": {"gpuId": "NVIDIA GeForce RTX 4090", "dataCenterId": "EU-RO-1"},
        "env": {"PUBLIC_KEY": "also-not-returned"},
    }

    def fake_capture(args, **_kwargs):
        return json.dumps(detail if "get" in args else pods)

    monkeypatch.setattr(runner, "_capture", fake_capture)
    monkeypatch.setattr(runner, "_pod_cache", (0.0, []))

    result = runner.list_pods(refresh=True)

    assert result == [
        {
            "id": "pod1234",
            "name": "gpu-test",
            "status": "running",
            "gpu": "NVIDIA GeForce RTX 4090",
            "vendor": "nvidia",
            "gpu_count": 1,
            "cost_per_hour": 0.74,
            "datacenter": "EU-RO-1",
            "uptime_seconds": None,
        }
    ]
    assert "SECRET" not in json.dumps(result)


# ─────────────────────────────────────────────────────────────────────────────
# Regressions: the three ways this dashboard could lie
# ─────────────────────────────────────────────────────────────────────────────
def test_logs_never_carry_a_live_key(monkeypatch):
    """Job logs are child stdout, and those children are handed the real API
    keys. One SDK traceback echoing its own credential would put a live key in
    the browser."""
    from gpushare.dashboard import runner

    monkeypatch.setenv("BASETEN_API_KEY", "hcf7ydyb.JPPeAvvJmRyvxVy4QE7bGAw6xjZ3CLL9")
    runner._known_secrets.cache_clear()
    try:
        for leak in (
            "auth failed for sk-proj-AbCdEfGhIjKlMnOpQrStUv123456",
            "RUNPOD_API_KEY=rpa_DTC1M3WRCLQMQUMN0L51S6KTNB1S01SHZ rejected",
            "baseten rejected hcf7ydyb.JPPeAvvJmRyvxVy4QE7bGAw6xjZ3CLL9",
            "token ghp_AbCdEfGhIjKlMnOpQrStUvWxYz0123",
        ):
            assert "REDACTED" in runner._redact(leak)
            assert "sk-proj-AbCdEfGhIjKlMnOpQrStUv123456" not in runner._redact(leak)
            assert "hcf7ydyb.JPPeAvvJmRyvxVy4QE7bGAw6xjZ3CLL9" not in runner._redact(leak)
    finally:
        runner._known_secrets.cache_clear()


def test_ordinary_log_lines_survive_redaction():
    """Over-redacting would make the live log useless, which is its own failure."""
    from gpushare.dashboard.runner import _redact

    for ok in ("step 42 loss 0.0051 0.137s", "json_parse_rate 1.000 exact_match 0.910"):
        assert _redact(ok) == ok


def test_optimization_without_a_reference_run_is_not_validated(tmp_path, monkeypatch):
    """A speed benchmark proves nothing about model quality. With no eval to
    compare against, the action must say `not_validated` — never `ok`. Saying
    `ok` here is the single most damaging thing this dashboard could do."""
    from gpushare.dashboard import runner

    monkeypatch.setattr(runner, "latest_run", lambda: None)
    status, quality = runner._validate_selection(
        runner.Job(id="x", kind="k", params={}), {}, {"dtype": "bf16"}, 3072
    )
    assert status["status"] == "not_validated"
    assert quality is None


def test_quality_gate_rejects_a_regression():
    """The gate that every validated action passes through."""
    from gpushare.dashboard.runner import _quality

    good = {"json_parse_rate": 1.0, "exact_match_rate": 0.91}
    assert _quality(good, good)["status"] == "ok"
    assert (
        _quality(good, {"json_parse_rate": 1.0, "exact_match_rate": 0.40})["status"] == "regressed"
    )
    # Within tolerance is not a regression: generation is not bit-identical
    # across chips, and a migration landing inside noise did not break anything.
    assert _quality(good, {"json_parse_rate": 1.0, "exact_match_rate": 0.90})["status"] == "ok"


def test_request_models_match_the_runner_signatures_they_splat_into():
    """Every request field must be a real parameter of the function it feeds.

    The routes call `generate(**req.model_dump())`, so an added field is a
    TypeError at request time, not at import — and only on the route that was
    not updated. That is exactly how `no_cache` shipped working on /api/generate
    and broken on /api/generate/stream: same request model, two functions, one
    of them forgotten. Pydantic cannot catch it and neither can a smoke test
    that never loads a model, because both fail earlier with 409.
    """
    import inspect

    from gpushare.dashboard.app import GenerateRequest, PrefixRequest

    for model, func in [
        (GenerateRequest, runner.generate),
        (GenerateRequest, runner.generate_stream),
        (PrefixRequest, runner.set_prefix),
    ]:
        params = inspect.signature(func).parameters
        missing = sorted(set(model.model_fields) - set(params))
        assert not missing, f"{func.__name__} is missing {missing}"
