"""Provider contracts with fake transports; these tests never contact model APIs."""

import json
import sys
from types import SimpleNamespace

import pytest

from gpushare.portability.providers import MODELS, ModelClient, ModelError


def response(content='{"ok":true}', finish="stop", usage=True):
    result = SimpleNamespace(choices=[SimpleNamespace(
        finish_reason=finish, message=SimpleNamespace(content=content))])
    if usage:
        result.usage = SimpleNamespace(prompt_tokens=20, completion_tokens=10)
    return result


class Transport:
    def __init__(self, reply=None, error=None):
        self.reply = reply if reply is not None else response()
        self.error = error
        self.calls = []
        self.chat = SimpleNamespace(completions=SimpleNamespace(create=self.create))

    def create(self, **kwargs):
        self.calls.append(kwargs)
        if self.error:
            raise self.error
        return self.reply


def test_usage_summaries_are_snapshots_not_live_history_references():
    client = ModelClient(transport=Transport())
    client.propose("system", {})
    first = client.summary()
    client.propose("system", {})
    client.history[0]["input_tokens"] = 999
    assert first["calls"] == 1
    assert len(first["requests"]) == 1
    assert first["requests"][0]["input_tokens"] == 20


@pytest.mark.parametrize("provider,key_name", [("baseten", "BASETEN_API_KEY"), ("openai", "OPENAI_API_KEY")])
def test_credentials_are_routed_to_their_own_provider_only(tmp_path, monkeypatch, provider, key_name):
    env = tmp_path / "provider.env"
    env.write_text("BASETEN_API_KEY=file-baseten-key\nOPENAI_API_KEY=file-openai-key\n")
    monkeypatch.setenv("BASETEN_API_KEY", "environment-baseten-key")
    monkeypatch.setenv("OPENAI_API_KEY", "environment-openai-key")
    captured = []

    def factory(**kwargs):
        captured.append(kwargs)
        return Transport()

    monkeypatch.setitem(sys.modules, "openai", SimpleNamespace(OpenAI=factory))
    client = ModelClient(provider=provider, env_file=env)
    assert captured[0]["api_key"] == "environment-" + provider + "-key"
    assert captured[0]["base_url"] == MODELS[provider][0]
    assert captured[0]["max_retries"] == 0
    assert client.propose("system", {"input": "fixture"}) == {"ok": True}
    request = client.transport.calls[0]
    expected = "max_tokens" if provider == "baseten" else "max_completion_tokens"
    assert request[expected] == client.max_output_tokens
    assert ("max_completion_tokens" if provider == "baseten" else "max_tokens") not in request
    assert key_name not in json.dumps(request)


@pytest.mark.parametrize("provider", ["baseten", "openai"])
def test_missing_provider_key_does_not_fall_back_to_other_credentials(tmp_path, monkeypatch, provider):
    other = "OPENAI_API_KEY" if provider == "baseten" else "BASETEN_API_KEY"
    monkeypatch.delenv(MODELS[provider][1], raising=False)
    monkeypatch.setenv(other, "unrelated-key")
    monkeypatch.setitem(sys.modules, "openai", SimpleNamespace(OpenAI=lambda **kwargs: pytest.fail("no client should be created")))
    with pytest.raises(ModelError, match="not configured"):
        ModelClient(provider=provider, env_file=tmp_path / "absent.env")


def test_limits_block_requests_before_transport_and_reserve_failed_call_cost():
    transport = Transport(error=RuntimeError("sensitive provider payload"))
    client = ModelClient(transport=transport, max_calls=1)
    with pytest.raises(ModelError):
        client.propose("system", {"input": "fixture"})
    assert client.reserved_cost_usd > 0
    with pytest.raises(ModelError, match="limit reached"):
        client.propose("system", {})
    assert len(transport.calls) == 1
    small = ModelClient(transport=Transport(), budget_usd=0.000001)
    with pytest.raises(ModelError, match="limit reached"):
        small.propose("system", {})
    assert small.transport.calls == []


def test_input_limit_and_accounted_cost_overage_prevent_more_calls():
    client = ModelClient(transport=Transport())
    with pytest.raises(ModelError, match="context size"):
        client.propose("x" * 100001, {})
    assert client.calls == 0
    client.transport.reply.usage.completion_tokens = 10000000
    client.propose("system", {})
    assert client.estimated_cost_usd > client.budget_usd
    with pytest.raises(ModelError, match="limit reached"):
        client.propose("system", {})
    assert len(client.transport.calls) == 1


@pytest.mark.parametrize("reply", [
    response("not JSON"), response("[]"), response("{}", finish="length"),
    response("{}", finish="content_filter"), SimpleNamespace(choices=[]),
    SimpleNamespace(), SimpleNamespace(choices=[SimpleNamespace(finish_reason="stop")]),
])
def test_malformed_or_truncated_output_is_a_safe_model_error(reply):
    client = ModelClient(transport=Transport(reply))
    with pytest.raises(ModelError):
        client.propose("system", {})
    assert client.calls == 1
    assert client.reserved_cost_usd > 0


def test_missing_or_malformed_usage_uses_conservative_estimate_without_echoing_fields():
    secret = "sk-" + "sensitive" * 5
    reply = response(usage=False)
    client = ModelClient(transport=Transport(reply))
    client.propose("system", {})
    assert client.estimated_cost_usd == client.reserved_cost_usd
    reply.usage = SimpleNamespace(prompt_tokens=secret, completion_tokens=None)
    client.propose("system", {})
    assert secret not in json.dumps(client.summary())


def test_provider_error_redacts_message_and_untrusted_status():
    secret = "sk-" + "sensitive" * 5
    error = RuntimeError("request headers Authorization: Bearer " + secret)
    error.status_code = secret
    client = ModelClient(transport=Transport(error=error))
    with pytest.raises(ModelError) as failure:
        client.propose("system", {})
    assert secret not in str(failure.value)
    assert secret not in json.dumps(client.summary())
    assert "unknown" in str(failure.value)


@pytest.mark.parametrize("config", [
    {"timeout_seconds": float("inf")}, {"timeout_seconds": 0}, {"timeout_seconds": 301},
    {"budget_usd": True}, {"max_calls": 1.5}, {"max_calls": True}, {"max_output_tokens": 512.5},
])
def test_invalid_limits_are_rejected_upfront(config):
    with pytest.raises(ValueError):
        ModelClient(transport=Transport(), **config)
