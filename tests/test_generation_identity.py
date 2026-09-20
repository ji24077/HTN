import hashlib
import json
from urllib import request

import pytest

from gpushare.dashboard import runner

MODEL_ID = "version-qwen-4b"
POD_ID = "pod-3090"
MODEL_REVISION = "cdbee75f17c01a7cc42f958dc650907174af0554"
BASE_MODEL = "Qwen/Qwen3-4B-Instruct-2507"
MANIFEST_SHA256 = "a" * 64
PROMPT_TEMPLATE = "You are Relay.\n{sentence}"
PREFIX_IDENTITY_SHA256 = "b" * 64


def _identity(**overrides):
    identity = {
        "model_id": MODEL_ID,
        "pod_id": POD_ID,
        "model_revision": MODEL_REVISION,
        "base_model": BASE_MODEL,
        "artifact_manifest_sha256": MANIFEST_SHA256,
        "prompt_template_sha256": hashlib.sha256(PROMPT_TEMPLATE.encode()).hexdigest(),
        "prefix_identity_sha256": PREFIX_IDENTITY_SHA256,
    }
    identity.update(overrides)
    return identity


def _expected_identity_arguments():
    return {
        "expected_model_id": MODEL_ID,
        "expected_pod_id": POD_ID,
        "expected_model_revision": MODEL_REVISION,
        "expected_base_model": BASE_MODEL,
        "expected_artifact_manifest_sha256": MANIFEST_SHA256,
        "expected_prompt_template": PROMPT_TEMPLATE,
        "expected_prefix_identity_sha256": PREFIX_IDENTITY_SHA256,
    }


@pytest.fixture
def exact_resident_deployment(monkeypatch):
    monkeypatch.setattr(
        runner,
        "_serve",
        {
            "model_id": MODEL_ID,
            "pod_id": POD_ID,
            "model_revision": MODEL_REVISION,
            "base_model": BASE_MODEL,
            "artifact_manifest_sha256": MANIFEST_SHA256,
            "prompt_template": PROMPT_TEMPLATE,
        },
    )


class _BlockingResponse:
    def __init__(self, payload):
        self.payload = payload

    def __enter__(self):
        return self

    def __exit__(self, *_args):
        return False

    def read(self):
        return json.dumps(self.payload).encode()


class _StreamingResponse:
    def __init__(self, frames):
        self.frames = frames
        self.frames_read = 0

    def __enter__(self):
        return self

    def __exit__(self, *_args):
        return False

    def __iter__(self):
        for frame in self.frames:
            self.frames_read += 1
            yield f"data: {json.dumps(frame)}\n\n".encode()


def test_generate_accepts_output_from_exact_deployment_identity(
    monkeypatch, exact_resident_deployment
):
    payload = {
        **_identity(),
        "raw_output": "verified answer",
        "output_tokens": 2,
    }
    monkeypatch.setattr(request, "urlopen", lambda *_args, **_kwargs: _BlockingResponse(payload))

    result = runner.generate(
        sentence="hello",
        max_new_tokens=16,
        **_expected_identity_arguments(),
    )

    assert result["raw_output"] == "verified answer"
    assert result["model_id"] == MODEL_ID


def test_generate_rejects_output_from_mismatched_deployment_identity(
    monkeypatch, exact_resident_deployment
):
    payload = {
        **_identity(pod_id="different-pod"),
        "raw_output": "must never be accepted",
    }
    monkeypatch.setattr(request, "urlopen", lambda *_args, **_kwargs: _BlockingResponse(payload))

    with pytest.raises(runner.JobError, match="different deployment identity"):
        runner.generate(
            sentence="hello",
            max_new_tokens=16,
            **_expected_identity_arguments(),
        )


def test_generate_stream_accepts_tokens_after_exact_identity(
    monkeypatch, exact_resident_deployment
):
    response = _StreamingResponse(
        [
            {"identity": True, **_identity()},
            {"done": False, "token": "verified"},
            {"done": True, "raw_output": "verified answer", **_identity()},
        ]
    )
    monkeypatch.setattr(request, "urlopen", lambda *_args, **_kwargs: response)

    frames = [
        json.loads(frame)
        for frame in runner.generate_stream(
            sentence="hello",
            max_new_tokens=16,
            **_expected_identity_arguments(),
        )
    ]

    assert [frame.get("token") for frame in frames if not frame.get("done")] == ["verified"]
    assert frames[-1]["raw_output"] == "verified answer"
    assert not any("error" in frame for frame in frames)


def test_generate_stream_rejects_mismatched_identity_before_reading_any_token(
    monkeypatch, exact_resident_deployment
):
    response = _StreamingResponse(
        [
            {"identity": True, **_identity(model_id="wrong-version")},
            {"done": False, "token": "must never be accepted"},
            {"done": True, "raw_output": "must never be accepted", **_identity()},
        ]
    )
    monkeypatch.setattr(request, "urlopen", lambda *_args, **_kwargs: response)

    frames = [
        json.loads(frame)
        for frame in runner.generate_stream(
            sentence="hello",
            max_new_tokens=16,
            **_expected_identity_arguments(),
        )
    ]

    assert response.frames_read == 1
    assert frames == [
        {
            "done": True,
            "error": "the chat connection reached a different deployment identity",
        }
    ]
    assert "must never be accepted" not in json.dumps(frames)
