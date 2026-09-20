# Qwen chat on Apple Metal

This example serves [Qwen3-0.6B](https://huggingface.co/Qwen/Qwen3-0.6B) through
Dispatch's service gateway. It uses PyTorch MPS and refuses CPU fallback.
It is a compact public chat model, not a custom fine-tune.

Start a Python service worker with:

```sh
WORKER_EXECUTOR=python_project WORKER_RUNTIME=auto uv run --project backend --python 3.12 --extra demo --with torch==2.14.0 --with numpy --with transformers==5.7.0 orchestrator-demo worker-b --port 8080
```

Download the weights once to persistent storage on the worker:

```sh
uv run --project backend --with huggingface-hub python - <<'PY'
from huggingface_hub import snapshot_download
snapshot_download(
    "Qwen/Qwen3-0.6B",
    revision="c1899de289a04d12100db370d81485cdf75e47ca",
    local_dir=".local/models/qwen3-0.6b",
    allow_patterns=["*.json", "*.safetensors", "*.txt", "*.model", "LICENSE", "README.md"],
)
PY
```

Upload `server.py` as a Service with runtime `mps`, zero dedicated VRAM, and
`args: ["--model-path", "/absolute/path/to/.local/models/qwen3-0.6b"]`.
Use a 180-second startup timeout. The readiness endpoint is `/health`; it becomes
available after the model loads and completes a real forward pass. The model
path must exist on every eligible replacement worker.

The service details show its stable access URL. API clients use a fleet bearer
token and append the following paths:

- `GET /serve/{job_id}/health` reports the model and actual device.
- `GET /serve/{job_id}/v1/models` lists the model.
- `POST /serve/{job_id}/v1/chat/completions` accepts text `messages`, `model`,
  `temperature`, `max_tokens`, and `stream`.

```json
{
  "model": "Qwen/Qwen3-0.6B",
  "messages": [{"role": "user", "content": "Say hello in one sentence."}],
  "max_tokens": 128,
  "stream": true
}
```

The example handles one generation at a time, up to 4096 input tokens and 256
output tokens. Concurrent generations return 429. Streaming uses SSE chat chunks
and ends with `[DONE]`; a disconnected streaming client requests generation
cancellation. Health checks remain responsive during inference.

Validated on the local Apple GPU with PyTorch 2.14.0 and Transformers 5.7.0:
model device `mps:0`, JSON chat responses, and multiple streamed chunks through
the gateway. This is a minimal chat-completions subset, not a full API-compatible
inference platform. Stop the service from its detail panel when finished.
