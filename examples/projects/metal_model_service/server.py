"""Small Qwen chat API on Apple Metal. Upload in Service mode with MPS requirements.

Start with --model-path pointing to downloaded Qwen/Qwen3-0.6B weights on the worker.
Requires torch, numpy, and transformers in the worker's Python environment.
Supports text messages, max_tokens, temperature, and optional SSE streaming.
"""

import argparse
import json
import os
import queue
import threading
import time
from http.server import BaseHTTPRequestHandler, ThreadingHTTPServer
from uuid import uuid4

import torch
from transformers import (
    AutoModelForCausalLM,
    AutoTokenizer,
    StoppingCriteria,
    StoppingCriteriaList,
    TextIteratorStreamer,
)

MODEL_ID = "Qwen/Qwen3-0.6B"
parser = argparse.ArgumentParser()
parser.add_argument("--model-path", required=True)
args = parser.parse_args()
if not torch.backends.mps.is_available():
    raise RuntimeError("This service requires Apple Metal/MPS; CPU fallback is disabled.")
print(f"Loading {MODEL_ID} on Apple Metal…", flush=True)
tokenizer = AutoTokenizer.from_pretrained(args.model_path, local_files_only=True)
model = (
    AutoModelForCausalLM.from_pretrained(
        args.model_path, dtype=torch.float16, local_files_only=True
    )
    .to("mps")
    .eval()
)
gate = threading.Lock()

# Readiness includes a real forward pass, not just a loaded checkpoint.
with torch.inference_mode():
    model(**tokenizer("Hello", return_tensors="pt").to("mps"))
torch.mps.synchronize()
print(f"Ready: {MODEL_ID}, device={model.device}", flush=True)


class Cancelled(StoppingCriteria):
    def __init__(self, event):
        self.event = event

    def __call__(self, input_ids, scores, **kwargs):
        return self.event.is_set()


class Handler(BaseHTTPRequestHandler):
    def log_message(self, message, *values):
        if self.path != "/health":
            super().log_message(message, *values)

    def reply(self, status, body):
        self.send_response(status)
        self.send_header("Content-Type", "application/json")
        self.end_headers()
        self.wfile.write(json.dumps(body).encode())

    def do_GET(self):
        if self.path == "/health":
            self.reply(200, {"status": "ready", "model": MODEL_ID, "device": str(model.device)})
        elif self.path == "/v1/models":
            self.reply(
                200,
                {
                    "object": "list",
                    "data": [{"id": MODEL_ID, "object": "model", "owned_by": "local"}],
                },
            )
        else:
            self.reply(404, {"error": "Use POST /v1/chat/completions"})

    def do_POST(self):
        if self.path != "/v1/chat/completions":
            self.reply(404, {"error": "Unknown route"})
            return
        try:
            size = int(self.headers.get("Content-Length", "0"))
            if not 0 < size <= 1024 * 1024:
                raise ValueError("Request body must be between 1 byte and 1 MiB")
            body = json.loads(self.rfile.read(size))
            if not isinstance(body, dict):
                raise ValueError("Expected a JSON object")
            if body.get("model", MODEL_ID) != MODEL_ID:
                raise ValueError(f"Available model: {MODEL_ID}")
            messages = body.get("messages")
            if not isinstance(messages, list) or not 1 <= len(messages) <= 32:
                raise ValueError("Provide 1 to 32 text messages")
            for message in messages:
                if (
                    not isinstance(message, dict)
                    or message.get("role") not in {"system", "user", "assistant"}
                    or not isinstance(message.get("content"), str)
                ):
                    raise ValueError("Messages require a role and text content")
            maximum = body.get("max_tokens", 128)
            if type(maximum) is not int or not 1 <= maximum <= 256:
                raise ValueError("max_tokens must be an integer from 1 to 256")
            temperature = body.get("temperature", 0.7)
            if type(temperature) not in {int, float} or not 0 <= temperature <= 2:
                raise ValueError("temperature must be from 0 to 2")
            streaming = body.get("stream", False)
            if type(streaming) is not bool:
                raise ValueError("stream must be a boolean")
            prompt = tokenizer.apply_chat_template(
                messages, tokenize=False, add_generation_prompt=True, enable_thinking=False
            )
            inputs = tokenizer(prompt, return_tensors="pt")
            if inputs.input_ids.shape[1] > 4096:
                raise ValueError("This demo accepts at most 4096 input tokens")
        except (ValueError, TypeError, KeyError) as error:
            self.reply(400, {"error": str(error)})
            return
        if not gate.acquire(blocking=False):
            self.reply(429, {"error": "Model is busy; retry when the current response finishes"})
            return

        stop = threading.Event()
        done = threading.Event()
        errors = []
        generated = []
        streamer = TextIteratorStreamer(
            tokenizer, skip_prompt=True, skip_special_tokens=True, timeout=1
        )
        identity = "chatcmpl-" + uuid4().hex
        created = int(time.time())

        def generate():
            try:
                options = {"do_sample": temperature > 0}
                if temperature > 0:
                    options.update(temperature=temperature, top_p=0.8, top_k=20)
                with torch.inference_mode():
                    result = model.generate(
                        **inputs.to("mps"),
                        max_new_tokens=maximum,
                        streamer=streamer,
                        stopping_criteria=StoppingCriteriaList([Cancelled(stop)]),
                        **options,
                    )
                generated.append(result.shape[1] - inputs.input_ids.shape[1])
            except Exception as error:
                errors.append(str(error))
            finally:
                done.set()
                gate.release()

        thread = threading.Thread(target=generate, daemon=True)
        thread.start()
        try:
            if streaming:
                self.send_response(200)
                self.send_header("Content-Type", "text/event-stream")
                self.send_header("Cache-Control", "no-cache")
                self.end_headers()
            parts = []
            while True:
                try:
                    part = next(streamer)
                except StopIteration:
                    break
                except queue.Empty:
                    if done.is_set():
                        break
                    continue
                parts.append(part)
                if streaming and part:
                    self.event(
                        {
                            "id": identity,
                            "object": "chat.completion.chunk",
                            "created": created,
                            "model": MODEL_ID,
                            "choices": [
                                {"index": 0, "delta": {"content": part}, "finish_reason": None}
                            ],
                        }
                    )
            thread.join()
            if errors:
                if streaming:
                    self.event({"error": errors[0]})
                else:
                    self.reply(500, {"error": errors[0]})
                return
            reason = "length" if generated and generated[0] >= maximum else "stop"
            if streaming:
                self.event(
                    {
                        "id": identity,
                        "object": "chat.completion.chunk",
                        "created": created,
                        "model": MODEL_ID,
                        "choices": [{"index": 0, "delta": {}, "finish_reason": reason}],
                    }
                )
                self.wfile.write(b"data: [DONE]\n\n")
                self.wfile.flush()
            else:
                self.reply(
                    200,
                    {
                        "id": identity,
                        "object": "chat.completion",
                        "created": created,
                        "model": MODEL_ID,
                        "choices": [
                            {
                                "index": 0,
                                "message": {"role": "assistant", "content": "".join(parts)},
                                "finish_reason": reason,
                            }
                        ],
                    },
                )
        except (BrokenPipeError, ConnectionResetError):
            pass
        finally:
            stop.set()
            thread.join(timeout=5)

    def event(self, body):
        self.wfile.write(("data: " + json.dumps(body) + "\n\n").encode())
        self.wfile.flush()


ThreadingHTTPServer(
    ("127.0.0.1", int(os.environ["DISPATCH_SERVICE_PORT"])), Handler
).serve_forever()
