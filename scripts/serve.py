"""Keep one model resident on the pod and answer generate requests.

    uv run python scripts/serve.py --model ckpt/run --port 8100

Loading Qwen takes ten to twenty seconds. A chat box that paid that on every
keystroke-to-answer would be unusable and would also make every latency number
meaningless, so the model is loaded once and held.

Binds to LOCALHOST ONLY. The pod exposes port 22 and nothing else; the
dashboard reaches this through an SSH local forward. Nothing here is
authenticated, so it must never be reachable from outside the pod.

WHAT THE LATENCY HERE DOES AND DOES NOT SHOW
  It shows what one interactive request costs — the number a person feels.
  It does NOT show the inference optimization. That optimization is about
  concurrency: at batch 1 there is nothing to batch, and the handbook says so
  directly. scripts/benchmark_inference.py is what measures throughput, at 64
  concurrent. Reporting a single-prompt latency as evidence that batching made
  things faster would be a lie the UI must not tell.
"""

from __future__ import annotations

import argparse
import contextlib
import json
import queue
import sys
import time
from http.server import BaseHTTPRequestHandler, ThreadingHTTPServer
from pathlib import Path
from threading import Thread

import torch

from gpushare.agent.task import MODEL_ID, PROMPT, parse_output

DTYPES = {"bf16": torch.bfloat16, "fp16": torch.float16, "fp32": torch.float32}
STATE: dict = {}


def load(model_ref: str, dtype: str):
    """A HF id, or a directory holding model.safetensors from scripts/train.py."""
    from transformers import AutoModelForCausalLM, AutoTokenizer

    td = DTYPES[dtype]
    tok = AutoTokenizer.from_pretrained(MODEL_ID, padding_side="left")
    if tok.pad_token_id is None:
        tok.pad_token = tok.eos_token

    p = Path(model_ref)
    if (p / "model.safetensors").exists():
        from safetensors.torch import load_file

        model = AutoModelForCausalLM.from_pretrained(MODEL_ID, dtype=td)
        missing, unexpected = model.load_state_dict(
            load_file(p / "model.safetensors"), strict=False
        )
        if missing or unexpected:
            # Loud: a partial load looks like a bad training run and would get
            # blamed on the config or the chip.
            print(
                f"state_dict: {len(missing)} missing, {len(unexpected)} unexpected",
                file=sys.stderr,
            )
    else:
        model = AutoModelForCausalLM.from_pretrained(model_ref, dtype=td)

    model = model.cuda().eval()
    return model, tok


@torch.no_grad()
def generate(sentence: str, *, max_new: int, greedy: bool) -> dict:
    model, tok = STATE["model"], STATE["tok"]
    prompt = PROMPT.format(sentence=sentence)
    enc = tok(prompt, return_tensors="pt", add_special_tokens=False)
    enc = {k: v.cuda() for k, v in enc.items()}

    torch.cuda.synchronize()
    t0 = time.perf_counter()
    out = model.generate(
        **enc,
        max_new_tokens=max_new,
        do_sample=not greedy,
        temperature=0.7 if not greedy else None,
        pad_token_id=tok.pad_token_id,
    )
    torch.cuda.synchronize()
    dt = time.perf_counter() - t0

    gen = out[0, enc["input_ids"].shape[1] :]
    raw = tok.decode(gen, skip_special_tokens=True)
    parsed = parse_output(raw)
    n_new = int(gen.shape[0])
    return {
        "prompt": prompt,
        "raw_output": raw,
        "parsed": parsed.model_dump() if parsed else None,
        "parsed_ok": parsed is not None,
        "new_tokens": n_new,
        "latency_s": dt,
        # Interactive latency at batch 1. NOT the throughput number — see the
        # module docstring.
        "tokens_per_s": n_new / dt if dt else 0.0,
        "model": STATE["model_ref"],
        "dtype": STATE["dtype"],
        "greedy": greedy,
    }


class TokenStreamer:
    """Emit every token the moment it exists.

    transformers' TextIteratorStreamer buffers to word boundaries, and a
    compact JSON record has almost none — measured on the pod, the entire
    answer arrived in THREE chunks. That reads as "it hung, then finished",
    not as generation, which defeats the point of streaming at this output
    length.

    Decoding the whole running prefix and emitting the delta (rather than
    decoding each id alone) is what keeps multi-byte characters intact: a
    token that is half a UTF-8 sequence produces no delta until the rest
    arrives, instead of a replacement character.
    """

    def __init__(self, tok):
        self._tok = tok
        self._q: queue.Queue = queue.Queue()
        self._ids: list[int] = []
        self._text = ""
        self._prompt_seen = False

    def put(self, value) -> None:
        # generate() calls put() once with the whole prompt before any new
        # token; echoing it back would replay the input into the answer.
        if not self._prompt_seen:
            self._prompt_seen = True
            return
        ids = value.tolist()
        if ids and isinstance(ids[0], list):
            ids = ids[0]
        self._ids.extend(ids)
        full = self._tok.decode(self._ids, skip_special_tokens=True)
        delta, self._text = full[len(self._text) :], full
        if delta:
            self._q.put(delta)

    def end(self) -> None:
        self._q.put(None)

    def __iter__(self):
        while True:
            item = self._q.get()
            if item is None:
                return
            yield item


@torch.no_grad()
def generate_stream(sentence: str, *, max_new: int, greedy: bool):
    """Yield tokens as they are produced, then a final summary frame.

    TIME TO FIRST TOKEN IS THE POINT. Total latency is what the GPU cost;
    TTFT is what a person experiences as "it started answering". They are
    different numbers and only one of them is why streaming feels fast, so
    both are reported and the UI shows both.

    Greedy still, so the streamed text is byte-identical to what the blocking
    endpoint returns — streaming is presentation, never a different answer.
    """
    model, tok = STATE["model"], STATE["tok"]
    prompt = PROMPT.format(sentence=sentence)
    enc = tok(prompt, return_tensors="pt", add_special_tokens=False)
    enc = {k: v.cuda() for k, v in enc.items()}

    streamer = TokenStreamer(tok)
    kwargs = dict(
        **enc,
        max_new_tokens=max_new,
        do_sample=not greedy,
        pad_token_id=tok.pad_token_id,
        streamer=streamer,
    )
    if not greedy:
        kwargs["temperature"] = 0.7

    torch.cuda.synchronize()
    t0 = time.perf_counter()
    # generate() blocks, so it runs on its own thread and the streamer is the
    # channel. Without this there is nothing to iterate until it has finished,
    # which is exactly the behaviour we are removing.
    thread = Thread(target=model.generate, kwargs=kwargs, daemon=True)
    thread.start()

    ttft = None
    pieces = []
    for text in streamer:
        if not text:
            continue
        if ttft is None:
            ttft = time.perf_counter() - t0
        pieces.append(text)
        yield {"token": text}

    thread.join()
    raw = "".join(pieces)
    dt = time.perf_counter() - t0
    parsed = parse_output(raw)
    yield {
        "done": True,
        "raw_output": raw,
        "parsed": parsed.model_dump() if parsed else None,
        "parsed_ok": parsed is not None,
        "ttft_s": ttft,
        "latency_s": dt,
        "model": STATE["model_ref"],
        "dtype": STATE["dtype"],
    }


class Handler(BaseHTTPRequestHandler):
    def _send(self, code: int, payload: dict) -> None:
        body = json.dumps(payload, ensure_ascii=False).encode()
        self.send_response(code)
        self.send_header("Content-Type", "application/json; charset=utf-8")
        self.send_header("Content-Length", str(len(body)))
        self.end_headers()
        self.wfile.write(body)

    def do_GET(self):  # noqa: N802 — BaseHTTPRequestHandler's naming
        if self.path == "/health":
            self._send(200, {"ok": True, "model": STATE["model_ref"], "dtype": STATE["dtype"]})
        else:
            self._send(404, {"error": "not found"})

    def _sse(self, req: dict) -> None:
        sentence = str(req.get("sentence", "")).strip()
        if not sentence:
            self._send(400, {"error": "sentence is required"})
            return
        self.send_response(200)
        self.send_header("Content-Type", "text/event-stream")
        # No Content-Length on purpose: the body length is unknown until the
        # last token. Declaring one here is what would make this buffer.
        self.send_header("Cache-Control", "no-cache")
        self.send_header("X-Accel-Buffering", "no")
        self.end_headers()
        try:
            for frame in generate_stream(
                sentence,
                max_new=min(int(req.get("max_new_tokens", 64)), 256),
                greedy=bool(req.get("greedy", True)),
            ):
                self.wfile.write(f"data: {json.dumps(frame, ensure_ascii=False)}\n\n".encode())
                self.wfile.flush()
        except (BrokenPipeError, ConnectionResetError):
            pass  # the browser navigated away mid-stream; nothing to report
        except Exception as e:  # noqa: BLE001
            with contextlib.suppress(Exception):
                err = json.dumps({"done": True, "error": f"{type(e).__name__}: {e}"})
                self.wfile.write(f"data: {err}\n\n".encode())
                self.wfile.flush()

    def do_POST(self):  # noqa: N802
        if self.path not in ("/generate", "/generate/stream"):
            self._send(404, {"error": "not found"})
            return
        try:
            n = int(self.headers.get("Content-Length", 0))
            req = json.loads(self.rfile.read(n) or b"{}")
            if self.path == "/generate/stream":
                self._sse(req)
                return
            sentence = str(req.get("sentence", "")).strip()
            if not sentence:
                self._send(400, {"error": "sentence is required"})
                return
            self._send(
                200,
                generate(
                    sentence,
                    max_new=min(int(req.get("max_new_tokens", 64)), 256),
                    greedy=bool(req.get("greedy", True)),
                ),
            )
        except Exception as e:  # noqa: BLE001 — one bad request must not kill the server
            self._send(500, {"error": f"{type(e).__name__}: {e}"})

    def log_message(self, *args):
        pass  # the dashboard already logs; this would double every line


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--model", default=MODEL_ID, help="HF id or a train.py output dir")
    ap.add_argument("--dtype", choices=sorted(DTYPES), default="bf16")
    ap.add_argument("--port", type=int, default=8100)
    a = ap.parse_args()

    if not torch.cuda.is_available():
        raise SystemExit("no GPU visible — run `make check` first")

    print(f"loading {a.model} ({a.dtype})...", file=sys.stderr, flush=True)
    model, tok = load(a.model, a.dtype)
    STATE.update(model=model, tok=tok, model_ref=a.model, dtype=a.dtype)

    # Threading: a stream holds its connection open for the whole generation,
    # and a single-threaded server would make /health time out behind it —
    # which the dashboard reads as "the model never became ready".
    # 127.0.0.1 on purpose: unauthenticated, reached only through an SSH forward.
    server = ThreadingHTTPServer(("127.0.0.1", a.port), Handler)
    print(f"READY on 127.0.0.1:{a.port}", file=sys.stderr, flush=True)
    server.serve_forever()


if __name__ == "__main__":
    main()
