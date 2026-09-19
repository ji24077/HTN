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

# Nothing is appended after the dynamic text. It used to be "\nJSON:\n", which
# silently forced one output shape: the same cached policy has to serve both a
# short deterministic benchmark answer and a long structured report, and the
# instruction that picks between them belongs to the caller. It is ~40 tokens
# against a 30,875-token prefix, so leaving it uncached costs nothing.
SUFFIX = ""


class PrefixCache:
    """KV for the static head of the prompt, computed once and reused.

    This is request-level prefix caching, which is a different thing from a
    static KV cache: a static cache avoids reallocation *within* one generation,
    while this avoids recomputing the shared prefill *across* generations. Only
    the second one removes seconds from a long-context workload.

    The prefix must be the HEAD of the prompt. Attention is causal, so a cached
    key/value is only valid if every token before it is unchanged — put the
    dynamic text first and nothing is reusable.
    """

    def __init__(self, model, tok, text: str) -> None:
        self.text = text
        ids = tok(text, return_tensors="pt", add_special_tokens=False)["input_ids"]
        self.ids = ids.cuda()
        self.n = int(self.ids.shape[1])
        torch.cuda.synchronize()
        t0 = time.perf_counter()
        with torch.no_grad():
            self.cache = model(input_ids=self.ids, use_cache=True).past_key_values
        torch.cuda.synchronize()
        self.build_s = time.perf_counter() - t0

    def matches(self, full_ids) -> bool:
        """Token-identity, not string prefix.

        BPE can merge across the seam, so `prefix + dynamic` does not always
        tokenize to `tokens(prefix) + tokens(dynamic)`. When it does not, the
        cached keys belong to different tokens than the ones the model is about
        to attend over, and the output silently changes. Checking ids is the
        only safe test; a str.startswith would pass and be wrong.
        """
        return full_ids.shape[1] > self.n and bool(
            torch.equal(full_ids[:, : self.n], self.ids)
        )

    def reset(self) -> None:
        """Drop everything generate() appended, keeping the prefix.

        crop() is why a request does not have to copy the cache. On a 30K
        prefix the KV is several GB; deep-copying it per request would cost
        more than the prefill it saves.
        """
        self.cache.crop(self.n)


def load(model_ref: str, dtype: str):
    """A HF id, or a directory holding model.safetensors from scripts/train.py."""
    from transformers import AutoModelForCausalLM, AutoTokenizer

    td = DTYPES[dtype]
    p = Path(model_ref)
    ours = (p / "model.safetensors").exists()
    # The tokenizer has to follow the weights. Our own checkpoints are Qwen2.5-0.5B
    # weights in a bare directory with no tokenizer files, so they borrow MODEL_ID's;
    # anything else brings its own. Hardcoding MODEL_ID here silently mistokenised
    # every other model — with a different vocab that is not an error, just wrong
    # text, which is the worst way for this to fail.
    tok = AutoTokenizer.from_pretrained(MODEL_ID if ours else model_ref, padding_side="left")
    if tok.pad_token_id is None:
        tok.pad_token = tok.eos_token

    if ours:
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


def build_prompt(sentence: str) -> tuple[str, PrefixCache | None]:
    """Static head first, dynamic text last — the order the cache requires."""
    pc = STATE.get("prefix")
    if pc is None:
        return PROMPT.format(sentence=sentence), None
    return pc.text + sentence + SUFFIX, pc


def encode_for(sentence: str, *, no_cache: bool = False):
    """Returns (input_ids, prefix_cache_or_None_if_miss, token breakdown).

    `no_cache` bypasses the cache while building the SAME prompt. Without it
    the only way to get an uncached baseline is to unset the prefix, which
    changes the prompt too — and then a difference in output cannot be
    attributed to the cache rather than to the wording. A speed claim needs the
    two runs to differ in exactly one thing.
    """
    tok = STATE["tok"]
    prompt, pc = build_prompt(sentence)
    ids = tok(prompt, return_tensors="pt", add_special_tokens=False)["input_ids"].cuda()
    hit = (not no_cache) and pc is not None and pc.matches(ids)
    info = {
        "prompt_tokens": int(ids.shape[1]),
        "prefix_tokens": pc.n if pc else 0,
        "dynamic_tokens": int(ids.shape[1]) - (pc.n if pc else 0),
        "cache": "hit" if hit else ("bypassed" if (pc and no_cache) else ("miss" if pc else "off")),
    }
    return ids, (pc if hit else None), info


@torch.no_grad()
def generate(sentence: str, *, max_new: int, greedy: bool, no_cache: bool = False) -> dict:
    model, tok = STATE["model"], STATE["tok"]
    ids, pc, info = encode_for(sentence, no_cache=no_cache)
    prompt = tok.decode(ids[0], skip_special_tokens=False)
    enc = {"input_ids": ids}

    torch.cuda.synchronize()
    t0 = time.perf_counter()
    kw = {"past_key_values": pc.cache} if pc else {}
    out = model.generate(
        **enc,
        **kw,
        max_new_tokens=max_new,
        do_sample=not greedy,
        temperature=0.7 if not greedy else None,
        pad_token_id=tok.pad_token_id,
    )
    torch.cuda.synchronize()
    dt = time.perf_counter() - t0
    if pc:
        pc.reset()

    gen = out[0, enc["input_ids"].shape[1] :]
    raw = tok.decode(gen, skip_special_tokens=True)
    parsed = parse_output(raw)
    n_new = int(gen.shape[0])
    return {
        "prompt": prompt,
        # Reported because prefill cost is driven by this number, not by the
        # character count a caller can see. Without it, a long-context latency
        # reading cannot be checked against the chip's FLOP ceiling — and an
        # impossible reading (more TFLOPS than the GPU has) is the signal that
        # the input was silently truncated rather than that serving got fast.
        "prompt_tokens": int(enc["input_ids"].shape[1]),
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
        **info,
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

    def ids_seen(self) -> list[int]:
        """Token count for the report panel — the length of what was generated.

        Counted from ids rather than from the decoded string: one token is not
        one character, and a "300-token report" claim has to come from tokens.
        """
        return self._ids

    def end(self) -> None:
        self._q.put(None)

    def __iter__(self):
        while True:
            item = self._q.get()
            if item is None:
                return
            yield item


@torch.no_grad()
def generate_stream(sentence: str, *, max_new: int, greedy: bool, no_cache: bool = False):
    """Yield tokens as they are produced, then a final summary frame.

    TIME TO FIRST TOKEN IS THE POINT. Total latency is what the GPU cost;
    TTFT is what a person experiences as "it started answering". They are
    different numbers and only one of them is why streaming feels fast, so
    both are reported and the UI shows both.

    Greedy still, so the streamed text is byte-identical to what the blocking
    endpoint returns — streaming is presentation, never a different answer.
    """
    model, tok = STATE["model"], STATE["tok"]
    ids, pc, info = encode_for(sentence, no_cache=no_cache)

    streamer = TokenStreamer(tok)
    kwargs = dict(
        input_ids=ids,
        max_new_tokens=max_new,
        do_sample=not greedy,
        pad_token_id=tok.pad_token_id,
        streamer=streamer,
    )
    if pc:
        kwargs["past_key_values"] = pc.cache
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
    if pc:
        pc.reset()
    raw = "".join(pieces)
    dt = time.perf_counter() - t0
    parsed = parse_output(raw)
    yield {
        "done": True,
        "raw_output": raw,
        "new_tokens": len(streamer.ids_seen()),
        "parsed": parsed.model_dump() if parsed else None,
        "parsed_ok": parsed is not None,
        # ttft is prefill plus one decode step, not prefill alone. Naming it
        # "prefill" would overstate how much prefix caching removes.
        "ttft_s": ttft,
        "latency_s": dt,
        "decode_s": (dt - ttft) if ttft is not None else None,
        "model": STATE["model_ref"],
        "dtype": STATE["dtype"],
        **info,
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
            pc = STATE.get("prefix")
            self._send(
                200,
                {
                    "ok": True,
                    "model": STATE["model_ref"],
                    "dtype": STATE["dtype"],
                    "prefix_tokens": pc.n if pc else 0,
                    "prefix_build_s": pc.build_s if pc else None,
                },
            )
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
                no_cache=bool(req.get("no_cache", False)),
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

    def _prefix(self, req: dict) -> None:
        text = req.get("prefix")
        # Drop the old cache BEFORE building the new one. Holding both is two
        # multi-GB KV tensors plus the new prefill's activations, which OOMs on
        # a 24 GB card — reloading the policy, the most ordinary thing to do
        # twice, was killing the server.
        STATE.pop("prefix", None)
        torch.cuda.empty_cache()
        if not text:
            self._send(200, {"prefix_tokens": 0, "cleared": True})
            return
        pc = PrefixCache(STATE["model"], STATE["tok"], str(text))
        STATE["prefix"] = pc
        # build_s IS the cold-path prefill for this prefix — the number the
        # cache removes. Reported so the speedup can be stated from a measured
        # baseline instead of a remembered one.
        self._send(200, {"prefix_tokens": pc.n, "build_s": pc.build_s, "cleared": False})

    def do_POST(self):  # noqa: N802
        if self.path not in ("/generate", "/generate/stream", "/prefix"):
            self._send(404, {"error": "not found"})
            return
        try:
            n = int(self.headers.get("Content-Length", 0))
            req = json.loads(self.rfile.read(n) or b"{}")
            if self.path == "/prefix":
                self._prefix(req)
                return
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
                    no_cache=bool(req.get("no_cache", False)),
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
