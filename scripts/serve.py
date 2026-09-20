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
import hashlib
import json
import queue
import sys
import time
import traceback
from http.server import BaseHTTPRequestHandler, ThreadingHTTPServer
from pathlib import Path
from threading import Lock, Thread

import torch

from gpushare.agent.task import MODEL_ID, PROMPT, parse_output
from gpushare.agent.timing import TimingTokenCollector
from gpushare.artifacts import ArtifactIntegrityError, verify_relay_adapter_contract

DTYPES = {"bf16": torch.bfloat16, "fp16": torch.float16, "fp32": torch.float32}
STATE: dict = {}
# ThreadingHTTPServer runs each request on its own thread so /health never
# blocks behind a stream (see main()). Nothing else made GPU access or
# STATE["prefix"] exclusive, so two overlapping requests — a /prefix reload
# racing a /generate/stream, or two "hit" generations sharing the same
# pc.cache — could crop() and read the same KV tensor at once and corrupt
# each other's output. One lock around every model/STATE touch serializes
# requests instead.
_MODEL_LOCK = Lock()

# How the KV cache is allocated during generation. None is transformers'
# default: a fresh dynamic cache that grows a step at a time. "static"
# preallocates it, which on a measured RTX 4090 took batch-1 median latency
# from 0.399s to 0.130s with p95 equally steady.
#
# It lives in STATE rather than in a launch flag because the point is to turn
# it on WITHOUT reloading the model: a deployment that must be recreated to be
# optimised has not been optimised, it has been replaced.
CACHE_MODES = {None, "static"}


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

    def __init__(self, model, tok, text: str, *, suffix: str = "") -> None:
        self.text = text
        # The tail is part of the prompt contract even though it cannot be in
        # the cached head: dynamic request text sits between the two. Keeping
        # it beside the cached head makes the applied server reproduce the
        # exact template used by the benchmark (for example ``Q: ...\nA:``).
        self.suffix = suffix
        identity = json.dumps(
            {"prefix": text, "suffix": suffix},
            ensure_ascii=False,
            sort_keys=True,
            separators=(",", ":"),
        )
        self.identity_sha256 = hashlib.sha256(identity.encode("utf-8")).hexdigest()
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
        return full_ids.shape[1] > self.n and bool(torch.equal(full_ids[:, : self.n], self.ids))

    def reset(self) -> None:
        """Drop everything generate() appended, keeping the prefix.

        crop() is why a request does not have to copy the cache. On a 30K
        prefix the KV is several GB; deep-copying it per request would cost
        more than the prefill it saves.
        """
        self.cache.crop(self.n)


def load(
    model_ref: str,
    dtype: str,
    *,
    revision: str | None = None,
    expected_base_model: str | None = None,
    expected_artifact_manifest_sha256: str | None = None,
):
    """Load a Hugging Face model, a full bundle, or a PEFT LoRA adapter."""
    from transformers import AutoModelForCausalLM, AutoTokenizer

    td = DTYPES[dtype]
    p = Path(model_ref)
    ours = (p / "model.safetensors").exists()
    adapter = (p / "adapter_config.json").exists()
    # The tokenizer has to follow the weights. Our own checkpoints are Qwen2.5-0.5B
    # weights in a bare directory with no tokenizer files, so they borrow MODEL_ID's;
    # anything else brings its own. Hardcoding MODEL_ID here silently mistokenised
    # every other model — with a different vocab that is not an error, just wrong
    # text, which is the worst way for this to fail.
    if adapter:
        from peft import PeftConfig

        manifest = None
        if expected_artifact_manifest_sha256 or (p / "relay-manifest.json").is_file():
            try:
                manifest, _training_contract = verify_relay_adapter_contract(
                    p,
                    expected_bundle_sha256=expected_artifact_manifest_sha256,
                    expected_base_model=expected_base_model,
                    expected_base_model_revision=revision,
                )
            except ArtifactIntegrityError as exc:
                raise ValueError(str(exc)) from exc
        adapter_config = PeftConfig.from_pretrained(model_ref, local_files_only=True)
        if (
            expected_base_model
            and adapter_config.base_model_name_or_path != expected_base_model
        ):
            raise ValueError("adapter config names a different base model")
        training_metadata_path = p / "relay-training.json"
        if training_metadata_path.is_file():
            training_metadata = json.loads(training_metadata_path.read_text(encoding="utf-8"))
            base_revision = training_metadata.get("base_model_revision")
            if base_revision is not None and (
                not isinstance(base_revision, str) or not base_revision.strip()
            ):
                raise ValueError(
                    "relay-training.json base_model_revision must be a non-empty string"
                )
        else:
            base_revision = getattr(adapter_config, "revision", None)
        if revision and base_revision and base_revision != revision:
            raise ValueError("adapter base revision does not match --revision")
        if revision and not base_revision:
            raise ValueError("adapter is missing the immutable base revision required by Relay")
        effective_revision = revision or base_revision
        tokenizer_ref = (
            model_ref
            if (p / "tokenizer_config.json").is_file()
            else adapter_config.base_model_name_or_path
        )
    else:
        adapter_config = None
        manifest = None
        base_revision = None
        effective_revision = revision
        if expected_base_model and model_ref != expected_base_model:
            raise ValueError("workspace base artifact does not match the expected model")
        tokenizer_ref = MODEL_ID if ours else model_ref
    tokenizer_kwargs = {"padding_side": "left"}
    if effective_revision and (tokenizer_ref != model_ref or not adapter):
        tokenizer_kwargs["revision"] = effective_revision
    tok = AutoTokenizer.from_pretrained(tokenizer_ref, **tokenizer_kwargs)
    if tok.pad_token_id is None:
        tok.pad_token = tok.eos_token

    if adapter:
        from peft import PeftModel

        base_kwargs = {"dtype": td}
        if effective_revision:
            base_kwargs["revision"] = effective_revision
        base = AutoModelForCausalLM.from_pretrained(
            adapter_config.base_model_name_or_path, **base_kwargs
        )
        model = PeftModel.from_pretrained(base, model_ref, local_files_only=True)
    elif ours:
        from safetensors.torch import load_file

        model = AutoModelForCausalLM.from_pretrained(
            MODEL_ID, dtype=td, revision=effective_revision
        )
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
        model = AutoModelForCausalLM.from_pretrained(
            model_ref, dtype=td, revision=effective_revision
        )

    model_config = getattr(model, "config", None)
    if adapter and getattr(model, "base_model", None) is not None:
        # PEFT wrappers may expose their adapter config at ``model.config``.
        # The immutable commit belongs to the underlying Qwen base, so unwrap
        # the same way the evaluation/benchmark paths do before attesting it.
        model_config = getattr(model.base_model, "config", model_config)
        model_config = getattr(getattr(model.base_model, "model", None), "config", model_config)
    actual_revision = getattr(model_config, "_commit_hash", None)
    if effective_revision and actual_revision != effective_revision:
        raise ValueError("loaded base model commit does not match --revision")
    model = model.cuda().eval()
    model._relay_model_revision = actual_revision
    model._relay_base_model = (
        adapter_config.base_model_name_or_path if adapter_config else expected_base_model or model_ref
    )
    model._relay_artifact_manifest_sha256 = (
        manifest.get("bundle_sha256") if manifest else None
    )
    return model, tok


def _cache_kwargs() -> dict:
    mode = STATE.get("cache_implementation")
    return {"cache_implementation": mode} if mode else {}


def public_identity() -> dict:
    """Return the immutable, path-free identity for this resident deployment."""

    pc = STATE.get("prefix")
    return {
        "model_id": STATE.get("deployment_model_id", "unregistered"),
        "pod_id": STATE.get("deployment_pod_id"),
        "base_model": STATE.get("base_model"),
        "model_revision": STATE.get("model_revision"),
        "artifact_manifest_sha256": STATE.get("artifact_manifest_sha256"),
        "prompt_template_sha256": hashlib.sha256(
            STATE.get("prompt_template", PROMPT).encode("utf-8")
        ).hexdigest(),
        "prefix_identity_sha256": pc.identity_sha256 if pc else None,
    }


def build_prompt(sentence: str, *, ignore_prefix: bool = False) -> tuple[str, PrefixCache | None]:
    """Static head first, dynamic text last — the order the cache requires.

    The template follows the model, because the wrong one is not a worse
    answer, it is a different question. A checkpoint trained on "Q: ...\nA:"
    served under "Extract:\n...\nJSON:\n" produced extraction-shaped JSON and
    never the string it was trained to emit — a model that looks broken while
    being asked something it was never taught.
    """
    # Quality/migration suites sometimes need the deployment's ordinary chat
    # prompt while a long-context optimization is resident. Bypass it for this
    # request without deleting or rebuilding the shared cache for everyone.
    pc = None if ignore_prefix else STATE.get("prefix")
    if pc is None:
        return STATE.get("prompt_template", PROMPT).format(sentence=sentence), None
    return pc.text + sentence + pc.suffix, pc


def encode_for(sentence: str, *, no_cache: bool = False, ignore_prefix: bool = False):
    """Returns (input_ids, prefix_cache_or_None_if_miss, token breakdown).

    `no_cache` bypasses the cache while building the SAME prompt. Without it
    the only way to get an uncached baseline is to unset the prefix, which
    changes the prompt too — and then a difference in output cannot be
    attributed to the cache rather than to the wording. A speed claim needs the
    two runs to differ in exactly one thing.
    """
    tok = STATE["tok"]
    prompt, pc = build_prompt(sentence, ignore_prefix=ignore_prefix)
    logical_ids = tok(prompt, return_tensors="pt", add_special_tokens=False)["input_ids"].cuda()
    hit = (not no_cache) and pc is not None and pc.matches(logical_ids)
    processed_ids = logical_ids[:, pc.n :] if hit else logical_ids
    logical_tokens = int(logical_ids.shape[1])
    processed_tokens = int(processed_ids.shape[1])
    info = {
        # ``prompt_tokens`` remains the compatibility alias for the logical
        # prompt length. ``processed_input_tokens`` is what this request
        # actually prefills after an optional shared-cache hit.
        "prompt_tokens": logical_tokens,
        "logical_input_tokens": logical_tokens,
        "processed_input_tokens": processed_tokens,
        "prefix_tokens": pc.n if pc else 0,
        "dynamic_tokens": logical_tokens - (pc.n if pc else 0),
        "cache": "hit" if hit else ("bypassed" if (pc and no_cache) else ("miss" if pc else "off")),
    }
    return processed_ids, (pc if hit else None), info


@torch.no_grad()
def generate(
    sentence: str,
    *,
    max_new: int,
    greedy: bool,
    no_cache: bool = False,
    fixed_output_tokens: bool = False,
    ignore_prefix: bool = False,
) -> dict:
    # Serialized: an overlapping /prefix reload or a second generate() sharing
    # this same pc.cache would crop() and read it concurrently. See _MODEL_LOCK.
    with _MODEL_LOCK:
        model, tok = STATE["model"], STATE["tok"]
        prompt, _selected_prefix = build_prompt(sentence, ignore_prefix=ignore_prefix)
        ids, pc, info = encode_for(sentence, no_cache=no_cache, ignore_prefix=ignore_prefix)
        enc = {"input_ids": ids}

        torch.cuda.synchronize()
        t0 = time.perf_counter()
        # A prefix hit already supplies its own cache object; asking for a
        # static one at the same time hands generate() two conflicting
        # instructions. The optimisation therefore applies to the path that
        # has no prefix — which is the chat path it was measured on.
        kw = {"past_key_values": pc.cache} if pc else _cache_kwargs()
        generation_kwargs = {}
        if fixed_output_tokens:
            generation_kwargs.update(min_new_tokens=max_new, eos_token_id=None)
        timing = TimingTokenCollector()
        out = model.generate(
            **enc,
            **kw,
            **generation_kwargs,
            max_new_tokens=max_new,
            do_sample=not greedy,
            temperature=0.7 if not greedy else None,
            pad_token_id=tok.pad_token_id,
            streamer=timing,
        )
        torch.cuda.synchronize()
        dt = time.perf_counter() - t0
        if pc:
            pc.reset()

        gen = out[0, enc["input_ids"].shape[1] :]
        raw = tok.decode(gen, skip_special_tokens=True)
        parsed = parse_output(raw)
        n_new = int(gen.shape[0])
        if timing.first_token_at is None:
            raise RuntimeError("generation produced no first token")
        return {
            "prompt": prompt,
            # Reported because prefill cost is driven by this number, not by the
            # character count a caller can see. Without it, a long-context latency
            # reading cannot be checked against the chip's FLOP ceiling — and an
            # impossible reading (more TFLOPS than the GPU has) is the signal that
            # the input was silently truncated rather than that serving got fast.
            "raw_output": raw,
            "parsed": parsed.model_dump() if parsed else None,
            "parsed_ok": parsed is not None,
            "new_tokens": n_new,
            "latency_s": dt,
            "ttft_s": timing.first_token_at - t0,
            # Interactive latency at batch 1. NOT the throughput number — see the
            # module docstring.
            "tokens_per_s": n_new / dt if dt else 0.0,
            **public_identity(),
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
def generate_stream(
    sentence: str,
    *,
    max_new: int,
    greedy: bool,
    no_cache: bool = False,
    fixed_output_tokens: bool = False,
    ignore_prefix: bool = False,
):
    """Yield tokens as they are produced, then a final summary frame.

    TIME TO FIRST TOKEN IS THE POINT. Total latency is what the GPU cost;
    TTFT is what a person experiences as "it started answering". They are
    different numbers and only one of them is why streaming feels fast, so
    both are reported and the UI shows both.

    Greedy still, so the streamed text is byte-identical to what the blocking
    endpoint returns — streaming is presentation, never a different answer.

    Holds _MODEL_LOCK for the whole generation, not just the GPU call: a
    /prefix reload landing mid-stream would crop() the same pc.cache this
    generation is still reading from, and a second concurrent stream would
    share it too. `with` releases the lock on early client disconnect
    (GeneratorExit) as well as on normal completion.
    """
    with _MODEL_LOCK:
        model, tok = STATE["model"], STATE["tok"]
        ids, pc, info = encode_for(sentence, no_cache=no_cache, ignore_prefix=ignore_prefix)

        streamer = TokenStreamer(tok)
        kwargs = dict(
            input_ids=ids,
            max_new_tokens=max_new,
            do_sample=not greedy,
            pad_token_id=tok.pad_token_id,
            streamer=streamer,
        )
        if fixed_output_tokens:
            # Benchmark-only: both arms must do the same decode work. Chat
            # keeps normal EOS behavior and never enables this flag.
            kwargs["min_new_tokens"] = max_new
            kwargs["eos_token_id"] = None
        if pc:
            kwargs["past_key_values"] = pc.cache
        else:
            # Keep streaming and blocking generation on the same runtime.
            # Prefix hits already provide a cache object, so combining the two
            # would be ambiguous; without a prefix, the selected static-cache
            # implementation is safe and should not silently disappear merely
            # because the caller requested SSE.
            kwargs.update(_cache_kwargs())
        if not greedy:
            kwargs["temperature"] = 0.7

        worker_errors: list[BaseException] = []
        thread: Thread | None = None
        thread_started = False

        def run_generate() -> None:
            try:
                model.generate(**kwargs)
            except BaseException as exc:  # noqa: BLE001 — cross the thread boundary
                # Exceptions raised in a Thread do not reach the request thread.
                # Retain the original exception so the SSE handler can surface it
                # instead of returning a plausible-looking partial answer.
                worker_errors.append(exc)
            finally:
                # Transformers normally ends the streamer itself, but an error
                # before that point otherwise leaves this request blocked forever
                # on Queue.get(). TokenStreamer.end() is idempotent for our
                # consumer: a second sentinel simply remains unused.
                streamer.end()

        try:
            torch.cuda.synchronize()
            t0 = time.perf_counter()
            # generate() blocks, so it runs on its own thread and the streamer is
            # the channel. Without this there is nothing to iterate until it has
            # finished, which is exactly the behaviour we are removing.
            thread = Thread(target=run_generate, daemon=True)
            thread.start()
            thread_started = True

            ttft = None
            pieces = []
            for text in streamer:
                if not text:
                    continue
                if ttft is None:
                    ttft = time.perf_counter() - t0
                pieces.append(text)
                yield {"token": text}

            # The sentinel can become visible just before run_generate returns.
            # Join before reading worker_errors so that publication is complete.
            thread.join()
            if worker_errors:
                raise worker_errors[0]

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
                **public_identity(),
                "dtype": STATE["dtype"],
                **info,
            }
        finally:
            # Closing this generator after a BrokenPipeError injects
            # GeneratorExit at its current yield. Do not release _MODEL_LOCK or
            # crop a shared prefix while the CUDA worker is still using either.
            if thread is not None and thread_started:
                thread.join()
            if pc:
                pc.reset()


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
                    **public_identity(),
                    "dtype": STATE["dtype"],
                    "prefix_tokens": pc.n if pc else 0,
                    "prefix_build_s": pc.build_s if pc else None,
                    "cache_implementation": STATE.get("cache_implementation"),
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
            # The proxy validates this frame before releasing even one token
            # to the browser.  It closes the deployment-switch TOCTOU window:
            # identity is now read from the same HTTP connection that will
            # produce the answer, not from an earlier /health request.
            identity = {"identity": True, **public_identity()}
            self.wfile.write(f"data: {json.dumps(identity, ensure_ascii=False)}\n\n".encode())
            self.wfile.flush()
            for frame in generate_stream(
                sentence,
                max_new=min(int(req.get("max_new_tokens", 64)), 256),
                greedy=bool(req.get("greedy", True)),
                no_cache=bool(req.get("no_cache", False)),
                fixed_output_tokens=bool(req.get("fixed_output_tokens", False)),
                ignore_prefix=bool(req.get("ignore_prefix", False)),
            ):
                self.wfile.write(f"data: {json.dumps(frame, ensure_ascii=False)}\n\n".encode())
                self.wfile.flush()
        except (BrokenPipeError, ConnectionResetError):
            pass  # the browser navigated away mid-stream; nothing to report
        except Exception:  # noqa: BLE001
            traceback.print_exc(file=sys.stderr)
            sys.stderr.flush()
            with contextlib.suppress(Exception):
                err = json.dumps(
                    {
                        "done": True,
                        "error": "generation failed; inspect the local pod log",
                    }
                )
                self.wfile.write(f"data: {err}\n\n".encode())
                self.wfile.flush()

    def _runtime(self, req: dict) -> None:
        """Change how generation allocates its cache, model stays resident."""
        mode = req.get("cache_implementation")
        if mode not in CACHE_MODES:
            self._send(
                400,
                {"error": f"cache_implementation must be one of {sorted(map(str, CACHE_MODES))}"},
            )
            return
        with _MODEL_LOCK:
            previous = STATE.get("cache_implementation")
            STATE["cache_implementation"] = mode
        # The previous value goes back in the response so a caller can put it
        # back without having had to remember it first.
        self._send(200, {"cache_implementation": mode, "previous": previous})

    def _prefix(self, req: dict) -> None:
        text = req.get("prefix")
        suffix = req.get("suffix", "")
        if suffix is None:
            suffix = ""
        if not isinstance(suffix, str):
            self._send(400, {"error": "suffix must be a string"})
            return
        # Locked so this never races a /generate or /generate/stream that is
        # still reading STATE["prefix"] — dropping/rebuilding it mid-request
        # is what produced degenerate output when a reload landed mid-stream.
        with _MODEL_LOCK:
            # Drop the old cache BEFORE building the new one. Holding both is two
            # multi-GB KV tensors plus the new prefill's activations, which OOMs on
            # a 24 GB card — reloading the policy, the most ordinary thing to do
            # twice, was killing the server.
            STATE.pop("prefix", None)
            torch.cuda.empty_cache()
            if not text:
                self._send(200, {"prefix_tokens": 0, "cleared": True})
                return
            pc = PrefixCache(STATE["model"], STATE["tok"], str(text), suffix=suffix)
            STATE["prefix"] = pc
            # build_s IS the cold-path prefill for this prefix — the number the
            # cache removes. Reported so the speedup can be stated from a measured
            # baseline instead of a remembered one.
            self._send(
                200,
                {
                    "prefix_tokens": pc.n,
                    "build_s": pc.build_s,
                    "prefix_identity_sha256": pc.identity_sha256,
                    "cleared": False,
                },
            )

    def do_POST(self):  # noqa: N802
        if self.path not in ("/generate", "/generate/stream", "/prefix", "/runtime"):
            self._send(404, {"error": "not found"})
            return
        try:
            n = int(self.headers.get("Content-Length", 0))
            req = json.loads(self.rfile.read(n) or b"{}")
            if self.path == "/prefix":
                self._prefix(req)
                return
            if self.path == "/runtime":
                self._runtime(req)
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
                    fixed_output_tokens=bool(req.get("fixed_output_tokens", False)),
                    ignore_prefix=bool(req.get("ignore_prefix", False)),
                ),
            )
        except Exception:  # noqa: BLE001 — one bad request must not kill the server
            # The traceback goes to the log even though the response cannot
            # carry it. An AssertionError stringifies to "", so the response
            # alone read as `{"error": "AssertionError: "}` — which says a
            # failure happened and nothing whatsoever about where.
            traceback.print_exc(file=sys.stderr)
            sys.stderr.flush()
            self._send(500, {"error": "request failed; inspect the local pod log"})

    def log_message(self, *args):
        pass  # the dashboard already logs; this would double every line


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--model", default=MODEL_ID, help="HF id or a train.py output dir")
    ap.add_argument("--dtype", choices=sorted(DTYPES), default="bf16")
    ap.add_argument(
        "--revision",
        default=None,
        help="Immutable Hugging Face base revision expected by this workspace version.",
    )
    ap.add_argument(
        "--expected-base-model",
        default=None,
        help="Exact Hugging Face base repository required by this workspace version.",
    )
    ap.add_argument(
        "--artifact-manifest-sha256",
        default=None,
        help="Canonical Relay adapter bundle hash required for LoRA versions.",
    )
    ap.add_argument(
        "--deployment-model-id",
        default=None,
        help="Browser-safe workspace version id; never an internal artifact path.",
    )
    ap.add_argument(
        "--deployment-pod-id",
        default=None,
        help="RunPod id bound to this server process for proxy identity checks.",
    )
    ap.add_argument("--port", type=int, default=8100)
    ap.add_argument(
        "--prompt-template",
        default=PROMPT,
        help="Must contain {sentence}. Has to match what the model was trained on.",
    )
    a = ap.parse_args()

    if not torch.cuda.is_available():
        raise SystemExit("no GPU visible — run `make check` first")

    print(f"loading {a.model} ({a.dtype})...", file=sys.stderr, flush=True)
    model, tok = load(
        a.model,
        a.dtype,
        revision=a.revision,
        expected_base_model=a.expected_base_model,
        expected_artifact_manifest_sha256=a.artifact_manifest_sha256,
    )
    if "{sentence}" not in a.prompt_template:
        raise SystemExit("--prompt-template must contain {sentence}")
    STATE.update(
        model=model,
        tok=tok,
        model_ref=a.model,
        deployment_model_id=a.deployment_model_id or a.model,
        deployment_pod_id=a.deployment_pod_id,
        base_model=getattr(model, "_relay_base_model", None),
        model_revision=getattr(model, "_relay_model_revision", None),
        artifact_manifest_sha256=getattr(
            model, "_relay_artifact_manifest_sha256", None
        ),
        dtype=a.dtype,
        prompt_template=a.prompt_template,
    )

    # Threading: a stream holds its connection open for the whole generation,
    # and a single-threaded server would make /health time out behind it —
    # which the dashboard reads as "the model never became ready".
    # 127.0.0.1 on purpose: unauthenticated, reached only through an SSH forward.
    server = ThreadingHTTPServer(("127.0.0.1", a.port), Handler)
    print(f"READY on 127.0.0.1:{a.port}", file=sys.stderr, flush=True)
    server.serve_forever()


if __name__ == "__main__":
    main()
