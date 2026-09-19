"""Baseten model: a pure completion endpoint over Qwen2.5-0.5B.

IT DOES NOT BUILD THE PROMPT. The caller sends the full prompt string, already
formatted. gpushare/agent/task.py owns PROMPT, and the training targets were
written against that exact text — a second copy here would drift, and the model
would start being asked a question it was never trained on. The endpoint stays
a completion service; the task semantics stay in one place.

Greedy by default for the same reason the local evaluator is greedy: a
before/after comparison has to vary one thing. Sampling would add a second
source of difference and leave the viewer unable to say which one moved.
"""

import time

import torch
from transformers import AutoModelForCausalLM, AutoTokenizer

MODEL_ID = "Qwen/Qwen2.5-0.5B"


class Model:
    def __init__(self, **kwargs):
        self._model = None
        self._tok = None
        self._config = kwargs.get("config", {})

    def load(self):
        self._tok = AutoTokenizer.from_pretrained(MODEL_ID, padding_side="left")
        if self._tok.pad_token_id is None:
            self._tok.pad_token = self._tok.eos_token
        self._model = AutoModelForCausalLM.from_pretrained(
            MODEL_ID, torch_dtype=torch.bfloat16
        )
        if torch.cuda.is_available():
            self._model = self._model.cuda()
        self._model.eval()

    @torch.no_grad()
    def predict(self, request: dict) -> dict:
        prompt = request.get("prompt")
        if not prompt:
            return {"error": "prompt is required (the caller formats it, not the server)"}

        max_new = min(int(request.get("max_new_tokens", 64)), 256)
        greedy = bool(request.get("greedy", True))

        enc = self._tok(prompt, return_tensors="pt", add_special_tokens=False)
        if torch.cuda.is_available():
            enc = {k: v.cuda() for k, v in enc.items()}
            torch.cuda.synchronize()

        t0 = time.perf_counter()
        out = self._model.generate(
            **enc,
            max_new_tokens=max_new,
            do_sample=not greedy,
            pad_token_id=self._tok.pad_token_id,
        )
        if torch.cuda.is_available():
            torch.cuda.synchronize()
        dt = time.perf_counter() - t0

        gen = out[0, enc["input_ids"].shape[1] :]
        return {
            "text": self._tok.decode(gen, skip_special_tokens=True),
            "new_tokens": int(gen.shape[0]),
            # Server-side generation only. The caller measures its own round
            # trip; reporting one as the other would blame the GPU for the
            # network, or hide the network behind the GPU.
            "latency_s": dt,
            "model": MODEL_ID,
            "greedy": greedy,
        }
