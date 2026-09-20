"""Provider-specific credentials and bounded JSON proposals; never execute model output.

Price estimates were checked against provider pricing on 2026-09-19. They are
estimates, not account billing. Unknown/custom models require explicit prices.
"""

from __future__ import annotations

import json
import math
import os
from dataclasses import dataclass
from pathlib import Path

from dotenv import dotenv_values

MODELS = {
    "baseten": ("https://inference.baseten.co/v1", "BASETEN_API_KEY", "zai-org/GLM-5.3-Flash", 0.15, 0.50),
    "openai": ("https://api.openai.com/v1", "OPENAI_API_KEY", "gpt-5.4-mini", 0.75, 4.50),
}


class ModelError(RuntimeError):
    """Safe diagnostic that never contains a key or raw provider exception."""


@dataclass
class ModelClient:
    provider: str = "baseten"
    env_file: Path = Path(".env")
    budget_usd: float = 1.0
    max_calls: int = 8
    max_output_tokens: int = 6000
    timeout_seconds: float = 90.0
    transport: object | None = None

    def __post_init__(self):
        if self.provider not in MODELS:
            raise ValueError("provider must be baseten or openai")
        if (type(self.budget_usd) not in {int, float} or not 0 < self.budget_usd <= 5
                or type(self.max_calls) is not int or not 1 <= self.max_calls <= 20):
            raise ValueError("require model budget <= $5 and 1..20 requests")
        if type(self.max_output_tokens) is not int or not 512 <= self.max_output_tokens <= 12000:
            raise ValueError("output token limit must be 512..12000")
        if (type(self.timeout_seconds) not in {int, float}
                or not math.isfinite(self.timeout_seconds) or not 0 < self.timeout_seconds <= 300):
            raise ValueError("request timeout must be finite and within 0..300 seconds")
        self.calls = 0
        self.estimated_cost_usd = 0.0
        self.reserved_cost_usd = 0.0
        self.history = []
        base, key_name, self.model, self.input_price, self.output_price = MODELS[self.provider]
        if self.transport is None:
            from openai import OpenAI

            env = {**dotenv_values(self.env_file), **os.environ}
            key = env.get(key_name)
            if not key:
                raise ModelError(f"{key_name} is not configured")
            self.transport = OpenAI(api_key=key, base_url=base, timeout=self.timeout_seconds, max_retries=0)

    def propose(self, system: str, request: dict) -> dict:
        content = json.dumps(request, ensure_ascii=False)
        # Counting every UTF-8 byte as a token deliberately overestimates input.
        upper_input = len((system + content).encode("utf-8")) + 1024
        upper_cost = (upper_input * self.input_price + self.max_output_tokens * self.output_price) / 1e6
        if upper_input > 100000:
            raise ModelError("request exceeds the supported script/context size")
        if self.calls >= self.max_calls or max(self.reserved_cost_usd, self.estimated_cost_usd) + upper_cost > self.budget_usd:
            raise ModelError("coding-model request or cost limit reached")
        self.calls += 1
        self.reserved_cost_usd += upper_cost
        args = {
            "model": self.model,
            "messages": [{"role": "system", "content": system}, {"role": "user", "content": content}],
            "response_format": {"type": "json_object"},
            "reasoning_effort": "low",
        }
        args["max_tokens" if self.provider == "baseten" else "max_completion_tokens"] = self.max_output_tokens
        try:
            response = self.transport.chat.completions.create(**args)
        except Exception as exc:
            status = getattr(exc, "status_code", None)
            status = status if type(status) is int else "unknown"
            raise ModelError(f"{self.provider} request failed (HTTP {status}); no secret response logged") from None
        usage = getattr(response, "usage", None)
        input_tokens = getattr(usage, "prompt_tokens", None)
        output_tokens = getattr(usage, "completion_tokens", None)
        if (type(input_tokens) is int and type(output_tokens) is int
                and 0 <= input_tokens <= 10**9 and 0 <= output_tokens <= 10**9):
            estimate = (input_tokens * self.input_price + output_tokens * self.output_price) / 1e6
            self.estimated_cost_usd += estimate
        else:
            input_tokens = output_tokens = None
            estimate = upper_cost
            self.estimated_cost_usd += upper_cost
        choices = getattr(response, "choices", None)
        if not isinstance(choices, (list, tuple)) or not choices:
            raise ModelError("coding-model response contained no completion choices")
        choice = choices[0]
        finish_reason = getattr(choice, "finish_reason", None)
        # Keep only the documented reason vocabulary in accounting diagnostics.
        safe_reason = finish_reason if finish_reason in ("stop", "length", "content_filter", "tool_calls", "function_call") else "unknown"
        self.history.append({"provider": self.provider, "model": self.model, "input_tokens": input_tokens,
                             "output_tokens": output_tokens, "estimated_cost_usd": estimate,
                             "finish_reason": safe_reason})
        if finish_reason != "stop":
            raise ModelError("coding-model response was incomplete; candidate was not applied")
        try:
            result = json.loads(getattr(getattr(choice, "message", None), "content", None) or "")
        except (ValueError, TypeError):
            raise ModelError("coding-model response was not a JSON object") from None
        if not isinstance(result, dict):
            raise ModelError("coding-model response was not a JSON object")
        return result

    def summary(self) -> dict:
        return {"provider": self.provider, "model": self.model, "calls": self.calls,
                "estimated_cost_usd": round(self.estimated_cost_usd, 6),
                "reserved_cost_upper_usd": round(self.reserved_cost_usd, 6),
                "budget_usd": self.budget_usd, "requests": [item.copy() for item in self.history]}
