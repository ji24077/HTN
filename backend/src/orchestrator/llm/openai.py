"""OpenAI Responses transport using the backend's existing async HTTP client."""

import asyncio
import math
import os
import re
from collections.abc import Sequence
from typing import Any

import httpx

from ..shared.protocol import json_loads, json_text
from .types import ModelClientError, ModelResponse, TokenUsage, ToolCall

RESPONSES_URL = "https://api.openai.com/v1/responses"
MAX_RESPONSE_BYTES = 4 * 1024 * 1024
TOOL_NAME = re.compile(r"[a-zA-Z0-9_-]{1,64}")


def function_tools(definitions: Sequence[dict[str, Any]]) -> list[dict[str, Any]]:
    """Adapt orchestrator.client.tool_definitions() without changing its schemas."""
    result = []
    names = set()
    for definition in definitions:
        name = definition.get("name")
        schema = definition.get("input_schema")
        description = definition.get("description", "")
        if (
            not isinstance(name, str)
            or not TOOL_NAME.fullmatch(name)
            or name in names
            or not isinstance(description, str)
            or not isinstance(schema, dict)
            or schema.get("type") != "object"
        ):
            raise ValueError("Tools need unique names, descriptions, and object input schemas")
        names.add(name)
        result.append(
            {
                "type": "function",
                "name": name,
                "description": description,
                "parameters": schema,
                # Existing schemas include defaults and arbitrary JSON payloads.
                # AgentTools remains responsible for validating execution inputs.
                "strict": False,
            }
        )
    return result


def response_from_json(data: Any, request_id: str | None) -> ModelResponse:
    """Expose completed text/calls while retaining every item for continuation."""
    try:
        if not isinstance(data, dict):
            raise ValueError
        status = data["status"]
        if status == "incomplete":
            raise ModelClientError(
                "incomplete_response",
                "OpenAI did not finish the response; no tool calls should be executed",
                request_id=request_id,
            )
        if status == "failed":
            raise ModelClientError(
                "response_failed", "OpenAI could not complete the response", request_id=request_id
            )
        if status != "completed" or data.get("error") is not None:
            raise ValueError
        response_id, model, output = data["id"], data["model"], data["output"]
        if not all(isinstance(v, str) and v for v in (response_id, model)):
            raise ValueError
        if not isinstance(output, list):
            raise ValueError
        texts, calls, refusals = [], [], []
        call_ids = set()
        for item in output:
            match item["type"]:
                case "message":
                    if item["role"] != "assistant" or item["status"] != "completed":
                        raise ValueError
                    if not isinstance(item["content"], list):
                        raise ValueError
                    for part in item["content"]:
                        if part["type"] == "output_text" and isinstance(part["text"], str):
                            texts.append(part["text"])
                        elif part["type"] == "refusal" and isinstance(part["refusal"], str):
                            refusals.append(part["refusal"])
                        else:
                            raise ValueError
                case "function_call":
                    call_id, name, arguments = item["call_id"], item["name"], item["arguments"]
                    if (
                        not all(isinstance(v, str) and v for v in (call_id, name))
                        or not isinstance(arguments, str)
                        or item.get("status", "completed") != "completed"
                        or call_id in call_ids
                    ):
                        raise ValueError
                    call_ids.add(call_id)
                    calls.append(ToolCall(call_id, name, arguments))
                case "reasoning":
                    pass  # Includes encrypted_content for store=false continuations.
                case _:
                    raise ValueError  # Only our function tools are requested.
        usage = None
        if data.get("usage") is not None:
            counts = [data["usage"][k] for k in ("input_tokens", "output_tokens", "total_tokens")]
            if not all(type(v) is int and v >= 0 for v in counts):
                raise ValueError
            usage = TokenUsage(*counts)
        return ModelResponse(
            id=response_id,
            model=model,
            text="".join(texts),
            tool_calls=tuple(calls),
            output=output,
            refusals=tuple(refusals),
            usage=usage,
            request_id=request_id,
        )
    except (KeyError, TypeError, ValueError):
        raise ModelClientError(
            "invalid_response", "Unexpected OpenAI response format", request_id=request_id
        ) from None


class OpenAIClient:
    """Reusable, stateless, single-call client. The caller owns tools and history."""

    def __init__(
        self,
        api_key: str,
        model: str,
        *,
        timeout_seconds: float = 60,
        max_output_tokens: int = 4096,
        transport: httpx.AsyncBaseTransport | None = None,
    ):
        if not api_key or not api_key.isascii() or any(c.isspace() for c in api_key):
            raise ValueError("A nonempty OpenAI API key without whitespace is required")
        if not model or any(c.isspace() for c in model):
            raise ValueError("An OpenAI model ID without whitespace is required")
        if (
            isinstance(timeout_seconds, bool)
            or not math.isfinite(timeout_seconds)
            or timeout_seconds <= 0
        ):
            raise ValueError("timeout_seconds must be finite and greater than zero")
        if type(max_output_tokens) is not int or max_output_tokens <= 0:
            raise ValueError("max_output_tokens must be a positive integer")
        self.model = model
        self.timeout_seconds = timeout_seconds
        self.max_output_tokens = max_output_tokens
        self._http = httpx.AsyncClient(
            headers={"Authorization": f"Bearer {api_key}", "Content-Type": "application/json"},
            timeout=httpx.Timeout(timeout_seconds, connect=min(5, timeout_seconds)),
            follow_redirects=False,
            trust_env=False,
            transport=transport,
        )

    @classmethod
    def from_env(cls) -> "OpenAIClient":
        """Read configuration explicitly; importing this module needs no secrets."""
        api_key = os.getenv("OPENAI_API_KEY", "")
        model = os.getenv("OPENAI_MODEL", "")
        if not api_key or not model:
            raise ValueError("Set OPENAI_API_KEY and OPENAI_MODEL on the backend")
        return cls(api_key, model)

    async def __aenter__(self) -> "OpenAIClient":
        return self

    async def __aexit__(self, *_args) -> None:
        await self.aclose()

    async def aclose(self) -> None:
        await self._http.aclose()

    async def respond(
        self,
        messages: str | Sequence[dict[str, Any]],
        *,
        tools: Sequence[dict[str, Any]] = (),
        instructions: str | None = None,
    ) -> ModelResponse:
        if isinstance(messages, str):
            if not messages.strip():
                raise ValueError("messages must not be empty")
            inputs = messages
        else:
            inputs = list(messages)
            if not inputs or not all(isinstance(item, dict) for item in inputs):
                raise ValueError("messages must contain Responses API input items")
        body = {
            "model": self.model,
            "input": inputs,
            "tools": function_tools(tools),
            "store": False,
            "include": ["reasoning.encrypted_content"],
            "parallel_tool_calls": False,
            "max_output_tokens": self.max_output_tokens,
        }
        if instructions is not None:
            if not isinstance(instructions, str):
                raise ValueError("instructions must be a string")
            body["instructions"] = instructions
        encoded = json_text(body).encode()
        request_id = None
        try:
            # One wall-clock deadline also bounds slow response streams. No
            # automatic retries: a failed request may still incur provider usage.
            async with asyncio.timeout(self.timeout_seconds):
                async with self._http.stream("POST", RESPONSES_URL, content=encoded) as response:
                    request_id = response.headers.get("x-request-id")
                    if not response.is_success:
                        status = response.status_code
                        codes = {
                            400: "invalid_request",
                            401: "authentication_failed",
                            403: "permission_denied",
                            404: "model_not_found",
                            429: "rate_limited",
                        }
                        raise ModelClientError(
                            codes.get(status, "provider_error"),
                            f"OpenAI request failed (HTTP {status})",
                            status=status,
                            retryable=status in {408, 409, 429} or status >= 500,
                            request_id=request_id,
                        )
                    data = bytearray()
                    async for chunk in response.aiter_bytes():
                        data.extend(chunk)
                        if len(data) > MAX_RESPONSE_BYTES:
                            raise ModelClientError(
                                "response_too_large",
                                "OpenAI response exceeded the client size limit",
                                request_id=request_id,
                            )
            try:
                parsed = json_loads(bytes(data))
            except (ValueError, RecursionError):
                raise ModelClientError(
                    "invalid_response", "OpenAI returned invalid JSON", request_id=request_id
                ) from None
            return response_from_json(parsed, request_id)
        except (TimeoutError, httpx.TimeoutException):
            raise ModelClientError(
                "timeout", "OpenAI request timed out", retryable=True, request_id=request_id
            ) from None
        except httpx.HTTPError:
            raise ModelClientError(
                "connection_error",
                "OpenAI request failed",
                retryable=True,
                request_id=request_id,
            ) from None
