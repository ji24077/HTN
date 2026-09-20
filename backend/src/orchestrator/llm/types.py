"""The call boundary a future agent loop can depend on and fake in tests."""

from collections.abc import Sequence
from dataclasses import dataclass, field
from typing import Any, Protocol

from ..shared.protocol import json_loads, json_text


class ModelClientError(Exception):
    """Safe error metadata; excludes provider bodies, prompts, and credentials."""

    def __init__(
        self,
        code: str,
        message: str,
        *,
        status: int | None = None,
        retryable: bool = False,
        request_id: str | None = None,
    ):
        super().__init__(message)
        self.code = code
        self.status = status
        self.retryable = retryable
        self.request_id = request_id


@dataclass(frozen=True)
class ToolCall:
    call_id: str
    name: str
    # Keep the original arguments so the loop can report invalid JSON back to
    # the model. Parsing/schema validation and tool execution are separate steps.
    arguments: str = field(repr=False)

    def parse_arguments(self) -> dict[str, Any]:
        try:
            value = json_loads(self.arguments)
        except (ValueError, RecursionError):
            raise ValueError("Tool arguments must be a JSON object") from None
        if not isinstance(value, dict):
            raise ValueError("Tool arguments must be a JSON object")
        return value

    def output(self, result: Any) -> dict[str, str]:
        """Format an already-executed tool result for the next explicit call."""
        return {
            "type": "function_call_output",
            "call_id": self.call_id,
            "output": json_text(result),
        }


@dataclass(frozen=True)
class TokenUsage:
    input_tokens: int
    output_tokens: int
    total_tokens: int


@dataclass(frozen=True)
class ModelResponse:
    id: str
    model: str
    text: str = field(repr=False)
    tool_calls: tuple[ToolCall, ...]
    # Opaque provider continuation items, including encrypted reasoning. The
    # caller owns history and must retain these, not just the visible text.
    output: list[dict[str, Any]] = field(repr=False)
    refusals: tuple[str, ...] = field(default=(), repr=False)
    usage: TokenUsage | None = None
    request_id: str | None = None


class ModelClient(Protocol):
    async def respond(
        self,
        messages: str | Sequence[dict[str, Any]],
        *,
        tools: Sequence[dict[str, Any]] = (),
        instructions: str | None = None,
    ) -> ModelResponse:
        """Make one model call. Never execute tools, retry, or retain history."""
        ...
