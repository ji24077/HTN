"""Bounded, replayable execution diagnostics; task results remain independently fenced."""

import json
from datetime import datetime
from typing import Literal

from pydantic import Field, JsonValue, field_validator

from .protocol import Identifier, Model
from .telemetry import Scrubber, secret_values


class ExecutionEvent(Model):
    sequence: int = Field(ge=1, le=1000, strict=True)
    at: datetime
    kind: Literal[
        "started",
        "step",
        "stdout",
        "stderr",
        "progress",
        "succeeded",
        "failed",
        "cancelled",
        "timed_out",
        "interrupted",
        "truncated",
    ]
    data: dict[str, JsonValue] = Field(default_factory=dict)

    @field_validator("at")
    @classmethod
    def timezone_required(cls, value):
        if value.tzinfo is None:
            raise ValueError("event timestamp requires timezone")
        return value

    @field_validator("data")
    @classmethod
    def bounded_data(cls, value):
        if (
            len(
                json.dumps(
                    value, ensure_ascii=False, separators=(",", ":"), allow_nan=False
                ).encode()
            )
            > 8192
        ):
            raise ValueError("execution event data exceeds 8 KiB")
        return value


class ExecutionBatch(Model):
    taskId: Identifier
    attempt: int = Field(ge=1, le=10, strict=True)
    events: list[ExecutionEvent] = Field(min_length=1, max_length=8)


def scrub_execution(data):
    return Scrubber(secret_values()).scrub(data)
