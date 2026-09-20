"""Bounded, replayable execution diagnostics; task results remain independently fenced."""

import functools
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
        "cleaned",
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
    # Assignment generations include declined offers, which do not spend retries.
    attempt: int = Field(ge=1, le=2_147_483_647, strict=True)
    events: list[ExecutionEvent] = Field(min_length=1, max_length=8)


@functools.cache
def _scrubber() -> Scrubber:
    # Scanning the environment per event is wasteful on the server's locked
    # ingest path; credentials are configured at startup, so one scan suffices.
    return Scrubber(secret_values())


def scrub_execution(data):
    return _scrubber().scrub(data)


def rejected_ack(payload: object) -> dict | None:
    """Acknowledge, as rejected, a batch that will never be stored.

    Without an acknowledgement the device replays the same malformed batch
    forever; with one it marks the sequences delivered and moves on.
    """
    if not isinstance(payload, dict) or not isinstance(payload.get("events"), list):
        return None
    try:
        return {
            "taskId": str(payload["taskId"]),
            "attempt": int(payload["attempt"]),
            "sequences": [int(item["sequence"]) for item in payload["events"]],
            "rejected": True,
        }
    except (KeyError, TypeError, ValueError):
        return None
