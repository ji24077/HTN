"""The sole executor is a cancellable stub; no user code is executed."""

import asyncio
import hashlib
from collections.abc import Callable

from pydantic import Field, JsonValue

from ...shared.protocol import Model, TaskSpec, json_text


class StubPayload(Model):
    duration_seconds: int = Field(default=0, ge=0, le=3600, strict=True)
    value: JsonValue = None
    fail: bool = Field(default=False, strict=True)


class StubExecutor:
    kind = "stub"

    async def execute(self, spec: TaskSpec, report: Callable[[float], None]) -> JsonValue:
        payload = StubPayload.model_validate(spec.payload)
        report(0)
        for step in range(payload.duration_seconds):
            await asyncio.sleep(1)
            report((step + 1) / payload.duration_seconds * 100)
        if payload.fail:
            raise ValueError("requested stub failure")
        digest = hashlib.sha256(json_text(payload.value).encode()).hexdigest()
        return {"value": payload.value, "sha256": digest}
