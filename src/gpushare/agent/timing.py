"""Shared token-timing collector for comparable batch-one GPU benchmarks."""

from __future__ import annotations

import time
from typing import Any


class TimingTokenCollector:
    """Record generated token IDs and the first-token timestamp.

    Transformers streamers receive the prompt once before generated tokens.
    Using this same collector in the resident serving process and the portable
    CUDA/ROCm benchmark keeps migration latency measurements on one harness.
    """

    def __init__(self) -> None:
        self.prompt_seen = False
        self.first_token_at: float | None = None
        self.ids: list[int] = []

    def put(self, value: Any) -> None:
        if not self.prompt_seen:
            self.prompt_seen = True
            return
        values = value.tolist()
        if values and isinstance(values[0], list):
            values = values[0]
        if values and self.first_token_at is None:
            self.first_token_at = time.perf_counter()
        self.ids.extend(values)

    def end(self) -> None:
        return None
