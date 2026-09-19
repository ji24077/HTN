"""Callable progress reporter plus durable task-scoped stdout/stderr/step hooks."""

import asyncio
import hashlib
import json
import math
import os
import time
from datetime import UTC, datetime
from pathlib import Path

from ..shared.execution import ExecutionBatch, ExecutionEvent, scrub_execution
from ..shared.protocol import json_text

TERMINAL = {"succeeded", "failed", "cancelled", "timed_out", "interrupted", "cleaned"}


class ExecutionJournal:
    def __init__(self, server, worker_id, root=None):
        root = Path(root or os.getenv("WORKER_EXECUTION_DIR", str(Path.home() / ".dwp/executions")))
        self.directory = root / hashlib.sha256(f"{server}\n{worker_id}".encode()).hexdigest()
        self.records = {}
        self.sent = {}
        self.sizes = {}
        self.dirty = set()
        self.flush_handle = None
        try:
            self.directory.mkdir(parents=True, exist_ok=True, mode=0o700)
            paths = sorted(
                self.directory.glob("*.json"), key=lambda p: p.stat().st_mtime, reverse=True
            )
            for index, path in enumerate(paths):
                if (
                    index >= 32
                    or time.time() - path.stat().st_mtime > 7 * 86400
                    or path.stat().st_size > 2 * 1024 * 1024
                ):
                    path.unlink()
                    continue
                try:
                    text = path.read_text()
                    record = json.loads(text)
                    if not isinstance(record["acked"], list):
                        continue
                    for item in record["events"]:
                        ExecutionEvent.model_validate(item)
                    key = (record["taskId"], record["attempt"])
                    self.records[key] = record
                    self.sizes[key] = len(text.encode())
                    if record["events"] and record["events"][-1]["kind"] not in TERMINAL:
                        self.emit(
                            *key, "interrupted", {"message": "Runner restarted before completion"}
                        )
                except (ValueError, KeyError, TypeError):
                    continue
        except OSError:
            pass

    def path(self, key):
        return self.directory / (
            hashlib.sha256(f"{key[0]}:{key[1]}".encode()).hexdigest() + ".json"
        )

    def persist(self, record):
        key = (record["taskId"], record["attempt"])
        self.dirty.discard(key)
        try:
            path = self.path(key)
            temporary = path.with_suffix(".tmp")
            with os.fdopen(
                os.open(temporary, os.O_WRONLY | os.O_CREAT | os.O_TRUNC, 0o600), "w"
            ) as out:
                out.write(json_text(record))
            temporary.replace(path)
        except OSError:
            pass  # Optional diagnostics cannot change a task's result.

    def schedule_persist(self, key):
        # Rewriting the whole journal per event is quadratic in a chatty task and
        # blocks the event loop that renews leases; coalesce ordinary output.
        self.dirty.add(key)
        if self.flush_handle is not None:
            return
        try:
            loop = asyncio.get_running_loop()
        except RuntimeError:
            self.persist_dirty()
            return
        self.flush_handle = loop.call_later(0.25, self.persist_dirty)

    def persist_dirty(self):
        self.flush_handle = None
        for key in list(self.dirty):
            if record := self.records.get(key):
                self.persist(record)
        self.dirty.clear()

    def emit(self, task_id, attempt, kind, data=None):
        key = (task_id, attempt)
        if key not in self.records:
            if len(self.records) >= 32:
                oldest = min(
                    self.records,
                    key=lambda key: (
                        self.records[key]["events"][0]["at"] if self.records[key]["events"] else ""
                    ),
                )
                del self.records[oldest]
                self.sent.pop(oldest, None)
                self.sizes.pop(oldest, None)
                self.dirty.discard(oldest)
                try:
                    self.path(oldest).unlink(missing_ok=True)
                except OSError:
                    pass
            self.records[key] = {"taskId": task_id, "attempt": attempt, "events": [], "acked": []}
        record = self.records[key]
        if len(record["events"]) >= 999 or (record.get("truncated") and kind not in TERMINAL):
            return
        clean = scrub_execution(data or {})
        size = len(json_text(clean).encode())
        if size > 8192:
            clean = {"message": "Event exceeded 8 KiB", "truncated": True}
            size = len(json_text(clean).encode())
        if kind not in TERMINAL and (
            len(record["events"]) >= 996 or self.sizes.get(key, 0) + size > 1000 * 1024
        ):
            kind, clean = (
                "truncated",
                {"message": "Output limit reached; completion is still recorded"},
            )
            record["truncated"] = True
        item = ExecutionEvent(
            sequence=len(record["events"]) + 1, at=datetime.now(UTC), kind=kind, data=clean
        )
        encoded = item.model_dump(mode="json")
        record["events"].append(encoded)
        self.sizes[key] = self.sizes.get(key, 0) + len(json_text(encoded).encode())
        # Durability matters most at the boundaries: the first event proves the
        # attempt started, and terminal events settle it. Coalesce the rest.
        if kind in TERMINAL or record.get("truncated") or len(record["events"]) == 1:
            self.persist(record)
        else:
            self.schedule_persist(key)

    def next_batch(self):
        for key, record in self.records.items():
            if time.monotonic() - self.sent.get(key, -100) < 5:
                continue
            pending = [e for e in record["events"] if e["sequence"] not in record["acked"]][:8]
            if pending:
                self.sent[key] = time.monotonic()
                return ExecutionBatch(taskId=key[0], attempt=key[1], events=pending).model_dump(
                    mode="json"
                )
        return None

    def acknowledge(self, batch):
        key = (batch["taskId"], batch["attempt"])
        if record := self.records.get(key):
            valid = {e["sequence"] for e in record["events"]}
            record["acked"] = sorted(set(record["acked"]) | (set(batch["sequences"]) & valid))
            self.sent.pop(key, None)
            self.persist(record)


class ExecutionReporter:
    """Existing executors call report(percent); new executors can call report.stdout(text)."""

    def __init__(self, journal, task, progress):
        self.journal, self.task, self.progress_callback = journal, task, progress
        self.last_percent = -1

    def emit(self, kind, data):
        self.journal.emit(self.task.spec.id, self.task.generation, kind, data)

    def __call__(self, value):
        if not math.isfinite(value):
            return
        percent = max(0, min(100, int(value)))
        self.progress_callback(percent)
        if percent != self.last_percent:
            self.last_percent = percent
            self.emit("progress", {"percent": percent})

    def step(self, message, **data):
        self.emit("step", {**data, "message": str(message)[:2048]})

    def stdout(self, text):
        self.emit("stdout", {"text": str(text)[:2048], "truncated": len(str(text)) > 2048})

    def stderr(self, text):
        self.emit("stderr", {"text": str(text)[:2048], "truncated": len(str(text)) > 2048})
