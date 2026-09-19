"""Callable progress reporter plus durable task-scoped stdout/stderr/step hooks."""

import hashlib
import json
import math
import os
import time
from datetime import UTC, datetime
from pathlib import Path

from ..shared.execution import ExecutionBatch, ExecutionEvent, scrub_execution
from ..shared.protocol import json_text

TERMINAL = {"succeeded", "failed", "cancelled", "timed_out", "interrupted"}


class ExecutionJournal:
    def __init__(self, server, worker_id, root=None):
        root = Path(root or os.getenv("WORKER_EXECUTION_DIR", str(Path.home() / ".dwp/executions")))
        self.directory = root / hashlib.sha256(f"{server}\n{worker_id}".encode()).hexdigest()
        self.records = {}
        self.sent = {}
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
                    record = json.loads(path.read_text())
                    if not isinstance(record["acked"], list):
                        continue
                    for item in record["events"]:
                        ExecutionEvent.model_validate(item)
                    key = (record["taskId"], record["attempt"])
                    self.records[key] = record
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
        try:
            path = self.path((record["taskId"], record["attempt"]))
            temporary = path.with_suffix(".tmp")
            with os.fdopen(
                os.open(temporary, os.O_WRONLY | os.O_CREAT | os.O_TRUNC, 0o600), "w"
            ) as out:
                out.write(json_text(record))
            temporary.replace(path)
        except OSError:
            pass  # Optional diagnostics cannot change a task's result.

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
                try:
                    self.path(oldest).unlink(missing_ok=True)
                except OSError:
                    pass
            self.records[key] = {"taskId": task_id, "attempt": attempt, "events": [], "acked": []}
        record = self.records[key]
        if len(record["events"]) >= 999 or (record.get("truncated") and kind not in TERMINAL):
            return
        clean = scrub_execution(data or {})
        if len(json_text(clean).encode()) > 8192:
            clean = {"message": "Event exceeded 8 KiB", "truncated": True}
        if kind not in TERMINAL and (
            len(record["events"]) >= 996 or len(json_text(record)) > 1000 * 1024
        ):
            kind, clean = (
                "truncated",
                {"message": "Output limit reached; completion is still recorded"},
            )
            record["truncated"] = True
        item = ExecutionEvent(
            sequence=len(record["events"]) + 1, at=datetime.now(UTC), kind=kind, data=clean
        )
        record["events"].append(item.model_dump(mode="json"))
        self.persist(record)

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
