import asyncio
import json
import tempfile
import unittest
from datetime import UTC, datetime

from orchestrator.shared.execution import ExecutionBatch
from orchestrator.shared.protocol import Task, TaskSpec
from orchestrator.worker.agent import execute
from orchestrator.worker.execution import ExecutionJournal, ExecutionReporter


class ExecutionTests(unittest.IsolatedAsyncioTestCase):
    async def test_success_failure_and_output_survive_restart_without_secrets(self):
        class Executor:
            async def execute(self, spec, report):
                report.step("starting user task")
                report.stdout("normal output")
                report.stderr("Bearer do-not-retain-this")
                report(50)
                if spec.payload.get("fail"):
                    raise ValueError("example failure")
                return {"ok": True}

        with tempfile.TemporaryDirectory() as root:
            journal = ExecutionJournal("https://fleet.test", "machine-1", root)
            for attempt, fail in ((1, False), (2, True)):
                task = Task(
                    spec=TaskSpec(
                        id="task-1",
                        job_id="job-1",
                        kind="user-task",
                        payload={"fail": fail},
                        requirements={"runtime": "cpu", "vram_mib": 0},
                        max_attempts=3,
                        timeout_seconds=10,
                    ),
                    state="running",
                    generation=attempt,
                    created_at=datetime.now(UTC),
                )
                result = await execute(
                    Executor(), task, ExecutionReporter(journal, task, lambda p: None)
                )
                self.assertEqual(result.type, "failed" if fail else "complete")
            restarted = ExecutionJournal("https://fleet.test", "machine-1", root)
            serialized = json.dumps(list(restarted.records.values()))
            self.assertIn("normal output", serialized)
            self.assertNotIn("do-not-retain-this", serialized)
            for record in restarted.records.values():
                self.assertEqual(record["events"][0]["kind"], "started")
                self.assertIn(record["events"][-1]["kind"], {"succeeded", "failed"})
            while batch := restarted.next_batch():
                ExecutionBatch.model_validate(batch)
                restarted.acknowledge(
                    {**batch, "sequences": [e["sequence"] for e in batch["events"]]}
                )
            self.assertIsNone(
                ExecutionJournal("https://fleet.test", "machine-1", root).next_batch()
            )
            self.assertIsNone(
                ExecutionJournal("https://other.test", "machine-1", root).next_batch()
            )

    async def test_output_is_coalesced_on_disk_and_bounded_by_bytes(self):
        with tempfile.TemporaryDirectory() as root:
            journal = ExecutionJournal("fleet", "worker", root)
            path = journal.path(("task-1", 1))
            kinds = lambda: [e["kind"] for e in json.loads(path.read_text())["events"]]  # noqa: E731
            journal.emit("task-1", 1, "started")
            journal.emit("task-1", 1, "stdout", {"text": "x" * 2048})
            self.assertEqual(kinds(), ["started"])
            await asyncio.sleep(0.4)
            self.assertEqual(kinds(), ["started", "stdout"])
            for _ in range(700):
                journal.emit("task-1", 1, "stdout", {"text": "x" * 2048})
            journal.emit("task-1", 1, "succeeded")
            events = kinds()
            self.assertEqual(events.count("truncated"), 1)
            self.assertLess(len(events), 600)
            self.assertEqual(events[-1], "succeeded")
            self.assertLess(path.stat().st_size, 1100 * 1024)

    async def test_interrupted_attempt_is_preserved_after_restart(self):
        with tempfile.TemporaryDirectory() as root:
            journal = ExecutionJournal("fleet", "worker", root)
            journal.emit("task-1", 1, "started")
            recovered = ExecutionJournal("fleet", "worker", root)
            self.assertEqual(recovered.next_batch()["events"][-1]["kind"], "interrupted")

    async def test_cancellation_records_interruption_and_propagates(self):
        class Executor:
            async def execute(self, spec, report):
                raise asyncio.CancelledError()

        with tempfile.TemporaryDirectory() as root:
            task = Task(
                spec=TaskSpec(
                    id="task-1",
                    job_id="job-1",
                    kind="stub",
                    payload={},
                    requirements={"runtime": "cpu", "vram_mib": 0},
                    max_attempts=1,
                    timeout_seconds=1,
                ),
                state="running",
                generation=1,
                created_at=datetime.now(UTC),
            )
            journal = ExecutionJournal("fleet", "worker", root)
            with self.assertRaises(asyncio.CancelledError):
                await execute(Executor(), task, ExecutionReporter(journal, task, lambda p: None))
            self.assertEqual(journal.records[("task-1", 1)]["events"][-1]["kind"], "interrupted")
