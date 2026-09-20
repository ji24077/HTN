"""Refused offers preserve retries and stale-assignment protection."""

import unittest
from contextlib import asynccontextmanager
from datetime import UTC, datetime, timedelta
from unittest.mock import AsyncMock, patch

from orchestrator.server.db.store import StaleAssignment, Store, task_events
from orchestrator.shared.execution import ExecutionBatch
from orchestrator.shared.protocol import Task, TaskSpec, task_ref
from orchestrator.supervisor.models import LogSearch


def assignment(*, generation=1, state="assigned"):
    now = datetime.now(UTC)
    return Task(
        spec=TaskSpec(
            id="deferred-task", job_id="job", kind="echo", payload={},
            requirements={"runtime": "cpu", "vram_mib": 0},
            max_attempts=2, timeout_seconds=60,
        ),
        state=state, generation=generation, worker_id="worker", session_id="session",
        lease_until=now + timedelta(seconds=45), deadline=now + timedelta(seconds=60),
        created_at=now,
    )


class DeferralTests(unittest.IsolatedAsyncioTestCase):
    async def test_many_refusals_do_not_spend_retries_but_real_failures_do(self):
        conn = AsyncMock()
        conn.fetchval.return_value = 12
        with patch("orchestrator.server.db.store.event", new_callable=AsyncMock):
            # Twelve refused offers precede the first real execution.
            await Store._fail(conn, assignment(generation=13, state="running"), "error", True)
            self.assertEqual(conn.execute.call_args.args[2], "queued")
            await Store._fail(conn, assignment(generation=14, state="running"), "error", True)
            self.assertEqual(conn.execute.call_args.args[2], "failed")
            await Store._fail(conn, assignment(generation=13, state="running"), "permanent", False)
            self.assertEqual(conn.execute.call_args.args[2], "failed")

    async def test_only_unaccepted_current_offers_can_be_deferred(self):
        conn = AsyncMock()
        store = object.__new__(Store)
        current = assignment(generation=13)

        @asynccontextmanager
        async def change():
            yield conn, datetime.now(UTC)

        store.change = change
        store._owned = AsyncMock(return_value=current)
        with patch("orchestrator.server.db.store.event", new_callable=AsyncMock) as event:
            await store.decline("worker", "session", task_ref(current))
            self.assertEqual(current.generation, 13)
            self.assertEqual(event.call_args.kwargs["reason"], "offer_declined")
            self.assertEqual(event.call_args.kwargs["generation"], 13)
            for state, worker, session, generation in [
                ("running", "worker", "session", 13),
                ("assigned", "other", "session", 13),
                ("assigned", "worker", "other", 13),
                ("assigned", "worker", "session", 12),
            ]:
                current.state = state
                ref = task_ref(current).model_copy(update={"generation": generation})
                with self.assertRaises(StaleAssignment):
                    await store.decline(worker, session, ref)
            self.assertEqual(conn.execute.await_count, 1)

    async def test_refusal_is_queued_in_diagnostics_not_failed(self):
        conn = AsyncMock()
        current = assignment()
        await task_events(conn, [(current.spec.id, "assigned", "queued",
                                 current.spec.model_dump(mode="json"),
                                 {"reason": "offer_declined", "generation": 1})])
        self.assertEqual(conn.execute.call_args.args[1][0]["kind"], "queued")

    async def test_diagnostics_accept_generation_beyond_retry_limit(self):
        batch = ExecutionBatch.model_validate({
            "taskId": "deferred-task", "attempt": 13,
            "events": [{"sequence": 1, "at": datetime.now(UTC), "kind": "started", "data": {}}],
        })
        self.assertEqual(batch.attempt, 13)
        self.assertEqual(LogSearch(attempt=13).attempt, 13)
