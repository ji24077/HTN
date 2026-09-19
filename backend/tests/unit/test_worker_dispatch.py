"""Queue handoff does not wait for a liveness heartbeat."""

import unittest
from datetime import UTC, datetime
from unittest.mock import AsyncMock, patch

from starlette.websockets import WebSocketDisconnect, WebSocketState

from orchestrator.server.worker_connection import serve_worker
from orchestrator.shared.protocol import Message, Task


class WorkerDispatchTests(unittest.IsolatedAsyncioTestCase):
    async def test_completion_ack_precedes_immediate_next_assignment(self):
        task = Task(
            spec={
                "id": "next",
                "job_id": "job",
                "kind": "stub",
                "payload": {},
                "requirements": {"runtime": "cpu", "vram_mib": 0},
                "max_attempts": 3,
                "timeout_seconds": 60,
            },
            state="assigned",
            generation=1,
            created_at=datetime.now(UTC),
        )
        store = AsyncMock()
        store.claim.return_value = task
        socket = AsyncMock()
        socket.application_state = WebSocketState.CONNECTED
        messages = [
            Message(
                type="hello", capabilities={"runtime": "cpu", "vram_mib": 0, "kinds": ["stub"]}
            ),
            Message(type="complete", ref={"task_id": "previous", "generation": 1}, result={}),
            WebSocketDisconnect(),
        ]
        with patch(
            "orchestrator.server.worker_connection.receive", AsyncMock(side_effect=messages)
        ):
            await serve_worker(socket, store, None, "worker-a")
        sent = [Message.model_validate_json(c.args[0]) for c in socket.send_text.call_args_list]
        self.assertEqual([m.type for m in sent], ["welcome", "result_accepted", "assign"])
        self.assertEqual(sent[-1].task.spec.id, "next")
        store.finish.assert_awaited_once()
        store.claim.assert_awaited_once()
        store.heartbeat.assert_not_awaited()
