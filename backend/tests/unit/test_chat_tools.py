"""Fleet tool bindings derive their catalog and tool subset from single sources."""

import unittest
from datetime import UTC, datetime
from unittest.mock import AsyncMock

from orchestrator.client.tools import TOOLS
from orchestrator.server.chat_tools import (
    EXAMPLE_PAYLOADS,
    FLEET_TOOLS,
    KINDS,
    FleetTools,
    ServerTaskClient,
    task_summary,
)
from orchestrator.shared.protocol import Task


def fake_request(store):
    app = type("App", (), {})()
    app.state = type("State", (), {})()
    app.state.store = store
    request = type("Request", (), {})()
    request.app = app
    return request


def task(task_id="task-1", payload=None):
    return Task(
        spec={
            "id": task_id,
            "job_id": "job-1",
            "kind": "walker_evolution",
            "payload": payload if payload is not None else {"parent": [0.0] * 308},
            "requirements": {"runtime": "cpu", "vram_mib": 0},
            "max_attempts": 3,
            "timeout_seconds": 120,
        },
        state="succeeded",
        generation=1,
        result={"scores": [1.0] * 4096},
        created_at=datetime.now(UTC),
    )


class ChatToolsTests(unittest.IsolatedAsyncioTestCase):
    def test_submittable_kinds_match_advertised_payloads(self):
        self.assertEqual(KINDS, frozenset(EXAMPLE_PAYLOADS))

    def test_fleet_tools_are_registry_tools_without_wait(self):
        self.assertTrue(set(FLEET_TOOLS) <= TOOLS.keys())
        self.assertNotIn("wait_task", FLEET_TOOLS)

    def test_definitions_are_built_once_and_exclude_wait(self):
        tools = FleetTools(fake_request(store=None))
        names = [tool["name"] for tool in tools.definitions()]
        self.assertEqual(names, [*FLEET_TOOLS, "list_workloads"])
        self.assertIs(tools.definitions(), tools.definitions())
        for name in ("list_tasks", "list_events"):
            description = next(t["description"] for t in tools.definitions() if t["name"] == name)
            self.assertIn("get_task", description)

    async def test_list_tasks_omits_payload_and_result(self):
        store = type("Store", (), {})()
        store.tasks = AsyncMock(return_value=[task(str(i)) for i in range(50)])
        rows = await ServerTaskClient(fake_request(store)).list_tasks()
        self.assertEqual(len(rows), 50)
        self.assertNotIn("payload", rows[0]["spec"])
        self.assertNotIn("result", rows[0])
        self.assertEqual(rows[0]["spec"]["kind"], "walker_evolution")
        self.assertEqual(rows[0]["state"], "succeeded")
        self.assertLess(len(str(rows)), 64 * 1024)

    def test_task_summary_keeps_identity_and_state(self):
        summary = task_summary(task("abc"))
        self.assertEqual(summary["spec"]["id"], "abc")
        self.assertEqual(summary["spec"]["job_id"], "job-1")
        self.assertIn("created_at", summary)
        self.assertNotIn("attestation", summary)
