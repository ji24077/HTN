"""Chat API + real PostgreSQL + fake model. Never uses configured Supabase or OpenAI."""

import asyncio
import os
import tempfile
import unittest
from pathlib import Path
from types import SimpleNamespace
from unittest.mock import AsyncMock
from uuid import uuid4

import httpx
from fastapi import HTTPException

from orchestrator.agent import AgentLoop
from orchestrator.llm import ModelResponse, ToolCall
from orchestrator.server.app import create_public_app, create_worker_app
from orchestrator.server.auth import SESSION_COOKIE
from orchestrator.server.chat_store import ChatStore
from orchestrator.server.db.store import Conflict, NotFound, Store
from orchestrator.shared.protocol import Capabilities, json_text

ORIGIN = "https://fleet.example.com"


def tool(call_id, name, arguments):
    call = ToolCall(call_id, name, json_text(arguments))
    return ModelResponse(
        "resp",
        "fake",
        "",
        (call,),
        [
            {
                "type": "function_call",
                "call_id": call_id,
                "name": name,
                "arguments": call.arguments,
            }
        ],
    )


def answer(text):
    return ModelResponse("resp", "fake", text, (), [{"role": "assistant", "content": text}])


@unittest.skipUnless(os.getenv("RUN_CHAT_E2E") == "1", "Set RUN_CHAT_E2E=1; needs demo extra")
class ChatIntegrationTests(unittest.IsolatedAsyncioTestCase):
    async def test_authenticated_conversation_tools_replay_and_recovery(self):
        import pgserver

        with tempfile.TemporaryDirectory(prefix="chat-test-") as directory:
            database = pgserver.get_server(Path(directory) / "postgres", cleanup_mode="stop")
            store = await Store.open(database.get_uri(), schema="chat_test")
            self.addAsyncCleanup(store.close)
            try:
                app = create_public_app()
                user_a, user_b = str(uuid4()), str(uuid4())

                async def verify(token):
                    if token not in {user_a, user_b}:
                        raise HTTPException(401, "unauthorized")
                    return {"sub": token}

                app.state.config = SimpleNamespace(
                    admin_token="test-admin-credential-123456789",
                    public_origin=ORIGIN,
                    worker_tokens={"worker-1": "worker-test-credential"},
                    supabase_admin_ids={user_a, user_b},
                )
                app.state.supabase_auth = SimpleNamespace(verify=verify)
                app.state.store = store
                app.state.chat_store = ChatStore(store.pool)
                model = SimpleNamespace(respond=AsyncMock())
                app.state.chat_agent = AgentLoop(model)
                # The worker need only be listed; this test verifies real queue submission.
                await store.register(
                    "worker-1", "session-1", Capabilities(runtime="cpu", vram_mib=0, kinds=["stub"])
                )
                spec = {
                    "id": "chat-task",
                    "job_id": "chat-job",
                    "kind": "stub",
                    "payload": {"duration_seconds": 2},
                    "requirements": {"runtime": "cpu", "vram_mib": 0},
                    "max_attempts": 3,
                    "timeout_seconds": 120,
                    "target_worker_id": "worker-1",
                }
                model.respond.side_effect = [
                    tool("call-1", "list_workers", {}),
                    tool("call-2", "submit_tasks", {"tasks": [spec]}),
                    answer("Queued chat-task."),
                    tool("call-3", "get_task", {"task_id": "chat-task"}),
                    answer("chat-task is queued."),
                ]
                headers = {"Authorization": f"Bearer {user_a}"}
                async with httpx.AsyncClient(
                    transport=httpx.ASGITransport(app=app), base_url=ORIGIN
                ) as client:
                    self.assertEqual((await client.get("/v1/chat/config")).status_code, 401)
                    chat_id = uuid4()
                    path = f"/v1/chat/{chat_id}"
                    body = {"request_id": str(uuid4()), "message": "Run a connection test"}
                    response = await client.post(path + "/messages", headers=headers, json=body)
                    self.assertEqual(response.status_code, 200, response.text)
                    self.assertEqual(response.json()["status"], "completed")
                    self.assertEqual(len(response.json()["tools"]), 2)
                    self.assertEqual((await store.task("chat-task")).state, "queued")
                    again = await client.post(path + "/messages", headers=headers, json=body)
                    self.assertEqual(again.json(), response.json())
                    self.assertEqual(model.respond.await_count, 3)
                    self.assertEqual(len(await store.tasks()), 1)
                    conflict = await client.post(
                        path + "/messages", headers=headers, json={**body, "message": "Different"}
                    )
                    self.assertEqual(conflict.status_code, 409)
                    followup = await client.post(
                        path + "/messages",
                        headers=headers,
                        json={"request_id": str(uuid4()), "message": "How is it doing?"},
                    )
                    self.assertEqual(followup.json()["reply"], "chat-task is queued.")
                    transcript = await client.get(path, headers=headers)
                    self.assertEqual(len(transcript.json()["turns"]), 2)
                    self.assertNotIn("history", transcript.json())
                    for method, suffix in (("GET", ""), ("POST", "/messages")):
                        denied = await client.request(
                            method,
                            path + suffix,
                            headers={"Authorization": f"Bearer {user_b}"},
                            json=body,
                        )
                        self.assertEqual(denied.status_code, 404)
                    client.cookies.set(SESSION_COOKIE, user_a)
                    cross_site = await client.post(
                        path + "/messages", headers={"Origin": "https://evil.example"}, json=body
                    )
                    self.assertEqual(cross_site.status_code, 401)
                    oversized = await client.post(
                        path + "/messages", headers=headers, content=b"x" * 32769
                    )
                    self.assertEqual(oversized.status_code, 413)
                    injected = await client.post(
                        path + "/messages", headers=headers, json={**body, "history": []}
                    )
                    self.assertEqual(injected.status_code, 400)
                    app.state.chat_agent = None
                    self.assertFalse(
                        (await client.get("/v1/chat/config", headers=headers)).json()["enabled"]
                    )
                    disabled = await client.post(path + "/messages", headers=headers, json=body)
                    self.assertEqual(disabled.status_code, 503)
                    app.state.chat_agent = AgentLoop(model)
                    # Interrupt after persisting a pending turn. The same request must
                    # return an interrupted result, never execute the model again.
                    interrupted_id = uuid4()
                    async with app.state.chat_store.edit("user:" + user_a, chat_id) as (
                        state,
                        save,
                    ):
                        from orchestrator.agent import Turn

                        state.turns.append(Turn(request_id=interrupted_id, message="Interrupted"))
                        state.history.append({"role": "user", "content": "Interrupted"})
                        await save(state)
                        with self.assertRaises(Conflict):
                            async with app.state.chat_store.edit("user:" + user_a, uuid4()):
                                self.fail("Concurrent principal lock was accepted")
                    recovered = await client.post(
                        path + "/messages",
                        headers=headers,
                        json={"request_id": str(interrupted_id), "message": "Interrupted"},
                    )
                    self.assertEqual(recovered.json()["status"], "failed")
                    self.assertEqual(model.respond.await_count, 5)
                    # A second store instance reads persisted history after restart.
                    persisted = await ChatStore(store.pool).read("user:" + user_a, chat_id)
                    self.assertEqual(persisted.turns[-1].status, "failed")
                    with self.assertRaises(NotFound):
                        await ChatStore(store.pool).read("user:" + user_b, chat_id)
                    app.state.chat_slots = asyncio.Semaphore(0)
                    self.assertEqual(
                        (
                            await client.post(path + "/messages", headers=headers, json=body)
                        ).status_code,
                        429,
                    )
                private = create_worker_app()
                async with httpx.AsyncClient(
                    transport=httpx.ASGITransport(app=private), base_url=ORIGIN
                ) as client:
                    self.assertEqual((await client.get("/v1/chat/config")).status_code, 404)
            finally:
                await store.close()
                database.cleanup()


if __name__ == "__main__":
    unittest.main()
