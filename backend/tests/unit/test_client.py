"""Client boundary checks; no database, workers, or remote resources needed."""

import asyncio
import json
import unittest

import httpx

from orchestrator.client import AgentTools, ClientError, OrchestratorClient
from orchestrator.client.http import validate_url
from orchestrator.shared.protocol import TaskSpec

SPEC = {
    "id": "test-001",
    "job_id": "test",
    "kind": "stub",
    "payload": {},
    "requirements": {"runtime": "cpu", "vram_mib": 0},
    "max_attempts": 3,
    "timeout_seconds": 60,
}


def task(state="running", **extra):
    return {
        "spec": SPEC,
        "state": state,
        "generation": 1,
        "created_at": "2026-09-19T00:00:00Z",
        **extra,
    }


class EventStream(httpx.AsyncByteStream):
    def __init__(self, frames=(), hang=False):
        self.frames, self.hang, self.closed = frames, hang, False

    async def __aiter__(self):
        for frame in self.frames:
            yield frame.encode()
        if self.hang:
            await asyncio.Event().wait()

    async def aclose(self):
        self.closed = True


class ClientTests(unittest.IsolatedAsyncioTestCase):
    async def client(self, handler):
        client = OrchestratorClient("https://control.example", "test-token")
        await client.http.aclose()
        client.http = httpx.AsyncClient(
            base_url="https://control.example/",
            transport=httpx.MockTransport(handler),
            headers={"Authorization": "Bearer test-token"},
            follow_redirects=False,
        )
        self.addAsyncCleanup(client.aclose)
        return client

    async def test_submit_does_not_retry_uncertain_write(self):
        requests = []

        def handler(request):
            requests.append(request)
            raise httpx.ReadTimeout("lost response", request=request)

        client = await self.client(handler)
        with self.assertRaises(ClientError) as caught:
            await client.submit_tasks([TaskSpec.model_validate(SPEC)])
        self.assertEqual(caught.exception.code, "connection_error")
        self.assertEqual(len(requests), 1)
        self.assertEqual(json.loads(requests[0].content)["tasks"][0]["id"], SPEC["id"])

    async def test_redirect_does_not_forward_credentials(self):
        requests = []

        def handler(request):
            requests.append(request)
            return httpx.Response(307, headers={"Location": "https://other.example/"})

        client = await self.client(handler)
        with self.assertRaises(ClientError):
            await client.list_workers()
        self.assertEqual(len(requests), 1)
        self.assertEqual(requests[0].url.host, "control.example")

    async def test_invalid_arguments_and_unknown_tools_make_no_requests(self):
        def handler(_request):
            self.fail("invalid tool call reached network")

        tools = AgentTools(await self.client(handler))
        for name, arguments, code in [
            ("get_task", {"task_id": "../escape"}, "invalid_arguments"),
            ("list_workers", {"token": "not-allowed"}, "invalid_arguments"),
            ("execute_shell", {}, "unknown_tool"),
        ]:
            result = await tools.call(name, arguments)
            self.assertFalse(result["ok"])
            self.assertEqual(result["error"]["code"], code)

    async def test_tool_subset_limits_definitions_and_dispatch(self):
        def handler(_request):
            self.fail("excluded tool reached network")

        tools = AgentTools(await self.client(handler), ["get_task", "list_workers"])
        self.assertEqual(
            [tool["name"] for tool in tools.definitions()], ["list_workers", "get_task"]
        )
        result = await tools.call("wait_task", {"task_id": "test-001"})
        self.assertEqual(result["error"]["code"], "unknown_tool")
        with self.assertRaises(ValueError):
            AgentTools(await self.client(handler), ["execute_shell"])

    async def test_wait_uses_stream_and_closes_it(self):
        calls = []
        # Comments, multiline data, and chunk boundaries are valid SSE.
        frame = (
            'event: snapshot\ndata: {"tasks":\ndata: '
            + json.dumps([task("succeeded", id="test-001")])
            + "}\n\n"
        )
        stream = EventStream([": keepalive\n\n", frame[:20], frame[20:]], hang=True)

        def handler(request):
            calls.append(request.url.path)
            if request.url.path.endswith("/updates"):
                return httpx.Response(200, stream=stream)
            return httpx.Response(200, json=task())

        client = await self.client(handler)
        result = await client.wait_task("test-001", 1)
        self.assertEqual(result.state, "succeeded")
        self.assertEqual(calls, ["/v1/tasks/test-001", "/v1/updates"])
        self.assertTrue(stream.closed)

    async def test_wait_timeout_does_not_cancel_and_closes_stream(self):
        calls = []
        stream = EventStream(hang=True)

        def handler(request):
            calls.append((request.method, request.url.path))
            if request.url.path.endswith("/updates"):
                return httpx.Response(200, stream=stream)
            return httpx.Response(200, json=task())

        client = await self.client(handler)
        with self.assertRaises(ClientError) as caught:
            await client.wait_task("test-001", 0.05)
        self.assertEqual(caught.exception.code, "wait_timeout")
        self.assertTrue(stream.closed)
        self.assertTrue(all(method == "GET" for method, _ in calls))

    async def test_wait_for_task_outside_snapshot_window(self):
        reads = 0
        stream = EventStream(['event: snapshot\ndata: {"tasks":[]}\n\n'], hang=True)

        def handler(request):
            nonlocal reads
            if request.url.path.endswith("/updates"):
                return httpx.Response(200, stream=stream)
            reads += 1
            return httpx.Response(200, json=task("running" if reads == 1 else "failed"))

        tools = AgentTools(await self.client(handler))
        result = await tools.call("wait_task", {"task_id": "test-001", "timeout_seconds": 1})
        self.assertTrue(result["ok"])
        self.assertEqual(result["result"]["state"], "failed")
        self.assertEqual(reads, 2)
        self.assertTrue(stream.closed)

    async def test_auth_failure_is_fatal_and_conflict_is_structured(self):
        calls = []

        def handler(request):
            calls.append(request.url.path)
            if request.method == "POST":
                return httpx.Response(
                    409, json={"error": "task ID already exists with a different specification"}
                )
            return httpx.Response(401, json={"detail": "unauthorized"})

        client = await self.client(handler)
        stream = client.updates()
        with self.assertRaises(ClientError) as caught:
            await anext(stream)
        self.assertEqual(caught.exception.code, "unauthorized")
        result = await AgentTools(client).call("submit_tasks", {"tasks": [SPEC]})
        self.assertEqual(result["error"]["code"], "conflict")
        self.assertFalse(result["error"]["retryable"])
        self.assertEqual(len(calls), 2)

    def test_url_transport_boundary(self):
        for url in (
            "http://remote.example",
            "https://user:secret@example.com",
            "file:///tmp/socket",
        ):
            with self.assertRaises(ValueError):
                validate_url(url)
        self.assertEqual(validate_url("http://127.0.0.1:8787/"), "http://127.0.0.1:8787")


if __name__ == "__main__":
    unittest.main()
