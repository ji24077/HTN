"""Single-call OpenAI contract checks; no API key or external requests required."""

import asyncio
import copy
import json
import os
import unittest
from unittest.mock import patch

import httpx

from orchestrator.client import tool_definitions
from orchestrator.llm import ModelClient, ModelClientError, OpenAIClient, ToolCall
from orchestrator.llm.openai import MAX_RESPONSE_BYTES, RESPONSES_URL

KEY = "test-openai-key-never-send-to-network"


def response(*output, **overrides):
    return {
        "id": "resp_test",
        "model": "test-model",
        "status": "completed",
        "output": list(output),
        "usage": {"input_tokens": 10, "output_tokens": 5, "total_tokens": 15},
        **overrides,
    }


def message(text="Ready."):
    return {
        "id": "msg_test",
        "type": "message",
        "role": "assistant",
        "status": "completed",
        "content": [{"type": "output_text", "text": text, "annotations": []}],
    }


def function_call(**overrides):
    return {
        "type": "function_call",
        "id": "fc_test",
        "call_id": "call_test",
        "name": "list_workers",
        "arguments": "{}",
        "status": "completed",
        **overrides,
    }


class ResponseStream(httpx.AsyncByteStream):
    def __init__(self, chunks=(), *, hang=False):
        self.chunks, self.hang, self.closed = chunks, hang, False
        self.started = asyncio.Event()

    async def __aiter__(self):
        self.started.set()
        for chunk in self.chunks:
            yield chunk
        if self.hang:
            await asyncio.Event().wait()

    async def aclose(self):
        self.closed = True


class OpenAIClientTests(unittest.IsolatedAsyncioTestCase):
    def client(self, handler, **options):
        client = OpenAIClient(KEY, "test-model", transport=httpx.MockTransport(handler), **options)
        self.addAsyncCleanup(client.aclose)
        return client

    async def test_single_request_maps_existing_tools_without_mutation(self):
        requests = []

        def handler(request):
            requests.append(request)
            return httpx.Response(
                200, json=response(message()), headers={"x-request-id": "req_test"}
            )

        definitions = tool_definitions()
        original = copy.deepcopy(definitions)
        client: ModelClient = self.client(handler, max_output_tokens=2048)
        result = await client.respond(
            "Which workers are available?", tools=definitions, instructions="Use fleet tools."
        )
        self.assertEqual(len(requests), 1)
        request = requests[0]
        self.assertEqual(str(request.url), RESPONSES_URL)
        self.assertEqual(request.method, "POST")
        self.assertEqual(request.headers["authorization"], f"Bearer {KEY}")
        body = json.loads(request.content)
        self.assertEqual(body["model"], "test-model")
        self.assertEqual(body["instructions"], "Use fleet tools.")
        self.assertEqual(body["max_output_tokens"], 2048)
        self.assertFalse(body["store"])
        self.assertFalse(body["parallel_tool_calls"])
        self.assertEqual(body["include"], ["reasoning.encrypted_content"])
        self.assertNotIn(KEY, request.content.decode())
        self.assertEqual(len(body["tools"]), 7)
        for supplied, encoded in zip(definitions, body["tools"], strict=True):
            self.assertEqual(encoded["parameters"], supplied["input_schema"])
            self.assertEqual(encoded["name"], supplied["name"])
            self.assertEqual(encoded["type"], "function")
            self.assertFalse(encoded["strict"])
        self.assertEqual(definitions, original)
        self.assertEqual(result.text, "Ready.")
        self.assertEqual(result.tool_calls, ())
        self.assertEqual(result.usage.total_tokens, 15)
        self.assertEqual(result.request_id, "req_test")

    async def test_caller_explicitly_continues_with_reasoning_and_tool_result(self):
        reasoning = {
            "type": "reasoning",
            "id": "rs_test",
            "summary": [],
            "encrypted_content": "opaque",
        }
        outputs = [reasoning, function_call(), message("Checking workers.")]
        requests = []

        def handler(request):
            requests.append(json.loads(request.content))
            return httpx.Response(
                200, json=response(*outputs) if len(requests) == 1 else response(message("One."))
            )

        client = self.client(handler)
        history = [{"role": "user", "content": "Which workers are available?"}]
        result = await client.respond(history, tools=tool_definitions())
        self.assertEqual(len(requests), 1)  # No automatic loop or tool execution.
        self.assertEqual(len(history), 1)  # Caller-owned history is unchanged.
        self.assertEqual(result.output, outputs)
        call = result.tool_calls[0]
        self.assertEqual(
            (call.call_id, call.name, call.parse_arguments()), ("call_test", "list_workers", {})
        )
        tool_result = call.output({"ok": True, "result": [{"id": "worker-1"}]})
        self.assertEqual(tool_result["call_id"], "call_test")
        history.extend(result.output)
        history.append(tool_result)
        followup = await client.respond(history, tools=tool_definitions())
        self.assertEqual(len(requests), 2)
        self.assertEqual(requests[1]["input"], history)
        self.assertEqual(requests[1]["input"][1]["encrypted_content"], "opaque")
        self.assertEqual(followup.text, "One.")

    async def test_stateless_reuse_and_multiple_calls_and_refusal(self):
        requests = []
        refusal = message()
        refusal["content"] = [{"type": "refusal", "refusal": "Cannot do that."}]

        def handler(request):
            requests.append(json.loads(request.content))
            return httpx.Response(
                200,
                json=response(
                    function_call(),
                    function_call(call_id="call_two", name="list_tasks"),
                    refusal,
                    usage=None,
                ),
            )

        client = self.client(handler)
        result = await client.respond("First")
        await client.respond("Second")
        self.assertEqual([item["input"] for item in requests], ["First", "Second"])
        self.assertEqual(len(result.tool_calls), 2)
        self.assertEqual(result.refusals, ("Cannot do that.",))
        self.assertEqual(result.text, "")
        self.assertIsNone(result.usage)

    async def test_incomplete_failed_and_malformed_responses_expose_no_calls(self):
        for body, code in [
            (response(function_call(), status="incomplete"), "incomplete_response"),
            (response(status="failed", error={"message": KEY}), "response_failed"),
            (response(status="in_progress"), "invalid_response"),
            (response(function_call(), function_call()), "invalid_response"),
            (response(function_call(arguments=None)), "invalid_response"),
            (response({"type": "unknown_tool"}), "invalid_response"),
            (response(message(), usage={}), "invalid_response"),
            (response(output=None), "invalid_response"),
            (response(None), "invalid_response"),
            ({"output": []}, "invalid_response"),
            ([], "invalid_response"),
        ]:
            with self.subTest(body=body):
                client = self.client(lambda _: httpx.Response(200, json=body))
                with self.assertRaises(ModelClientError) as caught:
                    await client.respond("hello")
                self.assertEqual(caught.exception.code, code)
                self.assertNotIn(KEY, str(caught.exception))

    async def test_invalid_json_and_nonfinite_numbers(self):
        for body in (b"not JSON", b'{"usage": NaN}', b"\xff"):
            client = self.client(lambda _: httpx.Response(200, content=body))
            with self.assertRaises(ModelClientError) as caught:
                await client.respond("hello")
            self.assertEqual(caught.exception.code, "invalid_response")

    async def test_http_errors_and_redirects_are_safe_and_never_retried(self):
        for status, code, retryable in [
            (401, "authentication_failed", False),
            (403, "permission_denied", False),
            (404, "model_not_found", False),
            (429, "rate_limited", True),
            (500, "provider_error", True),
            (307, "provider_error", False),
        ]:
            with self.subTest(status=status):
                requests = []

                def handler(request):
                    requests.append(request)
                    return httpx.Response(
                        status,
                        json={"error": {"message": KEY}},
                        headers={"Location": "https://other.example", "x-request-id": "req_error"},
                    )

                client = self.client(handler)
                with self.assertRaises(ModelClientError) as caught:
                    await client.respond("private prompt")
                error = caught.exception
                self.assertEqual(
                    (error.code, error.retryable, error.status), (code, retryable, status)
                )
                self.assertEqual(error.request_id, "req_error")
                self.assertNotIn(KEY, str(error))
                self.assertNotIn("private prompt", str(error))
                self.assertEqual(len(requests), 1)
                self.assertEqual(requests[0].url.host, "api.openai.com")

    async def test_transport_failure_does_not_retry_or_expose_transport_message(self):
        requests = []

        def handler(request):
            requests.append(request)
            raise httpx.ConnectError(KEY, request=request)

        client = self.client(handler)
        with self.assertRaises(ModelClientError) as caught:
            await client.respond("hello")
        self.assertEqual(caught.exception.code, "connection_error")
        self.assertTrue(caught.exception.retryable)
        self.assertNotIn(KEY, str(caught.exception))
        self.assertEqual(len(requests), 1)

    async def test_wall_clock_timeout_closes_response_stream(self):
        stream = ResponseStream(hang=True)
        client = self.client(lambda _: httpx.Response(200, stream=stream), timeout_seconds=0.02)
        with self.assertRaises(ModelClientError) as caught:
            await client.respond("hello")
        self.assertEqual(caught.exception.code, "timeout")
        self.assertTrue(stream.closed)

    async def test_cancellation_propagates_and_closes_response_stream(self):
        stream = ResponseStream(hang=True)
        client = self.client(lambda _: httpx.Response(200, stream=stream))
        pending = asyncio.create_task(client.respond("hello"))
        await asyncio.wait_for(stream.started.wait(), 1)
        pending.cancel()
        with self.assertRaises(asyncio.CancelledError):
            await pending
        self.assertTrue(stream.closed)

    async def test_oversized_response_is_bounded_and_closed(self):
        stream = ResponseStream([b"x" * (MAX_RESPONSE_BYTES + 1)])
        client = self.client(lambda _: httpx.Response(200, stream=stream))
        with self.assertRaises(ModelClientError) as caught:
            await client.respond("hello")
        self.assertEqual(caught.exception.code, "response_too_large")
        self.assertTrue(stream.closed)

    async def test_invalid_inputs_never_reach_network(self):
        def handler(_):
            self.fail("Invalid input reached the provider")

        client = self.client(handler)
        for prompt in ("", "  ", [], ["hello"]):
            with self.assertRaises(ValueError):
                await client.respond(prompt)
        for tools in ([{}], [tool_definitions()[0]] * 2):
            with self.assertRaises(ValueError):
                await client.respond("hello", tools=tools)
        with self.assertRaises(ValueError):
            await client.respond([{"role": "user", "content": float("nan")}])

    async def test_context_manager_closes_owned_transport(self):
        class Transport(httpx.AsyncBaseTransport):
            closed = False

            async def aclose(self):
                self.closed = True

        transport = Transport()
        async with OpenAIClient(KEY, "test-model", transport=transport):
            self.assertFalse(transport.closed)
        self.assertTrue(transport.closed)

    async def test_environment_configuration_is_explicit(self):
        with patch.dict(os.environ, {}, clear=True):
            with self.assertRaisesRegex(ValueError, "OPENAI_API_KEY and OPENAI_MODEL"):
                OpenAIClient.from_env()
        with patch.dict(os.environ, {"OPENAI_API_KEY": KEY, "OPENAI_MODEL": "chosen-model"}):
            async with OpenAIClient.from_env() as client:
                self.assertEqual(client.model, "chosen-model")

    def test_configuration_validation(self):
        for key, model, kwargs in [
            ("", "model", {}),
            ("key\nvalue", "model", {}),
            (KEY, "", {}),
            (KEY, "model", {"timeout_seconds": float("nan")}),
            (KEY, "model", {"timeout_seconds": 0}),
            (KEY, "model", {"max_output_tokens": True}),
        ]:
            with self.assertRaises(ValueError):
                OpenAIClient(key, model, **kwargs)

    def test_tool_arguments_and_outputs_are_explicit_json_boundaries(self):
        for value in ("not json", "[]", "null", '{"value": NaN}'):
            call = ToolCall("call", "tool", value)
            with self.assertRaisesRegex(ValueError, "JSON object"):
                call.parse_arguments()
        call = ToolCall("call", "tool", '{"value": 1}')
        self.assertEqual(call.parse_arguments(), {"value": 1})
        self.assertEqual(json.loads(call.output({"ok": True})["output"]), {"ok": True})
        with self.assertRaises(ValueError):
            call.output({"value": float("inf")})


if __name__ == "__main__":
    unittest.main()
