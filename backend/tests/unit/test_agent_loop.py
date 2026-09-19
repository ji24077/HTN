"""The agent loop executes tools; the model client remains a single-call dependency."""

import asyncio
import copy
import unittest
from unittest.mock import AsyncMock
from uuid import uuid4

from orchestrator.agent import AgentLoop, Conversation, Turn
from orchestrator.agent.loop import finish_interrupted
from orchestrator.llm import ModelClientError, ModelResponse, ToolCall


def answer(text="Done."):
    return ModelResponse("response", "fake", text, (), [{"role": "assistant", "content": text}])


def calls(*items):
    return ModelResponse(
        "response",
        "fake",
        "",
        tuple(items),
        [
            {"type": "reasoning", "encrypted_content": "opaque", "summary": []},
            *[
                {
                    "type": "function_call",
                    "call_id": item.call_id,
                    "name": item.name,
                    "arguments": item.arguments,
                }
                for item in items
            ],
        ],
    )


def conversation():
    return Conversation(
        turns=[Turn(request_id=uuid4(), message="Run a test")],
        history=[{"role": "user", "content": "Run a test"}],
    )


class FakeTools:
    def __init__(self):
        self.call = AsyncMock(return_value={"ok": True, "result": {"task_id": "test-task"}})

    def definitions(self):
        return [
            {"name": "submit_tasks", "description": "Submit", "input_schema": {"type": "object"}}
        ]


class AgentLoopTests(unittest.IsolatedAsyncioTestCase):
    async def test_tool_result_and_reasoning_continue_then_followup(self):
        state, tools = conversation(), FakeTools()
        saved = []

        async def checkpoint(value):
            saved.append(copy.deepcopy(value))

        model = type("Model", (), {})()
        model.respond = AsyncMock(
            side_effect=[calls(ToolCall("call-1", "submit_tasks", "{}")), answer()]
        )
        turn = await AgentLoop(model).run(state, tools, checkpoint)
        self.assertEqual(turn.status, "completed")
        self.assertEqual(turn.tools[0].status, "completed")
        tools.call.assert_awaited_once_with("submit_tasks", {})
        self.assertTrue(
            any(s.turns[0].tools and s.turns[0].tools[0].status == "running" for s in saved)
        )
        self.assertTrue(any(item.get("encrypted_content") == "opaque" for item in state.history))
        self.assertTrue(any(item.get("type") == "function_call_output" for item in state.history))
        state.turns.append(Turn(request_id=uuid4(), message="How is it doing?"))
        state.history.append({"role": "user", "content": "How is it doing?"})
        model.respond = AsyncMock(return_value=answer("Queued."))
        await AgentLoop(model).run(state, tools, checkpoint)
        self.assertEqual(state.turns[-1].reply, "Queued.")
        self.assertIn("test-task", str(model.respond.call_args.args[0]))

    async def test_invalid_and_unknown_calls_are_reported_without_execution(self):
        state, tools = conversation(), FakeTools()
        model = type("Model", (), {})()
        model.respond = AsyncMock(
            side_effect=[
                calls(
                    ToolCall("call-1", "submit_tasks", "not json"),
                    ToolCall("call-2", "execute_shell", "{}"),
                ),
                answer("Could not run those tools."),
            ]
        )
        turn = await AgentLoop(model).run(state, tools, AsyncMock())
        tools.call.assert_not_awaited()
        self.assertEqual(
            [item.result["error"]["code"] for item in turn.tools],
            ["invalid_arguments", "unknown_tool"],
        )

    async def test_checkpoint_failure_prevents_execution(self):
        state, tools = conversation(), FakeTools()
        model = type("Model", (), {})()
        model.respond = AsyncMock(return_value=calls(ToolCall("call-1", "submit_tasks", "{}")))
        with self.assertRaises(ConnectionError):
            await AgentLoop(model).run(state, tools, AsyncMock(side_effect=ConnectionError))
        tools.call.assert_not_awaited()

    async def test_model_error_after_submission_preserves_completed_activity(self):
        state, tools = conversation(), FakeTools()
        model = type("Model", (), {})()
        model.respond = AsyncMock(
            side_effect=[
                calls(ToolCall("call-1", "submit_tasks", "{}")),
                ModelClientError("rate_limited", "private detail"),
            ]
        )
        turn = await AgentLoop(model).run(state, tools, AsyncMock())
        self.assertEqual(turn.status, "failed")
        self.assertEqual(turn.tools[0].status, "completed")
        self.assertIn("test-task", str(turn.tools[0].result))
        self.assertNotIn("private detail", turn.reply)
        tools.call.assert_awaited_once()

    async def test_unknown_mutation_outcome_stops_further_model_calls(self):
        state, tools = conversation(), FakeTools()
        tools.call.return_value = {"ok": False, "error": {"code": "connection_error"}}
        model = type("Model", (), {})()
        model.respond = AsyncMock(return_value=calls(ToolCall("call-1", "submit_tasks", "{}")))
        turn = await AgentLoop(model).run(state, tools, AsyncMock())
        self.assertEqual(turn.status, "failed")
        model.respond.assert_awaited_once()

    async def test_tool_and_model_budgets_stop_without_pending_output_items(self):
        state, tools = conversation(), FakeTools()
        model = type("Model", (), {})()
        model.respond = AsyncMock(
            return_value=calls(
                ToolCall("call-1", "submit_tasks", "{}"), ToolCall("call-2", "submit_tasks", "{}")
            )
        )
        turn = await AgentLoop(model, max_steps=1).run(state, tools, AsyncMock())
        self.assertEqual(turn.status, "failed")
        tools.call.assert_awaited_once()
        self.assertEqual(
            len([item for item in state.history if item.get("type") == "function_call_output"]), 2
        )

    async def test_timeout_marks_pending_mutation_unknown_and_does_not_replay(self):
        state, tools = conversation(), FakeTools()

        async def hang(*args):
            await asyncio.Event().wait()

        tools.call.side_effect = hang
        model = type("Model", (), {})()
        model.respond = AsyncMock(return_value=calls(ToolCall("call-1", "submit_tasks", "{}")))
        turn = await AgentLoop(model, timeout_seconds=0.01).run(state, tools, AsyncMock())
        self.assertEqual(turn.status, "failed")
        self.assertEqual(turn.tools[0].result["error"]["code"], "outcome_unknown")
        tools.call.assert_awaited_once()

    async def test_checkpoint_timeout_is_a_persistence_error_not_the_deadline(self):
        state, tools = conversation(), FakeTools()
        model = type("Model", (), {})()
        model.respond = AsyncMock(return_value=calls(ToolCall("call-1", "submit_tasks", "{}")))
        with self.assertRaises(TimeoutError):
            await AgentLoop(model).run(state, tools, AsyncMock(side_effect=TimeoutError))
        tools.call.assert_not_awaited()
        self.assertEqual(state.turns[0].status, "running")

    async def test_process_interruption_closes_unexecuted_calls(self):
        state = conversation()
        state.history.extend(calls(ToolCall("call-1", "submit_tasks", "{}")).output)
        finish_interrupted(state, "Interrupted")
        self.assertEqual(state.turns[0].status, "failed")
        result = next(item for item in state.history if item.get("type") == "function_call_output")
        self.assertIn("not_executed", result["output"])

    async def test_repeated_tool_id_is_not_executed_again(self):
        state, tools = conversation(), FakeTools()
        model = type("Model", (), {})()
        model.respond = AsyncMock(return_value=calls(ToolCall("call-1", "submit_tasks", "{}")))
        turn = await AgentLoop(model).run(state, tools, AsyncMock())
        self.assertEqual(turn.status, "failed")
        self.assertIn("repeated", turn.reply)
        tools.call.assert_awaited_once()


if __name__ == "__main__":
    unittest.main()
