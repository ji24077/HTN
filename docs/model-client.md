# Standalone OpenAI client

`orchestrator.llm.OpenAIClient` makes one asynchronous OpenAI Responses request.
It handles configuration, function-schema conversion, response parsing, timeouts,
and provider errors. It does not execute tools, maintain a conversation, retry
requests, or start an agent loop. It has no dependency on the server or workers.
The separate agent loop accepts the `ModelClient` protocol and uses a fake in tests.

Set `OPENAI_API_KEY` and `OPENAI_MODEL` in the backend environment, or pass both
explicitly to the constructor. The configured model is `gpt-6-astra` (GPT-6 Astra),
also set in `.env.example`; it must be available to your OpenAI project.
The client does not load `.env`
itself; use the app launcher or `uv run --project backend --env-file .env`.
Importing the package does not require configuration or open connections.

```python
from orchestrator.client import tool_definitions
from orchestrator.llm import OpenAIClient

async def ask(message: str):
    async with OpenAIClient.from_env() as model:
        return await model.respond(
            message,
            instructions="Help plan tasks. Inspect workers before proposing placement.",
            tools=tool_definitions(),
        )
```

For service use, create one client during startup, reuse it across calls, and
`await model.aclose()` at shutdown. `OpenAIClient(api_key, model,
timeout_seconds=60, max_output_tokens=4096)` configures each call's wall-clock
deadline and output-token budget. `transport=` accepts an HTTPX async transport
for tests. The endpoint is fixed to OpenAI HTTPS; redirects and environment
proxies are disabled.

## Return contract

- `text`: concatenated assistant output text.
- `tool_calls`: `ToolCall` objects with `call_id`, `name`, and original JSON
  `arguments`. `parse_arguments()` requires a finite JSON object; the existing
  `AgentTools.call()` remains responsible for validating each tool's schema.
- `output`: all provider output items, including encrypted reasoning. Retain
  these for continuation; visible text alone loses tool and reasoning state.
- `refusals`: any refusal text, separate from normal output text.
- `usage`: input/output/total token counts when supplied by OpenAI.
- `id`, `model`, `request_id`: provider response/model identifiers and request ID.

Tools use the existing `{name, description, input_schema}` definitions. The
adapter explicitly uses non-strict function schemas because our task payloads
accept arbitrary JSON and optional fields. No tool call is executed by the client.
Only custom function tools are supported in this first client; hosted tools,
streaming, and uploads are outside its current contract.

## Explicit continuation

The caller supplies history on every call. Responses use `store=false` and
request encrypted reasoning for stateless continuation. The client does not
mutate the supplied history or remember it between requests.

Given a completed response and an independently validated/executed tool result:

```python
history = [{"role": "user", "content": "Which workers are available?"}]
first = await model.respond(history, tools=tool_definitions())
history.extend(first.output)

# The separate loop owns authorization, dispatch, and limits. For each returned
# call, it obtains a result (or validation error) from its allowed tool dispatcher.
for call in first.tool_calls:
    result = await allowed_tools.call(call.name, call.parse_arguments())
    history.append(call.output(result))

# Only call again when the caller actually wants another model turn.
second = await model.respond(history, tools=tool_definitions())
```

This snippet illustrates the client boundary. The dashboard uses the separate
[`AgentLoop`](fleet-assistant.md) to manage dispatch and conversation state. Supply
the desired `instructions` on each direct request. Never place credentials in
messages or tool arguments.

## Failures and testing

`ModelClientError` provides `code`, HTTP `status` when applicable, `retryable`,
and `request_id`. Error messages exclude provider response bodies and transport
exception text. A retryable error is a classification, not an automatic retry;
a timed-out request may already have incurred usage. Incomplete responses,
including output-token exhaustion, raise `incomplete_response` instead of
exposing partially generated calls. Caller cancellation propagates normally.

Mocked HTTP tests cover tool conversion, explicit continuation with reasoning,
refusals, malformed and incomplete responses, HTTP errors, redirects, timeouts,
cancellation, response-size limits, configuration, and resource cleanup:

```sh
uv run --project backend python -m unittest discover -s backend/tests/unit -p test_llm.py -v
```

API contract references: [function calling](https://developers.openai.com/api/docs/guides/function-calling)
and [conversation state](https://developers.openai.com/api/docs/guides/conversation-state).
