# Fleet assistant

The dashboard's **Fleet assistant** accepts natural-language requests and uses
the existing task tools. Try “Which workers are available?”, “Run a connection
test on an available worker”, then “How is that task doing?”. Completed tool
calls show their arguments and results in expandable rows. Current tool activity
is fetched every 1.5 seconds while a message is pending; fleet updates still use SSE.

Set `OPENAI_API_KEY` and `OPENAI_MODEL=gpt-6-astra` in the backend environment and
restart the backend. With no key, the panel reports that chat is not configured;
the rest of the dashboard continues to work. The model key never reaches the browser.

## Separate components

- `orchestrator.llm.OpenAIClient`: one model request; no dispatch or history.
- `orchestrator.agent.AgentLoop`: repeated model calls, validated tool dispatch,
  completion, time limits, and persistence checkpoints. It depends on the
  `ModelClient` protocol, a tools object, and an async checkpoint callback.
- `server/chat_tools.py`: binds the existing `AgentTools` dispatcher to the
  caller's authorization and the task store. It checks fleet access before every
  tool action, including reads, and shares API target-worker submission checks.
- `server/chat.py` and `chat_store.py`: authenticated HTTP and private PostgreSQL
  conversation storage. HTTP/session concerns stay outside the agent loop.
- `frontend/src/components/ChatPanel.tsx`: messages, follow-ups, tool activity,
  recovery, and starting a fresh conversation.

The loop's reusable call boundary is:

```python
from uuid import uuid4
from orchestrator.agent import AgentLoop, Conversation, Turn

state = Conversation(
    history=[{"role": "user", "content": message}],
    turns=[Turn(request_id=uuid4(), message=message)],
)
reply = await AgentLoop(model).run(state, authorized_tools, save_checkpoint)
```

The caller serializes access to a conversation and supplies the tools and
persistence callback. `authorized_tools` exposes `definitions()` and async
`call(name, arguments)`. The backend uses `OpenAIClient`; tests use a fake model.

## Tools and limits

Chat exposes `list_workers`, `list_tasks`, `get_task`, `submit_tasks`, `cancel_task`,
`list_events`, and `list_workloads`. The latter returns payload templates and the
configured inference fixture. Submissions allow up to ten tasks per tool call,
using only `stub`, `echo`, `walker_evolution`, and `cpu_inference_batch`.
Long `wait_task` calls are omitted: users can ask for status in a follow-up.
Rendering, arbitrary code, uploads, and job splitting remain unimplemented.

A turn is limited to eight model requests/eight tool calls and 90 seconds. Model
requests retain the single-call client's 60-second deadline and token budget.
History is bounded at approximately 512 KiB; each conversation allows 40 messages.
There is one active turn per principal across backend processes and two active
turns per process. A PostgreSQL session advisory lock serializes turns without
holding a transaction or the scheduler lock during model requests.

## API and recovery

All routes require the existing approved fleet access and only appear on the
public/combined listener. Cookie mutations require the exact browser origin.

| Method | Route | Result |
| --- | --- | --- |
| GET | `/v1/chat/config` | Whether chat is configured |
| GET | `/v1/chat/{conversation_id}` | Caller-owned visible turns and tool activity |
| POST | `/v1/chat/{conversation_id}/messages` | `{request_id, message}` → completed/failed turn |

Both IDs are UUIDs. The browser retains only its conversation ID in session
storage, scoped to the displayed account; the backend independently enforces
ownership. Browser requests cannot supply history, instructions, or tool results.
Raw model/reasoning continuation items stay server-side.

Each message's request ID is an idempotency key within its conversation. Reusing
the ID and text returns the saved result; changing its text returns 409. While
a turn is active, another turn by the same principal returns 409. **Check response**
reuses the original IDs after a lost HTTP response, including after a refresh.
It does not generate a new task submission.

Tool intent is checkpointed before execution, and its result after execution.
If the process stops between those checkpoints, the recovered message reports
an unknown outcome and does not replay tools. Inspect the recorded task IDs to
resolve it. Completed actions remain visible even if a later model request fails.
Cancelling a browser request or encountering a model error does not undo tasks
already submitted; use the existing task cancellation action if needed.

Conversations live in `chat_conversations` in the private backend schema, with
no browser database grants and no inclusion in fleet snapshots. Fleet tools retain
the existing fleet-admin scope; conversation ownership does not introduce task
tenant isolation. There is no automatic history cleanup yet; the prototype caps
each principal at 100 conversations. **New chat** starts a new conversation and
does not delete previous server history.

## Validation

```sh
uv run --project backend python -m unittest discover -s backend/tests -v
RUN_CHAT_E2E=1 uv run --project backend --python 3.12 --extra demo python -m unittest discover -s backend/tests/integration -p test_chat.py -v
npm --prefix frontend test
npm --prefix frontend run build
```

The integration test uses isolated local PostgreSQL and a fake model; it never
reads `.env` or configured Supabase/OpenAI credentials. A separate live GPT-6 Astra
smoke check verified workload discovery, worker discovery, one real queue submission,
and a task lookup in a follow-up against a temporary local database.
