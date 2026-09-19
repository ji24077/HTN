# Agent and automation interface

Agents can use the control plane without interacting with the dashboard. The
Python client and CLI use the same authenticated HTTP API and SSE update stream.
Worker connections remain WebSockets. No model provider or agent framework is
required, and these commands do not provision infrastructure.

## Local demo commands

Run these from the repository root with the demo server and workers running:

```sh
uv run --project backend orchestrator tools
uv run --project backend orchestrator --demo workers
uv run --project backend orchestrator --demo submit examples/tasks.json
uv run --project backend orchestrator --demo task demo-frame-0001
uv run --project backend orchestrator --demo wait demo-frame-0001 --timeout 120
uv run --project backend orchestrator --demo events --after 0
```

To cancel explicitly:

```sh
uv run --project backend orchestrator --demo cancel demo-frame-0001
```

`--demo` reads the admin credential from `.demo/config.json` and defaults to
`http://127.0.0.1:8787`. It refuses to send that credential to a non-loopback host.
For another deployment, set `ORCHESTRATOR_URL` and `ORCHESTRATOR_TOKEN` in the
host environment and omit `--demo`. `ADMIN_TOKEN` is also accepted. HTTP is
allowed only on loopback; use HTTPS elsewhere. Redirects are not followed.
Credentials are never model tool arguments or CLI command-line flags.

Commands emit one JSON envelope on stdout:

```json
{"ok": true, "result": {"state": "succeeded"}}
```

The result above is abbreviated; task responses include the spec, assignment,
progress, accepted result, and failure. Errors use
`{"ok":false,"error":{"code":"...","message":"..."}}` and a nonzero exit
code. `ok` means the API/tool operation succeeded: `wait` can successfully return
a task whose state is `failed` or `cancelled`. Inspect `result.state` to decide
whether the job succeeded. Waiting has a maximum one-hour deadline. Timing out
or interrupting the client does **not** cancel a task.

## Callable agent tools

`orchestrator.client.tool_definitions()` returns JSON Schema definitions
with `name`, `description`, and `input_schema`. `AgentTools.call(name, arguments)`
validates inputs and dispatches only the seven listed operations:

| Tool | Arguments | Purpose |
| --- | --- | --- |
| `list_workers` | `{}` | State and reported capabilities, up to 500 workers |
| `list_tasks` | `{}` | Newest 500 tasks |
| `get_task` | `task_id` | One task, progress, result, and failure |
| `submit_tasks` | `tasks: [TaskSpec, ...]` | Atomically submit 1–100 already-split tasks |
| `cancel_task` | `task_id` | Cancel a task |
| `wait_task` | `task_id`, optional `timeout_seconds` | Await a terminal state over SSE |
| `list_events` | Optional `after` event ID | Read audit history, up to 500 events |

A host can bind these functions to its agent framework or an MCP adapter. This
prototype provides the callable dispatcher and schemas, not an MCP server or an
autonomous agent loop. The host supplies credentials and decides which tools its
agent may invoke. The prototype still uses an admin credential, not scoped roles.

```python
import os
from orchestrator.client import OrchestratorClient
from orchestrator.client import AgentTools, tool_definitions

definitions = tool_definitions()  # Give these schemas to your tool adapter.

async def handle_agent_call(name: str, arguments: dict):
    async with OrchestratorClient(
        os.environ["ORCHESTRATOR_URL"], os.environ["ORCHESTRATOR_TOKEN"]
    ) as client:
        return await AgentTools(client).call(name, arguments)
```

The CLI also dispatches tool calls directly from a file or stdin:

```sh
uv run --project backend orchestrator --demo call get_task <<'JSON'
{"task_id":"demo-frame-0001"}
JSON
```

For typed Python code, `OrchestratorClient` exposes the same seven methods;
submission accepts `list[TaskSpec]`, and worker/task methods return Pydantic
models. `updates()` is an async iterator over pushed fleet snapshots. The client
reconnects after transient stream failures and gets current state; it does not
promise delivery of every intermediate transition. Use `list_events` with its
cursor for audit history. `wait_task` uses SSE; if its task falls outside the
500-task snapshot window, it reads that task on change notifications.

## Example agent workflow

An executable example discovers workers, submits a 15-second stub to Worker A,
and waits for its result:

```sh
uv run --project backend python examples/agent_task.py --demo --task-id my-agent-task-001
```

Use a stable task ID for retries. Repeating the same ID/spec returns the existing
task; changing the spec under that ID produces `conflict` (HTTP 409). If a submit
request times out, it may already have committed: inspect or resubmit the same
ID/spec. The client never silently retries a write. Choose a new ID only when
you intend to create new work.

An agent may choose `target_worker_id` when submitting, or leave it null for
automatic placement. Runtime, executor kind, available VRAM, and one-slot
capacity still apply. `allow_failover=true` permits another compatible worker
on a retry; false keeps attempts pinned. The server owns leases and retries.
Tasks still need to be split by the caller, and the only executor currently
available is the CPU stub. Real job splitting, GPU execution, and scoring remain
separate work.

Client boundary tests run without a server:

```sh
uv run --project backend python -m unittest discover -s backend/tests -v
```
