# Orchestration prototype

Python control plane for dispatching independent tasks to unreliable worker
machines. Workers connect outbound over WebSockets; the dashboard receives
server-pushed updates. PostgreSQL owns leases and accepted results.

**Status:** working local prototype with two CPU stub workers, a dashboard,
a Python client, JSON CLI, and callable agent tools. Real GPU workloads, job
splitting, machine scoring, and remote fleet validation remain separate work.

## Quick start

From this directory, install and build the React frontend (Node 24.15+):

```sh
npm --prefix frontend ci
npm --prefix frontend run build
```

Then run each command in its own terminal:

```sh
uv run --python 3.12 --extra demo orchestrator-demo server
uv run --python 3.12 --extra demo orchestrator-demo worker-a
uv run --python 3.12 --extra demo orchestrator-demo worker-b
```

Open **http://127.0.0.1:8787** to select workers and dispatch tasks. The demo starts
local PostgreSQL and stores its credentials/data in the ignored `.demo/` folder.
No cloud or GPU resources are required.

For frontend development, run `npm --prefix frontend run dev` in another terminal
and open **http://127.0.0.1:5173** for hot reload. It uses the same backend and
workers. See [frontend setup](frontend/README.md).

For programmatic access:

```sh
uv run orchestrator tools
uv run orchestrator --demo workers
uv run python examples/agent_task.py --demo --task-id my-agent-task-001
```

## Project layout

```text
orchestration-prototype/
├── src/orchestrator/
│   ├── shared/               # Models, protocol contracts, credential helpers
│   ├── server/
│   │   ├── app.py            # App assembly and service lifecycle
│   │   ├── routes.py         # Task, worker, audit, and SSE API
│   │   ├── auth.py           # Admin and local-demo authorization
│   │   ├── worker_connection.py
│   │   ├── scheduler.py      # Lease-expiry recovery loop
│   │   ├── updates.py        # PostgreSQL notification subscriptions
│   │   ├── config.py
│   │   ├── db/               # Store and packaged SQL schema
│   │   └── web/              # Generated React build, included in packages
│   ├── worker/
│   │   ├── agent.py          # Outbound connection and execution lifecycle
│   │   ├── config.py
│   │   └── executors/        # Workload implementations; stub for now
│   ├── client/               # Python API, agent tools, JSON CLI
│   └── demo.py               # Local service launcher
├── frontend/                # React + TypeScript + Vite source and UI tests
├── tests/unit/               # Isolated tests, no services required
├── examples/                 # Runnable client example and task fixture
├── docs/                     # Architecture, development, usage, validation
├── pyproject.toml            # Package, commands, dependencies, tool settings
├── uv.lock
├── Dockerfile
├── compose.yaml
└── .env.example
```

`shared/` defines the contracts. Server, worker, and client implement their own
sides of those contracts. New workload executors belong under `worker/executors/`;
the future job planner can use the client API to submit its split tasks.

The command names are stable: `orchestrator-server`, `orchestrator-worker`,
`orchestrator-demo`, and `orchestrator`. Module entry points are also available:
`python -m orchestrator.server`, `python -m orchestrator.worker`, and
`python -m orchestrator.client`.

## Guides

- [Architecture and API](docs/architecture.md): leases, scheduling, protocols, and limits.
- [Frontend](frontend/README.md): React development, hot reload, and builds.
- [Development](docs/development.md): local setup, containers, checks, and contribution boundaries.
- [Agent interface](docs/agent-interface.md): client, CLI, tool schemas, and examples.
- [Validation](docs/validation.md): completed checks and remaining failure scenarios.

## Checks

```sh
npm --prefix frontend test
npm --prefix frontend run build
uv run python -m unittest discover -s tests -v
uvx ruff check src tests examples/agent_task.py --select F,I
uv build
```
