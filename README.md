# HTN · GPU orchestration

Python control plane for dispatching independent tasks to unreliable worker
machines. Workers connect outbound over WebSockets; the dashboard receives
server-pushed updates. PostgreSQL owns leases and accepted results.

**Status:** working local prototype with two CPU stub workers, a dashboard,
a Python client, JSON CLI, and callable agent tools. Real GPU workloads, job
splitting, machine scoring, and remote fleet validation remain separate work.

## Quick start

From the repository root, install and build the React frontend (Node 24.15+):

```sh
npm --prefix frontend ci
npm --prefix frontend run build
```

Then run each command in its own terminal:

```sh
uv run --project backend --python 3.12 --extra demo orchestrator-demo server
uv run --project backend --python 3.12 --extra demo orchestrator-demo worker-a
uv run --project backend --python 3.12 --extra demo orchestrator-demo worker-b
```

Open **http://127.0.0.1:8787** to select workers and dispatch tasks. The demo starts
local PostgreSQL and stores its credentials/data in the ignored `.demo/` folder.
No cloud or GPU resources are required.

For frontend development, run `npm --prefix frontend run dev` in another terminal
and open **http://127.0.0.1:5173** for hot reload. It uses the same backend and
workers. See [frontend setup](frontend/README.md).

For programmatic access:

```sh
uv run --project backend orchestrator tools
uv run --project backend orchestrator --demo workers
uv run --project backend python examples/agent_task.py --demo --task-id my-agent-task-001
```

## Project layout

```text
HTN/
├── frontend/                # React website, browser auth, and UI tests
├── backend/
│   ├── src/orchestrator/
│   │   ├── server/          # Public API, private worker gateway, auth, database
│   │   ├── worker/          # Worker agent and workload executors
│   │   ├── client/          # Python client, CLI, and agent tools
│   │   ├── shared/          # Protocol models and credential helpers
│   │   └── demo.py          # Local demo launcher
│   ├── tests/               # Unit and local PostgreSQL integration tests
│   ├── pyproject.toml       # Python dependencies and entry points
│   └── uv.lock
├── deploy/                  # Dockerfile, local Compose, public HTTPS proxy
├── docs/                    # Architecture, setup, API, and validation guides
├── examples/                # Runnable client and task fixtures
├── .env.example             # Supabase / public website configuration template
└── .env.local.example       # Local Docker development configuration template
```

`shared/` defines the contracts. Server, worker, and client implement their own
sides of those contracts. New workload executors belong under `worker/executors/`;
the future job planner can use the client API to submit its split tasks.

Hosted entry points are `orchestrator-public` and
`orchestrator-worker-gateway`. Other command names are stable: `orchestrator-server`, `orchestrator-worker`,
`orchestrator-demo`, and `orchestrator`. Module entry points are also available:
`python -m orchestrator.server`, `python -m orchestrator.worker`, and
`python -m orchestrator.client`.

## Guides

- [Public website and private workers](docs/networking.md): Supabase login/database,
  separate public and worker listeners, and Tailscale Serve setup.
- [Architecture and API](docs/architecture.md): leases, scheduling, protocols, and limits.
- [Frontend](frontend/README.md): React development, hot reload, and builds.
- [Development](docs/development.md): local setup, containers, checks, and contribution boundaries.
- [Agent interface](docs/agent-interface.md): client, CLI, tool schemas, and examples.
- [Validation](docs/validation.md): completed checks and remaining failure scenarios.

## Checks

```sh
npm --prefix frontend test
npm --prefix frontend run build
uv run --project backend python -m unittest discover -s backend/tests -v
uvx ruff check --config backend/pyproject.toml backend/src backend/tests examples/agent_task.py --select F,I
uv build --project backend
```

Your root `.env` holds local Supabase configuration and is ignored by Git.
`.demo/` holds local demo data; `.local/` holds local certificates and tooling.
The Python environment lives in `backend/.venv/`. All are ignored.
The local demo works without the unfinished Supabase admin ID or public domain.
