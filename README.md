# HTN · GPU orchestration

Python control plane for dispatching independent tasks to unreliable worker
machines. Workers connect outbound over WebSockets; the dashboard receives
server-pushed updates. PostgreSQL owns leases and accepted results.

**Status:** working local prototype with two CPU stub workers, a dashboard,
a Python client, JSON CLI, and callable agent tools. Real GPU workloads, job
splitting, machine scoring, and remote fleet validation remain separate work.

## Run the app

The app uses Supabase authentication and PostgreSQL. Configure the root `.env`
from `.env.example`, including your public origin, Supabase project, database,
worker tokens, and approved admin emails or user IDs. Then:

```sh
npm --prefix frontend ci
./scripts/start-app.sh
```

This starts the public API, private worker gateway, and frontend. In this
workspace, open **https://localhost:5174** to create an account and sign in.
Approved email addresses receive fleet access only after email confirmation.
The dashboard shows registered workers and real database state.

See [frontend setup](frontend/README.md) for HTTPS and Supabase email redirects,
and [networking](docs/networking.md) for hosting the website and connecting
workers. The optional `orchestrator-demo` commands remain available for isolated
tests; the app launcher does not use them.

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
├── transport/tailscale/     # Embedded worker tunnel (Go / tsnet)
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

- [Automatic worker enrollment](docs/worker-enrollment.md): Supabase sign-in,
  backend-issued Tailscale keys, and the worker setup command.
- [Bundled backend](docs/bundled-backend.md): website API and private gateway on one host,
  with Tailscale included; supports local development.
- [Bundled worker](docs/bundled-worker.md): embedded Tailscale, enrollment, and worker image.
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
The app requires an approved admin email or user ID and a matching public origin.
