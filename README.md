# HTN · GPU orchestration

Python control plane for dispatching independent tasks to unreliable worker
machines. Workers connect outbound over WebSockets; the dashboard receives
server-pushed updates. PostgreSQL owns leases and accepted results.

**Status:** one Python/Supabase fleet control plane with a React dashboard, Python
workers, and paired desktop/iOS workers. The separate Relay console adds verified
RunPod execution, GPU offer scoring, owner lease policies, and approval-gated model
training, inference, verification, and migration. Commercial payments and broad
provider onboarding remain later-stage marketplace work; simulated catalog cards can
be planned against but are deliberately blocked from execution.

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

Set `OPENAI_API_KEY` and `OPENAI_MODEL=gpt-6-astra` in the backend environment to
enable the **Fleet assistant** chat panel. It can inspect the fleet, submit the
existing workloads, and inspect or cancel tasks. Conversations and tool activity
are saved privately on the backend. Restart the backend after changing configuration.

See [frontend setup](frontend/README.md) for HTTPS and Supabase email redirects,
and [networking](docs/networking.md) for hosting the website and connecting
workers. The optional `orchestrator-demo` commands remain available for isolated
tests; the app launcher does not use them.

## Add a machine

A machine joins the compute network by running one container:

```sh
docker run -d --restart unless-stopped -v dwp-agent-data:/data -p 127.0.0.1:43117:43117 \
  -e DWP_INVITE='https://your-control-service/join?code=CODE' dwp-agent:latest
```

Nothing is compiled per platform and nothing needs a port open: the agent dials out, so
a machine in another country joins the same way one in the next room does. The window at
`http://127.0.0.1:43117/` is the same desktop app as before — status, recent work, the
pause switch — served by the agent process itself.

Whoever you invite does not need this repository. The image is published to
`ghcr.io/<owner>/dwp-agent` by `.github/workflows/agent-image.yml`, and the `/join` page
their invite link points at gives them that command with their code already in it. See
[the agent as a container](docs/docker-agent.md) for publishing, several agents on one
host, optional ML workloads, and putting the container on a tailnet.

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
├── deploy/                  # Dockerfiles (backend, worker, agent), Compose, HTTPS proxy
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

## Relay · AI Compute Agent + GPU Marketplace

The integrated `gpushare` console is Relay's MVP control surface. A user can state
“Fine-tune Qwen2.5-0.5B under $20, deploy it, and keep quality unchanged,” compare
managed and community GPU cards, approve each external change, run an agent-off
baseline, chat with before/after models, and execute six specialized agents.

- **Training Optimization Agent** measures LoRA/QLoRA, batch/accumulation,
  checkpointing and GPU cost, and only publishes a candidate after the same
  held-out evaluation passes.
- **Inference Optimization Agent** measures eager/compiled runtimes, batching,
  decode KV caching, BF16/INT8/NF4 and route cost with fixed generated work.
- **Chip Migration Agent** transfers or resumes checkpoints across NVIDIA/AMD
  pods and refuses to update the active route when parse or exact-match quality
  regresses beyond the configured tolerance.
- **Job Allocation Agent** filters and ranks offers by VRAM, total cost, completion
  time, network, latency, trust, failure rate, privacy, region, and availability.
- **Verification Agent** produces pass/reject/rollback decisions from the same
  held-out hash, JSON validity, exact match, and required safety result.
- **Provider Agent** enforces owner schedules, minimum price, workload and data
  restrictions, region, runtime, and GPU-memory limits.

Spending, checkpoint migration, and live traffic switching are three independent
approvals. A connected RunPod is a verified executable offer; the four-chip MVP
catalog (RTX 4090, A5000, L40S, MI300X) is explicitly simulated and plan-only.
DiLoCo is selected only for cross-provider or slow-network training—not rendering
or inference. See [Relay architecture and API](docs/relay.md).

Configure the optional GPUShare fields in `.env`, then run:

```sh
make ui
```

Open **http://127.0.0.1:8080**. The original authenticated fleet UI remains at
**https://localhost:5174** when started with `./scripts/start-app.sh`.

## Guides

- [The agent as a container](docs/docker-agent.md): one image instead of five binaries,
  joining from an invite in the environment, several agents on one host, and the
  end-to-end fleet check.
- [Desktop and iOS integration](docs/jack-integration.md): device invites, real workloads,
  signed results, releases, Sentry, and validation.
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
- [Model client](docs/model-client.md): standalone OpenAI calls, separate from the agent loop.
- [Fleet assistant](docs/fleet-assistant.md): chat UI, agent loop, tools, and recovery.
- [Validation](docs/validation.md): completed checks and remaining failure scenarios.

## Checks

```sh
npm --prefix frontend test
npm --prefix frontend run build
pnpm install --frozen-lockfile
pnpm typecheck
pnpm test
pnpm check:gui
pnpm docker:build && pnpm docker:test -- --agents 4 --tasks 24
uv run --project backend python -m unittest discover -s backend/tests -v
uvx ruff check --config backend/pyproject.toml backend/src backend/tests examples/agent_task.py --select F,I
uv build --project backend
```

Your root `.env` holds local Supabase configuration and is ignored by Git.
`.demo/` holds local demo data; `.local/` holds local certificates and tooling.
The Python environment lives in `backend/.venv/`. All are ignored.
The app requires an approved admin email or user ID and a matching public origin.
