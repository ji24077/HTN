# Development

Run all commands from the `orchestration-prototype/` project root.
The root contains package/build configuration, Compose, and the Dockerfile;
service code, documentation, tests, and example inputs have their own folders.

## Interactive demo

Use Python 3.12 and the optional bundled PostgreSQL dependency to run locally
without Docker. Install/build the frontend with Node 24.15+ first:

```sh
npm --prefix frontend ci
npm --prefix frontend run build
```

Then start each command in its own terminal:

```sh
uv run --python 3.12 --extra demo orchestrator-demo server
uv run --python 3.12 --extra demo orchestrator-demo worker-a
uv run --python 3.12 --extra demo orchestrator-demo worker-b
```

Open **http://127.0.0.1:8787**. Pick Worker A, Worker B, or automatic assignment;
name the task; choose a duration; and select **Send task**. **Run one on each**
queues a separate example on each process. Progress comes from worker heartbeats;
it is not a browser animation pretending to do work. The task stream exposes
cancellation and accepted result details.

The browser holds one authenticated Server-Sent Events connection (`/v1/updates`).
PostgreSQL `LISTEN/NOTIFY` wakes it after committed worker, task, or audit changes;
there is no periodic dashboard snapshot fetch. Reconnecting sends a fresh snapshot,
and bursts coalesce so a slow client does not accumulate an unbounded queue.
Cards and buttons stay mounted while changed values update. Submission and
cancellation use HTTP; worker traffic uses the separate bidirectional WebSocket.
The stream sends a keepalive comment after 15 quiet seconds. The server still
checks lease expiry every second, independently of the dashboard.

The launcher uses the same server, worker, PostgreSQL store, and WebSocket
protocol as the main prototype. It starts PostgreSQL locally through `pgserver`;
the optional Redis cache is disabled for this demo. Configuration and durable
data live in the ignored `.demo/` directory. Stop the three terminals with Ctrl-C.
Restarting the server reopens the existing database. No GPU or cloud resources
are provisioned. The launcher defaults to port 8787; pass the same `--port` to all
three commands to change it.

The dashboard is available only with `DEMO_UI=true` to loopback clients on
localhost/loopback hostnames. It uses a local HttpOnly, SameSite cookie and checks
the origin for browser API calls. Worker credentials stay in the server process.
Normal deployments continue to use bearer authentication and do not expose the
dashboard automatically.

## Frontend development

Run `npm --prefix frontend run dev` alongside the demo backend/workers and open
http://127.0.0.1:5173. Vite provides hot reload and proxies authenticated API/SSE
traffic to port 8787. The built UI remains available directly on port 8787.
See [the frontend guide](../frontend/README.md) for component layout, proxy
configuration, tests, and packaging.

## Container development

Requires Docker Compose for the services and `uv` with Python 3.12–3.14 for a
local worker. The Docker image uses Python 3.12. Dependencies are resolved in
`uv.lock`. The local Python demo has been exercised; the container path below
has not been built or validated yet.

From this directory:

```sh
cp .env.example .env
```

Replace the example admin and worker tokens with different random values of at
least 24 characters. Hex tokens are convenient for Compose interpolation. For
example, `openssl rand -hex 32` generates one token. Keep `WORKER_TOKENS` in sync
if running the server directly. Placeholder tokens are rejected at startup.
`.env` is ignored by Git.

When ready to build and start the local services:

```sh
docker compose up --build -d
```

Compose reads `.env` and enrolls `worker-1`. It exposes the API on loopback port
8080, PostgreSQL on 54329, and Redis on 63799. Credentials in the Compose file
are for this local prototype only. The database lives in a named volume.

In another terminal in this directory, start a worker using the same `.env`:

```sh
uv run --frozen --env-file .env orchestrator-worker
```

To run the server directly instead of in a container, first start only its
dependencies, then use the local URLs in `.env`:

```sh
docker compose up -d postgres redis
uv run --frozen --env-file .env orchestrator-server
```

The Python modules read environment variables. `uv --env-file` and Compose load
`.env`; importing the package opens no database or network connections. The
entry points can also be invoked as `python -m orchestrator.server` and
`python -m orchestrator.worker` in an installed environment.

To submit the fixture later:

```sh
export ADMIN_TOKEN='<your admin token>'
curl --fail-with-body http://127.0.0.1:8080/v1/tasks \
  -H "Authorization: Bearer $ADMIN_TOKEN" \
  -H 'Content-Type: application/json' \
  --data-binary @examples/tasks.json
```

## Checks and packaging

```sh
npm --prefix frontend ci
npm --prefix frontend test
npm --prefix frontend run build
uv sync --python 3.12 --extra demo
uv run python -m unittest discover -s tests -v
uvx ruff check src tests examples/agent_task.py --select F,I
uvx ruff format --check src tests examples/agent_task.py
uv build
```

`tests/unit/` contains isolated client and asset-routing tests. The detailed
validation record and remaining integration scenarios are in
[validation.md](validation.md). A live example is available in
[the agent interface guide](agent-interface.md).

SQL and compiled web assets live inside the Python package so wheel/container
installs include them. React source is in `frontend/`; Vite generates
`src/orchestrator/server/web/`. Build it before `uv build` or backend asset tests.
The generated directory is ignored by Git and explicitly included by Hatch in
release archives. Docker performs this build in its Node stage; Node is not
required in the runtime Python image.

## Where changes belong

- `frontend/`: React UI components, TypeScript API types, SSE hook, and UI tests.
- `shared/`: transport models, planner/executor contracts, and credential helpers.
  It must not import server, worker, or client implementations.
- `server/app.py`: application construction and resource startup/shutdown.
- `server/routes.py` and `server/auth.py`: client HTTP/SSE routes and authorization.
- `server/worker_connection.py`: worker WebSocket messages and assignment delivery.
- `server/db/`: durable state transitions and SQL schema. Scheduling policy is
  currently in `Store.claim`; lease-expiry scanning is in `server/scheduler.py`.
- `worker/`: outbound connection and execution lifecycle; add new executors under
  `worker/executors/` behind the shared `Executor` protocol.
- `client/`: typed HTTP client, agent tools, and CLI. Import its public API from
  `orchestrator.client`; the client does not import server or worker code.
- `demo.py`: local development launcher, which composes the real server and worker.

Server and worker each own their configuration. Both depend on `shared/`;
neither depends on the other or on the automation client. The demo is the
composition layer that launches them.

Tests mirror the component being tested. Put service-independent checks under
`tests/unit/`; add service-backed suites separately when those scenarios are
implemented. Keep illustrative inputs and runnable client examples in `examples/`.
