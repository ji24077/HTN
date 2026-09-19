# Dispatch frontend

React + TypeScript + Vite. The UI talks to the existing FastAPI control plane;
workers keep their separate WebSocket connections.

Run commands below from the prototype root. Use Node 24.15+ (Node 24 is used
in the container build), npm, and the Python demo environment.

## First run

```sh
npm --prefix frontend ci
npm --prefix frontend run build
uv run --python 3.12 --extra demo orchestrator-demo server
```

Start `orchestrator-demo worker-a` and `orchestrator-demo worker-b` in their own
terminals, using the same `uv run --python 3.12 --extra demo` prefix. The built
React UI is served at **http://127.0.0.1:8787**.

## Develop with hot reload

Keep the demo backend/workers running, then:

```sh
npm --prefix frontend run dev
```

Open **http://127.0.0.1:5173**. Vite proxies `/v1` and `/demo/session` to
`http://127.0.0.1:8787`. The proxy preserves the browser Host header so FastAPI's
same-origin checks still apply. The backend creates a local HttpOnly session
cookie; no admin token is stored in frontend code or Vite environment variables.
The development server binds only to loopback.

If using a different backend port, set `ORCHESTRATOR_BACKEND_URL` when starting
Vite. It is a Node-side proxy setting and is not bundled into browser code.
The demo endpoint remains loopback-only; this does not implement public user login.

## Layout

```text
src/
  api/           # Typed API contracts and mutation requests
  components/    # Worker picker, task composer/list/details, activity
  hooks/         # Session bootstrap and a single SSE subscription
  lib/           # Display helpers
  test/          # Interaction tests and browser test setup
  App.tsx        # Page composition and mutations
  main.tsx       # React entry point
  styles.css     # Dashboard styling
```

Selection and form drafts live separately from server snapshots. Stable React
keys preserve controls as progress arrives. `useFleet` opens SSE after session
bootstrap, cleans up on unmount, and reconnects on failures; it does not poll
snapshots. Task mutations use HTTP. The API contract types mirror
`src/orchestrator/shared/protocol.py`; update both sides if the protocol changes.

## Checks

```sh
npm --prefix frontend run typecheck
npm --prefix frontend test
npm --prefix frontend run format:check
npm --prefix frontend run build
```

Tests cover selection/focus/draft preservation across snapshots, targeted
submission, cancellation, details, reconnect state, and mutation failures.

## Build and package

`npm run build` writes generated files into `src/orchestrator/server/web/`,
clearing the previous build. Edit files in `frontend/src/`, not that directory.
The generated output is ignored by Git and explicitly included in Python wheels
and source distributions. A fresh checkout must build the frontend before
running the demo, backend asset tests, or `uv build`:

```sh
npm --prefix frontend ci
npm --prefix frontend run build
uv build
```

The Dockerfile builds the frontend in a Node stage, then copies the compiled
assets into the Python build. The runtime container does not need Node or npm.
Rebuild and reload port 8787 to see source changes there; port 5173 updates live.
