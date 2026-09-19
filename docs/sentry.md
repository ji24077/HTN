# Sentry

Error, trace, log, and replay reporting is optional. A blank `SENTRY_DSN`
disables it and nothing is sent.

| Component | Sentry project | Setting |
| --- | --- | --- |
| Control plane (`orchestrator-public`, `-worker-gateway`, `-server`) | `htn-backend` | `SENTRY_DSN` in `.env` |
| Worker (`orchestrator-worker`) | `htn-backend` | `SENTRY_DSN` in `.env.worker` |
| Dashboard | `htn-frontend` | `SENTRY_FRONTEND_DSN` in `.env` |

Backend and worker events share a project and are told apart by the
`component` tag (`server` or `worker`), plus `surface` or `worker_id`. The
dashboard reads its DSN from `GET /telemetry/config` at runtime, so one built
bundle works in every deployment. Set `SENTRY_ENVIRONMENT` and optionally
`SENTRY_RELEASE` on each process.

The Go Tailscale transport is not instrumented; tunnel failures surface as
worker connection errors.

## What is sent

Everything useful for debugging: 100% of traces and profiles, request bodies,
task payloads, stack-frame variables, application logs, and a session replay
whenever the dashboard hits an error.

Credentials are removed before anything leaves the process
(`backend/src/orchestrator/shared/telemetry.py`, `frontend/src/telemetry.ts`):

- values under keys such as `authorization`, `cookie`, `token`, `secret`,
  `password`, and `api_key`;
- bearer headers, JWTs, Tailscale and Supabase keys, and URL passwords found
  inside free text;
- the exact value of every configured credential (`ADMIN_TOKEN`,
  `WORKER_TOKENS`, `WORKER_TOKEN`, OAuth secrets, database passwords), wherever
  it appears, including inside an object `repr`.

Replays mask all input fields. Worker tokens issued at runtime by browser
enrollment are not in the environment, so they are covered by the key and
pattern rules only. Treat user-submitted task payloads as visible to everyone
in the Sentry organization.

## Source maps

The dashboard build emits hidden source maps. Upload them after each build so
browser stack traces point at the TypeScript source:

```sh
npm --prefix frontend run build
SENTRY_ORG=phineas-truong SENTRY_PROJECT=htn-frontend \
  sentry sourcemap upload backend/src/orchestrator/server/web/assets
```

## Reading Sentry from an agent

`.mcp.json` registers Sentry's hosted MCP server for this organization. Claude
Code prompts to approve it on first open; run `/mcp` and sign in once. It holds
no token, so every teammate authenticates as themselves.

The `sentry` CLI reads the same data from a terminal and prints JSON with
`--json`:

```sh
npm install -g sentry && sentry auth login
sentry issue list phineas-truong/htn-backend
sentry issue view HTN-BACKEND-1
sentry trace view phineas-truong/<trace-id>
sentry log list phineas-truong/htn-backend
sentry replay list phineas-truong/htn-frontend
```

Use `sentry auth login --read-only` for an agent that should not change
Sentry state.

## Checking it works

```sh
export SENTRY_DSN=... SENTRY_FRONTEND_DSN=... SENTRY_ENVIRONMENT=demo
uv run --project backend --python 3.12 --extra demo orchestrator-demo server
uv run --project backend --python 3.12 --extra demo orchestrator-demo worker-a
```

Open the dashboard and submit a stub task whose payload includes
`"fail": true`. A `ValueError: requested stub failure` issue, a
`task.execute` trace, and the worker's logs appear in `htn-backend` within
seconds.
