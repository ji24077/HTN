# Sentry

Error, trace, log, and replay reporting is optional. A blank `SENTRY_DSN`
disables it and nothing is sent.

| Component | Sentry project | Setting |
| --- | --- | --- |
| Control plane (`orchestrator-public`, `-worker-gateway`, `-server`) | `htn-backend` | `SENTRY_DSN` in `.env` |
| Worker (`orchestrator-worker`) | `htn-backend` | `SENTRY_DSN` in `.env.worker` |
| Dashboard | `htn-frontend` | `SENTRY_FRONTEND_DSN` in `.env` |
| Installed desktop runner | Backend project by default | Automatically provisioned during pairing and refreshed on reconnect |

## Automatic runner configuration

For successful-run history, task output, and repair-agent diagnostics, see
[execution tracking](execution-tracking.md). Those records live in the task API
and dashboard independently of Sentry; exception events link through `execution_id`.

Set `SENTRY_DSN` on the platform and restart it. Desktop pairing sends the public
ingest DSN, environment, and release to the runner, which saves them in its device
profile. The same startup code reads that profile for Node, desktop apps, compiled
binaries, and installed login services. No `.env` file or Sentry API token is needed
on the runner. A device's Sentry errors still contain worker/task tags and pass
through the existing credential scrubber; ordinary desktop connection logs remain
local JSONL logs, not Sentry Logs.

An updated desktop runner that was already paired acquires these settings on its
next connection. Later reconnects also refresh changed settings or disable reporting
when the platform sends an empty DSN. Cached settings cover startup errors before
the connection is established. Older servers that omit telemetry remain compatible;
switching a device to a different server clears its cached telemetry configuration.
Changing platform settings requires restarting the platform; connected desktop
runners receive the change when they reconnect.

`SENTRY_WORKER_DSN` on the platform optionally selects a separate worker project.
Leaving that variable unset inherits `SENTRY_DSN`; setting it explicitly empty
disables managed worker reporting. Explicit `SENTRY_DSN`, `SENTRY_ENVIRONMENT`, and
`SENTRY_RELEASE` values in a desktop runner's environment override the managed
values. An explicitly empty runner `SENTRY_DSN` is a local opt-out.

Python automatic enrollment writes the same public settings to its generated worker
environment file alongside the Tailscale configuration. Existing Python profiles
need their Sentry settings updated manually; their private worker protocol does not
refresh configuration. iOS continues to report device failures through the backend;
this does not add a native Swift Sentry SDK.

Distribution still requires shipping an updated runner build: binaries already
installed before this feature cannot load profile telemetry until updated. No
release is automatically signed or published by this source change.

Backend and worker events share a project and are told apart by the
`component` tag (`server` or `worker`), plus `surface` or `worker_id`. The
dashboard reads its DSN from `GET /telemetry/config` at runtime, so one built
bundle works in every deployment. Set `SENTRY_ENVIRONMENT` and optionally
`SENTRY_RELEASE` on each process.

The Go Tailscale transport is not instrumented; tunnel failures surface as
worker connection errors.

## What is sent

Everything useful for debugging: 100% of traces and profiles, request bodies,
task payloads, stack traces, application logs, and a session replay whenever
the dashboard hits an error. Stack-frame locals are omitted because model
representations can hide sensitive field boundaries before scrubbing runs.

Credentials are removed before anything leaves the process
(`backend/src/orchestrator/shared/telemetry.py`, `frontend/src/telemetry.ts`):

- values under keys such as `authorization`, `cookie`, `token`, `secret`,
  `password`, and `api_key`;
- bearer headers, JWTs, Tailscale and Supabase keys, and URL passwords found
  inside free text;
- the exact value of every configured credential (`ADMIN_TOKEN`,
  `WORKER_TOKENS`, `WORKER_TOKEN`, OAuth secrets, database passwords), wherever
  it appears, including inside an object `repr`.

Replays mask all input fields. Auth callback page loads containing codes or
tokens are excluded from replay, even if authentication removes the parameters
during telemetry startup; replay resumes on the next ordinary page load.
Replay metadata and custom recording events also pass through the scrubber.
Worker tokens issued at runtime by browser
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
