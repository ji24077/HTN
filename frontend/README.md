# Dispatch frontend

React + TypeScript + Vite. The UI talks to the existing FastAPI control plane;
workers keep their separate WebSocket connections.

Run commands below from the repository root. Use Node 24.15+ (Node 24 is used
in the container build), npm, and uv.

## Run the app

The normal app requires Supabase sign-in and uses the configured PostgreSQL
backend. From the repository root, configure `.env` using `.env.example`, install
frontend dependencies, then start the public API, private worker gateway, and
frontend together:

```sh
npm --prefix frontend ci
uv run --project backend python scripts/setup-local-https.py
./scripts/start-app.sh
```

For this workspace, open **https://localhost:5174**. `PUBLIC_ORIGIN` sets the
browser origin and Vite port. `FRONTEND_TLS_CERT_FILE` and `FRONTEND_TLS_KEY_FILE`
point to the locally trusted TLS certificate and private key; relative paths
resolve from the repository root. These settings are read only by Vite's Node
process. No database or worker credentials are bundled into the frontend.

For localhost HTTPS, set `PUBLIC_ORIGIN=https://localhost:5174`,
`FRONTEND_TLS_CERT_FILE=.local/tls/localhost.pem`, and
`FRONTEND_TLS_KEY_FILE=.local/tls/localhost-key.pem` in `.env`. The HTTPS helper
creates a localhost-only development CA and certificate, and trusts the CA in
the macOS user keychain. Other platforms need to trust `.local/tls/root.pem` in
their browser. Certificate files and keys stay in the ignored `.local/` directory.
Re-run the helper to renew the 90-day server certificate.

Vite proxies `/auth` and `/v1` to the public API on `PUBLIC_PORT` (8080 by default),
preserving the browser Host and Origin. `ORCHESTRATOR_BACKEND_URL` can override
the target. Workers connect to the separate gateway on `WORKER_PORT` (8081 by
default). The app shows only registered workers; it does not start sample workers.

The older isolated demo is optional and requires explicitly starting
`orchestrator-demo`. It is not used by `start-app.sh`.

## Supabase accounts

The public website supports email/password sign-in, account creation, confirmation
links, password reset, session restoration, and sign-out. It loads the project URL
and public key from `/auth/config`; no frontend environment variables are needed.
The app launcher uses the real authentication flow, including on localhost HTTPS.

In Supabase Authentication, enable the Email provider and account signups if you
want users to register. Set the Site URL to your `PUBLIC_ORIGIN`, and allow these
redirect URLs (replace the example origin):

- `https://fleet.example.com/`
- `https://fleet.example.com/?auth=recovery`

Email templates must preserve Supabase's confirmation URL and requested redirect.
Confirmation and recovery use the SDK's PKCE flow: open email links in the same
browser that requested them. The recovery screen accepts a new password before
checking fleet authorization. Supabase enforces the project's password policy;
the form also requires at least eight characters for new passwords.

Registration does not grant access to the shared fleet. Configure approved
addresses in `SUPABASE_ADMIN_EMAILS`, or approved Auth UUIDs in
`SUPABASE_ADMIN_IDS`, and restart the backend. Email approvals require a confirmed
email in Supabase's account record; editable profile metadata does not grant access. The UI shows denied
access errors from the server. Existing sessions refresh through Supabase and
exchange their access token for the backend's HttpOnly cookie used by SSE.

See the [Supabase password recovery reference](https://supabase.com/docs/reference/javascript/auth-resetpasswordforemail)
and [auth events reference](https://supabase.com/docs/reference/javascript/auth-onauthstatechange).

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
`backend/src/orchestrator/shared/protocol.py`; update both sides if the protocol changes.

## Checks

```sh
npm --prefix frontend run typecheck
npm --prefix frontend test
npm --prefix frontend run format:check
npm --prefix frontend run build
```

Tests cover Supabase sign-in, denied access, session restoration, account creation,
password recovery, invalid links, and form errors, as well as selection/focus/draft
preservation across snapshots, targeted submission, cancellation, details,
reconnect state, and mutation failures. Auth SDK calls are mocked; live email
delivery and project configuration must be verified against your Supabase project.

## Build and package

`npm run build` writes generated files into `backend/src/orchestrator/server/web/`,
clearing the previous build. Edit files in `frontend/src/`, not that directory.
The generated output is ignored by Git and explicitly included in Python wheels
and source distributions. A fresh checkout must build the frontend before
running the demo, backend asset tests, or `uv build --project backend`:

```sh
npm --prefix frontend ci
npm --prefix frontend run build
uv build --project backend
```

The Dockerfile builds the frontend in a Node stage, then copies the compiled
assets into the Python build. The runtime container does not need Node or npm.
Rebuild and reload port 8787 to see source changes there; port 5173 updates live.
