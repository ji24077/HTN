# Backend with embedded Tailscale

The website API and private worker gateway run on the same machine. One
`orchestrator-backend` command owns both processes and the bundled Tailscale
helper. No separate Tailscale app, daemon, or Serve configuration is required.
The backend joins the tailnet as `orch-backend`; each bundled worker joins as
its own node. Browsers use the website normally without Tailscale.

```text
Browser → website HTTPS proxy → public API :8080 ──→ Supabase
Worker  → encrypted Tailscale → private WSS :8443 → worker gateway :8081 ──→ Supabase
                               └──── same backend machine ────┘
```

The public API listens on loopback by default; the worker gateway always listens
on loopback. The helper exposes only `/v1/worker` and `/healthz` through its
tailnet TLS listener. It never publishes the website or management API to that
listener. No Funnel is used. Workers still require their enrolled tokens.

## Local development

Build the helper with Go 1.26.6+ and install the backend:

```sh
mkdir -p .local/bin
go -C transport/tailscale build -trimpath -o ../../.local/bin/orchestrator-tunnel .
uv sync --project backend --frozen
```

Keep the existing Supabase and local HTTPS website settings in the root `.env`.
Add these backend-only settings:

```dotenv
TAILSCALE_HELPER=.local/bin/orchestrator-tunnel
TAILSCALE_HOSTNAME=orch-backend
TAILSCALE_STATE_DIR=.local/tailscale/backend
TS_AUTHKEY=
```

Run the backend and frontend together with `bash scripts/start-app.sh`. To run
only the backend (for example when Vite is already running):

```sh
uv run --project backend --frozen --env-file .env orchestrator-backend
```

Stop separate `orchestrator-public` / `orchestrator-worker-gateway` processes
before switching; the bundle uses their same ports and configuration. Ctrl-C
stops the bundle's children. Losing either API process stops the bundle. A
Tailscale enrollment or connectivity failure leaves the local website running
and retries the tunnel, so local development can continue.

On first start, Tailscale prints a browser authorization link. Sign in to the
intended tailnet and approve the backend node if your policy requires it. Enable
MagicDNS and HTTPS certificates in the Tailscale admin DNS page. The helper
prints a `gateway_ready` event containing the worker `SERVER_URL` when its private
listener is ready. Certificate issuance and connectivity are verified separately
when a worker connects. An HTTPS certificate makes the node's certificate name
visible in public certificate transparency logs; it does not make the service
publicly reachable.

Set that URL in the worker's `.env.worker`, following the
[bundled worker guide](bundled-worker.md). Allow the worker identity to reach
the backend on TCP 8443 in tailnet policy. Keep the backend's identity directory
private and persistent; do not copy it into worker packages or Git. Enrollment
can also use a single-use `TS_AUTHKEY` supplied locally. Remove it after enrollment.

The Mac must remain awake and online for remote workers to connect. Moving to a
server later uses the same application, a server-specific node identity, and a
new worker URL. Local operation does not deploy or publish the website.

## Backend image

`deploy/Dockerfile` includes the helper and launches `orchestrator-backend` by
default. It sets `PUBLIC_BIND_HOST=0.0.0.0` for the container's public API;
`WORKER_PORT` stays on loopback. Publish only the public API behind an HTTPS
proxy and persist `/state` in a backend-specific volume. No TUN device,
privileged mode, or host Tailscale installation is needed. The local demo
Compose file explicitly retains its original combined server command.

For a future container deployment, override the local `TAILSCALE_HELPER` and
`TAILSCALE_STATE_DIR` paths with `/usr/local/bin/orchestrator-tunnel` and `/state`.
Mount the Supabase CA and update its path in `DATABASE_URL`; local absolute paths
will not exist inside the container. Do not bake `.env` or node identity into
the image.

Reference: [Tailscale embedded server API](https://tailscale.com/docs/reference/tsnet-server-api).
