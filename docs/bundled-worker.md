# Worker with embedded Tailscale

The worker launches a small Go helper built with Tailscale's `tsnet` library.
Each worker joins the tailnet as its own node. Users do not install the Tailscale
desktop app, run `tailscaled`, grant network-extension permissions, or expose a
worker port. The website still works in an ordinary browser.

```text
Python worker → local TCP socket → bundled tsnet → private gateway → PostgreSQL
                └──────── verified WSS connection ──────────┘
```

The helper carries opaque TLS bytes to one configured `.ts.net` destination.
The Python worker verifies that destination's TLS certificate and sends its
worker token inside the encrypted connection. The helper does not receive the
worker token, database URL, or Supabase credentials in its environment. It is
not a general-purpose proxy and does not change host routes. Other local
processes can reach the loopback socket, but still need gateway credentials.

## Server prerequisite

For automatic sign-in and configuration, use the
[worker enrollment command](worker-enrollment.md). Manual enrollment below
remains available for existing development workers.

The private gateway must already have a reachable Tailscale HTTPS address.
Run the [bundled backend](bundled-backend.md) on the same machine as the website
API; it includes Tailscale and exposes the worker-only listener privately on
port 8443. Bundling the worker does not create or deploy this server. Allow the worker's identity to
reach only this gateway/port in your tailnet policy.

## Run locally (macOS or Linux)

Build the helper once using Go 1.26.6 or newer, then install Python dependencies:

```sh
mkdir -p .local/bin
go -C transport/tailscale build -trimpath -o ../../.local/bin/orchestrator-tunnel .
uv sync --project backend --frozen
cp .env.worker.example .env.worker
chmod 600 .env.worker
```

Edit the ignored `.env.worker`: supply the actual gateway URL, unique worker ID,
and that worker's enrolled token. Set a unique `TAILSCALE_HOSTNAME`. The template
already selects the built helper and an ignored persistent state directory.

```sh
uv run --project backend --frozen --env-file .env.worker orchestrator-worker
```

With `TS_AUTHKEY` blank, the first start prints a Tailscale sign-in URL in the
terminal. Open it and authorize this worker in the intended tailnet. If device
approval is enabled, an administrator must also approve the device. Startup
waits up to five minutes; restart if enrollment times out. Supabase login is
separate and does not enroll a worker into Tailscale.

For unattended enrollment, set `TS_AUTHKEY` in this worker's local environment
file using a single-use auth key scoped to the worker's permitted tag. Remove
the key after successful enrollment. Do not bake it into an image or distribute
a shared reusable key. Node credentials persist in `TAILSCALE_STATE_DIR`, which
is restricted to its owner. Give each worker a separate directory; never commit
or clone this identity state. Tailscale connects to its coordination/relay
services and uses its standard diagnostic logging behavior.

The worker owns the helper's lifetime. Ctrl-C stops both; losing the parent
closes the helper's stdin and stops it. A helper failure stops the worker without
falling back to a direct connection. Ordinary gateway disconnects use the
worker's existing reconnect backoff.

## Bundled Linux container

The image contains Python, the worker, and the compiled helper:

```sh
docker build -f deploy/Dockerfile.worker -t htn-worker .
docker run --rm --init --name htn-worker \
  --env-file .env.worker \
  -e TAILSCALE_HELPER=/usr/local/bin/orchestrator-tunnel \
  -e TAILSCALE_STATE_DIR=/state \
  -v htn-worker-1-state:/state \
  htn-worker
```

No published ports, privileged mode, `/dev/net/tun`, or `NET_ADMIN` are needed.
Use a distinct persistent volume per worker. For interactive enrollment, read
the sign-in link in the container's logs. An image build does not enroll nodes.
The executor is still the existing CPU stub; this does not add GPU execution.

## Checks

```sh
go -C transport/tailscale test -race ./...
uv run --project backend python -m unittest discover -s backend/tests/unit -p test_tunnel.py -v
```

The tests cover fixed-destination forwarding, cancellation, helper cleanup,
credential isolation, and TLS hostname verification before sending worker
credentials. Real enrollment, tailnet policy, and remote worker dispatch must
also be verified against the configured server; local tests cannot prove them.

References: [tsnet](https://tailscale.com/docs/features/tsnet),
[Server API](https://tailscale.com/docs/reference/tsnet-server-api).
