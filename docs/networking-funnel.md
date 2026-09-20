# Two ways onto the network, and when to use each

Main carries both. They are not layers; pick one per deployment.

| | **Funnel** (`scripts/share.ts`) | **tsnet** (`transport/tailscale`) |
|---|---|---|
| What the server gets | one permanent public HTTPS name | a private tailnet address |
| What a worker installs | **nothing** | an embedded Tailscale node (~29 MB helper) |
| What a worker needs | a pairing code | an auth key + device approval |
| Tailscale devices used | **1** (the server) | one per worker, against the plan limit |
| Publicly reachable | yes, authenticated | no |
| Data path | via a Tailscale relay, unpublished bandwidth cap | direct WireGuard where NAT allows |

## Why Funnel is the default here

Every machine on this fleet joined by pasting one command — a Windows laptop, a Mac on
campus wifi, and an iPhone, all behind NAT, none of them owned by the person running the
server. That is the workload this project was built for, and it is the thing tsnet cannot
do without an account and an auth key per machine.

Being public is not the weaker position. The device protocol authenticates every
connection with a 120-second Ed25519 assertion signed by a key that never leaves the
machine, replay-protected by a `jti` cache, and every result carries its own signature.
Measured against the live server: a forged assertion carrying a real host's ID, correct
audience and a valid TTL is rejected with `bad-signature`. A stolen URL buys nothing.

## When to use tsnet instead

When nothing should be publicly reachable at all, and you control every machine — a
company fleet rather than borrowed laptops. It is also the better data path for bulk
transfer: direct WireGuard has no funnel bandwidth cap, which matters if you start
shipping large model artifacts rather than small task payloads.

## The trap that cost us an evening

A worker configured for one network will not appear on the other, and **nothing reports an
error** — it connects successfully to the server it was pointed at.

A friend's laptop showed `wss://orch-backend.tail1f6128.ts.net:8443/v1/worker` and was
"connected", while the fleet owner saw no new device. Both were true. `tail1f6128` and
`taile7f048` are different tailnets, and `/v1/worker` is the token-authenticated worker
protocol, not `/agent/connect`. The machine had joined a different person's orchestrator.

Before debugging a missing device, compare two things on the worker: the **tailnet id** in
its URL, and whether it ends in `/v1/worker` (token worker) or `/agent/connect` (signed
device). Those two facts identify which network it actually joined.

## Using Funnel

```sh
node scripts/share.ts --port 8080
```

It refuses to publish a port nothing is serving, waits until the public name genuinely
answers rather than claiming success early, and prints the join link. Stop sharing with
`tailscale funnel --https=443 off`.
