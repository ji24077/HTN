# Distributed Work Platform

Coordinate work across computers you trust, see precisely where each task runs, and
remotely operate an authorized browser as another task type.

**Current phase: connectivity works over the public internet.** A host pairs through a
public HTTPS address, connects over `wss://`, and returns signed results. Eleven network
scenarios and 25 adversarial input cases pass. See
[`docs/04-connectivity-gate-report.md`](docs/04-connectivity-gate-report.md).

## Share your machine

```bash
pnpm doctor         # check this machine is ready, and what to fix if not
pnpm db:up
pnpm share          # public address + an invite link, in one command
pnpm invite <name>  # another invite, without restarting
```

Full walkthrough: [`docs/05-running-it.md`](docs/05-running-it.md).

Send a friend the printed `…/join?code=…` link. It shows them the two commands to run.
Press Enter for another invite. See [`docs/03-connectivity.md`](docs/03-connectivity.md)
for a permanent address, which you want before relying on this.

## Watch what is happening

```bash
pnpm logs                 # live, as it happens
pnpm logs --story         # per-computer summary: joins, drops, refusals
pnpm logs --level warn    # only problems
pnpm logs --since 30m --no-follow
```

## Test it

```bash
pnpm sim                  # 11 network scenarios against real processes
pnpm sim --list           # what each covers and why
pnpm sim:edge             # adversarial input: forged tokens, junk frames, bad clocks
node --env-file-if-exists=.env scripts/gate-security.ts
```

## Status

- [x] **Pass 0** — Architecture review · [`docs/01-architecture.md`](docs/01-architecture.md)
- [x] **Pass 1** — Network spike · pairing, transport, echo dispatch, leases, recovery,
      security gate, browser egress
- [x] **Connectivity** — public address, join flow, structured logging, network simulator.
      **Open:** a permanent address, and a real second computer
- [ ] **Pass 2** — Durable batch CPU inference + measured baseline
- [ ] **Pass 3** — Trusted-host remote browser session
- [ ] **Pass 4** — Dashboard
- [ ] **Pass 5** — Hardening and demo rehearsal

## Run it

Requires Node 24+, pnpm 10, Docker.

```bash
pnpm install
cp .env.example .env          # edit BOOTSTRAP_PASSWORD before sharing anything
pnpm db:up                    # Postgres 18 in Docker on :5433
pnpm control                  # control service on :8787, migrates and seeds on boot
```

There is no build step: Node 24 strips types directly, including across workspace links.
`pnpm typecheck` runs TypeScript as a checker only.

### Give it a public address

Agents dial out to whatever `PUBLIC_ORIGIN` says, so it must be reachable from the host's
network. For a spike:

```bash
pnpm tunnel                                        # prints https://<name>.trycloudflare.com
PUBLIC_ORIGIN=https://<name>.trycloudflare.com pnpm control
```

A quick tunnel is fine for Pass 1 and must not become the deployment — see architecture §14.

### Enroll a computer

```bash
# On the machine running the dashboard, get a code (10 min, single use):
curl -s -c /tmp/c -X POST $ORIGIN/auth/login -H 'content-type: application/json' \
  -d '{"email":"you@example.com","password":"change-me"}'
curl -s -b /tmp/c -X POST $ORIGIN/hosts/pair-code -H 'content-type: application/json' \
  -d '{"label":"studio-mac"}'

# On the computer being enrolled:
pnpm agent pair --server $ORIGIN --code ABCD-1234
pnpm agent run
```

The agent generates an Ed25519 keypair locally at `~/.dwp/agent.key` (mode 600) and
refuses to start if the permissions are wider. The private half never leaves the machine.

`DWP_HOME=/tmp/agent-b` runs a second agent on the same computer for local testing —
useful, but it is not the cross-network proof.

### Run work

```bash
curl -s -b /tmp/c -X POST $ORIGIN/jobs -H 'content-type: application/json' \
  -d '{"adapter":"echo","mode":"each","count":3,"sleepMs":150}'
curl -s -b /tmp/c $ORIGIN/jobs/<jobId>
```

`mode: "each"` pins one task per online host — the shape that shows each host ran its own
work. `mode: "queue"` leaves tasks unpinned so any eligible host can claim them.

### Host controls

```bash
pnpm agent pause     # local kill switch; works with the control service unreachable
pnpm agent resume
pnpm agent status
```

## Gates

```bash
node --env-file-if-exists=.env scripts/gate-security.ts   # 7 rejection checks
node --env-file-if-exists=.env scripts/verify-run.ts <jobId>
pnpm agent browser-probe --url $ORIGIN/whoami
```

`verify-run` checks every accepted result's Ed25519 attestation against the enrolled host
public keys, reading the database directly. It ignores the control service's event log on
purpose — the control service writes that log, so it cannot be evidence about itself.

`browser-probe` launches a real Chromium in an ephemeral profile, asserts no TCP debugging
port is exposed, and reports the address the target site actually observed.

## Documents

| Doc | What it is |
| --- | --- |
| [`docs/00-handoff.md`](docs/00-handoff.md) | The original MVP / architecture brief |
| [`docs/01-architecture.md`](docs/01-architecture.md) | Pass 0: contracts, scheduling, threat model, gates, backlog, go/no-go |
| [`docs/02-pass1-gate-report.md`](docs/02-pass1-gate-report.md) | Pass 1: what passed, with evidence, and what did not |
| [`docs/03-connectivity.md`](docs/03-connectivity.md) | Getting friends connected: why a tunnel is required here, choosing a permanent address, watching a live connection |
| [`docs/04-connectivity-gate-report.md`](docs/04-connectivity-gate-report.md) | Connectivity pass: what was proven, the bugs found, what is still open |
| [`docs/05-running-it.md`](docs/05-running-it.md) | How to host a network, get a friend on, and debug it |

Review page: https://claude.ai/artifact/GrRXxq48Z62UmCtwswZo4h

## Non-goals

Untrusted public hosts · arbitrary code or shell execution · real logged-in accounts on
stranger hosts · mobile host control · cross-chip translation · marketplace payments ·
end-to-end media confidentiality.
