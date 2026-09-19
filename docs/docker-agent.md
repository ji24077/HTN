# The agent as a container

A machine joins the network by running one image. No toolchain, no per-platform binary,
no install script, and the same bytes whether the machine is a MacBook in this room or a
Linux box in another country.

```sh
docker run -d --name dwp-agent --restart unless-stopped \
  -v dwp-agent-data:/data \
  -p 127.0.0.1:43117:43117 \
  -e DWP_INVITE='https://your-control-service/join?code=ABCDEF' \
  ghcr.io/your-org/dwp-agent:latest
```

That is the whole thing. The container joins, connects, and starts accepting work. The
window is at `http://127.0.0.1:43117/` — open it and the agent tells you the rest.

## Why this replaced the compiled binaries

The agent used to ship as five cross-compiled executables plus two desktop app bundles,
built from one developer's machine with `bun build --compile`. That worked, and it cost:

| | Binaries | Image |
|---|---|---|
| Artefacts to build and sign | 5 binaries + 2 apps | 1 image |
| Machine-learning workload | impossible — native `.so`/`.dylib` files cannot live inside a single-file executable | `--target ml`, same as everything else |
| "What version is that machine on?" | the compile-time stamp, the release string, or the file on disk — three answers that drifted | the image tag |
| Updating | download, verify a signature, unpack over the install, reconcile dependencies, restart, and on Windows defer the parts held open by the running process | `docker compose pull && docker compose up -d` |
| A Windows machine that stops on battery | a Scheduled Task with three wrong defaults, corrected by hand-written XML | the restart policy |

None of the binary machinery was wrong. It was all load-bearing, and all of it exists to
work around the fact that the same program had to be five different artefacts. Making it
one artefact deletes the problem rather than solving it again.

The binary path is still there and still works. This is the recommended way to add a
machine, not the only one.

## What the container changes about the agent

Three behaviours differ inside a container, decided by `isContainer()` in
`packages/agent/src/runtime.ts` and overridable with `DWP_CONTAINER`:

**It does not update itself.** The image is the version. A self-update would write into
a layer that `--force-recreate` discards, so it would hold until the next recreate and
then silently revert — and in between, the agent would report a version no image
anywhere has. `applyUpdate` refuses, and the window says how to update instead.

**It does not register a login service.** There is no login. Whether the agent comes back
is your restart policy, and a systemd unit written inside a container is a file nothing
will ever read. The window's "start at login" row becomes a statement of that fact.

**It serves the window on `0.0.0.0:43117` and opens no browser.** Loopback inside a
container is the container's own, which no published port can reach. The path token is
still required, the Host header is still checked, and the port is still published to the
host's loopback by default — so the window is no more reachable than before.

Everything else is the same code doing the same thing: the same page, the same "Recent
work" panel, the same pause switch, the same leave-and-join-another-network flow, the
same signed results.

## Joining without a person

A container has nobody to paste an invite link into a window, so the link arrives in the
environment. Any one of these works:

```sh
-e DWP_INVITE='https://control.example.com/join?code=ABCDEF'   # the link, as sent
-e DWP_SERVER=https://control.example.com -e DWP_CODE=ABCDEF   # the two halves
-e DWP_INVITE_FILE=/run/secrets/dwp_invite                     # a Docker/Compose secret
```

Prefer the file form in anything shared. An invite is a credential, and `environment:`
puts it in `docker inspect` and in every log that dumps the environment.

An agent that already has a config in `/data` ignores all three, so bringing a stack up
twice does not try to re-enrol and burn a code that has already been spent. Invites
expire after ten minutes and work exactly once; a control service also allows ten per
owner per hour, which is worth knowing before you script forty containers at once.

If the control service is still starting — the ordinary case under Compose, where
`depends_on` can wait for a container but not for the server inside it — the agent
retries for about two minutes rather than exiting and taking the invite with it.

## The volume is the machine

`/data` holds this machine's Ed25519 private key, which network it joined, whether it is
paused, and everything it has run. **Without a volume mounted there, recreating the
container produces a different machine**: a new identity, a new invite needed, and an
empty history.

With a volume, a container can be destroyed and recreated freely — it comes back as the
same host, with the same id on the dashboard and its "Recent work" intact. The
end-to-end suite checks exactly this by killing a container mid-task and restarting it.

## Several agents on one host

```sh
DWP_INVITE_1=… DWP_INVITE_2=… docker compose -f deploy/compose.fleet.yaml up -d --build
```

Do **not** reach for `docker compose up --scale agent=4`. Replicas share the service's
named volume, so all four would load the same private key and claim the same host id.
The control service hands the connection to whoever arrived last, so the four would
supersede each other in an endless round, abandoning work at every handover — and the
dashboard would show one flapping machine rather than four healthy ones.

`compose.fleet.yaml` gives each agent its own service, its own volume and its own
published port, which makes that impossible rather than merely discouraged. Each needs
its own invite.

## Reaching the control service

The agent dials **out**. Nothing about a containerised agent needs a port open to the
internet, a fixed address, or a DNS name — which is what makes "someone in another
country builds this image and joins" work with no extra setup beyond the invite link.

If your control service is only reachable inside a tailnet — running on a laptop at
home, say, with no public address — put the container on the tailnet:

```sh
docker compose -f deploy/compose.agent.yaml -f deploy/compose.tailscale.yaml up -d
```

That overlay runs `tailscale/tailscale` as a sidecar and puts the agent in its network
namespace. The agent itself knows nothing about Tailscale: it dials the same `wss://`
URL, which now resolves inside the tailnet. The tunnel moving into Docker is the point —
no host `tailscaled`, no per-OS install, and a Windows box and a Mac now reach the
network by identical means. `NET_ADMIN` is held by the Tailscale container; the agent
stays unprivileged.

If the control service already has a public HTTPS address, you need none of this.

## Optional workloads

The default image runs `echo` and `walker_evolution`, which are pure JavaScript and need
nothing. ONNX inference is a separate build target, because it is ~85 MB that most
machines have no use for:

```sh
docker build -f deploy/Dockerfile.agent --target ml -t dwp-agent:ml .
```

The build prunes the platform binaries the image can never execute, so it pays for one
platform rather than three.

## Configuration

| Variable | Default | What it does |
|---|---|---|
| `DWP_INVITE` / `DWP_SERVER`+`DWP_CODE` / `DWP_INVITE_FILE` | — | how to join; ignored once `/data` holds a config |
| `DWP_LABEL` | the container id | what this machine is called on the dashboard |
| `DWP_HOME` | `/data` | where identity, pause state and history live |
| `DWP_CONTAINER` | detected | force the container behaviours on (`1`) or off (`0`) |
| `DWP_GUI_HOST` | `0.0.0.0` | what the window binds to |
| `DWP_GUI_PORT` | `43117` | the port it binds; pinned, not walked |
| `DWP_GUI_ALLOWED_HOSTS` | — | extra Host names the window answers to, comma separated |
| `DWP_IMAGE` | — | the tag, so the window can say which image it is |
| `DWP_NO_WINDOW` | — | never try to open a browser, container or not |

## Checking it works

```sh
pnpm docker:build
pnpm docker:test -- --agents 4 --tasks 24
```

`scripts/docker-fleet-test.ts` starts a control service on its own port and its own
database schema — it cannot disturb one already running — builds the image, starts N
containers with nothing but an invite each, and then checks the claims that matter:

- every container appears as its own machine, with its own host id
- work spreads across them and every task comes back
- the results are **right**: echo nonces come back unchanged, and the deterministic
  walker's fitness numbers are recomputed locally and compared exactly
- the lease the server issued, the `hostId` in the output and the signed attestation all
  name the same machine — the check that would catch two connections getting crossed
- each container's own window agrees with the server about how many runs it did
- pausing stops work and resuming restarts it
- killing a container mid-flight loses no tasks, and restarting it brings back the same
  machine with its history intact

Useful flags: `--agents N`, `--tasks M`, `--skip-build`, `--skip-chaos`, `--keep` (leave
everything running to poke at it).

Measured on one laptop: 8 containers, 300 tasks, all verified, 41 tasks/second.
