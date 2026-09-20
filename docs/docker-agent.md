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

## Putting it on your other machines

Three things have to line up, and `node scripts/add-machine.ts` checks all three and
prints the command to paste:

```
  Control service  https://your-machine.tailnet.ts.net
  Reachable from   anywhere on the internet
  Image            ghcr.io/<owner>/dwp-agent:latest

  Machine 1 of 1. Paste this on that computer:

      docker run -d --name dwp-agent --restart unless-stopped -v dwp-agent-data:/data …
```

**The address has to be one that other machine can reach.** This is the mistake that
costs the most time, because a `127.0.0.1` URL is perfectly valid — for a completely
different computer. `add-machine` refuses to hand one out, and says which of the two
fixes you want:

```sh
LISTEN_HOST=0.0.0.0 ./scripts/start-fleet.sh --replace   # machines on this network
node scripts/share.ts --port 8080                        # machines anywhere
```

`share.ts` puts the control service on a permanent public HTTPS name with Tailscale
Funnel. Only the machine running the control service needs Tailscale; the machines
joining need nothing but Docker and working wifi, because they dial out over ordinary
WSS.

**A proxy in front means the server must be told.** Funnel terminates TLS and forwards
plain HTTP to loopback, so without `TRUSTED_PROXY_IPS` the server believes it is serving
`http://` and hands every joining agent a `ws://host/agent/connect` URL — port 80, where
nothing is listening. Pairing succeeds and the connection that follows is refused
forever, which reads as a broken agent rather than a mis-derived URL. `start-fleet.sh`
now sets `TRUSTED_PROXY_IPS=127.0.0.1`, which is safe: a machine on the LAN connects from
its own address and cannot spoof those headers, and anything already on this host can
read the admin token anyway.

**The invite expires.** Ten minutes, one use, ten per hour per owner. Mint several at
once with `--count 4`.

## How another machine gets the image

Someone joining does not have this repository and should not need it. They need two
things: the image, and an invite.

**The image is published to a registry.** `.github/workflows/agent-image.yml` builds it
for `linux/amd64` and `linux/arm64` on every push to `main` and pushes it to
`ghcr.io/<owner>/dwp-agent`. Both architectures are required, not a nicety: a
single-arch image does not fail helpfully on the wrong CPU — Docker Desktop runs it
under emulation at a fraction of the speed, which reads as "that machine is slow"
rather than "that image is for a different processor".

To publish by hand, or from a fork:

```sh
echo <a GitHub token with write:packages> | docker login ghcr.io -u <you> --password-stdin
node scripts/publish-image.ts                 # builds both arches, pushes, prints the digest
node scripts/publish-image.ts --dry-run       # build both, push nothing
```

It works out the registry from the repository's git remote, so a fork publishes to its
own namespace rather than to someone else's. Three tags go out each time: `latest` for
the docs, the version for an operator who wants to pin, and the commit SHA so a running
container can be traced back to a line of code.

**The invite carries the command.** The `/join?code=…` page every invite link points at
now leads with the Docker command, with that invite already substituted in, and a copy
button. So the whole of "how do I add my machine" is: open the link you were sent, copy
one line, paste it into a terminal. The code is filled in by the page's own script from
its address bar rather than rendered into the HTML, so it stays out of server logs and
proxy caches. The command is deliberately one long line — backslash continuations are
POSIX shell syntax and would break every line after the first when pasted into
PowerShell, which is exactly where a Windows contributor will paste it.

Set `DWP_AGENT_IMAGE` on the control service if you publish somewhere other than the
default, so that page names your image rather than the default one.

**With no registry at all**, for a machine you can copy a file to:

```sh
docker save dwp-agent:latest | gzip -1 > dwp-agent.tgz     # ~89 MB
# copy it across by whatever means, then on that machine:
gunzip -c dwp-agent.tgz | docker load
```

Then run the command `add-machine` printed, with `dwp-agent:latest` as the image. This
works offline and needs no accounts, and it is per-machine and per-update — which is
exactly the cost a registry removes.

**For someone who prefers a compose file**, `deploy/compose.agent.remote.yaml` pulls the
published image and needs nothing else from this repository:

```sh
curl -O https://raw.githubusercontent.com/ji24077/HTN/main/deploy/compose.agent.remote.yaml
DWP_INVITE='https://your-control-service/join?code=CODE' \
  docker compose -f compose.agent.remote.yaml up -d
```

## What the rest of the fleet sees

Nothing about a containerised machine looks different from anywhere else, which is the
point — the container is a packaging decision, not a protocol one.

- **On the fleet dashboard**, it is an ordinary worker: its label, the adapters it can
  run, its CPU and RAM, whether it is alive, and every task it has been given. The
  scheduler does not know or care that it is a container.
- **On the machine itself**, the window at `http://127.0.0.1:43117/` shows the same app
  as a laptop install: connection status, what is running right now, the "Recent work"
  panel with its timeline and per-adapter totals, and the pause switch. It adds one row
  — `Runs in`, naming the image and container — and swaps the login-service and update
  rows for what is true in a container.
- **From another machine**, nothing. The window is published to the host's loopback by
  default and gated by a path token. Add a name to `DWP_GUI_ALLOWED_HOSTS` and publish
  the port more widely only if you actually want that.

Results are signed by a key that never leaves `/data`, so a container proves which
machine ran a task in exactly the way a laptop does.

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

### Python and PyTorch on CPU

The standard image includes Python 3.12, CPU PyTorch 2.13.0, and the uploaded-project
executor. It reports Python/PyTorch versions after a real CPU tensor check. Blender
is outside the current scope. GPU-enabled Python workers are described below.
The PyTorch wheel comes from the [official CPU index](https://download.pytorch.org/whl/cpu).

```sh
docker build -f deploy/Dockerfile.agent -t dwp-agent:python-cpu .
```

Upload the three scripts in [the PyTorch example](../examples/projects/pytorch/README.md)
and request 200 training steps, a 2-step probe, and checkpoint validation. The agent
selects dependencies from source and uploaded manifests. The worker installs extra
Python libraries in a private environment, preserving the image's exact PyTorch
version. Incompatible version pins fail setup instead of downloading a CUDA build.

The paired agent retains identity, leases, cancellation, execution logs and signed
results. Artifacts use authenticated HTTP transfers rather than the result socket.
One task runs per device; multiple containers require separate identity volumes.

The opt-in integration test starts a temporary paired Docker container and database,
trains and validates a real model, and verifies signed results and output downloads:

```sh
RUN_PYTHON_AGENT_E2E=1 PYTHONPATH=backend/tests \
  uv run --project backend --python 3.12 --extra demo \
  python -m unittest integration.test_python_agent -v
```

### Ordinary fleet containers

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

## Updates, and how they are gated

**No device ever rebuilds anything.** One build produces one multi-architecture image;
every machine pulls the same bytes and runs them. That is the difference from the binary
story, where a release meant five compiles, two app bundles, and a signed manifest, and
"is that machine on the new version" was a question with three possible answers.

Nor is there a build per architecture *for the operator*: `linux/amd64` and `linux/arm64`
go out under one tag, and Docker picks the right one on each machine. A contributor on an
Apple laptop and one on a cloud VM run `docker pull` on the same name.

**A containerised agent never updates itself.** This is deliberate (see above), and it is
also what makes updates gateable: nothing changes on a machine until someone pulls. The
gate is which reference the machine is pointed at.

| You point a machine at | It moves when | Use it for |
|---|---|---|
| `dwp-agent:latest` with `pull_policy: always` | every `docker compose up -d` | a lab, or your own machines |
| `dwp-agent:0.4.0` | you edit the file | a fleet you want to move deliberately |
| `dwp-agent@sha256:…` | you edit the file | production, and anything you need to be certain about |

`node scripts/publish-image.ts` prints the digest for exactly this reason. A tag can be
moved by whoever can push to the registry; a digest names specific bytes and cannot.
Pinning by digest is the container equivalent of the pinned release-signing key the
binary path used — the guarantee is "these exact bytes", arrived at differently.

For a staged rollout, publish a second tag and point some machines at it:

```sh
docker buildx imagetools create -t ghcr.io/<owner>/dwp-agent:canary ghcr.io/<owner>/dwp-agent:latest
# a week later, if nothing broke
docker buildx imagetools create -t ghcr.io/<owner>/dwp-agent:stable ghcr.io/<owner>/dwp-agent:canary
```

That retags without rebuilding, so `stable` is provably the same bytes that ran as
`canary`.

**Seeing what each machine is on.** Every agent reports its image as its version, so the
fleet dashboard answers "who is still on the old one" directly:

```sh
curl -sH "authorization: Bearer $ADMIN_TOKEN" $SERVER/v1/workers \
  | python3 -c 'import sys,json;[print(w["id"][:8], w["capabilities"]["machine"]["agent_version"]) for w in json.load(sys.stdin)]'
```

This is why `DWP_IMAGE` is set in the compose files. Docker does not tell a container
what image it came from, so without it the agent has nothing truthful to report — and it
used to report the release version the *server* was offering when it paired, which is a
fact about the server and identical on every container regardless of what it was running.

**If a machine's owner wants updates to be automatic**, that is their choice to make on
their machine, not something the fleet does to them. A cron entry or a systemd timer
running `docker compose pull && docker compose up -d` is enough; so is
[Watchtower](https://containrrr.dev/watchtower/) if they prefer something with a UI.
Work in flight is handed back rather than lost, so this is safe to run unattended.

## What a container reports about itself

A container is limited by its cgroup, and `/proc` inside it is the host's. So
`os.cpus()` and `os.totalmem()` describe the machine the container is on, not the
container — an agent capped at `--cpus 1.5 --memory 512m` reported 15 cores and 12 GB
until this was fixed, which is what a scheduler would have used to decide how much work
it could take.

The agent now reads `cpu.max` and `memory.max` from the cgroup (v2, falling back to v1)
and reports those, falling back to the host's figures when the container genuinely has no
limit:

| Container | Reports |
|---|---|
| `--cpus 1.5 --memory 512m` | 2 cores, 512 MB total, free memory from `memory.current` |
| `--cpus 4 --memory 2g` | 4 cores, 2048 MB |
| no limits | the host's cores and memory, which is the truth |

A fractional CPU quota is rounded rather than floored, because `--cpus 1.5` is
meaningfully more than one core's worth of work and flooring would make every
fractionally limited machine look like the smallest possible worker.

## Optional workloads

The default image runs Python/PyTorch CPU projects as well as the existing
`echo` and `walker_evolution` JavaScript workloads. ONNX inference is a separate build target, because it is ~85 MB that most
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
- stopping a container hands its work back rather than abandoning it: `docker stop`
  exits cleanly in under a second, the task is requeued in well under a second instead
  of waiting out its 45-second lease, and it does not come back to the machine that is
  shutting down
- killing a container mid-flight loses no tasks, and restarting it brings back the same
  machine with its history intact

Useful flags: `--agents N`, `--tasks M`, `--skip-build`, `--skip-chaos`, `--keep` (leave
everything running to poke at it).

Measured on one laptop: 8 containers, 300 tasks, all verified, 41 tasks/second.

## Stopping and updating

`docker stop`, `docker compose down`, and the stop half of `docker compose up -d` all
send SIGTERM. The agent answers it by withdrawing consent, handing back whatever it is
holding, and closing the connection — measured at under 300ms, with the task picked up
by another machine within a second.

That ordering matters and was got wrong first. Declining the task *before* withdrawing
consent handed it straight back to the agent that was in the middle of dying: the server
finishes a declined task and immediately looks for another to offer the same open
connection. The task was reassigned to the same host 200ms after being returned, sat
there until the 45-second lease expired, and burned one of its three attempts doing it.
A fleet-wide restart could have exhausted a task's retries on hand-offs alone.

So updating a fleet is just:

```sh
docker compose -f deploy/compose.agent.remote.yaml pull
docker compose -f deploy/compose.agent.remote.yaml up -d
```

Work in flight moves to another machine rather than stalling, and each container comes
back as the same machine with its history intact.

## Uploaded Python projects on GPU workers

The default image remains CPU PyTorch. An existing NVIDIA host can run the same agent
with GPU-enabled PyTorch by rebuilding with the GPU compose override:

```sh
docker compose --env-file .env.agent -f deploy/compose.agent.yaml -f deploy/compose.agent.gpu.yaml up -d --build
```

The override exposes GPUs and selects the official `cu130` PyTorch wheel index.
Set `PYTORCH_INDEX_URL` in `.env.agent` to a compatible official CUDA index if your host
requires another build of the pinned PyTorch version. The NVIDIA driver and Container Toolkit
must already expose the GPU to Docker. Allow enough host/container RAM for PyTorch and the
project via `DWP_MEMORY`. The existing agent identity volume is retained.

For a custom image build, pass `--build-arg PYTORCH_INDEX_URL=https://download.pytorch.org/whl/cu130`
and run with `--gpus all`. The existing `--target cuda` image serves ONNX workloads;
use the default `agent` target for uploaded Python projects.

The agent runs a real PyTorch tensor check using `DWP_PROGRAM_PYTHON` and reports its
runtime, device, VRAM and PyTorch version. A CUDA-enabled wheel without GPU access reports
CPU. CPU preference remains respected. On native macOS, the same probe supports Apple MPS.
Native Windows bridging is not enabled; use the Linux Docker worker on NVIDIA desktops.

Deploy the updated backend too, so the upload planner accepts GPU requirements. Restart
the worker after changing its Python environment, confirm `python_program` and the measured
GPU runtime in its capabilities, and resubmit any job rejected under the former CPU policy.
This setup does not install host drivers or allocate GPU pods.

For a real GPU worker acceptance test (requires PyTorch and the selected device):

```sh
RUN_PYTHON_GPU_TESTS=1 PYTHON_GPU_RUNTIME=cuda PYTHONPATH=backend/tests \
  backend/.venv/bin/python -m unittest integration.test_python_gpu -v
```

Use `PYTHON_GPU_RUNTIME=mps` on macOS. This runs dependency setup, a probe, training,
checkpoint reload and validation through the worker executor; an unavailable requested
GPU fails the test instead of falling back to CPU.
