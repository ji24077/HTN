# Getting friends connected to your machine

**The situation:** your Mac is the server. Friends anywhere on the internet connect to it.
This document is how that works, what we verified, and what is still open.

---

## 1. Why you cannot skip the tunnel

Your Mac's Wi-Fi interface has the address `100.66.86.135`. That is inside `100.64.0.0/10`,
which is **carrier-grade NAT** — your ISP shares one real public address among many
customers. Checked against an external service, your machine appears as `138.51.79.158`,
an address you do not control and cannot forward ports on.

This is decisive:

- **Port forwarding cannot work.** There is no router setting that fixes it; the NAT is
  in your ISP's network, not your house.
- **Dynamic DNS cannot work either.** It would point at an address shared with strangers.
- Every connection must therefore be **started from your machine, outwards**, and a relay
  must join it to incoming visitors. That is exactly what a tunnel is.

The architecture already assumed outbound-only connections for host agents. It turns out
the same constraint applies to the server itself.

---

## 2. What we verified

| Check | Result |
| --- | --- |
| Tunnel reaches Cloudflare | **Pass** — QUIC and HTTP/2 both connect; prechecks all green |
| A request from the public internet reaches the service | **Pass** — verified by resolving through a public DNS server and connecting to the edge |
| An agent completes a WebSocket handshake over `wss://` through the tunnel | **Pass** |
| A job submitted through the public URL runs and returns a signed result | **Pass** |
| The agent opens no inbound ports | **Pass** — zero listening sockets; one outbound connection on 443 |
| A third-party server on an unrelated network reaches it | **Not yet** — see §4 |

## 3. Two real problems found, and what they mean

### Your Mac cannot resolve its own tunnel address

`dig` against `1.1.1.1` and `8.8.8.8` both answer instantly for the tunnel hostname. Your
system resolver returns nothing at all.

The practical effect is misleading: **the address works for everyone else and looks dead
from your own machine.** Opening your own link in your browser may fail while a friend
connects fine.

`scripts/lib/net-probe.ts` now handles this — it falls back to a public resolver and
connects directly to the address, and `share.ts` tells you when this is happening instead
of reporting a working tunnel as broken.

### Account-less tunnels register only one edge connection

A healthy Cloudflare tunnel registers about four connections across several regions.
Every account-less "quick tunnel" we opened registered exactly **one**, in Toronto. A
request arriving at that edge is served correctly; a request arriving anywhere else gets
`522`, and never reaches your machine at all.

That is consistent with what Cloudflare says about these tunnels: *"no uptime guarantee"*,
intended for experimenting. They are also rate-limited, and we created several in quick
succession while testing.

**Conclusion: a quick tunnel is fine for trying things out, and is not the foundation to
build on.** It also gets a brand-new random hostname on every restart, and agents remember
the address they paired with — so every restart breaks every friend's agent.

---

## 4. Choosing a permanent address

`node scripts/share.ts --url https://your-address` uses an address you control. Pick one:

| Option | Cost | Needs | Stable address | Notes |
| --- | --- | --- | --- | --- |
| **Cloudflare named tunnel** | Free | Cloudflare account + a domain (~$10/yr) | Yes | Proper multi-edge routing. The intended production path. |
| **Tailscale Funnel** | Free | Tailscale account | Yes, a `*.ts.net` name | No domain needed. Friends need nothing installed — it is a normal public URL. Easiest if you do not own a domain. |
| **ngrok** | Free tier | ngrok account | One reserved domain on free tier | Simple, well documented. |
| **A small VPS** | ~$5/mo | — | Yes | Runs the server itself rather than tunnelling to your Mac. Removes your laptop from the critical path entirely. |

**Recommendation: Tailscale Funnel if you do not own a domain, a named Cloudflare tunnel
if you do.** Either takes about ten minutes and removes both problems above.

The rotating-address problem has a workaround for the meantime — a friend can re-point an
already-enrolled agent without re-pairing:

```bash
pnpm agent set-server --server https://the-new-address
```

Their identity, keys and history survive; only the address changes.

---

## 5. What a friend actually does

Send them `https://your-address/join?code=ABCD-1234`. The page shows them:

```bash
pnpm install
pnpm agent pair --server https://your-address --code ABCD-1234
pnpm agent run
```

The pairing code is single-use and expires in ten minutes. `pnpm share` prints a fresh
invite link and gives you another each time you press Enter.

They can stop at any time with `pnpm agent pause`, which works even when your server is
unreachable, because it is a local file rather than a request.

**They still need Node 24, pnpm, and a copy of this repo.** That is the biggest remaining
friction for non-technical friends, and is worth solving with a packaged installer before
inviting anyone who is not comfortable in a terminal.

---

## 6. Safety, given this is your personal laptop

Exposing a machine to the internet deserves care, so the following are enforced rather
than documented:

- **The server refuses to start** with a default or weak password when it is publicly
  reachable, and refuses a non-HTTPS public address. It is a startup failure, not a
  warning, because warnings scroll past.
- **`pnpm share` generates a strong password** on first run and writes it to `.env`.
- **It listens on `127.0.0.1` only.** The tunnel reaches it over loopback, so nothing on
  your local network can touch the port.
- **Login and pairing are rate limited** (10 and 20 attempts per 5 minutes per address).
- **Agents cannot be sent arbitrary code.** Only the task types this repo ships will run,
  and a friend's machine never receives a shell command.
- **Internal errors are not returned to callers**, only logged locally.

What this does *not* protect against: anyone with your operator password has full control,
and the pairing link in a chat message is a valid invite for ten minutes. Treat both like
passwords.

---

## 7. Watching a real connection

When a friend first tries to join, run this in a second terminal:

```bash
pnpm logs                       # everything, live
pnpm logs --grep connect        # just connection activity
pnpm logs --level warn          # only problems
pnpm logs --story               # per-host summary: joins, drops, refusals
pnpm logs --since 30m --no-follow   # what happened earlier
```

`GET /diagnostics` (signed in) shows the same thing as JSON, including which hosts keep
dropping and why any authentication was refused.

The question to answer first is always **"did their request reach me at all?"** — if
nothing appears in the log, the problem is the address or their network, not the software.
If `agent.auth_rejected` appears, the reason field says exactly what was wrong.

---

## 8. How this was tested without a second computer

We cannot yet stage a real friend on a real other network, so the conditions that matter
are reproduced locally. `sim/` runs the **real** control service and **real** agent
processes — the same code a friend would run — and puts a fault-injecting proxy between
them (`sim/netsim.ts`). The proxy can add latency and jitter, throttle bandwidth, refuse
connections, silently swallow traffic, block WebSocket upgrades, and cut connections on a
schedule.

```bash
pnpm sim              # all scenarios
pnpm sim --list       # what each one covers and why
pnpm sim flaky-wifi   # just one
pnpm sim:edge         # adversarial input rather than adverse conditions
```

Timings are compressed for the suite (2s heartbeat instead of 15s) via the same
environment variables production uses, so the mechanisms exercised are the real ones.

**This is not a substitute for a real second machine.** It cannot reproduce ISP routing,
corporate TLS interception, IPv6-only networks, or genuine cross-continent latency
variance. What it does prove is that the *software* handles disruption correctly, so when
a real laptop joins, anything that goes wrong is likely to be environmental — and the logs
will say which.
