# Connectivity pass — gate report

**Date:** 2026-09-18
**Goal:** friends anywhere on the internet can connect to a server running on one Mac.
**Method:** real processes throughout. A public Cloudflare tunnel for the live path, and a
fault-injecting network proxy (`sim/`) for conditions that cannot be staged on one desk.

---

## Verdict

**The connection works end to end over the public internet.** A host paired through a
public HTTPS URL, connected over `wss://`, received work and returned signed results.

**It is not yet ready to rely on**, for one reason that is environmental rather than a
software defect: the address in use is a throwaway tunnel that changes on every restart
and routes through a single edge. §4 says what to switch to.

---

## What was proven live

| Step | Evidence |
| --- | --- |
| Server reachable on a public HTTPS address | `HTTP 200` on `/health` through the tunnel |
| The join page renders with the code filled in | `HTTP 200`, correct `pnpm agent pair …` command |
| A host pairs over the public URL | `friend-laptop` enrolled, config stored `wss://…trycloudflare.com` |
| That host connects over `wss://` | handshake in **375 ms**, clock skew **38 ms** |
| Work runs on it and comes back signed | 5/5 tasks succeeded, all Ed25519 signatures verified |
| The agent opens no inbound port | zero listening sockets; one outbound connection on 443 |

## Network conditions — 11 scenarios, 42 checks, all passing

Real control service, real agent processes, simulated network between them.

| Scenario | What it reproduces |
| --- | --- |
| `happy-path` | Two computers, clean network |
| `high-latency` | 400 ms each way, 80 ms jitter |
| `flaky-wifi` | Connection dropped every 5 s during a 100-task job |
| `laptop-sleep` | Link silently swallowed — no close, no error |
| `sleep-mid-task` | A computer dies while holding leased work |
| `blocked-websockets` | Network allows HTTPS but blocks WebSocket upgrades |
| `server-unreachable` | Host machine asleep when a friend tries to join |
| `control-restart` | Server restarted mid-job |
| `nat-idle-timeout` | Router reclaiming idle connections every 8 s |
| `reconnect-storm` | Six computers lose the network simultaneously |
| `slow-link` | 32 KB/s with 150 ms latency |

In every case involving work: **every task completed exactly once, with no duplicates.**
`flaky-wifi` completed all 100 tasks across 5 genuine disruptions, 8 needing a retry.

## Adversarial input — 25 checks, all passing

Expired tokens, clocks an hour fast, over-long token lifetimes, `alg: none`, algorithm
confusion, wrong audience, tampered payloads, malformed frames, 64 KB of junk, unicode and
markup in labels, unauthenticated access to every endpoint.

---

## Bugs found and fixed

Every one of these was found by testing, not by reading the code. Several would have
looked like "it just doesn't work" to a friend.

| # | Bug | Consequence if shipped |
| --- | --- | --- |
| 1 | Agent listened for `'pong'` instead of `'ping'`, so the server's own keepalives counted as silence | A healthy idle agent declared the server dead and reconnected every ~6 s — 200 reconnects where 12 were correct |
| 2 | No WebSocket handshake timeout | A laptop waking from sleep hung in CONNECTING **forever**, with no error |
| 3 | Server used `close()` on silent peers | A sleeping laptop could never be shed: repeated timeouts, connection stuck in CLOSING |
| 4 | Agent had no liveness check at all | If the *server* went silent, the agent waited indefinitely |
| 5 | Backoff reset on every `open` | A flapping link became a hot reconnect loop against the server |
| 6 | `BOOTSTRAP_PASSWORD` changes had no effect on an existing account | Rotating the password produced an unexplained `401` |
| 7 | Validation failures returned **HTTP 500** | A friend's misconfigured client looked like a server fault |
| 8 | Presence never reset on restart | The fleet listed dead machines as online after every restart |
| 9 | Tunnel pointed at `localhost`, service bound to `127.0.0.1` | macOS resolves `localhost` to `::1` first, so the tunnel reached nothing |
| 10 | Pairing failures printed a raw stack trace | The first error anyone joining would ever see was unreadable |
| 11 | `share.ts` called its own API through the public URL | Broke whenever local DNS could not resolve the tunnel hostname |

Two test defects were also fixed, both of the same dangerous kind — **a green check that
tested nothing**: a forged-result check that used a random task id and was rejected before
reaching the signature verification, and a `flaky-wifi` run whose job finished before the
first disruption fired.

---

## Findings about this specific machine

Both are environmental, and both matter for what to do next.

**You are behind carrier-grade NAT.** Your Wi-Fi interface holds `100.66.86.135`, inside
`100.64.0.0/10`, and your resolver is `172.20.10.1` — an iPhone Personal Hotspot gateway.
Port forwarding is impossible on this connection, not merely inconvenient.

**Your DNS does not resolve freshly created subdomains.** The system resolver answers
`google.com`, `cloudflare.com` and `trycloudflare.com` correctly, but returns nothing for a
tunnel subdomain created seconds earlier, which `1.1.1.1` answers instantly. The practical
effect is that **a working address looks broken from your own machine while working fine
for everyone else.**

Both are now handled rather than merely documented: `share.ts` detects the DNS case and
says so instead of reporting failure, and agents accept `DWP_DNS_FALLBACK=1`, which tries
the system resolver first and falls back to public resolvers. The live pairing above was
done that way.

---

## Still open

1. **A permanent address.** Quick tunnels rotate their hostname on every restart, and
   each registered only one edge connection — requests arriving at other edges got `522`.
   See `docs/03-connectivity.md` §4; Tailscale Funnel or a named Cloudflare tunnel.
2. **A real second computer.** Everything here ran on one machine. The simulator covers
   the software's behaviour, not ISP routing, corporate TLS interception, or IPv6-only
   networks.
3. **Getting the agent onto a friend's machine.** They still need Node 24, pnpm and a
   copy of the repo. That is the largest remaining barrier for anyone non-technical.
