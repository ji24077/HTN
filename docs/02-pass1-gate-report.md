# Pass 1 — Network spike: gate report

**Date:** 2026-09-18
**Code under test:** `packages/{protocol,control,agent}`, commit at time of run
**Harness:** `scripts/gate-security.ts`, `scripts/verify-run.ts`

> **Headline:** six of the seven Pass 1 items pass on real code with real evidence.
> The `network` gate is **INCOMPLETE** and stays open: both hosts ran on one machine
> on one network, because a second machine was not available. Everything below is
> reported at the strength the evidence actually supports.

---

## Environment

| | |
| --- | --- |
| Machine | macOS 26.3.1, arm64, 15 logical cores, 24 GB |
| Runtime | Node 24.14.0 — type stripping, no build step |
| Typecheck | TypeScript 7.0.2, clean |
| Database | PostgreSQL 18 (Docker), `dwp-db` |
| Public endpoint | Cloudflare quick tunnel → `https://<name>.trycloudflare.com` |
| Public egress of this machine | `138.51.79.158` (measured independently via api.ipify.org) |

## Gate results

| Item | Result | Evidence |
| --- | --- | --- |
| 1.1 Protocol package | **PASS** | `tsc --noEmit` clean across all three packages on TS 7.0.2 |
| 1.2 Public HTTPS/WSS control service | **PASS** | Agent `charlie` connected to `wss://<tunnel>/agent/connect`; job submitted and completed entirely through the public URL |
| 1.3 Pairing | **PASS** | Code single-use and time-boxed; key file created at mode 600; agent refuses to start on wider permissions |
| 1.4 Agent transport | **PASS** | EdDSA assertion per connect; replay rejected; heartbeat and presence working |
| 1.5 Echo adapter and dispatch | **PASS** | Each host returned its own `hostId` and the server-issued nonce; results independently signature-verified |
| 1.6 Presence | **PASS** | Disconnect flips `online=false`; leases expire on their own clock |
| 1.7 Chromium + `/whoami` | **PASS (mechanism)** | Site observed `138.51.79.158`, matching this machine's independently measured egress. No TCP debug port; driven over `--remote-debugging-pipe` |
| **`network` — two distinct networks** | **INCOMPLETE** | Both agents ran on one machine. Not the cross-network proof, and not presented as one. |

## `gate:security` — 7 / 7

```
PASS  unenrolled host is rejected                              — unknown-host
PASS  invalid pairing code is rejected                         — HTTP 400
PASS  pairing code cannot be redeemed twice                    — first 200, second 400
PASS  assertion signed by the wrong key is rejected            — bad-signature
PASS  revoked host is rejected                                 — unknown-host
PASS  a replayed assertion is rejected                         — first ACCEPTED, replay replayed
PASS  a result signed with an unrelated key is rejected        — task left 'leased', never accepted
```

The control service recorded `result.signature_invalid` for the forged result — the
rejection happened in the code path that matters, not in an early bail-out.

## `gate:recovery` — kill a host mid-batch

60 unpinned echo tasks at 2 s each, concurrency cap 2 per host. `SIGKILL` on `bravo` at t+8 s.

```
tasks total             60
accepted results        60      exactly one per task
tasks with >1 attempt    2      the two leases bravo held when it died
lease.expired events     2
duplicate_result         0
per host                 alpha 52, bravo 8
```

Recovery behaved as designed: only the unfinished leases came back, after their 30 s
expiry, and completed work was never redone or double-counted.

## `gate:distribution` — provenance without trusting the server

`scripts/verify-run.ts` reads the database directly and verifies every accepted
result's Ed25519 attestation against the enrolled host public keys. It deliberately
ignores the control service's own event log, because the control service writes that log.

```
accepted results : 60
distinct hosts   : 2
  alpha   52 verified
  bravo    8 verified
gate:distribution PASS
```

The signing key never leaves its host; the control service stores only public keys.

## No inbound listeners

```
alpha   (agent)   0 listening sockets   out: 127.0.0.1:52863 -> 127.0.0.1:8787
charlie (agent)   0 listening sockets   out: 100.66.86.135   -> 104.16.231.132:443
control           1 listening socket    :8787
```

Neither agent opens a listening socket. `charlie`'s only connection is outbound on 443.

## Findings that changed the architecture

1. **Node 24 strips types across pnpm workspace links.** Verified directly. The repo
   needs no build step and no `dist/`; TypeScript is a typechecker only. §4 simplifies.
2. **TypeScript 7.0.2 typechecks the workspace with zero errors**, so the §4 fallback
   trigger never fired and 7.0.2 is pinned as originally specified.
3. **Playwright drives Chromium over `--remote-debugging-pipe`** — read from the live
   process table, not taken from the docs. §2.5 moves from **[D]** to **[V]** on macOS.
   `chromium.launch()` also rejects a `--user-data-dir` argument; the profile directory
   must come from `launchPersistentContext()`, which is what the agent now does.
4. **Concurrency bug, found and fixed.** Overlapping dispatch passes each read a host's
   free capacity before either had claimed, so a host with a cap of 2 was handed ~12
   tasks — measured at 12× the intended concurrency before the fix. A per-host dispatch
   lock in `hub.ts` resolves it. This is the same shape of race the architecture warned
   about for leases; it appeared one level up, in the dispatcher.
5. **A gate can pass for the wrong reason.** The first forged-result check used a random
   task id, so the server rejected it before reaching the signature verification. It
   printed PASS while testing nothing. Rewritten to forge a result for a task the host
   genuinely holds.

## What Pass 1 does *not* establish

- **Cross-network operation.** One machine, one network. The `network` gate is open.
- **A remote browser session.** 1.7 launches Chromium and proves egress and the absence
  of a debug port. Live frames, input forwarding, and the enforcing proxy are Pass 3.
- **Any performance claim.** The echo adapter computes nothing. Speedup is Pass 2.
- **Tunnel as deployment.** A quick tunnel is a spike tool. The control service needs a
  managed host with a stable name before Pass 2 measurements mean anything.
