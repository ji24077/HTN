# Distributed Work Platform — Implementation Architecture (Pass 0)

**Status:** For review. No implementation authorized by this document.
**Date:** 2026-09-18
**Input:** [`00-handoff.md`](./00-handoff.md)
**Verification convention:** every factual claim is tagged **[V]** verified in this session, **[D]** documented by the vendor but not re-verified here, or **[A]** assumption to be tested during a named pass.

---

## 0. Executive answer

**A two-computer cross-network demo is achievable under the stated constraints.** Nothing in the handoff's Checkpoint A requires a capability that does not exist. The topology — host-initiated outbound WSS on 443 to a public control service — is the same mechanism used by every mainstream remote-access and CI-runner product, and it needs no inbound ports, no VPN, and no UPnP.

Three things in the handoff are overclaimed or underspecified and are corrected below:

1. **The 1.2× stretch speedup has a hard floor on batch size.** With the illustrative host rates in §7, the batch needs **≥ ~680 items** before 1.2× is even arithmetically reachable, regardless of scheduling quality. Smaller batches cannot pass, and the team should know that before choosing the demo dataset. (§7.4)
2. **"Proof of distribution" is not provable from the control service's own logs.** The service writes those logs; a demo that fakes distribution would produce identical ones. Host-signed result attestations are added so provenance is verifiable by a reviewer who does not trust the server. (§12.5)
3. **Storing `observed egress IP` on the `Host` record**, as the handoff's record shape suggests, is both unstable (CGNAT, tethering, VPN toggles) and a standing privacy cost. It becomes an on-demand probe displayed live and recorded per browser session only. (§5.3)

**Recommendation: GO for Pass 1**, conditional on the two deployment answers in §14.

---

## 1. Components, paths, and secret visibility

### 1.1 Components

| Component | Runs on | Responsibility |
| --- | --- | --- |
| `web` | User's browser | Dashboard: fleet, job submission, run map, browser live view, history. Holds no host credentials. |
| `control` | Public host, TLS-terminated | AuthN/AuthZ, pairing, placement, lease grants, event log, media relay. Single writer to the DB. |
| `db` | Managed Postgres | Durable jobs/tasks/leases/events. Source of truth for placement. |
| `agent` | Each opted-in computer | Outbound WSS client, capability probe, adapter host, consent enforcement, local kill switch. |
| `adapter:cpu_inference_batch` | Inside `agent` | ONNX Runtime CPU execution of pinned model over assigned input slices. |
| `adapter:remote_browser_session` | Inside `agent` | Playwright Chromium in an ephemeral context, behind an agent-local enforcing proxy. |

`web`, `control`, and `agent` are three distinct trust principals. The handoff's terminology (browser client / host agent / remote browser) is kept verbatim.

### 1.2 Three planes

```
CONTROL PLANE   web --HTTPS/WSS--> control <--WSS (host-initiated)-- agent
DATA PLANE      agent --HTTPS GET /artifacts/:sha256--> control        (model + inputs, content-addressed)
                agent --WSS task.result--> control                     (results, size-capped)
MEDIA PLANE     agent --WSS binary frames--> control --WSS--> web      (Pass 3; relay)
                web --WSS input events--> control --WSS--> agent
```

Only the selected host fetches the target website. The control service never proxies the remote browser's page traffic — that would defeat the entire egress feature. This is an invariant, tested by the Gate `browser-egress` check in §11.

### 1.3 Secret visibility matrix

| Secret / sensitive datum | Created by | Stored where | Can be read by |
| --- | --- | --- | --- |
| User password hash | control | `db` (Argon2id) | control only |
| User session cookie | control | Browser, httpOnly/Secure/SameSite=Strict | web (opaque), control |
| Pairing code | control | `db`, 10 min TTL, single use | User (shown once), control |
| Host private key (Ed25519) | agent, on the host | `~/.dwp/agent.key`, mode 0600 | **Host owner and root on that machine only.** Never transmitted. |
| Host public key | agent | `db` | control, anyone with DB read |
| Browser-session control token | control | Memory + `db` row, session TTL | Session owner, control |
| Remote browser page content / cookies | remote Chromium | Ephemeral profile dir, deleted on stop | **Host owner (full), platform operator (frames in transit), session owner** |
| Job inputs and results | submitter | `db` + content-addressed store | Submitter, assigned host owner, platform operator |

The two bolded rows are the trust-model core and must appear in the UI, not only in this document. The platform makes **no end-to-end confidentiality claim** in the MVP: relayed frames transit the control service in plaintext-over-TLS and are decryptable there.

---

## 2. Pairing, authentication, revocation, NAT, transport

### 2.1 Enrollment (host-initiated, code-mediated)

1. Signed-in user clicks *Enroll computer* → `POST /hosts/pair-code` → `{code: "K7M2-P4QX", expiresAt}`. 8 chars, Crockford base32, single use, 10-minute TTL, bound to `user_id`.
2. On the target machine: `dwp-agent pair --server https://control.example --code K7M2-P4QX`.
3. Agent generates an **Ed25519 keypair locally** and writes the private key to `~/.dwp/agent.key` (0600, parent dir 0700). It refuses to start if permissions are wider. **[A: macOS Keychain / Windows DPAPI storage deferred to hardening; file permissions are the MVP control.]**
4. `POST /hosts/pair {code, publicKey, hostInfo}` → server validates the code, marks it consumed inside the same transaction, creates the `hosts` row, returns `hostId`.
5. The code is never reusable, never logged, and never grants anything by itself — it only authorizes the binding of one public key.

**Why asymmetric rather than a bearer token:** the control database then holds no credential that can impersonate a host. A DB read compromise cannot forge host results, which is what makes the attestation in §12.5 meaningful.

### 2.2 Session authentication

The agent connects to `wss://control.example/agent/connect` presenting an EdDSA-signed JWT in the `Authorization` header: `{iss: hostId, aud: "dwp-control", iat, exp: iat+120, jti}`. Server verifies the signature against the stored public key, rejects `exp` skew > 120s, and rejects a replayed `jti` from a 10-minute cache. Reconnects mint a fresh assertion. No long-lived bearer token exists on the wire.

### 2.3 Revocation and pause — two different things

| Action | Who | Effect | Reversible |
| --- | --- | --- | --- |
| **Pause** | Host owner (local CLI/tray) or dashboard | `consent.paused = true`; no new placements; running tasks finish; browser sessions end | Yes, instantly |
| **Revoke** | Host owner or admin | `hosts.revoked_at = now()`; live socket closed with code `4003`; agent stops retrying and self-disables | No — re-pair required |

The local pause switch must work with the control service unreachable. It is a local file flag the agent reads, not a server round-trip.

### 2.4 NAT and transport — [V] for the mechanism, [A] for this team's networks

Outbound TCP/443 with TLS and a WebSocket upgrade traverses ordinary home NAT without configuration, because the connection is established from inside. No inbound listener exists on the host. Keepalive: WS ping every 15 s; server marks a host `offline` after 45 s without a pong; agent reconnects with full-jitter exponential backoff (1 s → 60 s cap).

The one environment this can fail in is a network with a TLS-inspecting middlebox or a proxy that blocks WS upgrades. **[A]** — to be confirmed on both real networks in Pass 1; fallback is long-poll/SSE + POST, which is strictly worse and should only be built if measured to be necessary.

### 2.5 No exposed CDP

- The agent launches Chromium via Playwright's `chromium.launch()`, which drives the browser over a stdio pipe rather than a TCP debugging port. **[D — Playwright docs; re-verify in Pass 3 by asserting no process has `--remote-debugging-port` and no port is listening.]**
- `launchServer()` and any `--remote-debugging-port` flag are **forbidden**; a lint rule and a startup assertion enforce this.
- Chrome documents that an exposed remote-debugging port permits cookie extraction from the profile. [Chrome remote-debugging security](https://developer.chrome.com/blog/remote-debugging-port) **[D]**
- The dashboard receives frames and sends semantic input events. It never receives a CDP endpoint, session ID, or WebSocket URL that reaches the browser directly.

---

## 3. Browser-only host: why it cannot be the host (finding, not preference)

| Requirement of a host | Web page | Installed agent |
| --- | --- | --- |
| Launch and drive a separate Chromium with an isolated profile | Impossible | Yes |
| Native-speed multithreaded ONNX CPU inference | WASM only; threads need cross-origin isolation (COOP/COEP) and SharedArrayBuffer **[D]** | Yes, full ORT CPU EP |
| Stay available with no tab open | No | Yes |
| Stable machine identity and a protected key file | No (origin-scoped storage, clearable) | Yes |
| Screen/desktop capture | Requires a user gesture and a picker per call **[D: MDN getDisplayMedia]** | N/A — the agent captures the browser it owns, not the desktop |
| Bounded, enforceable CPU/RAM caps | No | Yes (OS-level) |

**Conclusion:** a browser tab can be a *viewer* and, as a later experiment, a WASM compute worker for a narrow task type. It cannot be the general host. This confirms the handoff's position; it is recorded here as a decision with reasons, so it is not relitigated.

---

## 4. Repository layout and technology choices

```
distributed-work-platform/
  pnpm-workspace.yaml
  packages/
    protocol/         # zod schemas: REST DTOs, WS envelope, adapter contracts. No runtime deps.
    control/          # Fastify + ws + drizzle. The only DB writer.
      src/{http,ws,scheduler,leases,events,artifacts}/
      migrations/
    agent/            # host agent
      src/{transport,consent,capability,adapters/{inference,browser},proxy}/
    web/              # Vite + React dashboard
      src/{fleet,new-work,run-map,live-view,history}/
  fixtures/           # model + input MANIFESTS (hashes, URLs) — not the blobs
  bench/              # measurement harness, JSONL output
  docs/
  scripts/
```

`protocol` is imported by all three runtime packages; a contract change that breaks a peer fails typecheck in CI. This is the cheapest available defense against the control/agent message drift that kills projects of this shape.

### Versions — [V] resolved from the npm registry on 2026-09-18

| Choice | Version | Note |
| --- | --- | --- |
| Node.js | 24.14.0 **[V local]** | Agent and control. Pin via `.nvmrc` + `engines`. |
| pnpm | 10.27.0 **[V local]** | Workspaces. |
| TypeScript | 7.0.2 **[V]** | Latest is the native compiler. **[A]** If any dependency's type tooling lags, fall back to the 5.9 line; decide in Pass 1, record the reason. |
| Fastify | 5.12.5 **[V]** | HTTP. `ws` 8.21.3 **[V]** for both WSS endpoints. |
| Drizzle ORM | 0.45.2 **[V]** + `pg` 8.23.0 **[V]** | SQL-first; the lease query is hand-written SQL regardless. |
| PostgreSQL | 18.3 **[V local]** | `FOR UPDATE SKIP LOCKED` for atomic lease claims. |
| Playwright | 1.63.0 **[V]** | Chromium contexts. |
| onnxruntime-node | 1.30.0 **[V]** | Package metadata declares `os: ['win32','darwin','linux']` **[V]**. CPU EP is the MVP target; GPU EP availability varies by OS (CUDA on win32/linux; CoreML on darwin) **[D]** and is explicitly out of scope. |
| React / Vite | 19.3.0 / 8.3.0 **[V]** | Dashboard. |
| Zod | 4.6.5 **[V]** | One schema per contract, shared by both sides of every wire. |

All versions are pinned exactly in `pnpm-lock.yaml`. "Latest" is recorded as of this date and is not assumed to stay current.

---

## 5. Contracts, schema, and the task state machine

### 5.1 REST

```
POST   /auth/login                      -> sets session cookie
POST   /hosts/pair-code                 -> { code, expiresAt }
POST   /hosts/pair                      -> { hostId }            (code-authenticated, not session)
GET    /hosts                           -> Host[]
POST   /hosts/:id/consent               -> update allow_compute / allow_browser / caps
POST   /hosts/:id/revoke                -> 204
POST   /jobs                            -> Job                    (adapter, manifestHash, constraints)
GET    /jobs/:id                        -> Job + task rollup
GET    /jobs/:id/tasks                  -> Task[]
POST   /jobs/:id/cancel                 -> 202
GET    /jobs/:id/results                -> streamed NDJSON
POST   /browser-sessions                -> BrowserSession + control token
POST   /browser-sessions/:id/stop       -> 204
GET    /artifacts/:sha256               -> bytes    (agent-authenticated, content-addressed)
```

### 5.2 WebSocket envelope

```ts
type Envelope<T = unknown> = {
  v: 1
  id: string          // ULID, unique per message
  replyTo?: string    // correlates a response to a request
  ts: string          // sender clock, ISO-8601 — for durations only, never for ordering
  type: string
  payload: T
}
```

Ordering is established by the server's monotonic `run_events.seq`, never by sender timestamps. Clock skew between a Mac and a Windows box is a real and under-appreciated source of nonsense run maps.

| Direction | Types |
| --- | --- |
| agent → control | `hello`, `heartbeat`, `capability.update`, `consent.update`, `task.accept`, `task.decline`, `lease.renew`, `task.progress`, `task.result`, `task.error`, `calib.result`, `browser.opened`, `browser.frame`, `browser.event`, `browser.closed` |
| control → agent | `hello.ack`, `task.offer`, `task.cancel`, `browser.open`, `browser.input`, `browser.close`, `revoked`, `drain` |
| control → web | `job.updated`, `task.updated`, `host.updated`, `run.event`, `frame`, `session.ended` |
| web → control | `input`, `session.keepalive` |

`task.result` payloads are capped at **256 KiB**; larger outputs are uploaded content-addressed and referenced by hash. A classifier over N inputs produces small results, so the cap will not bind in the MVP — it exists so the WS relay can never be a bulk data path by accident.

### 5.3 Schema (abridged; full DDL lands with Pass 2)

```sql
users(id, email, password_hash, role, created_at)

hosts(id, owner_id, public_key, trust_tier, allow_compute, allow_browser,
      os, arch, cpu_model, logical_cores, total_ram_mb, agent_version,
      max_concurrency, cpu_quota_pct, mem_limit_mb,
      self_reported_region, online, last_heartbeat_at, paused,
      created_at, revoked_at)
-- NOTE: no observed_egress_ip column. See below.

pair_codes(code_hash, user_id, expires_at, consumed_at)

jobs(id, owner_id, adapter, status, manifest_hash, total_items,
     constraints jsonb, created_at, started_at, ended_at, cancel_requested_at)

tasks(id, job_id, seq, input_ref, state, assigned_host_id,
      lease_id, lease_expires_at, attempts, idempotency_key,
      output_ref, output_hash, error_class,
      queued_at, started_at, finished_at)

task_attempts(id, task_id, attempt, host_id, started_at, finished_at,
              outcome, host_signature, host_reported_ms)

browser_sessions(id, owner_id, host_id, start_url, allowlist_id, viewport jsonb,
                 state, observed_egress_ip, created_at, expires_at, ended_at, end_reason)

run_events(seq bigserial, id, job_id, task_id, session_id, host_id,
           actor, category, type, payload jsonb, server_ts)

calibrations(host_id, adapter, model_hash, agent_version,
             items_per_sec, setup_ms, measured_at)   -- PK: all four; TTL 24h
artifacts(sha256, kind, bytes, content_type, created_at)
```

**Deviation from the handoff:** `observed_egress_ip` is *not* on `hosts`. A host's egress address changes with tethering, VPN toggles, and CGNAT reassignment, so a stored value is stale and misleading precisely where honesty matters most. It is probed on demand for the fleet card (displayed, not persisted) and recorded on `browser_sessions` where it is evidence about one specific run.

### 5.4 Task state machine

```
          offer                accept              start
pending ─────────> offered ───────────> leased ───────────> running
   ^                  │                    │                   │
   │ lease expiry /   │ decline            │ lease expiry      │ result
   │ decline / nack   │                    │                   ▼
   └──────────────────┴────────────────────┴──────────── succeeded
                                                              │
   cancel (any non-terminal) ──> cancelled     max attempts ──> failed
```

- **Lease:** granted for 30 s. The agent renews every 10 s while `leased`/`running`. A sweeper every 5 s runs
  `UPDATE tasks SET state='pending', assigned_host_id=NULL, lease_id=NULL, attempts=attempts+1
   WHERE state IN ('offered','leased','running') AND lease_expires_at < now()`.
- **Claim:** `SELECT ... FROM tasks WHERE job_id=$1 AND state='pending' ORDER BY seq FOR UPDATE SKIP LOCKED LIMIT $n` inside one transaction that also writes the lease. Two schedulers cannot hand the same task to two hosts.
- **Idempotency:** the terminal write is `UPDATE tasks SET state='succeeded', ... WHERE id=$1 AND state <> 'succeeded'`. A late result from a superseded attempt matches zero rows and is recorded as a `duplicate_result` run event rather than an error. Every attempt is preserved in `task_attempts`, so the run map can show "ran twice, first accepted".
- **Cancellation:** sets `cancel_requested_at`, which (a) stops new leases, (b) sends `task.cancel` to holders. Already-succeeded results are kept and the job ends `cancelled_partial` with an explicit accepted-item count. Partial results are surfaced, never silently discarded.
- **Max attempts:** 3, then `failed` with an `error_class`. A task that fails on two different hosts is flagged as an input problem rather than a host problem.

---

## 6. Adapter interface and the two adapters

Each adapter is two halves: a **planner** on the control service and an **executor** in the agent. They share a type from `protocol/`, which is what keeps them honest.

```ts
// agent side
interface WorkloadExecutor<I, O> {
  readonly type: string
  readonly version: string
  probe(): Promise<CapabilityRecord>                       // at connect + on change
  calibrate?(spec: CalibSpec): Promise<ThroughputSample>   // { itemsPerSec, setupMs }
  execute(input: I, ctx: ExecCtx): Promise<O>
}

interface ExecCtx {
  signal: AbortSignal                  // fires on cancel, lease loss, or pause
  onProgress(done: number, total: number): void
  artifacts: { fetch(sha256: string): Promise<string> }   // verifies hash before returning path
  tmpdir: string                       // removed on completion
  limits: { cpuQuotaPct: number; memLimitMb: number; wallClockMs: number }
}

// control side
interface WorkloadPlanner<S, I> {
  readonly type: string
  validate(spec: S): Result<ValidatedSpec>
  eligible(host: HostRecord, spec: ValidatedSpec): Eligibility   // { ok } | { ok: false, reason }
  partition(spec: ValidatedSpec): I[]
}
```

`Eligibility` returns a **reason** on rejection, and the reasons are shown in the "New work" screen before launch. A scheduler that silently excludes a host is indistinguishable from a broken one.

### 6.1 `cpu_inference_batch`

- **Spec:** `{ modelHash, preprocessingId, inputManifestHash, batchSize, perItemTimeoutMs, tolerance }`
- **Eligibility:** `allow_compute ∧ ¬paused ∧ online ∧ os/arch supported by onnxruntime-node ∧ free_ram ≥ model_footprint × 1.5 ∧ concurrency < max_concurrency ∧ user under quota`
- **Execution:** fetch model + inputs by hash (verify SHA-256 before use — a mismatch is a hard failure, never a warning), create an ORT session with `intraOpNumThreads` bounded by `cpu_quota_pct`, run assigned items, emit one result per input ID.
- **Output:** `{ inputId, prediction, logitsHash, ms }[]` plus per-task timing.
- **Correctness:** the same input on two hosts must agree within a documented tolerance. Floating-point CPU kernels are not bit-identical across microarchitectures, so the gate is `max |Δlogit| ≤ tol` with `tol` published, not `==`. **[A: tol to be measured in Pass 2, not guessed.]**
- **Hard rule:** one model request is never split across machines. Parallelism is across independent items only.

### 6.2 `remote_browser_session`

- **Spec:** `{ hostId, startUrl, allowlistId, viewport, ttlSeconds }`
- **Eligibility:** `allow_browser ∧ trust_tier = 'trusted' ∧ ¬paused ∧ online ∧ no active session ∧ startUrl ∈ allowlist ∧ requester may use this host`
- **Execution:** launch an agent-local enforcing proxy (§9.5), launch Chromium with `--proxy-server=127.0.0.1:<port>` and a fresh `userDataDir` under `tmpdir`, create a new context, navigate, start a screencast.
- **Lifecycle:** **pinned to one host**. Disconnect ends the session, closes the context, and deletes the profile directory. No cookie or live-state migration between machines — that is the boundary that keeps the trust model coherent.
- **Output:** frames (never persisted), an append-only session event log, and the observed egress address.

---

## 7. Scheduling

### 7.1 Filters

Eligibility (§6) is evaluated first and produces a per-host reason string. Only surviving hosts enter placement.

### 7.2 Objective

Minimize **makespan** — the finish time of the slowest host — not average utilization and not cost. For host `h` given `n_h` items:

```
T_h = q_h + s_h·(1 − cached_h) + n_h / r_h
```

- `q_h` — estimated drain time of work already queued on `h`
- `s_h` — setup + transfer: `model_bytes / measured_bandwidth + warmup_ms`
- `r_h` — measured items/sec from `calibrations`
- `cached_h` — 1 if the host already holds this `model_hash`

### 7.3 Allocation — water-filling with a tail pool

Binary-search the makespan `T` such that `Σ_h max(0, r_h·(T − q_h − s_h)) = N`. Each host gets `n_h = max(0, r_h·(T − q_h − s_h))`, rounded. A host whose `n_h` computes to ≤ 0 is **excluded** — meaning its setup cost exceeds any contribution it could make, which is a real and common outcome for a cold slow host on a small batch, and the UI says so.

**Hold back 10% of items as a tail pool**, dispatched on request as hosts finish. This is the handoff's "conservative reserve" made concrete: it absorbs calibration error and turns a host going offline near the end into a small tail rather than a stranded block.

**Calibration cost control:** calibration is itself overhead. Results are cached per `(host, adapter, model_hash, agent_version)` with a 24 h TTL, and a calibration batch is capped at `min(64 items, 2 s)`. Recalibrating on every job would destroy exactly the small-batch case that is already marginal.

### 7.4 Worked two-host example — *illustrative rates, to be replaced by measured ones*

`N = 1000`. Host A: `r_A = 42` items/s, model cached so `s_A = 0`, `q_A = 0`. Host B: `r_B = 17` items/s, cold so `s_B = 6.8` s, `q_B = 0`.

```
42·T + 17·(T − 6.8) = 1000
59·T = 1115.6      →  T = 18.9 s
n_A = 42 × 18.9 = 794      n_B = 17 × (18.9 − 6.8) = 206
```

Single-host baseline on the faster host: `1000 / 42 = 23.8 s`. Predicted speedup **1.26×**.

**The batch-size floor.** Generalizing with `q = 0`, `s_A = 0`:

```
T = (N + r_B·s_B) / (r_A + r_B)
S = (N/r_A) / T = N·(r_A + r_B) / (r_A·(N + r_B·s_B))
```

`S` rises with `N` toward the ceiling `(r_A + r_B)/r_A = 1.405`. Solving `S ≥ 1.2` for these rates gives **N ≥ 677**.

This is the most important number in this document. It means:

- The 1.2× stretch target is **unreachable on any batch smaller than ~680 items** with this host pair, no matter how good the scheduler is.
- The demo dataset must be sized from measured `r_A`, `r_B`, `s_B` **before** the benchmark is run, not chosen for convenience and explained afterwards.
- If the real measured ceiling `(r_A + r_B)/r_A` is itself below 1.2 — which happens whenever one host is more than 5× slower than the other — the target is unreachable at **any** batch size, and the correct response is to report that, not to hunt for a flattering configuration.

### 7.5 Browser placement

Exactly one eligible trusted host, chosen explicitly by the user or by first-fit among eligible hosts. Exclusive control for the session owner for the session's lifetime. No migration.

---

## 8. Authorization and host-consent matrix

**Roles:** `admin`, `member`. **Relationships:** a host has one `owner`; MVP hosts are shared within a single team.

| Action | Host owner | Other member | Admin | Also requires |
| --- | --- | --- | --- | --- |
| Enroll a host | ✅ (their own) | ✅ (their own) | ✅ | Valid pairing code |
| Pause / resume host | ✅ | ❌ | ✅ | — |
| Revoke host | ✅ | ❌ | ✅ | — |
| Change host consent flags/caps | ✅ | ❌ | ❌ | — |
| Submit compute job (auto-place) | ✅ | ✅ | ✅ | Under user quota |
| Target a specific host for compute | ✅ | ✅ | ✅ | `allow_compute ∧ ¬paused` |
| Open browser session on host | ✅ | ✅ | ✅ | `allow_browser ∧ trust_tier='trusted' ∧ URL ∈ allowlist` |
| View frames / send input | Session owner only | ❌ | ❌ | Valid session control token |
| View run results | Job owner | ❌ | ✅ | — |
| View fleet metrics | ✅ | ✅ | ✅ | — |

Two checks are worth stating because they are the ones usually missed: **the host owner does not automatically get to watch another user's browser session on their own machine** through the dashboard (they can inspect the machine directly — that is the unavoidable trust boundary, but the platform does not hand them a viewer), and **every frame/input message is re-authorized against the session token on each message**, not only at socket open.

---

## 9. Threat model

### 9.1 Malicious or curious host owner
Can read the remote browser's memory, profile directory, and network traffic; can install a root CA and inspect TLS; can tamper with results.
**Mitigations:** `trust_tier`, browser sessions restricted to trusted hosts, test accounts and team-owned sites only, prominent "Running on Host B — its owner can see this session" labeling. Result tampering is detectable only for verifiable workloads (agreement across hosts on overlapping items — the tail pool gives a few of these for free).
**Residual risk: accepted and documented.** A host owner is inside their own machine. No MVP control changes that.

### 9.2 Malicious job submitter
Wants the host as an open proxy, a LAN scanner, or a code-execution surface.
**Mitigations:** no arbitrary code or shell — only the two adapters; URL allowlist; private-range blocking (§9.5); per-user quotas on items, concurrent jobs, and session minutes; wall-clock caps per task.

### 9.3 Platform operator
Sees relayed frames, input keystrokes, job inputs and results, and all metadata.
**Mitigations:** frames are relayed, never written to disk; `run_events` excludes page content by default; log retention is bounded. **No end-to-end encryption claim is made.** The UI states that the operator can see the session. E2E media would require the control service to relay opaque bytes it cannot decode — a genuine possibility with WebRTC + insertable streams later, and explicitly not a Pass 1–5 property.

### 9.4 Target sites
See automated traffic from the host's egress. **Mitigation:** allowlist of team-operated domains only; robots/ToS respected; no marketing of geo-restriction bypass.

### 9.5 Accidental LAN and metadata access — the sharpest edge

A URL allowlist alone is **not sufficient**, because DNS rebinding lets an allowlisted hostname resolve to `192.168.1.1` on the second lookup, and redirects can walk out of the allowlist. Enforcement therefore sits at an **agent-local HTTP/HTTPS proxy** that the remote Chromium is launched against (`--proxy-server=127.0.0.1:<ephemeral>`, bound to loopback):

1. Host must match the allowlist (exact host or explicit suffix rule).
2. Resolve the name **once**; reject unless every returned address is public.
3. **Pin** the connection to the checked IP — the browser never re-resolves for itself.
4. Deny `127.0.0.0/8`, `10/8`, `172.16/12`, `192.168/16`, `169.254/16` (including `169.254.169.254` metadata), `100.64/10` CGNAT, `::1`, `fc00::/7`, `fe80::/10`, and any non-global IPv6.
5. Re-run steps 1–4 on **every redirect hop**, with a hop cap.
6. Deny non-HTTP(S) schemes, and `file:`/`chrome:` navigation via Chromium policy flags.

Requests that fail become session events visible in the UI. The proxy is a small, fully unit-testable component, and its test suite is a Pass 3 gate — this is the one place where a quiet bug turns a friend's laptop into an attack surface against their own network.

---

## 10. Stream transport: choice, cost, and the upgrade trigger

### 10.1 Pass-3 transport — CDP screencast over the authenticated relay

`Page.startScreencast({format:'jpeg', quality:60, maxWidth:1280, maxHeight:800, everyNthFrame:2})`, frames forwarded as binary WS messages, rendered to a `<canvas>` in the dashboard. Input travels back as semantic events (`Input.dispatchMouseEvent` / `dispatchKeyEvent` through the Playwright CDP session).

**Backpressure rule — non-negotiable:** at most **one** frame in flight per session. If a frame is pending when the next arrives, the older is dropped, never queued. Unbounded frame queues are the standard way this architecture dies: latency grows without bound while throughput looks fine.

**Expected cost [A — to be measured, not assumed]:** at 1280×800 / q60, roughly 30–120 KB per frame; at ~10 fps that is 0.3–1.2 MB/s per session. Latency = `RTT(web↔control) + RTT(control↔agent) + encode + queue`.

The UI displays the **measured** refresh rate. It never implies 60 fps.

### 10.2 Concrete upgrade trigger

Move media to WebRTC (with TURN fallback) **only if**, over a 2-minute scripted interaction on the real two-host pair, any of:

- p50 input-to-visible-frame **> 250 ms**, or p95 **> 600 ms**
- sustained host uplink **> 1.5 MB/s** per session
- dropped-frame rate **> 20%**

### 10.3 What the upgrade actually costs — stated plainly

WebRTC is not a drop-in. The agent must become a media sender: the screencast is a stream of JPEGs, so it needs a real encode pipeline (VP8/H.264) and a WebRTC stack in Node, plus ICE, and a TURN server for the case where both peers are behind symmetric NAT — which relays the media anyway and returns the operator to the plaintext position. That is why it is conditional on measurement rather than planned. [MDN WebRTC protocols](https://developer.mozilla.org/en-US/docs/Web/API/WebRTC_API/Protocols) **[D]**

---

## 11. Test protocol and phase gates

**Environment record for every run:** OS, CPU model, logical cores, RAM, network type and measured up/down, agent version, model hash, manifest hash, item count, cache state (cold/warm), timestamp, and the git SHA of all three packages.

**Protocol:** ≥ 3 warm trials per configuration; report **median and full range**, never a single best run. Baseline is the **faster single eligible host** on identical inputs, model, and preprocessing. Speedup = `single-host wall-clock / distributed wall-clock`. Values < 1 are reported as-is. The harness (`bench/`) writes one JSONL record per trial and the run-history screen renders exactly those records — the UI cannot show a number the harness did not produce.

| Gate | Pass criterion | Evidence artifact |
| --- | --- | --- |
| `network` | Two hosts on distinct networks enroll and execute with outbound-only connections; no port forwarding | Traceroute/IP evidence from both, socket listing showing no inbound listener |
| `distribution` | Results carry valid host signatures from two distinct public keys (§12.5) | Signature verification report |
| `correctness` | Every input ID has exactly one accepted result; cross-host agreement within published tolerance | Diff report + measured tolerance |
| `recovery` | Killing a host mid-batch re-queues only unfinished leases; no duplicate accepted results; browser session ends visibly | Event log excerpt + task_attempts table |
| `browser-egress` | Team-owned `/whoami` observes the **host's** egress, not the control service's | Screenshot pairing dashboard label with site response |
| `security` | Revoked host rejected; non-owner cannot control a session; pause works offline; private-IP and off-allowlist targets rejected | Proxy test suite + manual matrix |
| `usability` | A fresh host is enrolled end-to-end from the README with no manual DB edits | Recorded walkthrough |
| `performance` | Measured speedup published with median and range; speedup claimed only if median > 1.0 | bench JSONL + run-history screenshot |

---

## 12. Top five failure modes

**12.1 Duplicate or lost results around lease expiry.** A host finishes as its lease expires; the task is already re-queued; two results arrive.
*Remedy:* conditional terminal `UPDATE`, all attempts retained, late results recorded as `duplicate_result`. *Test:* `SIGKILL` an agent at 60% of a batch and assert `accepted == total` with `attempts > total`.

**12.2 Distributed is slower than one host.** Setup and transfer dominate; the water-fill excludes a host or the speedup lands below 1.
*Remedy:* the §7.4 floor is computed and **shown before launch** — "with your current hosts, this batch needs ≥ 680 items to beat Host A." The run map always shows the overhead breakdown. A result below 1× is a finding, not a bug to be hidden.

**12.3 Media backpressure collapse.** Frames queue, latency climbs, the session becomes unusable while metrics look healthy.
*Remedy:* 1-frame-in-flight cap, drop-oldest, adaptive quality, hard session TTL, and latency measured end-to-end rather than at the encoder.

**12.4 SSRF / LAN reach from the remote browser.** A redirect or DNS rebind points the browser at the owner's router or a metadata endpoint.
*Remedy:* the §9.5 enforcing proxy with IP pinning and per-hop revalidation; its unit tests are a release gate.

**12.5 Fake distribution — the demo-integrity failure.** The most damaging outcome is a demo that *looks* distributed but isn't, and the control service's own logs cannot disprove it because the control service writes them.
*Remedy:* each result carries `sign(host_privkey, {taskId, attempt, outputHash, startedAt, finishedAt, hostId})`, stored in `task_attempts.host_signature`. `scripts/verify-run.ts` takes a job ID and the enrolled public keys and verifies every accepted result independently of the server's narrative. The run map shows a per-task verification badge. A reviewer who does not trust the operator can still check that two distinct machines did the work.

---

## 13. Prioritized backlog with acceptance criteria

### Pass 1 — Network spike (Gate A)
| # | Item | Acceptance |
| --- | --- | --- |
| 1.1 | `protocol` package: envelope, `hello`, `heartbeat`, `echo` task | Typecheck passes; schemas round-trip in tests |
| 1.2 | Control service: HTTPS + WSS, deployed publicly with a real certificate | `wss://` reachable from both networks |
| 1.3 | Pairing: code issue, keypair generation, `POST /hosts/pair` | Code is single-use and expires; key file is 0600 or the agent refuses to start |
| 1.4 | Agent transport: EdDSA assertion, reconnect with jitter, heartbeat | Kill the network for 60 s; agent reconnects without a restart |
| 1.5 | `echo` adapter + dispatch | Each host returns its own `hostId` + server nonce; `distribution` gate passes |
| 1.6 | Presence | Disconnect shows `offline` within 45 s |
| 1.7 | `/whoami` demo site + first Chromium launch on the trusted host | Site logs the **host's** egress IP; no listening debug port (asserted, not assumed) |

### Pass 2 — Durable batch compute
2.1 schema + migrations · 2.2 `SKIP LOCKED` claim and lease sweeper · 2.3 `cpu_inference_batch` executor with hash verification · 2.4 manifest + pinned fixtures · 2.5 calibration with caching and TTL · 2.6 water-fill placement + tail pool · 2.7 host-signed attestation and `verify-run` · 2.8 bench harness and baseline.
**Acceptance:** 1000 items across two real hosts; exactly one accepted result per input; cross-host agreement within published tolerance; `SIGKILL` recovery clean; median speedup published with range.

### Pass 3 — Browser workload
3.1 enforcing proxy + its test suite · 3.2 ephemeral context lifecycle and profile deletion · 3.3 screencast relay with backpressure · 3.4 input forwarding with per-message authorization · 3.5 one scripted workflow with pause/human-takeover/resume · 3.6 session TTL, stop, and disconnect semantics.
**Acceptance:** `browser-egress` and `security` gates pass; measured latency recorded against the §10.2 trigger; profile directory provably gone after stop.

### Pass 4 — Interface
Fleet · New work (with eligibility reasons and the batch-size floor shown pre-launch) · Run map (per-task host, timing, verification badge, overhead breakdown) · Live view (host label, egress, measured fps, control owner) · History (manifest hashes, baseline run ID, methodology).
**Acceptance:** the entire demo runs with no DB edits and no impersonated host events.

### Pass 5 — Hardening and rehearsal
Caps enforcement · revocation · cancellation with partial results · offline-capable local pause · quotas · README from-scratch walkthrough · gate table green · every number in the UI traceable to a bench record.

---

## 14. Go / no-go and unresolved decisions

**Recommendation: GO for Pass 1.** The topology is sound, the stack is available on all three target OSes, and the risky parts are isolated behind measurement gates rather than assumed away.

**Two decisions gate Pass 1 and cannot be assumed:**

1. **Where does the control service run publicly?** It needs a real TLS certificate and a stable hostname reachable from both home networks. A small managed host is the straightforward answer; a tunnel is acceptable for the spike but must not become the architecture.
2. **Which two physical machines, on which two networks?** The handoff is explicit that a VM cannot substitute for the cross-network proof. If only one network is available today, Pass 1 stalls at `network` regardless of code quality.

**Decisions deferred with a named trigger, not left open:**

| Decision | Trigger |
| --- | --- |
| WebRTC + TURN for media | §10.2 thresholds exceeded on real hardware |
| TypeScript 5.9 fallback | Any dependency's type tooling fails under TS 7 in Pass 1 |
| Keychain/DPAPI key storage | Pass 5 hardening |
| Result upload path for >256 KiB outputs | First adapter whose outputs exceed the cap |
| GPU execution providers | After Pass 2 publishes CPU numbers |

**Explicit non-goals, unchanged from the handoff:** untrusted public hosts, arbitrary code or shell execution, real logged-in accounts on stranger hosts, mobile host control, cross-chip translation, marketplace payments, and any end-to-end media confidentiality claim.
