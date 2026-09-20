# What a compute machine reports

**For:** whoever builds task allocation.
**Companion to:** `docs/task-allocation.md`, which covers where allocation code goes.
**Written:** 2026-09-19.

Everything here already crosses the wire. Nothing in this document needs an agent change
or a protocol version bump — it is a description of what arrives today.

Until 2026-09-19 the server kept only the adapter list and discarded the rest. It now
persists the hardware facts too (`dwp.py`, `hello()`), so the fields marked **stored**
below are readable from `workers.capabilities` right now.

## The four moments data arrives

| moment | how often | carries |
| --- | --- | --- |
| pairing | once, ever | public key, display name |
| `hello` | every reconnect | hardware + consent (the table below) |
| `heartbeat` | every 5s | live memory, running count |
| task result | per task | timings + a signed attestation |

A reconnect happens on every network blip, sleep/wake and update-restart, so `hello`
data is fresh in practice — but it is a *snapshot at connect*, not a live feed.

## `hello` — hardware

Source: `packages/agent/src/capability.ts`, schema `packages/protocol/src/messages.ts:24`.

| field | type | example | stored | notes |
| --- | --- | --- | --- | --- |
| `adapters` | string[] | `["echo","walker_evolution"]` | ✅ as `kinds` | what it can run. The eligibility filter |
| `agentVersion` | string | `"0.4.0"` | ✅ | pair with any timing you record — see traps |
| `os` | enum | `win32` \| `darwin` \| `linux` \| `ios` \| `android` | ✅ | |
| `arch` | string | `"x64"`, `"arm64"` | ✅ | |
| `cpuModel` | string | `"AMD Ryzen 5 5600U with Radeon Graphics     "` | ✅ | vendor string, padded; server strips it |
| `logicalCores` | int | `12` | ✅ | **not** usable parallelism — see `maxConcurrency` |
| `totalRamMb` | int | `15000` | ✅ | the machine's size |
| `freeRamMb` | int | `8000` | ❌ transient | resent every heartbeat; see below |

## `hello` — consent

Source: `ConsentState`, `packages/protocol/src/messages.ts:37`.

| field | type | stored | notes |
| --- | --- | --- | --- |
| `maxConcurrency` | int | ✅ | **how many tasks the owner agreed to run at once.** The number to respect, not `logicalCores` |
| `paused` | bool | ✅ as `workers.paused` | owner's kill switch; works offline |
| `allowCompute` | bool | ✅ folded into `paused` | withdrawal of consent |
| `allowBrowser` | bool | ❌ | browser workloads are not offered to devices |

Consent is re-sent as `consent.update` the moment it changes, not only at connect.

**A phone uses this to withdraw itself.** When `fitForWork` flips — too hot, Low Power
Mode, battery too low — iOS sends `consent.update` rather than declining offers one by
one (`Transport.swift:269`). The reasoning in that comment is worth reading before
designing any allocator: declining is the wrong mechanism, because the server releases
the task and re-offers it immediately, so an unfit device spins in a decline loop that
hammers the control service and burns exactly the battery Low Power Mode was trying to
save. **So a hot phone already looks paused to the scheduler** — the self-gating reaches
the server even though the thermal reading does not.

## Phones only — sent in the **heartbeat**, every 5s

The TypeScript schema allows a `mobile` block on `CapabilityRecord` (`messages.ts:13`),
but the iOS agent does not put it there: it attaches it to every heartbeat instead
(`Transport.swift:278`), which is right, because thermal state and battery change between
connections and `hello` only fires on reconnect.

**The server currently discards it.** `Heartbeat` in `dwp.py` models only `freeRamMb` and
`running`, and pydantic ignores the rest — so the most actionable data in the protocol
arrives every five seconds and is thrown away:

| field | type | why it matters |
| --- | --- | --- |
| `fitForWork` | bool | the device's own verdict. Prefer it to re-deriving policy |
| `thermal` | enum | `nominal` → `critical`. A throttling phone is slow, not broken |
| `lowPowerMode` | bool | the OS will throttle regardless of what you schedule |
| `batteryLevel` | 0–1 | optional |
| `charging` | bool | optional |
| `availableMemoryMb` | int | |

## `heartbeat` — every 5 seconds

| field | type | notes |
| --- | --- | --- |
| `freeRamMb` | number | live. Answers "can this task start", not "how big is this machine" |
| `running` | int | tasks in flight on that machine right now |

Not persisted today — used only for presence. **This is the natural place to hang live
scheduling state**, and it already arrives at exactly the cadence a scheduler wants.

## Task results — the only true measure of speed

Per task: `startedAt`, `finishedAt`, `hostReportedMs`, `outputHash`, and a `signature`
over `(taskId, attempt, hostId, outputHash, startedAt, finishedAt)` made with a key the
server has never held. Stored as `tasks.attestation`, alongside `tasks.started_at`.

**Nothing derives a throughput figure from these.** That is the one input an allocator
needs that does not exist yet, and the raw material is already in the table. A rolling
median of `hostReportedMs` per `(worker, kind)` would do it.

## Reading it

```sql
SELECT id,
       capabilities->'machine'->>'cpu_model'          AS cpu,
       (capabilities->'machine'->>'logical_cores')::int   AS cores,
       (capabilities->'machine'->>'total_ram_mb')::int    AS ram_mb,
       (capabilities->'machine'->>'max_concurrency')::int AS concurrency,
       capabilities->'kinds'                          AS kinds,
       state, paused, last_seen
FROM workers
WHERE state = 'alive' AND NOT paused;
```

`machine` is `NULL` for workers that registered before this existed, and for native
(non-app) workers configured from environment variables. **Always handle the null** — a
machine with no specs must still be schedulable, just not preferentially.

## Five traps

1. **`cpuModel` is a marketing string, not a speed.** Measure; do not parse.
2. **The runtime matters more than the CPU.** Identical deterministic arithmetic measured
   ~3.5x apart between Node/V8 and Bun/JSC. Store `agentVersion` next to any timing, or
   you will learn a JavaScript engine and call it a processor.
3. **`logicalCores` ≠ parallelism.** `maxConcurrency` is the consented cap and is
   deliberately lower on machines someone is using.
4. **`freeRamMb` falls under load** — it is a moment, not a property.
5. **`runtime` is hard-coded `cpu` and `vram_mib` to `0`** for app devices, honestly: no
   device path dispatches to a GPU. Measured, the GPU backends were *slower* for models
   this size — CoreML's GPU path at 0.86x of plain CPU, WebGPU at 0.23x. Treat GPU as a
   property to match against task size, never as a tier to prefer.

## Different platforms, same field names — the trap that is already live

This is not a future problem. Three implementations exist today (TypeScript desktop,
Swift `DWPAgentKit`, and a stub for Android that has never been written), and the schema
makes **every** hardware field required — `packages/protocol/src/messages.ts:24`, where
only `mobile` is optional. A platform that cannot answer honestly must therefore answer
anyway.

The danger is not that platforms send *different* fields. It is that they send the *same
field name carrying a different quantity*, which no type checker and no schema can catch:

| field | macOS / Windows | iOS |
| --- | --- | --- |
| `cpuModel` | `"Apple M5 Pro"`, `"AMD Ryzen 5 5600U with Radeon Graphics"` — a processor | `"iPhone17,1"` — a *device model*, not a processor |
| `freeRamMb` | free system memory | `os_proc_available_memory()` — this **process's** jetsam budget, ~4 GB on an 8 GB phone. Nothing to do with free RAM |
| `agentVersion` | `"0.4.0"` | `"0.4.0-ios"` — deliberately suffixed, so string equality fails |
| `logicalCores` | cores the scheduler may use | cores the OS will *throttle* under thermal pressure |

`Capability.swift` is explicit about the memory one, and right to be: reporting
device-wide free memory "would tell the scheduler a number that has almost nothing to do
with whether the next slice fits". The iOS behaviour is correct. What is wrong is that
both quantities travel under one name.

An allocator that ranks machines by `freeRamMb` today is comparing a desktop's spare
memory against a phone's allocation budget and will prefer the phone.

### What to do about it, before Android exists

There is no Android implementation yet. That is the opportunity: a third implementation
will cement whatever convention it finds, so the convention is worth fixing first.

1. **Make platform-dependent fields optional, and mean it.** `null` is honest; a
   fabricated value is not. A scheduler can handle "unknown" — it cannot detect a lie.
2. **Rename to the quantity, not the concept.** `freeRamMb` should be
   `allocatableMemoryMb` if that is what every platform can actually answer, or it should
   be split. Either is better than one name for two things.
3. **Namespace anything platform-specific**, the way `mobile` already is. That block is
   the right pattern: a phone adds thermal state and battery, a desktop omits it, and the
   scheduler may use or ignore it without either side pretending.
4. **Prefer the device's own verdict for anything policy-shaped.** `mobile.fitForWork`
   is the best-designed field in the protocol: each platform answers *"should I be given
   work right now?"* in its own terms, and the scheduler gets one comparable boolean
   instead of re-deriving five platform policies it cannot test.
5. **Prefer measured throughput over any of it.** It is the one signal that is
   platform-neutral by construction, because it measures work completed rather than
   hardware described. A phone and a desktop that both finish 40 walker evaluations a
   second are, for scheduling purposes, the same machine.

The general rule: **comparable things in shared fields, platform truths in namespaced
blocks, and a per-platform verdict where the platform knows best.**

## Choosing a machine for simulation, rendering, or ML

Every allocation decision is three questions, and they need different data. The schema
today answers the first partly, the second not at all, and the third barely.

1. **Can it run at all?** Hard gates — adapter, RAM, disk, VRAM, OS/arch. Binary. Get one
   wrong and the task fails rather than runs slowly.
2. **How fast will it be?** Measured throughput. Not derivable from any spec.
3. **Should it, right now?** Live state — busy, hot, on battery, about to sleep.

### Simulation — what this fleet runs today

Bottleneck is scalar floating point times concurrency. Inputs and outputs are bytes,
duration is seconds, and nothing is gated on memory.

**Needs nothing new.** Measured evals/sec per machine, concurrency headroom, and CPU busy%
cover it — and the first is already sitting in `tasks.attestation`, underived.

### Rendering

Different in kind, not degree: the unit of work is minutes to hours, the scene is often
gigabytes, and the machine is occupied wholly rather than partly.

- **Hard gates:** RAM to hold the scene, **free disk to stage assets**, VRAM if the
  renderer is GPU-based. None of these are reported today.
- **Dominant cost is usually data movement, not compute.** Shipping 4 GB of assets to a
  laptop on campus wifi can exceed the render itself.
- **Sustained availability matters more than speed.** A laptop that suspends every twenty
  minutes should never be given a forty-minute frame. The agent already detects this
  (`machine.woke`, `suspendedForMs`) and never reports it.

### Machine learning

- **Inference:** batch size times model size against available memory; on CPU the
  precision support and matrix units decide, not the core count. This fleet's measured
  example: CoreML's CPU kernels beat ONNX Runtime's by 1.18x on identical arithmetic.
- **Training:** VRAM is the gate, and nothing else matters until it is satisfied. Then
  memory bandwidth, then supported precisions (fp16/bf16/int8).
- **Weights are gigabytes.** Which machine already holds them usually decides placement.

### The signal that would change scheduling most, and is free

**Artifact locality.** The agent already caches every artifact by SHA-256 under
`~/.dwp/artifacts` and checks `existsSync` before downloading
(`packages/agent/src/adapters/inference.ts:119`) — and it never tells the server what it
holds.

For a 5 GB model or a 2 GB scene, *"who already has this?"* beats every hardware fact on
this page. A machine half as fast that already has the weights will finish first, and the
scheduler cannot currently tell. Reporting held hashes in the heartbeat — a list while it
is small, a digest once it is not — costs nothing and is specific to work this project
already does.

### What to add, in order, and only when a workload needs it

1. **Artifact inventory** — free, and the largest effect for anything with big inputs.
2. **Free disk** — free (`statfsSync`, verified working in the shipped binary). A staging
   gate: rendering and ML fail late and expensively without it.
3. **Sustained-availability score** — free, from uptime plus the suspension history the
   agent already computes. The gate for long jobs.
4. **Real GPU model and VRAM** — costs native code. Only once GPU work exists.
5. **Supported precisions** — ML only, and only for training.

### Do not add GPU fields speculatively

Measured on this project: CoreML's GPU path ran at **0.86x** of plain CPU and WebGPU at
**0.23x**, because a small model spends longer on dispatch than on arithmetic. A `vram_mib`
field that nothing consumes is a field that invites an allocator to prefer hardware which,
for the tasks this fleet actually runs, is slower. Add it with the workload, not before.

## What else is available, ranked

The list above is what arrives today, not the limit of what could. Everything in this
section was **measured on 2026-09-19 inside a compiled Bun binary** — the artefact that
actually ships — not read from documentation. Anything the shipped binary cannot do is
marked as such.

Ordered by what a scheduler would gain, not by what is easy.

### Tier 1 — free, and the biggest gaps

No new dependency, no native code, works in the binary. `capability.ts` and the heartbeat
are the only files that change.

| signal | source | cost | why it matters |
| --- | --- | --- | --- |
| **CPU busy %** | `os.cpus()[].times` | two samples, ~1s apart | The single largest blind spot. Today a machine already pinned at 100% by its owner's work looks identical to an idle one. Belongs in the **heartbeat**, since it needs two samples to mean anything |
| **observed throughput** | `tasks.attestation` | none — already stored | Rolling median `hostReportedMs` per `(worker, kind)`. Beats every static spec below, because it measures the thing you actually care about |
| **load average** | `os.loadavg()` | free | Confirmed working: `[3.22, 2.81, 3.86]`. **Returns zeros on Windows** — treat absent, not idle |
| **network RTT** | `ws.ping()`/`pong` | free | Already round-tripping every heartbeat and discarded. A machine 400ms away should not get short tasks |
| **dial latency** | `dialMs` | free | Already computed at `connect.established` and only logged |
| **recent suspensions** | heartbeat drift | free | Already computed — `machine.woke` with `suspendedForMs`. A laptop that sleeps every 20 minutes is a bad home for a long task, and the agent already knows |

### Tier 2 — free, smaller wins

| signal | source | notes |
| --- | --- | --- |
| CPU MHz | `os.cpus()[0].speed` | `2400` on Apple Silicon — nonzero, so usable. Zero on some platforms; treat 0 as unknown |
| per-core detail | `os.cpus()` | full array, not just `[0]`. Distinguishes performance from efficiency cores |
| disk free | `fs.statfsSync()` | confirmed: `13 GB free of 926 GB`. Matters once tasks stage artefacts |
| uptime | `os.uptime()` | proxy for stability |
| agent's own CPU/RSS | `process.resourceUsage()` | how much the agent itself is costing its host |

### Tier 3 — costs a subprocess or native code

| signal | why it is not free |
| --- | --- |
| battery level / on mains | `pmset -g batt` (macOS), WMI (Windows), `/sys/class/power_supply` (Linux). Three implementations, one spawn each |
| thermal state on desktop | Not exposed by any runtime API. Phones already report it; desktops cannot without platform code |
| GPU model / VRAM | Needs native bindings or a spawn. And see the trap below before wanting it |
| real network throughput | Only knowable by transferring — the release download already measures it incidentally |

### What to do first

**Derive throughput from the attestations you already store.** It requires no agent
change, no protocol negotiation, and no iOS/Android port — and it is a better predictor
than every static field combined, because it measures the machine doing the actual work
rather than describing the machine.

Then add **CPU busy %** to the heartbeat, because "how fast is this machine" and "is it
free right now" are different questions and nothing currently answers the second.

### The cost of the rest

Every new field in `CapabilityRecord` is a protocol negotiation across three
implementations — the TypeScript agent, the Swift `DWPAgentKit`, and Android — and
`.claude/skills/port-agent-change/SKILL.md` exists because that has already bitten this
project once. Adding an optional field is backward compatible in both directions
(zod ignores absent optionals, the server's `Capability` model ignores unknown keys —
verified), so the cost is coordination, not breakage. Add a field when a scheduling
decision turns on it; not before.
