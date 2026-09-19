# [TEAM NAME] — Team Handbook

**Rent GPUs from anyone. An agent pools them and makes them fast.**

This is the only document. Everything you need is here.

| Part | Who reads it | What's in it |
|---|---|---|
| **0. Start here** | All three | What we're building, the contracts, the three blockers |
| **1. Jack** | Jack | Network + sharing marketplace |
| **2. Ethan** | Ethan | Training execution engine |
| **3. Ji** | Ji | Agents + optimization + dashboard |
| **4. Integration** | All three | Merge checkpoints, what to cut |
| **5. Demo** | All three (presenter especially) | Stage plan, fallbacks, Q&A |
| **A. Optimization levers** | Ji | Full rule tables and measurement design |

**Two things still undecided — settle them today:** the team name, and whether renting is credits or real money. This doc assumes **credits** (see below).

---
---

# PART 0 — START HERE

## What we're building

**A marketplace where people lend and borrow each other's idle GPUs — plus an agent that pools whatever you borrowed and runs it well.**

Two layers, and the fact that they're joined is the whole point.

```
Vast.ai            sells you chips.       Pooling them is your problem.
Ray / torchft      pools chips for you.   Finding them is your problem.
Us                 both — and the agent decides what to rent in the first place.
```

Minji has no GPU. She borrows from three friends. Then there's nothing left for her to do: the agent pools them, picks the settings, and moves the job when a cheaper chip frees up. Junho is on the other side, earning credits from a 4090 that sits idle while he's in class.

## The three mechanisms

Twelve features on the pitch deck. Underneath, **three mechanisms:**

1. **Acquiring chips** — who's in your pool (rent, return, credits)
2. **Changing the worker set** — who's in the pod at a sync boundary (join, leave, migrate, drop a straggler)
3. **Changing the config** — probe, measure, re-pick parameters

Mechanisms 1 and 2 are nearly the same code — renting a chip and a worker joining the pod are the same event at different scopes. Every optimization lever is mechanism 3.

**And that maps one-to-one onto the agent tiers:**

```
Router          decides the worker set  → mechanisms 1 and 2
Chip agents     decide the config       → mechanism 3
```

Keep this in mind and the codebase stays small.

## The agent, in two tiers

This is the differentiator, so be precise about it.

```
Router  (sees the whole pool)
  · which chips to rent — cheapest that's sufficient, old generations included
  · when and where to migrate
  · who to drop for being slow
  · H, the sync interval — depends on measured network conditions
        ↓ delegates
Chip agents  (one per chip type)
  Ampere24GB (3090) · Ada24GB (4090) · MI250 · V100 · Default
  · dtype, batch size, attention, checkpointing on this chip
  · "I know the 3090"
```

**Why splitting by chip is defensible:** chip knowledge is empirical and doesn't transfer. What works on a 3090 (24GB, Ampere, no NVLink) isn't what works on an MI250, and you can't derive it from spec sheets — you have to run it. **This is also exactly how the data moat works:** run history accumulates per chip type, and the 3090 agent only gets smarter from 3090 data. A new chip shows up, a new agent starts from defaults and learns.

**In P0 these are rule-based Python classes.** That's fine — but build the *structure* for real (Part 3, S-4), because "we have a dedicated agent per chip" has to be true in the code or it backfires when a judge reads it.

## Core design rule

| Situation | What we do |
|---|---|
| Same vendor (NVIDIA↔NVIDIA, AMD↔AMD) | **Run together** — distributed training + inference |
| Different vendor (NVIDIA↔AMD) | **Migration only** — move the whole job |

We do not try to weave CUDA and ROCm together inside one training run. Drawing this line is what makes the project buildable.

**Related, and say it this way:** we don't "convert code." The same PyTorch code already runs on both vendors (CUDA build and ROCm build). **The only thing that moves is a weights file.** If you say "the agent migrates the code," a judge hears "you transpile CUDA kernels to ROCm," which isn't true and you lose the room.

## Credits, not money (assumed)

Renting settles in **credits**: contribute GPU-hours, spend them later. No payouts, no disputes, no KYC. "Lend it when idle, borrow it when you need it."

If you decide on real money instead, tell everyone before hour 3 — it adds settlement, trust scores, and dispute handling, which is out of scope for a hackathon. **A judge will ask. Have the answer ready.**

## How the work splits

| | Owns | Tickets | Load |
|---|---|---|---|
| **Jack** | Network + sharing marketplace | N-1…N-6, S-1, S-8 | ~12h |
| **Ethan** | Training execution engine | T-1…T-8 | ~13h |
| **Ji** | Agents + optimization + dashboard | S-2…S-7, S-9 | ~10.5h |

**The boundary, in one line:** Ji decides, Ethan executes, Jack connects the two.

Three rules that keep you from blocking each other:

- Ji never touches the training loop.
- Ethan never hardcodes a rule. Configs arrive as JSON and get executed.
- **There is no component called "the scheduler."** Scheduling, chip selection, rental decisions, and optimization are all one thing: **the agent** (router + chip agents). Two names means you build it twice or nobody builds it.

## Tech stack

| Layer | Choice | Fallback |
|---|---|---|
| Distributed training | torchft (DiLoCo + fault tolerance) | diloco_simple / server-mediated averaging |
| Inference | vLLM | — |
| Network | Tailscale (WireGuard overlay) | — |
| Collective backend | **gloo** (never NCCL — see below) | — |
| Migration | safetensors | — |
| Framework | PyTorch (CUDA + ROCm) | — |
| Server | FastAPI + WebSocket | — |
| Dashboard | Single HTML + WebSocket, or React + Recharts | — |
| Agents | Rule-based Python classes | Learned per-chip models in P2 |
| DB | SQLite | Postgres |

---

## HOUR 0 — Freeze the contracts (all three, 60 min)

**Do not skip this.** If you scatter without it, your three codebases won't connect and you'll discover that at hour 14. One hour here buys twelve later.

Sit at one screen, agree on the following, commit it as `contracts.py`. Changes after this need all three of you.

### Contract 1 — Worker → Server events (WebSocket JSON)

```json
{"type":"worker.register","worker_id":"w1","owner":"junho","vendor":"nvidia",
 "gpu":"RTX 3090","vram_gb":24,"cc":8.6,"chip_class":"ampere_24gb",
 "share":{"enabled":true,"not_gaming":true,"always_between":["00:00","08:00"]},
 "credits_per_hour":1.0,"ts":1737000000.0}

{"type":"worker.heartbeat","worker_id":"w1","gpu_util":0.93,"mem_used_gb":21.2,"ts":...}

{"type":"train.step","worker_id":"w1","step":1234,"loss":1.83,
 "step_time_s":0.21,"tokens":32768,"ts":...}

{"type":"train.sync","round":7,"participants":["w1","w2"],"bytes":41943040,
 "duration_s":2.1,"global_loss":1.79,"ts":...}

{"type":"worker.left","worker_id":"w1","reason":"timeout|user|game","ts":...}

{"type":"credits.update","owner":"junho","earned_gpu_hours":3.2,"balance":128.5,"ts":...}

{"type":"migration.progress","job_id":"j1","from":"w1","to":"w3",
 "phase":"waiting_sync|saving|transferring|loading|resumed","pct":0.4,"ts":...}

{"type":"probe.result","worker_id":"w1","chip_class":"ampere_24gb",
 "config_name":"baseline|optimized","t_step_median_s":2.31,
 "tokens_per_s":14200,"peak_vram_gb":9.1,"gpu_util":0.62,"t_sync_s":2.0}
```

### Contract 2 — Server → Worker commands

```json
{"type":"job.start","job_id":"j1","config":{ ...Contract 3... }}
{"type":"job.stop","job_id":"j1"}
{"type":"checkpoint.save","job_id":"j1","round":7}
{"type":"checkpoint.load","job_id":"j1","uri":"file:///.../round_7.safetensors"}
{"type":"config.update","micro_batch":32}
```

### Contract 3 — Job config (the router's output = Ethan's input)

```json
{"job_id":"j1","model":"nanogpt-124m","dtype":"bf16","attention":"sdpa",
 "micro_batch":32,"grad_accum":1,"global_batch_tokens":32768,
 "H":190,"backend":"gloo","workers":["w1","w2"],"total_steps":8000}
```

**This single JSON is the entire interface between Ji and Ethan.**

### Contract 4 — Checkpoint layout

```
ckpt/{job_id}/round_{N}.safetensors   ← weights only
ckpt/{job_id}/round_{N}.meta.json     ← {round, global_step, global_loss, config_hash, ts}
```

### Contract 5 — `chip_class` strings

Three people key on this. Agree the exact strings now.

```
ampere_24gb   (RTX 3090)
ada_24gb      (RTX 4090)
turing_16gb   (V100 etc.)
cdna_amd      (MI250, 7900)
default       (unknown — cold start)
```

Jack writes it in `worker.register` and SQLite. Ethan tags probes with it. Ji dispatches the router on it. **If these three disagree on the string, per-chip learning silently doesn't happen.**

**Done when:** all three repos import the same `contracts.py` and each of you has emitted one dummy event with it.

---

## The three blockers

Nothing else matters until these resolve. **Answers by T+3.**

| ID | Owner | What | If it fails |
|---|---|---|---|
| **N-1** | Jack | Tailscale peers report `direct`, not `relay` | Relay means 2–35 Mbit/s. Raise H, shrink the model |
| **N-2** | Jack | gloo all-reduce works machine-to-machine | Plan change — two processes on one box, or server-mediated averaging |
| **T-2a** | Ethan | torchft nightly actually runs here | Fall back to diloco_simple → T-3 grows 2h → 5h |

> **Ethan: run T-2a right now, before anything else on your list.**

**One thing to know up front: NCCL does not work over Tailscale across NAT.** Documented (NVIDIA/nccl #1606, closed "not planned"). Don't spend time on it. We use gloo, and DiLoCo's infrequent syncing is what makes that fine. Not a compromise — it's why we chose DiLoCo.

---
---

# PART 1 — JACK: Network + Sharing Marketplace

> "Connect scattered machines, make them talk, and let their owners lend them."

Jack owns **both ends of the wire**: the worker daemon on each GPU machine, and the server hub everyone connects to. Same protocol on both sides — one person writes both, or you get protocol bugs. He also owns the supply side of the marketplace: share rules, instant reclaim, and credits.

## User story

When Minji logs in she sees **her friend A's 3090, friend B's 3090, and the lab's AMD 7900 as a list, with what each costs in credits.** Three apartments, three routers. On her screen it's just a menu.

Junho installs the app, sets "only when I'm not gaming," and forgets about it. His 4090 works while he's in class and earns him credits; when he launches a game it drops out immediately. **From Junho's side nothing interrupts his game. From Minji's side one worker turns grey.** Same event, neither party inconvenienced — that's what Jack builds.

Later, Junho needs eight GPUs for his own project and spends what he banked.

## Key decisions

| Thing | Choice | Why |
|---|---|---|
| Overlay network | Tailscale (WireGuard) | Fastest path to meshing NAT'd home machines |
| Collective backend | **gloo** | NCCL fails across NAT. Don't try it |
| Server | FastAPI + WebSocket | Everything else is Python |
| Checkpoint transfer | HTTP multipart + sha256 | Simplicity wins. P2P is overkill |
| Settlement | **Credits**, not money | No payouts, no disputes, no KYC |
| DB | SQLite | Fine for now |

**Question Jack must answer on stage:** *"Does NCCL work across NAT?"*
→ "No. That's why we use gloo. NCCL is optimized around communicating every step; we sync once every H steps, so CPU-side collectives are enough. **Making networks that NCCL can't cross usable is exactly why we chose DiLoCo.**"

## Tickets

| ID | Ticket | Time | P |
|---|---|---|---|
| N-1 | Tailscale mesh + direct check | 0.5h | **P0 ★blocker** |
| N-2 | gloo all-reduce across machines | 1h | **P0 ★blocker** |
| N-3 | Worker daemon | 2h | P0 |
| S-1 | FastAPI + WebSocket hub | 1.5h | P0 |
| N-4 | Checkpoint transfer | 2h | P0 |
| N-5 | Worker reclaim/rejoin CLI | 1h | P0 |
| **N-6a** | **Share rules + credits ledger** | **0.75h** | **P0** ← was P1 |
| S-8 | SQLite run logging | 1h | P0 |
| N-6b | Automatic game detection | 1.25h | P1 |

### N-1 · Tailscale mesh + direct check (0.5h) ★ blocker

```bash
# scripts/netcheck.sh
tailscale status --json | jq -r '.Peer[] | "\(.HostName) \(.CurAddr) \(if .Relay=="" then "DIRECT" else "RELAY:"+.Relay end)"'
iperf3 -c <peer-100.x-ip> -t 5     # actual bandwidth
```

- **`RELAY` is an emergency.** DERP relay collapses to 2–35 Mbit/s. Tell Ji immediately so the router raises H.
- **Re-run this on site.** Venue wifi often enables client isolation, so peers that were `direct` at home can drop to `relay` at the event.

**Done when:** a table of peer × direct/relay × measured bandwidth, reproducible with one command.

### N-2 · gloo all-reduce across machines (1h) ★ blocker

```python
# scripts/gloo_check.py
import os, time, torch, torch.distributed as dist
os.environ["GLOO_SOCKET_IFNAME"] = "tailscale0"
dist.init_process_group("gloo",
    init_method=f"tcp://{MASTER_TAILSCALE_IP}:29500",
    rank=RANK, world_size=WORLD)

t = torch.ones(25_000_000)                      # 100MB fp32
dist.all_reduce(t)                              # warmup
t0 = time.time(); dist.all_reduce(t); dt = time.time() - t0
print(f"100MB all-reduce: {dt:.2f}s → {100/dt:.1f} MB/s")
```

- **This `dt` feeds the router's H formula directly.** Scale it to model size and hand Ji the resulting `T_sync`.
- Do not attempt NCCL.

**Done when:** two machines complete an all-reduce and you have a `T_sync` number.

### N-3 · Worker daemon (2h)

`worker/daemon.py` — runs on every GPU machine.

```python
def collect_spec():
    p = torch.cuda.get_device_properties(0)
    return {
        "gpu": p.name,
        "vram_gb": round(p.total_memory / 1e9, 1),
        "cc": float(f"{p.major}.{p.minor}"),
        "vendor": "amd" if torch.version.hip else "nvidia",
        "chip_class": classify(p),           # Contract 5 — the router keys on this
    }
```

- On start `worker.register` (with `owner`, `share` rules, `credits_per_hour`), then `worker.heartbeat` every second
- gpu_util: `pynvml.nvmlDeviceGetUtilizationRates` (NVIDIA), `rocm-smi --showuse` (AMD)
- Reconnect with exponential backoff
- On `job.start`, **spawn Ethan's trainer as a subprocess** and relay its stdout events

**Done when:** two workers register and heartbeat. Kill one → server emits `worker.left` within 5s.

### S-1 · FastAPI + WebSocket hub (1.5h)

```
WS   /ws/worker             worker connections
WS   /ws/ui                 dashboard connections (fan-out)
GET  /workers               registry, with share status and credit rate
POST /jobs                  create job (the router's config JSON)
POST /jobs/{id}/migrate     trigger migration
PUT  /ckpt/{job}/{round}    upload checkpoint
GET  /ckpt/{job}/{round}    download checkpoint
GET  /credits/{owner}       balance
```

- Registry is an in-memory dict; events fan out verbatim to `/ws/ui`
- **Heartbeat watchdog:** 5s of silence → emit `worker.left` (reason=timeout)
- Mirror every event into SQLite (S-8)

**Done when:** worker events reach the dashboard uninterrupted, so Ji can drop mock data.

### N-4 · Checkpoint transfer (2h)

- `PUT /ckpt/{job_id}/{round}` multipart with sha256 header; `GET` to retrieve
- Stream progress as `migration.progress` (phase=`transferring`, pct)
- **That progress bar appears in demo beat 5.** "In progress" reads better than silent success

**Done when:** 500MB round-trip succeeds, duration logged.

### N-5 · Worker reclaim/rejoin CLI (1h) ★ typed on stage

```bash
$ python -m worker.cli stop --reason game
  ✓ w1 (RTX 3090, junho) reclaimed by owner — round continues with remaining workers
  ✓ credited junho 1.4 GPU-hours  ·  balance 128.5

$ python -m worker.cli start
  ✓ w1 rejoined — will participate from the next sync (round 12)
```

**Output has to be clean.** Judges will be looking at this terminal. No stack traces, no warnings. **Note the credit line** — it turns the dropout demo into a marketplace demo for free.

### N-6a · Share rules + credits ledger (0.75h) ★ promoted to P0

The marketplace is half the product, so this can't sit in P1.

```yaml
# ~/.gpushare/config.yaml
owner: junho
share:
  enabled: true
  not_gaming: true
  always_between: ["00:00", "08:00"]
credits_per_hour: 1.0
```

- Ship the rules in `worker.register`; the router reads them when choosing what to rent
- Ledger: accumulate contributed GPU-hours per owner, emit `credits.update`, expose `GET /credits/{owner}`
- Manual reclaim (N-5) already gives you the demo. **Auto-detection is N-6b and is not required.**

**Done when:** the dashboard shows a credit balance rising while a worker contributes and falling when its owner borrows.

### N-6b · Automatic game detection (1.25h, P1)

GPU occupied by a PID that isn't our trainer → auto `stop --reason game`. Nice to have; the manual path demos identically.

### S-8 · SQLite run logging (1h)

```sql
workers(worker_id, owner, gpu, vendor, vram_gb, cc, chip_class, first_seen)
jobs(job_id, config_json, started_at, ended_at)
steps(job_id, worker_id, step, loss, step_time_s, tokens, ts)
syncs(job_id, round, participants_json, bytes, duration_s, global_loss, ts)
migrations(job_id, from_worker, to_worker, round, duration_s, ts)
probes(job_id, chip_class, config_name, t_step_median_s, tokens_per_s, peak_vram_gb, gpu_util)
credits(owner, delta_gpu_hours, reason, ts)
```

**`probes.chip_class` is the moat.** That column is what makes "the 3090 agent learns from 3090 data" real rather than a slogan. Same string as Contract 5.

**We show these tables during the pitch.** Empty tables make the vision sound hollow.

## Jack's dependencies

| | |
|---|---|
| Waiting on | Hour-0 contracts only |
| Who waits on me | **Everyone.** N-2 determines Ethan's approach; without S-1, Ji can't leave mock data |
| After T+8 | My work is mostly done. **Go help Ethan** |

---
---

# PART 2 — ETHAN: Training Execution Engine

> "Execute the config you're given and report measurements. You don't decide what to do."

Ethan holds the demo's two strongest moments: T-3 (survives a worker being reclaimed) and T-5 (migration). **If the schedule slips, cut everything else and protect these two.**

## User story

Minji hits "start training" and **the loss starts dropping.** Two borrowed GPUs train independently and merge periodically; the dashboard blinks at each sync.

An hour in, friend A comes home and launches Valorant. His 3090 is reclaimed. **The loss curve doesn't break.** Minji only notices because of the notification.

Later the agent moves her to a 4090. A vertical line appears on the chart and **the curve continues.** All she knows is "it moved and training didn't break."

## Key decisions

| Thing | Choice | Note |
|---|---|---|
| Framework | PyTorch (CUDA + ROCm) | **Same code already runs on both vendors. We move weights, not code** |
| DiLoCo | torchft (primary) / diloco_simple (fallback) | torchft's DiLoCo path is **experimental, nightly-only**. Pin the version |
| Backend | gloo | No NCCL |
| Checkpoint | safetensors | Vendor-neutral. **Weights only** |
| Model | nanoGPT-scale decoder-only (~124M) | Two 24GB cards are not one 48GB card. No memory pooling |

### DiLoCo structure

```
each worker:  θ_before = θ.clone()
              for _ in range(H):  inner AdamW step
              pseudo_grad = θ_before − θ_after        ← this is what gets sent
globally:     outer optimizer (SGD + Nesterov) applies the averaged pseudo_grad
              → broadcast new global θ to all workers
```

**Key property:** right after a sync, every worker holds an identical global θ. **That instant is the only safe point for migration** — and it's why we never move inner optimizer state.

## Tickets

### T-2a · torchft verdict (1h) ★ do this first, ignore ticket order

Install nightly → run the DiLoCo/LocalSGD example → **pin the version in `requirements.txt`** (experimental packages break overnight).

**Done when:** a clear yes/no is shared with the team. **By T+3.**

### T-1 · Single-GPU training loop (1.5h)

`trainer/loop.py` — consumes the Contract 3 config verbatim.

```python
def build(cfg):
    dtype = {"bf16": torch.bfloat16, "fp16": torch.float16,
             "fp32": torch.float32}[cfg["dtype"]]
    model = NanoGPT(cfg["model"]).to("cuda", dtype=dtype)
    if cfg["attention"] == "sdpa":
        model.use_sdpa = True                     # F.scaled_dot_product_attention
    scaler = torch.amp.GradScaler() if cfg["dtype"] == "fp16" else None
    return model, scaler

# the fixed-work invariant
assert cfg["micro_batch"] * cfg["grad_accum"] * len(cfg["workers"]) * SEQ_LEN \
       == cfg["global_batch_tokens"]
```

- Emit `train.step` as one JSON line on stdout per step
- **Keep that assert.** When a chip agent raises the batch size, if this breaks we're doing *less work*, not faster work — and the before/after story collapses under questioning

### T-2 · DiLoCo across 2 workers (3h) ★ core

- inner AdamW for H steps → pseudo-grad → outer SGD (Nesterov) → broadcast
- Emit `train.sync` with `participants`, `bytes`, `duration_s`, `global_loss`
- Count `bytes` for real — T-8 uses it

**Done when:** loss drops across two machines, syncs land exactly every H steps, bytes logged.

### T-3 · Barrier timeout + average over survivors (2h, or 5h on fallback) ★★

**Put a timeout on the sync barrier and average over whoever responded.** Don't block waiting for a worker that's gone. That's the whole feature.

- **torchft path:** wire up the lighthouse quorum. Heartbeat health checks already exist.
- **Fallback — server-mediated averaging (recommended):**
  ```
  worker → POST pseudo_grad to server (with deadline)
  server → average whatever arrived by the deadline → broadcast new global θ
  ```
  A star topology instead of a collective. **Fault tolerance comes free** — if the server doesn't wait, that *is* the tolerance. With 2–4 workers and a large H, bandwidth is a non-issue, and it removes the gloo dependency entirely.
  → If torchft fails, take this route. Do not hand-roll fault tolerance on top of dist collectives.

**Done when:** an owner reclaiming their GPU mid-training doesn't stop training, and the next sync's `participants` list is shorter.
**This is all of demo beat 4, and half of T-5.**

### T-4 · Checkpoint save/load (1h)

```python
from safetensors.torch import save_file
save_file(model.state_dict(), f"ckpt/{job}/round_{n}.safetensors")
json.dump({"round": n, "global_step": s, "global_loss": gl,
           "config_hash": h, "ts": time.time()},
          open(f"ckpt/{job}/round_{n}.meta.json", "w"))
```

**Weights only.** No inner optimizer state — we only migrate at sync boundaries, so it isn't needed.

### T-5 · Migration execution (1.5h) ★★

```
1. wait for next sync          migration.progress phase=waiting_sync
2. save                        phase=saving
3. upload (Jack's N-4)         phase=transferring  pct=...
4. target worker loads         phase=loading
5. resume                      phase=resumed
```

**Mechanically, migration = T-3 (worker leaves) + a new worker joining.** Not a new mechanism. Step 1 is the whole trick; the rest is moving a file.

**Surface the wait.** When the user clicks migrate they'll wait seconds-to-minutes for the next sync. Show `waiting for sync` rather than hiding it — why it waits *is* the explanation of why it's safe.

**Done when:** **NVIDIA→NVIDIA works first.** `global_loss` continues across the migration. Swap the target to AMD once ROCm is up — identical mechanism, so if ROCm never works the demo still stands and we only lose the word "cross-vendor."

### T-6 · Probe runner (1.5h)

```python
def probe(cfg, steps=40, warmup=10):
    ts = [run_one_step(cfg) for _ in range(steps)]
    t_step = statistics.median(ts[warmup:])          # median, not mean
    return {"t_step_median_s": t_step,
            "tokens_per_s": cfg["global_batch_tokens"] / t_step,
            "peak_vram_gb": torch.cuda.max_memory_allocated() / 1e9,
            "gpu_util": sample_util(),
            "chip_class": CHIP_CLASS}                # Contract 5

def search_batch(cfg, target=0.85):
    mb = cfg["micro_batch"]
    while True:
        try:
            probe({**cfg, "micro_batch": mb * 2}, steps=3, warmup=1)
            if peak_vram_frac() > target: return mb
            mb *= 2
        except torch.cuda.OutOfMemoryError:
            torch.cuda.empty_cache(); return mb
```

**Always discard the first 10 steps** (cudnn autotune, allocator warmup, compile). Including them in the baseline makes the baseline artificially slow — the most common way teams fool themselves.

**Tag every probe with `chip_class`** so it lands in the right chip agent's history.

### T-7 · Global loss eval (0.5h)

Evaluate on a **fixed held-out batch** at every sync, attach to `train.sync.global_loss`.

**This is the only proof that migration preserved training.** Per-step loss bumps slightly after a migration (inner optimizer moments rebuild). Global loss is continuous **by construction** — the thing we moved *is* that global θ.

### T-8 · Bytes-on-wire + DDP baseline (1h, P1)

Run DDP once on the same model and step count, measure bytes. **The headline is a ratio: `DDP 12.4 GB → ours 40 MB (310×)`.** An absolute number alone means nothing to a listener. Prep work, not live.

## Ethan's dependencies

| | |
|---|---|
| Waiting on | N-2 (gloo) — but T-1 can start immediately |
| Who waits on me | Ji, though his mock data carries him to T+8 |
| Risk | If T-2a fails, T-3 goes 2h → 5h. **Jack joins you then** |

---
---

# PART 3 — JI: Agents + Optimization + Dashboard

> "This is the differentiator, and it has to be true in the code."

Rental decisions, scheduling, chip selection, config optimization, and migration all live here as **two tiers: the router and the chip agents.**

Ji also owns the dashboard. What judges actually see is one screen saying "the agent did something smart," so the person writing the rules should be the person drawing the screen.

## User story

Minji doesn't know what bf16 is, and doesn't need to. Before training starts the agent speaks first:

```
Ampere24GB agent measured your 3090s for 30 seconds.
  As configured:      5h 08m   ·  38 credits
  With my settings:   3h 42m   ·  28 credits
  Changed: fp32→bf16 · batch 8→32 (accum 4→1) · sdpa · H=190
  Tokens per optimizer step: 32,768 → 32,768  ✓
                                  [Use these] [Keep defaults]
```

Later:

```
💡 Router: AMD 7900 idle · 42% cheaper per hour
   → migrate?                    [Accept] [Decline]
```

**What Ji builds is everything Minji never has to learn.** Why NCCL doesn't work, why H is 190, what a 3090 wants that a 4090 doesn't. Those decisions having been made *for* her is the product.

**Question Ji must answer on stage:** *"Isn't this just a scheduler?"*
→ "A scheduler fills empty slots at placement time. We pick what to rent, what settings each chip type wants, and when to move — **all from measurements taken during the run.** And those measurements accumulate per chip type."

## Tickets

| ID | Ticket | Time | P |
|---|---|---|---|
| S-2 | Mock event emitter | 0.5h | **P0 ★do first** |
| S-3 | Dashboard core | 2.5h | P0 |
| S-4 | **Router + chip agent structure** | 2h | P0 |
| S-5 | H computation + straggler detection | 1h | P0 |
| S-6 | Before/after optimization screen | 1.5h | P0 |
| S-7 | Migration proposal card + visualization | 1.5h | P0 |
| **S-9** | **Credit pricing + rental decision** | **1.5h** | **P0** ← was P1 |

### S-2 · Mock event emitter (0.5h) ★★ do this first

A script that fires Contract 1 events. A 30-second scenario: two workers training → an owner reclaims one → a migration → credits ticking.

**Half an hour of work with the highest return on the board.** Ji is never blocked from T+2 to T+20. Without it you sit idle until training runs, then build the dashboard in a panic — and the dashboard is the demo's only output device.

### S-3 · Dashboard core (2.5h)

```
┌──────────────────────────────────────────────────┐
│  Training: small-llm-v2            Status: ● live │
│  Borrowed:                                        │
│   • friendA 3090  [████████] active  1.0 cr/h     │
│   • friendB 3090  [████████] active  1.0 cr/h     │
│   • lab AMD7900   [available]        0.6 cr/h     │
│  Loss: 2.4 → 1.1 ↓                                │
│   ─ global loss (bold, one point per sync)        │
│   ┄ step loss   (thin grey)                       │
│  Sync: every H=190 ●    Transfer: 40MB/round      │
│  Spent: 12.4 cr    ·    AWS equivalent: $6.20     │
└──────────────────────────────────────────────────┘
```

- Worker reclaimed → grey out with a transition
- Migration → vertical line with before/after values
- Provider view: credit balance ticking up

### S-4 · Router + chip agent structure (2h) ★ build the structure for real

**The rules are thin in P0. The structure must not be.** If a judge opens this file and finds one function with three if-statements, "a dedicated agent per chip" becomes a liability.

```python
class ChipAgent:
    """One per chip type. Rules today, a learned model later — same interface."""
    chip_class: str
    def decide(self, job, probe) -> dict: ...

class Ampere24GB(ChipAgent):        # RTX 3090
    chip_class = "ampere_24gb"
    def decide(self, job, probe):
        return {"dtype": "bf16", "attention": "sdpa",
                "micro_batch": probe.searched_batch}

class Ada24GB(ChipAgent):           # RTX 4090
    chip_class = "ada_24gb"
    ...

class DefaultAgent(ChipAgent):      # unknown chip — cold start
    chip_class = "default"
    def decide(self, job, probe):
        return {"dtype": "bf16" if probe.cc >= 8.0 else
                         "fp16" if probe.cc >= 7.0 else "fp32",
                "attention": "sdpa", "micro_batch": probe.searched_batch}

class Router:
    def agent_for(self, worker) -> ChipAgent:
        return REGISTRY.get(worker.chip_class, DefaultAgent())

    def plan(self, job, workers, probes):
        per_chip = {w.id: self.agent_for(w).decide(job, probes[w.id])
                    for w in workers if w.trainable}
        cfg = reconcile(per_chip)                            # pool-wide settling
        cfg["H"] = compute_H(probes.t_sync, probes.t_step)   # pool-level, S-5
        cfg["grad_accum"] = job.global_batch_tokens // (
            cfg["micro_batch"] * len(workers) * SEQ_LEN)     # fixed-work invariant
        return cfg
```

Two notes:

- **`worker.chip_class` is Contract 5, the same string Jack writes to SQLite.** That's what makes per-chip learning real rather than a slogan.
- **`reconcile()` matters:** workers can disagree (a 3090 and a 4090 want different batch sizes). For P0, take the most conservative dtype across the pool and let batch size stay per-worker.

**Lever order matters.** Memory-saving levers first (bf16 → sdpa → checkpointing), *then* batch search. Reversed, you redo the batch search after enabling bf16. Full lever table in Appendix A.

### S-5 · H computation + straggler detection (1h) ★ the router's signature lever

```python
def compute_H(t_sync, t_step, rho=0.05):
    h = int(t_sync / t_step * (1 - rho) / rho)
    return max(50, min(500, h))                    # clamp
```

| Situation | T_step | T_sync | H | Note |
|---|---|---|---|---|
| Tailscale direct | 0.20s | 2s | 190 | normal |
| DERP relay | 0.20s | 25s | 500 | **ceiling hit → warn** |
| Same LAN, wired | 0.20s | 0.5s | 50 | floor |

**This is the lever to put your name on.** bf16 and batch tuning are standard techniques everyone uses, so they're weak against "did the agent actually do that?" H is different: it's **the one lever whose value changes in response to measured network conditions**, and no human would bother computing it. When Jack says "we dropped to relay," H rising automatically becomes a demo moment.

H is pool-wide, so it belongs to the router, not a chip agent.

- On ceiling: show `"Network too slow — H at ceiling. Consider a smaller model."`
- For the demo raise ρ to 0.2 so H is small enough to show two syncs in five minutes. **Say that you did.**

```python
def check_stragglers(workers):
    med = statistics.median(w.t_step for w in workers if w.trainable)
    for w in workers:
        if w.t_step > 3 * med:      reassign(w, "preprocess")
        elif w.t_step > 1.2 * med:  w.micro_batch = int(w.micro_batch * med / w.t_step)
    # ⚠️ different batch sizes → outer average must be weighted by sample count.
    #    Agree this with Ethan.
```

### S-6 · Before/after optimization screen (1.5h)

Renders from two `probe.result` events. **Three lines are your defense:**

- `tokens per optimizer step: 32,768 = 32,768 ✓` — same amount of work
- `measured: 30 steps (10 warmup dropped) → projected` — we're not pretending we timed 5 hours
- `loss curves overlap ●` — faster, not worse

Show **credits saved** next to time saved. Same screen, and it ties the two layers together.

**Watch your baseline.** fp32 with batch 1 produces a dramatic number and dies to "nobody configures it that way." Use **PyTorch defaults plus what a beginner would plausibly set**, and state what the baseline was before showing the number. Full protocol in Appendix A.

### S-7 · Migration proposal card + visualization (1.5h)

- Rule: target available && (faster || cheaper in credits) && expected gain > migration cost
- Accept → `POST /jobs/{id}/migrate` → render the five `migration.progress` phases, **including `waiting for sync`**
- On completion: vertical line plus `before 1.13 → resumed 1.13`

**Keep the accept button.** Fully automatic migration is P2 and saying so is better than claiming it. The agent decides, the user stays in control — and a visible click demos far better than something happening silently.

### S-9 · Credit pricing + rental decision (1.5h) ★ promoted to P0

The router's first decision is *what to rent*, so this isn't a display feature.

```python
def pick_chips(job, market):
    """Cheapest chips that are actually sufficient — old generations included."""
    fits = [c for c in market if c.vram_gb >= job.min_vram
                              and c.cc >= 7.0 and c.available]
    return sorted(fits, key=lambda c: c.credits_per_hour)[:job.want]
```

Display:
```
GPU-hours 0.42 × peer rate 1.0 cr/h  = 12.4 credits
GPU-hours 0.42 × AWS on-demand $X/h  = $6.20 equivalent
```

**Put the formula on screen next to the number.** Otherwise the first Q&A question is "where did that come from?"

Stage line: *"Her job doesn't need a 4090. The router found chips that are merely sufficient, at a fraction of the rate."*

## Ji's dependencies

| | |
|---|---|
| Waiting on | Hour-0 contracts only. **S-2 means nothing blocks you after that** |
| Who waits on me | Ethan consumes the router's config JSON — freeze that shape early |
| Needs agreement | `chip_class` strings (Contract 5) with Jack · sample-count weighting with Ethan |

---
---

# PART 4 — INTEGRATION

Working alone too long means it won't merge. **Force a merge at fixed times.**

| Time | Merging | Success criterion |
|---|---|---|
| **T+3** | Jack N-2 · Ethan T-2a | gloo yes/no, torchft yes/no. **Revise the plan here if needed** |
| **T+8** | Jack N-3, S-1 + Ethan T-2 + Ji S-3 | Real training events on the real dashboard. **Mock data retired** |
| **T+13** | Ethan T-3 + Jack N-5, N-6a + Ji S-3 | Owner reclaim shows on screen, credits tick (beat 4 done) |
| **T+17** | Ethan T-5 + Jack N-4 + Ji S-7 | Full migration visible, global loss continuous (beat 5 done) |
| **T+19** | All | Three clean demo runs. **Code freeze.** Rehearsal and backup video only |

**No new features after T+19.**

## Things that span more than one person — name the owner

**Migration:**

| Part | Ticket | Owner |
|---|---|---|
| Transfer | N-4 | Jack |
| Execution (wait, save, load, resume) | T-5 | Ethan |
| Decision, proposal card, visualization | S-7 | Ji |

It's the demo's climax split three ways. **All three sit together at T+17.**

**`chip_class` (Contract 5):** Jack writes it (N-3, S-8), Ethan tags probes with it (T-6), Ji dispatches on it (S-4). If the three disagree on the string, per-chip learning silently doesn't happen. **Agree it at hour 0.**

## Load imbalance

Jack's work is **front-loaded** (mostly done by T+8). Ethan's is **back-loaded**, and T-3 balloons from 2h to 5h if torchft fails.
→ **After T+8, Jack helps Ethan.** This is the plan, not an exception.

## Cut order when time runs short

Cut from the top. Everything below the line breaks the demo.

1. N-6b automatic game detection → **use the manual CLI; it demos identically**
2. T-8 DDP baseline → show absolute bytes only
3. S-6 optimization screen → show the agent's decision in logs
4. Chip agents beyond two classes → keep `Ampere24GB` and `DefaultAgent`, enough to prove the structure
5. AMD migration → **run NVIDIA→NVIDIA.** Same mechanism; you only lose the adjective
6. ─────── do not cut below this line ───────
7. S-9 rental decision + credits *(half the product)*
8. N-6a share rules + ledger *(half the product)*
9. T-5 migration
10. T-3 fault tolerance
11. T-2 DiLoCo

> Note what moved: **credits and rental are now below the line.** Cut them and you lose the marketplace half of the pitch, and the positioning stops making sense.

---
---

# PART 5 — DEMO

Five minutes. **One sentence has to land:**

> "She had no GPU. She borrowed three from friends, an agent tuned and pooled them, and training never stopped — even when an owner took one back."

Everything else is evidence, and **the evidence is the loss curve.**

## Before you present

### On arrival (T-30) — network first, nothing else

| Check | How | If it fails |
|---|---|---|
| Tailscale `direct` or DERP `relay` | `tailscale status` | Relay → raise H (50→200), shrink model. Decide on the spot |
| gloo all-reduce works | the pre-written test script | Hotspot → wired → two local processes |
| `GLOO_SOCKET_IFNAME=tailscale0` set | env on each worker | — |

### Before walking on stage (T-10)

| Item | Owner |
|---|---|
| **Training job is already running** (a few hundred steps in) | Ethan |
| Workers A and B connected, credits already accruing | Jack |
| Migration target chip **warmed up** — model already loaded | Ethan |
| Fixed held-out eval batch confirmed | Ethan |
| Dashboard fullscreen, zoom set | Ji |
| Backup video tab open (hidden) | Ji |

**Never start anything on stage.** No initialization, no model loading, no first connection. The only two commands typed live are **one reclaim and one migration accept.**

Venue wifi killing the initial handshake is the number one way these demos die. Joining an already-running job fails gracefully; starting on stage does not.

## The beats

### 1 — Hook (0:00–0:30)
Hold up the laptop. *"This MacBook has no GPU. Right now a model is training on it — on three GPUs she doesn't own."* Cut to the dashboard: loss descending, two borrowed workers, credits ticking.
No architecture, no stack, no team intro. The screen appears within 30 seconds.

### 2 — The two halves (0:30–1:15)
*"GPUs aren't missing, they're scattered and idle. Vast.ai will sell you a datacenter but you still have to pool it yourself. Ray will pool chips but you have to find them. We do both — **and an agent decides what to rent in the first place.**"*
Show the borrow menu with credit rates. Plant that the third option is a **different vendor** — setup for beat 5.

### 3 — Pooled and synced (1:15–2:15)
*"Each chip trains on its own and syncs once every 190 steps. Communication drops by orders of magnitude, so home wifi is enough — not a datacenter fabric."*
Wait for a sync to blink and point at it. **Headline as a ratio:** `DDP would be 12.4 GB · ours 40 MB (310×)`.

### 4 — The owner comes home (2:15–3:00) ★ strongest moment
*"Her friend gets home and launches Valorant."* **Actually reclaim a worker in the terminal.**
One worker greys out, its owner's credit balance jumps. The loss curve keeps descending.
*"His GPU went back to him instantly, he got paid for the three hours, and her training didn't stop. Both sides got what they wanted."*

**⚠️ Wording:** do not say "DiLoCo is robust to dropout." DiLoCo gives you cheap communication, not recovery. Anyone who knows the paper will pounce:
> *"DiLoCo just makes leaving cheap. The recovery logic is ours, layered on top."*

**Reclaim a secondary worker.** Never touch the chip you need for beat 5.

### 5 — The agent decides (3:00–4:00) ★ where the differentiator lands

```
💡 Router: AMD 7900 idle · 42% cheaper per hour
   → migrate?   [Accept] [Decline]
```

*"This is the part we built. Two tiers: a router that decides what to rent and when to move, and a separate agent per chip type that knows how that specific chip wants to be configured."*

Click Accept. Phases appear: `waiting for sync → saving → transferring → loading on AMD → resumed`.

Point at the first phase: *"It waits. At a sync boundary every worker holds the same weights, so there's nothing else to move."*

Then the decisive line:
> *"The loss continues. What was saved on NVIDIA came back alive on AMD. **We don't convert code — the same PyTorch runs on both. We moved one vendor-neutral weights file.**"*

**Show the numbers explicitly:** `before 1.13 → resumed 1.13`. A curve alone is indistinguishable from a curve drawn to look continuous.

Don't rush this minute. It's the whole talk.

### 6 — Both sides of the ledger (4:00–4:30)
*"Her run: 12.4 credits. The same thing on AWS: $6.20."* **Formula on screen.**
*"And her job never needed a 4090 — the router picked chips that were merely sufficient."*
Then flip: *"Junho, whose 3090 you saw leave, earned credits while he was in class. He'll spend them on his own project next month. Lend it when it's idle, borrow it when you need it."*

### 7 — Vision and close (4:30–5:00)
*"Today a human clicked accept. Next, the router moves it alone. And every run we log goes into the agent for that chip — the 3090 agent only gets smarter from 3090 data. That's the moat."*
> *"You don't need a datacenter to train. The chips already exist — they're just idle, in someone else's room."*

## Roles on stage

| | Role | Does |
|---|---|---|
| **Ji** | Presenter | All narration. Drives the dashboard. Clicks Accept |
| **Ethan** | Training | Starts the job beforehand. Types the reclaim in beat 4. Technical Q&A |
| **Jack** | Network | Watches connections. **Silently** reconnects if anything drops. Calls fallbacks |

**The presenter does not stop, ever.** Jack fixes quietly; if it isn't back in 30 seconds, switch to the agreed fallback.

## Fallbacks

| What breaks | Fallback | What to say |
|---|---|---|
| Wifi dies | ① phone hotspot → ② wired → ③ two processes on one machine | "Network is unstable, so we'll show the same structure with two local processes" |
| Tailscale on DERP relay | Raise H 50→200, shrink model. **Recompute beat 3 timings** | (say nothing) |
| Worker won't connect | Continue with the already-running job, skip the join | (say nothing) |
| **No AMD machine / ROCm broken** | **Run NVIDIA→NVIDIA.** Mechanism identical, demo fully stands | "Same-vendor movement here, but the checkpoint is vendor-neutral — only the target changes" |
| AMD works but fails on stage | Recorded clip + show the **actual checkpoint file** in a terminal | "This part is pre-recorded" — honesty beats getting caught |
| Loss diverges / NaN | Resume from a pre-baked checkpoint | "Recovering from a checkpoint" (becomes a feature demo) |
| Dashboard dies | Fall back to terminal logs | Format the log output readably in advance |
| Everything dies | 3-minute full recording | Last resort. **Have it ready** |

**The backup video is not optional.** Record it the day before.

## Q&A

**Credits or real money?** ★ will be asked
> Credits today — contribute GPU-hours, spend them later. That keeps us out of settlement, disputes, and KYC, and reciprocity fits the users we want: people who both lend and borrow. Cash payouts are a business decision, not a technical one.

**Why would anyone lend me their GPU?**
> Because it's idle and it costs them nothing — the owner gets instant priority back, which you just saw. And most of our users are on both sides: they lend while at work and borrow when they need scale.

**Does NCCL work across NAT? You're on Tailscale.** ★ likely
> No. That's why we use gloo. NCCL assumes per-step communication; we sync every H steps, so CPU-side collectives suffice. **Making networks NCCL can't cross usable is why we chose DiLoCo.** torchft also uses gloo across replica groups.

**Do you transpile CUDA to ROCm?**
> No, and we don't need to. The same PyTorch code already runs on both vendors. We move a vendor-neutral weights file, not code.

**Isn't this just a scheduler? Couldn't Kubernetes do it?**
> A scheduler fills empty slots at placement time. We decide what to rent, what settings each chip type wants, and when to move — from measurements taken during the run. And those measurements accumulate per chip type.

**How is this different from Vast.ai?**
> Vast.ai is a rental marketplace and pooling is your problem after you rent. We do the pooling, and the agent picks what to rent in the first place.

**How is this different from Petals or Hivemind?**
> They pool chips you already have, inference-first, single stack. We're a marketplace plus orchestration across mixed vendors, with training as the main case.

**How do you run CUDA and ROCm together?**
> We don't. That's the design rule. Same vendor runs concurrently; different vendors migrate. Drawing that line is what let us avoid the hard part.

**How do you move optimizer state? Doesn't momentum get lost?**
> We don't move it. Migrating at a sync boundary means there's nothing to move — global weights are the entire shared state at that instant, and the new worker starts an inner loop exactly as a freshly joining worker would. Mid-inner-loop migration is a separate problem and it's next.

**Your "chip agents" are just if-statements, aren't they?**
> They're rules today, yes. But each chip type has its own agent behind one interface, and every run logs into that agent's history. When there's enough data, rules get replaced by learned models one class at a time without changing anything around them.

**Did you just use torchft? What did you build?**
> We use proven components for the training core. We built the marketplace, the heterogeneous pool management, cross-vendor migration, and the agents that decide. Rewriting DiLoCo from scratch isn't what a hackathon is for.

**Why should I put my data on someone else's GPU?**
> We haven't solved that. It's trust-based today; encryption, trust scores, and TEE are on the roadmap. — **Admit it.** Getting caught overstating costs far more.

**Does it actually converge at scale?**
> What we showed is a small model. We haven't validated large scale; our evidence goes as far as the DiLoCo literature does.

**What about stragglers holding everyone up?**
> The router pulls slow chips out of training and puts them on data preprocessing. (Demo it if built; otherwise "it's designed and it's next.")

## Rehearsal checklist

- [ ] Full run-through **three times clean**
- [ ] Each fallback path executed at least once for real
- [ ] Wifi cut deliberately, recovery time measured
- [ ] No "DiLoCo is robust to dropout" phrasing left in the script
- [ ] No "we migrate the code" phrasing left in the script
- [ ] Credit formula and AWS equivalent visible on screen
- [ ] Credit balance visibly moves during beat 4
- [ ] Before/after loss values visible as numbers at migration
- [ ] Backup video recorded
- [ ] Battery / charger / HDMI adapter / hotspot data

---
---

# APPENDIX A — Optimization levers and measurement (Ji)

## The invariant everything rests on

> **Optimization means doing the same work faster. Not doing less work.**

Doubling batch size halves the step count. Steps per second look better, but you may not have done the same work. So **every measurement here uses tokens/sec (training) or fixed-length output tokens/sec (inference).** Never steps/sec or requests/sec.

| Invariant | Training | Inference |
|---|---|---|
| Amount of work | tokens per optimizer step fixed (batch↑ offset by grad_accum↓) | request count, prompts, output tokens fixed (`ignore_eos` + `max_tokens`) |
| Quality | same seed, same data order → overlay loss curves | compare perplexity or sample outputs when quantizing |

## Decision pipeline

```
Phase 0  static read     (0s)      model spec, chip_class, versions → route to a chip agent
Phase 1  probe           (30–60s)  20–50 real steps → step time, peak mem, util, sync time
Phase 2  apply rules     (instant) chip agent decides config · router decides H and pool
Phase 3  monitor         (ongoing) straggler detection, H re-tune, migration decisions
```

**Phase 1 is the point.** Deciding from specs alone is just a lookup table, and "isn't that hardcoded?" has no good answer. Measuring and then deciding is the differentiator.

## Training levers

| # | Lever | Rule | Effect | Risk | Owner |
|---|---|---|---|---|---|
| 1 | bf16 / fp16 | `cc ≥ 8.0` → bf16; `7.0–8.0` → fp16 + GradScaler; `< 7.0` → drop from training | 1.5–2× | near zero on bf16; fp16 needs loss scaling | chip agent |
| 2 | SDPA / Flash Attention | dtype is fp16/bf16 and head_dim supported | large memory savings, speed at long seq | custom attention needs swapping | chip agent |
| 3 | Batch size search | double micro-batch until peak VRAM hits **85%**; OOM → previous value. **Hold global batch fixed via grad_accum** | largest lever when util is 60–95% | OOM — always try/except with rollback | chip agent |
| 4 | H (sync interval) | formula below | removes comm wait | too large degrades convergence | **router** |
| 5 | Straggler handling | rules below | removes sync wait | fewer workers | **router** |
| 6 | Dataloader tuning | util < 90% **and** data-wait share is high → more `num_workers`, `pin_memory` | big only when data-bound | none | chip agent |
| 7 | torch.compile | `projected_runtime × 0.15 > compile_cost (~60s)` | 1.1–1.4× | **net loss on short jobs** | chip agent |
| 8 | Activation checkpointing | **only when OOM** at target batch | 30–40% memory | ~30% more compute; a loss if not memory-bound | chip agent |
| 9 | Outer gradient quantization | measured bandwidth below threshold → int8 pseudo-grad | large sync-time reduction | implementation effort | router |

### Lever order

```
1. bf16               → halve memory
2. SDPA               → more memory saved
3. (if needed) checkpointing
4. THEN batch search   ← cash in the headroom from 1–3
5. probe for step time
6. measure sync time → compute H
7. compare per-worker step times → straggler handling
```

**Memory-saving levers first, memory-spending levers last.**

### Lever 4 — H formula

```
overhead = T_sync / (H × T_step + T_sync) ≤ ρ
→  H ≥ (T_sync / T_step) × (1 − ρ) / ρ
```

With ρ = 0.05, that's `H ≥ 19 × T_sync / T_step`. Clamp to **[50, 500]** — outside the range the DiLoCo literature covers, convergence isn't something we can claim. On the ceiling, warn the user and recommend a smaller model.

### Lever 5 — straggler rules

Round time is set by the slowest worker, so one straggler costs `(T_slow − T_median) × H` per round.

1. **Try batch balancing first** — allocate micro-batch inversely to step time so step times match. The slow chip still contributes.
   **⚠️ Then the outer average must be weighted by each worker's sample count.** A plain mean over-weights the slow worker's smaller sample.
2. **Otherwise drop it from training** — if `T_step > 3 × median` or `cc < 7.0` (no bf16/fp16 tensor cores, e.g. a GTX 970). Reassign to preprocessing / tokenization.

Stage line: *"Forcing a 970 into training makes everything run at 970 speed. Pull it out for preprocessing and it still contributes while everything else speeds up."*

## Inference levers

| # | Lever | Rule | Effect |
|---|---|---|---|
| 1 | vLLM (continuous batching, PagedAttention) | whenever concurrency > 1 | gap grows with concurrency. **At concurrency 1, barely any difference** |
| 2 | Quantization (int8 / AWQ / GPTQ) | VRAM pressure or throughput below target | ~2× throughput, half the memory. Check perplexity |
| 3 | Raise `gpu_memory_utilization` | queue backing up from KV cache shortage | more concurrent sequences |
| 4 | Lower `max_model_len` | actual prompts are short | more KV cache headroom |
| 5 | Chip right-sizing | model fits on an older card | fewer credits, slightly slower |

**Lever 1's condition dictates the demo design: measure at high concurrency (e.g. 64).** At concurrency 1 vLLM's advantage doesn't appear.

**Note for the pitch:** DiLoCo has nothing to do with inference — it's a training algorithm. Distributed inference here means independent per-request routing: each request is served whole by one GPU. That's why mixed vendors are a non-issue on the inference side.

## Before/after measurement

### Training — probe and extrapolate, transparently

You can't run five hours twice. Measure short, convert long, **and show the conversion on screen.**

```
1. run 40 steps on the baseline config
     → drop the first 10 (autotune, allocator warmup, compile)
     → take the MEDIAN of the remaining 30
2. run 40 steps on the optimized config, same seed, same data order
3. tokens/sec = global_batch_tokens / t_step   ← compare on this
4. ETA = total_steps × t_step + (total_steps / H) × t_sync
5. credits = GPU-hours × rate, for both configs
```

```
┌──────────────────────────────────────────────┐
│  Before / After        (Ampere24GB agent)     │
│  setting      default        agent            │
│  ───────────────────────────────────────      │
│  dtype        fp32           bf16             │
│  batch        8 (×4 accum)   32 (×1 accum)    │
│  attention    eager          sdpa             │
│  H            —              190              │
│                                               │
│  tokens/optimizer step  32,768   32,768   ✓   │
│  tokens/sec             14,200   19,800       │
│  step time (median)     2.31s    1.66s        │
│                                               │
│  measured: 30 steps (10 warmup dropped)       │
│  projected: 8,000 × 1.66s + 42 syncs × 2.0s   │
│             = 5h 08m → 3h 42m  (PROJECTED)    │
│  credits:     38 → 28                         │
│  loss curves overlap: ●                       │
└──────────────────────────────────────────────┘
```

### Cheating list — don't

| Cheat | Why it gets caught |
|---|---|
| Raise batch without lowering grad_accum | You did less work. tokens/sec barely moves |
| Include warmup steps in the baseline | You made the baseline artificially slow. Most common self-deception |
| Use the mean | One outlier swings it. Use the median |
| Deliberately terrible baseline (fp32 + batch 1) | "Nobody configures it that way" ends the conversation. **Baseline = a beginner's reasonable defaults** |
| Different seed or data order | Loss comparison becomes meaningless |
| Present a projection as a measurement | Fatal. Label it **projected** on screen and out loud |

### Inference — measure it for real

```
fixed load:  64 prompts, sent concurrently, exactly 128 output tokens each
             (ignore_eos=True, max_tokens=128  ← nail the output length down)

A) baseline:  sequential HuggingFace generate loop
B) optimized: vLLM

metric: wall clock for the whole set → output tokens/sec = (64 × 128) / elapsed
```

`ignore_eos` is the point. Without it A and B generate different amounts of work and the comparison collapses. **This is the inference version of the same-work invariant.**

```
64 concurrent requests · 128 output tokens each (fixed)

baseline (sequential)  ████████████████████  41.2s
vLLM (continuous)      ███                    6.8s

output tokens/sec:  199 → 1,204
output tokens: 8,192 = 8,192 ✓
```

Two bars growing live needs no explanation. If quantization is on, show two sample outputs side by side to pre-empt "did it get worse?"

## Fitting optimization into the demo

Three agent moments (optimization, reclaim, migration) will blow the five minutes. **Tie optimization and migration into one narrative:**

```
One question the agent answers two ways: "faster and cheaper — how?"
        ├─ change the config  (chip agent)  → 5h 08m → 3h 42m · 38 → 28 cr
        └─ change the chip    (router)      → 3h 42m → 2h 50m · 28 → 24 cr
```

> *"The agent answers one question two ways: change the settings, or change the chip. **Deciding which is the part we built.**"*

- **Put the optimization decision in the 30 seconds *before* training starts.** Not its own beat.
- **Keep the live inference before/after as a separate demo** for Q&A or the booth. One screen with two growing bars, 30 seconds.

## Implementation checklist (Ji)

- [ ] `ChipAgent` base class + at least `Ampere24GB` and `DefaultAgent`
- [ ] `Router.agent_for()` dispatch on `chip_class` (Contract 5, same string as Jack's SQLite)
- [ ] `reconcile()` for pools with mixed chip types
- [ ] Probe runner returning median step time, peak mem, util, sync time
- [ ] Batch search with OOM rollback
- [ ] Global-batch invariant enforced (`micro_batch × grad_accum × workers = const`)
- [ ] tokens/sec displayed (never steps/sec)
- [ ] H function with clamp and ceiling warning
- [ ] Per-worker step time → straggler decision → balance or reassign
- [ ] Sample-count weighting in the outer average (agreed with Ethan)
- [ ] ETA formula with a **projected** label
- [ ] Credit cost shown next to time, with the formula
- [ ] `pick_chips()` — cheapest sufficient, not fastest
- [ ] Baseline config defined as beginner defaults, its log preserved
- [ ] Inference load generator: 64 concurrent, `ignore_eos`, fixed `max_tokens`
