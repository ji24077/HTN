# JACK — Network + Sharing Marketplace

Your brief. Self-contained — you don't need any other file.

---

# 1. The project in 60 seconds

**A marketplace where people lend and borrow each other's idle GPUs — plus an agent that pools whatever you borrowed and runs it well.**

```
Vast.ai            sells you chips.       Pooling them is your problem.
Ray / torchft      pools chips for you.   Finding them is your problem.
Us                 both — and the agent decides what to rent in the first place.
```

Minji has no GPU. She borrows from three friends. The agent pools them, picks the settings, and moves the job when a cheaper chip frees up. Junho is on the other side, earning credits from a 4090 that's idle while he's in class.

**Renting settles in credits**, not money — contribute GPU-hours, spend them later. No payouts, no KYC.

## Who does what

| | Owns |
|---|---|
| **Jack (you)** | Network + sharing marketplace — N-1…N-6, S-1, S-8 |
| **Ethan** | Training execution engine — T-1…T-8 |
| **Ji** | Agents + optimization + dashboard — S-2…S-7, S-9 |

**The boundary:** Ji decides, Ethan executes, **you connect the two.**

## Your part

> "Connect scattered machines, make them talk, and let their owners lend them."

You own **both ends of the wire**: the worker daemon on each GPU machine, and the server hub everyone connects to. Same protocol on both sides — one person writes both, or you get protocol bugs. You also own the supply side of the marketplace: share rules, instant reclaim, credits.

### User story

When Minji logs in she sees **friend A's 3090, friend B's 3090, and the lab's AMD 7900 as a list with credit rates.** Three apartments, three routers. On her screen it's just a menu.

Junho sets "only when I'm not gaming" and forgets about it. His 4090 works while he's in class and earns credits; when he launches a game it drops out immediately. **From Junho's side nothing interrupts his game. From Minji's side one worker turns grey.** Same event, neither party inconvenienced — that's what you build.

---

# 2. Hour 0 — freeze the contracts (all three, 60 min)

**Do not skip this.** Without it the three codebases won't connect and you'll find out at hour 14.

Sit at one screen, agree on all five, commit as `contracts.py`. Changes after this need all three of you.

### Contract 1 — Worker → Server events (you emit these)

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

`train.*` and `probe.result` come from Ethan's trainer on stdout — you just relay them.

### Contract 2 — Server → Worker commands (you send these)

```json
{"type":"job.start","job_id":"j1","config":{ ...Contract 3... }}
{"type":"job.stop","job_id":"j1"}
{"type":"checkpoint.save","job_id":"j1","round":7}
{"type":"checkpoint.load","job_id":"j1","uri":"file:///.../round_7.safetensors"}
{"type":"config.update","micro_batch":32}
```

### Contract 3 — Job config (Ji → Ethan; you carry it)

```json
{"job_id":"j1","model":"nanogpt-124m","dtype":"bf16","attention":"sdpa",
 "micro_batch":32,"grad_accum":1,"global_batch_tokens":32768,
 "H":190,"backend":"gloo","workers":["w1","w2"],"total_steps":8000}
```

### Contract 4 — Checkpoint layout

```
ckpt/{job_id}/round_{N}.safetensors   ← weights only
ckpt/{job_id}/round_{N}.meta.json     ← {round, global_step, global_loss, config_hash, ts}
```

### Contract 5 — `chip_class` strings ★ you write these

```
ampere_24gb   (RTX 3090)
ada_24gb      (RTX 4090)
turing_16gb   (V100 etc.)
cdna_amd      (MI250, 7900)
default       (unknown — cold start)
```

You write it in `worker.register` and SQLite. Ethan tags probes with it. Ji dispatches the router on it. **If the three of you disagree on the string, per-chip learning silently doesn't happen — no error, just nothing.**

---

# 3. Key decisions

| Thing | Choice | Why |
|---|---|---|
| Overlay network | Tailscale (WireGuard) | Fastest path to meshing NAT'd home machines |
| Collective backend | **gloo** | NCCL fails across NAT. Don't try it |
| Server | FastAPI + WebSocket | Everything else is Python |
| Checkpoint transfer | HTTP multipart + sha256 | Simplicity wins. P2P is overkill |
| Settlement | **Credits**, not money | No payouts, no disputes, no KYC |
| DB | SQLite | Fine for now |

**NCCL does not work over Tailscale across NAT.** Documented (NVIDIA/nccl #1606, closed "not planned"). Use gloo. DiLoCo's infrequent syncing is what makes that fine — not a compromise, it's why we chose DiLoCo.

---

# 4. Your tickets

| ID | Ticket | Time | P |
|---|---|---|---|
| N-1 | Tailscale mesh + direct check | 0.5h | **P0 ★blocker** |
| N-2 | gloo all-reduce across machines | 1h | **P0 ★blocker** |
| N-3 | Worker daemon | 2h | P0 |
| S-1 | FastAPI + WebSocket hub | 1.5h | P0 |
| N-4 | Checkpoint transfer | 2h | P0 |
| N-5 | Worker reclaim/rejoin CLI | 1h | P0 |
| **N-6a** | **Share rules + credits ledger** | **0.75h** | **P0** |
| S-8 | SQLite run logging | 1h | P0 |
| N-6b | Automatic game detection | 1.25h | P1 |

> S-1 and S-8 have `S` IDs but they're yours — the worker daemon and the server hub are two ends of one protocol.

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

- **This `dt` feeds Ji's H formula directly.** Scale it to model size and hand Ji the resulting `T_sync`.
- Do not attempt NCCL.

**Done when:** two machines complete an all-reduce and you have a `T_sync` number.
**If it fails:** plan change — two processes on one box, or Ethan's server-mediated averaging (which removes the gloo dependency entirely).

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
        "chip_class": classify(p),           # Contract 5
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
POST /jobs                  create job (Ji's router config JSON)
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

**Done when:** 500MB round-trip succeeds, duration logged (used for migration ETA display).

### N-5 · Worker reclaim/rejoin CLI (1h) ★ you type this on stage

```bash
$ python -m worker.cli stop --reason game
  ✓ w1 (RTX 3090, junho) reclaimed by owner — round continues with remaining workers
  ✓ credited junho 1.4 GPU-hours  ·  balance 128.5

$ python -m worker.cli start
  ✓ w1 rejoined — will participate from the next sync (round 12)
```

**The output has to be clean.** Judges will be looking at this terminal. No stack traces, no warnings. **Note the credit line** — it turns the dropout demo into a marketplace demo for free.

### N-6a · Share rules + credits ledger (0.75h) ★ P0, was P1

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

- Ship the rules in `worker.register`; Ji's router reads them when choosing what to rent
- Ledger: accumulate contributed GPU-hours per owner, emit `credits.update`, expose `GET /credits/{owner}`
- Manual reclaim (N-5) already gives you the demo. **Auto-detection is N-6b and is not required.**

**Done when:** the dashboard shows a credit balance rising while a worker contributes and falling when its owner borrows.

### N-6b · Automatic game detection (1.25h, P1)

GPU occupied by a PID that isn't our trainer → auto `stop --reason game`. Nice to have; the manual path demos identically. **This is the first thing to cut.**

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

---

# 5. Your dependencies

| | |
|---|---|
| Waiting on | Hour-0 contracts only. Independent afterwards |
| Who waits on you | **Everyone.** N-2 determines Ethan's approach; without S-1, Ji can't leave mock data |
| **After T+8** | Your work is mostly done. **Go help Ethan** — his T-3 balloons from 2h to 5h if torchft fails. This is the plan, not an exception |

## Integration checkpoints

| Time | What merges | Success criterion |
|---|---|---|
| **T+3** | Your N-2 · Ethan's T-2a | gloo yes/no, torchft yes/no. **Revise the plan here if needed** |
| **T+8** | Your N-3, S-1 + Ethan T-2 + Ji S-3 | Real training events on the real dashboard |
| **T+13** | Ethan T-3 + your N-5, N-6a + Ji S-3 | Owner reclaim shows on screen, credits tick |
| **T+17** | Ethan T-5 + your N-4 + Ji S-7 | Full migration visible, global loss continuous |
| **T+19** | All | Three clean demo runs. **Code freeze** |

---

# 6. Demo — your role

**You watch the connections and fix things silently.** The presenter (Ji) never stops. If something drops and it isn't back in 30 seconds, call the agreed fallback.

You also type the reclaim command in beat 4.

## On arrival (T-30) — your job, do it first

| Check | How | If it fails |
|---|---|---|
| Tailscale `direct` or DERP `relay` | `tailscale status` | Relay → tell Ji to raise H (50→200), shrink model |
| gloo all-reduce works | your test script | Hotspot → wired → two local processes |
| `GLOO_SOCKET_IFNAME=tailscale0` set | env on each worker | — |

## Before stage (T-10) — yours

- Workers A and B connected, credits already accruing

**Never start anything on stage.** Joining an already-running job fails gracefully; starting on stage does not.

## Beat 4 — the reclaim (2:15–3:00), the strongest moment

Ji says *"Her friend gets home and launches Valorant."* — you type the reclaim.
One worker greys out, the owner's credit balance jumps, the loss curve keeps descending.

**Reclaim a secondary worker.** Never touch the chip needed for beat 5.

## Fallbacks you call

| What breaks | Fallback |
|---|---|
| Wifi dies | ① phone hotspot → ② wired → ③ two processes on one machine |
| Tailscale on DERP relay | Tell Ji: raise H 50→200, shrink model |
| Worker won't connect | Continue with the already-running job, skip the join |
| No AMD / ROCm broken | Run NVIDIA→NVIDIA. Mechanism identical, demo fully stands |
| Everything dies | The 3-minute backup recording |

## Questions you answer in Q&A

**Does NCCL work across NAT? You're on Tailscale.** ★ likely
> No. That's why we use gloo. NCCL assumes per-step communication; we sync every H steps, so CPU-side collectives suffice. **Making networks NCCL can't cross usable is why we chose DiLoCo.** torchft also uses gloo across replica groups.

**Credits or real money?**
> Credits today — contribute GPU-hours, spend them later. That keeps us out of settlement, disputes, and KYC, and reciprocity fits our users: people who both lend and borrow. Cash payouts are a business decision, not a technical one.

**Why would anyone lend me their GPU?**
> Because it's idle and it costs them nothing — the owner gets instant priority back, which you just saw. And most of our users are on both sides: they lend while at work and borrow when they need scale.

**Why should I put my data on someone else's GPU?**
> We haven't solved that. It's trust-based today; encryption, trust scores, and TEE are on the roadmap. — **Admit it.** Getting caught overstating costs far more.

---

# 7. Start here, right now

1. `netcheck.sh` — is Tailscale `direct` or `relay`? (N-1)
2. gloo all-reduce across two machines, record `T_sync` (N-2)
3. Report both to the team at T+3

Everything else waits on those two answers.
