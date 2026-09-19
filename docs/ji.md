# JI — Agents + Optimization + Dashboard

Your brief. Self-contained — you don't need any other file.

---

# 1. The project in 60 seconds

**A marketplace where people lend and borrow each other's idle GPUs — plus an agent that pools whatever you borrowed and runs it well.**

```
Vast.ai            sells you chips.       Pooling them is your problem.
Ray / torchft      pools chips for you.   Finding them is your problem.
Us                 both — and the agent decides what to rent in the first place.
```

Minji has no GPU. She borrows from three friends. The agent pools them, picks the settings, and moves the job when a cheaper chip frees up. Junho is on the other side, earning credits from a 4090 that's idle while he's in class. **Renting settles in credits**, not money.

## Who does what

| | Owns |
|---|---|
| **Jack** | Network + sharing marketplace — N-1…N-6, S-1, S-8 |
| **Ethan** | Training execution engine — T-1…T-8 |
| **Ji (you)** | Agents + optimization + dashboard — S-2…S-7, S-9 |

**The boundary: you decide**, Ethan executes, Jack connects the two. You never touch the training loop; you emit a config JSON and Ethan runs it.

## Your part

> "This is the differentiator, and it has to be true in the code."

Rental decisions, scheduling, chip selection, config optimization, and migration all live here as **two tiers: the router and the chip agents.** You also own the dashboard — what judges actually see is one screen saying "the agent did something smart," so the person writing the rules should be the person drawing the screen.

**There is no component called "the scheduler."** Scheduling, chip selection, rental decisions, and optimization are all one thing: **the agent**. Two names means it gets built twice or not at all.

---

# 2. The agent, in two tiers

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

**In P0 these are rule-based Python classes.** That's fine — but **build the structure for real** (S-4). "We have a dedicated agent per chip" has to be true in the code or it backfires when a judge reads it.

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

**What you build is everything Minji never has to learn.** Why NCCL doesn't work, why H is 190, what a 3090 wants that a 4090 doesn't. Those decisions having been made *for* her is the product.

---

# 3. Hour 0 — freeze the contracts (all three, 60 min)

**Do not skip this.** Without it the three codebases won't connect and you'll find out at hour 14.

### Contract 1 — events you consume (from Jack's `/ws/ui`)

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

### Contract 3 — job config ★ this is your output

```json
{"job_id":"j1","model":"nanogpt-124m","dtype":"bf16","attention":"sdpa",
 "micro_batch":32,"grad_accum":1,"global_batch_tokens":32768,
 "H":190,"backend":"gloo","workers":["w1","w2"],"total_steps":8000}
```

**This single JSON is the entire interface between you and Ethan.** Freeze its shape early — he's building against it.

### Server endpoints you call (Jack's S-1)

```
GET  /workers               registry, with share status and credit rate
POST /jobs                  create job (your config JSON)
POST /jobs/{id}/migrate     trigger migration
GET  /credits/{owner}       balance
```

### Contract 5 — `chip_class` strings ★ your router dispatches on these

```
ampere_24gb   (RTX 3090)
ada_24gb      (RTX 4090)
turing_16gb   (V100 etc.)
cdna_amd      (MI250, 7900)
default       (unknown — cold start)
```

Jack writes it. Ethan tags probes with it. **You dispatch on it.** If the three of you disagree on the string, per-chip learning silently doesn't happen — no error, just nothing.

---

# 4. Your tickets

| ID | Ticket | Time | P |
|---|---|---|---|
| **S-2** | **Mock event emitter** | 0.5h | **P0 ★★ do first** |
| S-3 | Dashboard core | 2.5h | P0 |
| S-4 | Router + chip agent structure | 2h | P0 ★ |
| S-5 | H computation + straggler detection | 1h | P0 ★ |
| S-6 | Before/after optimization screen | 1.5h | P0 |
| S-7 | Migration proposal card + visualization | 1.5h | P0 |
| **S-9** | **Credit pricing + rental decision** | 1.5h | **P0** |

### S-2 · Mock event emitter (0.5h) ★★ do this first

A script that fires Contract 1 events. A 30-second scenario: two workers training → an owner reclaims one → a migration → credits ticking.

**Half an hour of work with the highest return on the board.** It means you're never blocked from T+2 to T+20. Without it you sit idle until Ethan's training runs, then build the dashboard in a panic at the end — and the dashboard is the demo's only output device.

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

**Global loss bold, per-step loss thin grey.** Per-step loss bumps right after a migration (inner optimizer moments rebuild); global loss is continuous by construction. If the bump is the prominent line, you'll be explaining instead of demoing.

Single HTML + WebSocket is plenty. React + Recharts if you prefer.

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

- `cc < 7.0` (no bf16/fp16 tensor cores, e.g. a GTX 970) → `trainable = False`, reassign to preprocessing
- **`reconcile()` matters:** a 3090 and a 4090 want different batch sizes. For P0, take the most conservative dtype across the pool and let batch size stay per-worker.
- **Lever order matters.** Memory-saving levers first (bf16 → sdpa → checkpointing), *then* batch search. Reversed, you redo the batch search after enabling bf16. Full table in the appendix.

### S-5 · H computation + straggler detection (1h) ★ your signature lever

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

H is pool-wide, so it belongs to the router, not a chip agent. Jack hands you `T_sync` from his gloo test.

- On ceiling: show `"Network too slow — H at ceiling. Consider a smaller model."`
- For the demo raise ρ to 0.2 so H is small enough to show two syncs in five minutes. **Say that you did.**

```python
def check_stragglers(workers):
    med = statistics.median(w.t_step for w in workers if w.trainable)
    for w in workers:
        if w.t_step > 3 * med:      reassign(w, "preprocess")
        elif w.t_step > 1.2 * med:  w.micro_batch = int(w.micro_batch * med / w.t_step)
    # ⚠️ different batch sizes → outer average must be weighted by sample count.
    #    AGREE THIS WITH ETHAN.
```

### S-6 · Before/after optimization screen (1.5h)

Renders from two `probe.result` events (baseline / optimized). **Three lines are your defense:**

- `tokens per optimizer step: 32,768 = 32,768 ✓` — same amount of work
- `measured: 30 steps (10 warmup dropped) → projected` — we're not pretending we timed 5 hours
- `loss curves overlap ●` — faster, not worse

Show **credits saved** next to time saved. Same screen, and it ties the marketplace and orchestration halves together.

**Watch your baseline.** fp32 with batch 1 produces a dramatic number and dies to "nobody configures it that way." Use **PyTorch defaults plus what a beginner would plausibly set**, and state what the baseline was before showing the number. Full protocol in the appendix.

### S-7 · Migration proposal card + visualization (1.5h)

- Rule: target available && (faster || cheaper in credits) && expected gain > migration cost
- Accept → `POST /jobs/{id}/migrate` → render the five `migration.progress` phases, **including `waiting for sync`**
- On completion: vertical line plus `before 1.13 → resumed 1.13` **as numbers**, not just a curve

**Keep the accept button.** Fully automatic migration is P2 and saying so is better than claiming it. The agent decides, the user stays in control — and a visible click demos far better than something happening silently.

**Show the `waiting for sync` phase.** Why it waits *is* the explanation of why it's safe.

### S-9 · Credit pricing + rental decision (1.5h) ★ P0, was P1

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

---

# 5. Your dependencies

| | |
|---|---|
| Waiting on | Hour-0 contracts only. **S-2 means nothing blocks you after that** |
| Who waits on you | Ethan consumes your config JSON — freeze that shape early |
| Needs agreement | `chip_class` strings with Jack · sample-count weighting with Ethan |

## Integration checkpoints

| Time | What merges | Success criterion |
|---|---|---|
| **T+3** | Jack N-2 · Ethan T-2a | gloo yes/no, torchft yes/no |
| **T+8** | Jack N-3, S-1 + Ethan T-2 + your S-3 | Real events on the real dashboard. **Mock data retired** |
| **T+13** | Ethan T-3 + Jack N-5, N-6a + your S-3 | Owner reclaim on screen, credits tick |
| **T+17** | Ethan T-5 + Jack N-4 + your S-7 | Full migration visible, global loss continuous |
| **T+19** | All | Three clean demo runs. **Code freeze** |

Your load is the lightest of the three on purpose — you're presenting, so the last hours go to rehearsal, and if something breaks on stage you need slack to fix the screen.

---

# 6. Demo — you present

Five minutes. **One sentence has to land:**

> "She had no GPU. She borrowed three from friends, an agent tuned and pooled them, and training never stopped — even when an owner took one back."

Everything else is evidence, and **the evidence is the loss curve.**

## Before stage (T-10) — yours

- Dashboard fullscreen, zoom set
- Backup video tab open (hidden)

**Nothing starts on stage.** The only two commands typed live are one reclaim and one migration accept.

## The beats

**1 — Hook (0:00–0:30).** Hold up the laptop. *"This MacBook has no GPU. Right now a model is training on it — on three GPUs she doesn't own."* Cut to the dashboard: loss descending, two borrowed workers, credits ticking. No architecture, no stack, no team intro.

**2 — The two halves (0:30–1:15).** *"GPUs aren't missing, they're scattered and idle. Vast.ai will sell you a datacenter but you still have to pool it yourself. Ray will pool chips but you have to find them. We do both — and an agent decides what to rent in the first place."* Show the borrow menu with credit rates. Plant that the third option is a **different vendor**.

**3 — Pooled and synced (1:15–2:15).** *"Each chip trains on its own and syncs once every 190 steps. Communication drops by orders of magnitude, so home wifi is enough — not a datacenter fabric."* Wait for a sync to blink and point at it. Headline as a ratio: `DDP would be 12.4 GB · ours 40 MB (310×)`.

**4 — The owner comes home (2:15–3:00) ★ strongest moment.** *"Her friend gets home and launches Valorant."* Someone reclaims a worker in the terminal. One worker greys out, its owner's credit balance jumps, the loss keeps descending. *"His GPU went back to him instantly, he got paid for the three hours, and her training didn't stop. Both sides got what they wanted."*

> **⚠️ Never say "DiLoCo is robust to dropout."** DiLoCo gives cheap communication, not recovery. Say: *"DiLoCo just makes leaving cheap. The recovery logic is ours, layered on top."*

**5 — The agent decides (3:00–4:00) ★ the differentiator.** The proposal card appears. *"This is the part we built. Two tiers: a router that decides what to rent and when to move, and a separate agent per chip type that knows how that specific chip wants to be configured."* Click Accept. Point at the first phase: *"It waits. At a sync boundary every worker holds the same weights, so there's nothing else to move."* Then:
> *"The loss continues. What was saved on NVIDIA came back alive on AMD. **We don't convert code — the same PyTorch runs on both. We moved one vendor-neutral weights file.**"*

Show `before 1.13 → resumed 1.13`. **Don't rush this minute. It's the whole talk.**

**6 — Both sides of the ledger (4:00–4:30).** *"Her run: 12.4 credits. The same thing on AWS: $6.20."* Formula on screen. *"And her job never needed a 4090 — the router picked chips that were merely sufficient."* Then flip: *"Junho, whose 3090 you saw leave, earned credits while he was in class. He'll spend them on his own project next month."*

**7 — Vision and close (4:30–5:00).** *"Today a human clicked accept. Next, the router moves it alone. And every run we log goes into the agent for that chip — the 3090 agent only gets smarter from 3090 data. That's the moat."*
> *"You don't need a datacenter to train. The chips already exist — they're just idle, in someone else's room."*

## Rules for you as presenter

**You do not stop, ever.** Jack fixes things silently; if it isn't back in 30 seconds, he calls the fallback and you keep talking.

| What breaks | What you say |
|---|---|
| Wifi dies | "Network is unstable, so we'll show the same structure with two local processes" |
| No AMD / ROCm broken | "Same-vendor movement here, but the checkpoint is vendor-neutral — only the target changes" |
| Pre-recorded segment | "This part is pre-recorded" — honesty beats getting caught |
| Loss diverges | "Recovering from a checkpoint" — it becomes a feature demo |

## Q&A — the ones aimed at you

**Isn't this just a scheduler? Couldn't Kubernetes do it?**
> A scheduler fills empty slots at placement time. We decide what to rent, what settings each chip type wants, and when to move — from measurements taken during the run. And those measurements accumulate per chip type.

**Your "chip agents" are just if-statements, aren't they?**
> They're rules today, yes. But each chip type has its own agent behind one interface, and every run logs into that agent's history. When there's enough data, rules get replaced by learned models one class at a time without changing anything around them.

**How is this different from Vast.ai?**
> Vast.ai is a rental marketplace and pooling is your problem after you rent. We do the pooling, and the agent picks what to rent in the first place.

**How is this different from Petals or Hivemind?**
> They pool chips you already have, inference-first, single stack. We're a marketplace plus orchestration across mixed vendors, with training as the main case.

**Credits or real money?**
> Credits today — contribute GPU-hours, spend them later. That keeps us out of settlement, disputes, and KYC. Cash payouts are a business decision, not a technical one.

**What about stragglers holding everyone up?**
> The router pulls slow chips out of training and puts them on data preprocessing. (Demo it if built; otherwise "it's designed and it's next.")

## Rehearsal checklist

- [ ] Full run-through **three times clean**
- [ ] Each fallback path executed at least once for real
- [ ] No "DiLoCo is robust to dropout" phrasing left in the script
- [ ] No "we migrate the code" phrasing left in the script
- [ ] Credit formula and AWS equivalent visible on screen
- [ ] Credit balance visibly moves during beat 4
- [ ] Before/after loss values visible as numbers at migration
- [ ] Backup video recorded

---
---

# APPENDIX — Optimization levers and measurement

## The invariant everything rests on

> **Optimization means doing the same work faster. Not doing less work.**

Doubling batch size halves the step count. Steps per second look better, but you may not have done the same work. So **every measurement uses tokens/sec (training) or fixed-length output tokens/sec (inference).** Never steps/sec or requests/sec.

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

With ρ = 0.05, that's `H ≥ 19 × T_sync / T_step`. Clamp to **[50, 500]** — outside the range the DiLoCo literature covers, convergence isn't something we can claim. On the ceiling, warn and recommend a smaller model.

### Lever 5 — straggler rules

Round time is set by the slowest worker, so one straggler costs `(T_slow − T_median) × H` per round.

1. **Try batch balancing first** — allocate micro-batch inversely to step time so step times match. The slow chip still contributes.
   **⚠️ Then the outer average must be weighted by each worker's sample count.** A plain mean over-weights the slow worker's smaller sample. Agree with Ethan.
2. **Otherwise drop it from training** — if `T_step > 3 × median` or `cc < 7.0`. Reassign to preprocessing / tokenization.

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

## Implementation checklist

- [ ] `ChipAgent` base class + at least `Ampere24GB` and `DefaultAgent`
- [ ] `Router.agent_for()` dispatch on `chip_class` (same string as Jack's SQLite)
- [ ] `reconcile()` for pools with mixed chip types
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

---

# 7. Start here, right now

1. **S-2** — the mock event emitter. Thirty minutes, and then nothing blocks you all weekend.
2. **S-3** — dashboard core against the mock data
3. Freeze the Contract 3 shape and tell Ethan, since he's building against it

Jack and Ethan's blockers resolve at T+3. You don't wait for either.
