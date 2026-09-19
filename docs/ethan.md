# ETHAN — Training Execution Engine

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

## Who does what

| | Owns |
|---|---|
| **Jack** | Network + sharing marketplace — N-1…N-6, S-1, S-8 |
| **Ethan (you)** | Training execution engine — T-1…T-8 |
| **Ji** | Agents + optimization + dashboard — S-2…S-7, S-9 |

**The boundary:** Ji decides, **you execute**, Jack connects the two.

## Your part

> "Execute the config you're given and report measurements. You don't decide what to do."

Ji's agent produces a config JSON. You consume it verbatim and run it. **If you find yourself wanting to hardcode a rule, that's a ticket for Ji, not for you.**

**You hold the demo's two strongest moments:** T-3 (training survives a worker being reclaimed) and T-5 (migration). **If the schedule slips, cut everything else and protect those two.**

### User story

Minji hits "start training" and **the loss starts dropping.** Two borrowed GPUs train independently and merge periodically; the dashboard blinks at each sync.

An hour in, friend A comes home and launches Valorant. His 3090 is reclaimed. **The loss curve doesn't break.** Minji only notices because of the notification.

Later the agent moves her to a 4090. A vertical line appears on the chart and **the curve continues.** All she knows is "it moved and training didn't break."

---

# 2. Hour 0 — freeze the contracts (all three, 60 min)

**Do not skip this.** Without it the three codebases won't connect and you'll find out at hour 14.

### Contract 1 — events you emit (one JSON line per event on stdout; Jack's daemon relays them)

```json
{"type":"train.step","worker_id":"w1","step":1234,"loss":1.83,
 "step_time_s":0.21,"tokens":32768,"ts":...}

{"type":"train.sync","round":7,"participants":["w1","w2"],"bytes":41943040,
 "duration_s":2.1,"global_loss":1.79,"ts":...}

{"type":"migration.progress","job_id":"j1","from":"w1","to":"w3",
 "phase":"waiting_sync|saving|transferring|loading|resumed","pct":0.4,"ts":...}

{"type":"probe.result","worker_id":"w1","chip_class":"ampere_24gb",
 "config_name":"baseline|optimized","t_step_median_s":2.31,
 "tokens_per_s":14200,"peak_vram_gb":9.1,"gpu_util":0.62,"t_sync_s":2.0}
```

### Contract 2 — commands you receive

```json
{"type":"job.start","job_id":"j1","config":{ ...Contract 3... }}
{"type":"job.stop","job_id":"j1"}
{"type":"checkpoint.save","job_id":"j1","round":7}
{"type":"checkpoint.load","job_id":"j1","uri":"file:///.../round_7.safetensors"}
{"type":"config.update","micro_batch":32}
```

### Contract 3 — job config ★ this is your entire input

```json
{"job_id":"j1","model":"nanogpt-124m","dtype":"bf16","attention":"sdpa",
 "micro_batch":32,"grad_accum":1,"global_batch_tokens":32768,
 "H":190,"backend":"gloo","workers":["w1","w2"],"total_steps":8000}
```

**This single JSON is the entire interface between you and Ji.** Freeze its shape early.

### Contract 4 — checkpoint layout

```
ckpt/{job_id}/round_{N}.safetensors   ← weights only
ckpt/{job_id}/round_{N}.meta.json     ← {round, global_step, global_loss, config_hash, ts}
```

### Contract 5 — `chip_class` strings ★ you tag probes with these

```
ampere_24gb   (RTX 3090)
ada_24gb      (RTX 4090)
turing_16gb   (V100 etc.)
cdna_amd      (MI250, 7900)
default       (unknown — cold start)
```

Jack writes it in `worker.register` and SQLite. **You tag every probe result with it.** Ji dispatches the router on it. If the three of you disagree on the string, per-chip learning silently doesn't happen — no error, just nothing.

---

# 3. Key decisions

| Thing | Choice | Note |
|---|---|---|
| Framework | PyTorch (CUDA + ROCm) | **Same code already runs on both vendors. We move weights, not code** |
| DiLoCo | torchft (primary) / diloco_simple (fallback) | torchft's DiLoCo path is **experimental, nightly-only**. Pin the version |
| Backend | gloo | **No NCCL** — it fails over Tailscale across NAT (NVIDIA/nccl #1606, closed "not planned") |
| Checkpoint | safetensors | Vendor-neutral. **Weights only** |
| Model | nanoGPT-scale decoder-only (~124M) | Two 24GB cards are not one 48GB card. No memory pooling |

## Design rule you're implementing

Same vendor (NVIDIA↔NVIDIA) runs together. Different vendor (NVIDIA↔AMD) migrates only. **We never weave CUDA and ROCm inside one training run.**

And we don't convert code — the same PyTorch already runs on both vendors. **The only thing that moves is a weights file.**

## DiLoCo structure

```
each worker:  θ_before = θ.clone()
              for _ in range(H):  inner AdamW step
              pseudo_grad = θ_before − θ_after        ← this is what gets sent
globally:     outer optimizer (SGD + Nesterov) applies the averaged pseudo_grad
              → broadcast new global θ to all workers
```

**Key property:** right after a sync, every worker holds an identical global θ. **That instant is the only safe point for migration** — and it's why we never move inner optimizer state. Internalize this; it makes T-5 easy.

---

# 4. Your tickets

| ID | Ticket | Time | P |
|---|---|---|---|
| **T-2a** | **torchft verdict** | 1h | **P0 ★ do this first** |
| T-1 | Single-GPU training loop | 1.5h | P0 |
| T-2 | DiLoCo across 2 workers | 3h | P0 ★ |
| T-3 | Barrier timeout + survivor averaging | 2h (5h fallback) | P0 ★★ |
| T-4 | Checkpoint save/load | 1h | P0 |
| T-5 | Migration execution | 1.5h | P0 ★★ |
| T-6 | Probe runner | 1.5h | P0 |
| T-7 | Global loss eval | 0.5h | P0 |
| T-8 | Bytes-on-wire + DDP baseline | 1h | P1 |

### T-2a · torchft verdict (1h) ★ do this first, ignore ticket order

Install nightly → run the DiLoCo/LocalSGD example → **pin the version in `requirements.txt`** (experimental packages break overnight).

**Done when:** a clear yes/no is shared with the whole team. **By T+3.**
**If it fails:** fall back to diloco_simple, and T-3 grows from 2h to 5h. Jack joins you after T+8.

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
- **Keep that assert.** When Ji's chip agent raises the batch size, if this breaks we're doing *less work*, not faster work — and the whole before/after story collapses under questioning

**Done when:** loss drops on one machine and events reach the server.

### T-2 · DiLoCo across 2 workers (3h) ★ core

- inner AdamW for H steps → pseudo-grad → outer SGD (Nesterov) → broadcast
- Emit `train.sync` with `participants`, `bytes`, `duration_s`, `global_loss`
- Count `bytes` for real — T-8 uses it

**Done when:** loss drops across two machines, syncs land exactly every H steps, transferred bytes are logged.

### T-3 · Barrier timeout + survivor averaging (2h, or 5h on fallback) ★★

**Put a timeout on the sync barrier and average over whoever responded.** Don't block waiting for a worker that's gone. That's the whole feature.

- **torchft path:** wire up the lighthouse quorum. Heartbeat health checks already exist.
- **Fallback — server-mediated averaging (recommended):**
  ```
  worker → POST pseudo_grad to server (with deadline)
  server → average whatever arrived by the deadline → broadcast new global θ
  ```
  A star topology instead of a collective. **Fault tolerance comes free** — if the server doesn't wait, that *is* the tolerance. With 2–4 workers and a large H, bandwidth is a non-issue, and it removes the gloo dependency entirely (so it survives even if Jack's N-2 fails).
  → If torchft fails, take this route. **Do not hand-roll fault tolerance on top of dist collectives.**

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

**Done when:** save → load in a different process → identical loss on the same batch.

### T-5 · Migration execution (1.5h) ★★

```
1. wait for next sync          migration.progress phase=waiting_sync
2. save                        phase=saving
3. upload (Jack's N-4)         phase=transferring  pct=...
4. target worker loads         phase=loading
5. resume                      phase=resumed
```

**Mechanically, migration = T-3 (worker leaves) + a new worker joining.** Not a new mechanism. Step 1 is the whole trick; the rest is moving a file.

**Surface the wait.** When the user clicks migrate they'll wait seconds-to-minutes for the next sync. Emit `waiting_sync` rather than hiding it — why it waits *is* the explanation of why it's safe.

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

**Always discard the first 10 steps** (cudnn autotune, allocator warmup, compile). Including them in the baseline makes the baseline artificially slow — the most common way teams fool themselves, and Ji's whole before/after screen rests on this being honest.

**Tag every probe with `chip_class`** so it lands in the right chip agent's history.

### T-7 · Global loss eval (0.5h)

Evaluate on a **fixed held-out batch** at every sync, attach to `train.sync.global_loss`.

**This is the only proof that migration preserved training.** Per-step training loss bumps slightly right after a migration (inner optimizer moments rebuild) — if a judge spots that bump first, you're explaining instead of demoing. Global loss is continuous **by construction**: the thing you moved *is* that global θ.

### T-8 · Bytes-on-wire + DDP baseline (1h, P1)

Run DDP once on the same model and step count, measure bytes. **The headline is a ratio: `DDP 12.4 GB → ours 40 MB (310×)`.** An absolute number alone means nothing to a listener. Prep work — no need to reproduce it live.

---

# 5. Your dependencies

| | |
|---|---|
| Waiting on | Jack's N-2 (gloo) — but T-1 can start immediately, and T-2a comes first anyway |
| Who waits on you | Ji, though his mock data carries him to T+8 |
| **Risk** | If T-2a fails, T-3 goes 2h → 5h. **Jack joins you after T+8** — that's the plan |
| Needs agreement | Sample-count weighting in the outer average, with Ji (see below) |

**One thing to settle with Ji:** if his straggler logic gives workers different micro-batch sizes, **the outer average must be weighted by each worker's sample count.** A plain mean over-weights the slow worker's smaller sample.

## Integration checkpoints

| Time | What merges | Success criterion |
|---|---|---|
| **T+3** | Jack N-2 · your T-2a | gloo yes/no, torchft yes/no. **Revise the plan here if needed** |
| **T+8** | Jack N-3, S-1 + your T-2 + Ji S-3 | Real training events on the real dashboard |
| **T+13** | Your T-3 + Jack N-5, N-6a + Ji S-3 | Owner reclaim shows on screen, credits tick |
| **T+17** | Your T-5 + Jack N-4 + Ji S-7 | Full migration visible, global loss continuous |
| **T+19** | All | Three clean demo runs. **Code freeze** |

---

# 6. Demo — your role

**You start the job before the talk and type the reclaim command in beat 4.** You also take technical Q&A.

## Before stage (T-10) — yours

| Item |
|---|
| **Training job is already running** (a few hundred steps in) |
| Migration target chip **warmed up** — model already loaded |
| Fixed held-out eval batch confirmed |

**Never start anything on stage.** No initialization, no model loading, no first connection. Venue wifi killing the initial handshake is the number one way these demos die. Joining an already-running job fails gracefully; starting on stage does not.

## Beat 4 — the reclaim (2:15–3:00), the strongest moment

Ji says *"Her friend gets home and launches Valorant."* — Jack types the reclaim (or you do, whoever's at the terminal).
One worker greys out, the owner's credit balance jumps, the loss curve keeps descending.

**⚠️ Wording you must not use:** *"DiLoCo is robust to dropout."* DiLoCo gives cheap communication, not recovery. Anyone who knows the paper will pounce. Say:
> *"DiLoCo just makes leaving cheap. The recovery logic is ours, layered on top."*

That scores better anyway — it shows what you built.

**Reclaim a secondary worker.** Never touch the chip needed for beat 5.

## Beat 5 — migration (3:00–4:00), the climax

Ji clicks Accept; your phases render: `waiting for sync → saving → transferring → loading on AMD → resumed`.

The line that matters:
> *"The loss continues. What was saved on NVIDIA came back alive on AMD. **We don't convert code — the same PyTorch runs on both. We moved one vendor-neutral weights file.**"*

**Make sure `before 1.13 → resumed 1.13` shows as numbers**, not just a curve. A curve alone is indistinguishable from a curve drawn to look continuous.

## Fallbacks that involve you

| What breaks | Fallback |
|---|---|
| No AMD / ROCm broken | **Run NVIDIA→NVIDIA.** Mechanism identical, demo fully stands |
| AMD works but fails on stage | Recorded clip + show the **actual checkpoint file** in a terminal. Say "this part is pre-recorded" — honesty beats getting caught |
| Loss diverges / NaN | Resume from a pre-baked checkpoint. Say "recovering from a checkpoint" — it becomes a feature demo |

## Questions you answer in Q&A

**How do you move optimizer state? Doesn't momentum get lost?**
> We don't move it. Migrating at a sync boundary means there's nothing to move — global weights are the entire shared state at that instant, and the new worker starts an inner loop exactly as a freshly joining worker would. Mid-inner-loop migration is a separate problem and it's next.

**Do you transpile CUDA to ROCm?**
> No, and we don't need to. The same PyTorch code already runs on both vendors. We move a vendor-neutral weights file, not code.

**How do you run CUDA and ROCm together?**
> We don't. That's the design rule. Same vendor runs concurrently; different vendors migrate. Drawing that line is what let us avoid the hard part.

**Did you just use torchft? What did you build?**
> We use proven components for the training core. We built the marketplace, the heterogeneous pool management, cross-vendor migration, and the agents that decide. Rewriting DiLoCo from scratch isn't what a hackathon is for.

**Does it actually converge at scale?**
> What we showed is a small model. We haven't validated large scale; our evidence goes as far as the DiLoCo literature does.

---

# 7. Start here, right now

1. **T-2a** — install torchft nightly, run the DiLoCo example, does it work? Pin the version. Report by T+3.
2. **T-1** — single-GPU loop, don't wait for the network

Everything downstream depends on the T-2a answer, so get it out of the way before you build anything else.
