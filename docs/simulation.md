# Simulation design — Ji

What the agent simulates, what it measures, and which of the two each number is.

**Status key used throughout:**

| | |
|---|---|
| **BUILT** | code exists, tests pass, you can run it today |
| **DESIGNED** | the equations and constraints are settled, no code yet |
| **OPEN** | known problem, no answer yet |

**One rule governs this whole document:** a simulator that has never been checked
against hardware is worse than no simulator, because you end up tuning the agent
against fiction. Every predicted number must be reportable next to its error.

---

## 0. Why simulate at all

The agent has to make four kinds of decision — what to rent, what config to run,
when to migrate, how to serve. Each decision needs measurements, and measurements
need rented GPUs. That makes the decision logic expensive and slow to develop,
and it makes experiments non-repeatable.

So we split measurement from decision behind one interface, and supply two
implementations. The agent develops for free against the simulator; money is
spent only on validation, and the gap between the two is itself a result.

---

## 1. The seam — `agent/simulate.py`  **BUILT**

```python
class Prober(Protocol):
    def probe(self, cfg: JobConfig, chip: ChipSpec) -> ProbeResult: ...

class SimProber:      # free, instant, no GPU            BUILT
class RunPodProber:   # real, costs money, slow          DESIGNED
```

Both return `contracts.ProbeResult`, so `Router` and the chip agents cannot tell
which one they got.

**Why `ProbeResult` and not a new type:** it is already in the frozen
`contracts.py`, already carries `chip_class` (Contract 5), and already commits to
the median-with-warmup-dropped convention. Inventing a parallel shape would mean
two things to keep in sync and a contract negotiation with Jack and Ethan for no
gain.

**What is deliberately NOT in `contracts.py`:** the four specs below. They are the
agent's internal cost model; neither Jack nor Ethan consumes them. `JobConfig.model`
stays a plain string and `specs.MODELS` maps it back. This keeps the frozen file
frozen — a change there needs all three people in the same room.

---

## 2. The four axes — `agent/specs.py`  **BUILT**

Split so each can be varied while the others are held fixed.

| Axis | Carries | Feeds |
|---|---|---|
| `ModelSpec` | `params, layers, d_model, n_heads, vocab` | sync bytes, memory, FLOPs |
| `DataSpec` | `tokens_total, seq_len` | `total_steps` |
| `ChipSpec` | `chip_class, vram_gb, cc, tflops_bf16, mem_bw_gbs, credits_per_hour` | rate, capacity, price |
| `NetSpec` | `bandwidth_mbps, latency_ms` | `T_sync`, and therefore `H` |

`ChipSpec` is mostly populated from the live RunPod catalog
(`GET /v2/catalog/gpus/{id}`), which returns `memory`, `price.secure`,
`price.community`, `availability`, and per-datacenter stock. Only `tflops_bf16`
and `mem_bw_gbs` come from a hand-maintained table.

---

## 3. Model — what we chose and why

### The spec, not the repo

The cost model needs five integers: `params, layers, d_model, n_heads, vocab`.
Whichever repo implements the training loop, those numbers are the same — so the
choice of implementation (Ethan's call, ticket T-1) does not affect anything here.

Registered in `specs.MODELS`:

| name | params | layers | d_model | heads | why it's in the list |
|---|---|---|---|---|---|
| `nanogpt-124m` | 124M | 12 | 768 | 12 | the handbook's reference size |
| `nanogpt-350m` | 350M | 24 | 1024 | 16 | headroom test — does the agent notice it won't fit? |
| `nanogpt-30m` | 30M | 6 | 384 | 6 | cheap smoke tests |

### Finding: nanoGPT is deprecated

`karpathy/nanoGPT`'s README now says *"nanoGPT has a new and improved cousin
called nanochat. It is very likely you meant to use/find nanochat instead."*

This matters for T-1, not for the simulator:

- **nanoGPT** — ~300 readable lines, easiest place to wire DiLoCo's inner/outer
  loop. But it is torch-2.0-era code and our lock is on **torch 2.11**, so
  breakage is a live risk.
- **nanochat** — current and maintained, sized by `--depth` (GPT-2 ≈ depth 26).
- **Pretrained `gpt2`** — loss starts low, so the demo curve barely descends.
  Rejected on those grounds.

### Finding: `nanogpt-30m` is not the fix for a slow link

`specs.py` originally described the 30M model as the fallback for when `H` hits
its ceiling. The simulator contradicted that (see §5.4) and the comment was
corrected. It cuts sync cost ~4x and still pins `H` at 500.

---

## 4. Data — what we chose and why

### The experiments need almost no data

The handbook's measurement protocol is **40 steps, first 10 dropped, median of
the remaining 30**. At 32,768 tokens per step:

| purpose | tokens | file size (uint16) |
|---|---|---|
| one probe (40 steps) | 1.3M | **2.6 MB** |
| full demo run (8,000 steps, 1 epoch) | 262M | **524 MB** |

Downloading OpenWebText's 9B tokens would burn hours of **$0.74/hr GPU time** on
the download alone, to use 0.01% of it.

Registered in `specs.DATASETS`:

| name | tokens | use |
|---|---|---|
| `tinyshakespeare` | 1.1M | probes and smoke tests — downloads in seconds |
| `openwebtext-1b` | 1B | reserve |
| `fineweb-10b` | 10B | demo curve; take a slice, don't materialise it |

Tokenizer is GPT-2 BPE via `tiktoken`. FineWeb-Edu publishes `sample-10BT` /
`sample-100BT` / `sample-350BT` subsets and streams.

**Operationally:** tokenize on a laptop, ship the `.bin`. Or write it once to a
RunPod **network volume** in the target datacenter and let every pod there mount
it — network volumes are datacenter-scoped, so one download serves the whole DC.
Never re-tokenize per pod.

---

## 5. Training cost model — `agent/simulate.py`  **BUILT**

### 5.1 Work

```
param_flops_per_token = 6 × params                              # fwd + bwd
attn_flops_per_token  = 12 × layers × d_model × seq_len
```

Split rather than summed because they behave differently: the first scales with
parameters, the second with sequence length. SDPA only touches the second, which
is why it is worth more at 2048 than at 512.

```
work = (param_flops + attn_flops / SDPA_SPEEDUP) × micro_batch × grad_accum × seq_len
SDPA_ATTENTION_SPEEDUP = 1.4
```

SDPA does not do fewer FLOPs — it does the same math without staging the
`seq × seq` matrix through HBM. Folding that into the work term keeps one number
to calibrate instead of two rates.

### 5.2 Rate

```
rate_before_mfu = tflops_for(dtype) × 1e12 × occupancy(micro_batch × seq_len)
                  × (COMPILE_SPEEDUP if compile else 1)

t_step = work / (rate_before_mfu × mfu)
```

- `tflops_for("fp32")` = `tflops_bf16 / 2` (the TF32 path).
- `COMPILE_SPEEDUP = 1.25`, after a ~60s warmup — which is why lever 7 is a net
  loss on short jobs.
- **`occupancy`** is a saturating curve normalised so `MFU_REFERENCE_TOKENS =
  32,768` scores exactly 1.0, with half-saturation at a third of that.

**Why occupancy exists.** The first version of this model had `t_step`
independent of `micro_batch` and `attention`. Everything still looked plausible
in the output table — but the batch search, which the handbook calls the largest
lever, was a **no-op in simulation**. The agent would have been tuned against a
model in which its biggest lever did nothing. Two regression tests now pin this:
`test_bigger_micro_batch_is_faster_at_identical_work` and
`test_sdpa_is_faster_not_only_smaller`.

With it, at a fixed 65,536 tokens per optimizer step on a 4090:

| micro_batch × accum | t_step | tokens/s | peak VRAM |
|---|---|---|---|
| 4 × 8 | 2.571s | 25,489 | 3.2 GB |
| 8 × 4 | 1.636s | 40,054 | 4.3 GB |
| 16 × 2 | 1.169s | 56,075 | 6.7 GB |
| 32 × 1 | 0.935s | 70,094 | 11.4 GB |

Same work, 2.75x the throughput, 3.6x the memory. That trade is the batch search.

### 5.3 Memory

```
static      = params×b + params×b + params×8 + (params×4 if mixed precision)
              weights    grads      AdamW m,v   fp32 master copy

act_linear  = micro_batch × seq_len × layers × d_model × 12 × b
act_attn    = micro_batch × heads × seq_len² × layers × b     # EAGER ONLY

peak_vram   = static + (act_linear + act_attn) × act_overhead
```

Two properties the batch search depends on, both under test:

- **`grad_accum` does not appear.** Accumulation exists so a large global batch
  fits in a small resident one. Modelling this wrong is how a batch search OOMs.
- **bf16 saves less than "half the dtype" suggests** — the fp32 master copy and
  fp32 optimizer moments dominate `static`. Honest modelling here is why the
  search finds less headroom than people expect.

`fits()` targets **85%** of VRAM, leaving room for fragmentation.

### 5.4 Sync — and therefore H

```
payload = params × dtype_bytes × 2(n−1)/n        # ring all-reduce
t_sync  = payload / (bandwidth_bytes_per_s × sync_eff) + 2 × latency
H       = clamp(⌈(t_sync / t_step) × (1−ρ)/ρ⌉, 50, 500)
```

Ring all-reduce moves `2(n−1)/n` of the payload through each worker, so 2 → 4
workers costs 1.5x, not 2x.

`NETS` registry:

| name | Mbps | note |
|---|---|---|
| `runpod-global` | **100** | measured ceiling, not an estimate — RunPod caps pod-to-pod global networking at 100 Mbps regardless of region |
| `tailscale-direct` | 200 | estimate, pending Jack's N-1 |
| `tailscale-derp` | 20 | relay, the emergency case |
| `lan-wired` | 1000 | floor case |

**Finding: at 100 Mbps, H pins to its ceiling.** A 124M bf16 pseudo-gradient is
248 MB, ~26s on that link. The ideal `H` is far past 500, so it clamps. Dropping
to 30M cuts sync to ~6.5s and **still** clamps. The honest agent output is
*"this model is too big for this link"*, not a quietly clamped number.

Worth noting what this costs in practice: at `H = 500` and `t_step = 0.935s`, a
round is 468s of compute against 26s of sync — about **5% overhead**, which is
the target. The ceiling binds, but the result is still workable.

**H does still vary — with the chip.** Same config, same link:

| chip | t_step | H |
|---|---|---|
| RTX 4090 | 0.935s | 500 (ceiling) |
| RTX 3090 | 2.169s | 232 |
| RTX A5000 | 2.778s | 181 |

A slower chip means a larger `t_step`, which means sync is cheaper relative to
compute, which means a smaller `H`. So `H` remains a real per-pool decision on
RunPod, driven by chip rather than by link.

### 5.5 Projection

```
ETA     = total_steps × t_step + (total_steps / H) × t_sync
credits = ETA_hours × n_workers × credits_per_hour
```

Dropping the second term promises a runtime the network cannot deliver. Anything
derived from these is **projected**, and the handbook's cheat list marks
presenting a projection as a measurement *fatal*.

---

## 6. Calibration — `agent/calibrate.py`  **BUILT**

### The moat, as code rather than as a claim

The handbook's moat is a story: *"run history accumulates per chip type."* These
constants are that story made concrete.

```python
@dataclass(frozen=True)
class Calibration:
    mfu: float           # fraction of peak FLOPS actually reached
    act_overhead: float  # multiplier on predicted activation bytes
    sync_eff: float      # fraction of nominal link bandwidth an all-reduce gets
    n_samples: int       # 0 == still a cold-start guess. The UI must say so.
```

`CalibrationStore` keys them by `chip_class`. The `ampere_24gb` constants are
fitted **only** from `ampere_24gb` probes. A new chip starts at defaults and
improves with its own data, and nothing above it changes.

### Fitted by inverting the cost model — no black box

```python
mfu          = work / (t_step_measured × rate_before_mfu)
act_overhead = (peak_vram_measured − static) / act_predicted
```

`observe()` and `predict_t_step()` call the **same two functions**
(`effective_flops`, `rate_before_mfu`), so they are exact inverses by
construction rather than by two copies of the same arithmetic staying in sync.
`test_observe_inverts_predict` pins it.

Combining multiple probes uses the **median**, for the same reason the probe
runner drops warmup and takes the median step time: one thermally throttled run
should not move the constant.

### Cold-start defaults

| chip_class | mfu | rationale |
|---|---|---|
| `ampere_24gb`, `ada_24gb`, `default` | 0.35 | mid-range, deliberately not optimistic |
| `turing_16gb` | 0.30 | no bf16; fp16 + GradScaler costs a little |
| `cdna_amd` | 0.25 | ROCm kernels lag CUDA on small models |

`act_overhead = 1.3`, `sync_eff = 0.75` everywhere.

**These are guesses.** See §10.

### Error reporting

```python
error_report(predicted, measured) -> {t_step_pct, tokens_per_s_pct,
                                      peak_vram_pct, t_sync_pct}
```

The denominator is the **measurement**. Predicting 2.2s when the card did 2.0s is
a 10% error; dividing by the prediction instead would flatter every over-estimate,
which is the direction we would be tempted to round.

Persistence today is in-memory. The durable version is Jack's S-8 `probes` table,
whose `chip_class` column is the same Contract 5 string — swapping the store for
a SQL read changes nothing above it.

---

## 7. Inference cost model — **DESIGNED, NOT BUILT**

Groundwork in place: `ChipSpec.mem_bw_gbs` is populated. No code yet.

Inference needs a **different cost model**, not a variant of the training one.

### Why it differs

```
prefill  compute-bound   ≈ 2 × params × prompt_tokens / (tflops × mfu)
decode   memory-bound    ≈ params × dtype_bytes / mem_bw     per forward pass
```

Decode reads the entire weight matrix to produce **one token per sequence**. That
is a memory-bandwidth problem, not a FLOPs problem, which is why `mem_bw_gbs` is
on `ChipSpec` and why an MI300X (5.3 TB/s) separates from a 4090 (1.0 TB/s) far
more on decode than its FLOPs ratio suggests.

### Why batching changes everything

A batched decode step reads the weights **once** for the whole batch. So decode
throughput scales close to linearly with concurrency until it turns
compute-bound. The cap is KV cache:

```
kv_per_seq   = 2 × layers × d_model × seq_len × dtype_bytes
max_seqs     = (vram − weights) × gpu_memory_utilization / kv_per_seq
```

This is the mechanism behind two of the handbook's inference levers: raising
`gpu_memory_utilization` and lowering `max_model_len` both buy KV headroom, which
buys concurrency, which buys throughput.

**It is also why the measurement protocol is what it is.** At concurrency 1 there
is nothing to batch and vLLM ≈ HuggingFace `generate`. The handbook specifies
**64 concurrent requests, `ignore_eos=True`, fixed `max_tokens`** — `ignore_eos`
is the inference form of the fixed-work invariant. Without it the two systems
generate different amounts of work and the comparison collapses.

### Planned additions

`InferenceSpec(concurrency, prompt_tokens, output_tokens, quantization)` and
`predict_output_tokens_per_s(...)`, reusing `ChipSpec` and `ModelSpec` unchanged.

---

## 8. Migration — **DESIGNED, NOT BUILT**

No code. What is settled is the mechanism and, more usefully, the constraints
found in the live RunPod API.

### Mechanism

Migration is not a new primitive: it is *a worker leaving* plus *a worker
joining*, at a sync boundary.

```
1. wait for next sync     phase=waiting_sync    ← the whole trick
2. save safetensors       phase=saving
3. transfer               phase=transferring
4. target loads           phase=loading
5. resume                 phase=resumed
```

Right after a sync every worker holds an identical global θ. That instant is the
only point where there is nothing else to move — which is why no inner-optimizer
state is migrated, and why step 1 must be **shown** rather than hidden. Why it
waits is the explanation of why it is safe.

### Constraints found in the RunPod API

1. **Global networking is NVIDIA-only.** An MI300X pod cannot join the private
   `.runpod.internal` network at all. The cross-vendor checkpoint has to move by
   network volume or the S3-compatible API — not over the pod mesh.
2. **Network volumes are datacenter-scoped.** Both pods must sit in the same DC
   to share one. Global volumes are region-independent but read-optimised and
   explicitly not for checkpointing.
3. **This is survivable because of a coincidence worth exploiting: MI300X (LOW)
   and RTX 4090 (HIGH) are both stocked in `EU-RO-1`.** One network volume there
   serves both sides of the cross-vendor migration. Pin both pods to `EU-RO-1`.
4. **Torch versions differ across the vendor line** — `2.11.0+cu128` against
   `2.10.0+rocm7.0`. safetensors is format-stable and vendor-neutral, so this
   should be fine; it is on the untested list until a real round-trip runs.

### Cost model to build

```
t_migrate = wait_for_sync + save + transfer + load
            ≈ (H/2 × t_step) + ckpt_bytes/disk_bw
              + ckpt_bytes/link_bw + ckpt_bytes/disk_bw
```

`wait_for_sync` averages half a round and **dominates everything else** at large
`H`. At `H = 500` and `t_step = 0.935s` that is ~4 minutes of waiting before any
bytes move. The proposal card has to surface that honestly, and the migration
rule — *expected gain > migration cost* — has to include it or the agent will
propose migrations that never pay back.

---

## 9. Technology choices

| Layer | Choice | Why |
|---|---|---|
| Contracts | pydantic v2, `extra="forbid"` | typos become `ValidationError` at the boundary; the fixed-work invariant is a model validator, not a convention |
| Packaging | `uv` + universal lock | torch is a different package per machine; one lock keeps three boxes identical |
| CUDA wheels | `cu128` | RunPod hosts report 12.8 / 13.x. cu124 is unavailable on most of them |
| ROCm wheels | `rocm7.0` | gives torch 2.10.0 against CUDA's 2.11.0 — near parity. rocm6.2 capped at 2.5.1 |
| GPU supply | RunPod REST API v2 | `GET /v2/catalog/gpus` is a real market with real prices — `pick_chips()` stops being a slogan |
| Collective | gloo | NCCL does not cross NAT; DiLoCo's infrequent sync makes CPU-side collectives sufficient |
| Checkpoint | safetensors | vendor-neutral, weights only |
| Agent tier 1 | rule-based Python | deterministic, demo-safe, honest in Q&A |
| Agent tier 2 | LLM picks among rule-generated candidates | **BLOCKED** — OpenAI key valid, balance $0 |
| Agent surface | MCP server | the server needs no LLM key; the client supplies it |

### On the MCP surface

An MCP server exposes tools; it does not call an LLM. The key belongs to the
client. The hybrid design chosen for the agent means the **rules layer completes
without any key**, and the LLM selection layer plugs in when credit is loaded —
so the $0 balance blocks one layer, not the project.

---

## 10. Verified against reality vs. assumed

The distinction this whole document exists to preserve.

### Verified — read from the live RunPod API

- GPU catalog: VRAM, `price.secure` / `price.community`, availability, per-DC stock
- Global networking capped at **100 Mbps**, NVIDIA-only, 17 datacenters
- Network volumes are datacenter-scoped
- Pod host CUDA is **13.0** on the 4090 we launched; 12.4 unavailable on most stock
- MI300X and RTX 4090 both stocked in `EU-RO-1`
- Account balance $515 — budget is not the constraint; **stock is**
- RunPod groups its own catalog by `pool`: `AMPERE_24`, `ADA_24`, `AMPERE_48`,
  `BLACKWELL_96`. The same axis as our `chip_class`

### Assumed — no hardware has checked these

- **Every calibration constant.** `mfu`, `act_overhead`, `sync_eff` are all
  cold-start guesses with `n_samples = 0`
- `tflops_bf16` and `mem_bw_gbs` in the `CHIPS` table — nominal peaks. They only
  need to be *self-consistent*, since fitted MFU multiplies through them and
  cancels a uniform error. Mixing sparse and dense numbers across chips would
  genuinely break it
- `_ACT_PER_TOKEN = 12`, `SDPA_ATTENTION_SPEEDUP = 1.4`, `COMPILE_SPEEDUP = 1.25`
- `occupancy`'s shape and its `MFU_REFERENCE_TOKENS = 32,768` anchor
- That cu128 wheels run on a CUDA 13.0 host
- That rocm7.0 wheels match the MI300X host driver

### The number that proves the point

Cold-start constants project **3.62x** for baseline → optimized. The handbook's
own worked example shows **1.39x** (14,200 → 19,800 tokens/s).

The model is currently far too optimistic. **Nothing from it goes on screen until
probes have fitted the constants and `error_report()` can be shown beside it.**

---

## 11. Open problems

**Cheapest rate is not cheapest job.** `pick_chips()` sorts by
`credits_per_hour`. The simulator says total cost to finish the same 8,000 steps
runs MI300X $2.32 < 4090 $3.25 < A5000 $3.51 < 3090 $5.07 — the *most* expensive
chip per hour finishes cheapest, because it is ~10x faster. The rule should sort
by projected cost-per-job. Deferred until the constants are calibrated, because
the MI300X figure rests on a guessed MFU of 0.25.

**`chip_class` conflates architecture with capacity.** `classify_chip()` only
checks `vram_gb >= 20`, so a 48 GB A6000 and a 24 GB 3090 both land in
`ampere_24gb`. Impact is limited — `micro_batch` is searched per worker and dtype
is the same for both — but their probes pool into one calibration. Fixing the
taxonomy means changing `contracts.py`, which requires all three of us.

**Two axes unbuilt.** Inference (§7) and migration (§8) have equations and
constraints but no code.

**`predict_gpu_util` is weak** and documented as such. It models neither
dataloader stalls nor launch gaps. Prefer a measured value wherever one exists.

**Nothing has been calibrated.** The first real probe changes every number in
this document, and that is the intended sequence.
