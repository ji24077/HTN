# Status — what is set up, what is left

Ji's working log. Design rationale lives in `simulation.md`; this file is the
operational picture: what exists, what is verified, what to do next.

Last updated at commit `84b07ec`. Demo operation and the mistakes worth not
repeating are in `demo.md`; design rationale is in `simulation.md`.

---

## TL;DR

**Measured on real GPUs.** Two RunPod machines, 88 tests green.

Training (RTX 4090, Qwen2.5-0.5B, 500 steps): `json_parse_rate` 0.000 → 1.000,
`exact_match` 0.000 → 0.835, `held_out_loss` 1.3553 → 0.0188. Per-field, the
two fields the model used to invent — `age` and `year` — now score 1.00 and
0.995 after the dataset was regenerated with nullable fields.

Long-context serving (RTX 3090, Qwen3-4B-Instruct, 30,857-token policy):
request-level prefix caching takes a verdict from **10.80s to 2.88s**, and time
to first token from **8.70s to 0.64s**. Same prompt, same model, greedy, warm-up
discarded. The claim is about the repeated prefix's prefill — generation itself
is unchanged, and the two paths are NOT byte-identical (see `demo.md`).

**The cost model's two halves did not fare the same.** Memory came in 1.4%
under a measured 13.80 GB peak and is now what job placement trusts. Time was
out by nearly 7x, which is a structural error rather than a calibration gap —
inverting it gives an MFU above 1.0. Unfixed; it needs a batch sweep.

**ROCm runs.** MI300X on RunPod: torch `2.10.0+rocm7.1.1`, `available True`,
`chip_class cdna_amd`, bf16 4096x4096 matmul in 207.8 ms. `check_env.py` passes
unmodified. What is *not* verified is the `rocm7.0` wheel index this repo pins —
the host is 7.1.1 and the image's own torch at `/opt/venv` is what ran.

**Migration runs.** `migrate-nextgen` 4090 → 3090: exact_match 0.835 → 0.825,
inside the 2% gate, `training_resumed`. AMD → NVIDIA is the remaining gap.

---

## 1. Accounts and keys

All three keys live in `.env`, which is gitignored (`.gitignore:4`) and has been
checked against `git status` after every commit. **No key is in the repo.**

| Service | Env var | Status |
|---|---|---|
| RunPod | `GPUSHARE_RUNPOD_API_KEY` | ✅ works · **balance $513.63** |
| Baseten (inference) | `BASETEN_API_KEY` | ✅ works · credit confirmed by a real completion |
| Baseten (MCP) | `BASETEN_MCP_API_KEY` | ✅ works · not wired to anything yet |
| OpenAI | `OPENAI_API_KEY` | ⚠️ key valid, **balance $0** — unused |

**A models-list 200 does not prove there is credit.** The OpenAI key returns 200
for `GET /v1/models` and then `429: You have no credits remaining` on an actual
completion. Always verify with a real call.

⚠️ All four keys were pasted in plaintext into a chat transcript. **Rotate them
after the event.**

### Naming rule

Prefix with `GPUSHARE_` only when the service has no standard env var name.
`OPENAI_API_KEY` and `BASETEN_API_KEY` are unprefixed because the SDK and CLI
read those exact names.

The two Baseten keys are split by purpose so either can be revoked alone.

---

## 2. Local tooling (Ji's Mac)

| | |
|---|---|
| `uv` | 0.11.5 |
| Python | 3.11.15 in `.venv` (`.python-version` pins 3.11) |
| `runpodctl` | 2.14.0 at `~/.local/bin/`, apiKey configured |
| `gh` | authenticated as `ji24077` |
| SSH | this Mac's `id_ed25519.pub` added to the RunPod account (the pre-existing key from another machine was preserved) |

---

## 3. Repo

`github.com/ji24077/gpushare` — private, `main` only, 5 commits, working tree clean.

`make test` → **38 passed**. `ruff check src/ tests/` → clean.

### Packaging fixes that were required to make `make setup-*` work at all

Both were found by running the thing, not by reading it:

1. **`pytorch-triton-rocm` was not mapped to the ROCm index.** The index is
   `explicit = true`, so uv looked on PyPI, and because the lock is universal
   **every machine failed to resolve — not just the AMD box.**
2. **`cu124` → `cu128`, `rocm6.2` → `rocm7.0`.** Checked against the live RunPod
   catalog: stocked hosts offer CUDA 12.8 / 13.x and report 12.4 unavailable.
   rocm6.2 also capped torch at 2.5.1 against CUDA's 2.11 — too wide a gap for
   the cross-vendor migration experiment. rocm7.0 gives 2.10.0, near parity.
   rocm7.0 renamed the triton package, so the dependency name moved with it.

Lock now carries `torch 2.11.0+cu128` and `torch 2.10.0+rocm7.0`.
**Neither has run on real hardware.**

---

## 4. Code inventory

| File | Lines | Status |
|---|---|---|
| `contracts.py` | 289 | ✅ frozen, shared. Contracts 1–5 as pydantic |
| `settings.py` | 36 | ✅ + LLM provider settings |
| `agent/specs.py` | 218 | ✅ ModelSpec / DataSpec / ChipSpec / NetSpec |
| `agent/simulate.py` | 233 | ✅ cost model + `SimProber` behind the `Prober` seam |
| `agent/calibrate.py` | 172 | ✅ per-`chip_class` constants, fitted by inverting the model |
| `agent/llm.py` | 176 | ✅ LLM picks among rule-generated candidates, falls back to rules |
| `agent/chips.py` | 52 | ⚠️ structure only — `decide()` bodies are thin |
| `agent/router.py` | 41 | ⚠️ `compute_H` / `pick_chips` / `check_stragglers` exist; **`plan()` and `reconcile()` are missing** |
| `agent/mock.py` | 8 | ❌ S-2 — still `raise SystemExit("not implemented")` |
| `dashboard/` | — | ❌ S-3 — empty |

Not Ji's, untouched, all still stubs: `worker/daemon.py` (N-3),
`worker/cli.py` (N-5), `trainer/loop.py` (T-1), and `server/main.py` still
points at a `server/app.py` that does not exist.

---

## 5. Verified vs assumed

### Read from the live RunPod API

- Pod-to-pod **global networking is capped at 100 Mbps**, NVIDIA-only, 17 DCs
- Network volumes are **datacenter-scoped**
- Launched pod reports host **CUDA 13.0**; 12.4 unavailable on most stock
- **Community cloud is effectively empty right now** — the cheap price column is
  not purchasable. Secure stock: RTX 4090 HIGH, L4 MEDIUM, A5000 / 3090 / A40 /
  A6000 / MI300X all LOW, **Tesla V100 NONE**
- **MI300X and RTX 4090 are both in `EU-RO-1`** — one network volume can serve
  both sides of a cross-vendor migration
- RunPod groups its own catalog by `pool`: `AMPERE_24`, `ADA_24`, `AMPERE_48`.
  Same axis as our `chip_class`

### Verified by calling it

- Baseten is OpenAI-compatible; `response_format: {"type":"json_object"}` works
  on GLM-5.3-Flash, DeepSeek-V4.1-Flash and gpt-oss-120b
- The LLM layer falls back to the rules in **366 ms** against a dead endpoint

### Assumed — no hardware has checked any of this

- **Every calibration constant** (`mfu`, `act_overhead`, `sync_eff`), all at
  `n_samples = 0`
- The `tflops_bf16` / `mem_bw_gbs` table
- That `cu128` wheels run on a CUDA 13.0 host
- That `rocm7.0` wheels match the MI300X driver
- That safetensors round-trips cleanly between torch 2.11+cu128 and 2.10+rocm7.0

**The number that keeps this honest:** cold-start constants project **3.62x**
for baseline → optimized where the handbook's own worked example shows **1.39x**.
The model is far too optimistic. Nothing from it goes on screen until probes
have fitted the constants and `error_report()` can be shown beside it.

---

## 6. What to do next

### Right now — a pod is burning money

`sp3yuevblqrj0h` (RTX 4090, EU-RO-1) has been **RUNNING for ~1h52m at $0.74/hr
and was never used** — SSH was blocked before it could be reached. Either use it
for step 1 or kill it:

```bash
runpodctl remove pod sp3yuevblqrj0h
```

### 1. One real probe — this unblocks more than anything else

```bash
rsync -az -e "ssh -i ~/.ssh/id_ed25519 -p <port>" \
  --exclude .venv --exclude .git --exclude .env \
  ./ root@<ip>:/workspace/gpushare/

# on the pod
curl -LsSf https://astral.sh/uv/install.sh | sh && source $HOME/.local/bin/env
cd /workspace/gpushare && cp .env.example .env && make setup-cuda && make check
```

**Exclude `.env` deliberately** — a rented pod is someone else's hardware and has
no business holding our API keys.

This single step settles four open questions at once: whether `cu128` works on a
CUDA 13.0 host, whether `classify_chip()` returns the right string on real
silicon, whether `make setup-cuda` resolves, and — once a probe runs — the first
measured value for `mfu`.

### 2. Calibrate, then report the error

Feed the measured probe through `calibrate.observe()`, store it, and print
`error_report(predicted, measured)`. The sim-vs-real gap is the deliverable, not
a diagnostic.

### 3. `S-2` mock emitter — still not written

`docs/ji.md` calls this the highest-return half hour on the board and says to do
it **first**. It is still a `SystemExit`. Nothing about the dashboard can start
until fake events exist.

### 4. `RunPodProber` behind the existing `Prober` seam

`agent/simulate.py` already defines the protocol. Implementing the real side
changes nothing above it.

### 5. Fill the gaps in the rules layer

- `router.plan()` and `reconcile()` — the ticket S-4 snippet has them, the file
  does not
- `pick_chips()` sorts by `credits_per_hour`; the simulator says **cost per
  finished job** ranks differently (MI300X $2.32 < 4090 $3.25 < A5000 $3.51 <
  3090 $5.07 — the most expensive chip per hour is the cheapest per job). Revisit
  after calibration, since the MI300X figure rests on a guessed MFU of 0.25

### 6. Then, in rough order

`S-3` dashboard · inference axis (`simulation.md` §7 has the equations) ·
migration decision logic `S-7` · MCP server exposing the agent's tools

### Blocked on other people

Neither of the two hour-0 blockers has an answer yet: **Jack's N-1/N-2** (is
Tailscale direct, does gloo work machine-to-machine) and **Ethan's T-2a** (does
torchft run here). The handbook wants both by T+3, and both change everyone's
plan.

---

## 7. Known problems

| | |
|---|---|
| **Nothing is calibrated** | every projected number is a guess; the first probe changes all of them |
| **OpenAI balance $0** | unused — Baseten covers the LLM layer, so this blocks nothing |
| **`chip_class` conflates architecture with capacity** | `classify_chip()` only checks `vram >= 20`, so a 48 GB A6000 and a 24 GB 3090 pool into one calibration. Fixing it means changing frozen `contracts.py` — needs all three of us |
| **nanoGPT is deprecated** | successor is nanochat; nanoGPT is torch-2.0-era code against our torch 2.11. Ethan's call (T-1), no impact on the simulator |
| **Stock, not budget, is the constraint** | $513 is ~16x what the whole plan needs. The 4090 is the only chip reliably available; V100 is at zero, so the `turing_16gb` class and the fp16 branch cannot be tested on RunPod at all |
| **H pins to its ceiling on a 100 Mbps link** | true for 124M *and* 30M — shrinking the model does not escape it. Overhead is still ~5%, so it is workable, but the agent should say so rather than quietly clamping |
