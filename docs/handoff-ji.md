# Handoff — Ji's area

`ji.md` is the plan written before anything ran. This is what is actually
true as of `8614b33`, separated into what was measured, what runs but was never
run, and what has never executed at all.

Operating the demo: `demo.md`. Design rationale: `simulation.md`.

---

## The three states things are in

| | Meaning |
|---|---|
| **Measured** | A number in this repo came from hardware, and the command that produced it is recorded |
| **Written, never run** | The code path exists, tests cover its logic, no job has ever executed it |
| **Never executed** | Not even once, and there is a known reason why |

Most of the disagreements today came from treating the second as the first.

---

## Measured

**Training** — RTX 4090, Qwen2.5-0.5B, 500 steps, bf16/sdpa, batch 16×1,
3,072 tokens/step.

```
                  before    after
json_parse_rate    0.000    1.000
exact_match        0.000    0.835
held_out_loss     1.3553   0.0188
hallucination        —      0.185   (27 nullable cases)
per-field: name 1.00 · age 1.00 · year 0.995 · org 0.93 · role 0.885
t_step 0.1374s · peak_vram 13.80 GB
```

`before`'s hallucination rate is not reported. It is 0.000 only because
`json_parse_rate` is 0 — nothing parses, so there is no field to invent into.

**Long-context serving** — RTX 3090, Qwen3-4B-Instruct, 30,857-token policy
prefix, greedy, warm-up discarded.

```
                 first token   total
cache bypassed       8.70s     10.80s
cache hit            0.64s      2.88s
                     13.6x       3.7x
```

Same 30,927 input tokens both ways. The claim is about the repeated prefix's
prefill — generation is unchanged, and the two paths are **not byte-identical**
(`"high"` vs `"High"` at character 14; each path reproduces itself exactly).

**Cost model, checked against the above.** Memory was 1.4% under the measured
13.80 GB peak, and is what job placement now trusts. Time was out by nearly 7x;
inverting it gives an MFU above 1.0, so it is a structural error and not a
calibration gap. Unfixed — it needs a batch sweep.

---

## Written, never run

### Migration — the one you asked about

`start_migration` in `runner.py:1185`. Two kinds, both wired to the UI, and
**zero migration jobs have ever executed.** Local job history:

```
train-and-evaluate 3 · serve-model 13 ·
optimize-inference-speed 1 · optimize-training-speed 1 · migration 0
```

What it does: pulls `model.safetensors` from the source pod, pushes it to the
target with the source's `eval/after.json`, re-runs `scripts/evaluate.py` on the
target over the same `n` samples, and gates the result through `_quality`
against the source's numbers. Vendor-neutral safetensors is the whole premise —
nothing CUDA-specific is serialised.

**`migrate-nextgen` is runnable today and has never been tried.** Preconditions
are all met right now:

```
source 4090   ckpt/model.safetensors 1.26 GB + meta.json   ✓
baseline      .gpushare/runs/38915adc.../eval/after.json   ✓
target 3090   disk 44.2 GB                                 ✓
target 3090   VRAM 0.9 GB                                  ✗ ← serving the 4B
```

The only blocker is that the 3090 is holding the long-context model. Testing it
means stopping that server, running the migration (~1.3 GB down then up, plus a
200-sample eval), and re-serving afterwards. Perhaps ten minutes and about
$0.20. **It is the single largest unverified claim in the demo** — four agent
actions are advertised and two of them are migrations.

Migration now refuses a busy target rather than OOMing on it. It refuses rather
than relocating, unlike training, because both ends are pinned: the source holds
the checkpoint and the target is the thing being argued about.

### Training optimization

`start_training_optimization`. Ran once, and it failed — the job record is in
`.gpushare/jobs/`. Not re-attempted since placement was added.

---

## Never executed

**ROCm. AMD→NVIDIA migration cannot be demonstrated.**

- No AMD pod has ever been rented
- `make setup-rocm` has never run, so the `rocm7.0` wheels are unverified
  against an MI300X driver
- RunPod's Global Networking is NVIDIA-only, so the private-network path is out
- MI300X is $2.39/hr, stock LOW, one datacentre (EU-RO-1)

`make setup-rocm && make check` on an MI300X — roughly $1.2 and half an hour —
is what would turn this from a claim into either a result or a known failure.
Until then the AMD→NVIDIA button will refuse: it requires an AMD source pod and
there is none.

**Full fine-tuning of Qwen3-4B.** 64.3 GB of optimizer state before activations
(bf16 weights 8.0 + grads 8.0 + Adam m/v fp32 16.1 each + fp32 master 16.1). No
24 GB or 48 GB card fits it; only the MI300X would. LoRA would fit in about 8.3
GB, but `train.py`, `evaluate.py` and `serve.py` all assume full-weight
safetensors, so the adapter path does not exist.

---

## Things that bit repeatedly, and what now stops them

| Failure | Now |
|---|---|
| Training on a pod that was serving → OOM two minutes in | Placement asks the GPU and moves the job to a pod with room |
| Finished 500-step run could not save → disk full | Placement checks disk too, and reports it separately, because the fix differs |
| Inference optimization on a pod with no checkpoint → `Repo id must be in the form...` | Refuses by name and says which pod has the checkpoint |
| Dashboard restart lost the served model → reload 8 GB + rebuild a 30K cache | Re-adopted at startup, but only if `/health` names the same weights |
| Reloading the policy OOMed | Old cache released before the new one is built |
| Chat sent messages bare → model continued the document | Wraps with the instruction and the assistant-turn marker when a policy is loaded |
| `JSON ✓/실패` badge scored a security verdict against the person schema | Badge removed; input tokens and cache state shown instead |

---

## Two agents, one repo

`static/index.html` was rewritten by another agent mid-change and every
measurement panel disappeared with it. They were merged back onto the new
structure as Step 5 and Step 6, and a test now asserts each panel's anchor is
present — these numbers exist in `/api/state` and have nowhere else to be seen,
so losing them leaves a page that still looks finished.

If both agents keep editing that file, this will happen again. Split the file or
give one owner.

---

## If you have thirty minutes

1. **Run `migrate-nextgen` once.** It is written, wired, gated, and has never
   executed. Stop the 3090's server, migrate 4090 → 3090, re-serve.
2. **Batch sweep on the 4090** to fix the 7x time error. Five minutes of GPU.
3. **ROCm survival check** if the cross-vendor story is being told at all.

In that order. The first turns two advertised buttons from a claim into a
result; the third decides whether the other two buttons should be on the screen.

---

## Running cost

Two pods, $1.24/hr combined — 4090 $0.74 (holds the fine-tuned checkpoint),
3090 $0.50 (serves the 4B). Stop both when the demo is over; the checkpoint
lives only on the 4090's disk, so pull `model.safetensors` first if it matters.
