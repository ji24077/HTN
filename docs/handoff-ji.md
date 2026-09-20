# Handoff — Ji's area

`ji.md` is the plan written before anything ran. This is what is actually
true as of `02e58a9`, separated into what was measured, what runs but was never
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

**AMD runs.** MI300X, RunPod, 2026-09-19:

```
torch        2.10.0+rocm7.1.1.gitd9556b05   hip 7.1.52802
backend      rocm      vendor amd      available True
gpu          AMD Instinct MI300X       vram_gb 206.1
bf16         True      chip_class cdna_amd     trainable True
bf16 4096x4096 matmul   207.8 ms, peak 0.18 GB
```

`classify_chip` returns `cdna_amd`/`trainable` off the real device, so
`check_env.py` passes unmodified on AMD.

**What this does NOT verify: the `rocm7.0` wheel index this repo pins.** The
host is ROCm 7.1.1 and the image ships its own torch at `/opt/venv`; that is
what ran. `make setup-rocm` was not used. Phin's runs made the same choice and
say why — the project lock would "silently choose a different framework
version". Treat `rocm7.0` in `pyproject.toml` as still unproven.

**Cost model, checked against the above.** Memory was 1.4% under the measured
13.80 GB peak, and is what job placement now trusts. Time was out by nearly 7x;
inverting it gives an MFU above 1.0, so it is a structural error and not a
calibration gap. Unfixed — it needs a batch sweep.

---

## Written, never run

### Migration — now run

`start_migration` in `runner.py`. **`migrate-nextgen` executed successfully**,
RTX 4090 → RTX 3090, on our own hardware:

```
before  json_parse 1.000 · exact_match 0.835   (200 cases)
after   json_parse 1.000 · exact_match 0.825
validation ok · tolerance 0.02 · delta_exact -0.010
  per-field  name 0 · age 0 · year 0 · org -0.005 · role -0.005
  "model quality preserved"
```

Phin's version, which replaced the original here, does more than copy a
checkpoint: it trains the source to step 4 of 8, pauses, relays the bundle, and
resumes on the target — `training_resumed` is in the result. Migration also now
refuses a busy target rather than OOMing on it, and refuses rather than
relocating, because both ends are pinned: the source holds the checkpoint and
the target is the thing being argued about.

**AMD→NVIDIA has not been run by us.** Phin measured MI300X quality (below);
moving a live job across vendors is the remaining gap.

### Training optimization

`start_training_optimization`. Ran once, and it failed — the job record is in
`.gpushare/jobs/`. Not re-attempted since placement was added.

---

## Never executed

**RunPod's Global Networking is NVIDIA-only**, so the private-network path
between an AMD and an NVIDIA pod is out. Transfers go over SSH/SCP.

(ROCm itself has now run — see below.)

**Full fine-tuning of Qwen3-4B.** 64.3 GB of optimizer state before activations
(bf16 weights 8.0 + grads 8.0 + Adam m/v fp32 16.1 each + fp32 master 16.1). No
24 GB or 48 GB card fits it; the MI300X at 206 GB would.

LoRA does exist — `train.py --method lora --lora-rank`, from the merge. An
earlier version of this file said the adapter path did not exist; that was
written before reading Phin's branch and was wrong.

---

## Renting an AMD pod — the recipe, because guessing cost $1.36

Three attempts. All three failure modes were already documented in
`docs/nvidia-amd-migration.md` and `demo/HANDOFF.md`; reading those first would
have cost nothing.

```json
{
  "imageName": "rocm/pytorch:rocm7.1.1_ubuntu24.04_py3.12_pytorch_release_2.10.0",
  "gpuTypeIds": ["AMD Instinct MI300X OAM"],
  "containerDiskInGb": 150,
  "ports": ["22/tcp"],
  "cloudType": "SECURE",
  "dockerStartCmd": ["bash", "-lc", "apt-get update -qq; apt-get install -y -qq openssh-server; mkdir -p /run/sshd /root/.ssh; printf '%s\\n' \"$PUBLIC_KEY\" > /root/.ssh/authorized_keys; chmod 700 /root/.ssh; chmod 600 /root/.ssh/authorized_keys; /usr/sbin/sshd -D -e"]
}
```

Three things that are not obvious:

**The AMD image does not start sshd.** RunPod's NVIDIA images do, so the same
pod body that works for a 4090 leaves an MI300X at `uptimeInSeconds: 0` with no
`publicIp` forever. It looks like a slow image pull. It is not. Twenty-two
minutes and $0.88 went here.

**torch is in `/opt/venv`, not on `python3`.** `python3 -c "import torch"` fails
on a working pod. Use `/opt/venv/bin/python`, and `pip install` the project's
non-torch deps into that venv rather than running `make setup-rocm`.

**Match the image to the host, not to `pyproject.toml`.** The first attempt used
`rocm6.4.1 / py3.10`: Python below this project's floor, and a ROCm major that
the pinned `rocm7.0` wheels would not have matched either. $0.48.

And the catalog lies about stock: MI300X reported `lowestPrice: null` and
`stockStatus: null` while creation succeeded three times. **Decide availability
by attempting to create, not by reading the catalog** — the "stock LOW" note in
earlier docs came from that same unreliable field.

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

1. **Cross-vendor migration, MI300X → NVIDIA.** Both pods are up and the AMD
   one is verified. This is the last advertised button with no result behind it.
2. **Batch sweep on the 4090** to fix the 7x time error. Five minutes of GPU.
3. **Re-run training optimization.** It failed once, before placement existed.

---

## Running cost

Four pods, **$3.90/hr** — MI300X $2.39, 4090 $0.74 (holds the fine-tuned
checkpoint), 3090 $0.50, A5000 $0.27. The MI300X is most of it. Stop them all
when the demo is over; the checkpoint
lives only on the 4090's disk, so pull `model.safetensors` first if it matters.
