# NVIDIA to AMD training continuation

This branch adds a migration test for the Qwen extraction model: train on NVIDIA,
pause at an optimizer-step boundary, move a complete checkpoint to AMD, compare
predictions before updating any weights, then continue the same training job.
Local tests validate checkpoint restoration and orchestration. Two paid attempts
stopped before model training; hardware migration has not passed.

This is a checkpoint-portability test, not a CUDA/HIP source translator. The
Qwen workload extracts structured fields from text; it has not been trained on
code translation. The product requirement now includes accepting an existing
workload and converting vendor-specific code in both directions. Neither that
conversion layer nor an AMD-to-NVIDIA return test is implemented by this runner.
Preserve this distinction in demos and teammate handoffs.

## Approved rental and current state

- Provider: RunPod, Secure Cloud, EU-RO-1.
- Source: one NVIDIA GeForce RTX 4090 (24 GB), quoted at $0.74/hour.
- Target: one AMD Instinct MI300X OAM (192 GB), quoted at $2.39/hour.
- Total authorization: $15 USD; delete both Pods after collecting results.
- Deadline: 90 minutes from session preparation, including setup time.
- Container disks only; no persistent or network volumes.

These are observed quotes, not reservations. The session guard rechecks stock
and prices before each creation. With a $0.10/hour combined storage allowance,
the planned 90-minute compute/storage estimate is $4.845. The guard rejects
a plan exceeding the authorized budget. The actual bill must be collected
separately; estimates do not include unknown account-specific taxes.

RunPod MCP is connected. Paid MI300X inventory changed from `NONE` at 09:45 UTC
to `LOW` at 09:52 UTC on 2026-09-19. A real allocation then succeeded for both
approved GPUs in EU-RO-1. The first attempt stopped during package setup when
Windows denied an atomic job-log replacement inside OneDrive. No model training
ran in that attempt. Both Pods were deleted and their absence independently
verified; the conservative elapsed-time cost estimate was $0.4194.

Job persistence now retries brief sharing locks and records permanent failures.
The retry controller keeps frequently rewritten job logs outside OneDrive and
reserves $0.50 for the first attempt, leaving a $14.50 ceiling for the retry.
Only successful training steps and process GPU memory measurements count as
workload evidence; provider `RUNNING` status or device-wide allocated memory
alone does not establish that our model ran. Alternative-provider research is
stopped because this experiment must use RunPod for the sponsor track.

The second session, `migration-20260919-3f0391f297eb`, ran from 10:05:04 to
10:10:10 UTC. RTX 4090 Pod `q8mm5vemo4vv4d` passed the CUDA BF16 smoke test.
MI300X Pod `6kyo6xfy67tepj` failed the ROCm BF16 smoke test: a 76 MiB allocation
failed with 191.98 GiB total GPU memory and zero bytes free. PyTorch reported
only 1,024 bytes allocated by this process and 2 MiB reserved but unallocated.
This happened before model loading or training. The underlying cause of the
unavailable device memory is not established; it must not be described as this
model using 192 GiB. Both Pods were deleted and MCP independently listed none.

The second elapsed-time cost estimate is $0.2746; combined with the first
attempt, the estimate is $0.6940. Provider billing records were not yet returned,
so these are estimates, not a final invoice. Both attempts completed zero model
optimizer steps. The second attempt's independent two-second GPU samples also
showed zero utilization throughout setup. There are no migrated predictions,
resumed training results, or conversion results to claim from these rentals.

The runner now probes each image's native PyTorch runtime before installing
anything, checking AMD first. It records free/total memory and runs a small BF16
forward/backward check. A failed probe preserves JSON evidence and blocks all
package installation and model training. This new guard has local regression
coverage; it has not yet been exercised on another paid rental.

The second job's frozen input copies are in
`.gpushare/runs/c84a0a2825544ee38c70e42ea855a333/input/`. Sanitized job logs,
session results, and utilization evidence are under `.cache/migration-results/`
with the session ID prefix. These are ignored runtime artifacts. The next run
must check native GPU allocation before package installation and maintain the
remaining cumulative budget; there is no active rental now.

RunPod's inspected REST v2 schema and runpodctl 2.14.0 expose no server-side Pod
termination deadline. Our watchdog is an independent local process, with a
second cleanup path in the controller's `finally` block. It cannot enforce a
hard provider-side spending cap if the computer sleeps, loses connectivity, or
powers off. Keep the machine awake and connected throughout the rental.

## What is preserved

The portable directory contains normal Hugging Face model or adapter files,
the saved tokenizer, and `training-manifest.json`, `training-state.json`, and
`training-state.safetensors`. It preserves trainable weights, optimizer tensors
and counters, global step, sampler state, RNG state, and scaler state when used.
Manifest hashes detect incomplete transfers and changed training data or
configuration. LoRA checkpoints also fingerprint the frozen base weights.
There is no pickle loading in this checkpoint format.

`--init-adapter` initializes weights for a new job. It does not resume an old
optimizer. `--resume` restores the complete job. `--steps` is the final global
step, and `--stop-after` is an absolute pause boundary:

```bash
python scripts/train.py --method lora --steps 8 --stop-after 4 \
  --job-id migration-test --out ckpt/source
python scripts/train.py --resume ckpt/source --steps 8 --out ckpt/resumed
```

CPU tests require exact uninterrupted-versus-resumed equivalence for full
training and LoRA, including a fresh-process restore. CUDA and ROCm do not
share bit-identical random generators or kernels. The first hardware test
therefore requires dropout-free training; a backend change with active dropout
fails. Cross-vendor equivalence is evaluated through step/state integrity,
finite training results, and held-out prediction quality.

## Runtime and SSH

Both Pods use an isolated `.migration-venv` with PyTorch 2.10.0,
Transformers 5.17.0, and PEFT 0.21.0. PyTorch comes from the `cu128` index on
NVIDIA and `rocm7.1` on AMD. The usual project's CUDA/ROCm dependency lock is
not used to silently choose a different framework version for this experiment.

The observed official NVIDIA template is `runpod-torch-v280`, image
`runpod/pytorch:1.0.2-cu1281-torch280-ubuntu2404`; the isolated environment
installs the matching test framework version. AMD's published Python 3.12 image
is `rocm/pytorch:rocm7.1.1_ubuntu24.04_py3.12_pytorch_release_2.10.0`.
AMD's documentation labels that image release a preview. Validate host driver,
GPU visibility, BF16 matmul/backward, and exact package versions before training.

A dedicated temporary SSH key has been generated outside the repository.
Its private key stays local. Expose `22/tcp` and use each Pod's direct SSH
endpoint; RunPod's proxy shell does not support SCP. The AMD image needs an
explicit SSH startup command. Account public-key registration must preserve
existing keys, and cleanup must remove only this test's key if registered.

The runner supports native Windows SSH/SCP with a tar transfer fallback.
Project upload includes source, scripts, data, and dependency files; it excludes
`.env`, local caches, and credentials. Frozen input data and full checkpoints
are hash-verified after transfer. RunPod credentials never go to the Pods.

## Bounded run sequence

1. Refresh the MCP connection and verify the Pod list.
2. Refresh catalog quotes, stage the exact Pod bodies and public key, and run
   local tests. Do this before starting the rental deadline.
3. Prepare a unique session with `scripts/runpod_session.py prepare`, then start
   `watchdog` in a separate hidden process. A fresh heartbeat is mandatory
   before `create` can send any paid request.
4. Create the pair once. On an ambiguous response, reconcile by the unique
   session name instead of blindly sending another creation request.
5. Poll direct SSH readiness with a bounded timeout, then run
   `run_resume_migration` from `gpushare.dashboard.migration`.
6. Initialize the source job from the validated `ckpt/laptop-lora-v2` adapter;
   this starts a new optimizer. Train NVIDIA to step 4 of 8, evaluate on 50
   frozen examples, and relay the full bundle locally to AMD. An all-invalid
   baseline cannot demonstrate useful prediction preservation.
7. Evaluate the transferred checkpoint on AMD before resuming. Require the
   same dataset/prompt identity, inference settings, and aggregate plus
   per-field regression gates. Then resume to step 8 and evaluate again.
8. Copy reports/checkpoints locally. Only a successful final gate may update
   the dashboard's latest result. Always call session cleanup in `finally` and
   verify both owned Pod IDs are absent, even if setup or training fails.

The helper accepts already-provisioned SSH endpoints and does not rent or
terminate infrastructure itself. Source and target dictionaries identify each
Pod and vendor; SSH dictionaries contain `ip`, `port`, and local private-key
path `key`. Start with `total_steps=8`, `stop_after=4`, and `eval_n=50`.

## Inputs, outputs, and interpretation

Inputs are `data/train.jsonl`, `data/heldout.jsonl`, optional initial adapter
weights, and the chosen training configuration. The runner freezes copies in
`.gpushare/runs/<job-id>/input/` before uploading them.

Outputs live under `.gpushare/runs/<job-id>/`: hardware preflight reports,
source and resumed checkpoint bundles, held-out evaluations, and a migration
report. Session ownership and termination evidence live under
`.gpushare/runpod-sessions/`. These runtime folders are ignored by Git.

An eight-step run tests the handoff plumbing; it does not establish that either
GPU is faster, cheaper for production training, or better for model quality.
A failed preflight or quality gate is a failed migration test, even when the
checkpoint loads successfully. Preserve that evidence before terminating Pods.

References: [RunPod SSH](https://docs.runpod.io/pods/configuration/use-ssh),
[RunPod networking](https://docs.runpod.io/pods/networking),
[PyTorch version installation](https://pytorch.org/get-started/previous-versions/),
[AMD ROCm 7.1.1 images](https://rocm.docs.amd.com/projects/install-on-linux/en/docs-7.1.1/install/3rd-party/pytorch-install.html).
