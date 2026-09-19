# Handoff for Ji: GPU migration, optimization and measured evidence

Date: 2026-09-19. Branch: **`phin/handoff-gpu-benchmarks`**.
PR destination: **`ji-phin-agentinfra`**. Nothing has been merged.

This branch gives you runnable benchmark/validation code, the exact saved
adapters, raw outputs, and the failures to investigate next. Start by reproducing
the saved report without renting anything. Then integrate the quality gate with
the model and serving path you intend to ship.

## Read this before integrating

This work starts at `febea54`, the head of
[PR #7: CUDA/HIP translation and checkpoint migration](https://github.com/ji24077/HTN/pull/7).
It also includes [PR #6: local training and evaluation](https://github.com/ji24077/HTN/pull/6)
through its ancestry. Both PRs were still open at handoff. The new PR therefore
overlaps those earlier PRs; it is not three independent sets of changes.

Your latest branch was `84b07ecd97a30490da6a55f537e89a8036e3fa2f` when checked.
It has newer nullable fields, data, streaming/prefix-cache behavior, placement
checks and chat optimization controls. **Those changes are not the application
snapshot validated by this benchmark.** A read-only merge preview found a
conflict in `scripts/train.py`. No merge or rebase was applied. Resolve training
changes deliberately and rerun tests against the latest schema/prompt/server;
an automatic textual merge of the other files does not establish compatibility.

For the changes added after PR #7:

```bash
git fetch origin phin/handoff-gpu-benchmarks phin/nvidia-amd-migration
git switch --track origin/phin/handoff-gpu-benchmarks
git diff origin/phin/nvidia-amd-migration...HEAD --stat
```

Use a clean checkout/worktree to preserve this measured snapshot while doing
integration separately. Do not overwrite your newer data or serving code with
this older snapshot just to make the saved benchmark pass.

## What was done

1. Improved local synthetic-data training and evaluation, including answer-only
   loss projection, padding reduction, memory/speed operating points, stricter
   quality comparisons and retained failed candidates. See
   [the training report](../docs/optimization-v3.md). The faster short local
   training profile was about 2.76x faster per step but used more GPU memory.
   It did not establish a better replacement model.
2. Built and ran the native CUDA -> HIP and HIP -> CUDA demonstration, including
   training and checkpoint/Adam-state continuation on RTX 4090 and MI300X.
   [The translation report](RUN_RESULTS.md) contains actual input/output source
   and evidence. This covers the embedded AXPY kernel and fixed small training
   job, not arbitrary CUDA projects/dependencies. The broad kernel speed gate
   failed; only the largest tested input passed.
3. Measured sentence-to-JSON inference on **10 distinct NVIDIA models plus AMD
   MI300X**. The expansion completed nine new runs and reused the earlier 4090
   and MI300X evidence. A 5090 container never finished starting within ten
   minutes; it was terminated, retained as a failure, and replaced by a 5080.
4. Validated candidates against the unchanged `scripts/evaluate.py` on 300 cases,
   retained per-case predictions, and tested the unchanged HTTP server through
   actual SSH tunnels. The nine new runs retain 5,400 validation predictions,
   1,170 timed requests and 117 actual HTTP requests. These are repeated runs
   of fixed suites, not that many distinct input sentences.
5. Added bounded rental profiles, exact GPU checks, aggregate budget reservations,
   ownership-restricted cleanup, report recomputation and regression tests.

## What the performance means

The headline times are median resident response times for one sentence with the
model already loaded. The fixed 13-sentence timing suite runs five rounds per
engine, with two warmups per sentence/engine. Loading, compilation and network
are separate from these times. Batch size, model, precision and decoding work
are fixed. Host CPU, vendor runtime and a single sampled host per GPU affect
results, so this is not a universal silicon ranking.

| Same GPU | Original response | Compiled response | Observed ratio | Exact values on all 300 cases |
|---|---:|---:|---:|---|
| RTX A5000 | 1.383 s | 0.234 s | 5.91x | Preserved; optimization accepted |
| L40S | 1.702 s | 0.254 s | 6.69x | Preserved; optimization accepted |
| RTX 4090 | 0.639 s | 0.142 s | 4.51x | 2 changed; rejected |
| MI300X | 1.136 s | 0.248 s | 4.58x | 13 changed; rejected |

Only A5000 and L40S passed the combined timing and 300-case checks against their
own unoptimized reference. **No optimized destination passed the full migration
gate from the original 4090.** In particular, compiled AMD changed 15/300 answers
against that 4090 reference, including five previously correct records becoming
incorrect. It must not be presented as an accepted faster migration.

The rule agreed for this experiment was identical parsed JSON values, not merely
unchanged average accuracy. Even a changed answer that improves a label match
fails that rule. Invalid JSON always fails. The two accepted optimizations each
preserve 270/300 correct records (90%); parity does not turn wrong answers into
correct ones. Full numbers and concrete failures are in the
[matrix](results/hardware-matrix-2026-09-19/README.md) and
[diagnostic notes](hardware/FINDINGS.md).

## Models, inputs and outputs

The speed experiment uses **experimental v3b**, which had already failed a
date-field regression gate. **v2 remains the extraction CLI default.** This
branch publishes both exact adapters; it does not promote v3b or establish v2
latency across the matrix. [Checkpoint setup and attribution](checkpoints/README.md)
explain how to download the public base into a dedicated cache.

| Item | Location |
|---|---|
| Default / experimental adapters | `ckpt/laptop-lora-v2/`, `ckpt/laptop-lora-v3b/` |
| Checkpoint and base file identities | `demo/checkpoints/manifest.json` |
| 13 latency inputs, labels and raw output runs | `demo/results/inference-latency-2026-09-19/` |
| 300 labelled regression cases and source manifest | `demo/validation/cases300.jsonl`, `cases300.manifest.json` |
| Existing-code comparisons and default-v2 smoke test | `demo/results/existing-code-validation-2026-09-19/` |
| Complete hardware report, index and file checksums | `demo/results/hardware-matrix-2026-09-19/` |
| Dated paid catalog and budget plans | `demo/hardware/` |
| Translation source, outputs, training/resume evidence | `demo/RUN_RESULTS.md`, `demo/results/translation-20260919-c77df0eb48ba/` |

The base is `Qwen/Qwen2.5-0.5B`, revision
`060db6499f32faf8b98477b0a26969ef7d8b9987`.
The combined base + v3b identity is
`02b0532346e314750330c6301670c2d6af40b874faf952f76f5fa326160ffbfb`.
The 300-case file SHA-256 is
`f1167e9094fa3ba5600d714cef22fc8045f932508ce61e5c56a9fc9cbe4ce1b9`.
The cases comprise 200 original held-out, 50 independent-review and 50 challenge
examples. They are now observed regression data, not a fresh blind test set.

## Pick it up without keys or GPUs

From the repository root, in a Python 3.11+ environment:

```bash
python -m pip install -e .
python scripts/verify_benchmark_handoff.py
python scripts/summarize_gpu_matrix.py \
  --index demo/results/hardware-matrix-2026-09-19/index.json \
  --out .cache/handoff-report
```

The verifier checks 148 evidence files, recomputes the saved report from raw
outputs, checks GPU/HTTP evidence, and hashes both included adapters. Only the
Python evaluator's LF/CRLF checkout differences are normalized; model, dataset
and evidence bytes must match exactly. It reads recorded cleanup/budget evidence;
it does not query the current account. The
second command creates a report in the ignored cache without rewriting the
frozen published report. Expected result: 11 measured models, two accepted
same-GPU optimizations, zero accepted optimized migrations from the 4090.

Development checks in an environment with the project/test dependencies:

```bash
python -m pip install -e '.[server,agent,dev]'
python -m pytest -q
ruff check scripts/benchmark_latency.py scripts/build_validation_suite.py \
  scripts/plan_gpu_matrix.py scripts/summarize_gpu_matrix.py \
  scripts/validate_existing_inference.py scripts/verify_benchmark_handoff.py \
  scripts/runpod_session.py src/gpushare/agent/latency.py \
  src/gpushare/agent/prediction_validation.py src/gpushare/agent/gpu_matrix.py \
  src/gpushare/agent/runpod_session.py tests/test_latency.py \
  tests/test_prediction_validation.py tests/test_gpu_matrix.py tests/test_runpod_session.py
```

The completed local suite passed **658 tests**, with one existing AnyIO
deprecation warning. Full numerical tests need the project's Torch/numpy/
Transformers/PEFT dependencies; a core-only install is enough to inspect and
recompute reports. On PowerShell, use `$env:PYTHONPATH = 'src'` if running directly
from source without installing. Windows pytest can use a fresh ignored
`--basetemp .cache/ji-pytest-1` directory.

## Reproduce on an already owned GPU

Use the pinned [NVIDIA image and package versions](hardware/README.md) or the
previously measured AMD image
`rocm/pytorch:rocm7.1.1_ubuntu24.04_py3.12_pytorch_release_2.10.0`.
Both used PyTorch 2.10, Transformers 5.17.0 and PEFT 0.21.0. Preserve the image's
CUDA/ROCm Torch build when installing packages; a default package sync can
replace it. The repository's generic ROCm setup is not an exact lock of this
measured image. Actual per-run environments are retained in the results.

Complete the [dedicated base-cache setup](checkpoints/README.md), leaving `BASE`,
`HF_HOME` and `HF_HUB_CACHE` set. On a compatible A5000, for example:

```bash
export PYTHONPATH=src
python scripts/validate_existing_inference.py \
  --model ckpt/laptop-lora-v3b --base "$BASE" \
  --expected-model-sha256 02b0532346e314750330c6301670c2d6af40b874faf952f76f5fa326160ffbfb \
  --data demo/validation/cases300.jsonl --n 300 --batch 1 \
  --engines compile-strict --out .cache/ji-a5000-validation
python scripts/benchmark_latency.py \
  --adapter ckpt/laptop-lora-v3b --base "$BASE" \
  --expected-model-sha256 02b0532346e314750330c6301670c2d6af40b874faf952f76f5fa326160ffbfb \
  --cases demo/results/inference-latency-2026-09-19/inputs.json \
  --expect-vendor nvidia --expect-gpu A5000 --pod-id YOUR_OWNED_POD_ID \
  --optimization compile-strict --out .cache/ji-a5000-latency.json
```

Use new output paths. For AMD, select the ROCm image and change the asserted
vendor/model to `amd` / `MI300X`. Both scripts use one actual GPU and reject CPU
fallback. Keep `scripts/train.py` available: the unchanged evaluator imports its
loss helpers. Candidate exit codes are **0 passed, 2 rejected, 1 failed**; a
completed timing run with exit 2 is not an accepted optimization. `--measure-only`
on the timing script explicitly disables its failure exit gate and should not
be used as evidence of acceptance.

The 13-case timing gate alone is insufficient. Recompute the 300-case comparison
for the actual source and destination before declaring a migration successful.
The retained HTTP checks exercise the original server; they do not prove that
the compiled engine is wired into production serving or the browser buttons.

## Rental, SSH and cleanup context

All campaign-owned pods were confirmed terminated. The cumulative estimate was
**$7.001 of the approved $15**, including **$3.086 for the expansion**. These are
duration times quoted compute prices plus storage allowances, not invoices.
There are no pods or live SSH tunnels for you to inherit. Windows helper
terminals from the campaign were closed.

The saved paid Secure/Community catalogs were queried without a CUDA filter.
They listed one AMD model, MI300X; its availability varied. Do not use an old
catalog as a current stock/price promise. The failed 5090 startup is included in
cost estimates. Obtain fresh quotes and a bounded plan before a new campaign.

`scripts/runpod_session.py` exposes `prepare`, `watchdog`, `create`, `status` and
`cleanup`; its module documentation describes the plan schema. Network actions
read **`RUNPOD_API_KEY`** from the caller's environment. The old local environment
used `GPUSHARE_RUNPOD_API_KEY`; map it privately if reusing that setup. Local
`.env`, `.gpushare` journals, private orchestration scripts and temporary SSH
keys are intentionally absent from the handoff. Generate your own SSH pair and
configure its public key in the new pod; use that pod's reported SSH address and
port. Do not reuse retired pod IDs or connection details from historical logs.

Start the watchdog before creation, start a worker as soon as each pod is ready,
and call owned-pod cleanup in a `finally` block. The local watchdog is best
effort, not a provider-side TTL; laptop sleep/network failure can delay deletion.
Verify termination through RunPod after collecting outputs. The public matrix
planner is offline and does not rent or run all GPUs automatically; campaign
orchestration remains integration work. On Windows, background helpers need
`CREATE_NO_WINDOW` (or `Start-Process -WindowStyle Hidden`) to avoid opening empty
terminal windows.

Baseten/OpenAI credentials are needed only for generating new AI translation
proposals. They are not needed for the fixed inference benchmarks, evidence
checks or public base download. No API/SSH keys are committed.

## Recommended next work

1. **Integrate the release model and latest schema.** Resolve the training conflict,
   keep your nullable/streaming updates, select the checkpoint deliberately, and
   establish baseline results on the actual app. Add fresh human-reviewed cases
   for absent facts, ambiguous dates and multiple people.
2. **Diagnose output drift before accepting migration.** Start with the concrete
   cases in `hardware/FINDINGS.md`. Compare the first divergent token/logits,
   cache implementation and backend behavior with fixed weights/precision.
   Keep an unoptimized fallback. Do not relax the agreed gate implicitly.
3. **Wire validation into the migration action.** Verify transferred checkpoint
   bytes and destination HTTP answers, then switch traffic only after the gate
   passes. Exercise rollback, cancellation, bad SSH, wrong model hashes and GPU
   OOM. A failed destination must leave the source usable.
4. **Measure the actual service.** Add cold/setup time, time to first token,
   end-to-end p50/p95, concurrency, throughput, memory and cost per accepted
   request. Current resident serial timings do not establish these properties.
5. **Broaden translation and skill learning.** Add reductions, custom backward,
   noncontiguous inputs and unsupported dependency fixtures. Feed measured
   outcomes into inactive skill candidates and use the trusted regression
   runner before promotion. General project conversion, target-specific tuning
   and automatic proven self-improvement remain unfinished.

The next product milestone is a real request surviving an optimization/GPU
switch with quality preserved, measured service latency and a working rollback.
The tests and evidence here provide the gates and known failures for that work.
