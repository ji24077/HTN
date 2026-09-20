# Optimization and migration before training

The uploaded training pipeline can now run:

**plan → baseline → optimize if useful → migrate if needed → full training → final validation**

This first integration supports the GPUShare native AXPY training harness from
`origin/ji-phin-agentinfra` at commit `2e3deab`. Its Python training code, data,
precision, optimizer, validation oracle and native ABI are fixed; only its embedded
CUDA/HIP source literal can change. Ordinary training projects retain their existing
probe/execution path. A required adaptation outside the supported contract needs
clarification, not an implicit skip. Qwen batch/accumulation optimization, arbitrary
Python rewriting, checkpoint transfer/resume and joint multi-machine training are
not implemented by this integration.

## Planning and execution

`ProgramPlan.preparation` is optional and selected by the planner. It includes a
rationale, `optimize`, `source_worker_id`, `source_vram_mib`, `source_vendor`,
`target_vendor`, and `validation_steps` (1–32, default 8). The ordinary plan's
worker and requirements select the final target **before** any optimization.
An explicit `preparation: null` preserves the previous program path. No upload
API or database schema change is required; existing in-flight plans remain valid.

Both workers must report `python_program`, enough memory, a CUDA-compatible PyTorch
runtime, and the correct accelerator provider: `cuda` for NVIDIA, `rocm` for AMD.
ROCm still uses PyTorch's `cuda` device interface. Preparation checks exactly one
visible GPU and the vendor before execution; original compilation failures stop
the pipeline before model proposals. GPU workers need the appropriate native
compiler already installed. The paired native benchmark requires Linux.

The unchanged pilot harness is packaged at
`backend/src/orchestrator/preprocessing/portability/native_training.py` and can be
uploaded as `train.py` with a final checkpoint validator. The full command must
contain exactly `--steps`, `--expect-vendor`, `--out`, and `--checkpoint` with their
values. The latter two use `{output_dir}/...`. Requested full steps and the uploaded
validator's metric bounds are frozen. Preparation uses separate bounded reference
steps and never substitutes those for the requested full run.

1. Run the original on the source worker and freeze its numerical report.
2. If optimization is enabled, propose a native-only candidate, persist its hash,
   run static checks, then execute it on the source worker. Compare inputs, step
   range, predictions, loss history, precision and training settings. Benchmark
   the original and candidate in the same GPU process, alternating their order,
   using 3 warmups and 21 measured pairs on each of four fixed input shapes.
   Accept only when each shape improves by more than 5% in median paired time,
   with at least 75% faster rounds and correct outputs. These measurements cover
   **native preprocessing**, not whole-training speedup.
3. When vendors differ, seed migration with the existing deterministic CUDA/HIP
   translator, then let the agent propose and repair native code. Run the same
   numerical checks on the selected target against the original reference. Base
   PyTorch versions, candidate hash, vendor and step range must match. Moving to
   another same-vendor machine needs target validation but no translation proposal.
   Staying on the source machine skips migration.
4. Freeze the accepted artifact and release full training on the tested target.
   The original final validator still checks the trained checkpoint and metrics.

Every candidate and rejection is persisted. Static failures never reach a worker.
Compilation, numerical and benchmark failures feed the next model proposal, along
with the previous candidate and accumulated measurements. Each preparation stage
has at most the submission's `max_adaptations` attempts; all stages share the
original job deadline and usage cap. Failed optional optimization retains the
original artifact and continues to migration, including model failures and exhausted
worker infrastructure retries. Optimization is skipped when its next attempt would
consume the reserved time for migration and training. Shared usage caps, cancellation
and the job deadline still apply. Failed required migration or baseline
ends the job without starting full training.

Candidates are durable before dispatch; restarts resume the saved phase. Pause,
cancellation, task leases and budget enforcement use the existing pipeline.
Prepared jobs stay pinned to their tested source/target workers: worker loss waits
for that allocation until the job deadline, rather than silently running on
untested hardware. Probe checkpoints are temporary; this is code preparation,
not continuation of the reference run's optimizer state.

## Feedback and evidence

The job status includes `training_preparation`, `versions`, `checks`, `decisions`
and `measurements`. The UI names each stage and shows stage outcomes and attempts.
Artifact hashes link rejected and accepted code to evidence. Migration proposals
consume the accepted optimization artifact. Rejected optimization code is never
used as the migration input. This is feedback through model context, not automatic
fine-tuning of the agent model's weights.

Each passed, rejected, skipped and fallback outcome is saved in
`training_preparation.telemetry`, along with the final preparation/training outcome.
After the database commit, new outcomes are exported as structured Sentry Logs
through the existing `SENTRY_DSN` configuration. Search
`event_name:training.preparation.outcome` and filter by `job_id`, `stage`, `outcome`,
`source_vendor`, or `target_vendor`. Success logs are enabled even when the root
logger defaults to WARNING. Fields include stable `preparation_event_id`, artifact
hashes, attempt count, model/usage counters when available, numerical results,
paired timing evidence and rejection reasons. Credentials are scrubbed before
persistence and export; source code is linked by hash rather than copied into logs.

Sentry export is best effort and cannot determine whether a job passes. Outcomes
remain in the database if reporting is disabled or delivery fails. A crash after
commit can miss an export; replay/export tooling must use `preparation_event_id`
for deduplication. There is no automatic delivery outbox in this integration.

Use the [AMD → NVIDIA fixture](../examples/projects/amd_training/README.md) for the
reverse conversion test. Its validator reloads the trained checkpoint and evaluates
unseen inputs after full training. Both live source and target workers are required.

## Reused source and verification

Ported from GPUShare commit `2e3deab`:

- `src/gpushare/portability/{kernels,scripts}.py`: native mapping and source guards.
- `demo/compare.py`: fixed numerical report comparison.
- `demo/train.py`: supported immutable Python harness.
- `demo/benchmark.py`: paired same-GPU native benchmark.

The synchronous SSH engineering runner is adapted into persisted worker tasks in
`preprocessing/training.py`. It uses the existing configured model client and worker
transport; it neither imports the other branch's credential loader nor rents GPUs.

Run contract and database-backed lifecycle tests with:

```sh
RUN_SUPERVISOR_TESTS=1 uv run --project backend --python 3.12 --extra demo python -m unittest discover -s backend/tests -p 'test_training*.py' -v
```

Those tests use synthetic GPU reports and model proposals. They verify orchestration
and gates, not live CUDA/HIP compilation, numerical behavior or a hardware speedup.
A real source/target GPU acceptance run is still required to establish that evidence.

`examples/live_training_preparation.py` exercises the persisted planner and
preparation state machine against two SSH-accessible GPU hosts using the configured
model. It creates a private temporary PostgreSQL database and executes the trusted
program driver on each host. SSH replaces worker enrollment and WebSocket transport;
this acceptance runner does not cover enrollment or the output download API.
It neither provisions nor terminates hosts; the caller must enforce the cloud
budget and clean up resources.

Supply a host configuration containing `key` and `known_hosts` paths plus `amd`
and `nvidia` objects with `host`, `port`, and the absolute `python` executable path.
Both hosts need the same base PyTorch version and their native compiler installed.
Then run:

```sh
uv run --project backend --python 3.12 --extra demo --env-file .env \
  examples/live_training_preparation.py --hosts /path/to/hosts.json \
  --output .local/training-preparation-live/run
```

The output directory records hardware probes, job state, per-task reports and logs,
and any final checkpoint/validation files. `run.json` records the job ID: use that
ID to find the corresponding Sentry Logs. The runner returns a nonzero exit status
when the job does not complete successfully.
