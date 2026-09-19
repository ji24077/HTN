# CUDA → AMD translation demo

Both translation directions and NVIDIA-to-AMD checkpoint continuation have now
passed on RunPod's RTX 4090 and MI300X. See [actual results, inputs, outputs and
costs](RUN_RESULTS.md). The native benchmark passed its speedup gate only for
the largest tested input; no whole-training speedup is claimed.

Paste **all of `train.py`** into the engineering agent. It is a standalone
PyTorch training script with its CUDA C++ kernel embedded as `CUDA_SOURCE`.
It needs no API key, external dataset, pretrained model, or imports from this
repository. The only Python dependency is PyTorch.

The custom kernel computes `features = 0.375 * x + y`. Those actual GPU results
become input to a small neural network trained with Adam in float32. This gives
the agent real CUDA APIs and a kernel launch to convert, alongside real training.
The original script is a **CUDA input**, not an already-translated AMD program.
Switching the PyTorch wheel or compiler alone does not convert its CUDA source.

## Inspect the input locally

From the repository root, with PyTorch installed:

```bash
python demo/train.py --inspect
```

This prints sample inputs, expected preprocessing output, and training shapes.
It does not compile anything or run a GPU. The normal training path has no CPU
fallback: a missing GPU, zero free GPU memory, or missing compiler stops the run.

## Run the original on NVIDIA

Use a Linux NVIDIA GPU environment containing the CUDA development toolkit
(`nvcc` on PATH) and a compatible host driver. A PyTorch wheel alone does not
provide the native compiler. For the pinned demo environment:

```bash
python -m pip install -r demo/requirements.txt --index-url https://download.pytorch.org/whl/cu128
python demo/train.py --expect-vendor nvidia --steps 8 \
  --out demo/artifacts/nvidia-baseline.json \
  --checkpoint demo/artifacts/nvidia-checkpoint.pt
```

The script writes and compiles its embedded kernel in `demo/artifacts/`, checks
it against an independent CPU reference at lengths including zero, one, and
nonmultiples of the block size, then trains on the GPU. Each completed optimizer
step prints its loss. Reports contain predictions, input identity, actual device,
kernel checks, optimizer state, and PyTorch memory measurements.

## Task for the engineering agent

1. Preserve the original script and its baseline report.
2. Make portable optimizations in a separate copy; preserve float32, generated
   inputs, model architecture, effective batch, learning rate, and step count.
3. Translate the custom CUDA source to HIP in a separate self-contained script,
   for example `demo/translated/train.py`. Preserve the CPU reference and checks.
   PyTorch's `torch.cuda` interfaces also operate on ROCm and should not simply
   be renamed to `torch.hip`.
4. On a Linux AMD development environment with ROCm PyTorch and `hipcc`, run
   that translated script with `--expect-vendor amd` and the same eight steps.
5. Compare the resulting report to the frozen baseline:

```bash
python demo/compare.py demo/artifacts/nvidia-baseline.json /path/to/amd-report.json
```

The comparator checks identity, predictions, and matching training-segment
losses within float32 tolerances. It requires the same kernel test shapes and
rejects custom-kernel errors above 1e-5 for this fixed workload. It does not infer translation correctness from
successful compilation alone. It reports timing only as a diagnostic and never
attributes a difference between GPU models to optimization.

## Checkpoint handoff

To test continuation, first run the original for four steps and keep its
checkpoint. Transfer that checkpoint with the translated script to AMD:

```bash
python demo/train.py --expect-vendor nvidia --steps 4 \
  --checkpoint demo/artifacts/step4.pt --out demo/artifacts/step4.json

# On AMD, after converting the source and copying the checkpoint:
python /path/to/translated/train.py --expect-vendor amd --resume /path/to/step4.pt \
  --steps 8 --checkpoint /path/to/step8.pt --out /path/to/amd-resumed.json
```

Resume restores weights, Adam moments and step counters, global step, and CPU
RNG. It checks saved prediction parity before taking another optimizer step.
Fixed full-batch data and a model without dropout avoid a cross-vendor GPU RNG
dependency. `--steps` is the final global step; using the saved step performs
evaluation without additional training. Read only checkpoints produced by this
demo; loading uses `weights_only=True`.

## What the measurements mean

Eight steps are a correctness smoke test, not a trustworthy performance benchmark.
Timing excludes compilation and preprocessing and includes training checks and
step logging. Memory figures cover the PyTorch allocator during training and
exclude native `cudaMalloc`/HIP allocations used by preprocessing. The naive
host/device copies deliberately provide something for the optimization agent
to improve, while preserving the calculation and independent checks.

The engineering-agent CLI implements bounded optimization and translation
proposals, with an optional remote GPU validator. Local checks and prepared
proposals do not establish GPU correctness or performance; those claims require
the execution and comparison reports from an actual run.

## Benchmark the native preprocessing stage

On the isolated Linux GPU host used for correctness validation, compare the
original and optimized scripts using the same backend and exactly one visible
GPU. Both native sources must first pass the translation guard and correctness
checks.

```bash
python demo/benchmark.py --baseline /path/to/original.py \
  --candidate /path/to/optimized.py --out demo/artifacts/native-benchmark.json
```

The output JSON path must be new. The harness extracts and compiles the embedded
native source without importing candidate Python. It measures only the float32
AXPY preprocessing call, including device allocations, transfers, kernel launch,
cleanup, and synchronization. It excludes compilation and training, so this
benchmark cannot establish a whole-training speedup.

After three warmup rounds, it alternates baseline/candidate order for at least
21 measured pairs at each fixed size: 1, 257, 65,537, and 1,048,576 elements. Every
output must pass the fixed CPU oracle. `performance_verified` is true only if
**every size exceeds 5% median paired improvement and the candidate is faster
in at least 75% of pairs**. A completed measurement can still fail that gate.
The report verifies the GPU environment and compiled native invocation; it does
not provide an independent profiler trace of GPU kernel execution.

References: [PyTorch ROCm interfaces](https://docs.pytorch.org/docs/main/notes/hip.html)
and [CUDA-to-HIP porting](https://rocm.docs.amd.com/projects/HIP/en/latest/how-to/hip_porting_guide.html).

## Run the engineering-agent CLI

From the repository root, install the agent dependencies and configure the
selected provider key in the ignored `.env` file. This command sends the embedded
native CUDA/HIP source and active skill instructions to the selected coding-model
provider. It writes proposals locally; without `--connections` it does not execute
on a GPU or rent a Pod.

```bash
python scripts/translation_agent.py demo/train.py \
  --out .gpushare/translation/demo-first --provider baseten \
  --target amd --model-budget 1 --attempts 3
```

Use `--provider openai` to select the configured OpenAI key. The output directory
must be new. `result.json` records the execution status, source and skill hashes,
model usage, proposals, and any candidate lesson IDs. `prepared_unverified` means
the generated code has not passed GPU execution checks. CLI progress is JSON,
followed by the complete final result JSON. `execution-result.json` preserves the
execution summary before lesson registration; candidates reference its exact file
hash so their supporting result can be checked later.

For existing GPU hosts that you own, `--connections path/to/connections.json`
accepts `source_info` and `target_info` objects containing `ip`, `port`, and a local
SSH private-key file path in `key`, plus `work_dir` set to
`/workspace/gpushare-translation/<session>`. Source and target vendors must match
the selected direction. The validator compiles and executes on those hosts; the
caller remains responsible for their rental budget and teardown. Keep connection
files under an ignored directory and never put private-key contents in them.

The CLI reads the active project skill from `.gpushare/skill-memory/demo` by
default. `--project-id NAME` selects another project; `--skills-dir PATH` sets an
explicit store location bound to that project. Successful and unverified lesson
proposals remain inactive candidates. A single demo never promotes a rule, even
when its GPU checks pass: promotion requires the separate, unchanged regression
suite and its real GPU evidence. Shared lessons and explicit rollback are
available through `gpushare.portability.memory.SkillStore`.

## Current implementation scope

The agent accepts demo-compatible scripts with an embedded `CUDA_SOURCE` or
`HIP_SOURCE` string implementing the existing stateless C ABI. Edits are confined
to that native source; the Python training workload and correctness checks remain
fixed. General project dependency translation and arbitrary custom GPU extensions
are not implemented. Target-specific tuning is also not implemented yet.

Lesson storage, versioning, rollback, and regression-gated promotion are
implemented. The CLI stores project candidates; automatic shared-rule promotion
remains unavailable until a trusted fixed GPU regression suite is connected.
