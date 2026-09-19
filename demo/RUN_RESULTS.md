# GPU translation demo results

Session: `translation-20260919-c77df0eb48ba`.

**Bidirectional translation and checkpoint resume are verified.** The broad performance gate did not pass. Scope: float32 native AXPY preprocessing feeding the fixed PyTorch MLP with Adam.

| Check | Result | Evidence |
| --- | --- | --- |
| NVIDIA CUDA → AMD HIP | verified | [report](results/translation-20260919-c77df0eb48ba/forward/result.json) |
| Independent AMD HIP → NVIDIA CUDA | verified | [report](results/translation-20260919-c77df0eb48ba/reverse/result.json) |
| Checkpoint and optimizer resume | verified | [report](results/translation-20260919-c77df0eb48ba/checkpoint-resume/validation.json) |
| Same-GPU native AXPY performance gate | measured; gate not passed | [report](results/translation-20260919-c77df0eb48ba/benchmark/report.json) |

Both owned Pods were terminated and cleanup was verified.

Final local verification: **585 tests passed** and Ruff passed. The test suite
reported one existing Starlette/AnyIO deprecation warning.

Two lessons from the verified repair attempts were stored as inactive project
candidates. Shared skill instructions have not been promoted automatically.

## Actual training evidence

| Stage | GPU | PyTorch | Completed steps | PyTorch peak allocated MiB |
| --- | --- | --- | ---: | ---: |
| forward baseline | NVIDIA GeForce RTX 4090 | 2.10.0+cu128 | 8 | 17.867 |
| forward optimize | NVIDIA GeForce RTX 4090 | 2.10.0+cu128 | 8 | 17.867 |
| forward translate | AMD Instinct MI300X | 2.10.0+rocm7.1.1.gitd9556b05 | 8 | 152.617 |
| reverse baseline | AMD Instinct MI300X | 2.10.0+rocm7.1.1.gitd9556b05 | 8 | 152.617 |
| reverse optimize | AMD Instinct MI300X | 2.10.0+rocm7.1.1.gitd9556b05 | 8 | 152.617 |
| reverse translate | NVIDIA GeForce RTX 4090 | 2.10.0+cu128 | 8 | 17.867 |
| resume source | NVIDIA GeForce RTX 4090 | 2.10.0+cu128 | 4 | 17.867 |
| resume target | AMD Instinct MI300X | 2.10.0+rocm7.1.1.gitd9556b05 | 4 | 152.617 |

The same checkpoint resumed from step 8 to 12 on both backends; optimizer restoration, predictions and continued loss history passed the fixed comparison.

## Native preprocessing measurements

Same GPU and work; compilation and training are excluded. All four shapes must exceed 5% median paired improvement and win at least 75% of at least 21 pairs, with every output passing the fixed oracle.

| Elements | Baseline median µs | Candidate median µs | Paired median improvement | Faster pairs | Gate |
| ---: | ---: | ---: | ---: | ---: | --- |
| 1 | 133.539 | 131.035 | 1.62% | 71.4% | fail |
| 257 | 131.556 | 129.932 | 0.91% | 71.4% | fail |
| 65,537 | 229.569 | 226.142 | 1.38% | 85.7% | fail |
| 1,048,576 | 1898.183 | 1763.753 | 10.46% | 100.0% | pass |

## Cost estimates

| Item | USD |
| --- | ---: |
| Earlier migration attempts | $0.694000 |
| Translation setup attempt 1 (failed) | $0.063100 |
| Translation setup attempt 2 (failed) | $0.042200 |
| This session GPU/storage upper estimate | $0.340200 |
| Prepared coding-model proposals | $0.004267 |
| Live repair model usage (cumulative meter counted once) | $0.001984 |
| **Total estimate** | **$1.145751** |

Budget: $15.00; within the approved total. These are duration/storage and model-token estimates, not provider invoices.

## Inputs, outputs and limits

[Artifact summary](results/translation-20260919-c77df0eb48ba/summary.json) · [Integrity manifest](results/translation-20260919-c77df0eb48ba/export-manifest.json).

- [Forward input](results/translation-20260919-c77df0eb48ba/forward/input.py)
- [Forward output](results/translation-20260919-c77df0eb48ba/forward/output.py)
- [Reverse input](results/translation-20260919-c77df0eb48ba/reverse/input.py)
- [Reverse output](results/translation-20260919-c77df0eb48ba/reverse/output.py)

- Embedded stateless CUDA_SOURCE/HIP_SOURCE C ABI only; arbitrary project/dependency translation is not implemented.
- Benchmark claims apply to native AXPY preprocessing only, not whole training or cross-GPU speedup.
- GPU environment/native invocation evidence is not an independent device-profiler trace.
- Target-specific tuning and a trusted fixed GPU suite for automatic shared-rule promotion are not implemented.
- Binary checkpoints and SSH/compilation stdout/stderr are excluded from this public artifact export.
