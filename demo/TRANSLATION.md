# Review translation inputs and outputs

These commands prepare source files locally. They use deterministic runtime API
and launch mappings, without calling a model, importing the input, compiling,
renting a GPU, or executing training. Run them from the repository root after
installing the package into your Python environment:

```powershell
.venv/Scripts/python.exe scripts/prepare_translation.py demo/train.py --target amd --out demo/prepared-cuda-to-hip
.venv/Scripts/python.exe scripts/prepare_translation.py demo/train_hip.py --target nvidia --out demo/prepared-hip-to-cuda
```

Each output folder must be new. It contains:

- `original.py`: the submitted script.
- `translated.py`: the candidate, with only its native source literal replaced.
- `result.json`: source hashes, applied mappings, static findings, and status.

`prepared_unverified` and `gpu_verified: false` mean a static draft exists.
They do not mean the GPU compiler accepted it, the GPU executed it, numerical
checks passed, or performance improved. Unsupported APIs, ambiguous source, or
changes to the protected Python contract produce `blocked` and a nonzero exit
code. Partial drafts remain available for inspection. Ordinary PyTorch-only
scripts have no native mapping to demonstrate; this initial CLI requires one
top-level `CUDA_SOURCE` or `HIP_SOURCE` string literal.

## Two independent native inputs

`train.py` starts with CUDA runtime calls and a CUDA kernel launch. Its native
implementation allocates three device buffers and uses a one-element-per-thread
AXPY kernel.

`train_hip.py` is independently authored HIP input, not output from converting
the CUDA fixture. Its native implementation allocates one contiguous device
buffer, splits it into three arrays, launches a capped grid with
`hipLaunchKernelGGL`, and uses a grid-stride AXPY kernel with explicit cleanup.
Both fixtures deliberately share the Python inputs, float32 CPU oracle, training
loop, checkpoint checks, and exported C ABI. This isolates native translation
from changes to the test's meaning. The independent implementation makes no
speedup claim.

Both demos are standalone PyTorch scripts. `--inspect` previews their inputs on
CPU. Normal execution requires a Linux GPU development environment with the
correct PyTorch build and `nvcc` or `hipcc`. PyTorch's `torch.cuda` API remains
unchanged when running its ROCm build.

## Validating a new translation

Both demo directions and checkpoint continuation passed on real RunPod GPUs;
see [recorded results and measured limits](RUN_RESULTS.md). New inputs and edits
still require the checks below.

Run each original on its own vendor first, then run its translated candidate on
the other vendor. Keep the numerical tolerances, generated inputs, precision,
step count, and independent reference unchanged. Collect native compilation
logs, GPU identity/use evidence, custom-kernel checks, training progress, and
checkpoint/prediction comparison reports. Measure optimization on the same
device with repeated timings before claiming a speedup. Static checks alone
cannot prove arbitrary native code preserves behavior or actually performs its
work on a GPU.
