# AMD → NVIDIA training preparation fixture

Upload `train.py` and `validate.py` together with workload **training** and this
description:

> Run the original HIP training harness on an AMD source worker. Optimize its
> embedded native kernel if repeated paired measurements prove an improvement.
> Then migrate the accepted code to a separate NVIDIA worker and validate its
> numerical results against the original AMD reference. Train for exactly 16
> steps on NVIDIA, save report.json and checkpoint.pt, and run validate.py. Require
> loss between 0 and 1.0 from metrics.json. Use 8 reference validation steps.
> Preserve the training code, dataset, optimizer and validator.

Allow at least 20 minutes for planning, compilation, bounded retries and execution.
The platform must have an online AMD worker advertising `rocm` and a separate
NVIDIA worker advertising `cuda`, both supporting `python_program`. Install matching
PyTorch base versions and the vendor's development compiler (`hipcc` / `nvcc`).
Each execution must see exactly one GPU.

`train.py` is the supported GPUShare harness with the embedded native source
translated to HIP. Its protected Python/docstring still describes the original
CUDA-to-HIP demo; the actual input source includes `hip/hip_runtime.h`. Migration
must replace this with CUDA before NVIDIA execution. It is a small AXPY
preprocessing kernel feeding a PyTorch MLP and Adam, with generated data and no
model downloads. Optimization measures the native preprocessing calls only.

The full-run command is:

```text
train.py --steps 16 --expect-vendor nvidia --out {output_dir}/report.json --checkpoint {output_dir}/checkpoint.pt
```

The final validator reloads the actual checkpoint on CPU, verifies Adam counters,
replays saved predictions, and evaluates unseen inputs. The loss threshold is a
fixed demo acceptance criterion, not a quality guarantee for production models.
Declare `checkpoint.pt`, `report.json`, and `metrics.json` as outputs.

Local integration tests use synthetic worker evidence. They prove ordering,
conversion, fallback, persistence and telemetry wiring; live AMD/NVIDIA execution
is required to prove compiler compatibility and GPU numerical results. No joint
training across machines or checkpoint continuation is performed.
