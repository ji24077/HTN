# Measured sentence-to-JSON latency: 19 September 2026

These are exploratory **13-case** measurements. The follow-up
[300-case validation against the unchanged evaluator](../existing-code-validation-2026-09-19/README.md)
is the broader correctness check. The pass labels below apply only to this
original fixed set and are not release approval. Raw measurements are preserved.

One GPU per RunPod pod; saved Qwen2.5-0.5B + laptop-lora-v3b; BF16 base and FP32 adapter; batch one; greedy decoding.
The fixed 13-sentence set includes the Finn/UofT example. Each engine has five measured repetitions per sentence (65 requests).
Times include the resident handler and complete JSON generation. Network, model load and warmup/compilation are separate.

| GPU | Baseline median | Candidate median | Candidate p95 | Measured latency ratio | Acceptance gate |
|---|---:|---:|---:|---:|---|
| RTX 3090 | 1.492 s | 0.214 s | 0.247 s | 6.96x | Pass |
| RTX 4090 | 0.639 s | 0.142 s | 0.162 s | 4.51x | Pass |
| AMD MI300X | 1.136 s | 0.248 s | 0.270 s | 4.58x | REJECTED |

The fastest accepted configuration within this 13-case experiment is the compiled RTX 4090. Both NVIDIA
candidates preserve every JSON value and score 13/13 exact cases in every round.
AMD's baseline also scores 13/13. Both AMD compiled variants change the same role
field, scoring 12/13; the precision-cast setting did not fix that regression.
Only the AMD baseline passed this small set's quality requirement. The rejected candidate
timings remain visible for diagnosis and are not accepted speedups.

The optimization uses static KV cache, torch.compile/reduce-overhead, and intermediate precision-cast emulation.
The precision policy, weights, prompt and outputs are checked independently of performance. Per-request raw outputs and timings are retained in the GPU JSON files.

## Migration from the 3090 baseline

| Destination | Measured latency ratio | Same JSON | Acceptance gate |
|---|---:|---|---|
| rtx4090-baseline | 2.33x | True | True |
| rtx4090-optimized | 10.53x | True | True |
| mi300x-baseline | 1.31x | True | True |
| mi300x-optimized | 6.02x | False | False |

## Regular 4090 compared with optimized 3090 and AMD

This reuses the measured runs above, with the regular RTX 4090 as the reference.
It does not represent another rental or a new timed run.

| Configuration | Median response time | Relative to regular 4090 | JSON quality |
|---|---:|---:|---|
| Regular RTX 4090 | 0.639 s | Reference | 13/13 correct |
| Optimized RTX 3090 | 0.214 s | 2.98x faster | All values preserved; accepted |
| Optimized AMD MI300X | 0.248 s | 2.58x shorter measured latency | Changed one role field; rejected |

The optimized 3090 beats the regular 4090 by about 3x while keeping the same
answers on the fixed test set. AMD's compiled result remains ineligible because
it changes the answer on one sentence in every repetition.

## Finn example

`In 2026, Finn, 18, became a student at the University of Toronto.`

Expected JSON: `{"name":"Finn","age":18,"org":"University of Toronto","role":"student","year":2026}`

| GPU | Baseline median | Optimized median |
|---|---:|---:|
| RTX 3090 | 1.232 s | 0.187 s |
| RTX 4090 | 0.526 s | 0.120 s |
| AMD MI300X | 0.940 s | 0.213 s |

The Finn example alone passes on AMD, but the full fixed test set rejects its
compiled path. A successful demo sentence does not override the failed gate.

## Setup, evidence and limits

| GPU | Model load | Baseline warmup | Optimized warmup incl. compile | Profiled GPU events |
|---|---:|---:|---:|---:|
| RTX 3090 | 0.82 s | 32.63 s | 94.02 s | 80799 |
| RTX 4090 | 0.48 s | 13.78 s | 52.21 s | 81039 |
| AMD MI300X | 1.18 s | 35.98 s | 74.02 s | 73262 |

Warmup includes all 26 warmup requests per engine. Provisioning, downloads and runtime installation add overhead; measured transport/setup phases are retained in summary.json. This is not a measurement of a production migration button or browser/network latency.

NVIDIA uses PyTorch 2.10.0/CUDA 12.8; AMD uses its PyTorch 2.10.0/ROCm 7.1.1 build. CPU hardware differs. These results compare deployments, not isolated chip performance.

Initial PyTorch 2.6 NVIDIA trials ran valid baselines but compilation failed. The original MI300X compiled trial reached 4.49x but changed one role field (12/13 exact cases); its result is retained and rejected. No case was removed or edited to make an optimization pass.

All GPUs were verified by device identity, model parameter placement, synchronized GPU computation and profiler device activity. Compiler counters record actual compiled graphs.

A 13-case exploratory test does not establish accuracy or latency for unseen inputs, other models, training, throughput, other batch sizes or different precisions. The speed gate (>5% median paired improvement and >=75% faster trials) is not a confidence interval.

All owned pods were terminated. Estimated inference campaign cost: $1.59; estimated cumulative cost including earlier migration work: $2.74 of the $15 cap. These are duration/rate estimates with a storage allowance, not invoice totals.

Inputs: [inputs.json](inputs.json). Outputs and timings: [rtx3090.json](rtx3090.json), [rtx4090.json](rtx4090.json), [mi300x.json](mi300x.json). All comparisons: [summary.json](summary.json). Evidence hashes: [manifest.json](manifest.json).
