# Existing-code validation: 19 September 2026

The unchanged evaluator completed 300 cases on the laptop RTX 4060, a RunPod
RTX 4090, and a RunPod AMD MI300X. Both cloud GPUs also ran the cache-only and
compiled candidates through the evaluator's existing functions. Every response
parsed, but **neither candidate preserved every reference value**. Both GPU
validators correctly exited 2. These results do not approve a new optimization.

The unchanged HTTP server also answered 13 real requests on each cloud GPU,
through SSH forwards from the laptop. All 26 responses were correct on that
small smoke set. All rented pods were terminated; the final RunPod inventory was
empty. [Machine-readable summary](summary.json) and [file hashes](manifest.json).

## Model and test scope

The model is the same frozen Qwen2.5-0.5B plus `laptop-lora-v3b` used in the
earlier speed experiment, with BF16 base and FP32 PEFT adapter weights for the
evaluation runs. Its full file identity is
`02b0532346e314750330c6301670c2d6af40b874faf952f76f5fa326160ffbfb`.

V3b is an experimental checkpoint that was previously rejected as a quality
replacement; the extraction CLI still defaults to v2. This work validates the
existing code with the benchmark checkpoint. It does not promote v3b or establish
the default v2 model's speed. The [300 cases and selection manifest](../../validation/README.md)
include all 200 original cases, all 50 independent-review cases, and 50 balanced
challenge cases. These previously observed sources are not a newly blind test.

## Accuracy and output parity

Each count below is recomputed from raw JSON and the fixed labels, with exact
field-value equality. A regression means a previously strictly correct case
became incorrect. Changes include improvements and capitalization differences;
the agreed preservation rule still rejects any changed value.

| GPU | Mode | Strictly correct / 300 | Values changed versus same-GPU reference | Previously correct cases regressed |
|---|---|---:|---:|---:|
| Local RTX 4060 Laptop | Original evaluator | 271 | — | — |
| RTX 4090 | Original evaluator | 268 | — | — |
| RTX 4090 | Static cache | 270 | 2 | 0 |
| RTX 4090 | Static cache + compile-strict | 270 | 2 | 0 |
| AMD MI300X | Original evaluator | 270 | — | — |
| AMD MI300X | Static cache | 269 | 14 | 6 |
| AMD MI300X | Static cache + compile-strict | 270 | 13 | 5 |

The original evaluator lowercases and trims values for its own score. It reports
269/300 for the 4090 reference, whereas exact value equality gives 268/300: it
accepts `Hospice chaplain` against the label `hospice chaplain`. That original
report is retained unchanged. The other 4090 candidate change corrects
`zoovet veterinarian` to `zoo veterinarian`. Neither change worsens a previously
correct answer, but both violate literal output preservation.

AMD's compiled candidate has the same total strict accuracy as its reference,
yet loses five previously correct cases. That is the concrete reason to check
individual answers instead of accepting equal aggregate scores. Cache-only mode
already introduces drift, so a compiler flag alone does not address every failure.
These observations identify a failing configuration, not the underlying numerical
cause or a general defect in AMD hardware.

## Comparisons between configurations

| Source → destination | Changed cases / 300 | Exact-value parity |
|---|---:|---|
| Original 4090 → original AMD | 2 | Reject |
| Original 4090 → static-cache AMD | 16 | Reject |
| Original 4090 → compiled AMD | 15 | Reject |
| Static-cache 4090 → compiled 4090 | 0 | Pass |
| Compiled 4090 → original AMD | 0 | Pass |

The two matching pairs are useful bounded observations. They do not reverse the
rejected optimization relative to the original 4090 or establish a speedup for
migration to the original AMD path. No universal answer-equivalence claim is made.

The local 4060 reference differs from the original 4090 on three cases. It runs
PyTorch 2.11.0+cu128; the cloud 4090 runs 2.10.0+cu128, and AMD runs
2.10.0+rocm7.1.1.gitd9556b05. These comparisons include runtime and host differences,
not just chip differences.

## Existing HTTP serving path

[4090 requests and responses](rtx4090/http.json) and
[AMD requests and responses](mi300x/http.json) retain every payload and timing.
Each run sends the earlier 13-case set to the unmodified `scripts/serve.py`
`/generate` endpoint, with the model already resident. All 13 responses on each
GPU match their labels; the 12 cases present in the 300-case set also match the
same-GPU reference evaluator. Finn is the additional smoke case.

This checks real baseline serving over an SSH connection. It does not test the
browser migration button, automatic traffic switching, concurrency, streaming,
or a compiled server. Client round-trip times include the network; the server's
generation timer has a different scope. Neither is a new controlled speed benchmark.

The separate [default-v2 smoke test](default-v2-smoke.json) runs the unchanged
extraction CLI with its default model. A complete Finn sentence produces the
correct record. A sentence with no year makes the model propose 2023; the existing
`--require-grounding` option returns `needs_review` and exit 2. The HTTP server
does not apply that CLI guard. Literal grounding also cannot detect a wrong date
that happens to occur elsewhere in the input.

## Evidence, implementation checks and cost

Each GPU directory contains the original evaluator output, candidate outputs,
per-case comparison report, HTTP responses, execution logs, GPU environment
proof and hashes of the uploaded source. The evaluator and serving scripts were
not edited. The candidates reuse the existing evaluator's loaders, tokenizer,
generation, scoring and loss functions.

These runs apply the cache/compile configuration through the existing evaluator.
The earlier latency harness has its own loading and timing path; the two should
not be presented as one benchmark. Evaluation runtime here includes compilation,
quality cases and held-out loss. No new resident-response speedup is claimed.

The new benchmark exit policy rejects an unverified speedup by default; an
explicit `--measure-only` option is required to treat completed measurements as
success without passing the acceptance gate. The local automated test suite
passed: **630 tests**.

This validation campaign cost an estimated **$1.18**, including failed setup
attempts; estimated cumulative spend including the earlier campaigns is **$3.92**
against the $15 cap. These are duration/rate estimates with a storage allowance,
not an invoice. The first attempts exposed a container package-management
restriction and an omitted `scripts/train.py` dependency used by the evaluator's
loss calculation. The unchanged public helper was supplied and the successful
4090 retry included a one-case preflight. Failed attempts are included in the
session accounting, not counted as completed validations.

See the [next product milestones](../../validation/NEXT_STEPS.md) for the work
needed to integrate these checks into migration and serving.
