# Sentence-to-JSON speed experiments

[Measured results, inputs and outputs](results/inference-latency-2026-09-19/README.md)

[Expanded hardware matrix: 10 NVIDIA models plus RunPod's AMD catalog](hardware/README.md)

This experiment measures the response time of the saved Qwen2.5-0.5B extraction
model after it is loaded. The same weights, tokenizer, prompt, input sentences,
greedy decoding, and precision policy are used on each GPU. The existing
evaluation policy uses BF16 base weights and FP32 PEFT adapter weights.

These speed measurements use the frozen experimental `laptop-lora-v3b` adapter.
The extraction CLI defaults to v2: v3b was [rejected as a model-quality
replacement](../docs/optimization-v3.md). Preserving v3b's answers during a speed
test is a separate decision from selecting the best extraction model.

The approved first comparison is one RTX 3090, one RTX 4090, and one AMD MI300X
on RunPod. NVIDIA uses PyTorch CUDA; AMD uses PyTorch ROCm. This measures migration
of the extraction model through those runtimes. Native CUDA/HIP source conversion
is covered separately by `train.py` and `train_hip.py`.

## Measurement

`scripts/benchmark_latency.py` keeps one model resident and makes serial,
batch-one requests. Each timed request includes prompt formatting, tokenization,
device transfer, generation, GPU synchronization, decoding and JSON parsing.
Network latency, model loading, compilation and warmup are excluded from that
number and reported separately where measured. Do not describe the handler timer
as browser-to-browser latency.

The baseline uses the current SDPA inference path and a dynamic attention cache.
The default candidate uses a static attention cache with `torch.compile` and
`emulate_precision_casts=True`, preserving intermediate precision casts that
compiler fusion can otherwise remove. This is a PyTorch Inductor setting tested
with PyTorch 2.10; it does not change model weights or the BF16/FP32 policy.
`--optimization compile` retains the original compiler behavior for comparison.
The optional
`--optimization static` mode measures the cache change without compilation.
Neither variant fuses adapter weights, quantizes the model, changes the prompt,
replaces the model, caches answers, or hard-codes extracted values.

Each exact prompt/engine is warmed twice. Five measured rounds alternate engine
order and rotate prompt order. Outputs and individual timings are retained,
including slower trials. A separate profiler sample records device activity
outside the latency measurements when the runtime supports profiling.

## Correctness and reporting

The first fixed set has 13 sentences: the first six original held-out examples,
the first six independent-review examples, and a fully specified Finn/UofT
example. Selection occurs before execution; examples are not filtered according
to the model's answers. This is a bounded experiment, not a general accuracy
certification.

Every repeated output must be valid JSON with the required typed fields. Every
field value must exactly match the corresponding baseline output; whitespace and
JSON key order may differ. Invalid output fails parity even when both engines
fail identically. Ground-truth accuracy is reported separately: preserving an
incorrect answer does not make it correct.

A speedup passes this experiment's gate only when the parity gate passes, median paired latency
improves by more than 5%, and at least 75% of matched trials are faster. Hardware
comparisons are labelled as migration results; comparisons within one device
are labelled as optimization results. Cross-host trials are not simultaneous,
and host CPU/network differences must be considered when interpreting results.
Five repetitions of 13 sentences are exploratory measurements, not a statistical
confidence interval or a guarantee for unseen inputs.

All local model files are hashed before loading. The exact base model revision
is prepositioned on each host; the learned adapter checkpoint is exported from
the 4090 and transferred to the other hosts. The full resulting model identity
must match the locally frozen identity. Transfer, setup, warmup and load durations
remain visible so migration overhead can be considered separately.

The rental profile has a 90-minute local watchdog and uses the existing $15 total
budget. After the initial setup attempts, it reserves $2.00 for earlier work and
at most $13.00 for the current session.
The watchdog is best effort and depends on the local machine and network; the
controller also terminates each owned pod after collecting its results and runs
final cleanup on failure.
Independent profiles for each approved GPU let a ready host run without waiting
for stock on another host. Each profile keeps the same GPU, cloud, region, count,
price limit, ownership checks and cleanup policy as the three-GPU profile.

## Runtime lessons from the real runs

The initial NVIDIA PyTorch 2.6 image ran the baseline but could not compile the
Transformers 5.17 Qwen wrapper. Its completed baseline measurements are retained.
The next NVIDIA trial uses official PyTorch 2.10.0 CUDA 12.8 wheels on hosts whose
NVIDIA 580 drivers support that runtime. AMD uses its ROCm 7.1.1 PyTorch 2.10 build.
These are deployment comparisons, including host CPU and vendor runtime effects.

Install Python dependencies in the image's existing GPU environment and constrain
its `torch` version. A nested environment can hide the parent ROCm installation
and cause pip to install a CUDA build. Verify the actual GPU and a synchronized
GPU calculation before proceeding. Hugging Face `.cache` download metadata is
excluded from model identity; weight, model configuration and tokenizer bytes
remain checked.

The original MI300X compiled candidate was 4.49x faster at the median but changed
the role field on one of 13 sentences. It is retained as a rejected candidate,
not an accepted speedup. That failure motivated the precision-cast setting and
another full benchmark against the same fixed cases.
The precision-cast rerun passed on both NVIDIA GPUs but retained the same AMD
regression. The setting name `compile-strict` describes a compiler option; it is
not a correctness guarantee. AMD's baseline remains the accepted path.

References: [official PyTorch version/install combinations](https://pytorch.org/get-started/previous-versions/),
[Transformers inference optimizations](https://github.com/huggingface/transformers/blob/main/docs/source/en/optimization_overview.md),
and [Inductor precision-cast implementation](https://github.com/pytorch/pytorch/blob/v2.10.0/torch/_inductor/lowering.py).

## Reproduce on an already-owned GPU

Provide a frozen local base-model directory, adapter directory, and cases JSON
(a list of objects with `id`, `sentence`, and `record`). Compute the expected model
hash with `gpushare.agent.latency.model_identity` before transferring the files.

```bash
PYTHONPATH=src python scripts/benchmark_latency.py \
  --base /workspace/base --adapter /workspace/adapter \
  --cases /workspace/cases.json --out /workspace/latency.json \
  --expected-model-sha256 "$MODEL_SHA256" \
  --expect-vendor nvidia --expect-gpu 3090 --pod-id "$OWNED_POD_ID" \
  --optimization compile-strict
```

Use `--expect-vendor amd --expect-gpu MI300X` on AMD. The command does not provision
GPUs or download weights. A failed compiled candidate retains baseline evidence
and is never automatically reported as a speedup.
Exit status zero requires `comparison.speedup_verified: true`. A completed but
rejected candidate exits 2; execution failures exit 1. `--measure-only` explicitly
allows exit zero for completed measurements without accepting the candidate.
Consumers must still check `comparison.speedup_verified`; `status: measured`
alone does not mean the quality or speed gate passed. Historical reports were
collected before this stricter exit-status behavior was added.

The [expanded validation suite](validation/README.md) checks 300 cases through the
existing evaluator and compares every output, rather than only aggregate scores.
