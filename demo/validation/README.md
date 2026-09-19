# Validation against the existing code

[Completed GPU results, individual outputs and HTTP checks](../results/existing-code-validation-2026-09-19/README.md).

The frozen speed-test model is `ckpt/laptop-lora-v3b`. It is an experimental
checkpoint, previously rejected as a replacement because of date-field
regressions; `scripts/extract.py` still defaults to v2. See the
[recorded model-selection decision](../../docs/optimization-v3.md). Running the
existing evaluator with v3b does not promote it or validate v2's speed. Keep the
checkpoint distinction explicit when presenting these results.

`cases300.jsonl` freezes 300 labelled sentences: all 200 original held-out cases,
all 50 independent-review cases, and 50 challenge cases balanced across ten
categories. `cases300.manifest.json` records source hashes, indices and category
counts. No sentence overlaps the saved `*train*.jsonl` files after case folding
and trimming outer whitespace. This checks sentence overlap, not semantic or template overlap. These sources
have been evaluated before; this is an expanded regression suite, not a newly
blind test set.

Build the cases with `python scripts/build_validation_suite.py`.

## What the validation does

`scripts/validate_existing_inference.py` runs `scripts/evaluate.py` unchanged in
a subprocess. It then runs static-cache and compiled candidates through that
evaluator's existing model loader, tokenizer, generation, scoring and loss
functions. It hashes the reference code before and after execution. No copied
replacement for the reference evaluator is used.

Every output is retained and matched by its original source index, because the
evaluator sorts failures first. Raw outputs are parsed again. Missing samples,
different inputs, different precision or decoding controls, invalid JSON, or any
changed field reject the comparison. Equal aggregate accuracy cannot conceal a
regression on one sentence and an improvement on another. Accuracy against the
labels is reported separately: preserving a wrong answer does not make it right.

The base-model and adapter files must match the frozen SHA-256 identity, and the
existing loader's offline Hugging Face cache must resolve to those same bytes.
No model downloads or GPU rentals are performed by this script.

```bash
export PYTHONPATH=src
export HF_HOME=/workspace/hf-cache
python scripts/validate_existing_inference.py \
  --model /workspace/adapter \
  --base /workspace/hf-cache/hub/models--Qwen--Qwen2.5-0.5B/snapshots/060db6499f32faf8b98477b0a26969ef7d8b9987 \
  --expected-model-sha256 02b0532346e314750330c6301670c2d6af40b874faf952f76f5fa326160ffbfb \
  --data demo/validation/cases300.jsonl --out /workspace/new-validation \
  --n 300 --batch 1 --engines static compile-strict
```

Use an existing CUDA or ROCm PyTorch environment. The measured deployment uses
PyTorch 2.10, Transformers 5.17 and PEFT 0.21. The original model loader uses the
cached `Qwen/Qwen2.5-0.5B` main reference: populate it with the exact revision above
before running. The script verifies the bytes rather than trusting the name.

Exit 0 means every candidate passed, exit 2 means a candidate was rejected, and
exit 1 means the validation could not complete. `validation.json` contains the
commands, code and model hashes, reference accuracy and per-case differences.
Raw outputs and execution logs are stored alongside it.

Runtime here includes compilation, all quality cases and held-out loss. It is
**not** a resident-response latency measurement; use the separate latency
benchmark for speed. A candidate passing this fixed set still needs unseen-case
testing and validation in the serving path before deployment.

## Diagnostic using the actual default model

The retained [two-request v2 smoke test](../results/existing-code-validation-2026-09-19/default-v2-smoke.json)
runs the unchanged extraction CLI with its default model and the existing
`--evidence --require-grounding` options. With all facts supplied, it returns the
correct Finn record. With no year supplied, the model proposes 2023, and the
grounding gate returns `needs_review` with exit 2. This confirms a useful existing
guard; it does not establish general abstention or semantic correctness. The
HTTP server does not currently apply that CLI guard.

See the [next product milestones](NEXT_STEPS.md) for the work that connects these
checks to real migration and serving behavior.
