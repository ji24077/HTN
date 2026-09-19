# Local optimization pass: v3b

This experiment continues the saved `ckpt/laptop-lora-v2` Qwen2.5-0.5B adapter on targeted examples. **The candidate failed the appointment-year regression gate and was not promoted. The default remains v2.** Experiments ran on the local RTX 4060 Laptop GPU. No RunPod, Baseten, or OpenAI key was required. Source, tests, generated data, and measured reports are retained. The later [Ji handoff](../demo/HANDOFF.md) also publishes both exact adapters with base-download instructions.

## Measured quality and selection

| Evaluation | Examples | Existing v2 exact match | Experimental v3b exact match |
|---|---:|---:|---:|
| Original held-out | 200 | 87.5% | 89.5% |
| Existing challenge | 800 | 81.0% | 81.875% |
| Independently authored diagnostic | 50 | 96.0% | 98.0% |

All three produced valid schema-conforming JSON on every example. Aggregate exact match hides an important tradeoff: challenge name accuracy rose from 90.0% to 96.125%, and age from 98.25% to 100%, but year accuracy fell from 91.75% to 85.5%. The strict per-field gate rejected this 6.25-point regression. Parenthetical-date exact match fell from 78.75% to 43.75%; irrelevant-number exact match fell from 18.75% to 8.75%. These remain development failures, even though more total records were correct.

The pipeline and optional training configurations are improved; this experiment does **not** establish a better replacement model. No further candidate was trained after seeing these results. The next training-data investigation should focus on appointment-year binding while preserving replay coverage, using a separate development split and a new untouched test set before claiming generalization.

## Input and output

- Training input: `data/experiments/binding-v3b/train_replay.jsonl` — 5,168 rows: all 1,768 original training records, 2,400 targeted records, and 1,000 deterministically selected v2 training records for replay.
- Experimental training output: `ckpt/laptop-lora-v3b/adapter_model.safetensors` and `meta.json`. This rejected candidate is retained for inspection. This is an adapter requiring the original Qwen base model, not a full standalone checkpoint. The later handoff includes the exact v2/v3b weight files; see [checkpoint setup](../demo/checkpoints/README.md).
- Evaluation outputs: `eval/v3b-{baseline,candidate}-{original,challenge,independent}.json`. Each includes every input, expected record, model output, wrong fields, category, and source index. Samples are ordered with failures first.
- Machine-readable results and checkpoint hashes: `experiments/optimization-v3-results.json`.
- Live CLI example using the retained v2 checkpoint: [input](../experiments/demo-input.txt) and [output with source spans](../experiments/demo-output.json). Its appointment year was correctly extracted as 2019 despite founding and publication dates in the same input.

The targeted examples pair a changed fact with its changed answer, or change a distracting fact while keeping the answer constant. They cover appointment dates, subject selection, names with accents and punctuation, and 561 distinct job titles. Number ranges overlap across the corpus to reduce easy date/age shortcuts. Founding and graduation still precede hiring in these templates, so they do not cover every temporal relationship. The initial `binding-v3` directory is an unused draft; this run uses `binding-v3b` exclusively.

Replay assembly is reproducible from the frozen training files: load `binding-v3b/train_mixed.jsonl`, append `random.Random(431).sample(v2_train_rows, 1000)`, then shuffle the combined list with `random.Random(1337)`. Source and output hashes are in `binding-v3b/replay_manifest.json`. Counterfactual partners share a `group_id`; keep them together if creating a future split.

## Training efficiency

A matched short profile used the same starting adapter, examples, shuffled order, effective batch of eight, learning rate, precision, and sequence cap. Each configuration ran 24 steps and discarded six warmup steps.

| Configuration | Median seconds/step | Peak allocated GPU GB |
|---|---:|---:|
| Microbatch 4, accumulation 2, decoder checkpointing, projection chunk 64 | 0.510 | 1.231 |
| Microbatch 8, accumulation 1, no decoder checkpointing, projection chunk 128 | 0.185 | 2.989 |

The faster configuration showed a 2.76× speedup per optimizer step in this short comparison, at higher memory use. These are two operating points, not simultaneous speed and memory improvements. The 300-step candidate used the faster configuration and measured about 0.181 seconds/step and 3.01 GB peak allocation. Laptop power and temperature can change timings; this was not a repeated controlled benchmark. BF16 trajectories can differ across microbatch partitions.

Both configurations retain answer-only vocabulary projection and trim right padding. CPU tests compare loss and every trainable gradient against the standard Transformers implementation, including nonzero LoRA adapters and decoder checkpointing. The implementation explicitly supports the project's Qwen2 architecture.

The sampler now uses deterministic shuffled epochs. This bounded run drew 2,400 unique examples, about 46.4% of the 5,168-row corpus; it did not train on every row. `--sampling replacement` reproduces the earlier sampling policy. Telemetry separates real input tokens, answer tokens, padded decoder positions, and nominal sequence capacity.

## Evaluation controls and limits

Both checkpoints use BF16, batch 32, greedy generation, 128 output tokens, sequence cap 160 for loss, and no adapter fusion or length bucketing. `--strict-inference` rejects mismatched settings, while the existing task-only comparison remains available for hardware/precision migrations. The regression gate checks parse rate, exact match, and every field, with a stated absolute tolerance of 0.02. Category reports remain diagnostic and are not separately gated.

The 50-example independent set was frozen before this comparison. A separate agent wrote it without reading the new training data, but knew prior failure types. It is a small diagnostic set, not blinded human evaluation or a representative deployment sample. It is now an observed benchmark; future development needs a new untouched test set. See [benchmark methodology](independent-benchmark.md).

The candidate receives additional optimization on a changed data mixture. A gain therefore cannot be attributed to synthetic data alone. Only one training seed was tested. All evaluated inputs explicitly supply all five fields; reliable abstention when facts are absent remains future work.

## Optional inference features

`scripts/extract.py --evidence` returns source spans alongside the predicted record. `--require-grounding` exits with code 2 and a `needs_review` result when a predicted value is absent from the input. The checker preserves original offsets and accents and handles canonical Unicode, case, whitespace, and numeric boundaries. It checks literal presence: a wrong founding year that appears in the sentence can still pass. It is not a correctness or confidence score.

Adapter fusion and prompt length bucketing are available as optional evaluation flags. They remain disabled by default. In a 64-example profile, fusion alone changed two previously correct answers into errors; the combined fused/bucketed run recovered the original aggregate score but did not establish output equivalence. The recorded time includes generation, secondary loss, scoring, and serialization, so it is total evaluation time, not request latency.

## Reproduce locally

Run from `ji-review` using the existing environment and cached model:

```powershell
$env:HF_HOME = Join-Path $PWD '.cache/huggingface'
$env:HF_HUB_OFFLINE = '1'
$env:PYTHONUTF8 = '1'

.venv/Scripts/python.exe scripts/train.py --data data/experiments/binding-v3b/train_replay.jsonl --out ckpt/my-v3b-run --method lora --init-adapter ckpt/laptop-lora-v2 --seq-len 160 --steps 300 --warmup-measure 10 --lr 0.0001 --micro-batch 8 --grad-accum 1 --loss-chunk 128 --no-gradient-checkpointing

.venv/Scripts/python.exe scripts/evaluate.py --model ckpt/laptop-lora-v2 --data data/heldout.jsonl --n 200 --seq-len 160 --batch 32 --out eval/my-baseline.json
.venv/Scripts/python.exe scripts/evaluate.py --model ckpt/my-v3b-run --data data/heldout.jsonl --n 200 --seq-len 160 --batch 32 --out eval/my-candidate.json --compare eval/my-baseline.json --strict-inference

.venv/Scripts/python.exe scripts/extract.py --model ckpt/laptop-lora-v2 --text "Mira Chen, 37, joined Aster Works in 2019 as a systems engineer." --evidence --require-grounding
```

Choose a new empty output directory for each training run. Warm starts load adapter weights only and create a fresh optimizer. For the lower-memory training configuration, use microbatch 4, accumulation 2, projection chunk 64, and `--gradient-checkpointing`.

Validation at experiment time: 160 CPU tests and Ruff passed. The real-tokenizer audit found no overlong training or independent-test examples at the selected cap. Exact sentence and full-record overlap checks found no matches between the new training mixture and the three evaluation sets; this does not rule out shared vocabulary or structural similarity.

Publication preserves generated data and measurement files byte-for-byte with `.gitattributes`. Original-source hashes describe the Windows working tree used for the experiment; existing original files and generator source may use different line endings in other checkouts. Before publication, the three warm-start metadata files had their personal absolute adapter paths replaced by repository-relative paths, and the summary's metadata hashes were refreshed. Measured values and model weights were unchanged.

The next substantial quality improvement should come from representative human-reviewed documents and an explicit policy for missing or ambiguous facts. More templated rows alone cannot establish production readiness.
