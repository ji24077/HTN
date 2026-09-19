# Local training experiments

For the subsequent targeted-data run, faster training profiles, evidence output,
and the rejected v3b candidate, see [the optimization report](optimization-v3.md).
The measurements below describe the earlier experiments; v2 remains the default.

Worktree: `ji-review`. Branch: `ji-phin-agentinfra`, starting at `5f4a27b`.
Experiments ran locally in an isolated worktree. The Sentry worktree was
unchanged. No paid generation API or rented GPU was used. Model weights and
tokenizer caches remain local; this report accompanies the source/evidence PR.

## What is working

The task is Qwen2.5-0.5B extracting name, age, organization, role and year from
short biographies. This is the trained extraction model, not the scheduling
agent's hosted language model.

- Installed the CUDA environment locally: Python 3.12.10, PyTorch 2.11.0+cu128,
  Transformers 5.17.0 and PEFT 0.21.0. Dependencies are in `uv.lock`.
- Trained an actual LoRA adapter for 200 steps on this laptop's 8 GB RTX 4060.
  Peak allocated memory was **1.225 GB**, peak reserved memory **1.319 GB**.
  The adapter trains 4,399,104 parameters and its weights occupy 17.64 MB.
- Saved adapter: `ckpt/laptop-lora-v2/`. It reloads successfully for inference.
  The base model remains required. Weights, tokenizers and caches stay local;
  checkpoint `meta.json` files remain visible to Git as measurement evidence.
  Adapters are not drop-in full-weight checkpoints for the migration engine.
- Matched evaluation on all 200 original held-out examples: **87.5% exact
  match / 100% valid JSON**, versus 88.5% for the control adapter.
  The two-point regression gate passed. An earlier batch-8 evaluation gave
  88%; the final comparison uses batch 32 for both models.

## Matched experiment result

Both adapters started from the same base model with the same seed, learning
rate, LoRA rank, effective batch size and 200-step training budget. The control
used only the original training file; the mixed model used original plus V2.
Both final evaluations use greedy decoding and generation batch size 32.

| Evaluation | Examples | Original-data control | Synthetic-mixed model |
|---|---:|---:|---:|
| Original heldout: exact match | 200 | 88.5% | 87.5% |
| Unseen synthetic challenge: exact match | 800 | 46.0% | 81.0% |
| Unseen synthetic challenge: valid JSON | 800 | 84.625% | 100% |
| Synthetic validation: exact match | 800 | Not evaluated | 91.75% |
| Synthetic validation: valid JSON | 800 | Not evaluated | 99.625% |

The synthetic examples improved this controlled challenge by 35 percentage
points while original-benchmark exact match fell by one point. This is one
training seed on finite template families, not a general accuracy guarantee.
The 800-example results supersede the preliminary 200-example challenge runs.
Of the 800 paired challenge predictions, the mixed model fixed 300 control
errors and introduced 20 new errors. On the original benchmark it fixed nine
and introduced eleven. Failed quoted-organization examples remain in the
reports; no benchmark labels were changed to improve the scores.

The detailed measurements and matched-data comparisons are in
`experiments/laptop-results.json` and the `eval/laptop-*.json` reports. Each
evaluation records the exact selected examples, settings, every prediction,
field accuracy, category scores and a confidence interval.

## Data available

| Dataset | Training | Validation | Challenge |
|---|---:|---:|---:|
| Original | 1,768 | 200 | — |
| `data/experiments/copy-stress` | 2,000 | 200 | 200 |
| `data/experiments/extraction-v2` | 8,000 | 800 | 800 |

Each experiment also contains `train_mixed.jsonl`: original training examples
plus that experiment's synthetic training split. V2 contains 9,768 mixed rows.
No original data or benchmark was overwritten.

V2 adds accented and hyphenated names, long job titles, quoted and braced
organization names, distracting ages/dates/people, explicit field labels and
multi-line records. Labels are constructed directly from the sentence facts.
These appointments are fictional. This remains a finite template benchmark;
success does not establish general extraction accuracy on arbitrary prose.

Synthetic splits have disjoint names and organizations. Challenge structures
never appear in synthetic training. V2 excludes complete records and normalized
sentences from the original data and the first experiment. The audit confirmed
zero train/challenge sentence, record or name overlaps.

The real tokenizer checked every V2 row and the original heldout: maximum
combined prompt/answer length **102 tokens**, maximum answer length **47**.
`--seq-len 128` therefore fits this dataset without dropping examples. See
`experiments/token-audit-v2.json` (the audit's configured cap was 256).

Generate another distinct version:

```powershell
.venv/Scripts/python.exe scripts/synthetic_lab.py generate --expanded --train-n 8000 --heldout-n 800 --challenge-n 800 --seed 43 --exclude-dir data/experiments/copy-stress --exclude-dir data/experiments/extraction-v2 --out data/experiments/extraction-v3
```

The generator refuses to overwrite an existing experiment directory. Its
manifest records counts, seed, content hashes and generator source hash.

## Memory and correctness changes

`src/gpushare/trainer/sft.py` computes the vocabulary projection only for
positions that predict answer tokens. Checkpointed chunks bound the vocabulary
projection's activation memory. The decoder still sees the full prompt.
This implementation is specific to this project's Qwen2 architecture.

Tests compare its loss and every trainable gradient with the standard Qwen2
loss, including LoRA and decoder checkpointing. Gradient accumulation now
normalizes by total supervised tokens, fixing unequal weighting of short and
long answers across micro-batches. Dynamic padding trims unused sequence tails
while preserving EOS even when EOS and padding share a token ID.

Training offers full-parameter or LoRA mode, optional decoder checkpointing,
configurable projection chunks, fixed RNG seeds and a small default micro-batch.
AdamW avoids the extra foreach tensor-list allocation. FP16 uses FP32 trainable
weights plus autocast/GradScaler. Long examples fail clearly rather than silently
changing the dataset. Modern HF checkpoint saving handles tied embeddings.

A short, matched full-training probe on this RTX 4060 used identical data,
sample order, two examples per step and eight steps, dropping two warmups:

| Path | Peak allocated | Peak reserved | Median step |
|---|---:|---:|---:|
| Standard loss, fixed padding | 4.716 GB | 5.864 GB | 0.1742 s |
| Answer-only loss, trimmed padding | 4.556 GB | 4.792 GB | 0.1482 s |

These are short observations, not robust throughput estimates. BF16 kernels
can produce small numerical differences after changing tensor shapes. The
LoRA run is a different training method and batch configuration, so its 1.225 GB
figure is not a controlled speed comparison against these full-training probes.
Ji's old 13.8 GB/4090 measurement is likewise a different configuration.

## Reproduce training and evaluation

From `ji-review`, use the cached model and installed environment:

```powershell
$env:HF_HOME=Join-Path $PWD '.cache/huggingface'
$env:HF_HUB_OFFLINE='1'
$env:PYTHONUTF8='1'
.venv/Scripts/python.exe scripts/extract.py --text "At age 34, Amara Okafor joined Cedar Quill Research Institute in 2023 as lead accessibility engineer."
.venv/Scripts/python.exe scripts/train.py --method lora --gradient-checkpointing --micro-batch 4 --grad-accum 2 --seq-len 128 --steps 200 --data data/experiments/extraction-v2/train_mixed.jsonl --out ckpt/my-lora-run
.venv/Scripts/python.exe scripts/evaluate.py --model ckpt/my-lora-run --data data/heldout.jsonl --seq-len 128 --n 200 --batch 32 --out eval/my-original.json
.venv/Scripts/python.exe scripts/evaluate.py --model ckpt/my-lora-run --data data/experiments/extraction-v2/challenge.jsonl --seq-len 128 --n 800 --batch 32 --out eval/my-challenge.json
```

Choose a fresh checkpoint directory for every run. This trainer starts from
the base model; it does not resume optimizer state. It fails before training if
the output directory is nonempty.

Compare only fresh reports over identical examples and decoding settings:

```powershell
.venv/Scripts/python.exe scripts/evaluate.py --model ckpt/my-lora-run --data data/heldout.jsonl --seq-len 128 --n 200 --batch 8 --out eval/my-original-comparison.json --compare eval/laptop-control-original.json
```

The comparison gate rejects missing/mismatched provenance and regressions in
parse rate, exact-match rate or individual field accuracy beyond tolerance.
Ji's old reports do not contain provenance, so regenerate them before using
this gate. The committed base score also used 100 rows while the old after
score used 200; they were not a perfectly paired comparison.

## Remaining data limitations

`experiments/data-audit.json` flags 106 original training rows and 12 held-out
rows with non-verbatim labels. These require review, not automatic rejection:
some legitimately normalize the text. For example, `hospital São Lucas` is
labeled `São Lucas Hospital`. Organization/department and role boundaries are
also inconsistent. Changing this benchmark after inspecting model errors would
make comparisons unreliable; agree a labeling policy for a new benchmark.

## Validation and completion sound

84 tests pass in the Python 3.12 environment. Ruff checks pass for all changed
Python files. Tests cover loss/gradient equivalence, variable-length gradient
accumulation, padding/EOS, checkpoint roundtrip, data leakage, repeatability,
overwrite protection, provenance and braces inside JSON strings.

```powershell
.venv/Scripts/python.exe -m pytest -q --basetemp .pytest_cache/local-checks
.venv/Scripts/python.exe scripts/completion_bell.py
```

The second command plays one decaying 880 Hz note for 0.55 seconds through the
Windows audio device. It needs no PowerShell execution-policy change.

API references: [Hugging Face Qwen2](https://huggingface.co/docs/transformers/model_doc/qwen2)
and [PEFT quicktour](https://huggingface.co/docs/peft/main/quicktour).
