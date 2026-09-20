# Expanded RunPod hardware measurements

Ten distinct NVIDIA models and RunPod's one listed AMD model, MI300X, have measured results. The unsuccessful 5090 startup attempt is also retained. Every time below comes from a GPU run. New and reused artifacts are indexed in `index.json`.

The checkpoint is the frozen experimental v3b adapter, not the default v2 release checkpoint. Times are median resident response seconds on 13 fixed sentences, five rounds per engine. Setup, compilation and network are excluded. This is one host per model, not a guaranteed GPU ranking.

| GPU | Original (s) | Compiled (s) | Observed ratio | Changed / 300 vs own original | Accepted optimization |
|---|---:|---:|---:|---:|---|
| RTX 3090 | 1.197 | 0.197 | 6.06x | 1 | Rejected |
| RTX 4090 | 0.639 | 0.142 | 4.51x | 2 | Rejected |
| RTX 5090 | — | — | — | — | Incomplete |
| RTX A5000 | 1.383 | 0.234 | 5.91x | 0 | Yes |
| RTX A6000 | 1.381 | 0.236 | 5.84x | 1 | Rejected |
| A40 | 1.450 | 0.268 | 5.41x | 1 | Rejected |
| L4 | 1.781 | 0.345 | 5.16x | 1 | Rejected |
| L40S | 1.702 | 0.254 | 6.69x | 0 | Yes |
| A100-SXM4-80GB | 1.593 | 0.256 | 6.23x | 3 | Rejected |
| H100 80GB HBM3 | 1.009 | 0.201 | 5.02x | 1 | Rejected |
| MI300X OAM | 1.136 | 0.248 | 4.58x | 13 | Rejected |
| RTX 5080 | 0.997 | 0.196 | 5.09x | 1 | Rejected |

An observed ratio is not an accepted speedup: both the repeated latency test and the expanded 300-case check must pass. Even a changed answer that improves accuracy fails the agreed exact-value policy. Ground-truth gains and per-case regressions remain in `summary.json`.

## Migration from the original 4090

| Destination, compiled | Changed / 300 vs original 4090 | Previously correct cases regressed | Accepted migration |
|---|---:|---:|---|
| rtx3090 | 1 | 0 | Rejected |
| rtx4090 | 2 | 0 | Rejected |
| rtx5090 | Not tested | — | Not validated |
| a5000 | 2 | 0 | Rejected |
| a6000 | 1 | 0 | Rejected |
| a40 | 1 | 0 | Rejected |
| l4 | 2 | 0 | Rejected |
| l40s | 3 | 0 | Rejected |
| a100 | 4 | 1 | Rejected |
| h100 | 2 | 0 | Rejected |
| mi300x | 15 | 5 | Rejected |
| rtx5080 | 1 | 0 | Rejected |

New session-duration estimate: **$3.086**. Cumulative estimate: **$7.001 / $15**. These are duration × quoted compute rate plus conservative storage allowances, not invoices.

All campaign-owned pods confirmed terminated: **True**.

[Concrete failure cases and next diagnoses](../../hardware/FINDINGS.md). The expanded code passes 658 local tests; the new nine completed GPU runs retain 5,400 validation predictions, 1,170 timed requests and 117 actual HTTP requests. These are repeated evaluations of the fixed suites, not that many distinct sentences.

## Recompute

```bash
PYTHONPATH=src python scripts/summarize_gpu_matrix.py \
  --index demo/results/hardware-matrix-2026-09-19/index.json \
  --out demo/results/hardware-matrix-2026-09-19
```

Raw inputs: [300-case suite](../../validation/cases300.jsonl), [latency sentences](../inference-latency-2026-09-19/inputs.json). Each target folder contains raw outputs, environment/provenance and execution logs where collected. The 4090 and MI300X link to earlier evidence. The 3090 receives the expanded check here. The 5090 allocation never finished container startup within ten minutes and was terminated; a separately budgeted 5080 run fills that slot, with the failed 5090 attempt retained.
