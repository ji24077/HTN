# A broader RunPod test matrix

The matrix contains **10 distinct NVIDIA models and every AMD model in RunPod's
paid Pod catalog: MI300X**. Ten MI300X instances would still be one GPU model, so
replicas are not counted as hardware diversity.

| Vendor | Distinct targets |
|---|---|
| NVIDIA | RTX 3090, RTX 4090, RTX 5080, RTX A5000, RTX A6000, A40, L4, L40S, A100 SXM 80 GB, H100 SXM 80 GB |
| AMD | MI300X 192 GB |

The complete Secure and Community catalog reads are saved alongside this file,
with timestamps and query parameters. Neither read applies a CUDA filter, which
could wrongly hide AMD. At the snapshot, MI300X was listed for paid Secure Cloud
but had no stock. The unsupported Community price field is not a rentable offer.
Availability changes; each rental must recheck the exact GPU, cloud, region,
driver requirement and quote. The existing MI300X measurements remain useful
even when a fresh instance is temporarily unavailable.

## What is actually measured

See [raw results and recomputed comparisons](../results/hardware-matrix-2026-09-19/README.md).
The RTX 4090 and MI300X results come from the earlier completed runs. The expanded
campaign tests eight new NVIDIA types and fills the 3090's missing 300-case gate.
The inventory alone never means a GPU has passed, is faster, or supports an
arbitrary CUDA project.

The original 5090 allocation did not finish container startup within its
ten-minute startup limit. It was terminated and retained as an unmeasured
failure; a separately budgeted 5080 allocation fills that slot. The dated initial
plan is preserved rather than rewritten to hide that failure.

Each new run uses the same frozen Qwen2.5-0.5B base revision and experimental v3b
adapter bytes. This is still not the default v2 release checkpoint. The checks are:

1. Verify the rented GPU name, one visible device, the expected GPU runtime, and
   a real synchronized matrix calculation. Refuse CPU fallback.
2. Run the unchanged `scripts/evaluate.py` on the fixed 300-case suite. Compare
   the compiled candidate's raw JSON values per case, not only aggregate accuracy.
3. Measure resident baseline and compiled latency on the same 13 sentences,
   five rounds per engine, with loading, compilation and network excluded.
4. Run the unchanged HTTP server and send real requests over an SSH tunnel.
5. Collect outputs and logs, terminate the owned pod, and verify termination.

The new NVIDIA runs use
`pytorch/pytorch:2.10.0-cuda12.8-cudnn9-runtime@sha256:b85566342b86d13a67712e9315d40cdc2dad7f8d86df1aff3831f80835edbcca`,
Transformers 5.17.0 and PEFT 0.21.0, preserving the image's GPU-enabled Torch.
Actual GPU names, library versions, device activity and full model-file hashes
are retained per run. [Reproduce the latency command](../LATENCY.md#reproduce-on-an-already-owned-gpu)
and the [300-case validation](../validation/README.md) on a compatible owned GPU.

These are deployment comparisons on one host per model, including host CPU and
vendor-runtime effects. They do not isolate silicon or establish performance
for other models, larger batches, training, multi-GPU jobs or custom kernels.
The input suite has already informed development and is not a fresh blind test.

## Bounded spending

The main expansion reserves **$7.62** across eight separate 45-minute sessions,
including **$0.15/hour per pod** for storage, plus **$4.25** reserved for earlier
work. The 3090 quality follow-up reserves another **$0.28**, so the combined
reservation is $12.15. The replacement 5080 adds **$0.41**, bringing the combined
maximum reservation to **$12.56 of $15**, including the full original 5090 reserve.
The earlier completed cost estimate was
$3.9154403115; the larger $4.25 reserve leaves accounting headroom.

Each pod receives an independent worker immediately; no paid pod waits in a
worker queue. A local watchdog and controller cleanup restrict deletion to owned
pods. The watchdog depends on this machine and its network, so this is not a
provider-enforced spending cap. The campaign does not retry uncertain creates.

`gpu_matrix.expansion_plan` rejects a set of reservations whose combined lifetime
cost exceeds the remaining budget. Individual `runpod_session` profiles still
check the exact model, count, region, quote and ownership. A catalog entry is not
added to the existing simulated `CHIPS` calibration table without measurements.

## Recreate the inventory plan without renting

Run from the repository root, with a new output filename:

```bash
PYTHONPATH=src python scripts/plan_gpu_matrix.py \
  --secure demo/hardware/catalog-secure-2026-09-19.json \
  --community demo/hardware/catalog-community-2026-09-19.json \
  --prior-reserve-usd 4.25 \
  --out demo/hardware/review-plan.json
```

This command is offline and read-only toward RunPod. It produces a plan with
unmeasured performance fields, not invented benchmark numbers. The snapshots
and price ceilings are dated; obtain fresh quotes before a future rental.
