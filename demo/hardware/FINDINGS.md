# What the wider hardware coverage found

The same frozen model can produce different answers when the GPU or inference
engine changes. The 13-sentence latency set missed regressions that the expanded
300-case existing-code check caught. Inspect the full raw comparisons in the
[hardware results](../results/hardware-matrix-2026-09-19/README.md).

| Compiled destination | Case | Correct original value | Regressed compiled value |
|---|---|---|---|
| RTX 3090, RTX 5080, A40, RTX A6000 | `original-133`, role | `zoo veterinarian` | `zoovet veterinarian` |
| L4 | `challenge-39`, year | `2013` | `1973` |
| A100 SXM | `independent-37`, organization | `Vale: Experimental Arts` | `Vale` |

A5000 and L40S preserved every original JSON value on this 300-case suite and
passed the repeated speed gate. H100 changed one answer in the direction of the
ground truth, but that still fails the agreed exact-value rule. The model itself
gets only about 90% of the complete records right; preserving its outputs is not
a certificate of extraction accuracy.

An optimization on one GPU and migration from the original 4090 are separate
decisions. A5000 and L40S can pass their own original-versus-compiled comparison
while still differing from the 4090 reference. The summary computes both gates
independently from indexed raw outputs.

The next useful diagnosis is to compare token logits at the first divergent
token for these cases, with model, prompt, dtype and decoding fixed. Record the
selected attention/cache/compiler paths and change one setting at a time. The
current experiments identify observable failures; they do not establish which
kernel or floating-point operation caused each failure.

The default release model is v2. These measurements remain labelled as the
experimental v3b checkpoint. A release decision needs the selected release
weights and a separate untouched evaluation set, followed by the same hardware,
speed and actual serving-path checks.
