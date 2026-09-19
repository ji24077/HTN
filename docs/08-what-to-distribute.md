# What is actually worth distributing

**Written:** 2026-09-19, from measurements on this fleet — 4 machines, 41 logical cores:
a 15-core Mac, a 12-core Windows laptop, an 8-core Mac on campus wifi, and a 6-core iPhone.

## The one-sentence test

**Split the work into pieces that never need to talk to each other. If you cannot, do not
distribute it.**

Everything below is a consequence of that.

## The three conditions

A job is worth spreading across machines when all three hold.

**1. The items are independent.** Piece 37 must not need piece 36's answer. Evolution
qualifies: score 320 mutated gaits, none of which knows the others exist. Training one
neural network by gradient descent does not — every step depends on the last.

**2. Compute per item exceeds transfer per item.** Sending an item costs a round trip. If
the machine finishes before the next item arrives, the network is the bottleneck and one
local core would have been faster.

**3. There are enough items.** This is the one people get wrong, and it is measurable
rather than a matter of taste.

## The batch-size floor

From `docs/01-architecture.md`, for two hosts with rates `r_A`, `r_B` and a cold-start
cost `s_B`:

```
S = N·(r_A + r_B) / (r_A·(N + r_B·s_B))
```

Speedup `S` rises with the number of items `N` toward a ceiling of `(r_A + r_B)/r_A`. For
this fleet's measured rates, reaching a 1.2× speedup needs **N ≥ 677 items**.

Two consequences worth internalising:

- **Below a few hundred items, distributing makes the job slower.** The transfer and
  cold-start costs are not amortised.
- **The ceiling is fixed by the rate ratio.** If one machine is more than ~5× slower than
  another, a 1.2× speedup is unreachable at *any* batch size. The right response is to say
  so, not to hunt for a flattering configuration.

## The barrier problem — measured here today

Raising `maxConcurrency` on the fast Mac took the fleet from 7 concurrent slots to 15. The
walker got **slower**: ~170–200 gaits/s before, ~135–155 after.

The cause is that a generation is a **synchronous barrier**. Population 320 at 20 gaits per
task is 16 tasks, and the next generation cannot start until all 16 return. With 15 slots
and 16 tasks, nearly everything is claimed at once and the generation ends when the
*slowest* machine finishes its single task. Extra slots on the fast machine bought
nothing, while ten concurrent tasks contended for cores also running Postgres and the
control service.

**With a barrier, throughput is bounded by the slowest participant, not by total
capacity.** Adding capacity to the fast machine cannot help; either give the fleet more
items per barrier so the fast machines absorb more, or size each machine's share to its
measured rate, or remove the barrier.

This is the most useful thing the fleet has taught us, and it was invisible until someone
added capacity and watched the number go down.

## Jobs that fit

- **Parameter sweeps and hyperparameter search.** Each configuration is independent. The
  canonical good case.
- **Evolutionary and population methods** — ES, CMA-ES, ARS, genetic algorithms. What the
  walker does: 128,000 gait simulations in 556 s across four machines.
- **Batch inference over many inputs.** MNIST here: 98.90% on 10,000 digits, one laptop at
  10,962 digits/s and another at 7,666/s. Split by item, never split a single model.
- **Monte Carlo and simulation ensembles.** Independent by construction.
- **Rendering frames**, one frame per task.
- **Brute-force search** over a partitioned space — hashes, combinatorics, test matrices.
- **Large test suites**, if the tests are genuinely isolated.

## Jobs that do not fit

- **Training one large model by gradient descent.** Every step needs the previous one, and
  synchronising gradients over home internet costs far more than it saves. This is what
  one GPU is for.
- **Anything needing shared memory** or a consistent view of mutable state.
- **Sequential simulation** where step *n+1* needs step *n*.
- **Small jobs.** Below the floor, one machine wins.
- **Latency-sensitive work.** A round trip is tens to hundreds of milliseconds.
- **Work whose data dwarfs its compute.** Shipping a gigabyte to do a second of work.

## Fleet versus one GPU

Not competitors — they answer different shapes of question.

| | This fleet | One GPU |
|---|---|---|
| Best at | many independent CPU tasks | dense linear algebra, one big tensor graph |
| Scales by | adding machines | buying a bigger card |
| Tolerates 400 ms latency | yes | no |
| Cost | idle machines you own | $/hour, or capital |
| A phone can join | yes | no |

A GPU will beat this fleet at training a network, and it is not close. This fleet will beat
a single GPU at *ten thousand independent small simulations*, because the GPU runs them one
after another while four machines run them at once — and no framework here needs installing
before a machine can help.

## Practical rule

Before distributing, ask: **how many independent pieces, and how long does each take?**

- Fewer than ~500 pieces → run it locally.
- Each piece under ~100 ms → make the pieces bigger first.
- Pieces that need each other's results → the wrong shape; reformulate or use one machine.
- Hundreds to millions of independent pieces, each taking real time → this is what the
  fleet is for.
