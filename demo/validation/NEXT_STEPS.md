# Work that would make the demo materially stronger

The next product milestone is a real request surviving an optimization or GPU
switch, with its answer preserved and its actual response time measured. The
existing kernel-translation demo and fixed-model speed experiments establish
parts of that flow; neither establishes the entire product.

## 1. Validate the model that will actually ship

The speed benchmark is frozen on experimental v3b. The extraction CLI defaults
to v2 because v3b failed its earlier appointment-year regression gate. Keep those
decisions separate. Select the release checkpoint using representative labelled
development data, then repeat the same speed and migration checks on that exact
checkpoint. Preserve the current evidence instead of relabelling it as v2.

The expanded 300-case suite is useful for regressions, but its sources have
already informed development. A separate, untouched test set should include
permitted real examples, human-reviewed labels, dates unrelated to appointment,
multiple people, and long or punctuated names and titles. Define what to return
when a required fact is absent or genuinely ambiguous. The current fixed schema
and complete-fact examples do not establish reliable abstention.

## 2. Connect the quality gate to the actual migration action

For the existing model-serving path, record the source checkpoint identity and
reference answers. Prepare the destination, verify the transferred files, load
the model, and send real HTTP requests before switching traffic. Permit a switch
only after the required per-case output checks and the selected performance
objective pass. A failed destination must leave the source usable.

Exercise interrupted transfers, wrong checkpoint hashes, failed SSH, insufficient
GPU memory and changed predictions. Verify that a failed or cancelled operation
does not display success, lose the source, or leave an owned test pod billing.
The current evaluator/HTTP validation does not exercise the browser's migration
button or prove that this traffic-switching behavior is integrated.

## 3. Measure the experience under realistic load

Extend the existing serial request measurements with distinct input lengths and
multiple concurrent clients. Measure time to first token, complete response time,
p50/p95 latency, errors, queue wait, throughput and peak memory. Check streaming
and non-streaming answers against the same reference. Keep compilation, model
loading, transfer and network time separate and visible.

Do not change model, precision, generation length or total work to manufacture a
speedup. Keep failed candidates and slow trials. A warmed 13-case result cannot
establish cold-start behavior or production tail latency.

## 4. Choose GPUs using the workload and cost

Use fresh provider quotes and measured results to compare response time and cost
per completed, quality-accepted request. Include preparation overhead when
deciding whether migration is worthwhile. For a fixed sequential workload, the
simple time break-even is extra setup time divided by time saved per request;
it applies only when the destination is actually faster and answer quality passes.

Require the expected request volume and load pattern instead of inventing them.
A larger or more expensive GPU is not automatically the best destination for
this small extraction model. Conversely, these measurements do not predict its
ranking for larger models or batched training.

## 5. Broaden translation coverage and feed evidence into skills

Add independent project fixtures beyond the existing AXPY custom kernel: a
reduction, noncontiguous tensors, a custom backward path, and a dependency with
different CUDA/ROCm support. Run each original and translation on its respective
GPU, retaining output, gradient and training-state comparisons. Unsupported
features should produce a concrete diagnostic before renting resources.

The repository already has a versioned skill store. Connect new measured
outcomes to that store as proposed lessons with exact GPU, library, model and
suite identities. A rejected optimization is useful evidence, but it must not
become an active recommendation. Promote only through the existing trusted
regression runner; keep prior versions available for rollback.

Recommended order: release-checkpoint validation, real serving-path integration,
failure recovery, then load/cost selection and broader translation fixtures.
Those steps turn isolated speed measurements into a credible migration product.
