---
name: portability
description: Optimize complete PyTorch scripts and translate their GPU dependencies and custom kernels between NVIDIA CUDA and AMD HIP, with real execution checks and regression-gated engineering lessons.
---

# GPU portability engineering

Accept a complete runnable script, its dependencies, and reproducible sample or
generated inputs. Inspect imported libraries, embedded GPU code, build settings,
precision, autograd behavior, and device assumptions before changing code.
If required context is missing, report exactly what prevents execution.

## Preserve the computation

Keep the original script, inputs, model, batch size, training steps, seed, and
precision as the reference. Preserve float32 unless the input uses another
precision; reduced precision requires an explicit user option. Do not reduce
work, remove the custom operation, detach gradients, substitute a CPU/reference
implementation, or weaken assertions to make a conversion pass. Set numerical
tolerances before evaluating a candidate; report failures against those fixed
tolerances. A compiler exit code alone does not establish correctness.

## Engineering loop

1. Run the unchanged source on its intended GPU. Record actual GPU identity,
   backend and library versions, precision, outputs, gradients, training loss,
   completed steps, process GPU memory, and synchronized timings after warmup.
2. Make portable optimizations to avoid unnecessary work and transfers. Verify
   correctness against the unchanged source before keeping any optimization.
3. Translate for the selected GPU. Resolve the dependency and build environment
   as well as source code. Use established conversion tooling where appropriate,
   then inspect and repair the result. An unchanged PyTorch operation that already
   runs on ROCm is portability evidence, not proof of a custom-kernel conversion.
4. Compile and execute on the target GPU. Compare outputs, gradients, and training
   behavior to the fixed reference. Verify that the intended custom kernel and
   GPU backend actually execute. Reject silent CPU or reference-code fallback.
5. Tune target-specific kernel configuration only after translation is correct;
   rerun the same correctness checks after each retained change. Measure speed
   against the same workload on the same GPU. Cross-GPU timings describe a
   migration tradeoff, not an isolated optimization speedup.
6. Retain artifacts for every stage: input hashes, patches, environment,
   compiler/runtime output, fixed-suite hash, correctness results, timing
   samples, and GPU execution evidence. Label unexecuted results as unverified.

Use a bounded repair loop and the caller's execution time and spending limits.
Report the failing operation when an unsupported feature blocks translation.
Do not invent support for architecture-specific assembly, proprietary libraries,
distributed collectives, or other features outside the tested scope. CUDA to HIP
and HIP to CUDA need independent execution evidence; one direction passing does
not establish the reverse direction.

## Learn from verified results

Propose a concise Markdown lesson with the tested GPU identifiers, exact library
versions, input/artifact hashes, and the evidence supporting it. A project-specific
fix belongs in that project's skill store. A reusable rule belongs in shared
memory with its applicability attached; never extrapolate one GPU result to every
architecture or library version.

Candidate lessons stay inactive until a trusted runner tests the combined skill
against the existing fixed regression suite. The runner, not the generated code
or the model, reports success and GPU execution. Preserve the suite and its hash:
the agent may propose additional cases but cannot delete, weaken, or alter the
tests deciding promotion. Every existing case must pass with real GPU evidence.
Automatically promote only after those checks; no additional human gate is
required. Keep immutable versions and support rollback to an explicit prior
version. Failed verification keeps the previous active skill.

Store only scrubbed evidence. Credentials and unrelated project data are not
learning material. Skill updates change future agent instructions; they do not
train or change the coding model's weights.
