# Uploaded compute jobs

The supported scope is Python projects and PyTorch on reported CPU, CUDA, or Apple MPS workers. Upload Python scripts or a
ZIP containing Python source and supporting data, describe the result, and optionally
set a spending cap. The agent selects the entrypoint, Python dependencies, compatible worker,
and execution settings from the original source. Blender, native scene rendering, GPU provisioning and non-Python execution remain unsupported.
Explicit GPU-only requests must not be silently downgraded to CPU.

The dashboard submits `execution_mode: "auto"`. Explicit `job` and `service`
submissions remain supported by the API. Automatic hosting requires a hosting
request in the user's description; merely uploading a server file is not enough.
Service planning selects the entrypoint, readiness path, runtime, memory, timeouts,
concurrency and requested lifetime. The scheduler assigns compatible capacity.

All uploaded jobs use the existing durable job state machine, reservations,
deadlines, cancellation, execution history, supervisor, and clarification flow.
`POST /v1/jobs` is the common upload endpoint. The existing `/v1/simulations`
endpoints remain compatible and default to the simulation adapter. Existing
simulation jobs keep their current state and validation behavior.

## Execution and validation

Simulations retain the measured planner: profile the original, propose an
adaptation, compare against fresh reference cases on two distinct workers,
schedule waves, and aggregate results.

Rendering, training, and other programs follow:

1. Inspect the uploaded source and validation code; ask for missing requirements.
2. Choose an available worker, runtime/VRAM requirements, a bounded probe, the
   full execution command, expected outputs, and validation requirements.
3. Run the original program with small probe arguments. A failed probe can
   trigger a revised execution plan within the submitted adaptation budget;
   original source is never rewritten.
4. Review probe timing and choose placement for the full run. Freeze the full
   command, validator, metric thresholds, and output contract. Worker loss can
   change placement, but cannot silently reduce the requested work.
5. Execute the full program and the uploaded validator on the worker. Validate
   deliverables, persist files, and return the final result. Failed full-run
   validation ends the job; it does not loosen the acceptance criteria.

The program adapter currently runs one Python program on one worker. It does
not partition rendering frames or implement distributed training, GPU rental,
checkpoint transfer/resume, or Ji/Phinn's optimization loop. Programs must support their selected runtime; the planner selects existing compatible workers but does not provision GPUs.

Workers must run the updated `WORKER_EXECUTOR=python_project` executor. It
advertises `python_program` support so older simulation-only workers are not
given program tasks. Python dependencies are installed automatically as described
below. The standard Docker agent includes Python 3.12 and CPU PyTorch 2.13.0.
Existing local process isolation
and credential stripping still apply; this change does not introduce a new
untrusted-code sandbox.

## Automatic Python dependencies

The planning agent selects additional Python packages from the original code,
including the validator and service entrypoint, and saves them in the plan's
`dependencies` list. It maps import names to distribution names (for example,
`PIL` to `Pillow`), excludes standard-library and uploaded local modules, and
explains its choices in the plan summary. A manifest is optional. Ambiguous
package identities use the existing clarification flow.

Before each execution, the worker finds the nearest `requirements.txt` or
`pyproject.toml`, walking from the selected working directory to the upload root.
`requirements.txt` takes precedence in the same directory; otherwise static
`[project].dependencies` are used. Nested requirements and relative wheel paths
are resolved from the manifest directory. Dynamic dependency metadata needs a
`requirements.txt`; Poetry-specific declarations and optional dependency groups
are not automatically selected.

Manifest requirements and agent-selected packages are resolved together by pip.
Explicit pins are preserved: incompatible selections fail setup. Uploaded source
and manifests are not rewritten. If neither declares packages, the existing worker
environment is used without an installation step.

Installation happens in a fresh environment inside the attempt's temporary
workspace, with access to the worker's preinstalled Python libraries (including
PyTorch). New packages never install into the worker environment. The bundled PyTorch version
is constrained during resolution so extra dependencies cannot replace the worker’s selected build. Execution
and validation use the resulting interpreter. The worker must provide Python's
`venv` and `ensurepip`, as the Python Docker images do. Install logs appear in task
history, setup time counts toward task/startup limits, and cancellation stops
installers with the project process group. Cleanup removes the environment.
Environments are not cached between attempts; pin versions for repeatable resolution.

This applies to simulations, Python programs, and hosted services. It does not
install OS packages, Blender, GPU drivers or model weights, or enable a GPU that the worker cannot access. Rebuild/redeploy Python worker
images to use the updated executor.

## Project contract

For custom Python programs, provide an entrypoint and a validation script. The entrypoint
must support a bounded probe without source modification, such as a small
frame size or training step count. The validator must fail with a nonzero exit
or exception when outputs do not meet the requested criteria. For training,
it should reload the saved checkpoint and evaluate it; for rendering, it should
decode outputs and check the requested frames and dimensions.

Both scripts receive `DISPATCH_OUTPUT_DIR`, a dedicated output directory. CLI
arguments may use `{output_dir}`; the worker substitutes its actual path.
Source and input files are preserved. Only declared program output files are
published. Paths are relative to that directory; glob patterns and links are
not supported. Ordinary simulation code may also write files to this directory;
those are uploaded from successful worker executions.

- Rendering requires at least one declared image plus the uploaded validator.
- Training requires a nonempty checkpoint, `metrics.json`, and at least one
  finite numeric metric with an acceptance bound. The agent must take bounds
  from the user request or the original evaluation policy, or ask the user.
- Other Python programs need declared output files and an uploaded validator.

The worker enforces metric bounds even if the validator exits successfully.
Image content checks and checkpoint-format checks belong to the supplied
validator; a nonempty file alone is not evidence of model or image quality.

Current input limits are 128 MiB expanded, 128 KiB of Python source and 100 files.
Outputs have no fixed per-file or per-job byte cap, with at most 100 files per job
including staged attempts. Available worker/server storage and the job deadline
still limit transfers. Defaults remain 30 minutes and three planning/adaptation
attempts. Uploading large pretrained models or datasets still needs a separate
input transport; increasing output capacity does not remove the input limit.

Planners can explicitly reject a job before dispatch or after a probe when
reported hardware, requirements or execution evidence show it will not fit, or
when the available tools cannot perform the requested operation on the inputs.
The reason and evidence are saved with the failed job. The running supervisor
can cancel it with an explanation after examining capacity and execution logs.
Neither agent may silently reduce the requested workload. Missing telemetry or
temporarily busy machines alone are not evidence of insufficient capacity.

## Downloading results

The job view lists output files and provides a final `result.json` download.
Worker uploads, PostgreSQL storage and server downloads use bounded chunks
(1 MiB storage chunks), rather than loading an entire checkpoint into memory.
The browser's download manager receives the file directly; JavaScript does not
build a full-file Blob. Download routes support HEAD for authorization checks.
Outputs remain separate from the 64 KiB task result and survive worker workspace
cleanup and service restarts. Existing inline output records remain readable.
Interrupted uploads are hidden and cleaned up, and Content-Length is required
and checked on worker uploads.

- `GET /v1/jobs/{job_id}`: shared project status and plan.
- `POST /v1/jobs/{job_id}/answer`: answer a planner clarification.
- `GET /v1/jobs/{job_id}/outputs`: metadata for accepted task outputs.
- `GET /v1/jobs/{job_id}/outputs/{output_id}`: download one file.
- `GET /v1/jobs/{job_id}/result`: download the completed root job result.

Downloads use the existing fleet-admin authentication. Worker uploads require
the active task token, worker identity, attempt number, and a valid running
lease; authorization is checked again when committing. Outputs from failed or
superseded attempts are not downloadable. Bytes are immutable within an
attempt, and retries with identical content are idempotent. Binary data never
rides in fleet snapshots or model context.

## Runnable examples

Start with [the PyTorch training example](../examples/projects/pytorch/README.md).
It uses real tensor operations, saves a checkpoint, reloads it, and validates held-out
error. The agent infers `torch` from its imports; no requirements file is needed.

Upload the two `.py` files in `examples/projects/rendering/` and request:

> Render the Mandelbrot image at 256 by 256 pixels. Use size 16 for the probe,
> validate the full image, and return mandelbrot.ppm.

Upload the two `.py` files in `examples/projects/training/` and request:

> Train for 200 steps, using 2 steps for the probe. Reload the saved checkpoint
> and require held-out MSE <= 0.001. Return checkpoint.json and metrics.json.

Both examples use only Python's standard library. They exercise real program
execution and file downloads without renting GPUs. Integration tests use real
PostgreSQL, worker subprocesses, and HTTP artifact transfers with controlled
model proposals; they do not establish live-model or physical-GPU performance.

```sh
RUN_SUPERVISOR_TESTS=1 uv run --project backend --python 3.12 --extra demo python -m unittest discover -s backend/tests -p test_uploaded_programs.py -v
```

For a live-model smoke test, use:

```sh
uv run --env-file .env --project backend --python 3.12 --extra demo python examples/uploaded_project_smoke.py
```

This creates temporary PostgreSQL, a loopback HTTP/WebSocket server, and two
local worker processes. It sends three jobs through the actual model planner:
rendering, successful training, and deliberately undertrained output that must
fail validation. It downloads accepted files, verifies their hashes and content,
and saves reports under `.local/project-smoke/`. The configured production
database and fleet are not used; only model API calls leave the machine.

## GPU worker eligibility

Finite programs may request CPU, CUDA, or MPS. The planner sees connected, unpaused,
free workers registered for Python project execution; program dispatch also requires
`python_program`. A CUDA badge from a different adapter (such as ONNX inference) does
not establish Python/PyTorch execution support. The serving Python interpreter must
pass a real tensor operation before advertising GPU capability. Restart workers after
installing PyTorch so they report the new runtime and libraries.

Automatic device selection prefers a compatible GPU. Explicit GPU-only requests never
fall back to CPU. The worker exposes `DISPATCH_DEVICE` to both the entrypoint and validator;
the original source must use it or otherwise support the selected device. Services and
simulation equivalence checks retain their CPU execution contract.

Deploy the updated backend as well as the worker: older backend plans still enforce CPU-only
requirements. Previously rejected jobs must be resubmitted. Native Windows Python bridging
is not supported; NVIDIA desktops can use the Linux Docker worker with GPU access.
