## **Bring your code. ChatGPU adapts, optimizes, and runs it across a network of idle GPUs.**

We want people to rent out their GPUs when they aren't using them, making spare capacity, including consumer chips, available for training, simulations, and experimentation.

But personal machines can disconnect at any time and vary enormously in hardware and software. Users cannot tune their code for every possible device. Our agents handle that adaptation, checking changes against the original behaviour and using telemetry to troubleshoot failures and improve future runs.

## How to use it

1. **Create a new job.** Drop in any relevant files, tell ChatGPU what you want it to do, and click **Submit project**.

   <img src="https://raw.githubusercontent.com/ji24077/HTN/main/docs/assets/devpost/new-job.png" alt="Create a job by uploading files and describing what you want ChatGPU to do." width="800" style="max-width: 100%; height: auto;">

2. **Watch your agent work.** Follow along as it plans, tests, and supervises your run.

   <img src="https://raw.githubusercontent.com/ji24077/HTN/main/docs/assets/devpost/run-supervision.png" alt="The agent&#x27;s planning history and job supervisor during a training run." width="800" style="max-width: 100%; height: auto;">

3. **Reap the rewards.** Review your results and download the outputs, such as your trained model and evaluation metrics.

   <img src="https://raw.githubusercontent.com/ji24077/HTN/main/docs/assets/devpost/completed-results.png" alt="A completed job with a model checkpoint and evaluation metrics ready to download." width="800" style="max-width: 100%; height: auto;">

## How it works

### Network

Each contributor runs a worker that connects to ChatGPU and reports its runtime,
memory, and supported workloads. Workers initiate authenticated outbound
connections, so contributors do not need to expose their computers to incoming
connections. Owners can pause their workers when they need their machines back.

The scheduler matches tasks to compatible, available workers. Supported
environments include CPU, NVIDIA CUDA, AMD ROCm, and Apple MPS, with eligibility
checked for each workload. The dashboard receives live updates, while PostgreSQL
stores job state, assignments, accepted results, and execution history.

<img src="https://raw.githubusercontent.com/ji24077/HTN/main/docs/assets/devpost/network.png" alt="The dashboard and durable control service coordinate contributed CPU and GPU workers through outbound connections." width="800" style="max-width: 100%; height: auto;">

### Agent to run: from request to results

We use **GPT-6 Astra** to power job planning and run supervision.
The planning agent reads the uploaded code and the user's request, identifies
dependencies and output requirements, and chooses compatible hardware. If a
requirement is unclear, it asks the user. It can also consult specialist agents
about dependencies, parallelization, and validation before deciding on a plan.

**Adaptation, migration, and optimization form one preparation stage.** The agent
applies the supported steps the workload needs, using small worker runs to
measure and validate its choices. Once preparation passes, workers execute the
full workload, validate its outputs, and save the results for download. A run
supervisor follows progress and investigates problems throughout the pipeline.

<img src="https://raw.githubusercontent.com/ji24077/HTN/main/docs/assets/devpost/run-pipeline.png" alt="Adaptation, migration, and optimization form one preparation stage between planning and full execution." width="800" style="max-width: 100%; height: auto;">

#### Inside adaptation, migration, and optimization

<img src="https://raw.githubusercontent.com/ji24077/HTN/main/docs/assets/devpost/preparation-loop.png" alt="The preparation stage proposes supported changes, tests them against the original, and revises them using measured feedback." width="800" style="max-width: 100%; height: auto;">

The loop starts with the original program and fixed acceptance criteria.
Candidates pass code checks before running on the relevant hardware. Compilation
errors, numerical differences, and performance measurements become feedback for
the next proposal. The agent can revise supported parts of the implementation
within its budget, while the definition of a correct result stays fixed.

- **Adaptation** changes how work is organized. Independent computations can be
  divided into batches, with checks that their combined results agree with the
  original on the tested cases.
- **Migration** adapts supported hardware-specific code for a different chip.
  The target's numerical results are checked against a reference from the
  original environment.
- **Optimization** compares alternative implementations using repeated
  measurements. A candidate must preserve the required behaviour and meet its
  performance criteria to replace the accepted version.

A rejected optional optimization leaves the last accepted code in place. A
required adaptation or migration must pass before full execution can begin.
Accepted code is frozen for the full run, and the user's requested workload and
quality requirements remain unchanged. The transformations available depend on
the workload and runtime; code that already fits can proceed without rewriting.

### When a run fails: the Sentry feedback loop

We use **Sentry** as our telemetry platform across the system, collecting errors,
logs, traces, and performance measurements. Failures can come from lost workers,
missing dependencies, compilation errors, resource limits, or outputs that fail
validation. ChatGPU records the affected job, worker, and attempt so the supervisor
can connect Sentry evidence to the run it is investigating.

The supervisor reads this evidence alongside current job state and its saved
findings. Its response depends on the cause:

| What happened | How the system responds |
| --- | --- |
| A temporary failure or worker loss | Retry eligible work within existing limits, or arrange compatible replacement capacity where the execution policy permits it. |
| A candidate fails compilation or validation | Feed the evidence back to the preparation agent for a bounded revision. Keep the accepted version when an optional optimization is rejected. |
| A required change cannot pass, or the workload cannot fit | Stop with an explanation, or ask the user for a missing decision. Preserve the original output and quality requirements. |
| Sentry is unavailable | Continue investigating from stored execution logs and job state. |

<img src="https://raw.githubusercontent.com/ji24077/HTN/main/docs/assets/devpost/sentry-failures.png" alt="Failures feed stored execution evidence and Sentry into diagnosis, followed by bounded recovery, candidate revision, a user decision, or an explained stop." width="800" style="max-width: 100%; height: auto;">

The planner owns code adaptation and validation; the supervisor manages the
run's supported recovery actions. Every action is checked against the job's scope
and limits, and the next observation establishes whether recovery worked.
Findings, actions, and follow-ups persist across supervisor runs. The scheduler
handles heartbeats, leases, and routine retries independently of model calls.

### Using telemetry to improve model optimization and migration

The same execution data can help us improve future runs. Preparation records
include hardware and runtime details, original and candidate code identities,
numerical checks, timings, rejection reasons, and final outcomes. Successful,
failed, skipped, and fallback outcomes are saved, with structured Sentry Logs
providing a way to search and investigate them.

**We plan to use this evidence to improve our model optimization and migration
algorithms:** identify which changes help particular workloads and chips, learn
from recurring compatibility failures, and turn those failures into regression
cases. Candidate changes can then be evaluated against both correctness and
performance evidence before adoption. A faster result must still meet the
original quality requirements.

Our existing development agent already investigates recorded incidents and
proposes fixes with tests for human review. We want to build on that foundation
with better migration strategies and optimization choices informed by previous
runs. Broader learning across runs and training agents on these records are
future uses of the data.

<img src="https://raw.githubusercontent.com/ji24077/HTN/main/docs/assets/devpost/telemetry-improvement.png" alt="Telemetry from successful and failed runs informs proposed migration and optimization improvements, which are tested and reviewed before use in future runs." width="800" style="max-width: 100%; height: auto;">

### Example flow: NVIDIA training code on an AMD GPU

A user brings a training project with supported GPU code written for NVIDIA and
asks to run it on AMD. The model, data, training settings, and acceptance criteria
must stay consistent. With compatible source and target workers available, the
flow is:

1. **Establish a reference.** Run a small, unchanged workload on NVIDIA and save
   its numerical results and measurements.
2. **Optimize where useful.** Test supported implementation changes against the
   original on the source GPU. Keep a candidate only when repeated measurements
   and correctness checks justify it; otherwise, retain the original.
3. **Migrate the accepted code.** Translate supported CUDA code to HIP for AMD.
   Compilation and execution feedback guide any permitted revisions.
4. **Validate on the target.** Run the bounded workload on AMD and compare its
   results with the fixed NVIDIA reference. A failed required migration returns
   to the revision loop within the job's limits, or stops with an explanation.
5. **Run the full job.** Freeze the validated code and train on AMD for the
   requested workload. Reload and evaluate the final checkpoint, then return
   the model and results if they pass the original checks.

<img src="https://raw.githubusercontent.com/ji24077/HTN/main/docs/assets/devpost/nvidia-to-amd.png" alt="An NVIDIA reference and optional optimization feed code migration, AMD validation, full training, and final checkpoint evaluation." width="800" style="max-width: 100%; height: auto;">

The small preparation runs establish evidence for the execution strategy. The
full training run still performs the requested work, and its final result must
satisfy the original validation requirements. This flow prepares code for the
target before the full run begins.

### Keeping runs recoverable and under user control

Worker assignments have renewable leases. A machine must keep reporting to
retain ownership. If it disappears, eligible work can be retried or replanned
within the job's limits. Each attempt has a distinct identity, so a late result
from an old worker cannot overwrite the accepted result. Work tied to a specific
validated machine must continue to respect that placement requirement.

<img src="https://raw.githubusercontent.com/ji24077/HTN/main/docs/assets/devpost/worker-recovery.png" alt="A disconnected worker loses its assignment; a permitted replacement attempt can complete while stale results are rejected." width="800" style="max-width: 100%; height: auto;">

Plans, execution history, validation evidence, and accepted files remain
available after worker cleanup. Users can inspect what happened, download their
outputs, or cancel outstanding work. Optional spending caps are expressed in
Canadian dollars and stop work when estimated usage reaches the cap. Enforcement
is periodic, so these are soft limits. Current estimates use CPU/RAM rates;
owner payouts and payment processing remain future marketplace work.

The same upload flow can host supported Python HTTP services when the user asks
for an endpoint. Readiness checks control routing, and compatible replacement
workers can restart the service behind the same authenticated URL.

## Challenges we ran into

This was the most technically challenging project we have ever built. There were
so many moving parts: the dashboard, scheduler, agents, database, remote workers,
GPU runtimes, and telemetry. Getting each piece working was a challenge. Getting
them to agree on what was happening throughout a job was an even bigger one.

- **Connections were not on our side.** Unreliable connections made integration
  and debugging harder. A worker could become unreachable while a job was still
  in progress, leaving us to distinguish slow execution from a lost machine.
  Heartbeats, renewable assignments, and recovery rules became essential.
- **Every machine brought different constraints.** Hardware, available memory,
  dependencies, and GPU runtimes all affected whether a workload could run.
  Choosing a worker meant checking what the code actually needed, then testing
  the execution plan on that hardware before committing to the full job.
- **Code changes needed evidence.** An agent could propose a migration or
  optimization, but we still needed to establish whether it preserved the
  original behaviour on the tested cases. That meant fixed acceptance criteria,
  reference results, numerical checks, and repeated performance measurements.
  A failed optional optimization also needed a reliable path back to accepted
  code.
- **Recovery created its own coordination problems.** Retries, delayed updates,
  and returning workers all had to fit into one consistent job history. We had
  to track individual attempts, reject stale results, and keep agent actions
  within the user's workload and spending limits.
- **Debugging meant following a job across the whole system.** A visible failure
  could originate in planning, environment setup, compilation, execution, or
  validation. Connecting Sentry telemetry to the relevant job, worker, and
  attempt gave both us and the supervising agent the context to investigate.

Through all of this, we wanted the user experience to stay simple: upload files,
describe the task, and collect the results. Making the system explain its
progress, surface useful failures, and return downloadable outputs took work
across every layer. The hardest part was making all those pieces work together
from submission through to a validated result.

## What we learned

How to stay awake.

## What's next

Sleep.

## For Sponsors

### Sentry: observability that drives agent decisions

**Sentry gives our agents evidence to investigate failures and gives our team a
way to understand behaviour across the whole system.** Our integration includes
Logs, Tracing, Profiling, and Session Replay alongside error monitoring, matching
the emphasis on depth and actionable observability in the
[Sentry sponsor challenge](https://hackthenorth2026.devpost.com/#prizes).

| Product | Technical integration | What it helps us investigate |
| --- | --- | --- |
| Logs | Python reporting enables Sentry Logs; the dashboard captures warning and error console messages. The supervisor can query job-scoped logs through the Sentry API. | What happened around a failure, on which worker, and during which attempt. |
| Tracing | Browser requests propagate trace headers to same-origin API calls. Python workers create `task.execute` transactions tagged with job, task, worker, and execution identities. | Request latency and worker execution time, correlated with the relevant job. |
| Profiling | Python telemetry configures trace-linked profiling, with trace and profile-session sampling set to 100% for the prototype. | Where time is spent inside instrumented Python processes. |
| Session Replay | The dashboard enables error-triggered replay with inputs masked and media blocked. Authentication callback pages are excluded. | The sequence of user interactions leading to a dashboard error. |

<img src="https://raw.githubusercontent.com/ji24077/HTN/main/docs/assets/devpost/sponsor-sentry.png" alt="Sentry evidence supports job-scoped recovery and a separate development loop that proposes tested fixes for review." width="800" style="max-width: 100%; height: auto;">

**The runtime loop is backed by durable state.** The supervisor polls Sentry error
events, saves pagination cursors, and deduplicates alerts into its job inbox. Its
`search_logs` and `get_alert_details` tools use a backend adapter that builds
job-scoped queries and checks returned rows against the same job. Findings,
follow-ups, and recovery actions persist in PostgreSQL. The next observation
checks whether the action helped, and stored execution history remains available
when Sentry cannot be reached.

**The development loop turns incidents into reviewable changes.** A separate
script uses the Sentry CLI to collect an issue, stack trace, recent events, and
logs from the same trace. A coding agent investigates in an isolated checkout,
proposes a fix with a regression test, and produces a pull request for human
review. Merged incident records become context for later investigations. The
export path can also pair recorded incidents with their fix diffs, providing a
foundation for our planned migration and optimization improvements.

Credential scrubbers run before events, logs, breadcrumbs, and transactions leave
the process. Workers receive public ingest configuration during enrollment;
Sentry API credentials stay in the backend.

Implementation: [telemetry coverage](https://github.com/ji24077/HTN/blob/main/docs/sentry.md),
[job supervisor](https://github.com/ji24077/HTN/blob/main/docs/job-supervisor.md), and
[incident-to-PR workflow](https://github.com/ji24077/HTN/blob/main/docs/self-heal.md).

### OpenAI: GPT-6 Astra inside an execution and feedback loop

**GPT-6 Astra turns a user's request into decisions the system can execute and
check.** It powers planning and run supervision through the OpenAI Responses API.
Uploaded code, available hardware, execution evidence, and the user's constraints
provide the context for choosing the next step.

Our model adapter sends JSON Schema function definitions and returns tool calls
to a separate execution loop, following the
[Responses API function-calling pattern](https://developers.openai.com/api/docs/guides/function-calling).
The backend validates tool arguments and authority before dispatch. Supervisor
tools include `get_job`, `list_eligible_workers`, `search_logs`, `take_action`, and
`remember`, so an investigation can progress from observation to action to a
saved finding.

<img src="https://raw.githubusercontent.com/ji24077/HTN/main/docs/assets/devpost/sponsor-openai.png" alt="GPT-6 Astra requests tools through a bounded execution loop that validates calls, saves state, executes actions, and returns observations." width="800" style="max-width: 100%; height: auto;">

**The engineering is in how the loop handles real execution.** The application
saves model output and pending tool activity before side effects, then saves
results before the next model call. Requests use `store=false` with caller-owned
history and encrypted reasoning items retained for continuation. Calls are
sequential, and the default shared loop limits a turn to eight model calls,
eight tool calls, and 90 seconds. Incomplete model responses cannot dispatch
partially generated calls.

If execution is interrupted, the loop records uncertain outcomes and inspects
durable task and action identities before further work. Duplicate tool-call IDs
are rejected. The supervisor also takes a per-job database lock to prevent
concurrent investigations from issuing competing actions. These mechanisms make
long-running supervision possible across multiple short model turns.

**Correctness comes from execution evidence.** The preparation pipeline checks
supported code changes against the original program's acceptance criteria.
Compiler errors, numerical comparisons, and measured performance guide revisions.
The supervisor works within the submitted limits, while the scheduler manages
leases and routine retries. Tests use a replaceable model client to exercise
malformed calls, interrupted turns, timeouts, and persistence behaviour.

**Codex with GPT-6 Astra was central to building ChatGPU.** We ran **16 Astra
worktrees in parallel across our team** for implementation. This was the most
technically complex project we had attempted, and what impressed us most was
Astra's ability to navigate all the moving parts and their dependencies.

We used Codex from the planning stage to scope the project and reason through
the architecture, then throughout implementation. With the dashboard, scheduler,
workers, agents, and telemetry all interacting, understanding how a change fit
into the wider system was a major part of the work. Astra helped us work through
that complexity while our team built different parts in parallel.

**Testing with screenshots was a highlight.** We could show Codex the interface,
review what the user would actually see, and iterate with visual feedback.
That was especially valuable for a flow spanning job submission, agent
supervision, and results. One of us described it as the best testing experience
they had ever had.

Implementation: [model transport](https://github.com/ji24077/HTN/blob/main/backend/src/orchestrator/llm/openai.py),
[agent loop](https://github.com/ji24077/HTN/blob/main/backend/src/orchestrator/agent/loop.py), and
[supervisor tools](https://github.com/ji24077/HTN/blob/main/backend/src/orchestrator/supervisor/tools.py).

Editable Mermaid diagram sources are included alongside the images in [assets/devpost](https://github.com/ji24077/HTN/tree/main/docs/assets/devpost).
