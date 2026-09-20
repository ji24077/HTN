# Concurrent analysis agents

Uploaded Python/PyTorch jobs and Monte Carlo jobs have one planning coordinator. At supported planning decisions it can call `delegate_analysis` with up to three independent questions: dependencies, parallelization, and validation. Delegation is optional; straightforward jobs can proceed directly.

The next job tick starts separate asynchronous model requests together. Each analyst has its own assignment and an immutable snapshot of the project context. The coordinator waits for that batch to finish, reads the saved reports, and decides how to proceed. Reports are advisory: existing runtime, validation, placement, and execution gates remain authoritative. Only the coordinator can ask the user, accept a plan, or dispatch worker tasks. Children have only a structured `report_analysis` tool and cannot execute code, edit files, or recursively delegate.

This makes **analysis concurrent**. Monte Carlo execution can separately use concurrent batches on available workers through the existing scheduler. Arbitrary training code is not automatically converted into distributed training.

## Limits and lifecycle

- At most three concurrent child calls per job and six child calls total across all phases and adaptations. Calls are reserved durably before starting, including failed or interrupted calls.
- Each child has a 60-second timeout bounded by the parent job deadline. The configured model client's output limit still applies per response; reported token usage is saved per child. This is a call budget, not an aggregate token or dollar cap, and model charges are not added to the worker compute ledger.
- Parent pause, cancellation, and deadline changes are monitored during inference. Outstanding local model requests are cancelled; this cannot guarantee a provider stops billing an already accepted request.
- Each completed report is checkpointed independently. On backend restart, saved reports remain available and calls left running become interrupted with unknown outcomes. They are never automatically replayed; the coordinator may delegate new work within the remaining budget.
- Revision checks and parent state checks reject stale or late results. The per-job advisory lock prevents two coordinators from running a batch for the same job concurrently.

Job status includes child assignments, status, reports, errors, and observed usage. The private input snapshot is omitted. Job Details displays these reports, and Overview shows running analysis agents.

`backend/tests/integration/test_parallel_analysis.py` uses real PostgreSQL checkpoints and controlled model calls. A barrier requires all three calls to start before any can complete, proving actual overlap without relying on timing comparisons. Tests also cover partial checkpoints, parent controls, deadlines, restart recovery, call limits, isolated failures, and both coordinator paths.
