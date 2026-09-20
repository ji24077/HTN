import { useEffect, useState } from "react";
import {
  ApiError,
  answerSimulation,
  cancelTask,
  executionEvents,
  readSimulation,
  type SimulationStatus,
} from "../api/client";
import type { ExecutionEvent } from "../api/types";
import { time } from "../lib/format";
import { ServicePanel } from "./ServicePanel";
import { JobOutputs } from "./JobOutputs";
import { AnalysisAgents } from "./AnalysisAgents";

export const phaseLabels: Record<string, string> = {
  pending: "Waiting for worker",
  starting: "Starting service",
  ready: "Ready",
  restarting: "Restarting service",
  stopped: "Stopped",
  submitted: "Inspecting request and project",
  service_planning: "Planning how to host your project",
  preparing: "Reserving preprocessing worker",
  profiling: "Measuring execution cost",
  planning: "Choosing execution policy",
  sampling: "Preparing validation cases",
  checking_aggregation: "Validating aggregation",
  scheduling: "Planning the next wave",
  aggregation_wait: "Reserving aggregation worker",
  inspecting: "Inspecting project",
  original: "Checking original code",
  adapting: "Adapting code",
  reference: "Generating reference cases",
  testing: "Testing adaptation",
  validation_wait: "Waiting for second worker",
  distributed_reference: "Generating independent cases",
  distributed_validation: "Validating across two workers",
  allocating: "Allocating full run",
  running: "Running simulation",
  aggregating: "Combining results",
  completed: "Completed",
  failed: "Failed",
  cancelled: "Cancelled",
  needs_input: "Needs your input",
  program_preparing: "Preparing Python worker",
  program_planning: "Planning the project",
  program_probe: "Testing the execution plan",
  program_placement: "Reviewing measured performance",
  program_ready: "Preparing the full run",
  program_running: "Executing and validating outputs",
};
export function SimulationDetails({
  jobId,
  cancelled,
  view = "Overview",
  onStatus,
}: {
  jobId: string;
  cancelled: boolean;
  view?: string;
  onStatus?: (status: SimulationStatus) => void;
}) {
  const [status, setStatus] = useState<SimulationStatus | null>(null);
  const [error, setError] = useState("");
  const [answer, setAnswer] = useState("");
  const [busy, setBusy] = useState(false);
  const [selected, setSelected] = useState("");
  const [logs, setLogs] = useState<ExecutionEvent[]>([]);
  const [logError, setLogError] = useState("");
  const [revision, setRevision] = useState(0);
  useEffect(() => {
    const controller = new AbortController();
    let timer: ReturnType<typeof setTimeout>;
    async function poll() {
      try {
        const next = await readSimulation(jobId, controller.signal);
        if (controller.signal.aborted) return;
        if (
          !next.limits ||
          !Array.isArray(next.tasks) ||
          !Array.isArray(next.checks) ||
          !Array.isArray(next.versions)
        )
          throw new Error("Invalid project status");
        setStatus(next);
        onStatus?.(next);
        setError("");
        if (
          ["completed", "failed", "cancelled", "stopped"].includes(
            next.phase,
          ) &&
          !next.cleanup?.pending_workers.length
        )
          return;
      } catch (cause) {
        if (controller.signal.aborted) return;
        setError("Project status unavailable.");
        if (cause instanceof ApiError && [401, 403, 404].includes(cause.status))
          return;
      }
      timer = setTimeout(poll, 2000);
    }
    void poll();
    return () => {
      controller.abort();
      clearTimeout(timer);
    };
  }, [jobId, revision, cancelled, onStatus]);
  const selectedState = status?.tasks.find(
    (task) => task.id === selected,
  )?.state;
  useEffect(() => {
    setLogs([]);
    setLogError("");
    if (!selected) return;
    const controller = new AbortController();
    let timer: ReturnType<typeof setTimeout>;
    let cursor = 0;
    const started = Date.now();
    async function poll() {
      try {
        const page = await executionEvents(selected, cursor, controller.signal);
        if (controller.signal.aborted) return;
        setLogs((previous) => [...previous, ...page.events].slice(-2000));
        cursor = page.next_cursor;
        if (
          !page.has_more &&
          ["succeeded", "failed", "cancelled"].includes(selectedState || "") &&
          Date.now() - started > 15000
        )
          return;
        timer = setTimeout(poll, page.has_more ? 0 : 1500);
      } catch {
        if (!controller.signal.aborted)
          setLogError(
            "Execution logs unavailable. Select the step again to retry.",
          );
      }
    }
    void poll();
    return () => {
      controller.abort();
      clearTimeout(timer);
    };
  }, [selected, selectedState]);
  if (!status)
    return <p className="inline-alert">{error || "Loading job progress…"}</p>;
  if (view === "Files") return null;
  if (status.service)
    return (
      <>
        {error && (
          <p role="status" className="inline-alert">
            {error}
          </p>
        )}
        {view === "Details" && <AnalysisAgents analysis={status.analysis} />}
        <ServicePanel
          status={status}
          view={view}
          refresh={() => setRevision((value) => value + 1)}
        />
      </>
    );
  return (
    <section className="simulation-progress" aria-label="Project progress">
      <div hidden={view !== "Overview"}>
        <div className="section-heading">
          <h3>{phaseLabels[status.phase] || status.phase}</h3>
        </div>

        {status.analysis?.children?.some((child) => child.status === "running") && (
          <p role="status">
            {status.analysis.children.filter((child) => child.status === "running").length}{" "}
            analysis agents running concurrently. Reports appear in Details.
          </p>
        )}
        {status.plan &&
          [
            "running",
            "aggregating",
            "completed",
            "failed",
            "cancelled",
          ].includes(status.phase) && (
            <p aria-label="Trial progress">
              {(status.trial_counts?.succeeded || 0).toLocaleString()} /{" "}
              {status.plan.trials.toLocaleString()} trials completed
              {status.trial_counts?.failed
                ? ` · ${status.trial_counts.failed.toLocaleString()} failed`
                : ""}
            </p>
          )}
        {status.cleanup && status.cleanup.required > 0 && (
          <p role="status" className="muted">
            {status.cleanup.pending_workers.length
              ? `Waiting for process and file cleanup on ${status.cleanup.pending_workers.join(", ")}. An offline worker must reconnect to confirm cleanup.`
              : "Worker cleanup confirmed: execution processes stopped and temporary project files removed."}
          </p>
        )}
        {!["completed", "failed", "cancelled"].includes(status.phase) && (
          <button
            className="outline-btn"
            disabled={busy}
            onClick={async () => {
              setBusy(true);
              try {
                await cancelTask(jobId);
                setRevision((value) => value + 1);
              } catch (cause) {
                setError(
                  cause instanceof Error
                    ? cause.message
                    : "Could not cancel job",
                );
              } finally {
                setBusy(false);
              }
            }}
          >
            {busy ? "Working…" : "Cancel & clean up workers"}
          </button>
        )}
        {error && (
          <p role="status" className="inline-alert">
            {error}
          </p>
        )}
        <div
          className="pipeline-steps"
          hidden={["failed", "cancelled", "completed"].includes(status.phase)}
        >
          {(status.program_plan
            ? ["Plan", "Probe", "Run & validate outputs"]
            : ["Preprocess", "Validate on two workers", "Full run"]
          ).map((label, i) => (
            <span
              key={label}
              className={
                i ===
                ([
                  "distributed_reference",
                  "distributed_validation",
                  "validation_wait",
                  "program_probe",
                  "program_placement",
                ].includes(status.phase)
                  ? 1
                  : [
                        "allocating",
                        "scheduling",
                        "aggregation_wait",
                        "running",
                        "aggregating",
                        "completed",
                        "program_running",
                        "program_ready",
                      ].includes(status.phase)
                    ? 2
                    : 0)
                  ? "current"
                  : ""
              }
            >
              {i + 1}. {label}
            </span>
          ))}
        </div>
        {status.workers.length > 0 &&
          !["completed", "failed", "cancelled"].includes(status.phase) && (
            <p className="muted">Workers: {status.workers.join(", ")}</p>
          )}
        {status.question && status.phase === "needs_input" && (
          <form
            className="question-callout"
            onSubmit={async (event) => {
              event.preventDefault();
              setBusy(true);
              try {
                await answerSimulation(jobId, answer);
                setAnswer("");
                setRevision((value) => value + 1);
              } catch (cause) {
                setError(
                  cause instanceof Error
                    ? cause.message
                    : "Could not send answer",
                );
              } finally {
                setBusy(false);
              }
            }}
          >
            <strong>Needs your input</strong>
            <p>{status.question}</p>
            <label className="field-label" htmlFor="simulation-answer">
              Your answer
            </label>
            <textarea
              id="simulation-answer"
              value={answer}
              onChange={(event) => setAnswer(event.target.value)}
              required
            />
            <button className="outline-btn" disabled={busy || !answer.trim()}>
              Send answer
            </button>
          </form>
        )}
      </div>
      {view === "Overview" && status.phase === "completed" && (
        <JobOutputs jobId={jobId} phase={status.phase} compact />
      )}
      <div hidden={view !== "Details"}>
        <AnalysisAgents analysis={status.analysis} />
        {status.plan && (
          <details>
            <summary>Execution plan</summary>
            <p>{status.plan.summary}</p>
            <p>
              {status.plan.trials.toLocaleString()} trials
              {status.plan.batch_size
                ? ` · batches of ${status.plan.batch_size}`
                : ""}
              {status.plan.workers ? ` · ${status.plan.workers} workers` : ""}
            </p>
            <small>
              Comparison: relative tolerance 1e-8, absolute tolerance 1e-10.
              Original files are preserved.
            </small>
          </details>
        )}
        {status.program_plan && (
          <details>
            <summary>Execution plan</summary>
            <p>{status.program_plan.summary}</p>
            <p>
              Selected machine: {status.program_plan.worker_id} ·{" "}
              {status.program_plan.requirements.runtime}
            </p>
            <p>Validation: {status.program_plan.validator}</p>
            {!!status.program_plan.dependencies?.length && (
              <p>
                Python dependencies:{" "}
                {status.program_plan.dependencies.join(", ")}
              </p>
            )}
            <ul>
              {status.program_plan.outputs.map((output) => (
                <li key={output.path}>
                  {output.path} · {output.kind}
                </li>
              ))}
            </ul>
            {status.program_plan.metrics.map((metric) => (
              <p key={metric.name}>
                {metric.name}:{" "}
                {metric.minimum !== null ? `minimum ${metric.minimum}` : ""}{" "}
                {metric.maximum !== null ? `maximum ${metric.maximum}` : ""}
              </p>
            ))}
          </details>
        )}
        {status.policy && (
          <details>
            <summary>Agent execution policy</summary>
            <p>{status.policy.rationale}</p>
            <p>
              {status.policy.local_cases} local validation cases ·{" "}
              {status.policy.independent_cases} independent cases ·{" "}
              {status.policy.aggregation} aggregation
            </p>
          </details>
        )}
        {status.schedule && (
          <details>
            <summary>
              Agent schedule · {status.schedule.batches.length} task
              {status.schedule.batches.length === 1 ? "" : "s"}
            </summary>
            <p>{status.schedule.rationale}</p>
            <ul>
              {status.schedule.batches.map((batch, index) => (
                <li key={index}>
                  {batch.trials.toLocaleString()} trials on {batch.worker_id} ·{" "}
                  {batch.timeout_seconds}s timeout
                </li>
              ))}
            </ul>
            <p className="muted">
              Aggregation: {status.schedule.aggregation_worker} ·{" "}
              {status.schedule.aggregation_timeout_seconds}s timeout
            </p>
          </details>
        )}
        {!!status.measurements?.length && (
          <details>
            <summary>Measured execution time</summary>
            <ul>
              {status.measurements.map((m, index) => (
                <li key={index}>
                  {phaseLabels[m.stage] || m.stage} · {m.worker_id} ·{" "}
                  {m.trials.toLocaleString()} trials
                  <br />
                  Compute {(m.compute_seconds * 1000).toFixed(1)} ms · worker
                  time {m.execution_seconds.toFixed(2)}s · total{" "}
                  {m.observed_wall_seconds.toFixed(2)}s
                </li>
              ))}
            </ul>
          </details>
        )}
        {!!status.decisions?.length && (
          <details>
            <summary>
              Agent planning history ({status.decisions.length})
            </summary>
            <ol>
              {status.decisions.map((d, index) => (
                <li key={index}>
                  <strong>{phaseLabels[d.stage] || d.stage}</strong>
                  <p>
                    {d.proposal.rationale ||
                      d.proposal.summary ||
                      d.proposal.explanation ||
                      d.tool}
                  </p>
                </li>
              ))}
            </ol>
          </details>
        )}
        {status.checks.length > 0 && (
          <div className="validation-results">
            <h4>Validation checks</h4>
            {status.checks.map((check, index) => (
              <details key={index}>
                <summary>
                  <span
                    className={`status-badge ${check.passed ? "succeeded" : "failed"}`}
                  >
                    {check.passed ? "Passed" : "Failed"}
                  </span>{" "}
                  Round {check.round} ·{" "}
                  {check.stage === "distributed_validation"
                    ? "Independent two-worker validation"
                    : phaseLabels[check.stage] || "Adaptation test"}
                </summary>
                <pre>{JSON.stringify(check, null, 2)}</pre>
              </details>
            ))}
          </div>
        )}
        {status.versions.length > 0 && (
          <details>
            <summary>
              Adapted code & versions ({status.versions.length})
            </summary>
            {status.versions.map((version) => (
              <details key={version.round}>
                <summary>
                  Version {version.round} · {version.digest.slice(0, 12)}
                </summary>
                <p>{version.explanation}</p>
                <pre>{version.code}</pre>
              </details>
            ))}
          </details>
        )}
        {status.validated_hash && (
          <p className="mono muted">
            {status.program_plan ? "Probed source" : "Validated package"}:{" "}
            {status.validated_hash.slice(0, 16)}
          </p>
        )}
      </div>
      <details
        className="pipeline-executions"
        open={view === "Activity"}
        hidden={view !== "Activity"}
      >
        <summary>Worker executions & logs ({status.tasks.length})</summary>
        <p className="muted">
          Select an execution to read its worker output. The timeline below
          shows job-level scheduler events.
        </p>
        <div className="pipeline-task-list">
          {status.tasks.map((task) => (
            <button
              key={task.id}
              type="button"
              className={selected === task.id ? "selected" : ""}
              onClick={() => setSelected(selected === task.id ? "" : task.id)}
            >
              <span>
                {task.role}
                <small>
                  {task.worker_id || "Queued"} · attempt {task.generation}
                </small>
              </span>
              <span className={`status-badge ${task.state}`}>{task.state}</span>
            </button>
          ))}
        </div>
        {selected && (
          <div
            className="log-output"
            tabIndex={0}
            aria-label="Preprocessing execution logs"
          >
            {logs.map((event) => (
              <div
                key={event.id}
                className={`log-line ${event.kind === "stderr" ? "error" : ""}`}
              >
                <time>{time(event.occurred_at)}</time>
                <span>{event.kind}</span>
                <pre>
                  {typeof event.data.text === "string"
                    ? event.data.text
                    : typeof event.data.message === "string"
                      ? event.data.message
                      : JSON.stringify(event.data)}
                </pre>
              </div>
            ))}
            {!logs.length && <p>{logError || "No logs recorded yet."}</p>}
          </div>
        )}
      </details>
    </section>
  );
}
