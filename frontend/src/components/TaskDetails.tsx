import { useEffect, useRef, useState } from "react";
import type { Task, ExecutionEvent } from "../api/types";
import { ApiError, executionEvents, getTask } from "../api/client";
import { record, taskTitle, time, workerName } from "../lib/format";
import { statusLabels } from "../lib/jobs";
import { SimulationDetails, phaseLabels } from "./SimulationDetails";
import { JobSupervisor } from "./JobSupervisor";
import { Icon } from "./Icon";

const SETTLED = new Set<Task["state"]>(["succeeded", "failed", "cancelled"]);
const labels: Record<string, string> = {
  queued: "Job queued",
  assigned: "Worker assigned",
  running: "Execution running",
  started: "Runner started",
  succeeded: "Result accepted",
  completed: "Execution completed",
  failed: "Attempt failed",
  cancelled: "Execution cancelled",
  step_started: "Step started",
  step_completed: "Step completed",
};
function message(event: ExecutionEvent) {
  const data = event.data;
  for (const value of [
    data.text,
    data.message,
    record(data.error).message,
    data.error,
    data.reason,
    data.step,
  ])
    if (typeof value === "string") return value;
  if (event.kind === "progress") return `${data.percent ?? 0}% complete`;
  return labels[event.kind] || event.kind.replaceAll("_", " ");
}
function level(event: ExecutionEvent) {
  if (typeof event.data.level === "string")
    return event.data.level.toLowerCase().replace(/^warn$/, "warning");
  return /fail|error|stderr/.test(event.kind)
    ? "error"
    : /warn/.test(event.kind)
      ? "warning"
      : "info";
}
export function TaskDetails({
  task,
  tasks = [],
  onSelectTask,
  onClose,
}: {
  task: Task | null;
  tasks?: Task[];
  onSelectTask?: (task: Task) => void;
  onClose: () => void;
}) {
  const ref = useRef<HTMLDialogElement>(null);
  const taskId = task?.spec.id;
  const [events, setEvents] = useState<ExecutionEvent[]>([]);
  /**
   * The same task re-read on its own, because listings no longer carry `result`.
   * Falls back to the listing copy, so the dialog renders immediately and fills in
   * the output when it arrives.
   */
  const [full, setFull] = useState<Task | null>(null);
  const [error, setError] = useState("");
  const [tab, setTab] = useState("Timeline");
  const [query, setQuery] = useState("");
  const [attempt, setAttempt] = useState("");
  const [severity, setSeverity] = useState("");
  const [worker, setWorker] = useState("");
  const [follow, setFollow] = useState(true);
  const logRef = useRef<HTMLDivElement>(null);
  const cursorRef = useRef({ taskId, value: 0 });
  const settled = task ? SETTLED.has(task.state) : false;
  const isService = task?.spec.kind === "python_service" ||
    record(task?.spec.payload).execution_mode === "service";
  useEffect(() => {
    setEvents([]);
    setError("");
    setTab("Timeline");
    setQuery("");
    setAttempt("");
    setSeverity("");
    setWorker("");
    cursorRef.current = { taskId, value: 0 };
  }, [taskId]);
  useEffect(() => {
    setFull(null);
    if (!taskId) return;
    const controller = new AbortController();
    getTask(taskId, controller.signal)
      .then((one) => {
        if (!controller.signal.aborted) setFull(one);
      })
      // A missing result is not worth an error banner; the panel simply stays empty.
      .catch(() => {});
    return () => controller.abort();
  }, [taskId]);
  useEffect(() => {
    if (!taskId) return;
    const controller = new AbortController();
    let timer: ReturnType<typeof setTimeout>;
    async function poll() {
      let more = false;
      try {
        const page = await executionEvents(
          taskId!,
          cursorRef.current.value,
          controller.signal,
        );
        if (controller.signal.aborted) return;
        if (!Array.isArray(page.events))
          throw new Error("Invalid execution history response");
        cursorRef.current.value = page.next_cursor;
        more = page.has_more;
        setEvents((previous) =>
          [
            ...new Map(
              [...previous, ...page.events].map((event) => [event.id, event]),
            ).values(),
          ].slice(-2000),
        );
        setError("");
        if (!more && settled) return;
      } catch (caught) {
        if (controller.signal.aborted) return;
        if (caught instanceof ApiError && [401, 403].includes(caught.status)) {
          setError("Sign in again to view execution history.");
          return;
        }
        setError("Execution history unavailable; retrying…");
      }
      if (!controller.signal.aborted) timer = setTimeout(poll, more ? 0 : 1500);
    }
    void poll();
    return () => {
      controller.abort();
      clearTimeout(timer);
    };
  }, [taskId, task?.generation, settled]);
  useEffect(() => {
    if (task && !ref.current?.open) ref.current?.showModal();
    else if (!task && ref.current?.open) ref.current.close();
  }, [!!task]);
  useEffect(() => {
    if (follow && tab === "Logs" && logRef.current)
      logRef.current.scrollTop = logRef.current.scrollHeight;
  }, [events, follow, tab]);
  const attempts = [
    ...new Set(events.map((event) => event.attempt).filter(Boolean)),
  ].sort((a, b) => a - b);
  const workers = [
    ...new Set(
      events.map((event) => event.worker_id).filter((id): id is string => !!id),
    ),
  ];
  const visible = events.filter(
    (event) =>
      (!attempt || String(event.attempt) === attempt) &&
      (!worker || event.worker_id === worker) &&
      (!severity || level(event) === severity) &&
      (!query ||
        `${message(event)} ${event.kind} ${JSON.stringify(event.data)}`
          .toLowerCase()
          .includes(query.toLowerCase())),
  );
  const milestones = events.filter(
    (event) =>
      (!attempt || String(event.attempt) === attempt) &&
      !["progress", "stdout", "stderr", "log"].includes(event.kind),
  );
  function download() {
    const url = URL.createObjectURL(
      new Blob([visible.map((event) => JSON.stringify(event)).join("\n")], {
        type: "application/x-ndjson",
      }),
    );
    const anchor = document.createElement("a");
    anchor.href = url;
    anchor.download = `${taskId}-logs.ndjson`;
    anchor.click();
    setTimeout(() => URL.revokeObjectURL(url), 1000);
  }
  return (
    <dialog
      ref={ref}
      id="result-dialog"
      className="detail-dialog"
      aria-labelledby="result-title"
      onClose={onClose}
    >
      <header className="detail-header">
        <div>
          <div className="eyebrow">JOB DETAILS</div>
          <h2 id="result-title">{task ? taskTitle(task) : "Task details"}</h2>
          <p className="mono muted">{task?.spec.job_id}</p>
        </div>
        <button
          className="icon-btn"
          aria-label="Close task details"
          onClick={onClose}
        >
          <Icon name="close" />
        </button>
      </header>
      {task && (
        <>
          <div className="detail-summary">
            <div>
              <span>Status</span>
              <strong className={`status-badge ${task.state}`}>
                <i />
                {task.spec.kind === "simulation_job"
                  ? phaseLabels[String(record(task.spec.payload).phase)] ||
                    statusLabels[task.state]
                  : statusLabels[task.state]}
              </strong>
            </div>
            <div>
              <span>
                {isService ? "Execution" : task.spec.kind === "simulation_job"
                  ? "Pipeline progress"
                  : "Progress"}
              </span>
              <strong>
                {isService ? "Persistent" : `${task.state === "succeeded" ? 100 : Math.round(task.progress)}%`}
              </strong>
            </div>
            <div>
              <span>
                {task.spec.kind === "simulation_job"
                  ? "Allocation"
                  : settled
                    ? "Last worker"
                    : "Current worker"}
              </span>
              <strong>
                {isService && task.spec.kind === "simulation_job" ? "One worker" : task.spec.kind === "simulation_job"
                  ? "Managed by phase"
                  : task.worker_id
                    ? workerName(task.worker_id)
                    : "Unassigned"}
              </strong>
            </div>
            <div>
              <span>
                {task.spec.kind === "simulation_job" ? "Launch" : "Attempts"}
              </span>
              <strong>
                {isService && task.spec.kind === "simulation_job" ? "Automatic recovery" : task.spec.kind === "simulation_job" ? (
                  "After validation"
                ) : (
                  <>
                    {task.generation}{" "}
                    <small>/ {task.spec.max_attempts} allowed</small>
                  </>
                )}
              </strong>
            </div>
          </div>
          {tasks.length > 1 && (
            <div className="task-selector">
              <label htmlFor="detail-task">Task in this job</label>
              <select
                id="detail-task"
                value={task.spec.id}
                onChange={(event) => {
                  const selected = tasks.find(
                    (item) => item.spec.id === event.target.value,
                  );
                  if (selected) onSelectTask?.(selected);
                }}
              >
                {tasks.map((item) => (
                  <option key={item.spec.id} value={item.spec.id}>
                    {taskTitle(item)} · {statusLabels[item.state]}
                  </option>
                ))}
              </select>
            </div>
          )}
          <div className="detail-columns">
            <div className="execution-column">
              {task.spec.kind === "simulation_job" && (
                <SimulationDetails
                  key={task.spec.job_id}
                  jobId={task.spec.job_id}
                  cancelled={task.state === "cancelled"}
                />
              )}
              {task.failure && (
                <div className="failure-banner">
                  <Icon name="warning" />
                  <div>
                    <strong>Latest failure</strong>
                    <p>{task.failure}</p>
                  </div>
                </div>
              )}
              <section
                className="attempt-section"
                hidden={task.spec.kind === "simulation_job"}
                aria-label="Execution attempts"
              >
                <div className="section-heading">
                  <h3>Execution attempts</h3>
                  <button
                    className="text-btn"
                    onClick={() => setAttempt("")}
                    disabled={!attempt}
                  >
                    Show all
                  </button>
                </div>
                <div className="attempt-list">
                  {attempts.map((number) => {
                    const history = events.filter(
                      (event) => event.attempt === number,
                    );
                    const failed = history.find(
                      (event) => event.kind === "failed",
                    );
                    const success =
                      (task.state === "succeeded" &&
                        task.generation === number) ||
                      history.some(
                        (event) =>
                          event.kind === "succeeded" &&
                          event.source === "server",
                      );
                    const cancelled = history.some(
                      (event) => event.kind === "cancelled",
                    );
                    const machine = history.find(
                      (event) => event.worker_id,
                    )?.worker_id;
                    return (
                      <button
                        className={`attempt-card ${attempt === String(number) ? "selected" : ""}`}
                        key={number}
                        aria-pressed={attempt === String(number)}
                        onClick={() =>
                          setAttempt(
                            attempt === String(number) ? "" : String(number),
                          )
                        }
                      >
                        <span className="attempt-number">{number}</span>
                        <div>
                          <strong>{workerName(machine || null)}</strong>
                          <small>
                            {failed
                              ? message(failed).replaceAll("_", " ")
                              : success
                                ? "Completed"
                                : cancelled
                                  ? "Cancelled"
                                  : "Execution started"}
                          </small>
                        </div>
                        <Icon
                          name={
                            failed ? "warning" : success ? "check" : "clock"
                          }
                          size={16}
                        />
                      </button>
                    );
                  })}
                  {!attempts.length && (
                    <p className="muted">Waiting for a worker assignment.</p>
                  )}
                </div>
              </section>
              <section
                className="execution-history"
                aria-label="Execution history"
              >
                <div
                  className="detail-tabs"
                  role="tablist"
                  aria-label="Execution views"
                >
                  {["Timeline", "Logs", "Result"].map((name) => (
                    <button
                      role="tab"
                      id={`tab-${name}`}
                      aria-controls={tab === name ? `panel-${name}` : undefined}
                      aria-selected={tab === name}
                      tabIndex={tab === name ? 0 : -1}
                      onKeyDown={(event) => {
                        const names = ["Timeline", "Logs", "Result"];
                        let next = -1;
                        if (event.key === "ArrowRight")
                          next = (names.indexOf(name) + 1) % names.length;
                        if (event.key === "ArrowLeft")
                          next =
                            (names.indexOf(name) + names.length - 1) %
                            names.length;
                        if (event.key === "Home") next = 0;
                        if (event.key === "End") next = names.length - 1;
                        if (next >= 0) {
                          event.preventDefault();
                          setTab(names[next]);
                          document
                            .getElementById(`tab-${names[next]}`)
                            ?.focus();
                        }
                      }}
                      key={name}
                      onClick={() => setTab(name)}
                    >
                      {name}
                      {name === "Logs" && <small>{events.length}</small>}
                    </button>
                  ))}
                </div>
                {error && (
                  <p role="status" className="inline-alert">
                    {error}
                  </p>
                )}
                {events.length >= 2000 && (
                  <p className="inline-alert">
                    Showing the latest 2,000 events. Earlier events remain
                    available through the task API.
                  </p>
                )}
                {tab === "Timeline" && (
                  <div
                    role="tabpanel"
                    id="panel-Timeline"
                    aria-labelledby="tab-Timeline"
                    className="timeline-panel"
                  >
                    <p className="panel-description">
                      Milestones across{" "}
                      {attempt ? `attempt ${attempt}` : "all attempts"}. Times
                      are local.
                    </p>
                    <ol className="timeline">
                      {milestones.map((event) => (
                        <li key={event.id} className={level(event)}>
                          <span className="timeline-marker" />
                          <div className="timeline-heading">
                            <strong>
                              {labels[event.kind] ||
                                event.kind.replaceAll("_", " ")}
                            </strong>
                            <time
                              dateTime={event.occurred_at}
                              title={new Date(
                                event.occurred_at,
                              ).toLocaleString()}
                            >
                              {time(event.occurred_at)}
                            </time>
                          </div>
                          <p>
                            {event.worker_id || "Scheduler"} ·{" "}
                            {event.attempt
                              ? `Attempt ${event.attempt}`
                              : "Submission"}
                          </p>
                          {message(event) !==
                            (labels[event.kind] ||
                              event.kind.replaceAll("_", " ")) && (
                            <p className="event-message">
                              {message(event).replaceAll("_", " ")}
                            </p>
                          )}
                          <details>
                            <summary>Event details</summary>
                            <pre>{JSON.stringify(event.data, null, 2)}</pre>
                          </details>
                        </li>
                      ))}
                    </ol>
                    {!milestones.length && (
                      <p className="empty-state">
                        No execution events recorded yet.
                      </p>
                    )}
                  </div>
                )}
                {tab === "Logs" && (
                  <div
                    role="tabpanel"
                    id="panel-Logs"
                    aria-labelledby="tab-Logs"
                  >
                    <div className="log-toolbar">
                      <label className="search-field">
                        <Icon name="search" size={16} />
                        <input
                          aria-label="Search execution logs"
                          placeholder="Search logs…"
                          value={query}
                          onChange={(event) => setQuery(event.target.value)}
                        />
                      </label>
                      <div className="log-filters">
                        <select
                          aria-label="Filter by attempt"
                          value={attempt}
                          onChange={(event) => setAttempt(event.target.value)}
                        >
                          <option value="">All attempts</option>
                          {attempts.map((number) => (
                            <option key={number} value={number}>
                              Attempt {number}
                            </option>
                          ))}
                        </select>
                        <select
                          aria-label="Filter by worker"
                          value={worker}
                          onChange={(event) => setWorker(event.target.value)}
                        >
                          <option value="">All workers</option>
                          {workers.map((id) => (
                            <option key={id}>{id}</option>
                          ))}
                        </select>
                        <select
                          aria-label="Filter by severity"
                          value={severity}
                          onChange={(event) => setSeverity(event.target.value)}
                        >
                          <option value="">All levels</option>
                          <option value="error">Errors</option>
                          <option value="warning">Warnings</option>
                          <option value="info">Info</option>
                        </select>
                      </div>
                    </div>
                    <div className="log-meta">
                      <span>
                        {visible.length} of {events.length} events · local time
                      </span>
                      <label>
                        <input
                          type="checkbox"
                          checked={follow}
                          onChange={(event) => setFollow(event.target.checked)}
                        />
                        Follow latest
                      </label>
                      <button
                        className="text-btn"
                        onClick={download}
                        disabled={!visible.length}
                      >
                        <Icon name="download" size={14} />
                        Export
                      </button>
                    </div>
                    <div
                      className="log-output"
                      ref={logRef}
                      tabIndex={0}
                      aria-label="Execution log output"
                    >
                      {visible.map((event) => (
                        <div
                          className={`log-line ${level(event)}`}
                          key={event.id}
                        >
                          <time>{time(event.occurred_at)}</time>
                          <span className="log-level">{level(event)}</span>
                          <div>
                            <span className="log-context">
                              {event.worker_id || "scheduler"} ·{" "}
                              {event.attempt
                                ? `#${event.attempt}`
                                : "submission"}{" "}
                              · {event.kind}
                            </span>
                            <pre>{message(event)}</pre>
                            <details>
                              <summary>Structured data</summary>
                              <pre>{JSON.stringify(event.data, null, 2)}</pre>
                            </details>
                          </div>
                        </div>
                      ))}
                      {!visible.length && (
                        <p className="empty-state">
                          No logs match these filters.
                        </p>
                      )}
                    </div>
                  </div>
                )}
                {tab === "Result" && (
                  <div
                    role="tabpanel"
                    id="panel-Result"
                    aria-labelledby="tab-Result"
                    className="result-panel"
                  >
                    <h3>
                      {task.state === "succeeded"
                        ? "Execution result"
                        : "No successful result yet"}
                    </h3>
                    {(full ?? task).attestation && (
                      <p className="verified">
                        <Icon name="check" size={16} />
                        Device signature verified when this result was accepted.
                      </p>
                    )}
                    {(full ?? task).result != null && (
                      <pre>
                        {typeof (full ?? task).result === "string"
                          ? ((full ?? task).result as string)
                          : JSON.stringify((full ?? task).result, null, 2)}
                      </pre>
                    )}
                    <details>
                      <summary>Task configuration & metadata</summary>
                      <pre id="result-json">
                        {JSON.stringify(full ?? task, null, 2)}
                      </pre>
                    </details>
                  </div>
                )}
              </section>
            </div>
            <div className="supervisor-column">
              <JobSupervisor jobId={task.spec.job_id} />
            </div>
          </div>
        </>
      )}
    </dialog>
  );
}
