import { useEffect, useState } from "react";
import {
  answerSimulation,
  executionEvents,
  serviceAction,
  type SimulationStatus,
} from "../api/client";
import type { ExecutionEvent } from "../api/types";

export function ServicePanel({
  status,
  refresh,
  view = "Overview",
}: {
  status: SimulationStatus;
  refresh: () => void;
  view?: string;
}) {
  const [busy, setBusy] = useState(false);
  const [error, setError] = useState("");
  const [copied, setCopied] = useState(false);
  const [answer, setAnswer] = useState("");
  const [logs, setLogs] = useState<ExecutionEvent[]>([]);
  const task = status.tasks.find((item) => item.id === status.service?.task_id);
  useEffect(() => {
    setLogs([]);
    if (!task) return;
    const controller = new AbortController();
    let timer: ReturnType<typeof setTimeout>;
    let cursor = 0;
    const poll = async () => {
      try {
        const page = await executionEvents(task.id, cursor, controller.signal);
        if (controller.signal.aborted) return;
        cursor = page.next_cursor;
        setLogs((previous) => [...previous, ...page.events].slice(-500));
      } catch {
        if (!controller.signal.aborted) setError("Service logs unavailable.");
      }
      if (!controller.signal.aborted) timer = setTimeout(poll, 2000);
    };
    void poll();
    return () => {
      controller.abort();
      clearTimeout(timer);
    };
  }, [task?.id]);
  async function act(operation: "stop" | "restart") {
    setBusy(true);
    setError("");
    try {
      await serviceAction(status.job_id, operation);
      refresh();
    } catch (cause) {
      setError(
        cause instanceof Error ? cause.message : "Service action failed",
      );
    } finally {
      setBusy(false);
    }
  }
  return (
    <section className="simulation-progress" aria-label="Service status">
      <div hidden={view !== "Overview"}>
        <h3>Service · {status.phase.replaceAll("_", " ")}</h3>
        <p>{status.message}</p>
        <label>Endpoint</label>
        <pre>
          {new URL(status.service!.endpoint + "/", window.location.origin).href}
        </pre>
        <button
          className="outline-btn"
          onClick={async () => {
            try {
              await navigator.clipboard.writeText(
                new URL(status.service!.endpoint + "/", window.location.origin)
                  .href,
              );
              setCopied(true);
            } catch {
              setError("Select and copy the endpoint above.");
            }
          }}
        >
          {copied ? "Copied" : "Copy endpoint"}
        </button>
        <p className="muted">
          Use your fleet access token to call this endpoint. HTTP responses can
          stream.
        </p>
        <p>
          {status.workers.join(", ") || "Waiting for a worker"} ·{" "}
          {status.service!.restarts} restarts
        </p>
        {status.service!.ready_at && (
          <p>
            Ready since {new Date(status.service!.ready_at).toLocaleString()}
          </p>
        )}
        <p>
          {status.deadline
            ? `Expires ${new Date(status.deadline).toLocaleString()}`
            : "Runs until stopped"}
        </p>
        {status.question && status.phase === "needs_input" && (
          <form
            onSubmit={async (event) => {
              event.preventDefault();
              setBusy(true);
              try {
                await answerSimulation(status.job_id, answer);
                refresh();
              } catch (cause) {
                setError(
                  cause instanceof Error
                    ? cause.message
                    : "Could not save answer",
                );
              } finally {
                setBusy(false);
              }
            }}
          >
            <label>
              {status.question}
              <input
                value={answer}
                onChange={(event) => setAnswer(event.target.value)}
              />
            </label>
            <button disabled={busy || !answer.trim()}>Submit answer</button>
          </form>
        )}
        <button
          className="outline-btn danger"
          disabled={busy || status.service!.desired === "stopped"}
          onClick={() => void act("stop")}
        >
          Stop service
        </button>{" "}
        <button
          className="outline-btn"
          disabled={busy}
          onClick={() => void act("restart")}
        >
          Restart service
        </button>
        {error && <p role="alert">{error}</p>}
      </div>
      <div hidden={view !== "Details"}>
        {status.service?.plan_summary && <p>{status.service.plan_summary}</p>}
        <details>
          <summary>Execution history ({status.tasks.length})</summary>
          {status.tasks.map((item) => (
            <p key={item.id}>
              {item.worker_id || "Unassigned"} · {item.state} {item.failure}
            </p>
          ))}
        </details>
      </div>
      {view === "Activity" && (
        <details open>
          <summary>Current execution logs</summary>
          <pre>
            {logs
              .map((item) =>
                typeof item.data.text === "string"
                  ? item.data.text
                  : typeof item.data.message === "string"
                    ? item.data.message
                    : JSON.stringify(item),
              )
              .join("\n") || "No service logs recorded."}
          </pre>
        </details>
      )}
    </section>
  );
}
