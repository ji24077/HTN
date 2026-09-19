import { useEffect, useRef, useState } from "react";
import type { Task, ExecutionEvent } from "../api/types";
import { ApiError, executionEvents } from "../api/client";
import { taskTitle } from "../lib/format";

const SETTLED = new Set<Task["state"]>(["succeeded", "failed", "cancelled"]);
export function TaskDetails({
  task,
  onClose,
}: {
  task: Task | null;
  onClose: () => void;
}) {
  const ref = useRef<HTMLDialogElement>(null);
  const open = task !== null;
  const taskId = task?.spec.id;
  const [events, setEvents] = useState<ExecutionEvent[]>([]);
  const [error, setError] = useState("");
  // The parent streams task updates; read the latest state without restarting the poller.
  const stateRef = useRef(task?.state);
  stateRef.current = task?.state;
  useEffect(() => {
    setEvents([]);
    setError("");
    if (!taskId) return;
    const controller = new AbortController();
    let timer: ReturnType<typeof setTimeout>;
    let cursor = 0;
    const poll = async () => {
      // Only a page fetched after the task settled can be its final page.
      const settled = stateRef.current !== undefined && SETTLED.has(stateRef.current);
      let more = false;
      try {
        const page = await executionEvents(taskId, cursor, controller.signal);
        if (controller.signal.aborted) return;
        if (!Array.isArray(page.events))
          throw new Error("Invalid execution history response");
        cursor = page.next_cursor;
        more = page.has_more;
        setEvents((previous) => [...previous, ...page.events].slice(-2000));
        setError("");
        if (!more && settled) return;
      } catch (caught) {
        if (controller.signal.aborted) return;
        if (caught instanceof ApiError && [401, 403].includes(caught.status)) {
          // Retrying cannot help and would re-trigger session renewal every tick.
          setError("Sign in again to view execution history.");
          return;
        }
        setError("Execution history unavailable; retrying…");
      }
      if (!controller.signal.aborted) timer = setTimeout(poll, more ? 0 : 1500);
    };
    void poll();
    return () => {
      controller.abort();
      clearTimeout(timer);
    };
  }, [taskId]);
  useEffect(() => {
    const dialog = ref.current;
    if (open && !dialog?.open) dialog?.showModal();
    else if (!open && dialog?.open) dialog.close();
  }, [open]);
  return (
    <dialog
      ref={ref}
      id="result-dialog"
      aria-labelledby="result-title"
      onClose={onClose}
    >
      <header>
        <h2 id="result-title">{task ? taskTitle(task) : "Task details"}</h2>
        <button
          className="icon-btn"
          id="close-dialog"
          aria-label="Close task details"
          onClick={onClose}
        >
          ×
        </button>
      </header>
      {task?.attestation && (
        <p className="attestation-status">
          Device signature verified when this result was accepted.
        </p>
      )}
      {task && (
        <section className="execution-history" aria-label="Execution history">
          <h3>Execution history</h3>
          <p>Steps and output from each attempt, including successful runs.</p>
          {error && <p role="status">{error}</p>}
          {!events.length && !error && <p>No execution events recorded yet.</p>}
          {events.length >= 2000 && (
            <p>
              Showing the latest 2,000 events. Earlier events remain available
              through the task API.
            </p>
          )}
          <ol>
            {events.map((event) => (
              <li key={event.id}>
                <div>
                  <time dateTime={event.occurred_at}>
                    {new Date(event.occurred_at).toLocaleTimeString()}
                  </time>
                  {" · "}
                  <strong>{event.kind.replaceAll("_", " ")}</strong>
                  {" · "}
                  {event.attempt ? `Attempt ${event.attempt}` : "Submission"}
                  {" · "}
                  {event.source === "server" ? "Platform" : "Runner"}
                </div>
                {event.worker_id && <small>Machine: {event.worker_id}</small>}
                <pre>{JSON.stringify(event.data, null, 2)}</pre>
              </li>
            ))}
          </ol>
        </section>
      )}
      <pre id="result-json">{task ? JSON.stringify(task, null, 2) : ""}</pre>
    </dialog>
  );
}
