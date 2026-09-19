import { useState, type SubmitEvent } from "react";
import type { TaskSpec, Worker } from "../api/types";
import { stubTask } from "../api/client";
import { workerName } from "../lib/format";

export function TaskComposer({
  selected,
  onSelect,
  workers,
  busy,
  onSubmit,
}: {
  selected: string;
  onSelect: (id: string) => void;
  workers: Worker[];
  busy: boolean;
  onSubmit: (tasks: TaskSpec[]) => Promise<void>;
}) {
  const [name, setName] = useState("Render preview");
  const [duration, setDuration] = useState(30);
  const [failover, setFailover] = useState(true);
  const ids = [
    ...new Set(["worker-a", "worker-b", ...workers.map((worker) => worker.id)]),
  ];
  function submit(event: SubmitEvent<HTMLFormElement>) {
    event.preventDefault();
    if (name.trim())
      void onSubmit([stubTask(name.trim(), selected, duration, failover)]);
  }
  return (
    <section className="composer">
      <div className="composer-header">
        <h2>Dispatch a task</h2>
        <span className="composer-arrow">↗</span>
      </div>
      <p>A small piece of work. Your destination.</p>
      <form id="task-form" onSubmit={submit}>
        <label className="field-label" htmlFor="task-name">
          Task name
        </label>
        <input
          id="task-name"
          type="text"
          value={name}
          onChange={(event) => setName(event.target.value)}
          maxLength={80}
          required
          autoComplete="off"
        />
        <label className="field-label" htmlFor="target">
          Send to
        </label>
        <select
          id="target"
          value={selected}
          onChange={(event) => onSelect(event.target.value)}
        >
          <option value="">Any available worker</option>
          {ids.map((id) => (
            <option key={id} value={id}>
              {workerName(id)}
            </option>
          ))}
        </select>
        <div className="field-label" id="duration-label">
          Simulated duration
        </div>
        <div
          className="duration-group"
          role="group"
          aria-labelledby="duration-label"
        >
          {[15, 30, 60].map((seconds) => (
            <button
              key={seconds}
              type="button"
              className={`duration ${duration === seconds ? "active" : ""}`}
              data-seconds={seconds}
              aria-pressed={duration === seconds}
              onClick={() => setDuration(seconds)}
            >
              {seconds} sec
            </button>
          ))}
        </div>
        <label className="toggle-row" htmlFor="failover">
          <span>Retry on another worker</span>
          <input
            type="checkbox"
            id="failover"
            checked={failover}
            onChange={(event) => setFailover(event.target.checked)}
          />
        </label>
        <p className="failover-help">
          If the worker disconnects, unfinished work can move to an available
          one.
        </p>
        <button
          type="submit"
          className="submit"
          id="send-button"
          disabled={busy}
        >
          <span>{busy ? "Sending…" : "Send task"}</span>
          <span aria-hidden="true">↗</span>
        </button>
        <p className="composer-foot">
          Simulated workload · Real connections &amp; leases
        </p>
      </form>
    </section>
  );
}
