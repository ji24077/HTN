import type { Task, Worker } from "../api/types";
import { active, age, healthy, taskTitle, workerName } from "../lib/format";

function MachineIcon() {
  return (
    <svg
      width="20"
      height="22"
      viewBox="0 0 24 24"
      fill="none"
      stroke="currentColor"
      strokeWidth="1.3"
      aria-hidden="true"
    >
      <rect x="4" y="3" width="16" height="7" rx="2" />
      <rect x="4" y="14" width="16" height="7" rx="2" />
      <path d="M8 6.5h5M8 17.5h5" />
      <circle cx="16.5" cy="6.5" r=".6" />
      <circle cx="16.5" cy="17.5" r=".6" />
    </svg>
  );
}
export function WorkerGrid({
  workers,
  tasks,
  selected,
  onSelect,
}: {
  workers: Worker[];
  tasks: Task[];
  selected: string;
  onSelect: (id: string) => void;
}) {
  const ids = [...workers]
    .sort(
      (a, b) =>
        Number(healthy(b)) - Number(healthy(a)) || a.id.localeCompare(b.id),
    )
    .map((worker) => worker.id);
  return (
    <section>
      <div className="section-heading">
        <h2>
          Registered workers{" "}
          <span className="worker-count">
            / {String(ids.length).padStart(2, "0")}
          </span>
        </h2>
        <small id="fleet-status">Choose a destination</small>
      </div>
      <div className="workers" id="workers">
        {ids.length === 0 && (
          <p className="empty-state">
            No workers connected yet. Workers will appear here when they join
            your fleet.
          </p>
        )}
        {ids.map((id) => {
          const worker = workers.find((worker) => worker.id === id);
          const task = tasks.find(
            (task) => task.worker_id === id && active(task),
          );
          const ready = !!worker && healthy(worker);
          const status = ready
            ? worker.paused
              ? "Paused"
              : task
                ? "Working"
                : "Ready"
            : worker?.state === "unhealthy"
              ? "Reconnecting"
              : "Offline";
          return (
            <button
              key={id}
              type="button"
              className={`worker ${selected === id ? "selected" : ""} ${ready ? "" : "waiting"}`}
              data-worker={id}
              aria-pressed={selected === id}
              aria-label={`Send to ${workerName(id)}`}
              onClick={() => onSelect(id)}
            >
              <div className="worker-top">
                <div className="machine-icon">
                  <MachineIcon />
                </div>
                <span className={`pill ${ready ? "" : "neutral"}`}>
                  <i
                    className="pulse"
                    style={ready ? undefined : { background: "#a0a794" }}
                  />
                  <span className="worker-status">{status}</span>
                </span>
              </div>
              <h3>{workerName(id)}</h3>
              <div className="worker-id">{id}</div>
              <div className="worker-specs">
                <span className="tag">
                  {worker?.capabilities.runtime === "cuda"
                    ? "CUDA"
                    : worker?.capabilities.runtime === "mps"
                      ? "Metal"
                      : "CPU"}
                </span>
                <span className="tag">1 execution slot</span>
              </div>
              <div className="worker-footer">
                <div className="work-status">
                  <span>
                    {task
                      ? taskTitle(task)
                      : ready
                        ? "Ready for your next task"
                        : "Waiting for connection"}
                  </span>
                  <span>{task ? `${Math.round(task.progress)}%` : ""}</span>
                </div>
                <div className="progress" hidden={!task}>
                  <i style={{ width: `${task?.progress || 0}%` }} />
                </div>
                <div className="heartbeat">
                  {worker
                    ? "heartbeat " + age(worker.last_seen)
                    : "no heartbeat yet"}
                </div>
              </div>
              <span className="selected-mark" />
            </button>
          );
        })}
      </div>
      <p className="hint">
        <span>↳</span> Select a worker to direct your next task. Busy workers
        keep it in their queue.
      </p>
    </section>
  );
}
