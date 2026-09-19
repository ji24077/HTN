import type { Task } from "../api/types";
import { active, age, record, taskTitle, workerName } from "../lib/format";
const statuses = {
  assigned: "Starting",
  queued: "Queued",
  running: "Running",
  succeeded: "Complete",
  failed: "Failed",
  cancelled: "Cancelled",
};
export function TaskList({
  tasks,
  cancelling,
  onCancel,
  onDetail,
}: {
  tasks: Task[];
  cancelling: Set<string>;
  onCancel: (id: string) => void;
  onDetail: (task: Task) => void;
}) {
  const ordered = [...tasks].sort(
    (a, b) =>
      Number(active(b) || b.state === "queued") -
        Number(active(a) || a.state === "queued") ||
      Date.parse(b.created_at) - Date.parse(a.created_at),
  );
  return (
    <section className="queue-section">
      <div className="section-heading">
        <h2>Task stream</h2>
        <small id="task-count">
          {tasks.length
            ? `${tasks.length} task${tasks.length === 1 ? "" : "s"} · most recent 12`
            : "Your next task starts here"}
        </small>
      </div>
      <div className="task-list" id="task-list">
        {ordered.slice(0, 12).map((task) => (
          <article
            className="task-row"
            key={task.spec.id}
            data-task={task.spec.id}
          >
            <div className="task-icon">
              {task.state === "succeeded"
                ? "✓"
                : active(task)
                  ? "↗"
                  : task.state === "queued"
                    ? "≡"
                    : "−"}
            </div>
            <div>
              <div className="task-name">{taskTitle(task)}</div>
              <div className="task-meta">
                <b>
                  {workerName(task.worker_id || task.spec.target_worker_id)}
                </b>{" "}
                · {String(record(task.spec.payload).duration_seconds ?? "—")}s
                {task.generation > 1 ? ` · retry ${task.generation - 1}` : ""} ·{" "}
                {age(task.created_at)}
              </div>
              <div className="task-progress" hidden={!active(task)}>
                <div className="progress">
                  <i style={{ width: `${task.progress}%` }} />
                </div>
              </div>
            </div>
            <div className="task-actions">
              <span className={`badge ${task.state}`}>
                {statuses[task.state]}
                {task.state === "running"
                  ? ` ${Math.round(task.progress)}%`
                  : ""}
              </span>
              <button
                className="icon-btn"
                data-detail={task.spec.id}
                aria-label={`View ${taskTitle(task)} details`}
                onClick={() => onDetail(task)}
              >
                ↗
              </button>
              <button
                className="icon-btn"
                data-cancel={task.spec.id}
                aria-label={`Cancel ${taskTitle(task)}`}
                disabled={cancelling.has(task.spec.id)}
                hidden={!active(task) && task.state !== "queued"}
                onClick={() => onCancel(task.spec.id)}
              >
                ×
              </button>
            </div>
          </article>
        ))}
        {!tasks.length && (
          <div className="empty">
            <div className="empty-icon">↗</div>No tasks yet.
            <br />
            Choose a worker and send your first piece of work.
          </div>
        )}
      </div>
    </section>
  );
}
