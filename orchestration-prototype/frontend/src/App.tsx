import { useEffect, useRef, useState } from "react";
import { cancelTask, stubTask, submitTasks } from "./api/client";
import type { Task, TaskSpec } from "./api/types";
import { ActivityFeed } from "./components/ActivityFeed";
import { TaskComposer } from "./components/TaskComposer";
import { TaskDetails } from "./components/TaskDetails";
import { TaskList } from "./components/TaskList";
import { WorkerGrid } from "./components/WorkerGrid";
import { useFleet } from "./hooks/useFleet";
import { active, healthy, time, workerName } from "./lib/format";

export default function App() {
  const { snapshot, status, updatedAt } = useFleet();
  const [selected, setSelected] = useState("");
  const [busy, setBusy] = useState(false);
  const sending = useRef(false);
  const [cancelling, setCancelling] = useState(new Set<string>());
  const pendingCancellations = useRef(new Set<string>());
  const [toast, setToast] = useState<{ message: string } | null>(null);
  const [detail, setDetail] = useState<Task | null>(null);
  useEffect(() => {
    if (!toast) return;
    const timer = setTimeout(() => setToast(null), 4500);
    return () => clearTimeout(timer);
  }, [toast]);
  const notify = (message: string) => setToast({ message });
  async function submit(tasks: TaskSpec[]) {
    if (sending.current) return;
    sending.current = true;
    setBusy(true);
    try {
      await submitTasks(tasks);
      notify(
        tasks.length === 1
          ? `Queued for ${tasks[0].target_worker_id ? workerName(tasks[0].target_worker_id) : "the next available worker"}.`
          : "One task queued for each worker.",
      );
    } catch (error) {
      notify(
        "Could not dispatch: " +
          (error instanceof Error ? error.message : "Request failed"),
      );
    } finally {
      sending.current = false;
      setBusy(false);
    }
  }
  async function cancel(id: string) {
    if (pendingCancellations.current.has(id)) return;
    pendingCancellations.current.add(id);
    setCancelling(new Set(pendingCancellations.current));
    try {
      await cancelTask(id);
      notify("Task cancelled. Worker will stop on its next heartbeat.");
    } catch (error) {
      notify(error instanceof Error ? error.message : "Could not cancel task");
    } finally {
      pendingCancellations.current.delete(id);
      setCancelling(new Set(pendingCancellations.current));
    }
  }
  const fleetSize = new Set([
    "worker-a",
    "worker-b",
    ...snapshot.workers.map((worker) => worker.id),
  ]).size;
  const metrics = [
    {
      id: "online-count",
      label: "Workers online",
      icon: "◉",
      value: snapshot.workers.filter(healthy).length,
      note: `of ${fleetSize} services`,
    },
    {
      id: "running-count",
      label: "In progress",
      icon: "↗",
      value: snapshot.tasks.filter(active).length,
      note: "tasks",
    },
    {
      id: "queued-count",
      label: "In the queue",
      icon: "≡",
      value: snapshot.tasks.filter((task) => task.state === "queued").length,
      note: "waiting",
    },
    {
      id: "complete-count",
      label: "Completed",
      icon: "✓",
      value: snapshot.tasks.filter((task) => task.state === "succeeded").length,
      note: "results accepted",
    },
  ];
  return (
    <>
      <aside>
        <div className="brand">
          <span className="brandmark">↗</span> dispatch
          <span className="brand-dot">.</span>
        </div>
        <div className="section-label">WORKSPACE</div>
        <div className="nav">
          <span>▦</span> Compute lab <small>01</small>
        </div>
        <p className="aside-note">
          A little playground for
          <br />
          distributed work.
        </p>
        <div className="sidebar-bottom">
          <span className="small-dot" />
          Local environment<small>Independent worker processes</small>
        </div>
      </aside>
      <main>
        <header className="topbar">
          <div className="crumb">
            <span>Workspace</span>
            <span className="crumb-divider">/</span>
            <strong>Compute lab</strong>
          </div>
          <div className="top-right">
            <span className="pill neutral">Prototype</span>
            <span className="avatar">ET</span>
          </div>
        </header>
        <div className="content">
          <div className="heading">
            <div>
              <div className="eyebrow">Orchestration playground</div>
              <h1>Your fleet. Your call.</h1>
              <p className="subtitle">
                Pick a worker, send a task, and watch the handoff happen.
              </p>
            </div>
            <button
              className="outline-btn"
              id="pair-button"
              disabled={busy}
              onClick={() =>
                void submit([
                  stubTask("Render preview · scene A", "worker-a", 30, true),
                  stubTask("Render preview · scene B", "worker-b", 45, true),
                ])
              }
            >
              <span aria-hidden="true">⇉</span> Run one on each
            </button>
          </div>
          <div
            id="connection-error"
            hidden={status !== "reconnecting"}
            role="alert"
          >
            Live connection interrupted — reconnecting automatically.
          </div>
          <section className="metrics" aria-label="Fleet overview">
            {metrics.map((metric) => (
              <div className="metric" key={metric.id}>
                <div className="metric-label">
                  {metric.label}
                  <span>{metric.icon}</span>
                </div>
                <div className="metric-value">
                  <span id={metric.id}>{updatedAt ? metric.value : "—"}</span>
                  <small>{metric.note}</small>
                </div>
              </div>
            ))}
          </section>
          <div className="layout">
            <div className="left-column">
              <WorkerGrid
                workers={snapshot.workers}
                tasks={snapshot.tasks}
                selected={selected}
                onSelect={setSelected}
              />
              <TaskList
                tasks={snapshot.tasks}
                cancelling={cancelling}
                onCancel={(id) => void cancel(id)}
                onDetail={setDetail}
              />
            </div>
            <div className="right-column">
              <TaskComposer
                selected={selected}
                onSelect={setSelected}
                workers={snapshot.workers}
                busy={busy}
                onSubmit={submit}
              />
              <ActivityFeed
                events={snapshot.events}
                tasks={snapshot.tasks}
                status={status}
              />
            </div>
          </div>
          <footer className="footer">
            <span>
              <b>Independent workers. One control plane.</b> No GPU or cloud
              resources in use.
            </span>
            <span id="updated">
              {updatedAt
                ? "Updated " + time(updatedAt)
                : "Connecting to localhost"}
            </span>
          </footer>
        </div>
      </main>
      <div id="toast" role="status" aria-live="polite" hidden={!toast}>
        {toast?.message}
      </div>
      <TaskDetails
        task={
          detail
            ? snapshot.tasks.find((task) => task.spec.id === detail.spec.id) ||
              detail
            : null
        }
        onClose={() => setDetail(null)}
      />
    </>
  );
}
