import { useEffect, useRef, useState } from "react";
import {
  ApiError,
  AuthLinkError,
  cancelTask,
  logout,
  openSession,
  workloadTask,
  submitTasks,
} from "./api/client";
import type { Task, TaskSpec } from "./api/types";
import { ActivityFeed } from "./components/ActivityFeed";
import { ChatPanel } from "./components/ChatPanel";
import { DeviceInvite } from "./components/DeviceInvite";
import { Login } from "./components/Login";
import { TaskComposer } from "./components/TaskComposer";
import { TaskDetails } from "./components/TaskDetails";
import { TaskList } from "./components/TaskList";
import { WorkerGrid } from "./components/WorkerGrid";
import { useFleet } from "./hooks/useFleet";
import { active, healthy, time, workerName } from "./lib/format";

export default function App() {
  const [session, setSession] = useState<
    "checking" | "login" | "demo" | "public" | "recovery" | "error"
  >("checking");
  const [email, setEmail] = useState("");
  const [authError, setAuthError] = useState("");
  const [revision, setRevision] = useState(0);
  useEffect(() => {
    const controller = new AbortController();
    const expired = () => setSession("login");
    const recovery = () => setSession("recovery");
    window.addEventListener("session-expired", expired);
    window.addEventListener("password-recovery", recovery);
    openSession(controller.signal)
      .then((result) => {
        if (!controller.signal.aborted) {
          setEmail(result.email || "");
          setAuthError("");
          setSession((current) =>
            current === "recovery" ? current : result.mode || "demo",
          );
        }
      })
      .catch((error) => {
        if (!controller.signal.aborted) {
          setAuthError(
            error instanceof AuthLinkError ||
              (error instanceof ApiError && error.status === 403)
              ? error.message
              : "",
          );
          setSession(
            error instanceof AuthLinkError ||
              (error instanceof ApiError && [401, 403].includes(error.status))
              ? "login"
              : "error",
          );
        }
      });
    return () => {
      controller.abort();
      window.removeEventListener("session-expired", expired);
      window.removeEventListener("password-recovery", recovery);
    };
  }, [revision]);
  if (session === "login" || session === "recovery")
    return (
      <Login
        key={session}
        recovery={session === "recovery"}
        initialError={authError}
        onLogin={() => {
          setSession("checking");
          setRevision((value) => value + 1);
        }}
      />
    );
  if (session === "checking")
    return <div className="login-page">Connecting…</div>;
  if (session === "error")
    return (
      <div className="login-page" role="alert">
        Could not connect to the server.
        <button
          onClick={() => {
            setSession("checking");
            setRevision((value) => value + 1);
          }}
        >
          Try again
        </button>
      </div>
    );
  return (
    <FleetApp
      remote={session === "public"}
      email={email}
      onLogout={() => setSession("login")}
    />
  );
}

function FleetApp({
  remote,
  email,
  onLogout,
}: {
  remote: boolean;
  email: string;
  onLogout: () => void;
}) {
  const { snapshot, status, updatedAt } = useFleet();
  const [selected, setSelected] = useState("");
  const [busy, setBusy] = useState(false);
  const [signingOut, setSigningOut] = useState(false);
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
  const fleetSize = snapshot.workers.length;
  const availableWorkers = snapshot.workers.filter(
    (worker) =>
      healthy(worker) &&
      !worker.paused &&
      (worker.capabilities.kinds.includes("stub") ||
        worker.capabilities.kinds.includes("echo")),
  );
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
          One place to manage
          <br />
          distributed work.
        </p>
        <div className="sidebar-bottom">
          <span className="small-dot" />
          {remote ? "Connected fleet" : "Local environment"}
          <small>Independent worker processes</small>
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
            {remote && (
              <button
                className="outline-btn"
                disabled={signingOut}
                onClick={async () => {
                  setSigningOut(true);
                  try {
                    await logout();
                    onLogout();
                  } catch {
                    notify("Could not sign out. Please try again.");
                  } finally {
                    setSigningOut(false);
                  }
                }}
              >
                {signingOut ? "Signing out…" : "Sign out"}
              </button>
            )}
            {!remote && <span className="pill neutral">Demo</span>}
            {remote && email && (
              <span className="account-email" title={email}>
                {email}
              </span>
            )}
            <span
              className="avatar"
              aria-label={remote ? email || "Fleet account" : "Local demo"}
            >
              {remote ? email.slice(0, 2).toUpperCase() || "U" : "LD"}
            </span>
          </div>
        </header>
        <div className="content">
          <div className="heading">
            <div>
              <div className="eyebrow">Fleet dashboard</div>
              <h1>Your fleet. Your call.</h1>
              <p className="subtitle">
                Pick a worker, send a task, and watch the handoff happen.
              </p>
            </div>
            <button
              className="outline-btn"
              id="pair-button"
              disabled={busy || availableWorkers.length === 0}
              onClick={async () => {
                const tasks = await Promise.all(
                  availableWorkers.map((worker) =>
                    workloadTask(
                      worker.capabilities.kinds.includes("echo")
                        ? "echo"
                        : "stub",
                      `Connection test · ${workerName(worker.id)}`,
                      worker.id,
                      30,
                      true,
                    ),
                  ),
                );
                void submit(tasks);
              }}
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
          <DeviceInvite />
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
              <ChatPanel
                key={remote ? email : "demo"}
                scope={remote ? email : "demo"}
              />
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
              <b>Independent workers. One control plane.</b> Python, desktop,
              and iOS workers.
            </span>
            <span id="updated">
              {updatedAt
                ? "Updated " + time(updatedAt)
                : "Connecting to server"}
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
