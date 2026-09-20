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
import { SimulationComposer } from "./components/SimulationComposer";
import { TaskComposer } from "./components/TaskComposer";
import { TaskDetails } from "./components/TaskDetails";
import { TaskList } from "./components/TaskList";
import { WorkerGrid } from "./components/WorkerGrid";
import { useFleet } from "./hooks/useFleet";
import { formatMoney, healthy, time, workerName } from "./lib/format";
import { groupJobs } from "./lib/jobs";
import { Icon } from "./components/Icon";

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
  const [view, setView] = useState<
    "Jobs" | "Workers" | "Activity" | "Assistant"
  >("Jobs");
  const [composeOpen, setComposeOpen] = useState(false);
  const [composeMode, setComposeMode] = useState("upload");
  const composerRef = useRef<HTMLDialogElement>(null);
  useEffect(() => {
    if (composeOpen && !composerRef.current?.open)
      composerRef.current?.showModal();
    else if (!composeOpen && composerRef.current?.open)
      composerRef.current.close();
  }, [composeOpen]);
  const [busy, setBusy] = useState(false);
  const [submitError, setSubmitError] = useState("");
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
  async function submit(tasks: TaskSpec[], usageCap?: string) {
    if (sending.current) return;
    sending.current = true;
    setSubmitError("");
    setBusy(true);
    try {
      const submitted = await submitTasks(tasks, usageCap);
      setComposeOpen(false);
      setView("Jobs");
      if (Array.isArray(submitted) && submitted[0]?.spec)
        setDetail(submitted[0]);
      notify(
        tasks.length === 1
          ? `Queued for ${tasks[0].target_worker_id ? workerName(tasks[0].target_worker_id) : "the next available worker"}.`
          : "One task queued for each worker.",
      );
    } catch (error) {
      setSubmitError(
        "Could not dispatch: " +
          (error instanceof Error ? error.message : "Request failed"),
      );
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
  const availableWorkers = snapshot.workers.filter(
    (worker) =>
      healthy(worker) &&
      !worker.paused &&
      (worker.capabilities.kinds.includes("stub") ||
        worker.capabilities.kinds.includes("echo")),
  );
  const jobs = groupJobs(snapshot.tasks);
  const online = snapshot.workers.filter(healthy).length;
  const metrics = [
    {
      id: "running-count",
      label: "Active jobs",
      value: jobs.filter((j) => ["running", "queued"].includes(j.state)).length,
      note: `${jobs.filter((j) => j.state === "queued").length} queued`,
      icon: "activity" as const,
    },
    {
      id: "complete-count",
      label: "Completed",
      value: jobs.filter((j) => j.state === "succeeded").length,
      note: "All tasks succeeded",
      icon: "check" as const,
    },
    {
      id: "failed-count",
      label: "Failed",
      value: jobs.filter((j) => j.state === "failed").length,
      note: "Open a job to investigate",
      icon: "warning" as const,
    },
    {
      id: "online-count",
      label: "Workers online",
      value: online,
      note: `${snapshot.workers.length} registered`,
      icon: "workers" as const,
    },
  ];
  const subtitles = {
    Jobs: "Track progress, investigate failures, and review results.",
    Workers: "Manage the machines that run your jobs.",
    Activity: "A live record of assignments, retries, and fleet changes.",
    Assistant: "Inspect your fleet and dispatch supported workloads.",
  };
  return (
    <>
      <aside className="app-sidebar">
        <div className="brand">
          <span className="brandmark">
            <Icon name="arrow" size={19} />
          </span>
          dispatch<span className="brand-dot">.</span>
        </div>
        <div className="workspace-label">
          <span className="workspace-avatar">C</span>
          <div>
            Compute workspace<small>Distributed execution</small>
          </div>
        </div>
        <div className="section-label">WORKSPACE</div>
        <nav className="main-nav" aria-label="Main navigation">
          {(
            [
              ["Jobs", "jobs"],
              ["Workers", "workers"],
              ["Activity", "activity"],
              ["Assistant", "assistant"],
            ] as const
          ).map(([label, icon]) => (
            <button
              key={label}
              aria-current={view === label ? "page" : undefined}
              onClick={() => setView(label)}
            >
              <Icon name={icon} />
              <span>{label}</span>
              {label === "Jobs" && <small>{jobs.length}</small>}
              {label === "Workers" && <small>{online}</small>}
            </button>
          ))}
        </nav>
        <div className="sidebar-bottom">
          <span className={`connection-dot ${status}`} />
          {remote ? "Connected fleet" : "Local environment"}
          <small>Changes update automatically</small>
        </div>
      </aside>
      <main className="app-main">
        <header className="topbar">
          <div className="crumb">
            <span>Workspace</span>
            <span className="crumb-divider">/</span>
            <strong>{view}</strong>
          </div>
          <div className="top-right">
            <div
              className="header-spend"
              role="group"
              aria-label={
                snapshot.account ? "Account balance" : "Total estimated spend"
              }
              title={
                snapshot.account
                  ? "Your remaining CAD credit after estimated usage. Open a run to see its cost."
                  : "Total recorded worker cost across all runs. Open a run for its usage and cap."
              }
            >
              <strong>
                {snapshot.account
                  ? formatMoney(snapshot.account.balance)
                  : snapshot.usage
                    ? formatMoney(snapshot.usage.cost)
                    : "—"}
              </strong>
              {snapshot.account && (
                <span className="balance-label">credit</span>
              )}
            </div>
            <span
              className={`connection-label ${status}`}
              role="status"
              aria-label={
                status === "live"
                  ? "Live updates"
                  : status === "connecting"
                    ? "Connecting"
                    : "Reconnecting"
              }
            >
              <i className={`connection-dot ${status}`} aria-hidden="true" />
              <span className="connection-text" aria-hidden="true">
                {status === "live"
                  ? "Live updates"
                  : status === "connecting"
                    ? "Connecting"
                    : "Reconnecting"}
              </span>
            </span>
            {remote && email && (
              <span className="account-email" title={email}>
                {email}
              </span>
            )}
            {remote && (
              <button
                className="text-btn"
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
              <div className="eyebrow">COMPUTE WORKSPACE</div>
              <h1>{view}</h1>
              <p className="subtitle">{subtitles[view]}</p>
            </div>
            <button
              className="primary-btn"
              onClick={() => setComposeOpen(true)}
            >
              <Icon name="plus" size={17} />
              New job
            </button>
          </div>
          <div
            id="connection-error"
            hidden={status !== "reconnecting"}
            role="alert"
          >
            Live connection interrupted — reconnecting automatically. Displayed
            data may be out of date.
          </div>
          <div hidden={view !== "Jobs"}>
            <section className="metrics" aria-label="Fleet overview">
              {metrics.map((metric) => (
                <div className="metric" key={metric.id}>
                  <div className="metric-label">
                    {metric.label}
                    <Icon name={metric.icon} size={16} />
                  </div>
                  <div className="metric-value" id={metric.id}>
                    {updatedAt ? metric.value : "—"}
                  </div>
                  <small>{metric.note}</small>
                </div>
              ))}
            </section>
            <TaskList
              tasks={snapshot.tasks}
              cancelling={cancelling}
              onCancel={(id) => void cancel(id)}
              onDetail={setDetail}
            />
            <div className="tracking-note">
              <Icon name="assistant" size={16} />
              <span>
                Open a job to follow its attempts, search execution logs, and
                see the supervisor’s decisions.
              </span>
            </div>
          </div>
          <div hidden={view !== "Workers"}>
            <div className="section-actions">
              <p className="muted">
                {online} online · {snapshot.workers.length - online} offline
              </p>
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
                Run one on each
              </button>
            </div>
            <WorkerGrid
              workers={snapshot.workers}
              tasks={snapshot.tasks}
              selected={selected}
              onSelect={(id) => {
                setSelected(id);
                setComposeMode("builtin");
                setComposeOpen(true);
              }}
            />
            <DeviceInvite />
          </div>
          <div hidden={view !== "Activity"}>
            <ActivityFeed
              events={snapshot.events}
              tasks={snapshot.tasks}
              status={status}
            />
          </div>
          <div hidden={view !== "Assistant"} className="assistant-page">
            <ChatPanel
              key={remote ? email : "demo"}
              scope={remote ? email : "demo"}
            />
          </div>
          <footer className="footer">
            <span>
              dispatch <span className="muted">/ Distributed compute</span>
            </span>
            <span id="updated">
              {updatedAt
                ? "Last update " + time(updatedAt)
                : "Connecting to server"}
            </span>
          </footer>
        </div>
      </main>
      <dialog
        ref={composerRef}
        className="compose-dialog"
        aria-labelledby="compose-title"
        onClose={() => setComposeOpen(false)}
      >
        <header>
          <div>
            <div className="eyebrow">DISPATCH</div>
            <h2 id="compose-title">New job</h2>
          </div>
          <button
            className="icon-btn"
            aria-label="Close new job"
            onClick={() => setComposeOpen(false)}
          >
            <Icon name="close" />
          </button>
        </header>
        {submitError && (
          <p role="alert" className="inline-alert compose-error">
            {submitError}
          </p>
        )}
        <div className="compose-mode" role="group" aria-label="Submission type">
          <button
            type="button"
            aria-pressed={composeMode === "upload"}
            onClick={() => setComposeMode("upload")}
          >
            Upload simulation
          </button>
          <button
            type="button"
            aria-pressed={composeMode === "builtin"}
            onClick={() => setComposeMode("builtin")}
          >
            Built-in tasks
          </button>
        </div>
        <div hidden={composeMode !== "upload"}>
          <SimulationComposer
            onCreated={(task) => {
              setComposeOpen(false);
              setView("Jobs");
              setDetail(task);
              notify(
                "Simulation submitted. Preprocessing will start on one worker.",
              );
            }}
          />
        </div>
        <div hidden={composeMode !== "builtin"}>
          <TaskComposer
            selected={selected}
            onSelect={setSelected}
            workers={snapshot.workers}
            busy={busy}
            onSubmit={submit}
          />
        </div>
      </dialog>
      <div
        id="toast"
        role="status"
        aria-label="Notification"
        aria-live="polite"
        hidden={!toast}
      >
        {toast?.message}
      </div>
      <TaskDetails
        task={
          detail
            ? snapshot.tasks.find((task) => task.spec.id === detail.spec.id) ||
              detail
            : null
        }
        tasks={
          detail
            ? snapshot.tasks.filter(
                (task) => task.spec.job_id === detail.spec.job_id,
              )
            : []
        }
        onSelectTask={setDetail}
        onClose={() => setDetail(null)}
      />
    </>
  );
}
