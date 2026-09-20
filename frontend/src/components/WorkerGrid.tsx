import { useState } from "react";
import { setRuntimePreference } from "../api/client";
import type { RuntimePreference, Task, Worker } from "../api/types";
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

/**
 * Where this machine runs work, and whether an operator may change it.
 *
 * The GPU side is disabled unless the machine has said it has a device. That is the
 * whole point of the control: a switch that can be set to something the machine cannot
 * do is a switch that lies, and the operator finds out only when work starts failing.
 * The machine's own `reason` becomes the tooltip, so "why is this greyed out" is
 * answered in place rather than by reading logs on someone else's laptop.
 *
 * A machine that reported no accelerator at all -- every agent built before the field
 * existed -- is not treated as having no GPU. It is treated as not having said, which is
 * a different sentence and a different fix (pull a newer image).
 */
function RuntimeToggle({ worker }: { worker: Worker }) {
  const accelerator = worker.capabilities.accelerator;
  const [pending, setPending] = useState(false);
  const [error, setError] = useState<string | null>(null);
  // The server echoes the machine's answer into capabilities, so this follows the
  // snapshot rather than local state: an optimistic flip would show a setting the
  // machine may have already refused.
  const current: RuntimePreference =
    worker.capabilities.runtime_preference ?? "auto";
  const silent = accelerator === undefined || accelerator === null;
  const usable = !!accelerator?.available;
  const why = silent
    ? "This machine has not reported its devices. Its agent predates the setting — pull a newer image."
    : accelerator?.reason || "";

  const choose = async (next: RuntimePreference) => {
    if (next === current || pending) return;
    setPending(true);
    setError(null);
    try {
      await setRuntimePreference(worker.id, next);
    } catch (err) {
      setError(err instanceof Error ? err.message : "could not set");
    } finally {
      setPending(false);
    }
  };

  if (worker.capabilities.machine?.runtime_control === "startup") {
    return <div className="runtime-toggle">
      <small className="runtime-why">
        {usable ? accelerator?.device : why}
        {usable && current === "cpu" ? " · CPU mode" : ""}
      </small>
      <small className="runtime-why">Runtime selected when this Python worker starts.</small>
    </div>;
  }

  return (
    <div className="runtime-toggle" onClick={(e) => e.stopPropagation()}>
      <div className="runtime-choices" role="group" aria-label="Where work runs">
        <button
          type="button"
          className={current === "cpu" ? "on" : ""}
          aria-pressed={current === "cpu"}
          disabled={pending}
          onClick={() => choose("cpu")}
          title="Keep all work on the CPU, even if a device is available"
        >
          CPU
        </button>
        <button
          type="button"
          className={current === "auto" ? "on" : ""}
          aria-pressed={current === "auto"}
          disabled={pending || !usable}
          onClick={() => choose("auto")}
          title={
            usable
              ? `Use ${accelerator?.device || "the best device"} when a workload can`
              : why || "No device available on this machine"
          }
        >
          GPU
        </button>
      </div>
      <small className="runtime-why">
        {error
          ? error
          : usable
            ? accelerator?.device ||
              `${worker.capabilities.runtime.toUpperCase()} available`
            : why}
      </small>
    </div>
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
          const serving = task?.spec.kind === "python_service";
          const service = serving ? tasks.find(item => item.spec.kind === "simulation_job" && item.spec.job_id === task.spec.job_id) : undefined;
          const status = ready
            ? task
              ? serving ? "Busy · serving" : "Working"
              : worker.paused ? "Paused" : "Ready"
            : worker?.state === "unhealthy"
              ? "Reconnecting"
              : "Offline";
          return (
            // A div rather than a button: the runtime control below is itself a
            // button, and a button inside a button is invalid HTML that browsers
            // resolve by dropping one of them. role/tabIndex/onKeyDown keep the card
            // reachable and operable from the keyboard exactly as it was.
            <div
              key={id}
              role="button"
              tabIndex={0}
              className={`worker ${selected === id ? "selected" : ""} ${ready ? "" : "waiting"}`}
              data-worker={id}
              aria-pressed={selected === id}
              aria-label={`Send to ${workerName(id)}`}
              onClick={() => onSelect(id)}
              onKeyDown={(e) => {
                if (e.key === "Enter" || e.key === " ") {
                  e.preventDefault();
                  onSelect(id);
                }
              }}
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
                {(() => {
                  const version = worker?.capabilities.machine?.agent_version;
                  if (!version) return null;
                  // A release string carries a content hash after "+". A bare version is
                  // the compile-time stamp, identical in every build, which is what a
                  // machine reports when it has never installed a release.
                  const updated = version.includes("+");
                  return (
                    <span
                      className={`tag ${updated ? "" : "stale"}`}
                      title={
                        updated
                          ? `Running release ${version}`
                          : `Reporting the build stamp ${version} — this machine has never installed a release, so it cannot auto-update`
                      }
                    >
                      {updated ? version : `${version} (not updated)`}
                    </span>
                  );
                })()}
              </div>
              {worker && <RuntimeToggle worker={worker} />}
              <div className="worker-footer">
                <div className="work-status">
                  <span>
                    {service ? taskTitle(service) : task
                      ? taskTitle(task)
                      : ready
                        ? "Ready for your next task"
                        : "Waiting for connection"}
                  </span>
                  <span>{serving ? "Slot reserved" : task ? `${Math.round(task.progress)}%` : ""}</span>
                </div>
                <div className="progress" hidden={!task || serving}>
                  <i style={{ width: `${task?.progress || 0}%` }} />
                </div>
                <div className="heartbeat">
                  {worker
                    ? "heartbeat " + age(worker.last_seen)
                    : "no heartbeat yet"}
                </div>
              </div>
              <span className="selected-mark" />
            </div>
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
