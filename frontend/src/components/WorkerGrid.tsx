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

/** Offer a policy choice only when the worker reports a usable accelerator. */
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

  if (silent || !usable) {
    return (
      <p className="runtime-summary">
        {silent
          ? "GPU availability has not been reported."
          : "GPU execution is unavailable on this worker."}
      </p>
    );
  }

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
    return (
      <div className="runtime-toggle">
        <small className="runtime-why">
          {accelerator?.device || "GPU available"}
          {current === "cpu" ? " · CPU mode" : ""}
        </small>
        <small className="runtime-why">
          {healthy(worker)
            ? "Runtime set at startup."
            : "Last reported runtime · worker offline."}
        </small>
      </div>
    );
  }

  return (
    <div className="runtime-toggle" onClick={(e) => e.stopPropagation()}>
      <span className="runtime-label">Scheduling policy</span>
      <div
        className="runtime-choices"
        role="group"
        aria-label="Scheduling policy"
      >
        <button
          type="button"
          className={current === "cpu" ? "on" : ""}
          aria-pressed={current === "cpu"}
          disabled={pending || !healthy(worker)}
          onClick={() => choose("cpu")}
          title="Keep all work on the CPU, even if a device is available"
        >
          CPU only
        </button>
        <button
          type="button"
          className={current === "auto" ? "on" : ""}
          aria-pressed={current === "auto"}
          disabled={pending || !healthy(worker)}
          onClick={() => choose("auto")}
          title={`Let the agent use ${accelerator?.device || "the GPU"} when the workload supports it`}
        >
          Automatic
        </button>
      </div>
      <small className="runtime-why">
        {accelerator?.device || "GPU available"}
      </small>
      {!healthy(worker) && (
        <small className="runtime-why">Reconnect to change this setting.</small>
      )}
      {pending && <small role="status">Updating policy…</small>}
      {error && (
        <small className="runtime-error" role="alert">
          {error}
        </small>
      )}
    </div>
  );
}

export function WorkerGrid({
  workers,
  loading = false,
  tasks,
  selected,
  onSelect,
}: {
  workers: Worker[];
  loading?: boolean;
  tasks: Task[];
  selected: string;
  onSelect: (id: string) => void;
}) {
  const [showOffline, setShowOffline] = useState(false);
  if (loading)
    return (
      <p className="loading-panel" role="status">
        Loading workers…
      </p>
    );
  const ids = [...workers]
    .filter((worker) => showOffline || healthy(worker))
    .sort(
      (a, b) =>
        Number(healthy(b)) - Number(healthy(a)) ||
        (a.name?.trim() || workerName(a.id)).localeCompare(
          b.name?.trim() || workerName(b.id),
          undefined,
          { sensitivity: "base" },
        ) ||
        a.id.localeCompare(b.id),
    )
    .map((worker) => worker.id);
  return (
    <section>
      <div className="section-heading">
        <h2>
          {showOffline ? "Registered workers" : "Available workers"}{" "}
          <span className="worker-count">
            / {String(ids.length).padStart(2, "0")}
          </span>
        </h2>
        <small id="fleet-status">The agent selects compatible machines</small>
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
          const machine = worker?.capabilities.machine;
          const name = worker?.name?.trim() || workerName(id);
          const serving = task?.spec.kind === "python_service";
          const service = serving
            ? tasks.find(
                (item) =>
                  item.spec.kind === "simulation_job" &&
                  item.spec.job_id === task.spec.job_id,
              )
            : undefined;
          const status = ready
            ? task
              ? serving
                ? "Busy · serving"
                : "Working"
              : worker.paused
                ? "Paused"
                : "Ready"
            : worker?.state === "unhealthy"
              ? "Reconnecting"
              : "Offline";
          return (
            <article
              key={id}
              className={`worker ${selected === id ? "selected" : ""} ${ready ? "" : "waiting"}`}
              data-worker={id}
              aria-label={name}
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
              <h3>{name}</h3>
              {machine?.os && (
                <p className="worker-platform">
                  {machine.os}
                  {machine.arch ? ` · ${machine.arch}` : ""}
                </p>
              )}
              <div className="worker-specs">
                <span className="tag">
                  {worker?.capabilities.runtime === "cuda"
                    ? "CUDA"
                    : worker?.capabilities.runtime === "mps"
                      ? "Metal"
                      : "CPU"}
                </span>
                {!!machine?.logical_cores && (
                  <span className="tag">{machine.logical_cores} cores</span>
                )}
                {!!machine?.total_ram_mb && (
                  <span className="tag">
                    {Math.round(machine.total_ram_mb / 1024)} GiB RAM
                  </span>
                )}
                {worker?.capabilities.python && (
                  <span
                    className="tag"
                    title={`Python ${worker.capabilities.python.version}`}
                  >
                    PyTorch {worker.capabilities.python.pytorch}
                  </span>
                )}
              </div>
              {worker && <RuntimeToggle worker={worker} />}
              <div className="worker-footer">
                <div className="work-status">
                  <span>
                    {service
                      ? taskTitle(service)
                      : task
                        ? taskTitle(task)
                        : ready
                          ? worker?.paused
                            ? "Scheduling paused"
                            : "No active job"
                          : "Waiting for connection"}
                  </span>
                  <span>
                    {serving
                      ? "Slot reserved"
                      : task
                        ? `${Math.round(task.progress)}%`
                        : ""}
                  </span>
                </div>
                <div className="progress" hidden={!task || serving}>
                  <i style={{ width: `${task?.progress || 0}%` }} />
                </div>
                <div className="heartbeat">
                  {worker
                    ? "Last seen " + age(worker.last_seen)
                    : "no heartbeat yet"}
                </div>
              </div>
              <button
                type="button"
                className="outline-btn worker-inspect"
                aria-label={
                  selected === id
                    ? `Hide details for ${name}`
                    : `View details for ${name}`
                }
                aria-expanded={selected === id}
                aria-controls={`worker-details-${id}`}
                onClick={() => onSelect(selected === id ? "" : id)}
              >
                {selected === id ? "Hide details" : "View details"}
                <span aria-hidden="true">{selected === id ? "−" : "+"}</span>
              </button>
              <div
                id={`worker-details-${id}`}
                hidden={selected !== id}
                className="machine-details"
              >
                {selected === id && worker && (
                  <dl>
                    <dt>Worker ID</dt>
                    <dd>{id}</dd>
                    <dt>Processor</dt>
                    <dd>{machine?.cpu_model || "Not reported"}</dd>
                    <dt>Supported workloads</dt>
                    <dd>{worker.capabilities.kinds.join(", ")}</dd>
                    {machine?.agent_version && (
                      <>
                        <dt>Agent build</dt>
                        <dd>{machine.agent_version}</dd>
                      </>
                    )}
                    {worker.capabilities.accelerator?.reason && (
                      <>
                        <dt>GPU diagnostics</dt>
                        <dd>{worker.capabilities.accelerator.reason}</dd>
                      </>
                    )}
                  </dl>
                )}
              </div>
              <span className="selected-mark" />
            </article>
          );
        })}
      </div>
      {workers.some((worker) => !healthy(worker)) && (
        <button
          className="text-btn offline-toggle"
          aria-expanded={showOffline}
          onClick={() => setShowOffline((value) => !value)}
        >
          {showOffline
            ? "Hide offline workers"
            : `Show offline workers (${workers.filter((worker) => !healthy(worker)).length})`}
        </button>
      )}
      <p className="hint">
        Hardware and runtime reflect the worker's last report. Automatic policy
        allows compatible GPU workloads; it does not install GPU software.
      </p>
    </section>
  );
}
