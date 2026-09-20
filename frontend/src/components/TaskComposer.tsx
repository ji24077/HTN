import { useState, type SubmitEvent } from "react";
import type { TaskSpec, Worker } from "../api/types";
import { workloadTask, type WorkloadKind } from "../api/client";
import { MaxSpendField, parseMaxSpend } from "./MaxSpendField";
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
  onSubmit: (tasks: TaskSpec[], usageCap?: string) => Promise<void>;
}) {
  const [maxSpend, setMaxSpend] = useState("");
  const [name, setName] = useState("Connection test");
  const [duration, setDuration] = useState(30);
  const [failover, setFailover] = useState(true);
  const [fail, setFail] = useState(false);
  const [kind, setKind] = useState<WorkloadKind>("stub");
  const [preparing, setPreparing] = useState(false);
  const [error, setError] = useState("");
  const ids = [
    ...new Set([
      ...workers
        .filter((worker) => worker.capabilities.kinds.includes(kind))
        .map((worker) => worker.id),
      ...(selected ? [selected] : []),
    ]),
  ];
  async function submit(event: SubmitEvent<HTMLFormElement>) {
    event.preventDefault();
    if (!name.trim() || preparing || busy) return;
    setPreparing(true);
    setError("");
    try {
      const usageCap = parseMaxSpend(maxSpend);
      const worker = workers.find((worker) => worker.id === selected);
      if (worker && !worker.capabilities.kinds.includes(kind))
        throw new Error("Choose a worker that supports this workload.");
      const task = await workloadTask(
        kind,
        name.trim(),
        selected,
        duration,
        failover,
      );
      if (kind === "stub" && fail)
        task.payload = {
          duration_seconds: 1,
          fail: true,
          value: { label: name.trim() },
        };
      await onSubmit([task], usageCap);
    } catch (cause) {
      setError(
        cause instanceof Error ? cause.message : "Could not prepare the task.",
      );
    } finally {
      setPreparing(false);
    }
  }
  return (
    <section className="composer">
      <p>
        Choose a workload and where it should run. Track its progress from the
        jobs page.
      </p>
      <form id="task-form" onSubmit={submit}>
        <label className="field-label" htmlFor="workload">
          Workload
        </label>
        <select
          id="workload"
          value={kind}
          onChange={(event) => {
            const next = event.target.value as WorkloadKind;
            setKind(next);
            setError("");
            if (
              selected &&
              !workers
                .find((worker) => worker.id === selected)
                ?.capabilities.kinds.includes(next)
            )
              onSelect("");
          }}
        >
          <option value="stub">Connection test (Python worker)</option>
          <option value="echo">Signed connection test</option>
          <option value="walker_evolution">
            Walker evolution · 8 candidates
          </option>
          <option value="cpu_inference_batch">
            MNIST inference · 100 digits
          </option>
        </select>
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
        {kind === "stub" && (
          <>
            <label className="toggle-row" htmlFor="failure-test">
              <span>Intentional failure (supervisor test)</span>
              <input
                id="failure-test"
                type="checkbox"
                checked={fail}
                onChange={(event) => setFail(event.target.checked)}
              />
            </label>
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
          </>
        )}
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
        <MaxSpendField
          value={maxSpend}
          onChange={setMaxSpend}
          disabled={busy || preparing}
        />
        <button
          type="submit"
          className="submit"
          id="send-button"
          disabled={busy || preparing}
        >
          <span>{busy || preparing ? "Sending…" : "Send task"}</span>
          <span aria-hidden="true">↗</span>
        </button>
        <p className="composer-foot">
          {kind === "stub"
            ? "Simulated workload · Real connections & leases"
            : "Runs on paired devices · Signed results"}
        </p>
        {error && <p role="alert">{error}</p>}
      </form>
    </section>
  );
}
