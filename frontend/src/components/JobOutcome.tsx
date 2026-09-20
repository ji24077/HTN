import type { Task } from "../api/types";
import type { SimulationStatus } from "../api/client";
import { record, taskTitle } from "../lib/format";
import { statusLabels } from "../lib/jobs";
import { Icon } from "./Icon";

export function parsedResult(value: unknown): Record<string, unknown> {
  if (typeof value === "string") {
    try {
      return record(JSON.parse(value));
    } catch {
      return { output: value };
    }
  }
  return record(value);
}

export function JobOutcome({
  task,
  status,
}: {
  task: Task;
  status?: SimulationStatus | null;
}) {
  const result = parsedResult(task.result);
  const output = record(result.output);
  // Only outcome metrics, never probe timings or arbitrary nested numbers.
  const metrics = Object.entries(
    record(output.metrics ?? result.metrics),
  ).filter(
    (entry): entry is [string, number] =>
      typeof entry[1] === "number" && Number.isFinite(entry[1]),
  );
  const aggregate = Object.entries(output)
    .filter(
      (entry): entry is [string, number] =>
        typeof entry[1] === "number" && Number.isFinite(entry[1]),
    )
    .filter(([name]) => name !== "output_count");
  const values = metrics.length ? metrics : aggregate;
  const completed = task.state === "succeeded";
  const actualWorkers = [
    ...new Set(
      status?.tasks
        .filter(
          (t) =>
            t.state === "succeeded" && /program_run|batch|execute/.test(t.role),
        )
        .map((t) => t.worker_id)
        .filter(Boolean) ?? [],
    ),
  ];
  const workers = actualWorkers.length
    ? actualWorkers
    : status?.workers.length
      ? status.workers
      : task.worker_id
        ? [task.worker_id]
        : [];
  const runtime = status?.program_plan?.requirements.runtime;
  const runtimeLabel =
    runtime === "mps"
      ? "Apple Metal (MPS)"
      : runtime === "cuda"
        ? "NVIDIA CUDA"
        : runtime === "cpu"
          ? "CPU"
          : "";
  const summary = completed
    ? status?.message ||
      "Execution completed. Review the recorded result and available files below."
    : task.state === "failed"
      ? task.failure ||
        status?.message ||
        "Execution failed. Open Activity for the recorded error."
      : task.state === "cancelled"
        ? "This job was cancelled. Any saved outputs remain available in Files."
        : status?.message ||
          "The agent is preparing your request. Progress and results will appear here.";
  const number = (value: number) =>
    value.toLocaleString(undefined, { maximumSignificantDigits: 6 });
  function downloadReport() {
    const lines = [
      `# ${taskTitle(task)}`,
      "",
      "ChatGPU · Recorded result summary",
      "",
      `Status: ${statusLabels[task.state]}`,
      "",
      summary,
      "",
      ...(workers.length ? [`Workers: ${workers.join(", ")}`, ""] : []),
      ...(runtimeLabel ? [`Planned runtime: ${runtimeLabel}`, ""] : []),
      ...(values.length
        ? [
            "## Results",
            "",
            ...values.map(([name, value]) => {
              const bound = status?.program_plan?.metrics.find(
                (m) => m.name === name,
              );
              return `- ${name}: ${value}${bound?.maximum != null ? ` (required ≤ ${bound.maximum})` : ""}${bound?.minimum != null ? ` (required ≥ ${bound.minimum})` : ""}`;
            }),
            "",
          ]
        : []),
      ...(status?.checks.length
        ? [
            "## Validation",
            "",
            ...status.checks.map(
              (c) =>
                `- ${c.stage}: ${c.passed ? "Passed" : "Failed"} (round ${c.round})`,
            ),
            "",
          ]
        : []),
      "## About this summary",
      "",
      "Generated from this job's recorded results. No new execution or analysis was performed. Source artifacts are available in ChatGPU's Files view.",
      "",
      `Job: ${task.spec.job_id}`,
      "",
    ];
    const url = URL.createObjectURL(
      new Blob([lines.join("\n")], { type: "text/markdown;charset=utf-8" }),
    );
    const anchor = document.createElement("a");
    anchor.href = url;
    anchor.download = `${task.spec.job_id}-summary.md`;
    anchor.click();
    setTimeout(() => URL.revokeObjectURL(url), 1000);
  }
  return (
    <section className={`outcome-card ${task.state}`} aria-label="Job outcome">
      <div className="outcome-heading">
        <div>
          <span className="eyebrow">
            {completed ? "YOUR RESULTS" : "CURRENT STATUS"}
          </span>
          <h3>
            {completed
              ? "Your run is complete"
              : task.state === "failed"
                ? "This run couldn’t finish"
                : task.state === "cancelled"
                  ? "Run cancelled"
                  : "Work in progress"}
          </h3>
        </div>
        {completed && task.result != null && (
          <button className="outline-btn" onClick={downloadReport}>
            <Icon name="download" size={16} />
            Download summary
          </button>
        )}
      </div>
      <p className="outcome-summary">{summary}</p>
      {workers.length > 0 && (
        <p className="outcome-environment">
          {workers.join(", ")}
          {runtimeLabel && ` · Planned runtime: ${runtimeLabel}`}
        </p>
      )}
      {!!values.length && (
        <div className="result-metrics">
          {values.map(([name, value]) => {
            const bound = status?.program_plan?.metrics.find(
              (m) => m.name === name,
            );
            const hasBound =
              bound && (bound.minimum != null || bound.maximum != null);
            const passed =
              hasBound &&
              (bound.minimum == null || value >= bound.minimum) &&
              (bound.maximum == null || value <= bound.maximum);
            const ratio =
              bound?.maximum != null && bound.maximum > 0 && value >= 0
                ? value / bound.maximum
                : null;
            return (
              <div className="result-metric" key={name}>
                <span>{name.replaceAll("_", " ")}</span>
                <strong>
                  {name === "hit_rate" && value >= 0 && value <= 1
                    ? `${number(value * 100)}%`
                    : number(value)}
                </strong>
                {hasBound && (
                  <small className={passed ? "metric-pass" : "metric-fail"}>
                    {passed
                      ? "Within required bounds"
                      : "Outside required bounds"}
                    {bound.minimum != null && ` · ≥ ${bound.minimum}`}
                    {bound.maximum != null && ` · ≤ ${bound.maximum}`}
                  </small>
                )}
                {name === "hit_rate" && value >= 0 && value <= 1 && (
                  <div
                    className="metric-scale"
                    role="img"
                    aria-label={`Observed hit rate: ${number(value * 100)}%`}
                  >
                    <i style={{ width: `${value * 100}%` }} />
                  </div>
                )}
                {ratio !== null && (
                  <div
                    className="metric-scale"
                    role="img"
                    aria-label={`${name}: ${number(value)}, maximum ${bound?.maximum}`}
                  >
                    <i
                      style={{
                        width: `${Math.min(100, Math.max(1, ratio * 100))}%`,
                      }}
                    />
                  </div>
                )}
              </div>
            );
          })}
        </div>
      )}
      {typeof result.output === "string" && (
        <p className="outcome-text">{result.output}</p>
      )}
      {completed && (
        <p className="outcome-note">
          Based on saved results. No new computation performed.
        </p>
      )}
    </section>
  );
}
