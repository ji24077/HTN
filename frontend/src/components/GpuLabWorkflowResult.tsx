import type { LabJob } from "../api/gpulab";
import type { GpuWorkflowPlan } from "../lib/gpuWorkflow";
import { JOB_LABEL, verification } from "../lib/gpulab";

export type WorkflowAction = Extract<GpuWorkflowPlan, { kind: "action" }>;
export type ActionState =
  "pending" | "submitting" | "running" | "done" | "dismissed" | "failed";

export function GpuLabWorkflowResult({
  action,
  state,
  job,
  disabled,
  onApprove,
  onDismiss,
}: {
  action: WorkflowAction;
  state: ActionState;
  job?: LabJob;
  disabled: boolean;
  onApprove: () => void;
  onDismiss: () => void;
}) {
  const active =
    job && ["queued", "running", "cancelling"].includes(job.status);
  const verdict = job && !active ? verification(job) : null;
  return (
    <div className="lab-workflow-result">
      <details>
        <summary>Review {action.title.toLowerCase()}</summary>
        {action.warnings?.map((warning) => (
          <p key={warning}>{warning}</p>
        ))}
        <dl className="lab-workflow-settings">
          {Object.entries(action.body as Record<string, unknown>).map(
            ([key, value]) => (
              <div key={key}>
                <dt>{key.replaceAll("_", " ")}</dt>
                <dd>{String(value)}</dd>
              </div>
            ),
          )}
        </dl>
      </details>
      {state === "pending" && (
        <div className="lab-row">
          <button
            className="primary-btn"
            disabled={disabled}
            onClick={onApprove}
          >
            Approve and run
          </button>
          <button className="text-btn" onClick={onDismiss}>
            Dismiss
          </button>
        </div>
      )}
      {state === "submitting" && (
        <p role="status">Submitting the approved action…</p>
      )}
      {state === "dismissed" && <p>Dismissed. Nothing was run.</p>}
      {state === "done" && !job && <p>Action completed.</p>}
      {job && (
        <details open={Boolean(active)}>
          <summary>
            {JOB_LABEL[job.kind] || job.kind}:{" "}
            {active ? job.stage || job.status : verdict?.label || job.status}
          </summary>
          {active && (
            <>
              <progress
                max={100}
                value={job.progress || 0}
                aria-label="Job progress"
              />
              <p>
                {job.progress || 0}% · {job.stage || job.status}
              </p>
            </>
          )}
          {verdict && <p>{verdict.detail}</p>}
          {job.result?.remote_checkpoint && (
            <p>
              Checkpoint: <code>{job.result.remote_checkpoint}</code>
            </p>
          )}
          <details>
            <summary>Execution log</summary>
            <pre className="lab-log">
              {(job.logs || []).slice(-100).join("\n") || "No log entries yet."}
            </pre>
          </details>
        </details>
      )}
    </div>
  );
}
