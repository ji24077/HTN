import { useEffect, useState } from "react";
import { Icon } from "./Icon";
import { ApiError, readSupervisor, type SupervisorStatus } from "../api/client";

export function JobSupervisor({ jobId }: { jobId: string }) {
  const [status, setStatus] = useState<SupervisorStatus | null>(null);
  const [error, setError] = useState("");
  useEffect(() => {
    setStatus(null);
    setError("");
    const controller = new AbortController();
    let timer: ReturnType<typeof setTimeout>;
    async function poll() {
      try {
        const next = await readSupervisor(jobId, controller.signal);
        if (controller.signal.aborted) return;
        if (
          !next.job?.memory ||
          !Array.isArray(next.runs) ||
          !Array.isArray(next.actions)
        )
          throw new Error("Invalid supervisor response");
        setStatus(next);
        setError("");
        if (next.job.finalized) return;
      } catch (cause) {
        if (controller.signal.aborted) return;
        if (cause instanceof ApiError && cause.status === 404) {
          setError("This job predates supervision.");
          return;
        }
        setError("Supervisor status unavailable.");
        if (cause instanceof ApiError && [401, 403].includes(cause.status))
          return;
      }
      if (!controller.signal.aborted) timer = setTimeout(poll, 3000);
    }
    void poll();
    return () => {
      controller.abort();
      clearTimeout(timer);
    };
  }, [jobId]);
  const latest = status?.runs.find((run) => run.reply);
  return (
    <section className="supervisor-card" aria-label="Job supervisor">
      <div className="supervisor-heading">
        <span className="supervisor-icon">
          <Icon name="assistant" />
        </span>
        <div>
          <h3>Job supervisor</h3>
          <small>
            {status
              ? !status.enabled
                ? "Disabled"
                : status.job.finalized
                  ? "Review complete"
                  : status.runs[0]?.status === "running"
                    ? "Reviewing this job…"
                    : "Monitoring"
              : "Connecting"}
          </small>
        </div>
      </div>
      {error && (
        <p role="status" className="inline-alert">
          {error}
        </p>
      )}
      {!status && !error && <p className="muted">Loading supervisor…</p>}
      {status && (
        <>
          <p className="supervisor-scope">
            Follows this job from submission to completion.
          </p>
          {!status.enabled && <p>Supervisor is not enabled on this server.</p>}
          {latest ? (
            <div className="supervisor-reply">
              <RichText text={latest.reply} />
            </div>
          ) : (
            <p className="muted">
              No review yet. Decisions will appear here as this job progresses.
            </p>
          )}
          {status.job.memory.questions.map((question, i) => (
            <div className="question-callout" key={i}>
              <strong>Needs your input</strong>
              <p>{question}</p>
            </div>
          ))}
          {status.actions.length > 0 && (
            <div className="supervisor-actions">
              <h4>
                Actions taken <span>{status.actions.length}</span>
              </h4>
              {status.actions.map((action) => (
                <div key={action.action_id}>
                  <strong>
                    {action.request.operation.replaceAll("_", " ")}
                  </strong>
                  <p>{action.request.reason}</p>
                  {action.result.state && <small>{action.result.state}</small>}
                </div>
              ))}
            </div>
          )}
          {status.job.memory.followups.map((followup, i) => (
            <div className="followup" key={i}>
              <strong>Next check</strong>
              <p>{followup.check}</p>
              <small>{new Date(followup.due_at).toLocaleString()}</small>
            </div>
          ))}
          {status.job.memory.findings.length > 0 && (
            <details className="supervisor-evidence">
              <summary>
                Findings & evidence{" "}
                <span>{status.job.memory.findings.length}</span>
              </summary>
              {status.job.memory.findings.map((finding, i) => (
                <div key={i}>
                  <strong>
                    {finding.kind === "hypothesis"
                      ? "Possible cause"
                      : "Finding"}
                  </strong>
                  <p>{finding.text}</p>
                  {finding.evidence.length > 0 && (
                    <small>Evidence: {finding.evidence.join(", ")}</small>
                  )}
                </div>
              ))}
            </details>
          )}
          {status.runs.length > 1 && (
            <details className="supervisor-evidence">
              <summary>
                Review history <span>{status.runs.length}</span>
              </summary>
              {status.runs.map((run) => (
                <div key={run.id}>
                  <small>
                    {new Date(run.created_at).toLocaleString()} · {run.status}
                  </small>
                  {run.reply && <RichText text={run.reply} />}
                </div>
              ))}
            </details>
          )}
          <div className="supervisor-foot">
            <Icon name="link" size={14} />
            {status.sentry_enabled
              ? "Execution logs + Sentry connected"
              : "Execution logs connected · Sentry not configured"}
          </div>
        </>
      )}
    </section>
  );
}
function inline(text: string) {
  return text
    .split(/(\*\*[^*]+\*\*|`[^`]+`)/g)
    .map((part, i) =>
      part.startsWith("**") ? (
        <strong key={i}>{part.slice(2, -2)}</strong>
      ) : part.startsWith("`") ? (
        <code key={i}>{part.slice(1, -1)}</code>
      ) : (
        part
      ),
    );
}
function RichText({ text }: { text: string }) {
  return (
    <>
      {text.split(/\n\n+/).map((paragraph, index) =>
        paragraph.split("\n").every((line) => /^[-*] /.test(line)) ? (
          <ul key={index}>
            {paragraph.split("\n").map((line, i) => (
              <li key={i}>{inline(line.slice(2))}</li>
            ))}
          </ul>
        ) : (
          <p key={index}>{inline(paragraph)}</p>
        ),
      )}
    </>
  );
}
