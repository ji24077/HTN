import { Icon } from "./Icon";
import { useGpushare } from "../hooks/useGpushare";
import type { GpuJob } from "../api/types";

const duration = (seconds: number) => {
  const hours = Math.floor(seconds / 3600);
  const minutes = Math.floor((seconds % 3600) / 60);
  return hours ? `${hours}h ${minutes}m` : `${minutes}m`;
};

const jobTime = (job: GpuJob) => {
  const end = job.finished_at ?? Date.now() / 1000;
  return duration(Math.max(0, Math.round(end - job.created_at)));
};

export function GpuPanel({ active }: { active: boolean }) {
  const { enabled, pods, jobs, serving, error, loading } = useGpushare(active);
  const running = pods.filter((pod) => pod.status === "running");
  // What the rented hardware costs while this panel is open. Rented GPUs bill
  // by the hour whether or not anything is running on them, and the pod that
  // is quietly still up is the one nobody remembers to stop.
  const burn = running.reduce((total, pod) => total + pod.cost_per_hour, 0);

  if (loading) return <p className="muted">Loading GPU state…</p>;
  // Order matters: a failed first read leaves `enabled` at its false default,
  // so checking `enabled` first would report a running-but-unreachable service
  // as one nobody configured — collapsing the same two cases the backend
  // separates into 502 and 503, and sending the operator to edit config when
  // what they need to do is start gpushare.
  if (error && !enabled) return <p className="muted">{error}</p>;
  if (!enabled)
    return (
      <p className="muted">
        No GPU service configured. Set <code>GPUSHARE_URL</code> to the gpushare
        console to show rented GPUs, training jobs, and the resident model here.
      </p>
    );

  return (
    <section aria-label="GPU fleet">
      {error && <p className="muted">{error}</p>}
      <div className="section-actions">
        <p className="muted">
          {running.length} GPU{running.length === 1 ? "" : "s"} rented
          {burn > 0 && <> · ${burn.toFixed(2)}/hr</>}
        </p>
      </div>

      <table className="job-table">
        <thead>
          <tr>
            <th>Pod</th>
            <th>GPU</th>
            <th>Vendor</th>
            <th>Up</th>
            <th>$/hr</th>
            <th>Serving</th>
          </tr>
        </thead>
        <tbody>
          {pods.map((pod) => (
            <tr key={pod.id}>
              <td>{pod.name}</td>
              <td>{pod.gpu}</td>
              <td>{pod.vendor}</td>
              <td>
                {pod.status === "running"
                  ? duration(pod.uptime_seconds)
                  : pod.status}
              </td>
              <td>${pod.cost_per_hour.toFixed(2)}</td>
              <td>
                {serving?.running && serving.pod_id === pod.id ? (
                  <span>
                    <Icon name="check" size={14} /> {serving.model_id}
                    {serving.prefix_tokens
                      ? ` · ${serving.prefix_tokens.toLocaleString()} tok cached`
                      : ""}
                  </span>
                ) : (
                  <span className="muted">—</span>
                )}
              </td>
            </tr>
          ))}
          {pods.length === 0 && (
            <tr>
              <td colSpan={6} className="muted">
                No pods rented.
              </td>
            </tr>
          )}
        </tbody>
      </table>

      <div className="section-label">GPU JOBS</div>
      <table className="job-table">
        <thead>
          <tr>
            <th>Job</th>
            <th>Status</th>
            <th>Stage</th>
            <th>Took</th>
          </tr>
        </thead>
        <tbody>
          {jobs.slice(0, 10).map((job) => (
            <tr key={job.id}>
              <td>{job.kind}</td>
              <td>{job.status}</td>
              {/* A failed job's reason is the useful cell, not the stage it
                  stopped at — that is always "queued" for an early refusal. */}
              <td title={job.error ?? undefined}>{job.error ?? job.stage}</td>
              <td>{jobTime(job)}</td>
            </tr>
          ))}
          {jobs.length === 0 && (
            <tr>
              <td colSpan={4} className="muted">
                No GPU jobs yet.
              </td>
            </tr>
          )}
        </tbody>
      </table>

      <p className="muted">
        Read-only. Training, serving and NVIDIA→AMD migration run from the
        gpushare console, where their live logs are.
      </p>
    </section>
  );
}
