import { useState } from "react";
import type { Task } from "../api/types";
import { active, age, workerName } from "../lib/format";
import { groupJobs, statusLabels } from "../lib/jobs";
import { Icon } from "./Icon";

const filters = ["All jobs", "Active", "Failed", "Completed"] as const;
export function TaskList({
  tasks,
  cancelling,
  onCancel,
  onDetail,
}: {
  tasks: Task[];
  cancelling: Set<string>;
  onCancel: (id: string) => void;
  onDetail: (task: Task) => void;
}) {
  const [filter, setFilter] = useState<(typeof filters)[number]>("All jobs");
  const [query, setQuery] = useState("");
  const [page, setPage] = useState(0);
  const jobs = groupJobs(tasks);
  const match = (state: string) =>
    filter === "All jobs" ||
    (filter === "Active" && ["running", "queued"].includes(state)) ||
    (filter === "Failed" && state === "failed") ||
    (filter === "Completed" && state === "succeeded");
  const filtered = jobs.filter(
    (job) =>
      match(job.state) &&
      [job.title, job.id, ...job.tasks.map((t) => t.spec.id), ...job.workers]
        .join(" ")
        .toLowerCase()
        .includes(query.toLowerCase()),
  );
  const currentPage = Math.min(
    page,
    Math.max(0, Math.ceil(filtered.length / 15) - 1),
  );
  const visible = filtered.slice(currentPage * 15, currentPage * 15 + 15);
  return (
    <section className="jobs-panel" aria-label="Job tracking">
      <div className="jobs-toolbar">
        <div className="filter-tabs" aria-label="Filter jobs">
          {filters.map((item) => (
            <button
              key={item}
              aria-pressed={filter === item}
              onClick={() => {
                setFilter(item);
                setPage(0);
              }}
            >
              {item}
              {item === "All jobs" && <small>{jobs.length}</small>}
            </button>
          ))}
        </div>
        <label className="search-field">
          <Icon name="search" size={16} />
          <input
            aria-label="Search jobs"
            placeholder="Search jobs, IDs or workers…"
            value={query}
            onChange={(event) => {
              setQuery(event.target.value);
              setPage(0);
            }}
          />
        </label>
      </div>
      <div className="job-table-wrap">
        <table className="job-table">
          <thead>
            <tr>
              <th scope="col">Job</th>
              <th scope="col">Status</th>
              <th scope="col">Progress</th>
              <th scope="col">Worker</th>
              <th scope="col">Submitted</th>
              <th scope="col">
                <span className="sr-only">Actions</span>
              </th>
            </tr>
          </thead>
          <tbody>
            {visible.map((job) => (
              <tr key={job.id} data-task={job.primary.spec.id}>
                <td>
                  <button
                    className="job-open"
                    aria-label={`View ${job.title} details`}
                    onClick={() => onDetail(job.primary)}
                  >
                    <span className={`job-glyph ${job.state}`}>
                      <Icon
                        name={
                          job.state === "succeeded"
                            ? "check"
                            : job.state === "failed"
                              ? "warning"
                              : "jobs"
                        }
                        size={17}
                      />
                    </span>
                    <span>
                      <strong>{job.title}</strong>
                      <small>
                        {job.tasks.length > 1
                          ? `${job.tasks.length} tasks · `
                          : ""}
                        {job.primary.spec.kind.replaceAll("_", " ")}
                        <span className="mobile-progress">
                          {["running", "queued"].includes(job.state)
                            ? ` · ${job.progress}%`
                            : ""}
                        </span>
                        <span className="job-short-id">
                          {" "}
                          ·{" "}
                          {job.id.length > 24
                            ? job.id.slice(0, 20) + "…"
                            : job.id}
                        </span>
                      </small>
                    </span>
                  </button>
                </td>
                <td>
                  <span className={`status-badge ${job.state}`}>
                    <i />
                    {statusLabels[job.state]}
                  </span>
                </td>
                <td>
                  <div className="table-progress">
                    <div
                      className="progress"
                      role="progressbar"
                      aria-label={`${job.title} progress`}
                      aria-valuenow={job.progress}
                      aria-valuemin={0}
                      aria-valuemax={100}
                    >
                      <i style={{ width: `${job.progress}%` }} />
                    </div>
                    <span>{job.progress}%</span>
                  </div>
                  <small className="muted">
                    {job.tasks.length > 1
                      ? `${job.tasks.filter((t) => t.state === "succeeded").length} / ${job.tasks.length} tasks`
                      : job.primary.generation > 1
                        ? `Attempt ${job.primary.generation} of ${job.primary.spec.max_attempts}`
                        : job.state === "queued"
                          ? "Waiting for capacity"
                          : job.primary.generation
                            ? "First attempt"
                            : "Not started"}
                  </small>
                </td>
                <td>
                  <span className="worker-cell" title={job.workers.join(", ")}>
                    {job.workers.length > 1
                      ? `${job.workers.length} workers`
                      : workerName(
                          job.workers[0] || job.primary.spec.target_worker_id,
                        )}
                  </span>
                </td>
                <td className="muted">
                  <time
                    dateTime={job.createdAt}
                    title={new Date(job.createdAt).toLocaleString()}
                  >
                    {age(job.createdAt)}
                  </time>
                </td>
                <td>
                  {job.tasks.length === 1 &&
                  (active(job.primary) || job.primary.state === "queued") ? (
                    <button
                      className="text-btn danger"
                      aria-label={`Cancel ${job.title}`}
                      disabled={cancelling.has(job.primary.spec.id)}
                      onClick={() => onCancel(job.primary.spec.id)}
                    >
                      Cancel
                    </button>
                  ) : (
                    <button
                      className="text-btn"
                      aria-label={`Open ${job.title}`}
                      onClick={() => onDetail(job.primary)}
                    >
                      <Icon name="chevron" size={15} />
                    </button>
                  )}
                </td>
              </tr>
            ))}
          </tbody>
        </table>
      </div>
      {!visible.length && (
        <div className="empty">
          <Icon name={query ? "search" : "jobs"} size={28} />
          <h3>{jobs.length ? "No matching jobs" : "Your work starts here"}</h3>
          <p>
            {jobs.length
              ? "Try another search or status filter."
              : "Create a job to track its progress, worker assignments and results."}
          </p>
        </div>
      )}
      <div className="table-footer">
        <span>
          {filtered.length
            ? `${currentPage * 15 + 1}–${Math.min(currentPage * 15 + 15, filtered.length)} of ${filtered.length} jobs`
            : "0 jobs"}
          {tasks.length >= 500 ? " · latest 500 tasks loaded" : ""}
        </span>
        <div>
          <button
            className="text-btn"
            disabled={currentPage === 0}
            onClick={() => setPage(currentPage - 1)}
          >
            Previous
          </button>
          <button
            className="text-btn"
            disabled={(currentPage + 1) * 15 >= filtered.length}
            onClick={() => setPage(currentPage + 1)}
          >
            Next
          </button>
        </div>
      </div>
    </section>
  );
}
