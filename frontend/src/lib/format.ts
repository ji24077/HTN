import type { AuditEvent, Task, Worker } from "../api/types";
export function formatMoney(value: string | null | undefined) {
  if (value == null || !value.trim()) return "—";
  const amount = Number(value);
  if (!Number.isFinite(amount)) return "—";
  if (amount > 0 && amount < 0.0001) return "<CA$0.0001";
  return new Intl.NumberFormat("en-US", {
    style: "currency",
    currency: "CAD",
    minimumFractionDigits: 2,
    maximumFractionDigits: 4,
  }).format(amount);
}

/**
 * The names machines were given when they were paired, by worker id.
 *
 * A module-level map rather than a prop, because the id is all most callers have: a
 * task row knows which worker ran it, not which device that worker is, and threading
 * the fleet into every table to answer that would put the same list in five places.
 * `rememberWorkers` is called from the one place a snapshot arrives, before the state
 * update that re-renders, so a name is never a render behind its worker.
 *
 * Entries are kept for machines that have left the snapshot. A finished task still
 * names the laptop that ran it after that laptop has gone offline, which is the whole
 * reason anyone reads the column.
 */
const names = new Map<string, string>();

export function rememberWorkers(workers: Worker[]) {
  for (const worker of workers) {
    const name = worker.name?.trim();
    if (name) names.set(worker.id, name);
  }
}

/**
 * What to call a worker on screen.
 *
 * Falls back to the raw id, which is a UUID for a paired device. That is unreadable but
 * honest: it is the only handle that definitely exists, and showing it is better than
 * inventing a name for a machine nobody has named.
 */
export const workerName = (id: string | null) =>
  id === "worker-a"
    ? "Worker A"
    : id === "worker-b"
      ? "Worker B"
      : (id && names.get(id)) || id || "Any worker";

/** A UUID trimmed to something a person can compare at a glance. */
export const shortId = (id: string) => (id.length > 12 ? id.slice(0, 8) : id);
export const active = (task: Task) =>
  task.state === "assigned" || task.state === "running";
export const healthy = (worker: Worker) =>
  worker.state === "alive" && Date.now() - Date.parse(worker.last_seen) < 17000;
export function record(value: unknown): Record<string, unknown> {
  return value !== null && typeof value === "object"
    ? (value as Record<string, unknown>)
    : {};
}
export function taskTitle(task: Task) {
  const label = record(record(task.spec.payload).value).label;
  return typeof label === "string" ? label : task.spec.id;
}
export function age(date: string) {
  const seconds = Math.max(
    0,
    Math.floor((Date.now() - Date.parse(date)) / 1000),
  );
  return seconds < 60
    ? `${seconds}s ago`
    : seconds < 3600
      ? `${Math.floor(seconds / 60)}m ago`
      : `${Math.floor(seconds / 3600)}h ago`;
}
export const time = (date: string | Date) =>
  new Date(date).toLocaleTimeString([], {
    hour: "2-digit",
    minute: "2-digit",
    second: "2-digit",
  });
export function eventText(event: AuditEvent, tasks: Task[]) {
  const who = workerName(
    typeof event.details.worker_id === "string"
      ? event.details.worker_id
      : event.entity === "worker" || event.entity === "enrollment"
        ? event.entity_id
        : null,
  );
  if (event.entity === "enrollment") {
    const reason = event.details.reason;
    const outcome =
      event.new_state === "active"
        ? "enrolled"
        : reason === "cancelled"
          ? "enrollment withdrawn"
          : reason === "expired"
            ? "enrollment expired"
            : event.new_state === "failed"
              ? "enrollment failed"
              : `enrollment ${event.new_state}`;
    return `${who} ${outcome}`;
  }
  const task = tasks.find((task) => task.spec.id === event.entity_id);
  const name = task ? taskTitle(task) : event.entity_id;
  if (event.entity === "worker")
    return `${who} ${event.new_state === "alive" ? "connected" : event.new_state === "unhealthy" ? "lost connection" : "went offline"}`;
  if (event.new_state === "assigned") return `Sent “${name}” to ${who}`;
  if (event.new_state === "running")
    return `${event.details.worker_id ? who : "Scheduler"} started “${name}”`;
  if (event.new_state === "succeeded") return `Result accepted for “${name}”`;
  if (event.new_state === "queued")
    return event.previous_state
      ? `Requeued “${name}” · ${event.details.reason || "retry"}`
      : `Queued “${name}”`;
  return `“${name}” ${event.new_state}`;
}
