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
export const workerName = (id: string | null) =>
  id === "worker-a"
    ? "Worker A"
    : id === "worker-b"
      ? "Worker B"
      : id || "Any worker";
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
  if (event.new_state === "running") return `${who} started “${name}”`;
  if (event.new_state === "succeeded") return `Result accepted for “${name}”`;
  if (event.new_state === "queued")
    return event.previous_state
      ? `Requeued “${name}” · ${event.details.reason || "retry"}`
      : `Queued “${name}”`;
  return `“${name}” ${event.new_state}`;
}
