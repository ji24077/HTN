import type { Task } from "../api/types";
import { active, record, taskTitle } from "./format";

export const statusLabels: Record<string, string> = {
  assigned: "Starting",
  queued: "Queued",
  running: "Running",
  succeeded: "Completed",
  failed: "Failed",
  cancelled: "Cancelled",
  paused: "Paused",
};
export interface JobGroup {
  id: string;
  title: string;
  tasks: Task[];
  primary: Task;
  state: Task["state"];
  progress: number;
  createdAt: string;
  workers: string[];
}
export function groupJobs(tasks: Task[]): JobGroup[] {
  const groups = new Map<string, Task[]>();
  for (const task of tasks)
    groups.set(task.spec.job_id, [
      ...(groups.get(task.spec.job_id) || []),
      task,
    ]);
  return [...groups]
    .map(([id, children]) => {
      const service = children.find(
        (t) =>
          t.spec.kind === "simulation_job" &&
          record(t.spec.payload).execution_mode === "service",
      );
      const primary =
        service ||
        children.find(active) ||
        children.find((t) => t.state === "queued") ||
        children.find((t) => t.state === "failed") ||
        children[0];
      const state: Task["state"] = service
        ? service.state
        : children.some(active)
          ? "running"
          : children.some((t) => t.state === "queued")
            ? "queued"
            : children.some((t) => t.state === "failed")
              ? "failed"
              : children.every((t) => t.state === "succeeded")
                ? "succeeded"
                : "cancelled";
      return {
        id,
        tasks: children,
        primary,
        state,
        title: service
          ? taskTitle(service)
          : children.length > 1 &&
              taskTitle(children[0]) === children[0].spec.id
            ? id
            : taskTitle(children[0]),
        progress: Math.round(
          children.reduce(
            (sum, t) => sum + (t.state === "succeeded" ? 100 : t.progress),
            0,
          ) / children.length,
        ),
        createdAt: children.reduce(
          (date, t) => (t.created_at < date ? t.created_at : date),
          children[0].created_at,
        ),
        workers: [
          ...new Set(
            children
              .map((t) => t.worker_id)
              .filter((id): id is string => Boolean(id)),
          ),
        ],
      };
    })
    .sort(
      (a, b) =>
        Number(["running", "queued"].includes(b.state)) -
          Number(["running", "queued"].includes(a.state)) ||
        Date.parse(b.createdAt) - Date.parse(a.createdAt),
    );
}

export function workloadLabel(task: Task): string {
  const payload = record(task.spec.payload);
  if (
    payload.execution_mode === "service" ||
    task.spec.kind === "python_service"
  )
    return "Hosted service";
  const names: Record<string, string> = {
    training: "Model training",
    rendering: "Rendering",
    simulation: "Simulation",
    python: "Python program",
  };
  if (typeof payload.workload === "string" && names[payload.workload])
    return names[payload.workload];
  return task.spec.kind === "simulation_job"
    ? "Compute project"
    : task.spec.kind === "stub" || task.spec.kind === "echo"
      ? "Connection test"
      : task.spec.kind.replaceAll("_", " ");
}
