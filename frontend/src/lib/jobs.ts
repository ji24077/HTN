import type { Task } from "../api/types";
import { active, taskTitle } from "./format";

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
      const primary =
        children.find(active) ||
        children.find((t) => t.state === "queued") ||
        children.find((t) => t.state === "failed") ||
        children[0];
      const state: Task["state"] = children.some(active)
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
        title:
          children.length > 1 && taskTitle(children[0]) === children[0].spec.id
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
