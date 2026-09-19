import type { Task, TaskSpec } from "./types";

async function request<T>(path: string, options: RequestInit = {}): Promise<T> {
  const response = await fetch(path, {
    credentials: "same-origin",
    ...options,
    headers: { "Content-Type": "application/json", ...options.headers },
  });
  if (!response.ok) {
    const body = await response.json().catch(() => null);
    const message = body?.detail || body?.error;
    throw new Error(
      typeof message === "string"
        ? message
        : `Request failed (${response.status})`,
    );
  }
  return response.json() as Promise<T>;
}
export const openSession = (signal: AbortSignal) =>
  request("/demo/session", { signal });
export const submitTasks = (tasks: TaskSpec[]) =>
  request<Task[]>("/v1/tasks", {
    method: "POST",
    body: JSON.stringify({ tasks }),
  });
export const cancelTask = (id: string) =>
  request<Task>(`/v1/tasks/${encodeURIComponent(id)}/cancel`, {
    method: "POST",
  });
export function stubTask(
  label: string,
  target: string,
  seconds: number,
  failover: boolean,
): TaskSpec {
  return {
    id: "task-" + crypto.randomUUID(),
    job_id: "playground",
    kind: "stub",
    payload: { duration_seconds: seconds, value: { label } },
    requirements: { runtime: "cpu", vram_mib: 0 },
    max_attempts: 3,
    timeout_seconds: seconds + 60,
    target_worker_id: target || null,
    allow_failover: failover,
  };
}
