import { act, render, screen } from "@testing-library/react";
import { afterEach, beforeEach, expect, it, vi } from "vitest";
import { TaskDetails } from "../components/TaskDetails";
import { ApiError } from "../api/client";
import type { Task } from "../api/types";

const api = vi.hoisted(() => ({ executionEvents: vi.fn() }));
vi.mock("../api/client", async (importOriginal) => ({
  ...(await importOriginal<typeof import("../api/client")>()),
  executionEvents: api.executionEvents,
}));

function task(state: Task["state"]): Task {
  return {
    spec: {
      id: "task-1",
      job_id: "job",
      kind: "stub",
      payload: {},
      requirements: { runtime: "cpu", vram_mib: 0 },
      max_attempts: 1,
      timeout_seconds: 10,
      target_worker_id: null,
      allow_failover: false,
    },
    state,
    generation: 1,
    worker_id: "worker-a",
    session_id: "session",
    lease_until: null,
    deadline: null,
    result: null,
    failure: "",
    created_at: new Date().toISOString(),
    progress: 0,
    started_at: null,
  };
}

const page = (id: number) => ({
  events: [
    {
      id,
      task_id: "task-1",
      attempt: 1,
      worker_id: "worker-a",
      source: "worker",
      sequence: id,
      kind: "stdout",
      occurred_at: new Date().toISOString(),
      data: { text: `line ${id}` },
      execution_id: "task-1:1",
    },
  ],
  next_cursor: id,
  has_more: false,
});

const tick = (ms: number) => act(() => vi.advanceTimersByTimeAsync(ms));

beforeEach(() => {
  vi.useFakeTimers();
  api.executionEvents.mockReset();
});
afterEach(() => vi.useRealTimers());

it("polls a live task, then stops after the final page of a settled task", async () => {
  api.executionEvents.mockImplementation(async (_id, after: number) =>
    page(after + 1),
  );
  const { rerender } = render(
    <TaskDetails task={task("running")} onClose={() => {}} />,
  );
  await tick(0);
  expect(api.executionEvents).toHaveBeenCalledTimes(1);
  await tick(1500);
  expect(api.executionEvents).toHaveBeenCalledTimes(2);
  rerender(<TaskDetails task={task("succeeded")} onClose={() => {}} />);
  // One more fetch after settling picks up the final events, then polling ends.
  await tick(1500);
  expect(api.executionEvents).toHaveBeenCalledTimes(3);
  await tick(10_000);
  expect(api.executionEvents).toHaveBeenCalledTimes(3);
  expect(screen.getByText(/line 3/)).toBeInTheDocument();
});

it("stops polling after an unrenewable session instead of re-triggering sign-in", async () => {
  api.executionEvents.mockRejectedValue(new ApiError("Please sign in", 401));
  render(<TaskDetails task={task("running")} onClose={() => {}} />);
  await tick(0);
  expect(
    screen.getByText("Sign in again to view execution history."),
  ).toBeInTheDocument();
  await tick(10_000);
  expect(api.executionEvents).toHaveBeenCalledTimes(1);
});

it("keeps retrying transient failures", async () => {
  api.executionEvents.mockRejectedValue(new Error("network"));
  render(<TaskDetails task={task("running")} onClose={() => {}} />);
  await tick(0);
  expect(screen.getByText(/retrying/)).toBeInTheDocument();
  await tick(1500);
  expect(api.executionEvents).toHaveBeenCalledTimes(2);
});
