import { act, fireEvent, render, screen } from "@testing-library/react";
import { afterEach, beforeEach, expect, it, vi } from "vitest";
import { TaskDetails } from "../components/TaskDetails";
import { ApiError } from "../api/client";
import type { Task } from "../api/types";

const api = vi.hoisted(() => ({ executionEvents: vi.fn(), getTask: vi.fn() }));
vi.mock("../api/client", async (importOriginal) => ({
  ...(await importOriginal<typeof import("../api/client")>()),
  executionEvents: api.executionEvents,
  getTask: api.getTask,
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
  api.getTask.mockReset().mockResolvedValue(task("running"));
});
afterEach(() => vi.useRealTimers());

it("keeps a loading job dialog open and provides recovery after a link fails", () => {
  const retry = vi.fn();
  const close = vi.fn();
  const { rerender } = render(
    <TaskDetails
      task={null}
      request={{ retryable: true }}
      onRetry={retry}
      onClose={close}
    />,
  );
  expect(screen.getByRole("dialog")).toBeVisible();
  expect(screen.getByRole("status")).toHaveTextContent("Loading your job");
  rerender(
    <TaskDetails
      task={null}
      request={{ error: "Could not load this job.", retryable: true }}
      onRetry={retry}
      onClose={close}
    />,
  );
  expect(screen.getByRole("alert")).toHaveTextContent(
    "Could not load this job",
  );
  fireEvent.click(screen.getByRole("button", { name: "Retry job" }));
  expect(retry).toHaveBeenCalledOnce();
  fireEvent.click(screen.getByRole("button", { name: "Back to jobs" }));
  expect(close).toHaveBeenCalledOnce();
});

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
  fireEvent.click(screen.getByRole("button", { name: "Activity" }));
  fireEvent.click(screen.getByRole("tab", { name: /Logs/ }));
  expect(
    screen.getByText(/line 3/, { selector: ".log-line > div > pre" }),
  ).toBeInTheDocument();
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

it("resumes logs after a failed task is retried on another worker, preserving earlier attempts", async () => {
  api.executionEvents.mockResolvedValueOnce(page(1)).mockResolvedValue({
    ...page(2),
    events: [
      {
        ...page(2).events[0],
        attempt: 2,
        worker_id: "worker-b",
        data: { text: "second worker output" },
      },
    ],
  });
  const { rerender } = render(
    <TaskDetails task={task("failed")} onClose={() => {}} />,
  );
  await tick(0);
  await tick(5000);
  expect(api.executionEvents).toHaveBeenCalledTimes(1);
  rerender(
    <TaskDetails
      task={{ ...task("running"), generation: 2, worker_id: "worker-b" }}
      onClose={() => {}}
    />,
  );
  await tick(0);
  expect(api.executionEvents).toHaveBeenLastCalledWith(
    "task-1",
    1,
    expect.any(AbortSignal),
  );
  fireEvent.click(screen.getByRole("button", { name: "Activity" }));
  fireEvent.click(screen.getByRole("tab", { name: /Logs/ }));
  expect(
    screen.getByText("line 1", { selector: ".log-line > div > pre" }),
  ).toBeInTheDocument();
  expect(
    screen.getByText("second worker output", {
      selector: ".log-line > div > pre",
    }),
  ).toBeInTheDocument();
  fireEvent.change(screen.getByLabelText("Filter by worker"), {
    target: { value: "worker-b" },
  });
  expect(
    screen.queryByText("line 1", { selector: ".log-line > div > pre" }),
  ).not.toBeInTheDocument();
  expect(
    screen.getByText("second worker output", {
      selector: ".log-line > div > pre",
    }),
  ).toBeInTheDocument();
  fireEvent.change(screen.getByLabelText("Search execution logs"), {
    target: { value: "does not exist" },
  });
  expect(screen.getByText("No logs match these filters.")).toBeInTheDocument();
});

it("refreshes the visible outcome on completion and lets a failed result fetch retry", async () => {
  api.executionEvents.mockResolvedValue({
    events: [],
    next_cursor: 0,
    has_more: false,
  });
  const view = render(
    <TaskDetails task={task("running")} onClose={() => {}} />,
  );
  await tick(0);
  api.getTask.mockRejectedValueOnce(new Error("network"));
  view.rerender(<TaskDetails task={task("succeeded")} onClose={() => {}} />);
  await tick(0);
  expect(screen.getByRole("alert")).toHaveTextContent(
    "Could not load the latest result",
  );
  api.getTask.mockResolvedValue({
    ...task("succeeded"),
    result: JSON.stringify({ output: { metrics: { mse: 0.002 } } }),
  });
  fireEvent.click(screen.getByRole("button", { name: "Retry result" }));
  await tick(0);
  expect(screen.getByText("0.002")).toBeVisible();
  expect(screen.queryByRole("alert")).not.toBeInTheDocument();
  expect(
    screen.getByRole("button", { name: "Download summary" }),
  ).toBeVisible();
});

it("preserves underscores in worker messages", async () => {
  api.executionEvents.mockResolvedValue({
    ...page(1),
    events: [
      {
        ...page(1).events[0],
        kind: "failed",
        data: { text: "gpu_upload.py requires DISPATCH_OUTPUT_DIR" },
      },
    ],
  });
  render(<TaskDetails task={task("failed")} onClose={() => {}} />);
  await tick(0);
  fireEvent.click(screen.getByRole("button", { name: "Activity" }));
  expect(
    screen.getByText("gpu_upload.py requires DISPATCH_OUTPUT_DIR", {
      selector: ".event-message",
    }),
  ).toBeVisible();
});
