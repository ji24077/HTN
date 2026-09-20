import { render, screen } from "@testing-library/react";
import userEvent from "@testing-library/user-event";
import { expect, it, vi } from "vitest";
import { TaskList } from "../components/TaskList";
import type { Task } from "../api/types";
import { groupJobs } from "../lib/jobs";

function task(
  id: string,
  job: string,
  state: Task["state"],
  worker = "worker-a",
): Task {
  return {
    spec: {
      id,
      job_id: job,
      kind: "stub",
      payload: { value: { label: job } },
      requirements: { runtime: "cpu", vram_mib: 0 },
      max_attempts: 3,
      timeout_seconds: 90,
      target_worker_id: null,
      allow_failover: true,
    },
    state,
    generation: 1,
    worker_id: worker,
    session_id: null,
    lease_until: null,
    deadline: null,
    result: null,
    failure: "",
    created_at: "2026-09-19T12:00:00Z",
    progress: state === "running" ? 20 : 0,
    started_at: null,
  };
}

it("groups tasks into jobs, opens the active task, and filters by state and worker", async () => {
  const user = userEvent.setup();
  const open = vi.fn();
  const active = task("two", "Simulation", "running", "worker-b");
  render(
    <TaskList
      tasks={[
        task("one", "Simulation", "succeeded"),
        active,
        task("three", "Broken render", "failed"),
      ]}
      cancelling={new Set()}
      onCancel={vi.fn()}
      onDetail={open}
    />,
  );
  expect(screen.getAllByRole("row")).toHaveLength(3);
  expect(
    screen.getByRole("progressbar", { name: "Simulation progress" }),
  ).toHaveAttribute("aria-valuenow", "60");
  await user.click(
    screen.getByRole("button", { name: "View Simulation details" }),
  );
  expect(open).toHaveBeenCalledWith(active);
  await user.click(screen.getByRole("button", { name: "Failed" }));
  expect(
    screen.queryByRole("button", { name: "View Simulation details" }),
  ).not.toBeInTheDocument();
  expect(
    screen.getByRole("button", { name: "View Broken render details" }),
  ).toBeInTheDocument();
  await user.click(screen.getByRole("button", { name: /All jobs/ }));
  await user.type(screen.getByLabelText("Search jobs"), "worker-b");
  expect(
    screen.getByRole("button", { name: "View Simulation details" }),
  ).toBeInTheDocument();
  expect(
    screen.queryByRole("button", { name: "View Broken render details" }),
  ).not.toBeInTheDocument();
});

it("opens the service root after refresh regardless of attempt ordering", async () => {
  const user = userEvent.setup();
  const open = vi.fn();
  const cancel = vi.fn();
  const root = task("service", "service", "running", "");
  root.spec.kind = "simulation_job";
  root.spec.payload = {
    execution_mode: "service",
    phase: "ready",
    description: "Hosted model",
  };
  const attempt = task("attempt", "service", "running");
  attempt.spec.kind = "python_service";
  render(
    <TaskList
      tasks={[attempt, root]}
      cancelling={new Set()}
      onCancel={cancel}
      onDetail={open}
    />,
  );
  expect(screen.getByText("Persistent service")).toBeInTheDocument();
  expect(screen.queryByRole("progressbar")).not.toBeInTheDocument();
  await user.click(screen.getByRole("button", { name: /View .* details/ }));
  expect(open).toHaveBeenCalledWith(root);
  await user.click(screen.getByRole("button", { name: /Stop / }));
  expect(cancel).toHaveBeenCalledWith(root.spec.id);
  // A historical failure must not make a deliberately stopped service look failed.
  attempt.state = "failed";
  root.state = "cancelled";
  expect(groupJobs([attempt, root])[0].state).toBe("cancelled");
});

it("does not mistake the initial snapshot load for an empty job history", () => {
  render(
    <TaskList
      tasks={[]}
      loading
      cancelling={new Set()}
      onCancel={() => {}}
      onDetail={() => {}}
    />,
  );
  expect(screen.getByRole("status")).toHaveTextContent("Loading your jobs");
  expect(screen.queryByText("Your work starts here")).not.toBeInTheDocument();
});
