import { render, screen } from "@testing-library/react";
import userEvent from "@testing-library/user-event";
import { expect, it, vi } from "vitest";
import { TaskList } from "../components/TaskList";
import type { Task } from "../api/types";

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
