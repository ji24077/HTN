import { render, screen } from "@testing-library/react";
import { expect, it } from "vitest";
import type { Task } from "../api/types";
import type { SimulationStatus } from "../api/client";
import { JobOutcome } from "../components/JobOutcome";

const task = {
  state: "succeeded",
  worker_id: "gpu-worker",
  spec: { id: "job", job_id: "job", payload: { value: { label: "Training" } } },
  result: JSON.stringify({ output: { metrics: { mse: 0.002 } } }),
} as Task;
it("renders only recorded metrics with their true acceptance bounds", () => {
  const status = {
    message: "Validation passed.",
    workers: ["gpu-worker"],
    tasks: [],
    checks: [],
    program_plan: {
      requirements: { runtime: "mps" },
      metrics: [{ name: "mse", minimum: null, maximum: 0.01 }],
    },
  } as unknown as SimulationStatus;
  const view = render(<JobOutcome task={task} status={status} />);
  expect(screen.getByText("0.002")).toBeVisible();
  expect(screen.getByText(/Within required bounds/)).toBeVisible();
  expect(screen.getByText(/Planned runtime: Apple Metal/)).toBeVisible();
  expect(screen.getByRole("img")).toHaveAccessibleName(
    "mse: 0.002, maximum 0.01",
  );
  expect(screen.queryByText(/loss curve/i)).not.toBeInTheDocument();
  view.rerender(
    <JobOutcome
      task={{ ...task, result: { output: { metrics: { mse: 0.02 } } } }}
      status={status}
    />,
  );
  expect(screen.getByText(/Outside required bounds/)).toBeVisible();
});
it("does not fabricate metrics or offer a summary before results are available", () => {
  render(<JobOutcome task={{ ...task, state: "running", result: null }} />);
  expect(screen.getByText("Work in progress")).toBeVisible();
  expect(screen.queryByRole("img")).not.toBeInTheDocument();
  expect(
    screen.queryByRole("button", { name: "Download summary" }),
  ).not.toBeInTheDocument();
});
