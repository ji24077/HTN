import { act, render, screen } from "@testing-library/react";
import { afterEach, expect, it, vi } from "vitest";
import { SimulationDetails } from "../components/SimulationDetails";
const api = vi.hoisted(() => ({
  readSimulation: vi.fn(),
  cancelTask: vi.fn(),
  answerSimulation: vi.fn(),
  executionEvents: vi.fn(),
  jobOutputs: vi.fn().mockResolvedValue({ files: [] }),
}));
vi.mock("../api/client", () => ({ ...api, ApiError: class extends Error {} }));
afterEach(() => vi.useRealTimers());
const status = {
  phase: "running",
  message: "Running",
  round: 1,
  limits: { adaptations: 3 },
  workers: ["worker-a"],
  tasks: [],
  checks: [],
  versions: [],
  plan: { summary: "Monte Carlo", trials: 10000, batch_size: 32, workers: 2 },
  trial_counts: { succeeded: 896 },
};

it("shows a training plan and output files without simulation trial controls", async () => {
  api.readSimulation.mockReset().mockResolvedValue({
    ...status,
    phase: "completed",
    workload: "training",
    plan: null,
    program_plan: {
      summary: "Train and evaluate a checkpoint",
      worker_id: "worker-b",
      requirements: { runtime: "cuda", vram_mib: 8192 },
      validator: "validate.py",
      outputs: [{ path: "checkpoint.json", kind: "checkpoint" }],
      metrics: [{ name: "mse", minimum: null, maximum: 0.001 }],
    },
  });
  api.jobOutputs.mockResolvedValueOnce({
    files: [{ id: "out-1", name: "checkpoint.json", size: 200, attempt: 1 }],
  });
  render(<SimulationDetails jobId="training-job" cancelled={false} />);
  expect(
    await screen.findByText("Train and evaluate a checkpoint"),
  ).toBeInTheDocument();
  expect(
    await screen.findByRole("button", { name: "Download checkpoint.json" }),
  ).toBeInTheDocument();
  expect(screen.getByText(/Selected machine: worker-b/)).toBeInTheDocument();
  expect(screen.getByText(/mse:.*maximum 0.001/)).toBeInTheDocument();
  expect(screen.queryByLabelText("Trial progress")).not.toBeInTheDocument();
  expect(screen.queryByText(/Validate on two workers/)).not.toBeInTheDocument();
});
it("cancels and waits for worker cleanup confirmation", async () => {
  vi.useFakeTimers();
  api.readSimulation
    .mockReset()
    .mockResolvedValueOnce(status)
    .mockResolvedValueOnce({
      ...status,
      phase: "cancelled",
      cleanup: { required: 1, confirmed: 0, pending_workers: ["worker-a"] },
    })
    .mockResolvedValue({
      ...status,
      phase: "cancelled",
      cleanup: { required: 1, confirmed: 1, pending_workers: [] },
    });
  api.cancelTask.mockResolvedValue({});
  render(<SimulationDetails jobId="sim-1" cancelled={false} />);
  await act(async () => {
    await vi.advanceTimersByTimeAsync(1);
  });
  expect(screen.getByLabelText("Trial progress")).toHaveTextContent(
    "896 / 10,000 trials completed",
  );
  await act(async () => {
    screen.getByRole("button", { name: "Cancel & clean up workers" }).click();
  });
  expect(api.cancelTask).toHaveBeenCalledWith("sim-1");
  expect(screen.getByRole("status")).toHaveTextContent(
    "Waiting for process and file cleanup on worker-a",
  );
  await act(async () => {
    await vi.advanceTimersByTimeAsync(2000);
  });
  expect(screen.getByRole("status")).toHaveTextContent(
    "Worker cleanup confirmed",
  );
  const reads = api.readSimulation.mock.calls.length;
  await act(async () => {
    await vi.advanceTimersByTimeAsync(10000);
  });
  expect(api.readSimulation.mock.calls.length).toBe(reads);
});

it("shows the measured agent schedule without legacy batch defaults", async () => {
  api.readSimulation.mockReset().mockResolvedValue({
    ...status,
    phase: "completed",
    plan: { summary: "Measured Monte Carlo run", trials: 10000 },
    policy: {
      rationale: "Startup dominates compute.",
      local_cases: 7,
      independent_cases: 9,
      aggregation: "inline",
      validation_timeout_seconds: 45,
    },
    schedule: {
      rationale: "One worker can finish before another transfer would pay off.",
      batches: [{ worker_id: "worker-b", trials: 10000, timeout_seconds: 37 }],
      aggregation_worker: "worker-b",
      aggregation_timeout_seconds: 23,
    },
    measurements: [
      {
        stage: "profiling",
        worker_id: "worker-b",
        tasks: 1,
        trials: 8,
        compute_seconds: 0.002,
        execution_seconds: 0.04,
        observed_wall_seconds: 1.5,
        output_bytes: 500,
      },
    ],
  });
  render(<SimulationDetails jobId="sim-agent" cancelled={false} />);
  expect(
    await screen.findByText("Agent schedule · 1 task"),
  ).toBeInTheDocument();
  expect(
    screen.getByText("10,000 trials on worker-b · 37s timeout"),
  ).toBeInTheDocument();
  expect(screen.getByText("Startup dominates compute.")).toBeInTheDocument();
  expect(screen.getByText("Measured execution time")).toBeInTheDocument();
  expect(screen.queryByText(/batches of 32/)).not.toBeInTheDocument();
});

it("shows Python dependencies and downloadable outputs", async () => {
  api.readSimulation.mockReset().mockResolvedValue({
    ...status,
    phase: "completed",
    plan: null,
    program_plan: {
      summary: "Train a CUDA model",
      worker_id: "worker-a",
      requirements: { runtime: "cuda", vram_mib: 8192 },
      outputs: [{ path: "model.pt", kind: "checkpoint" }],
      metrics: [],
      entrypoint: "train.py",
      validator: "validate.py",
      dependencies: ["torch", "numpy"],
    },
  });
  api.jobOutputs.mockResolvedValue({
    files: [{ id: "checkpoint-1", name: "model.pt", size: 2048, attempt: 1 }],
  });
  render(<SimulationDetails jobId="python-1" cancelled={false} />);
  expect(
    await screen.findByText("Python dependencies: torch, numpy"),
  ).toBeInTheDocument();
  expect(
    await screen.findByRole("button", { name: "Download model.pt" }),
  ).toBeInTheDocument();
  expect(screen.getByText(/Selected machine: worker-a · cuda/)).toBeInTheDocument();
  expect(screen.getByText("3. Run & validate outputs")).toHaveClass("current");
  expect(screen.queryByLabelText("Trial progress")).not.toBeInTheDocument();
});

it("shows concurrent analysts and their saved reports", async () => {
  api.readSimulation.mockReset().mockResolvedValue({
    ...status,
    phase: "planning",
    analysis: {
      calls_used: 3,
      rationale: "Review independent questions",
      children: [
        { id: "a", role: "dependencies", status: "running", question: "Which libraries?" },
        { id: "b", role: "parallelization", status: "running", question: "Can trials overlap?" },
        { id: "c", role: "validation", status: "completed", question: "How to validate?",
          report: { summary: "Compare seeded outputs", evidence: ["simulate(seed)"], questions: ["Which tolerance?"] } },
      ],
    },
  });
  const { rerender } = render(<SimulationDetails jobId="parallel" cancelled={false} />);
  expect(await screen.findByText(/2 analysis agents running concurrently/)).toBeVisible();
  rerender(<SimulationDetails jobId="parallel" cancelled={false} view="Details" />);
  expect(screen.getByRole("region", { name: "Analysis agents" })).toBeVisible();
  expect(screen.getByText("dependencies · running")).toBeVisible();
  expect(screen.getByText("parallelization · running")).toBeVisible();
  await act(async () => { screen.getByText("validation · completed").click(); });
  expect(screen.getByText("Compare seeded outputs")).toBeVisible();
  expect(screen.getByText("simulate(seed)")).toBeVisible();
  expect(screen.getByText("Which tolerance?")).toBeVisible();
});
