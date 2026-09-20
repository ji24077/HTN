import { act, render, screen } from "@testing-library/react";
import { afterEach, beforeEach, expect, it, vi } from "vitest";
import { JobSupervisor } from "../components/JobSupervisor";

const read = vi.hoisted(() => vi.fn());
vi.mock("../api/client", async (original) => ({
  ...(await original<typeof import("../api/client")>()),
  readSupervisor: read,
}));

const status = (finalized: boolean) => ({
  enabled: true,
  sentry_enabled: false,
  job: {
    state: "failed",
    finalized,
    memory: {
      findings: [
        {
          kind: "observation",
          text: "<script>failure</script>",
          evidence: ["task:1"],
        },
      ],
      questions: [],
      followups: [],
    },
  },
  runs: [
    {
      id: "run-1",
      status: "completed",
      reply: "No retry: deterministic failure.",
    },
  ],
  actions: [],
});

beforeEach(() => {
  vi.useFakeTimers();
  read.mockReset();
});
afterEach(() => vi.useRealTimers());

it("keeps monitoring a failed task until the supervisor finishes its final review", async () => {
  read.mockResolvedValueOnce(status(false)).mockResolvedValue(status(true));
  const view = render(<JobSupervisor jobId="job-1" />);
  await act(() => vi.advanceTimersByTimeAsync(0));
  expect(
    screen.getByText("No retry: deterministic failure."),
  ).toBeInTheDocument();
  expect(view.container.querySelector("script")).toBeNull();
  await act(() => vi.advanceTimersByTimeAsync(3000));
  expect(screen.getByText(/Review complete/)).toBeInTheDocument();
  await act(() => vi.advanceTimersByTimeAsync(9000));
  expect(read).toHaveBeenCalledTimes(2);
});

it("reports malformed or unavailable responses without crashing task details", async () => {
  read.mockResolvedValue({});
  render(<JobSupervisor jobId="job-1" />);
  await act(() => vi.advanceTimersByTimeAsync(0));
  expect(
    screen.getByText("Supervisor status unavailable."),
  ).toBeInTheDocument();
});

it("shows the run's estimated spend and refreshes when its cap is reached", async () => {
  const usage = {
    currency: "CAD",
    estimated: true,
    cost: "0.016667",
    cap: "0.020000",
    remaining: "0.003333",
    duration_seconds: "60",
    cap_reached: false,
    active_attempts: 1,
    attempts: 1,
  };
  read.mockResolvedValueOnce({ ...status(false), usage }).mockResolvedValue({
    ...status(true),
    usage: {
      ...usage,
      cost: "0.020100",
      cap_reached: true,
      active_attempts: 0,
    },
  });
  render(<JobSupervisor jobId="job-1" />);
  await act(() => vi.advanceTimersByTimeAsync(0));
  expect(screen.getByRole("region", { name: "Run usage" })).toBeInTheDocument();
  expect(screen.getByText("CA$0.0167")).toBeInTheDocument();
  expect(screen.getByText("Run cap: CA$0.02")).toBeInTheDocument();
  await act(() => vi.advanceTimersByTimeAsync(3000));
  expect(screen.getByText("CA$0.0201")).toBeInTheDocument();
  expect(
    screen.getByText("Run cap: CA$0.02 · Cap reached"),
  ).toBeInTheDocument();
});

it("shows uncapped usage even when supervision is disabled and resets on run changes", async () => {
  read.mockResolvedValue({
    ...status(false),
    enabled: false,
    usage: {
      cost: "0.000001",
      cap: null,
      cap_reached: false,
    },
  });
  const view = render(<JobSupervisor jobId="job-1" />);
  await act(() => vi.advanceTimersByTimeAsync(0));
  expect(screen.getByText("<CA$0.0001")).toBeInTheDocument();
  expect(screen.getByText("No usage cap")).toBeInTheDocument();
  read.mockImplementation(() => new Promise(() => {}));
  view.rerender(<JobSupervisor jobId="job-2" />);
  expect(screen.queryByText("<CA$0.0001")).not.toBeInTheDocument();
});

it("archives historical questions after completion while final review is pending", async () => {
  const pending = status(false);
  read.mockResolvedValue({
    ...pending,
    job: {
      ...pending.job,
      memory: {
        ...pending.job.memory,
        questions: ["Confirm 2000 steps?"],
        followups: [
          { check: "Wait for execution", due_at: "2026-09-20T00:00:00Z" },
        ],
      },
    },
  });
  render(<JobSupervisor jobId="job-1" terminal />);
  await act(() => vi.advanceTimersByTimeAsync(0));
  expect(screen.getByText("Final review pending")).toBeVisible();
  expect(screen.queryByText("Needs your input")).not.toBeInTheDocument();
  expect(screen.queryByText("Next check")).not.toBeInTheDocument();
  expect(screen.getByText("Confirm 2000 steps?")).not.toBeVisible();
});
