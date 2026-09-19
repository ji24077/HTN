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
