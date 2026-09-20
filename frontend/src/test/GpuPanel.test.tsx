import { act, render, screen } from "@testing-library/react";
import { afterEach, beforeEach, expect, it, vi } from "vitest";
import { GpuPanel } from "../components/GpuPanel";
import { ApiError } from "../api/client";

const config = vi.hoisted(() => vi.fn());
const pods = vi.hoisted(() => vi.fn());
const models = vi.hoisted(() => vi.fn());
const jobs = vi.hoisted(() => vi.fn());
vi.mock("../api/client", async (original) => ({
  ...(await original<typeof import("../api/client")>()),
  gpushareConfig: config,
  gpusharePods: pods,
  gpushareModels: models,
  gpushareJobs: jobs,
}));

const pod = (id: string, cost: number, status = "running") => ({
  id,
  name: `pod-${id}`,
  status,
  gpu: "NVIDIA GeForce RTX 3090",
  vendor: "nvidia",
  gpu_count: 1,
  cost_per_hour: cost,
  datacenter: "EU-CZ-1",
  uptime_seconds: 7200,
});

beforeEach(() => {
  vi.useFakeTimers();
  for (const mock of [config, pods, models, jobs]) mock.mockReset();
  config.mockResolvedValue({ enabled: true });
  pods.mockResolvedValue({ pods: [] });
  models.mockResolvedValue({ serving: { running: false } });
  jobs.mockResolvedValue({ jobs: [] });
});
afterEach(() => vi.useRealTimers());

it("adds up what the rented GPUs cost per hour, counting only the running ones", async () => {
  // The idle-but-still-rented pod is the one nobody remembers to stop, so the
  // total has to be what is billing now, not what was ever rented.
  pods.mockResolvedValue({
    pods: [pod("a", 2.39), pod("b", 0.5), pod("c", 9.99, "exited")],
  });
  render(<GpuPanel active />);
  await act(() => vi.advanceTimersByTimeAsync(0));
  expect(screen.getByText(/2 GPUs rented/)).toBeInTheDocument();
  expect(screen.getByText(/\$2\.89\/hr/)).toBeInTheDocument();
});

it("marks which pod holds the resident model and how much prompt it has cached", async () => {
  pods.mockResolvedValue({ pods: [pod("a", 0.5), pod("b", 0.27)] });
  models.mockResolvedValue({
    serving: {
      running: true,
      pod_id: "a",
      model_id: "finetuned",
      prefix_tokens: 30857,
    },
  });
  render(<GpuPanel active />);
  await act(() => vi.advanceTimersByTimeAsync(0));
  expect(screen.getByText(/finetuned/)).toBeInTheDocument();
  expect(screen.getByText(/30,857 tok cached/)).toBeInTheDocument();
});

it("shows a failed job's reason rather than the stage it stopped at", async () => {
  // An early refusal never leaves "queued", and "queued" tells an operator
  // nothing about why the run did not happen.
  jobs.mockResolvedValue({
    jobs: [
      {
        id: "j1",
        kind: "optimize-inference-speed",
        status: "failed",
        stage: "queued",
        progress: 0,
        created_at: 1000,
        finished_at: 1024,
        error: "pod 2fe3 has no trained checkpoint",
      },
    ],
  });
  render(<GpuPanel active />);
  await act(() => vi.advanceTimersByTimeAsync(0));
  expect(
    screen.getByText("pod 2fe3 has no trained checkpoint"),
  ).toBeInTheDocument();
  expect(screen.queryByText("queued")).toBeNull();
});

it("tells the operator to configure the service instead of showing an empty fleet", async () => {
  config.mockResolvedValue({ enabled: false });
  render(<GpuPanel active />);
  await act(() => vi.advanceTimersByTimeAsync(0));
  expect(screen.getByText(/GPUSHARE_URL/)).toBeInTheDocument();
  expect(pods).not.toHaveBeenCalled();
});

it("separates a service that is down from one that was never configured", async () => {
  config.mockRejectedValue(new ApiError("gpushare did not answer", 502));
  render(<GpuPanel active />);
  await act(() => vi.advanceTimersByTimeAsync(0));
  expect(screen.getByText(/not answering/)).toBeInTheDocument();
});

it("does not poll a rented-GPU service while its tab is hidden", async () => {
  render(<GpuPanel active={false} />);
  await act(() => vi.advanceTimersByTimeAsync(10000));
  expect(config).not.toHaveBeenCalled();
});

it("stops polling once told the service is not configured", async () => {
  config.mockResolvedValue({ enabled: false });
  render(<GpuPanel active />);
  await act(() => vi.advanceTimersByTimeAsync(0));
  await act(() => vi.advanceTimersByTimeAsync(30000));
  expect(config).toHaveBeenCalledTimes(1);
});
