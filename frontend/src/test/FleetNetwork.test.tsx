import { render, screen, waitFor, within } from "@testing-library/react";
import userEvent from "@testing-library/user-event";
import { beforeEach, expect, it, vi } from "vitest";
import { FleetNetwork } from "../components/FleetNetwork";
import type { Snapshot, Task, Worker } from "../api/types";

const readSimulation = vi.hoisted(() => vi.fn());
vi.mock("../api/client", () => ({ readSimulation }));

const worker = (id: string, extra: Partial<Worker> = {}): Worker => ({
  id,
  name: id === "a" ? "Studio Mac" : "Lab Box",
  session_id: `${id}-session`,
  capabilities: {
    runtime: "cpu",
    vram_mib: 0,
    kinds: ["echo"],
    machine: { os: "macOS", logical_cores: 10, cpu_model: "Apple M3" },
  },
  state: "alive",
  last_seen: new Date().toISOString(),
  paused: false,
  ...extra,
});
const job = (): Task =>
  ({
    spec: {
      id: "t1",
      job_id: "job-1",
      kind: "simulation_job",
      payload: { value: { label: "Wind tunnel" } },
      requirements: { runtime: "cpu", vram_mib: 0 },
      max_attempts: 1,
      timeout_seconds: null,
      target_worker_id: null,
      allow_failover: true,
    },
    state: "running",
    generation: 1,
    worker_id: "a",
    session_id: null,
    lease_until: null,
    deadline: null,
    result: null,
    failure: "",
    created_at: new Date().toISOString(),
    progress: 0.4,
    started_at: null,
  }) as Task;
const snapshot: Snapshot = {
  workers: [worker("a"), worker("b", { paused: true })],
  tasks: [],
  events: [],
};

beforeEach(() => readSimulation.mockReset());

it("renders nothing at all while its tab is closed", () => {
  const { container } = render(
    <FleetNetwork snapshot={snapshot} active={false} />,
  );
  expect(container).toBeEmptyDOMElement();
  expect(readSimulation).not.toHaveBeenCalled();
});

it("names every machine and the exchange on its line", async () => {
  const user = userEvent.setup();
  render(<FleetNetwork snapshot={snapshot} active />);
  expect(
    screen.getByRole("img", {
      name: "Fleet topology: 2 machines connected to the control plane",
    }),
  ).toBeInTheDocument();
  await user.click(screen.getByRole("button", { name: "Activity log" }));
  const log = screen.getByRole("dialog", { name: "Activity log" });
  const connections = within(log).getByRole("list", { name: "Connections" });
  expect(connections).toHaveTextContent("Control plane → Studio Mac");
  expect(connections).toHaveTextContent("idle · heartbeat");
  expect(connections).toHaveTextContent("Control plane → Lab Box");
  expect(connections).toHaveTextContent("paused · accepting nothing");
});

it("keeps the log popup closed until it is asked for", () => {
  render(<FleetNetwork snapshot={snapshot} active />);
  expect(
    screen.queryByRole("dialog", { name: "Activity log" }),
  ).not.toBeInTheDocument();
  expect(screen.getByRole("button", { name: "Activity log" })).toHaveAttribute(
    "aria-expanded",
    "false",
  );
});

it("says so when no machine has joined", () => {
  render(
    <FleetNetwork snapshot={{ workers: [], tasks: [], events: [] }} active />,
  );
  expect(screen.getByText(/No machines have joined yet/)).toBeInTheDocument();
});

it("keeps the view controls switchable", async () => {
  const user = userEvent.setup();
  render(<FleetNetwork snapshot={snapshot} active />);
  const labels = screen.getByRole("button", { name: "All labels" });
  expect(labels).toHaveAttribute("aria-pressed", "false");
  await user.click(labels);
  expect(labels).toHaveAttribute("aria-pressed", "true");
  const spin = screen.getByRole("button", { name: "Auto-rotate" });
  expect(spin).toHaveAttribute("aria-pressed", "true");
  await user.click(spin);
  expect(spin).toHaveAttribute("aria-pressed", "false");
});

it("reads the subagents of a running job and lists them", async () => {
  readSimulation.mockResolvedValue({
    analysis: {
      rationale: "Split the review",
      children: [
        { id: "c1", role: "dependencies", status: "running", question: "q" },
        { id: "c2", role: "validation", status: "completed" },
      ],
    },
  });
  const user = userEvent.setup();
  render(<FleetNetwork snapshot={{ ...snapshot, tasks: [job()] }} active />);
  await waitFor(() =>
    expect(readSimulation).toHaveBeenCalledWith("job-1", expect.anything()),
  );
  await user.click(screen.getByRole("button", { name: "Activity log" }));
  const log = screen.getByRole("dialog", { name: "Activity log" });
  await waitFor(() =>
    expect(
      within(log).getByRole("list", { name: "Analysis agents" }),
    ).toHaveTextContent("Wind tunnel · dependencies"),
  );
  // Each agent hangs off the machine running the job, not off the control plane.
  expect(
    within(log).getByRole("list", { name: "Connections" }),
  ).toHaveTextContent("Studio Mac → dependencies");
});

it("asks for nothing when no simulation job is running", async () => {
  render(<FleetNetwork snapshot={snapshot} active />);
  await waitFor(() =>
    expect(screen.getByRole("button", { name: "Activity log" })).toBeEnabled(),
  );
  expect(readSimulation).not.toHaveBeenCalled();
});
