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

it("lists every finished run, not an arbitrary first few", async () => {
  // Forty finished jobs: enough to trip any cap sitting between the snapshot and
  // the list, which is what made the run list stop partway with no explanation.
  const runs = Array.from({ length: 40 }, (_, index) => {
    const done = job();
    done.spec.id = `t${index}`;
    done.spec.job_id = `job-${index}`;
    (done.spec.payload as { value: { label: string } }).value.label =
      `run-${index}.py`;
    done.state = "succeeded";
    done.created_at = new Date(Date.now() - index * 60_000).toISOString();
    return done;
  });
  const user = userEvent.setup();
  render(<FleetNetwork snapshot={{ ...snapshot, tasks: runs }} active />);
  await user.click(screen.getByRole("button", { name: "Replay a run" }));
  const panel = screen.getByRole("dialog", { name: "Past runs" });
  const listed = within(panel).getAllByRole("button");
  // Every run, plus the popup's own close button.
  expect(listed).toHaveLength(41);
  expect(within(panel).getByText("run-0.py")).toBeInTheDocument();
  expect(within(panel).getByText("run-39.py")).toBeInTheDocument();
  // And it says how far back the list actually reaches.
  expect(panel).toHaveTextContent("40 finished runs");
});
