import { render, screen } from "@testing-library/react";
import userEvent from "@testing-library/user-event";
import { beforeEach, expect, it, vi } from "vitest";
import { WorkerGrid } from "../components/WorkerGrid";
import type { Accelerator, Task, Worker } from "../api/types";

const setRuntimePreference = vi.fn();
vi.mock("../api/client", () => ({
  setRuntimePreference: (...args: unknown[]) => setRuntimePreference(...args),
}));

beforeEach(() => setRuntimePreference.mockReset().mockResolvedValue({}));

it("reserves a worker for an idle service and marks it ready after stop", () => {
  const w = worker("metal", { available: true, reason: "", device: "Apple GPU" });
  const attempt = { spec: { id: "attempt", job_id: "model", kind: "python_service", payload: {} },
    state: "running", worker_id: w.id, progress: 0 } as Task;
  const root = { ...attempt, worker_id: null, spec: { ...attempt.spec, id: "model", kind: "simulation_job", payload: { value: { label: "Hosted model" } } } } as Task;
  const view = render(<WorkerGrid workers={[w]} tasks={[root, attempt]} selected="" onSelect={() => {}} />);
  expect(screen.getByText("Busy · serving")).toBeInTheDocument();
  expect(screen.getByText("Hosted model")).toBeInTheDocument();
  expect(screen.getByText("Slot reserved")).toBeInTheDocument();
  expect(screen.queryByText("0%")).not.toBeInTheDocument();
  view.rerender(<WorkerGrid workers={[w]} tasks={[{ ...attempt, state: "cancelled" }]} selected="" onSelect={() => {}} />);
  expect(screen.getByText("Ready")).toBeInTheDocument();
  expect(screen.queryByText("Busy · serving")).not.toBeInTheDocument();
});

it("reports local Python devices without offering unsupported remote settings", () => {
  const w = worker("python", { available: true, reason: "", device: "Apple GPU" });
  w.capabilities.machine = { runtime_control: "startup" };
  show(w);
  expect(screen.getByText("Apple GPU")).toBeInTheDocument();
  expect(screen.queryByRole("button", { name: "CPU" })).not.toBeInTheDocument();
  expect(screen.queryByRole("button", { name: "GPU" })).not.toBeInTheDocument();
  expect(screen.queryByText(/predates/)).not.toBeInTheDocument();
});

function worker(
  id: string,
  accelerator?: Accelerator | null,
  runtime_preference: "auto" | "cpu" = "auto",
): Worker {
  return {
    id,
    session_id: "s1",
    capabilities: {
      runtime: accelerator?.available ? "cuda" : "cpu",
      vram_mib: accelerator?.available ? 8192 : 0,
      kinds: ["echo"],
      machine: { logical_cores: 8 },
      ...(accelerator === undefined ? {} : { accelerator }),
      runtime_preference,
    },
    state: "alive",
    last_seen: new Date().toISOString(),
    paused: false,
  } as Worker;
}

function show(w: Worker) {
  render(
    <WorkerGrid workers={[w]} tasks={[]} selected="" onSelect={() => {}} />,
  );
}

it("refuses to offer GPU on a machine that reported no device", async () => {
  show(
    worker("m1", {
      available: false,
      reason: "no inference runtime in this image (build --target ml)",
    }),
  );
  expect(screen.getByRole("button", { name: "GPU" })).toBeDisabled();
  // The machine's own words, so "why is this greyed out" is answered in place.
  expect(screen.getByText(/no inference runtime in this image/)).toBeTruthy();
  await userEvent.click(screen.getByRole("button", { name: "GPU" }));
  expect(setRuntimePreference).not.toHaveBeenCalled();
});

it("distinguishes a machine with no device from one that never said", async () => {
  // Absent is not false. One means "it looked and found nothing"; the other means the
  // agent predates the field, and the fix is a newer image rather than new hardware.
  show(worker("m2", undefined));
  expect(screen.getByRole("button", { name: "GPU" })).toBeDisabled();
  expect(screen.getByText(/has not reported its devices/)).toBeTruthy();
});

it("offers GPU where the machine reported one, and names it", async () => {
  // Starts on cpu, so clicking GPU is a real change rather than a no-op.
  show(
    worker(
      "m3",
      { available: true, reason: "", device: "NVIDIA GeForce RTX 4060" },
      "cpu",
    ),
  );
  const gpu = screen.getByRole("button", { name: "GPU" });
  expect(gpu).not.toBeDisabled();
  expect(screen.getByText("NVIDIA GeForce RTX 4060")).toBeTruthy();
  await userEvent.click(gpu);
  expect(setRuntimePreference).toHaveBeenCalledWith("m3", "auto");
});

it("sends cpu when CPU is chosen on a machine that has a device", async () => {
  show(worker("m4", { available: true, reason: "", device: "RTX 4060" }));
  await userEvent.click(screen.getByRole("button", { name: "CPU" }));
  expect(setRuntimePreference).toHaveBeenCalledWith("m4", "cpu");
});

it("does not re-send the setting the machine is already on", async () => {
  show(worker("m5", { available: true, reason: "" }, "cpu"));
  await userEvent.click(screen.getByRole("button", { name: "CPU" }));
  expect(setRuntimePreference).not.toHaveBeenCalled();
});

it("keeps the card selectable without the toggle stealing the click", async () => {
  const onSelect = vi.fn();
  render(
    <WorkerGrid
      workers={[worker("m6", { available: true, reason: "" })]}
      tasks={[]}
      selected=""
      onSelect={onSelect}
    />,
  );
  await userEvent.click(screen.getByRole("button", { name: "CPU" }));
  // The control sits inside the card; without stopPropagation, setting the runtime
  // would also retarget the operator's next task to this machine.
  expect(onSelect).not.toHaveBeenCalled();
  await userEvent.click(screen.getByRole("button", { name: /Send to/ }));
  expect(onSelect).toHaveBeenCalledWith("m6");
});
