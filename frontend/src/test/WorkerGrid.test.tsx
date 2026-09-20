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
  const w = worker("metal", {
    available: true,
    reason: "",
    device: "Apple GPU",
  });
  const attempt = {
    spec: {
      id: "attempt",
      job_id: "model",
      kind: "python_service",
      payload: {},
    },
    state: "running",
    worker_id: w.id,
    progress: 0,
  } as Task;
  const root = {
    ...attempt,
    worker_id: null,
    spec: {
      ...attempt.spec,
      id: "model",
      kind: "simulation_job",
      payload: { value: { label: "Hosted model" } },
    },
  } as Task;
  const view = render(
    <WorkerGrid
      workers={[w]}
      tasks={[root, attempt]}
      selected=""
      onSelect={() => {}}
    />,
  );
  expect(screen.getByText("Busy · serving")).toBeInTheDocument();
  expect(screen.getByText("Hosted model")).toBeInTheDocument();
  expect(screen.getByText("Slot reserved")).toBeInTheDocument();
  expect(screen.queryByText("0%")).not.toBeInTheDocument();
  view.rerender(
    <WorkerGrid
      workers={[w]}
      tasks={[{ ...attempt, state: "cancelled" }]}
      selected=""
      onSelect={() => {}}
    />,
  );
  expect(screen.getByText("Ready")).toBeInTheDocument();
  expect(screen.queryByText("Busy · serving")).not.toBeInTheDocument();
});

it("reports local Python devices without offering unsupported remote settings", () => {
  const w = worker("python", {
    available: true,
    reason: "",
    device: "Apple GPU",
  });
  w.capabilities.machine = { runtime_control: "startup" };
  show(w);
  expect(screen.getByText("Apple GPU")).toBeInTheDocument();
  expect(
    screen.queryByRole("button", { name: "CPU only" }),
  ).not.toBeInTheDocument();
  expect(
    screen.queryByRole("button", { name: "Automatic" }),
  ).not.toBeInTheDocument();
  expect(screen.queryByText(/predates/)).not.toBeInTheDocument();
});

function worker(
  id: string,
  accelerator?: Accelerator | null,
  runtime_preference: "auto" | "cpu" = "auto",
  name?: string,
): Worker {
  return {
    id,
    ...(name === undefined ? {} : { name }),
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
  expect(
    screen.queryByRole("button", { name: "Automatic" }),
  ).not.toBeInTheDocument();
  expect(
    screen.queryByRole("button", { name: "CPU only" }),
  ).not.toBeInTheDocument();
  expect(
    screen.getByText("GPU execution is unavailable on this worker."),
  ).toBeInTheDocument();
  expect(setRuntimePreference).not.toHaveBeenCalled();
});

it("distinguishes a machine with no device from one that never said", async () => {
  // An absent capability report must not be presented as a confirmed lack of GPU.
  show(worker("m2", undefined));
  expect(
    screen.queryByRole("button", { name: "Automatic" }),
  ).not.toBeInTheDocument();
  expect(
    screen.getByText("GPU availability has not been reported."),
  ).toBeInTheDocument();
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
  const gpu = screen.getByRole("button", { name: "Automatic" });
  expect(gpu).not.toBeDisabled();
  expect(screen.getByText("NVIDIA GeForce RTX 4060")).toBeTruthy();
  await userEvent.click(gpu);
  expect(setRuntimePreference).toHaveBeenCalledWith("m3", "auto");
});

it("sends cpu when CPU is chosen on a machine that has a device", async () => {
  show(worker("m4", { available: true, reason: "", device: "RTX 4060" }));
  await userEvent.click(screen.getByRole("button", { name: "CPU only" }));
  expect(setRuntimePreference).toHaveBeenCalledWith("m4", "cpu");
});

it("does not re-send the setting the machine is already on", async () => {
  show(worker("m5", { available: true, reason: "" }, "cpu"));
  await userEvent.click(screen.getByRole("button", { name: "CPU only" }));
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
  await userEvent.click(screen.getByRole("button", { name: "CPU only" }));
  // The control sits inside the card; without stopPropagation, setting the runtime
  // would also retarget the operator's next task to this machine.
  expect(onSelect).not.toHaveBeenCalled();
  await userEvent.click(screen.getByRole("button", { name: /View details/ }));
  expect(onSelect).toHaveBeenCalledWith("m6");
});

it("never selects Automatic as a confirmed policy when capabilities were not reported", () => {
  const w = worker("legacy");
  delete w.capabilities.runtime_preference;
  show(w);
  expect(
    screen.queryByRole("button", { name: "Automatic" }),
  ).not.toBeInTheDocument();
  expect(
    screen.queryByRole("button", { name: "CPU only" }),
  ).not.toBeInTheDocument();
});

it("keeps offline runtime settings read-only", async () => {
  show({
    ...worker("offline", { available: true, reason: "", device: "GPU" }),
    state: "offline",
  });
  await userEvent.click(
    screen.getByRole("button", { name: /Show offline workers/ }),
  );
  expect(screen.getByRole("button", { name: "CPU only" })).toBeDisabled();
  expect(screen.getByRole("button", { name: "Automatic" })).toBeDisabled();
});

it("keyboard activation of a policy never inspects or submits a worker", async () => {
  const onSelect = vi.fn();
  render(
    <WorkerGrid
      workers={[worker("gpu", { available: true, reason: "" })]}
      tasks={[]}
      selected=""
      onSelect={onSelect}
    />,
  );
  screen.getByRole("button", { name: "CPU only" }).focus();
  await userEvent.keyboard("{Enter}");
  expect(setRuntimePreference).toHaveBeenCalledWith("gpu", "cpu");
  expect(onSelect).not.toHaveBeenCalled();
});

it("heads the card with the machine's name, not its id", () => {
  const id = "6f31f4bc-7a83-4877-b7f1-2bdde3b4b830";
  show(worker(id, null, "auto", "ethans-desktop"));
  expect(screen.getByRole("heading", { name: "ethans-desktop" })).toBeTruthy();
  // Identifiers remain in machine details rather than crowding the card.
  expect(screen.queryByText(id)).toBeNull();
  // Screen reader users choose a destination by name too.
  expect(
    screen.getByRole("button", { name: "View details for ethans-desktop" }),
  ).toBeTruthy();
});

it("falls back to the id for a machine nobody named", () => {
  // A worker that joined with a shared token has no device row and so no name. Showing
  // the id is honest; inventing a name is not.
  show(worker("token-worker-1"));
  expect(screen.getByRole("heading", { name: "token-worker-1" })).toBeTruthy();
});

it("orders the grid by name, with ready machines first", async () => {
  render(
    <WorkerGrid
      workers={[
        { ...worker("id-c", null, "auto", "zulu") },
        { ...worker("id-a", null, "auto", "alpha"), state: "offline" },
        { ...worker("id-b", null, "auto", "mike") },
      ]}
      tasks={[]}
      selected=""
      onSelect={() => {}}
    />,
  );
  await userEvent.click(
    screen.getByRole("button", { name: /Show offline workers/ }),
  );
  const headings = screen
    .getAllByRole("heading", { level: 3 })
    .map((node) => node.textContent);
  // "alpha" sorts first alphabetically but is offline, so it goes last.
  expect(headings).toEqual(["mike", "zulu", "alpha"]);
});
