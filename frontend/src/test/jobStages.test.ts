import { expect, it } from "vitest";
import { stageTasks } from "../lib/jobStages";
import { buildFleetGraph } from "../lib/fleetGraph";
import { buildReplay, replaySnapshot } from "../lib/fleetReplay";
import { groupJobs } from "../lib/jobs";
import type { SimulationStatus } from "../api/client";
import type { Snapshot, Task, Worker } from "../api/types";

const now = Date.parse("2026-09-20T12:00:00Z");
const worker = (id: string): Worker => ({
  id,
  name: id,
  session_id: `${id}-s`,
  capabilities: { runtime: "cpu", vram_mib: 0, kinds: ["python_project"] },
  state: "alive",
  last_seen: new Date(now - 1000).toISOString(),
  paused: false,
});

/** What the dashboard snapshot actually holds for a simulation: the root, no machine. */
const root = (state: Task["state"] = "running"): Task =>
  ({
    spec: {
      id: "sim-1",
      job_id: "sim-1",
      kind: "simulation_job",
      payload: { value: { label: "calibrate.py" } },
      requirements: { runtime: "cpu", vram_mib: 0 },
      max_attempts: 1,
      timeout_seconds: null,
      target_worker_id: null,
      allow_failover: true,
    },
    state,
    generation: 1,
    worker_id: null,
    session_id: null,
    lease_until: null,
    deadline: null,
    result: null,
    failure: "",
    created_at: new Date(now - 60_000).toISOString(),
    progress: 0,
    started_at: null,
  }) as Task;

const status = {
  tasks: [
    {
      id: "t-prof",
      role: "profiling",
      state: "succeeded",
      worker_id: "pyworker-1",
      generation: 1,
      failure: "",
      progress: 1,
    },
    {
      id: "t-b0",
      role: "batch-0",
      state: "running",
      worker_id: "pyworker-1",
      generation: 1,
      failure: "",
      progress: 0.4,
    },
    {
      id: "t-b60",
      role: "batch-60",
      state: "running",
      worker_id: "pyworker-2",
      generation: 1,
      failure: "",
      progress: 0.2,
    },
    {
      id: "t-b120",
      role: "batch-120",
      state: "queued",
      worker_id: null,
      generation: 1,
      failure: "",
      progress: 0,
    },
  ],
} as unknown as SimulationStatus;

const snapshot = (tasks: Task[]): Snapshot => ({
  workers: [worker("pyworker-1"), worker("pyworker-2")],
  tasks,
  events: [],
});

it("puts a simulation job on the machines its stages are running on", () => {
  // Without the stages the snapshot says only that a job exists somewhere, and the
  // job is drawn hanging off the control plane with nothing running it.
  const bare = buildFleetGraph(snapshot([root()]), now);
  // Its only line runs back to the control plane; no machine is named.
  expect(
    bare.links.filter(
      (link) => link.kind === "job" && link.source === "job:sim-1",
    ),
  ).toHaveLength(0);

  const stages = stageTasks(status, root(), "sim-1");
  const graph = buildFleetGraph(snapshot([root(), ...stages]), now);
  const routes = graph.links.filter(
    (link) => link.kind === "job" && link.source === "job:sim-1",
  );
  expect(routes.map((link) => link.target).sort()).toEqual([
    "pyworker-1",
    "pyworker-2",
  ]);
  // The job keeps its own name; the stages say where it is.
  expect(graph.nodes.find((node) => node.kind === "job")?.label).toBe(
    "calibrate.py",
  );
  // And each machine's own line names the stage it is running.
  expect(
    graph.links.find(
      (link) => link.kind === "control" && link.target === "pyworker-2",
    )?.label,
  ).toBe("running · batch-60 · 20%");
});

it("replays a finished run across the machines that actually ran it", () => {
  const finished = {
    tasks: status.tasks.map((stage) => ({
      ...stage,
      state: "succeeded",
      worker_id: stage.worker_id || "pyworker-3",
    })),
  } as unknown as SimulationStatus;
  const stages = stageTasks(finished, root("succeeded"), "sim-1");
  const group = groupJobs([root("succeeded"), ...stages])[0];
  const replay = buildReplay(group, []);
  // Every machine that took a stage is part of the replay, from the run's own record.
  expect(replay.workers.sort()).toEqual([
    "pyworker-1",
    "pyworker-2",
    "pyworker-3",
  ]);
  const last = replay.steps[replay.steps.length - 1];
  const frame = replaySnapshot(replay, last, group, [
    worker("pyworker-1"),
    worker("pyworker-2"),
  ]);
  expect(frame.workers.map((w) => w.id).sort()).toEqual([
    "pyworker-1",
    "pyworker-2",
    "pyworker-3",
  ]);
  const graph = buildFleetGraph(frame, last.at);
  expect(graph.nodes.filter((node) => node.kind === "worker").length).toBe(3);
});

it("keeps a finished stage on its machine while the run is still going", () => {
  const stages = stageTasks(status, root(), "sim-1");
  const graph = buildFleetGraph(snapshot([root(), ...stages]), now);
  // pyworker-1 holds a finished profiling stage and a running batch: the live one
  // names the line, and the finished one has not taken the machine off the picture.
  const one = graph.links.find(
    (link) => link.kind === "job" && link.target === "pyworker-1",
  );
  expect(one?.label).toBe("running on · pyworker-1");

  const settling = stageTasks(
    {
      tasks: status.tasks.map((stage) =>
        stage.worker_id === "pyworker-2"
          ? { ...stage, state: "succeeded" }
          : stage,
      ),
    } as unknown as SimulationStatus,
    root(),
    "sim-1",
  );
  const later = buildFleetGraph(snapshot([root(), ...settling]), now);
  const two = later.links.find(
    (link) => link.kind === "job" && link.target === "pyworker-2",
  );
  expect(two).toMatchObject({
    state: "result",
    label: "finished on · pyworker-2",
    intensity: 0,
  });
});

it("leaves machines that went offline long ago out of the picture", () => {
  const gone: Worker = {
    ...worker("retired"),
    state: "offline",
    last_seen: new Date(now - 36_000_000).toISOString(),
  };
  const recent: Worker = {
    ...worker("just-dropped"),
    state: "offline",
    last_seen: new Date(now - 30_000).toISOString(),
  };
  const graph = buildFleetGraph(
    { workers: [gone, recent, worker("live")], tasks: [], events: [] },
    now,
  );
  const drawn = graph.nodes.filter((node) => node.kind === "worker");
  // A machine that dropped a moment ago is news; one that left yesterday is not.
  expect(drawn.map((node) => node.id).sort()).toEqual(["just-dropped", "live"]);
});
