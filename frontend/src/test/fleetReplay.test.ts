import { expect, it } from "vitest";
import { buildReplay, replaySnapshot } from "../lib/fleetReplay";
import { buildFleetGraph } from "../lib/fleetGraph";
import { groupJobs } from "../lib/jobs";
import type { ExecutionEvent, Task, Worker } from "../api/types";

const start = Date.parse("2026-09-20T10:00:00Z");
const task = (id: string, extra: Partial<Task> = {}): Task =>
  ({
    spec: {
      id,
      job_id: "job-1",
      kind: "simulation_job",
      payload: { value: { label: `Trial ${id}` } },
      requirements: { runtime: "cpu", vram_mib: 0 },
      max_attempts: 1,
      timeout_seconds: null,
      target_worker_id: null,
      allow_failover: true,
    },
    state: "succeeded",
    generation: 1,
    worker_id: "a",
    session_id: null,
    lease_until: null,
    deadline: null,
    result: null,
    failure: "",
    created_at: new Date(start).toISOString(),
    progress: 1,
    started_at: null,
    ...extra,
  }) as Task;
const event = (
  taskId: string,
  kind: ExecutionEvent["kind"],
  offset: number,
  data: Record<string, unknown> = {},
  worker = "a",
): ExecutionEvent =>
  ({
    id: offset,
    execution_id: "e1",
    task_id: taskId,
    attempt: 1,
    worker_id: worker,
    source: "worker",
    sequence: offset,
    kind,
    occurred_at: new Date(start + offset * 1000).toISOString(),
    received_at: new Date(start + offset * 1000).toISOString(),
    data,
  }) as ExecutionEvent;

it("folds the execution log into the state of the run at each moment", () => {
  const group = groupJobs([
    task("t1"),
    task("t2", { worker_id: "b", state: "failed" }),
  ])[0];
  const replay = buildReplay(group, [
    event("t1", "started", 1),
    event("t2", "started", 2, {}, "b"),
    event("t1", "progress", 3, { progress: 0.5 }),
    event("t1", "succeeded", 4),
    event("t2", "failed", 5, { error: "out of memory" }, "b"),
  ]);
  expect(replay.workers).toEqual(["a", "b"]);
  // Opens with everything queued, before any machine has it.
  expect(replay.steps[0].states.t1).toMatchObject({
    state: "queued",
    worker: null,
  });
  expect(replay.steps[0].caption).toContain("queued · 2 tasks");
  expect(replay.steps[1].states.t1).toMatchObject({
    state: "running",
    worker: "a",
  });
  // t2 starting does not disturb t1.
  expect(replay.steps[2].states.t1.state).toBe("running");
  expect(replay.steps[3].states.t1.progress).toBe(0.5);
  expect(replay.steps[4].states.t1.state).toBe("succeeded");
  expect(replay.steps[5].caption).toContain("out of memory");
  expect(replay.steps[replay.steps.length - 1].caption).toContain(
    "1 of 2 tasks succeeded",
  );
});

it("still replays a run whose machines logged nothing", () => {
  const group = groupJobs([task("t1")])[0];
  const replay = buildReplay(group, []);
  expect(replay.steps).toHaveLength(2);
  // Start and recorded outcome: thin, but honest and still watchable.
  expect(replay.steps[0].states.t1.state).toBe("queued");
  expect(replay.steps[1].states.t1).toMatchObject({
    state: "succeeded",
    worker: "a",
  });
  expect(replay.steps[1].at).toBeGreaterThan(replay.steps[0].at);
});

it("rebuilds a moment as a snapshot the graph can draw", () => {
  const group = groupJobs([task("t1")])[0];
  const replay = buildReplay(group, [
    event("t1", "started", 1),
    event("t1", "succeeded", 9),
  ]);
  const fleet: Worker[] = [
    {
      id: "a",
      name: "Studio Mac",
      session_id: "s",
      capabilities: { runtime: "cpu", vram_mib: 0, kinds: [] },
      state: "offline",
      last_seen: new Date(start).toISOString(),
      paused: true,
    },
  ];
  const mid = replaySnapshot(replay, replay.steps[1], group, fleet);
  // The machine is drawn as it was during the run, not as it is now.
  expect(mid.workers[0]).toMatchObject({ state: "alive", paused: false });
  expect(mid.tasks[0]).toMatchObject({ state: "running", worker_id: "a" });
  const graph = buildFleetGraph(mid, replay.steps[1].at);
  expect(graph.links.find((link) => link.kind === "control")).toMatchObject({
    state: "busy",
    label: "running · Trial t1",
  });
  // Only the machines that took part appear.
  expect(graph.nodes.filter((node) => node.kind === "worker")).toHaveLength(1);
});
