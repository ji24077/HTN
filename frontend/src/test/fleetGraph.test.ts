import { expect, it } from "vitest";
import { buildFleetGraph, CONTROL_ID } from "../lib/fleetGraph";
import {
  DEFAULT_CAMERA,
  project,
  reconcileBodies,
  screenDelta,
  stepLayout,
  type Body,
} from "../lib/fleetScene";
import type { Snapshot, Task, Worker } from "../api/types";

const now = Date.parse("2026-09-20T12:00:00Z");
const worker = (id: string, extra: Partial<Worker> = {}): Worker => ({
  id,
  name: id.toUpperCase(),
  session_id: `${id}-session`,
  capabilities: {
    runtime: "cpu",
    vram_mib: 0,
    kinds: ["echo"],
    machine: { os: "macOS", logical_cores: 10, cpu_model: "Apple M3" },
  },
  state: "alive",
  last_seen: new Date(now - 1000).toISOString(),
  paused: false,
  ...extra,
});
const task = (id: string, extra: Partial<Task> = {}): Task =>
  ({
    spec: {
      id,
      job_id: "job-1",
      kind: "echo",
      payload: { value: { label: `Task ${id}` } },
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
    created_at: new Date(now).toISOString(),
    progress: 0.5,
    started_at: null,
    ...extra,
  }) as Task;
const snapshot = (over: Partial<Snapshot> = {}): Snapshot => ({
  workers: [],
  tasks: [],
  events: [],
  ...over,
});

it("links every machine to the control plane and names the exchange", () => {
  const graph = buildFleetGraph(
    snapshot({
      workers: [worker("a"), worker("b")],
      tasks: [task("t1")],
    }),
    now,
  );
  expect(graph.nodes.map((node) => node.id)).toEqual([
    CONTROL_ID,
    "a",
    "b",
    "job:job-1",
  ]);
  const busy = graph.links.find((link) => link.target === "a");
  expect(busy?.state).toBe("busy");
  expect(busy?.label).toBe("running · Task t1 · 50%");
  // Progress streams back up the wire, so the packets travel toward the control plane.
  expect(busy?.flow).toBe(-1);
  const quiet = graph.links.find((link) => link.target === "b");
  expect(quiet).toMatchObject({
    state: "idle",
    label: "idle · heartbeat",
    flow: 0,
  });
});

it("reads dispatch, pause, and lost heartbeats off the fleet state", () => {
  const graph = buildFleetGraph(
    snapshot({
      workers: [
        worker("a"),
        worker("b", { paused: true }),
        worker("c", { last_seen: new Date(now - 60_000).toISOString() }),
        worker("d", { state: "offline" }),
      ],
      tasks: [task("t1", { state: "assigned" })],
    }),
    now,
  );
  const state = (id: string) =>
    graph.links.find((link) => link.target === id)?.state;
  expect(graph.links.find((link) => link.target === "a")).toMatchObject({
    state: "busy",
    label: "dispatching · Task t1",
    flow: 1,
  });
  expect(state("b")).toBe("paused");
  expect(state("c")).toBe("offline");
  expect(state("d")).toBe("offline");
  expect(graph.nodes.find((node) => node.id === "c")?.state).toBe("unhealthy");
});

it("carries no traffic on a line with nothing running on it", () => {
  const done = task("t1", { state: "succeeded" });
  const graph = buildFleetGraph(
    snapshot({ workers: [worker("a")], tasks: [done] }),
    now,
  );
  // A finished job is not traffic. The line stays up and stays empty.
  expect(graph.links[0]).toMatchObject({
    state: "idle",
    label: "idle · heartbeat",
    intensity: 0,
  });
  const paused = buildFleetGraph(
    snapshot({ workers: [worker("a", { paused: true })] }),
    now,
  );
  expect(paused.links[0].intensity).toBe(0);
});

it("only pairs machines that are both running the job right now", () => {
  const assigned = buildFleetGraph(
    snapshot({
      workers: [worker("a"), worker("b")],
      tasks: [task("t1"), task("t2", { worker_id: "b", state: "assigned" })],
    }),
    now,
  );
  expect(assigned.links.filter((link) => link.kind === "peer")).toHaveLength(0);
});

it("draws one peer line between machines sharing a job, and none otherwise", () => {
  const shared = buildFleetGraph(
    snapshot({
      workers: [worker("a"), worker("b")],
      tasks: [task("t1"), task("t2", { worker_id: "b" })],
    }),
    now,
  );
  const peers = shared.links.filter((link) => link.kind === "peer");
  expect(peers).toHaveLength(1);
  expect(peers[0].label).toBe("co-running · Task t1");
  const alone = buildFleetGraph(
    snapshot({ workers: [worker("a"), worker("b")], tasks: [task("t1")] }),
    now,
  );
  expect(alone.links.filter((link) => link.kind === "peer")).toHaveLength(0);
});

it("seats new machines away from the origin and drops ones that leave", () => {
  const bodies = new Map<string, Body>();
  const nodes = [
    { id: CONTROL_ID, kind: "control" as const },
    { id: "a", kind: "worker" as const },
    { id: "b", kind: "worker" as const },
  ];
  reconcileBodies(bodies, nodes);
  expect([...bodies.keys()]).toEqual([CONTROL_ID, "a", "b"]);
  expect(bodies.get(CONTROL_ID)?.pinned).toBe(true);
  const seated = bodies.get("a")!;
  expect(Math.hypot(seated.x, seated.y, seated.z)).toBeGreaterThan(10);
  const moved = { ...seated };
  reconcileBodies(bodies, nodes.slice(0, 2));
  expect([...bodies.keys()]).toEqual([CONTROL_ID, "a"]);
  // A machine that stayed keeps the place it already had.
  expect(bodies.get("a")).toMatchObject({ x: moved.x, z: moved.z });
});

it("settles the layout around a pinned control plane", () => {
  const bodies = reconcileBodies(new Map<string, Body>(), [
    { id: CONTROL_ID, kind: "control" },
    { id: "a", kind: "worker" },
    { id: "b", kind: "worker" },
    { id: "c", kind: "worker" },
  ]);
  const links = ["a", "b", "c"].map((id) => ({
    source: CONTROL_ID,
    target: id,
    kind: "control" as const,
  }));
  let motion = 0;
  for (let step = 0; step < 600; step += 1)
    motion = stepLayout(bodies, links, 1 / 60);
  expect(motion).toBeLessThan(1);
  expect(bodies.get(CONTROL_ID)).toMatchObject({ x: 0, y: 0, z: 0 });
  for (const id of ["a", "b", "c"]) {
    const body = bodies.get(id)!;
    const radius = Math.hypot(body.x, body.y, body.z);
    expect(radius).toBeGreaterThan(120);
    expect(radius).toBeLessThan(700);
    expect(Number.isFinite(radius)).toBe(true);
  }
});

it("projects depth into perspective and puts the origin at the centre", () => {
  const centre = project({ x: 0, y: 0, z: 0 }, DEFAULT_CAMERA, 800, 600);
  expect(centre).toMatchObject({ x: 400, y: 300 });
  const near = project(
    { x: 60, y: 0, z: 0 },
    { yaw: 0, pitch: 0, zoom: 1 },
    800,
    600,
  );
  const far = project(
    { x: 60, y: 0, z: 300 },
    { yaw: 0, pitch: 0, zoom: 1 },
    800,
    600,
  );
  // The further point is drawn smaller and pulled toward the vanishing point.
  expect(far.scale).toBeLessThan(near.scale);
  expect(far.x - 400).toBeLessThan(near.x - 400);
  expect(far.depth).toBeGreaterThan(near.depth);
});

it("hangs each subagent off the machine running its job", () => {
  const graph = buildFleetGraph(
    snapshot({ workers: [worker("a"), worker("b")], tasks: [task("t1")] }),
    now,
    [
      {
        jobId: "job-1",
        title: "Wind tunnel",
        workers: ["a"],
        children: [
          { id: "c1", role: "dependencies", status: "running" },
          { id: "c2", role: "validation", status: "completed", summary: "ok" },
          { id: "c3", role: "parallelization", status: "queued" },
        ],
      },
    ],
  );
  // Only the agent actually working is drawn; the finished and queued ones would
  // crowd the machine without saying anything about what is happening now.
  const drawn = graph.nodes.filter((node) => node.kind === "agent");
  expect(drawn.map((node) => node.label)).toEqual(["dependencies"]);
  const lines = graph.links.filter((link) => link.kind === "agent");
  // Attached to the machine, not to the control plane.
  expect(lines).toHaveLength(1);
  expect(lines[0]).toMatchObject({
    source: "a",
    label: "analysing · dependencies",
    state: "busy",
  });
  expect(lines[0].intensity).toBeGreaterThan(0);
});

it("falls back to the control plane for a job no machine has started", () => {
  const graph = buildFleetGraph(snapshot({ workers: [worker("a")] }), now, [
    {
      jobId: "job-9",
      title: "Queued project",
      workers: [],
      children: [{ id: "c1", role: "dependencies", status: "running" }],
    },
  ]);
  expect(graph.links.find((link) => link.kind === "agent")?.source).toBe(
    CONTROL_ID,
  );
});

it("turns a screen drag into the world move that produces it", () => {
  const camera = { yaw: 0.6, pitch: -0.32, zoom: 1 };
  const origin = { x: 40, y: -12, z: 25 };
  const move = screenDelta(30, -18, camera, 1);
  const before = project(origin, camera, 800, 600);
  const after = project(
    { x: origin.x + move.x, y: origin.y + move.y, z: origin.z + move.z },
    camera,
    800,
    600,
  );
  // The node follows the pointer, however the scene happens to be turned.
  expect(after.x - before.x).toBeCloseTo(30, 0);
  expect(after.y - before.y).toBeCloseTo(-18, 0);
});

it("shows a launched job and the machine it is heading for", () => {
  const graph = buildFleetGraph(
    snapshot({
      workers: [worker("a"), worker("b")],
      tasks: [
        task("t1", {
          state: "queued",
          worker_id: null,
          spec: { ...task("t1").spec, target_worker_id: "b" },
        }),
      ],
    }),
    now,
  );
  const node = graph.nodes.find((item) => item.kind === "job");
  expect(node).toMatchObject({ label: "Task t1", state: "paused" });
  const route = graph.links.find((link) => link.kind === "job");
  // Queued for a named machine: the route is drawn before anything moves along it.
  expect(route).toMatchObject({
    source: "job:job-1",
    target: "b",
    state: "idle",
    label: "queued for · B",
    intensity: 0,
  });
});

it("hangs an untargeted job off the control plane with what it needs", () => {
  const queued = task("t1", { state: "queued", worker_id: null });
  queued.spec.requirements = { runtime: "cuda", vram_mib: 8192 };
  const graph = buildFleetGraph(
    snapshot({ workers: [worker("a")], tasks: [queued] }),
    now,
  );
  expect(graph.links.find((link) => link.kind === "job")).toMatchObject({
    source: CONTROL_ID,
    target: "job:job-1",
    label: "1 queued · needs CUDA GPU · 8 GiB VRAM",
  });
});

it("turns the route solid once the job is executing on a machine", () => {
  const graph = buildFleetGraph(
    snapshot({ workers: [worker("a")], tasks: [task("t1")] }),
    now,
  );
  const route = graph.links.find((link) => link.kind === "job");
  expect(route).toMatchObject({
    target: "a",
    state: "busy",
    label: "running on · A",
  });
  // The job is placed, so nothing is left waiting on the control plane.
  expect(
    graph.links.filter(
      (link) => link.kind === "job" && link.source === CONTROL_ID,
    ),
  ).toHaveLength(0);
});

it("keeps a finished job on the machine that ran it, then lets it go", () => {
  const done = task("t1", { state: "succeeded", worker_id: "b" });
  const ended = (at: number) => ({
    id: 1,
    at: new Date(at).toISOString(),
    entity: "task",
    entity_id: "t1",
    previous_state: "running",
    new_state: "succeeded",
    details: {},
  });
  const fresh = buildFleetGraph(
    snapshot({
      workers: [worker("a"), worker("b")],
      tasks: [done],
      events: [ended(now - 5_000)],
    }),
    now,
  );
  const route = fresh.links.find((link) => link.kind === "job");
  // It answers the only question left: where did that run?
  expect(route).toMatchObject({
    source: "job:job-1",
    target: "b",
    state: "result",
    label: "succeeded on · B",
    intensity: 0,
  });
  expect(fresh.nodes.find((node) => node.kind === "job")?.state).toBe(
    "offline",
  );
  const later = buildFleetGraph(
    snapshot({
      workers: [worker("a"), worker("b")],
      tasks: [done],
      events: [ended(now - 120_000)],
    }),
    now,
  );
  expect(later.nodes.filter((node) => node.kind === "job")).toHaveLength(0);
});
