/**
 * The fleet, expressed as nodes and the relationships between them.
 *
 * Pure derivation from the snapshot the dashboard already receives, plus whatever
 * analysis subagents the caller has read: no requests, no state, no canvas.
 *
 * One rule runs through all of it: a line only carries traffic when work is moving
 * across it at this instant. A machine that is connected but has nothing to do gets a
 * quiet line, because a five-second liveness ping is not a job being sent anywhere.
 */
import type {
  AuditEvent,
  Snapshot,
  Task,
  TaskSpec,
  Worker,
} from "../api/types";
import { taskTitle, workerName } from "./format";
import { groupJobs } from "./jobs";

/** The orchestrator itself. Not a worker, so it cannot collide with a worker id. */
export const CONTROL_ID = "control-plane";

/** Matches `healthy` in lib/format, but against the clock this build was given, so a
 *  graph can be derived for any instant rather than only for right now. */
const HEARTBEAT_MS = 17_000;

export type NodeState = "online" | "paused" | "unhealthy" | "offline";
export type LinkState = "busy" | "result" | "idle" | "paused" | "offline";
export type NodeKind = "control" | "worker" | "agent" | "job";

/** Active jobs drawn at once. More than this and the picture stops being readable. */
const MAX_JOBS = 8;
/** How long a finished job stays on screen, attached to the machine that ran it.
 *  A job that blinked out the moment it completed left no answer to the only
 *  question worth asking afterwards: where did that run? */
const LINGER_MS = 45_000;
/** A machine that went offline yesterday is not part of what is happening now. It
 *  stays on the Workers tab, which is the place for the roll of everything ever
 *  paired; here it would be one more grey dot between the reader and the live fleet. */
const STALE_MS = 600_000;

/** One analysis subagent working under a job. */
export interface AgentChild {
  id: string;
  role: string;
  status: string;
  question?: string;
  summary?: string;
}

/** A job's subagents, and the machines that job is running on. */
export interface AgentGroup {
  jobId: string;
  title: string;
  rationale?: string;
  children: AgentChild[];
  workers: string[];
}

export interface GraphNode {
  id: string;
  kind: NodeKind;
  label: string;
  /** One line of hardware or context, shown when the node is in focus. */
  detail: string;
  state: NodeState;
  /** Work in hand right now. Drives node size and the pulse ring. */
  load: number;
}

export interface GraphLink {
  id: string;
  source: string;
  target: string;
  /** A machine's line to the orchestrator, two machines on one job, or an agent. */
  kind: "control" | "peer" | "agent" | "job";
  state: LinkState;
  /** What the two ends are exchanging, printed on the line. */
  label: string;
  /** 1 source→target, -1 target→source, 0 both ways. */
  flow: -1 | 0 | 1;
  /** 0 nothing moving … 1 saturated. Zero means no packets are drawn at all. */
  intensity: number;
}

export interface FleetGraph {
  nodes: GraphNode[];
  links: GraphLink[];
}

function nodeState(worker: Worker, now: number): NodeState {
  if (worker.state === "offline") return "offline";
  if (
    worker.state !== "alive" ||
    now - Date.parse(worker.last_seen) >= HEARTBEAT_MS
  )
    return "unhealthy";
  return worker.paused ? "paused" : "online";
}

function detail(worker: Worker) {
  const machine = worker.capabilities.machine;
  const accelerator = worker.capabilities.accelerator;
  const chip =
    (accelerator?.available && accelerator.device) ||
    machine?.cpu_model ||
    machine?.os ||
    null;
  const cores = machine?.logical_cores
    ? `${machine.logical_cores} cores`
    : null;
  return (
    [chip, cores].filter(Boolean).join(" · ") ||
    worker.capabilities.runtime.toUpperCase()
  );
}

function controlLink(
  worker: Worker,
  running: Task[],
  assigned: Task[],
  now: number,
): Pick<GraphLink, "state" | "label" | "flow" | "intensity"> {
  const state = nodeState(worker, now);
  if (state === "offline" || state === "unhealthy")
    return {
      state: "offline",
      label: state === "offline" ? "disconnected" : "no heartbeat",
      flow: 0,
      intensity: 0,
    };
  if (worker.paused)
    return {
      state: "paused",
      label: "paused · accepting nothing",
      flow: 0,
      intensity: 0,
    };
  if (running.length)
    return {
      state: "busy",
      // Progress is the thing actually travelling up the wire while a task runs.
      label:
        running.length > 1
          ? `streaming progress · ${running.length} tasks`
          : `running · ${taskTitle(running[0])}${
              running[0].progress > 0
                ? ` · ${Math.round(running[0].progress * 100)}%`
                : ""
            }`,
      flow: -1,
      intensity: Math.min(1, 0.55 + 0.2 * running.length),
    };
  if (assigned.length)
    return {
      state: "busy",
      label: `dispatching · ${taskTitle(assigned[0])}`,
      flow: 1,
      intensity: 0.7,
    };
  // Connected with nothing to do. The only traffic is the five-second liveness
  // ping, which is not work moving, so the line is drawn quiet and empty.
  return { state: "idle", label: "idle · heartbeat", flow: 0, intensity: 0 };
}

/**
 * Machines cooperating on one job, connected in a ring so the picture stays legible.
 * Only machines actually executing a task right now: two idle machines that happen to
 * share a finished job are not exchanging anything.
 */
function peerLinks(workers: Worker[], tasks: Task[]): GraphLink[] {
  const jobs = new Map<string, { workers: string[]; task: Task }>();
  const known = new Set(workers.map((worker) => worker.id));
  for (const task of tasks) {
    if (task.state !== "running" || !task.worker_id) continue;
    if (!known.has(task.worker_id)) continue;
    const entry = jobs.get(task.spec.job_id) || { workers: [], task };
    if (!entry.workers.includes(task.worker_id))
      entry.workers.push(task.worker_id);
    jobs.set(task.spec.job_id, entry);
  }
  const links: GraphLink[] = [];
  for (const [jobId, { workers: members, task }] of jobs) {
    if (members.length < 2) continue;
    const ring = [...members].sort();
    for (let index = 0; index < ring.length; index += 1) {
      const source = ring[index];
      const target = ring[(index + 1) % ring.length];
      // Two members make one line, not a doubled-back pair.
      if (ring.length === 2 && index === 1) break;
      links.push({
        id: `peer:${jobId}:${source}:${target}`,
        source,
        target,
        kind: "peer",
        state: "busy",
        label: `co-running · ${taskTitle(task)}`,
        flow: 0,
        intensity: 0.35,
      });
    }
  }
  return links;
}

/** What a job is asking for, which is what says where it can land. */
function requirement(spec: TaskSpec) {
  const runtime =
    spec.requirements.runtime === "cuda"
      ? "CUDA GPU"
      : spec.requirements.runtime === "mps"
        ? "Apple GPU"
        : "CPU";
  const vram = spec.requirements.vram_mib;
  return vram > 0
    ? `${runtime} · ${vram >= 1024 ? `${Math.round(vram / 1024)} GiB` : `${vram} MiB`} VRAM`
    : runtime;
}

/**
 * Jobs, as their own nodes.
 *
 * A job appears the moment it is launched, before any machine has picked it up, and
 * draws a line to the machine it is aimed at. That line is what says where the work
 * is about to go: dotted while the job is still queued for that machine, solid once
 * it is actually executing there. A job with no target hangs off the control plane,
 * waiting for the next free machine.
 */
function endedAt(ids: Set<string>, events: AuditEvent[]) {
  let last = 0;
  for (const event of events) {
    if (event.entity !== "task" || !ids.has(event.entity_id)) continue;
    if (!["succeeded", "failed", "cancelled"].includes(event.new_state))
      continue;
    last = Math.max(last, Date.parse(event.at));
  }
  return last || null;
}

function jobGraph(
  tasks: Task[],
  names: Map<string, string>,
  events: AuditEvent[],
  now: number,
): FleetGraph {
  // The same label the machine's own node carries, so a route names what it points at.
  const call = (id: string) => names.get(id) || workerName(id);
  const nodes: GraphNode[] = [];
  const links: GraphLink[] = [];
  const groups = groupJobs(tasks)
    .map((group) => ({
      group,
      ended: ["running", "queued"].includes(group.state)
        ? null
        : endedAt(new Set(group.tasks.map((task) => task.spec.id)), events),
    }))
    .filter(
      ({ group, ended }) =>
        ["running", "queued"].includes(group.state) ||
        (ended !== null && now - ended < LINGER_MS),
    )
    .slice(0, MAX_JOBS);
  for (const { group, ended } of groups) {
    const id = `job:${group.id}`;
    const running = group.tasks.filter((task) => task.state === "running");
    const need = requirement(group.primary.spec);
    const over = ended !== null;
    nodes.push({
      id,
      kind: "job",
      label: group.title,
      detail: over
        ? `${group.state} · ${need}`
        : `${need} · ${group.tasks.length} task${group.tasks.length === 1 ? "" : "s"}`,
      state: over ? "offline" : running.length ? "online" : "paused",
      load: running.length,
    });
    // One line per machine this job touches, whether it is there yet or not, and
    // whether or not it is still there. A stage that has finished keeps its line
    // while the job runs on: four batches lighting up and then settling one by one is
    // the shape of the run, and dropping each as it completes hides exactly that.
    // Where a machine holds several stages, the liveliest one names the line.
    const RANK = { running: 3, assigned: 2, queued: 1, done: 0 };
    const placed = new Map<string, keyof typeof RANK>();
    const put = (worker: string | null, stage: keyof typeof RANK) => {
      if (!worker) return;
      const current = placed.get(worker);
      if (!current || RANK[stage] > RANK[current]) placed.set(worker, stage);
    };
    for (const task of group.tasks) {
      if (over) put(task.worker_id, "done");
      else if (task.state === "running") put(task.worker_id, "running");
      else if (task.state === "assigned") put(task.worker_id, "assigned");
      else if (task.state === "queued" && task.spec.target_worker_id)
        put(task.spec.target_worker_id, "queued");
      else if (task.state === "succeeded") put(task.worker_id, "done");
    }
    for (const [worker, stage] of placed) {
      if (!names.has(worker)) continue;
      links.push({
        id: `job:${group.id}:${worker}`,
        source: id,
        target: worker,
        kind: "job",
        state:
          stage === "done" ? "result" : stage === "queued" ? "idle" : "busy",
        label:
          stage === "running"
            ? `running on · ${call(worker)}`
            : stage === "assigned"
              ? `starting on · ${call(worker)}`
              : stage === "queued"
                ? `queued for · ${call(worker)}`
                : over
                  ? `${group.state} on · ${call(worker)}`
                  : `finished on · ${call(worker)}`,
        flow: 1,
        intensity: stage === "assigned" ? 0.4 : stage === "running" ? 0.25 : 0,
      });
    }
    const waiting = over
      ? 0
      : group.tasks.filter(
          (task) => task.state === "queued" && !task.spec.target_worker_id,
        ).length;
    if (waiting || ![...placed.keys()].some((worker) => names.has(worker)))
      links.push({
        id: `job:${group.id}:queue`,
        source: CONTROL_ID,
        target: id,
        kind: "job",
        state: "idle",
        label: over
          ? `${group.state} · nowhere to show`
          : waiting
            ? `${waiting} queued · needs ${need}`
            : `waiting · needs ${need}`,
        flow: 0,
        intensity: 0,
      });
  }
  return { nodes, links };
}

/**
 * The agent layer. Each subagent hangs directly off the machine whose work it is
 * analysing, rather than through an intermediate node, so the picture says plainly
 * which process an agent belongs to. A job with no machine running it yet falls back
 * to the control plane, which is where its work is still sitting.
 *
 * Only the agents actually working are drawn. One that has filed its report is no
 * longer part of what is happening, and a dozen finished diamonds crowd the machines
 * they hang off; the activity log lists every agent and its status instead.
 */
function agentGraph(groups: AgentGroup[], known: Set<string>): FleetGraph {
  const nodes: GraphNode[] = [];
  const links: GraphLink[] = [];
  for (const group of groups) {
    const anchor =
      [...group.workers].filter((id) => known.has(id)).sort()[0] || CONTROL_ID;
    for (const child of group.children) {
      if (child.status !== "running") continue;
      const id = `agent:${group.jobId}:${child.id}`;
      nodes.push({
        id,
        kind: "agent",
        label: child.role.replaceAll("_", " "),
        detail: [group.title, child.summary || child.question]
          .filter(Boolean)
          .join(" · "),
        state: "online",
        load: 1,
      });
      links.push({
        id: `agent:${id}`,
        source: anchor,
        target: id,
        kind: "agent",
        state: "busy",
        label: `analysing · ${child.role.replaceAll("_", " ")}`,
        flow: -1,
        intensity: 0.55,
      });
    }
  }
  return { nodes, links };
}

export function buildFleetGraph(
  snapshot: Snapshot,
  now = Date.now(),
  agents: AgentGroup[] = [],
): FleetGraph {
  const workers = [...snapshot.workers]
    .filter(
      (worker) =>
        nodeState(worker, now) !== "offline" ||
        now - Date.parse(worker.last_seen) < STALE_MS,
    )
    .sort((a, b) => a.id.localeCompare(b.id));
  const online = workers.filter(
    (worker) => nodeState(worker, now) === "online",
  );
  const nodes: GraphNode[] = [
    {
      id: CONTROL_ID,
      kind: "control",
      label: "Control plane",
      detail: `${online.length} of ${workers.length} machines connected`,
      state: "online",
      load: snapshot.tasks.filter((task) => task.state === "running").length,
    },
  ];
  const links: GraphLink[] = [];
  for (const worker of workers) {
    const mine = snapshot.tasks.filter((task) => task.worker_id === worker.id);
    const running = mine.filter((task) => task.state === "running");
    const assigned = mine.filter((task) => task.state === "assigned");
    nodes.push({
      id: worker.id,
      kind: "worker",
      label: worker.name?.trim() || workerName(worker.id),
      detail: detail(worker),
      state: nodeState(worker, now),
      load: running.length + assigned.length,
    });
    links.push({
      id: `control:${worker.id}`,
      source: CONTROL_ID,
      target: worker.id,
      kind: "control",
      ...controlLink(worker, running, assigned, now),
    });
  }
  const names = new Map(
    nodes
      .filter((node) => node.kind === "worker")
      .map((node) => [node.id, node.label] as const),
  );
  const queue = jobGraph(snapshot.tasks, names, snapshot.events, now);
  const layer = agentGraph(agents, new Set(names.keys()));
  return {
    nodes: [...nodes, ...queue.nodes, ...layer.nodes],
    links: [
      ...links,
      ...peerLinks(workers, snapshot.tasks),
      ...queue.links,
      ...layer.links,
    ],
  };
}
