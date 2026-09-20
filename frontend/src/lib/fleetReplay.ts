/**
 * Replaying a finished run.
 *
 * A run leaves a per-task execution log behind — started, progress, succeeded and the
 * rest, each stamped with when it happened and which machine it happened on. Folding
 * that log forward gives the state of every task at each moment of the run, and each
 * of those states is turned back into an ordinary snapshot. The renderer does not
 * know it is looking at the past: it draws a replayed snapshot exactly as it draws a
 * live one.
 */
import type {
  AuditEvent,
  ExecutionEvent,
  Snapshot,
  Task,
  Worker,
} from "../api/types";
import { taskTitle, workerName } from "./format";
import type { JobGroup } from "./jobs";

export interface ReplayStep {
  /** When this moment happened, epoch ms. */
  at: number;
  caption: string;
  states: Record<
    string,
    { state: Task["state"]; worker: string | null; progress: number }
  >;
}

export interface Replay {
  jobId: string;
  title: string;
  steps: ReplayStep[];
  /** Machines that took part, in the order they first appeared. */
  workers: string[];
}

const TERMINAL: Record<string, Task["state"]> = {
  succeeded: "succeeded",
  failed: "failed",
  cancelled: "cancelled",
  timed_out: "failed",
  interrupted: "failed",
};

function text(data: Record<string, unknown>, ...keys: string[]) {
  for (const key of keys) {
    const value = data[key];
    if (typeof value === "string" && value.trim())
      return value.trim().length > 70
        ? `${value.trim().slice(0, 69)}…`
        : value.trim();
  }
  return "";
}

export function buildReplay(group: JobGroup, events: ExecutionEvent[]): Replay {
  const titles = new Map(
    group.tasks.map((task) => [task.spec.id, taskTitle(task)] as const),
  );
  const name = (id: string) => titles.get(id) || id;
  const states: ReplayStep["states"] = {};
  for (const task of group.tasks)
    states[task.spec.id] = { state: "queued", worker: null, progress: 0 };
  const seen: string[] = [];
  const steps: ReplayStep[] = [
    {
      at: Date.parse(group.createdAt),
      caption: `“${group.title}” queued · ${group.tasks.length} task${group.tasks.length === 1 ? "" : "s"}`,
      states: structuredClone(states),
    },
  ];
  const ordered = [...events].sort(
    (one, two) =>
      Date.parse(one.occurred_at) - Date.parse(two.occurred_at) ||
      one.sequence - two.sequence,
  );
  for (const event of ordered) {
    const current = states[event.task_id];
    if (!current) continue;
    if (event.worker_id && !seen.includes(event.worker_id))
      seen.push(event.worker_id);
    if (event.worker_id) current.worker = event.worker_id;
    const who = event.worker_id ? workerName(event.worker_id) : "A machine";
    let caption = "";
    if (event.kind === "started") {
      current.state = "running";
      caption = `${who} started “${name(event.task_id)}”`;
    } else if (event.kind === "progress") {
      current.state = "running";
      const value = Number(event.data.progress ?? event.data.fraction);
      if (Number.isFinite(value))
        current.progress = value > 1 ? value / 100 : value;
      caption = `${name(event.task_id)} · ${Math.round(current.progress * 100)}%`;
    } else if (TERMINAL[event.kind]) {
      current.state = TERMINAL[event.kind];
      if (event.kind === "succeeded") current.progress = 1;
      const why = text(event.data, "error", "message", "reason");
      caption = `“${name(event.task_id)}” ${event.kind.replaceAll("_", " ")}${why ? ` · ${why}` : ""}`;
    } else if (event.kind === "cleaned") {
      caption = `${who} cleaned up after “${name(event.task_id)}”`;
    } else {
      const line = text(event.data, "message", "name", "line", "text");
      if (!line) continue;
      caption = `${name(event.task_id)} · ${line}`;
    }
    steps.push({
      at: Date.parse(event.occurred_at),
      caption,
      states: structuredClone(states),
    });
  }
  // The log can be incomplete — an older agent, a machine that died mid-task — so the
  // run always ends on the outcome the database actually recorded.
  for (const task of group.tasks) {
    states[task.spec.id] = {
      state: task.state,
      worker: task.worker_id,
      progress: task.state === "succeeded" ? 1 : task.progress,
    };
    if (task.worker_id && !seen.includes(task.worker_id))
      seen.push(task.worker_id);
  }
  const done = group.tasks.filter((task) => task.state === "succeeded").length;
  steps.push({
    at: Math.max(steps[steps.length - 1].at + 1, Date.parse(group.createdAt)),
    caption: `Run ${group.state} · ${done} of ${group.tasks.length} task${group.tasks.length === 1 ? "" : "s"} succeeded`,
    states: structuredClone(states),
  });
  return { jobId: group.id, title: group.title, steps, workers: seen };
}

/**
 * One moment of a replay, as a snapshot the graph can be built from. Only the
 * machines that took part are included: a replay is about this run, not about whoever
 * else happens to be online while you watch it.
 */
export function replaySnapshot(
  replay: Replay,
  step: ReplayStep,
  group: JobGroup,
  fleet: Worker[],
): Snapshot {
  const stamp = new Date(step.at).toISOString();
  const workers: Worker[] = replay.workers.map((id) => {
    const live = fleet.find((worker) => worker.id === id);
    return live
      ? { ...live, state: "alive", paused: false, last_seen: stamp }
      : {
          id,
          name: workerName(id),
          session_id: `${id}-replay`,
          capabilities: { runtime: "cpu", vram_mib: 0, kinds: [] },
          state: "alive",
          last_seen: stamp,
          paused: false,
        };
  });
  const tasks = group.tasks.map((task) => {
    const at = step.states[task.spec.id];
    return {
      ...task,
      state: at?.state ?? "queued",
      worker_id: at?.worker ?? null,
      progress: at?.progress ?? 0,
    };
  });
  // A replay has no audit trail, and the audit trail is what tells the graph a job
  // has just finished rather than never existed. Without this the run's last frame
  // drops the job entirely — the one frame where "where did that run?" is the only
  // question left. So the moments that have already happened are stated as events.
  const events: AuditEvent[] = tasks
    .filter((task) => ["succeeded", "failed", "cancelled"].includes(task.state))
    .map((task, index) => ({
      id: index + 1,
      at: stamp,
      entity: "task",
      entity_id: task.spec.id,
      previous_state: "running",
      new_state: task.state,
      details: task.worker_id ? { worker_id: task.worker_id } : {},
    }));
  return { workers, tasks, events };
}
