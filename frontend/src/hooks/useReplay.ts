import { useEffect, useMemo, useRef, useState } from "react";
import { executionEvents, readSimulation } from "../api/client";
import type { ExecutionEvent, Snapshot } from "../api/types";
import type { AgentGroup } from "../lib/fleetGraph";
import { buildReplay, type Replay } from "../lib/fleetReplay";
import { stageTasks } from "../lib/jobStages";
import { groupJobs, type JobGroup } from "../lib/jobs";

/** Tasks whose execution log is read for one replay. */
const MAX_TASKS = 12;
/** Steps per second during playback. Slow enough to read the caption. */
const RATE = 3;

/**
 * Loading and playing back a finished run.
 *
 * Nothing is fetched until a run is chosen, and everything is dropped when the replay
 * is closed, so this costs the live view nothing while it is not in use.
 */
export function useReplay(snapshot: Snapshot) {
  // Every finished run the snapshot knows about, with no cap of its own. There is
  // already a bound — the snapshot is the newest 500 task rows — and adding a second,
  // smaller one on top of it only made the list stop somewhere unexplained.
  const past = useMemo(
    () =>
      groupJobs(snapshot.tasks).filter((group) =>
        ["succeeded", "failed", "cancelled"].includes(group.state),
      ),
    [snapshot.tasks],
  );
  const [replay, setReplay] = useState<{
    replay: Replay;
    group: JobGroup;
    agents: AgentGroup[];
  } | null>(null);
  const [index, setIndex] = useState(0);
  const [playing, setPlaying] = useState(false);
  const [loading, setLoading] = useState("");
  const [error, setError] = useState("");
  const request = useRef<AbortController | null>(null);

  useEffect(() => () => request.current?.abort(), []);

  useEffect(() => {
    if (!playing || !replay) return;
    const timer = setInterval(() => {
      setIndex((current) => {
        if (current + 1 >= replay.replay.steps.length) {
          setPlaying(false);
          return current;
        }
        return current + 1;
      });
    }, 1000 / RATE);
    return () => clearInterval(timer);
  }, [playing, replay]);

  async function open(group: JobGroup) {
    request.current?.abort();
    const controller = new AbortController();
    request.current = controller;
    setLoading(group.id);
    setError("");
    try {
      // The snapshot holds one row for a simulation job and it has no machine. The
      // stages behind the simulation endpoint are the run: which machine took each
      // batch, in what order. Without them a replay is a job on its own in an empty
      // room, which is the one thing a replay must not be.
      const status = await readSimulation(group.id, controller.signal).catch(
        () => null,
      );
      const stages = status ? stageTasks(status, group.primary, group.id) : [];
      const detailed: JobGroup = stages.length
        ? {
            ...group,
            // The root row stays at the head: it is what the job is called. The
            // stages after it are what it did and where.
            tasks: [...group.tasks, ...stages],
            workers: [
              ...new Set(
                stages
                  .map((task) => task.worker_id)
                  .filter((id): id is string => Boolean(id)),
              ),
            ],
          }
        : group;
      const logs = await Promise.all(
        detailed.tasks
          .filter((task) => task.worker_id)
          .slice(0, MAX_TASKS)
          .map((task) =>
            executionEvents(task.spec.id, 0, controller.signal)
              .then((page) => page.events)
              .catch((): ExecutionEvent[] => []),
          ),
      );
      // A finished run's analysis agents are still on file; show them alongside it.
      const agents: AgentGroup[] = status?.analysis?.children?.length
        ? [
            {
              jobId: group.id,
              title: group.title,
              rationale: status.analysis.rationale,
              workers: detailed.workers,
              children: status.analysis.children.map((child) => ({
                id: child.id,
                role: child.role,
                status: child.status,
                question: child.question,
                summary: child.report?.summary,
              })),
            },
          ]
        : [];
      if (controller.signal.aborted) return;
      setReplay({
        replay: buildReplay(detailed, logs.flat()),
        group: detailed,
        agents,
      });
      setIndex(0);
      setPlaying(true);
    } catch {
      if (!controller.signal.aborted)
        setError("Could not load that run. Try another.");
    } finally {
      if (!controller.signal.aborted) setLoading("");
    }
  }

  function close() {
    request.current?.abort();
    setReplay(null);
    setPlaying(false);
    setIndex(0);
    setLoading("");
  }

  const steps = replay?.replay.steps;
  const at = steps ? Math.min(index, steps.length - 1) : 0;
  return {
    past,
    replay,
    step: steps ? steps[at] : null,
    index: at,
    total: steps?.length ?? 0,
    playing,
    loading,
    error,
    open,
    close,
    seek: (value: number) => {
      setPlaying(false);
      setIndex(value);
    },
    toggle: () =>
      setPlaying((value) => {
        if (!value && steps && at >= steps.length - 1) setIndex(0);
        return !value;
      }),
  };
}
