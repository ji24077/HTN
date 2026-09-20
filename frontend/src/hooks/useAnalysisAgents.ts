import { useEffect, useMemo, useRef, useState } from "react";
import { readSimulation } from "../api/client";
import type { Snapshot, Task } from "../api/types";
import type { AgentGroup } from "../lib/fleetGraph";
import { stageTasks } from "../lib/jobStages";
import { groupJobs } from "../lib/jobs";

/** Jobs polled per round. A fleet with more running keeps the newest few. */
const MAX_JOBS = 3;
const INTERVAL_MS = 6000;

/**
 * What the snapshot cannot say about a running simulation: which machines its stages
 * are on, and which analysis subagents are working under it.
 *
 * Both live behind the per-job simulation endpoint, so this polls, and deliberately
 * stays cheap about it: only while the topology tab is open, only for simulation jobs
 * that are actually running, a handful at a time, six seconds apart, and never while
 * the page is hidden. A failed read is dropped rather than surfaced; this view is an
 * extra and must not make noise for the rest of the dashboard.
 */
export function useAnalysisAgents(snapshot: Snapshot, active: boolean) {
  const [groups, setGroups] = useState<AgentGroup[]>([]);
  const [stages, setStages] = useState<Task[]>([]);
  const jobs = useMemo(
    () =>
      groupJobs(snapshot.tasks)
        .filter(
          (job) =>
            ["running", "queued"].includes(job.state) &&
            job.tasks.some((task) => task.spec.kind === "simulation_job"),
        )
        .slice(0, MAX_JOBS),
    [snapshot.tasks],
  );
  // The effect keys off the job ids alone; titles and workers are read at poll time
  // so a progress update does not restart the timer.
  const key = jobs.map((job) => job.id).join(",");
  const jobsRef = useRef(jobs);
  jobsRef.current = jobs;

  useEffect(() => {
    if (!active || !key) {
      setGroups([]);
      setStages([]);
      return;
    }
    let stopped = false;
    const controller = new AbortController();
    async function round() {
      if (stopped || document.hidden) return;
      const found: AgentGroup[] = [];
      const placed: Task[] = [];
      for (const job of jobsRef.current) {
        if (stopped) return;
        try {
          const status = await readSimulation(job.id, controller.signal);
          placed.push(...stageTasks(status, job.primary, job.id));
          const children = status.analysis?.children || [];
          if (children.length)
            found.push({
              jobId: job.id,
              title: job.title,
              rationale: status.analysis?.rationale,
              workers: job.workers,
              children: children.map((child) => ({
                id: child.id,
                role: child.role,
                status: child.status,
                question: child.question,
                summary: child.report?.summary,
              })),
            });
        } catch {
          // A job without an analysis stage, or a read that failed, simply
          // contributes no nodes this round.
        }
      }
      if (!stopped) {
        setGroups(found);
        setStages(placed);
      }
    }
    void round();
    const timer = setInterval(() => void round(), INTERVAL_MS);
    return () => {
      stopped = true;
      clearInterval(timer);
      controller.abort();
    };
  }, [key, active]);

  return { agents: groups, stages };
}
