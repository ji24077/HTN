import { useCallback, useEffect, useRef, useState } from "react";
import {
  lab,
  LIVE_STATUSES,
  type Experiment,
  type LabJob,
  type LabModel,
  type Pod,
  type Serving,
} from "../api/gpulab";

export type GpuLabState = {
  status: "connecting" | "live" | "offline";
  error: string;
  podError: string;
  experiment: Experiment | null;
  jobs: LabJob[];
  models: LabModel[];
  serving: Serving;
  pods: Pod[];
};

const empty: GpuLabState = {
  status: "connecting",
  error: "",
  podError: "",
  experiment: null,
  jobs: [],
  models: [],
  serving: { running: false },
  pods: [],
};

/** Experiment server state, with the live job polled until it settles. */
export function useGpuLab(active: boolean, onSettled: (job: LabJob) => void) {
  const [state, setState] = useState(empty);
  const settled = useRef(onSettled);
  settled.current = onSettled;

  const refresh = useCallback(async () => {
    try {
      const [stateData, jobData, modelData] = await Promise.all([
        lab<{ experiment: Experiment }>("/api/state"),
        lab<{ jobs?: LabJob[] }>("/api/jobs"),
        lab<{ models?: LabModel[]; serving?: Serving }>("/api/models"),
      ]);
      let pods: Pod[] = [];
      let podError = "";
      try {
        pods = (await lab<{ pods?: Pod[] }>("/api/pods")).pods || [];
      } catch (cause) {
        podError = cause instanceof Error ? cause.message : "Request failed";
      }
      setState({
        status: "live",
        error: "",
        podError,
        experiment: stateData.experiment,
        jobs: jobData.jobs || [],
        models: modelData.models || [],
        serving: modelData.serving || { running: false },
        pods,
      });
    } catch (cause) {
      setState((current) => ({
        ...current,
        status: "offline",
        error: cause instanceof Error ? cause.message : "Request failed",
      }));
    }
  }, []);

  useEffect(() => {
    if (active) void refresh();
  }, [active, refresh]);

  // The experiment server is often started after this page; keep knocking.
  useEffect(() => {
    if (!active || state.status !== "offline") return;
    const timer = setTimeout(() => void refresh(), 5000);
    return () => clearTimeout(timer);
  }, [active, state, refresh]);

  const liveId = state.jobs.find((job) =>
    LIVE_STATUSES.includes(job.status),
  )?.id;
  useEffect(() => {
    if (!active || !liveId) return;
    let stopped = false;
    let timer: ReturnType<typeof setTimeout>;
    async function poll() {
      try {
        const job = await lab<LabJob>("/api/jobs/" + liveId);
        if (stopped) return;
        setState((current) => ({
          ...current,
          jobs: current.jobs.map((item) => (item.id === job.id ? job : item)),
        }));
        if (!LIVE_STATUSES.includes(job.status)) {
          settled.current(job);
          await refresh();
          return;
        }
      } catch {
        /* Polling is best effort; the next tick retries. */
      }
      if (!stopped) timer = setTimeout(() => void poll(), 1800);
    }
    timer = setTimeout(() => void poll(), 1800);
    return () => {
      stopped = true;
      clearTimeout(timer);
    };
  }, [active, liveId, refresh]);

  const start = useCallback(async (path: string, body: unknown) => {
    const job = await lab<LabJob>(path, body);
    setState((current) => ({ ...current, jobs: [job, ...current.jobs] }));
    return job;
  }, []);

  return { ...state, refresh, start };
}
