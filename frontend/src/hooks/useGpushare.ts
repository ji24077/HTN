import { useEffect, useState } from "react";
import {
  ApiError,
  gpushareConfig,
  gpushareJobs,
  gpushareModels,
  gpusharePods,
} from "../api/client";
import type { GpuJob, GpuPod, GpuServing } from "../api/types";

export interface GpushareState {
  enabled: boolean;
  pods: GpuPod[];
  jobs: GpuJob[];
  serving: GpuServing | null;
  error: string;
  loading: boolean;
}

const EMPTY: GpushareState = {
  enabled: false,
  pods: [],
  jobs: [],
  serving: null,
  error: "",
  loading: true,
};

/**
 * gpushare state, polled.
 *
 * Deliberately separate from useFleet: that hook owns the /v1/updates stream
 * and throws on any payload that is not {workers, tasks, events}, so widening
 * it to carry GPU state would make one malformed GPU field take the whole
 * dashboard down. gpushare is also a different process with its own uptime —
 * it going away should grey out one panel, not the page.
 */
export function useGpushare(active: boolean): GpushareState {
  const [state, setState] = useState<GpushareState>(EMPTY);
  useEffect(() => {
    // Polling a rented-GPU service costs a RunPod API call; do not pay it
    // while the operator is looking at another tab.
    if (!active) return;
    const controller = new AbortController();
    let timer: ReturnType<typeof setTimeout>;
    async function poll() {
      try {
        const { enabled } = await gpushareConfig(controller.signal);
        if (controller.signal.aborted) return;
        if (!enabled) {
          setState({ ...EMPTY, loading: false });
          return; // Not configured is a permanent answer; stop polling.
        }
        const [pods, models, jobs] = await Promise.all([
          gpusharePods(controller.signal),
          gpushareModels(controller.signal),
          gpushareJobs(controller.signal),
        ]);
        if (controller.signal.aborted) return;
        setState({
          enabled: true,
          pods: pods.pods ?? [],
          jobs: jobs.jobs ?? [],
          serving: models.serving ?? null,
          error: "",
          loading: false,
        });
      } catch (cause) {
        if (controller.signal.aborted) return;
        const message =
          cause instanceof ApiError && cause.status === 502
            ? "gpushare is configured but not answering. Is the console running?"
            : "GPU state unavailable.";
        setState((previous) => ({
          ...previous,
          error: message,
          loading: false,
        }));
        if (cause instanceof ApiError && [401, 403].includes(cause.status))
          return;
      }
      if (!controller.signal.aborted) timer = setTimeout(poll, 5000);
    }
    void poll();
    return () => {
      controller.abort();
      clearTimeout(timer);
    };
  }, [active]);
  return state;
}
