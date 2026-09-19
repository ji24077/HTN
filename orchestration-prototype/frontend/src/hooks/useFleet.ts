import { useEffect, useState } from "react";
import { openSession } from "../api/client";
import type { ConnectionStatus, Snapshot } from "../api/types";

const empty: Snapshot = { workers: [], tasks: [], events: [] };
export function useFleet() {
  const [snapshot, setSnapshot] = useState(empty);
  const [status, setStatus] = useState<ConnectionStatus>("connecting");
  const [updatedAt, setUpdatedAt] = useState<Date | null>(null);
  useEffect(() => {
    let disposed = false;
    let stream: EventSource | undefined;
    let retry: ReturnType<typeof setTimeout> | undefined;
    let controller: AbortController | undefined;
    function stop() {
      controller?.abort();
      stream?.close();
      clearTimeout(retry);
    }
    function scheduleRetry() {
      if (disposed) return;
      setStatus("reconnecting");
      clearTimeout(retry);
      retry = setTimeout(() => void connect(), 2000);
    }
    async function connect() {
      stop();
      controller = new AbortController();
      const signal = controller.signal;
      try {
        await openSession(signal);
        if (disposed || signal.aborted) return;
        stream = new EventSource("/v1/updates");
        stream.addEventListener("snapshot", (event: MessageEvent<string>) => {
          if (disposed) return;
          try {
            const next: Snapshot = JSON.parse(event.data);
            if (
              !Array.isArray(next.workers) ||
              !Array.isArray(next.tasks) ||
              !Array.isArray(next.events)
            )
              throw new Error("Invalid snapshot");
            setSnapshot(next);
            setUpdatedAt(new Date());
            setStatus("live");
          } catch {
            setStatus("reconnecting");
          }
        });
        stream.addEventListener("unavailable", () => setStatus("reconnecting"));
        stream.onerror = () => {
          if (disposed) return;
          setStatus("reconnecting");
          // EventSource handles normal reconnects. A fatal auth response closes
          // it; establish a new local session before retrying in that case.
          if (stream?.readyState === EventSource.CLOSED) scheduleRetry();
        };
      } catch {
        if (!disposed && !signal.aborted) scheduleRetry();
      }
    }
    const resume = (event: PageTransitionEvent) => {
      if (event.persisted) void connect();
    };
    window.addEventListener("pagehide", stop);
    window.addEventListener("pageshow", resume);
    void connect();
    return () => {
      disposed = true;
      stop();
      window.removeEventListener("pagehide", stop);
      window.removeEventListener("pageshow", resume);
    };
  }, []);
  return { snapshot, status, updatedAt };
}
