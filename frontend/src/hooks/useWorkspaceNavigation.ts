import { useEffect, useState } from "react";
import { getTask } from "../api/client";
import type { Task } from "../api/types";

const destinations = [
  "Jobs",
  "Workers",
  "Assistant",
  "Activity",
  "Topology",
  "Experiments",
] as const;
export const jobViews = ["Overview", "Files", "Activity", "Details"] as const;
export type JobView = (typeof jobViews)[number];
type Destination = (typeof destinations)[number];

function readRoute() {
  const [destination, encodedId, tab] = window.location.hash
    .replace(/^#\/?/, "")
    .split("/");
  const view =
    destinations.find((name) => name.toLowerCase() === destination) || "Jobs";
  const jobView =
    jobViews.find((name) => name.toLowerCase() === tab) || "Overview";
  let taskId = "";
  let invalid = false;
  if (view === "Jobs" && encodedId) {
    try {
      taskId = decodeURIComponent(encodedId);
    } catch {
      invalid = true;
    }
  }
  return { view, taskId, jobView, invalid };
}

function jobPath(taskId: string, view: JobView = "Overview") {
  return `/jobs/${encodeURIComponent(taskId)}${view === "Overview" ? "" : `/${view.toLowerCase()}`}`;
}

export function useWorkspaceNavigation() {
  const [route, setRoute] = useState(readRoute);
  const [lookup, setLookup] = useState<{
    id: string;
    task?: Task;
    error?: string;
  }>({ id: "" });
  const [revision, setRevision] = useState(0);

  useEffect(() => {
    const restore = () => setRoute(readRoute());
    window.addEventListener("hashchange", restore);
    window.addEventListener("popstate", restore);
    return () => {
      window.removeEventListener("hashchange", restore);
      window.removeEventListener("popstate", restore);
    };
  }, []);

  const taskId = route.taskId;
  const detail = lookup.id === taskId ? (lookup.task ?? null) : null;
  useEffect(() => {
    if (!taskId || detail) return;
    const controller = new AbortController();
    setLookup({ id: taskId });
    getTask(taskId, controller.signal)
      .then((task) => {
        if (task?.spec?.id !== taskId) throw new Error("Invalid job response");
        if (!controller.signal.aborted && readRoute().taskId === taskId)
          setLookup({ id: taskId, task });
      })
      .catch(() => {
        if (!controller.signal.aborted && readRoute().taskId === taskId)
          setLookup({
            id: taskId,
            error:
              "We couldn’t load this job. Try again, or return to your jobs.",
          });
      });
    return () => controller.abort();
  }, [taskId, detail, revision]);

  function go(path: string) {
    if (window.location.hash !== `#${path}`)
      window.history.pushState(null, "", `#${path}`);
    setRoute(readRoute());
  }

  return {
    view: route.view,
    jobView: route.jobView,
    detail,
    jobRequest: route.invalid
      ? {
          error:
            "This job link is invalid. Return to your jobs to select a job.",
          retryable: false,
        }
      : taskId && !detail
        ? {
            error: lookup.id === taskId ? lookup.error : undefined,
            retryable: true,
          }
        : null,
    retryJob: () => setRevision((value) => value + 1),
    openDetail(task: Task) {
      setLookup({ id: task.spec.id, task });
      go(jobPath(task.spec.id));
    },
    changeJobView(view: JobView) {
      if (taskId) go(jobPath(taskId, view));
    },
    closeDetail: () => go("/jobs"),
    navigate(view: Destination) {
      go(`/${view.toLowerCase()}`);
      window.scrollTo({ top: 0 });
    },
  };
}
