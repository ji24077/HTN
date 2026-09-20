/**
 * The stages a simulation job actually ran, as ordinary tasks.
 *
 * The dashboard snapshot carries one row per simulation job — the root — and that row
 * has no worker, because no single machine runs a job. The stages that do have machines
 * (the profiling pass, the independent validation, each batch, the aggregation) are only
 * behind the per-job simulation endpoint. Drawing the fleet from the snapshot alone
 * therefore shows a job hanging off the control plane with nothing running it, which is
 * exactly backwards: the job is the one thing in the picture that is definitely on a
 * machine somewhere.
 *
 * Turning those stages back into `Task` rows means everything downstream — grouping,
 * the job node, the lines to each machine, the replay — works on them unchanged,
 * without a second notion of what a unit of work is.
 */
import type { SimulationStatus } from "../api/client";
import type { Task } from "../api/types";

const STATES: Task["state"][] = [
  "queued",
  "assigned",
  "running",
  "succeeded",
  "failed",
  "cancelled",
];

const asState = (value: string): Task["state"] =>
  (STATES as string[]).includes(value) ? (value as Task["state"]) : "queued";

export function stageTasks(
  status: SimulationStatus,
  template: Task,
  jobId: string,
): Task[] {
  return (status.tasks || [])
    .filter((stage) => stage.id)
    .map((stage) => ({
      ...template,
      spec: {
        ...template.spec,
        id: stage.id,
        job_id: jobId,
        target_worker_id: null,
        // The role is what this stage is: "batch-60", "aggregate", "profiling".
        payload: { value: { label: stage.role || stage.id } },
      },
      state: asState(stage.state),
      worker_id: stage.worker_id,
      progress: typeof stage.progress === "number" ? stage.progress : 0,
      generation: stage.generation ?? template.generation,
      failure: stage.failure || "",
    }));
}
