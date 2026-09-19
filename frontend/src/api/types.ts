export interface Requirements {
  runtime: "cpu" | "cuda" | "mps";
  vram_mib: number;
}
export interface Worker {
  id: string;
  session_id: string;
  capabilities: Requirements & { kinds: string[] };
  state: "alive" | "unhealthy" | "offline";
  last_seen: string;
  paused: boolean;
}
export interface TaskSpec {
  id: string;
  job_id: string;
  kind: string;
  payload: unknown;
  requirements: Requirements;
  max_attempts: number;
  timeout_seconds: number;
  target_worker_id: string | null;
  allow_failover: boolean;
}
export interface Task {
  spec: TaskSpec;
  state:
    "queued" | "assigned" | "running" | "succeeded" | "failed" | "cancelled";
  generation: number;
  worker_id: string | null;
  session_id: string | null;
  lease_until: string | null;
  deadline: string | null;
  result: unknown;
  failure: string;
  created_at: string;
  progress: number;
  started_at: string | null;
  attestation?: Record<string, unknown> | null;
}
export interface AuditEvent {
  id: number;
  at: string;
  entity: string;
  entity_id: string;
  previous_state: string;
  new_state: string;
  details: Record<string, unknown>;
}
export interface Snapshot {
  workers: Worker[];
  tasks: Task[];
  events: AuditEvent[];
}
export type ConnectionStatus = "connecting" | "live" | "reconnecting";
