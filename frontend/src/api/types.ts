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
  account?: {
    currency: "CAD";
    credited: string;
    spent: string;
    balance: string;
  } | null;
  workers: Worker[];
  tasks: Task[];
  events: AuditEvent[];
  usage?: {
    currency: "CAD";
    estimated: true;
    cost: string;
    attempts: number;
  };
}
export type ConnectionStatus = "connecting" | "live" | "reconnecting";

export interface ChatToolActivity {
  call_id: string;
  name: string;
  arguments: Record<string, unknown> | null;
  status: "running" | "completed" | "failed";
  result: Record<string, unknown> | null;
}

export interface ChatTurn {
  request_id: string;
  message: string;
  status: "running" | "completed" | "failed";
  reply: string;
  tools: ChatToolActivity[];
}

export interface ChatMessage {
  request_id: string;
  message: string;
}

export interface ExecutionEvent {
  id: number;
  execution_id: string;
  task_id: string;
  attempt: number;
  worker_id: string | null;
  source: "server" | "worker";
  sequence: number;
  kind: string;
  occurred_at: string;
  received_at: string;
  data: Record<string, unknown>;
}
