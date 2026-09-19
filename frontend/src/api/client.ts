import { createClient, type SupabaseClient } from "@supabase/supabase-js";
import type {
  ChatMessage,
  ChatTurn,
  Task,
  TaskSpec,
  ExecutionEvent,
} from "./types";

export function executionEvents(
  taskId: string,
  after: number,
  signal: AbortSignal,
) {
  return request<{
    events: ExecutionEvent[];
    next_cursor: number;
    has_more: boolean;
  }>(
    `/v1/tasks/${encodeURIComponent(taskId)}/execution-events?after=${after}`,
    { signal },
  );
}

export class ApiError extends Error {
  constructor(
    message: string,
    public status: number,
  ) {
    super(message);
  }
}

async function rawRequest<T>(
  path: string,
  options: RequestInit = {},
): Promise<T> {
  const response = await fetch(path, {
    credentials: "same-origin",
    ...options,
    headers: { "Content-Type": "application/json", ...options.headers },
  });
  if (!response.ok) {
    const body = await response.json().catch(() => null);
    const message = body?.detail || body?.error;
    throw new ApiError(
      typeof message === "string"
        ? message
        : `Request failed (${response.status})`,
      response.status,
    );
  }
  return response.json() as Promise<T>;
}

let authClient: Promise<SupabaseClient> | undefined;
let recoveringPassword = false;
let signingOut = false;

function clearAuthRedirect() {
  recoveringPassword = false;
  window.history.replaceState(null, "", window.location.pathname);
}

export class AuthLinkError extends Error {}

function supabase(): Promise<SupabaseClient> {
  if (!authClient) {
    authClient = rawRequest<{ url: string; publishableKey: string }>(
      "/auth/config",
    )
      .then(async ({ url, publishableKey }) => {
        const client = createClient(url, publishableKey, {
          auth: { flowType: "pkce" },
        });
        const { data: listener } = client.auth.onAuthStateChange((event) => {
          if (event === "PASSWORD_RECOVERY") {
            recoveringPassword = true;
            // Keep recovery visible across reloads after the SDK consumes the link.
            window.history.replaceState(null, "", "?auth=recovery");
            window.dispatchEvent(new Event("password-recovery"));
          }
          if (event === "SIGNED_OUT") {
            clearAuthRedirect();
            if (signingOut) return;
            // Defer work outside the SDK's auth callback lock.
            setTimeout(() => {
              void rawRequest("/auth/session", { method: "DELETE" }).catch(
                () => {},
              );
              window.dispatchEvent(new Event("session-expired"));
            }, 0);
          }
        });
        const { error } = await client.auth.initialize();
        if (error) {
          listener.subscription.unsubscribe();
          throw new AuthLinkError(
            "Could not verify this email link. Please request a new one and open it in the same browser.",
          );
        }
        return client;
      })
      .catch((error) => {
        authClient = undefined;
        throw error;
      });
  }
  return authClient;
}

async function exchange(token: string) {
  await rawRequest("/auth/session", {
    method: "POST",
    headers: { Authorization: `Bearer ${token}` },
  });
}

let refreshing: Promise<void> | undefined;
async function renewSession() {
  if (!refreshing) {
    refreshing = (async () => {
      const client = await supabase();
      const { data, error } = await client.auth.getSession();
      if (error || !data.session) throw new ApiError("Please sign in", 401);
      await exchange(data.session.access_token);
    })().finally(() => {
      refreshing = undefined;
    });
  }
  await refreshing;
}

async function request<T>(path: string, options: RequestInit = {}): Promise<T> {
  try {
    return await rawRequest<T>(path, options);
  } catch (error) {
    if (!(error instanceof ApiError) || error.status !== 401) throw error;
    try {
      await renewSession();
      return await rawRequest<T>(path, options);
    } catch (renewalError) {
      if (
        renewalError instanceof ApiError &&
        [401, 403].includes(renewalError.status)
      ) {
        window.dispatchEvent(new Event("session-expired"));
      }
      throw renewalError;
    }
  }
}

export async function openSession(
  signal: AbortSignal,
): Promise<{ mode: "demo" | "public" | "recovery"; email?: string }> {
  const params = new URLSearchParams(window.location.hash.slice(1));
  const query = new URLSearchParams(window.location.search);
  if (params.has("error") || query.has("error")) {
    throw new AuthLinkError(
      "This email link is invalid or has expired. Please request a new one.",
    );
  }
  // Consume confirmation links even when the browser still has another
  // account's (possibly denied) API cookie.
  if (query.has("code") || params.has("access_token")) {
    const client = await supabase();
    const { data, error } = await client.auth.getSession();
    if (error || !data.session) {
      throw new AuthLinkError(
        "Could not verify this email link. Please request a new one.",
      );
    }
    if (
      !recoveringPassword &&
      query.get("auth") !== "recovery" &&
      params.get("type") !== "recovery"
    ) {
      await exchange(data.session.access_token);
    }
  }
  if (
    recoveringPassword ||
    query.get("auth") === "recovery" ||
    params.get("type") === "recovery"
  ) {
    const client = await supabase();
    const { data, error } = await client.auth.getSession();
    if (error || !data.session) {
      throw new AuthLinkError(
        "This password reset link is invalid or has expired. Please request a new one.",
      );
    }
    return { mode: "recovery" };
  }
  try {
    const result = await rawRequest<{ mode: "demo" | "public" }>(
      "/auth/session",
      { signal },
    );
    if (result.mode === "public") {
      const client = await supabase();
      const { data, error } = await client.auth.getSession();
      if (error || !data.session) throw new ApiError("Please sign in", 401);
      if (recoveringPassword) return { mode: "recovery" };
      // A confirmation link can switch Supabase accounts while an old API
      // cookie is still valid. Authorize the current account before showing data.
      await exchange(data.session.access_token);
      return { ...result, email: data.session?.user?.email };
    }
    return result;
  } catch (error) {
    if (!(error instanceof ApiError) || error.status !== 401) throw error;
    try {
      await renewSession();
      const result = await rawRequest<{ mode: "public" }>("/auth/session", {
        signal,
      });
      const client = await supabase();
      const { data } = await client.auth.getSession();
      if (recoveringPassword) return { mode: "recovery" };
      return { ...result, email: data.session?.user?.email };
    } catch (renewalError) {
      if (
        renewalError instanceof ApiError &&
        [401, 403].includes(renewalError.status)
      ) {
        window.dispatchEvent(new Event("session-expired"));
      }
      throw renewalError;
    }
  }
}

export async function login(email: string, password: string) {
  clearAuthRedirect();
  const client = await supabase();
  const { data, error } = await client.auth.signInWithPassword({
    email,
    password,
  });
  if (error) throw new Error(error.message);
  if (!data.session) throw new Error("Sign in did not return a session");
  await exchange(data.session.access_token);
}

export async function signUp(email: string, password: string) {
  clearAuthRedirect();
  const client = await supabase();
  const { data, error } = await client.auth.signUp({
    email,
    password,
    options: { emailRedirectTo: window.location.origin + "/" },
  });
  if (error) throw new Error(error.message);
  if (!data.session) return false;
  await exchange(data.session.access_token);
  return true;
}

export async function sendPasswordReset(email: string) {
  clearAuthRedirect();
  const client = await supabase();
  const { error } = await client.auth.resetPasswordForEmail(email, {
    redirectTo: window.location.origin + "/?auth=recovery",
  });
  if (error) throw new Error(error.message);
}

export async function updatePassword(password: string) {
  const client = await supabase();
  const { error } = await client.auth.updateUser({ password });
  if (error) throw new Error(error.message);
  clearAuthRedirect();
}

export async function logout() {
  signingOut = true;
  try {
    await rawRequest("/auth/session", { method: "DELETE" });
    const client = await supabase();
    const { error } = await client.auth.signOut({ scope: "local" });
    if (error) throw new Error(error.message);
    clearAuthRedirect();
  } finally {
    signingOut = false;
  }
}

export const submitTasks = (tasks: TaskSpec[]) =>
  request<Task[]>("/v1/tasks", {
    method: "POST",
    body: JSON.stringify({
      tasks,
      ...(tasks.some(
        (task) =>
          task.kind === "stub" &&
          typeof task.payload === "object" &&
          task.payload !== null &&
          "fail" in task.payload &&
          task.payload.fail === true,
      )
        ? {
            instructions:
              "This is an intentional failure-handling test. Allow the first execution attempt to run even if you can predict its failure; investigate the observed failure afterward. Do not change the payload.",
          }
        : {}),
    }),
  });
export const cancelTask = (id: string) =>
  request<Task>(`/v1/tasks/${encodeURIComponent(id)}/cancel`, {
    method: "POST",
  });

export const chatConfig = (signal?: AbortSignal) =>
  request<{ enabled: boolean }>("/v1/chat/config", { signal });
export const readChat = (id: string, signal?: AbortSignal) =>
  request<{ id: string; turns: ChatTurn[] }>(
    `/v1/chat/${encodeURIComponent(id)}`,
    { signal },
  );
export const sendChat = (
  id: string,
  message: ChatMessage,
  signal?: AbortSignal,
) =>
  request<ChatTurn>(`/v1/chat/${encodeURIComponent(id)}/messages`, {
    method: "POST",
    body: JSON.stringify(message),
    signal,
  });
export const createDeviceInvite = () =>
  request<{ code: string; expires_in: number; server: string }>(
    "/v1/device-invites",
    { method: "POST" },
  );

export type WorkloadKind =
  "stub" | "echo" | "walker_evolution" | "cpu_inference_batch";

export async function workloadTask(
  kind: WorkloadKind,
  label: string,
  target: string,
  seconds: number,
  failover: boolean,
): Promise<TaskSpec> {
  const task = stubTask(label, target, seconds, failover);
  task.kind = kind;
  if (kind === "echo") task.payload = { nonce: label, sleepMs: 1000 };
  if (kind === "walker_evolution") {
    task.payload = {
      generation: 0,
      parent: Array(308).fill(0),
      sigma: 0.1,
      seeds: [1, 2, 3, 4, 5, 6, 7, 8],
      steps: 600,
    };
    task.timeout_seconds = 120;
  }
  if (kind === "cpu_inference_batch") {
    const catalog = await request<{
      inference: Record<string, unknown> | null;
    }>("/v1/workloads");
    if (!catalog.inference)
      throw new Error("The server has no inference model configured.");
    task.payload = catalog.inference;
    task.timeout_seconds = 300;
  }
  return task;
}
export function stubTask(
  label: string,
  target: string,
  seconds: number,
  failover: boolean,
): TaskSpec {
  return {
    id: "task-" + crypto.randomUUID(),
    job_id: "job-" + crypto.randomUUID(),
    kind: "stub",
    payload: { duration_seconds: seconds, value: { label } },
    requirements: { runtime: "cpu", vram_mib: 0 },
    max_attempts: 3,
    timeout_seconds: seconds + 60,
    target_worker_id: target || null,
    allow_failover: failover,
  };
}

export type SupervisorStatus = {
  enabled: boolean;
  sentry_enabled: boolean;
  job: {
    state: string;
    finalized: boolean;
    memory: {
      findings: { kind: string; text: string; evidence: string[] }[];
      questions: string[];
      followups: { check: string; expected_outcome: string; due_at: string }[];
    };
  };
  runs: { id: string; status: string; reply: string; created_at: string }[];
  actions: {
    action_id: string;
    request: { operation: string; reason: string };
    result: { state?: string };
  }[];
};

export const readSupervisor = (jobId: string, signal?: AbortSignal) =>
  request<SupervisorStatus>(
    `/v1/jobs/${encodeURIComponent(jobId)}/supervisor`,
    { signal },
  );

export interface SimulationStatus {
  trial_counts?: Record<string, number>;
  cleanup?: { required: number; confirmed: number; pending_workers: string[] };
  job_id: string;
  phase: string;
  message: string;
  description: string;
  round: number;
  deadline: string;
  original_hash: string;
  validated_hash?: string;
  limits: { adaptations: number; runtime_seconds: number; workers: number };
  workers: string[];
  question?: string;
  plan?: {
    summary: string;
    trials: number;
    batch_size?: number;
    workers?: number;
    reference: string;
    aggregate: string;
  };
  policy?: {
    rationale: string;
    local_cases: number;
    independent_cases: number;
    validation_timeout_seconds: number;
    aggregation: string;
  };
  schedule?: {
    rationale: string;
    batches: { worker_id: string; trials: number; timeout_seconds: number }[];
    aggregation_worker: string;
    aggregation_timeout_seconds: number;
  };
  measurements?: {
    stage: string;
    worker_id: string;
    tasks: number;
    trials: number;
    compute_seconds: number;
    execution_seconds: number;
    observed_wall_seconds: number;
    output_bytes: number;
  }[];
  decisions?: {
    stage: string;
    tool: string;
    proposal: { rationale?: string; summary?: string; explanation?: string };
  }[];
  versions: {
    round: number;
    digest: string;
    explanation: string;
    code: string;
  }[];
  checks: {
    round: number;
    stage: string;
    passed: boolean;
    cases?: number;
    details?: unknown;
    seeds: number[];
  }[];
  tasks: {
    id: string;
    state: string;
    generation: number;
    worker_id: string | null;
    role: string;
    failure: string;
    progress: number;
  }[];
}
export const readSimulation = (jobId: string, signal?: AbortSignal) =>
  request<SimulationStatus>(`/v1/simulations/${encodeURIComponent(jobId)}`, {
    signal,
  });
export const answerSimulation = (jobId: string, message: string) =>
  request(`/v1/simulations/${encodeURIComponent(jobId)}/answer`, {
    method: "POST",
    body: JSON.stringify({ message }),
  });
export async function uploadSimulation(
  files: File[],
  description: string,
  requestId: string,
) {
  const encoded = await Promise.all(
    files.map(async (file) => {
      const bytes = new Uint8Array(await file.arrayBuffer());
      let text = "";
      for (let i = 0; i < bytes.length; i += 8192)
        text += String.fromCharCode(...bytes.subarray(i, i + 8192));
      return { name: file.name, content: btoa(text) };
    }),
  );
  return request<Task>("/v1/simulations", {
    method: "POST",
    body: JSON.stringify({
      request_id: requestId,
      description,
      files: encoded,
    }),
  });
}
