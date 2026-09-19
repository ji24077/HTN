import { createClient, type SupabaseClient } from "@supabase/supabase-js";
import type { Task, TaskSpec } from "./types";

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
function supabase(): Promise<SupabaseClient> {
  if (!authClient) {
    authClient = rawRequest<{ url: string; publishableKey: string }>(
      "/auth/config",
    )
      .then(({ url, publishableKey }) => {
        const client = createClient(url, publishableKey);
        client.auth.onAuthStateChange((event) => {
          if (event === "SIGNED_OUT") {
            // Defer work outside the SDK's auth callback lock.
            setTimeout(() => {
              void rawRequest("/auth/session", { method: "DELETE" }).catch(
                () => {},
              );
              window.dispatchEvent(new Event("session-expired"));
            }, 0);
          }
        });
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
): Promise<{ mode: "demo" | "public" }> {
  try {
    const result = await rawRequest<{ mode: "demo" | "public" }>(
      "/auth/session",
      { signal },
    );
    if (result.mode === "public") await supabase();
    return result;
  } catch (error) {
    if (!(error instanceof ApiError) || error.status !== 401) throw error;
    try {
      await renewSession();
      return await rawRequest("/auth/session", { signal });
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
  const client = await supabase();
  const { data, error } = await client.auth.signInWithPassword({
    email,
    password,
  });
  if (error) throw new Error(error.message);
  if (!data.session) throw new Error("Sign in did not return a session");
  await exchange(data.session.access_token);
}

export async function logout() {
  await rawRequest("/auth/session", { method: "DELETE" });
  const client = await supabase();
  const { error } = await client.auth.signOut({ scope: "local" });
  if (error) throw new Error(error.message);
}

export const submitTasks = (tasks: TaskSpec[]) =>
  request<Task[]>("/v1/tasks", {
    method: "POST",
    body: JSON.stringify({ tasks }),
  });
export const cancelTask = (id: string) =>
  request<Task>(`/v1/tasks/${encodeURIComponent(id)}/cancel`, {
    method: "POST",
  });
export function stubTask(
  label: string,
  target: string,
  seconds: number,
  failover: boolean,
): TaskSpec {
  return {
    id: "task-" + crypto.randomUUID(),
    job_id: "playground",
    kind: "stub",
    payload: { duration_seconds: seconds, value: { label } },
    requirements: { runtime: "cpu", vram_mib: 0 },
    max_attempts: 3,
    timeout_seconds: seconds + 60,
    target_worker_id: target || null,
    allow_failover: failover,
  };
}
