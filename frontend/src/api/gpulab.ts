// Client for the GPUShare experiment server (src/gpushare/dashboard). It is a
// separate process from the orchestrator, so requests carry no credentials.
export const GPULAB_URL = (
  (import.meta.env.VITE_GPUSHARE_URL as string | undefined) ||
  "http://127.0.0.1:8080"
).replace(/\/$/, "");

export type Pod = {
  id: string;
  name: string;
  gpu: string;
  vendor: string;
  status: string;
  cost_per_hour: number;
};

export type LabModel = {
  id: string;
  label: string;
  kind: string;
  pod_id?: string | null;
};

export type Serving = {
  running: boolean;
  model_id?: string;
  pod_id?: string;
  dtype?: string;
  prefix_tokens?: number;
  prefix_build_s?: number;
};

export type Evaluation = {
  exact_match_rate?: number;
  field_accuracy?: Record<string, number>;
  hallucination_rate?: number;
  omission_rate?: number;
  held_out_loss?: number;
  n_nullable?: number;
};

export type TrainRun = {
  t_step_median_s?: number;
  peak_vram_gb?: number;
  dtype?: string;
  attention?: string;
  steps?: number;
  micro_batch?: number;
  grad_accum?: number;
  tokens_per_step?: number;
  loss_history?: number[];
  start_step?: number;
  loss_curve?: { step: number; loss: number }[];
};

export type Experiment = {
  before?: Evaluation | null;
  after?: Evaluation | null;
  train?: TrainRun | null;
  data: { train: number; heldout: number };
};

type Throughput = { elapsed_s?: number; tokens_per_second?: number };

export type LabJob = {
  id: string;
  kind: string;
  status: string;
  stage?: string;
  progress?: number;
  started_at?: number;
  finished_at?: number;
  error?: string;
  logs?: string[];
  result?: {
    train?: TrainRun;
    after?: Evaluation;
    chosen?: { micro_batch?: number; grad_accum?: number };
    candidates?: (TrainRun & { micro_batch: number; grad_accum: number })[];
    quality?: { chosen?: Evaluation };
    baseline?: Throughput;
    optimized?: Throughput;
    speedup?: number;
    validation?: { status?: string };
  };
};

export type GenerateBody = {
  sentence: string;
  max_new_tokens: number;
  greedy: boolean;
  no_cache: boolean;
};

export type Generation = {
  raw_output?: string;
  latency_s: number;
  ttft_s?: number;
  prompt_tokens?: number;
  prefix_tokens?: number;
  new_tokens?: number;
  cache?: string;
};

export type PrefixResult = {
  cleared?: boolean;
  prefix_tokens: number;
  build_s: number;
};

export const LIVE_STATUSES = ["queued", "running", "cancelling"];

export async function lab<T>(path: string, body?: unknown): Promise<T> {
  const response = await fetch(GPULAB_URL + path, {
    method: body === undefined ? "GET" : "POST",
    headers: { "Content-Type": "application/json" },
    body: body === undefined ? undefined : JSON.stringify(body),
  });
  const data = await response.json().catch(() => ({}));
  if (!response.ok)
    throw new Error(
      typeof data.detail === "string"
        ? data.detail
        : `Request failed (${response.status})`,
    );
  return data as T;
}

/** Streams one reply, reporting each token; resolves with the final timing frame. */
export async function streamGenerate(
  body: GenerateBody,
  onToken: (token: string) => void,
  signal?: AbortSignal,
): Promise<Generation> {
  const response = await fetch(GPULAB_URL + "/api/generate/stream", {
    method: "POST",
    headers: { "Content-Type": "application/json" },
    body: JSON.stringify(body),
    signal,
  });
  if (!response.ok || !response.body)
    throw new Error(`Stream failed (${response.status})`);
  const reader = response.body.getReader();
  const decoder = new TextDecoder();
  let buffer = "";
  let done: Generation | null = null;
  for (;;) {
    const part = await reader.read();
    if (part.done) break;
    buffer += decoder.decode(part.value, { stream: true });
    const frames = buffer.split("\n\n");
    buffer = frames.pop() || "";
    for (const frame of frames) {
      const line = frame.split("\n").find((x) => x.startsWith("data: "));
      if (!line) continue;
      const data = JSON.parse(line.slice(6));
      if (data.token) onToken(data.token);
      if (data.error) throw new Error(data.error);
      if (data.done) done = data;
    }
  }
  if (!done) throw new Error("The stream ended before the reply finished.");
  return done;
}
