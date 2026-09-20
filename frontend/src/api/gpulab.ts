// The dev server proxies /api to GPUShare on its own port. An explicit public
// URL is optional; backend credentials must never use a VITE_ variable.
export const GPULAB_URL = (
  (import.meta.env.VITE_GPUSHARE_URL as string | undefined) || ""
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
  ref?: string;
  detail?: string;
};

export type Serving = {
  running: boolean;
  model_id?: string;
  model_ref?: string;
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
  comparison?: { status: string; detail: string };
};

export type EvidenceVerdict = {
  status: "passed" | "rejected" | "not_validated";
  cases: number;
  changed_cases: number | null;
  reference_correct: number | null;
  candidate_correct: number | null;
  examples?: {
    source_index?: number;
    id?: string;
    sentence?: string;
    fields: string[];
    before: unknown;
    after: unknown;
    expected: unknown;
  }[];
};

export type EvidenceRecheck = {
  gpu_key: string;
  comparison: "optimization" | "migration_from_4090";
  source: "saved_outputs";
  checked_at: string;
  verdict: EvidenceVerdict;
  status: EvidenceVerdict["status"];
  model_sha256: string;
  dataset_sha256: string;
  recorded_speed_gate_passed: boolean;
  detail: string;
};

export type RecordedEvidence = {
  kind: "recorded_hardware_evidence";
  recorded_at: string;
  checkpoint: string;
  model_sha256: string;
  dataset_sha256: string;
  model_status: string;
  timing_scope: string;
  sampling_note: string;
  quality_note: string;
  rows: {
    key: string;
    gpu: string;
    vendor: string;
    measured: boolean;
    baseline_s: number | null;
    candidate_s: number | null;
    latency_ratio: number | null;
    optimization: EvidenceVerdict;
    migration_from_4090: EvidenceVerdict;
  }[];
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
    remote_checkpoint?: string;
    model?: string;
    migration_mode?: string;
    train?: TrainRun;
    after?: Evaluation;
    chosen?: { micro_batch?: number; grad_accum?: number };
    candidates?: (TrainRun & { micro_batch: number; grad_accum: number })[];
    quality?: { chosen?: Evaluation };
    baseline?: Throughput;
    optimized?: Throughput;
    speedup?: number;
    validation?: {
      status?: string;
      detail?: string;
      tolerance?: number;
      policy?: string;
      cases?: number;
      changed_outputs?: number;
      output_preservation_verified?: boolean;
      evaluation?: { dataset_sha256?: string };
    };
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
