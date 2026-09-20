import { lab, type Generation, type LabJob } from "../api/gpulab";

export const pct = (v?: number | null) =>
  v == null ? "—" : (Number(v) * 100).toFixed(1) + "%";
export const secs = (v?: number | null, digits = 2) =>
  v == null || !Number.isFinite(Number(v))
    ? "—"
    : Number(v).toFixed(digits) + "s";
export const gpuName = (gpu?: string) =>
  (gpu || "?").replace("NVIDIA ", "").replace("AMD ", "");
export const duration = (job?: LabJob | null) =>
  job?.started_at && job.finished_at ? job.finished_at - job.started_at : null;

// What the agent is doing, in the words of someone watching it — not the
// internal job kind.
export const JOB_LABEL: Record<string, string> = {
  "train-and-evaluate": "Training + automatic evaluation",
  "optimize-training-speed": "Training setup optimization",
  "optimize-inference-speed": "Inference throughput optimization",
  "migrate-nextgen": "Cross-generation migration",
  "migrate-amd-nvidia": "AMD → NVIDIA migration",
  "migrate-nvidia-amd": "NVIDIA → AMD migration",
  "serve-model": "Loading model",
  "generate-data": "Generating training data",
};

export function friendlyFailure(job?: LabJob | null) {
  const text = (job?.error || "") + " " + (job?.logs || []).join(" ");
  if (/No space left on device|disk free|disk space/i.test(text))
    return "The GPU pod ran out of disk space and could not save the checkpoint. Free up disk or run on another pod.";
  if (/no pod has room/i.test(text))
    return "No pod has enough VRAM or disk for training. Unload the running model or choose another pod.";
  if (/no model is loaded/i.test(text))
    return "The inference model is not ready. Pick a model and load it onto a GPU first.";
  return job?.error || "The job did not finish. Check the detailed log.";
}

// ── long context: the policy is the cached prefix ───────────────────────────
const POLICY_SECTION =
  "Section {n}. Access control and credential handling. Employees must not transmit " +
  "credentials, API keys, or private tokens to any system that has not been approved by the " +
  "security review board. Approved destinations are listed in the internal registry and are " +
  "reviewed quarterly. Any upload of secret material to an external or unrecognised domain is " +
  "treated as a credential exposure event and must be reported within one hour. The on-call " +
  "responder revokes the affected credential, rotates dependent secrets, and opens an incident " +
  "record. Severity is assigned on a scale of zero through five, where four or higher requires " +
  "notification of the data protection officer. Contractors are held to the same standard as " +
  "full time staff. Repeated violations escalate to account suspension pending review. ";

// The POLICY alone is the cached prefix. The instruction is deliberately NOT
// in it: one cache serves both a short benchmark answer and a longer verdict,
// and the instruction is what picks between them.
export function policyText() {
  let out = "";
  for (let i = 1; i <= 215; i++)
    out += POLICY_SECTION.replace("{n}", String(i));
  // Qwen's user-turn opener belongs at the head of the cached prefix.
  return "<|im_start|>user\n" + out + "\n\n---\n";
}

// Closes the user turn and opens the assistant's. Without it an instruct model
// fed raw text continues the document instead of answering.
export const TURN_END = "<|im_end|>\n<|im_start|>assistant\n";

// Short and deterministic, so the benchmark isolates prefill.
const BENCH_INSTRUCTION =
  "Apply the policy above. For the event below return only JSON with keys " +
  "risk, severity, action.\n\nEvent:\n";

// Four short lines: long enough to watch appear, short enough that the total
// is dominated by prefill — which is the thing being optimised.
export const VERDICT_INSTRUCTION =
  "Apply the policy above. For the event below reply with exactly four lines " +
  "and nothing else:\nRISK: <one phrase>\nSEVERITY: <0-5>\nACTION: <one sentence>\n" +
  "ESCALATE: <yes or no, and to whom>\n\nEvent:\n";

export const EXAMPLE_EVENT =
  "A contractor emailed an API key to a personal address.";

const REPEATS = 3;
const pctile = (xs: number[], q: number) => {
  const a = xs.slice().sort((m, n) => m - n);
  return a[Math.min(a.length - 1, Math.floor(q * (a.length - 1) + 0.5))];
};

export type BenchmarkResult = {
  repeats: number;
  deterministic: boolean;
  identical: boolean;
  diverge: string;
  promptTokens: number;
  prefixTokens: number;
  reusePct: number;
  missP50: number;
  missP95: number;
  hitP50: number;
  hitP95: number;
};

/** Cache-miss vs cache-hit prefill, three runs each after a discarded warm-up. */
export async function runPrefillBenchmark(
  event: string,
  onProgress: (note: string) => void,
): Promise<BenchmarkResult> {
  const body = (no_cache: boolean) => ({
    sentence: BENCH_INSTRUCTION + event + TURN_END,
    max_new_tokens: 48,
    greedy: true,
    no_cache,
  });
  // The first call after a load pays one-off CUDA and allocator costs that
  // belong to neither arm.
  onProgress("Warming up (not measured)…");
  await lab<Generation>("/api/generate", body(true));
  const off: number[] = [];
  const on: number[] = [];
  const offOut: string[] = [];
  const onOut: string[] = [];
  let last: Generation | null = null;
  for (let i = 0; i < REPEATS; i++) {
    onProgress(`Cache off ${i + 1}/${REPEATS}…`);
    const a = await lab<Generation>("/api/generate", body(true));
    onProgress(`Cache on ${i + 1}/${REPEATS}…`);
    const b = await lab<Generation>("/api/generate", body(false));
    off.push(a.latency_s);
    on.push(b.latency_s);
    offOut.push(a.raw_output || "");
    onOut.push(b.raw_output || "");
    last = b;
  }
  // Each arm must repeat itself exactly. Across arms that is NOT guaranteed:
  // different tensor shapes pick different kernels, so a bf16 near-tie can flip.
  const identical = offOut[0] === onOut[0];
  let diverge = "";
  if (!identical) {
    let k = 0;
    while (k < offOut[0].length && offOut[0][k] === onOut[0][k]) k++;
    diverge = `character ${k}: off ${JSON.stringify(offOut[0].slice(k, k + 14))} vs on ${JSON.stringify(onOut[0].slice(k, k + 14))}`;
  }
  const promptTokens = last?.prompt_tokens || 0;
  const prefixTokens = last?.prefix_tokens || 0;
  return {
    repeats: REPEATS,
    deterministic:
      offOut.every((o) => o === offOut[0]) &&
      onOut.every((o) => o === onOut[0]),
    identical,
    diverge,
    promptTokens,
    prefixTokens,
    reusePct: promptTokens
      ? Math.floor((prefixTokens / promptTokens) * 1000) / 10
      : 0,
    missP50: pctile(off, 0.5),
    missP95: pctile(off, 0.95),
    hitP50: pctile(on, 0.5),
    hitP95: pctile(on, 0.95),
  };
}
