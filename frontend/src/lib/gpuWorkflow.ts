import {
  LIVE_STATUSES,
  type Experiment,
  type LabJob,
  type LabModel,
  type Pod,
  type RecordedEvidence,
  type Serving,
} from "../api/gpulab";
import { migrationAction, verification } from "./gpulab";

export type GpuWorkflowContext = {
  pods: Pod[];
  models: LabModel[];
  experiment: Experiment | null;
  jobs: LabJob[];
  evidence: RecordedEvidence | null;
  readOnly: boolean;
  selectedPodId?: string;
  selectedModelId?: string;
  serving?: Serving;
};

export type GpuWorkflowPlan =
  | {
      kind: "action";
      title: string;
      explanation: string;
      path: string;
      body: Record<string, unknown>;
      requiresApproval: true;
      warnings?: string[];
    }
  | {
      kind: "evidence";
      gpuKey?: string;
      comparison?: "optimization" | "migration_from_4090";
      recheck?: boolean;
    }
  | { kind: "status" }
  | { kind: "help"; text: string }
  | { kind: "blocked"; text: string };

const HELP =
  "Workflow commands create a reviewable plan; they do not call an AI planner. " +
  "Try /train-baseline, /optimize-training, /optimize-inference, /generate-data, " +
  "/migrate RTX 4090 -> MI300X, /serve ft on RTX 4090, /unload, " +
  "/compare saved MI300X, /recheck saved MI300X, /status, or /help. " +
  "Choose a pod and model above, or name them explicitly. Run one command at a time. " +
  "Budget caps, deadlines, and automatic deployment are not supported. Use Model mode for model replies.";

const blocked = (text: string): GpuWorkflowPlan => ({ kind: "blocked", text });
const compact = (value: string) =>
  value.toLowerCase().replace(/[^a-z0-9]/g, "");

function gpuAliases(gpu: string, vendor: string): string[] {
  const short = gpu
    .replace(/^(?:nvidia|amd)\s+/i, "")
    .replace(/^geforce\s+/i, "");
  return [
    gpu,
    short,
    `${vendor} ${short}`,
    short.replace(/^rtx\s*/i, ""),
    vendor,
    ...(short.match(/\b(?:mi\d+[a-z]*|[ahl]\d+[a-z]*|\d{4}(?:\s*ti)?)\b/gi) ||
      []),
  ].map(compact);
}

type Selection<T> = { value: T } | { error: string };

function choosePod(query: string, context: GpuWorkflowContext): Selection<Pod> {
  const name = query.trim().replace(/^(?:the\s+)?(?:gpu|pod)\s+/i, "");
  let matches: Pod[];
  if (name) {
    const wanted = compact(name);
    matches = context.pods.filter((pod) =>
      [
        compact(pod.id),
        compact(pod.name),
        ...gpuAliases(pod.gpu, pod.vendor),
      ].includes(wanted),
    );
  } else if (context.selectedPodId) {
    matches = context.pods.filter((pod) => pod.id === context.selectedPodId);
  } else {
    matches = context.pods.filter((pod) => pod.status === "running");
  }
  if (!matches.length)
    return {
      error: name
        ? `No available pod matches “${name}”. Choose a listed pod by name or ID.`
        : "Select an available GPU pod before planning this command.",
    };
  if (matches.length > 1)
    return {
      error: `More than one pod matches${name ? ` “${name}”` : " this request"}. Specify a pod name or ID: ${matches.map((pod) => `${pod.name} (${pod.id})`).join(", ")}.`,
    };
  if (matches[0].status !== "running")
    return {
      error: `${matches[0].name} is not running. This planner does not rent or start GPU pods.`,
    };
  return { value: matches[0] };
}

function chooseModel(
  query: string,
  context: GpuWorkflowContext,
): Selection<LabModel> {
  const name = query
    .trim()
    .replace(/^the\s+/i, "")
    .replace(/^model\s+/i, "");
  const generic = !name || name === "model";
  const matches = generic
    ? context.selectedModelId
      ? context.models.filter((model) => model.id === context.selectedModelId)
      : context.models
    : context.models.filter((model) => {
        const aliases = [model.id, model.label, `${model.kind} model`];
        if (model.kind === "finetuned")
          aliases.push("fine-tuned", "fine-tuned model");
        if (model.kind === "agent") aliases.push("optimized model");
        return aliases.map(compact).includes(compact(name));
      });
  if (matches.length !== 1)
    return {
      error: matches.length
        ? "More than one model matches. Select a model or specify its exact ID."
        : "No available model matches. Select a listed model or specify its exact ID.",
    };
  return { value: matches[0] };
}

function acceptedBaseline(context: GpuWorkflowContext): boolean {
  return context.jobs.some(
    (job) =>
      job.kind === "train-and-evaluate" && verification(job).label === "Passed",
  );
}

function action(
  title: string,
  explanation: string,
  path: string,
  body: Record<string, unknown>,
  warnings: string[] = [],
): GpuWorkflowPlan {
  return {
    kind: "action",
    title,
    explanation,
    path,
    body,
    requiresApproval: true,
    warnings,
  };
}

function savedEvidence(
  text: string,
  context: GpuWorkflowContext,
): GpuWorkflowPlan {
  if (!context.evidence)
    return blocked(
      "Saved GPU evidence is unavailable. Refresh the lab and try again.",
    );
  const recheck = /^(?:recheck|verify)\b/.test(text);
  const comparison =
    /\b(?:from|against)\s+(?:the\s+)?(?:nvidia\s+)?(?:rtx\s*)?4090\b/.test(text)
      ? "migration_from_4090"
      : /\b(?:own baseline|same[- ]gpu|optimization)\b/.test(text)
        ? "optimization"
        : undefined;
  const query = text
    .replace(/^(?:compare|recheck|verify)(?:\s+the)?\s*/, "")
    .replace(
      /\b(?:from|against)\s+(?:the\s+)?(?:nvidia\s+)?(?:rtx\s*)?4090\b/,
      "",
    )
    .replace(
      /\b(?:own baseline|same[- ]gpu|optimization|saved|recorded|evidence|results|runs?|gpus?)\b/g,
      "",
    )
    .trim();
  if (!query) return { kind: "evidence", comparison, recheck };
  const rows = context.evidence.rows.filter((row) =>
    [compact(row.key), ...gpuAliases(row.gpu, row.vendor)].includes(
      compact(query),
    ),
  );
  if (rows.length !== 1)
    return blocked(
      rows.length
        ? "Name one saved GPU to compare or recheck; the vendor matches several chips."
        : `No saved GPU matches “${query}”. Try /compare to browse the recorded chips.`,
    );
  if (recheck && !rows[0].measured)
    return blocked(
      `${rows[0].gpu} has no completed measurement to recheck. Use /compare to inspect the limitation.`,
    );
  return { kind: "evidence", gpuKey: rows[0].key, comparison, recheck };
}

/** Pure command planning. Call only in Workflow mode; never executes a request. */
export function planGpuWorkflow(
  input: string,
  context: GpuWorkflowContext,
): GpuWorkflowPlan {
  const text = input
    .trim()
    .toLowerCase()
    .replace(/^please\s+/, "")
    .replace(/^\//, "")
    .replace(/\s+please[.!]?$/, "")
    .replace(/[.!?]+$/, "")
    .replace(/\s+/g, " ");
  if (!text || /^(?:help|commands|what can you do)$/.test(text))
    return { kind: "help", text: HELP };
  if (
    /^(?:status|show status|job status|what is running|what's running)$/.test(
      text,
    )
  )
    return { kind: "status" };

  // A request must not lose a constraint just because this fixed workflow has
  // no field for it. These requirements need a different planning capability.
  if (
    /[$€£]|\b(?:budget|dollars?|usd|cad|eur|deadline|tomorrow|tonight|today|by\s+\d|within|under\s+\d|less than|at most|no more than|before\s+\d|in\s+\d+\s*(?:minutes?|hours?|days?))\b/.test(
      text,
    )
  )
    return blocked(
      "This planner cannot enforce a spending cap or completion deadline. No job was planned. Remove that requirement only if you intend to approve an uncapped experiment.",
    );
  if (
    /\b(?:deploy|traffic|rollback|quantiz\w*|qlora|lora|vllm|tensorrt|rent|lease|cheapest|fastest|automatically)\b/.test(
      text,
    )
  )
    return blocked(
      "That request needs unsupported deployment, provisioning, runtime, or optimization choices. No job was planned. Use /help for the bounded workflows available here.",
    );
  if (/^(?:compare|recheck|verify)\b/.test(text))
    return savedEvidence(text, context);

  let command:
    "train" | "training" | "inference" | "data" | "migrate" | "serve" | "stop";
  let tail = "";
  const patterns: [typeof command, RegExp][] = [
    [
      "train",
      /^(?:train-baseline|train(?: the| a)? baseline|train|run baseline training)(?:\s+(.*))?$/,
    ],
    [
      "training",
      /^(?:optimize-training|optimize training|run(?: the)? training agent)(?:\s+(.*))?$/,
    ],
    [
      "inference",
      /^(?:optimize-inference|optimize inference|optimize batching|run(?: the)? (?:inference|batching) agent)(?:\s+(.*))?$/,
    ],
    [
      "data",
      /^(?:generate-data|generate data|generate training data|data)(?:\s+(.*))?$/,
    ],
    ["migrate", /^(?:migrate|test migration)(?:\s+(.*))?$/],
    ["serve", /^(?:serve|load)(?:\s+(.*))?$/],
    ["stop", /^(?:stop|unload)(?:\s+(.*))?$/],
  ];
  const parsed = patterns
    .map(([kind, pattern]) => ({ kind, match: text.match(pattern) }))
    .find((entry) => entry.match);
  if (!parsed?.match)
    return blocked(
      "I could not map that request to one supported workflow. Try /help, or use Model mode for a model reply.",
    );
  command = parsed.kind;
  tail = parsed.match[1] || "";
  if (context.readOnly)
    return blocked(
      "This server is in recorded-evidence mode. Live jobs and model changes are disabled; /compare, /recheck, and /status are available.",
    );
  if (context.jobs.some((job) => LIVE_STATUSES.includes(job.status)))
    return blocked(
      "A job is still active. Wait for it to finish or cancel it in the job controls before planning another change.",
    );

  if (command === "data") {
    if (tail)
      return blocked(
        "Data generation supports the displayed defaults only: 2,000 examples, 200 held out, 8 workers. Use /generate-data for that plan.",
      );
    return action(
      "Generate training data",
      "Generate 2,000 examples with 200 held out using 8 workers.",
      "/api/jobs/data",
      { total: 2000, heldout: 200, workers: 8 },
      [
        "Uses the configured data-generation service; its cost is not capped or estimated.",
      ],
    );
  }
  if (command === "stop") {
    if (tail && !/^(?:the )?(?:model|inference|serving)$/.test(tail))
      return blocked(
        "Use /unload to stop the currently served model. This command does not cancel jobs or stop GPU pods.",
      );
    if (!context.serving)
      return blocked("Refresh serving status before requesting an unload.");
    if (!context.serving.running)
      return blocked(
        "No model is currently loaded. There is nothing to unload.",
      );
    return action(
      "Unload the current model",
      "Stop the currently served model. This frees its serving resources and interrupts inference; the rented pod remains running.",
      "/api/serve/stop",
      {},
      ["This stops serving; it does not stop pod billing."],
    );
  }
  if (command === "migrate") {
    if (!acceptedBaseline(context))
      return blocked(
        "A baseline with a passing current evaluation is required before a migration experiment. Recorded metrics alone are not an accepted baseline.",
      );
    const pair = tail.replace(/^from\s+/, "").split(/\s*(?:->|→)\s*|\s+to\s+/);
    if (pair.length !== 2 || !pair[0] || !pair[1])
      return blocked(
        "Specify both endpoints, for example: /migrate RTX 4090 -> MI300X. This tests checkpoint portability; it does not switch live traffic.",
      );
    const source = choosePod(pair[0], context);
    const target = choosePod(pair[1], context);
    if ("error" in source) return blocked(source.error);
    if ("error" in target) return blocked(target.error);
    const route = migrationAction(source.value, target.value);
    if (!route)
      return blocked(
        "That GPU pair has no supported migration workflow. Choose distinct NVIDIA generations, NVIDIA to AMD, or AMD to NVIDIA.",
      );
    return action(
      route.label,
      route.detail,
      `/api/jobs/action/${route.name}`,
      {
        source_pod_id: source.value.id,
        target_pod_id: target.value.id,
        ...(route.name === "migrate-nvidia-amd"
          ? { total_steps: 8, stop_after: 4, eval_n: 50 }
          : {}),
      },
      [
        "This is a bounded experiment, not a production deployment or automatic traffic switch.",
        "Both pods may incur charges; this plan enforces no budget cap.",
      ],
    );
  }
  if (command === "serve") {
    if (!context.serving)
      return blocked("Refresh serving status before loading a model.");
    if (context.serving.running)
      return blocked(
        "A model is already serving. Review and approve /unload first; loading will not silently replace the active model.",
      );
    const split =
      tail.match(/^(.*?)\s+on\s+(.+)$/) || tail.match(/^(on)\s+(.+)$/);
    const modelQuery = split ? (split[1] === "on" ? "" : split[1]) : tail;
    const pod = choosePod(split?.[2] || "", context);
    const model = chooseModel(modelQuery, context);
    if ("error" in pod) return blocked(pod.error);
    if ("error" in model) return blocked(model.error);
    if (
      ["finetuned", "agent"].includes(model.value.kind) &&
      !acceptedBaseline(context)
    )
      return blocked(
        "An accepted baseline is required before loading a trained candidate. A saved benchmark pass does not approve this model.",
      );
    return action(
      `Load ${model.value.label}`,
      `Load ${model.value.label} on ${pod.value.name} (${pod.value.gpu}) with bf16 precision. This starts inference serving; it does not approve the model's quality.`,
      "/api/serve",
      { pod_id: pod.value.id, model_id: model.value.id, dtype: "bf16" },
      ["GPU costs continue while the pod runs. No spending cap is enforced."],
    );
  }

  if (tail && !/^on\s+.+$/.test(tail))
    return blocked(
      "Extra settings or multiple actions are not supported in one command. Name a GPU with “on <pod name or ID>”, or use /help.",
    );
  const pod = choosePod(tail.replace(/^on\s+/, ""), context);
  if ("error" in pod) return blocked(pod.error);
  if (command === "train") {
    if (!context.experiment?.data.train || !context.experiment.data.heldout)
      return blocked(
        "Training and held-out data are required first. Review /generate-data, then plan baseline training.",
      );
    return action(
      "Train the baseline",
      `Run the configured Qwen baseline on ${pod.value.name}: 500 steps, bf16, SDPA, micro-batch 16, accumulation 1, followed by evaluation.`,
      "/api/jobs/train",
      {
        pod_id: pod.value.id,
        steps: 500,
        dtype: "bf16",
        attention: "sdpa",
        micro_batch: 16,
        grad_accum: 1,
      },
      [
        "This uses the existing training workflow, not automatic LoRA/QLoRA selection. No spending cap is enforced.",
      ],
    );
  }
  if (!acceptedBaseline(context))
    return blocked(
      "Finish a baseline with a passing current evaluation first. Historical metrics or a completed job without accepted validation are insufficient.",
    );
  return command === "training"
    ? action(
        "Optimize training settings",
        `Measure batch and accumulation candidates on ${pod.value.name}, retrain with the selected configuration, and evaluate against the accepted baseline.`,
        "/api/jobs/action/optimize-training",
        { pod_id: pod.value.id },
        [
          "Quality is evaluated after the experiment; this is not a guarantee that every answer stays unchanged.",
          "No spending cap is enforced.",
        ],
      )
    : action(
        "Compare sequential and batched inference",
        `Measure the same checkpoint and 64 inputs sequentially and batched on ${pod.value.name}. Accept only the recorded quality and output checks.`,
        "/api/jobs/action/optimize-inference",
        { pod_id: pod.value.id },
        [
          "This tests batching, not arbitrary runtime, quantization, routing, or cache optimization.",
          "No spending cap is enforced.",
        ],
      );
}
