import { describe, expect, test } from "vitest";
import type { LabJob, Pod, RecordedEvidence } from "../api/gpulab";
import { planGpuWorkflow, type GpuWorkflowContext } from "../lib/gpuWorkflow";

const nvidia: Pod = {
  id: "nv-a",
  name: "trainer",
  gpu: "NVIDIA GeForce RTX 4090",
  vendor: "nvidia",
  status: "running",
  cost_per_hour: 0.74,
};
const amd: Pod = {
  id: "amd-a",
  name: "candidate",
  gpu: "AMD MI300X",
  vendor: "amd",
  status: "running",
  cost_per_hour: 2,
};
const l40: Pod = {
  ...nvidia,
  id: "nv-b",
  name: "l40-host",
  gpu: "NVIDIA L40S",
};
const baseline: LabJob = {
  id: "accepted-baseline",
  kind: "train-and-evaluate",
  status: "complete",
  result: {
    validation: {
      status: "ok",
      tolerance: 0,
      policy: "aggregate_no_regression",
      evaluation: { dataset_sha256: "frozen-data" },
    },
  },
};
const verdict = {
  status: "rejected" as const,
  cases: 300,
  changed_cases: 13,
  reference_correct: 270,
  candidate_correct: 270,
};
const evidence: RecordedEvidence = {
  kind: "recorded_hardware_evidence",
  recorded_at: "2026-09-19",
  checkpoint: "experimental v3b",
  model_sha256: "model-hash",
  dataset_sha256: "data-hash",
  model_status: "Experimental",
  timing_scope: "Resident inference",
  sampling_note: "One measured run",
  quality_note: "Exact JSON values",
  rows: [
    {
      key: "mi300x",
      gpu: "AMD MI300X",
      vendor: "amd",
      measured: true,
      baseline_s: 1.136,
      candidate_s: 0.248,
      latency_ratio: 4.58,
      optimization: verdict,
      migration_from_4090: { ...verdict, changed_cases: 15 },
    },
    {
      key: "4090",
      gpu: "NVIDIA RTX 4090",
      vendor: "nvidia",
      measured: true,
      baseline_s: 0.639,
      candidate_s: 0.142,
      latency_ratio: 4.5,
      optimization: { ...verdict, changed_cases: 2 },
      migration_from_4090: verdict,
    },
    {
      key: "5090",
      gpu: "NVIDIA RTX 5090",
      vendor: "nvidia",
      measured: false,
      baseline_s: null,
      candidate_s: null,
      latency_ratio: null,
      optimization: { ...verdict, status: "not_validated" },
      migration_from_4090: { ...verdict, status: "not_validated" },
    },
  ],
};
const context = (
  changes: Partial<GpuWorkflowContext> = {},
): GpuWorkflowContext => ({
  pods: [nvidia, amd, l40],
  models: [
    { id: "ft", label: "Fine-tuned Qwen", kind: "finetuned" },
    { id: "base", label: "Base Qwen", kind: "base" },
  ],
  experiment: {
    data: { train: 1800, heldout: 200 },
    after: { exact_match_rate: 0.9 },
  },
  jobs: [baseline],
  evidence,
  readOnly: false,
  selectedPodId: nvidia.id,
  selectedModelId: "ft",
  serving: { running: false },
  ...changes,
});

describe("bounded commands map to the existing API", () => {
  test.each([
    "/train-baseline",
    "/train baseline",
    "train the baseline",
    "Please run baseline training!",
  ])("baseline defaults for %s", (command) => {
    expect(planGpuWorkflow(command, context())).toMatchObject({
      kind: "action",
      path: "/api/jobs/train",
      requiresApproval: true,
      body: {
        pod_id: "nv-a",
        steps: 500,
        dtype: "bf16",
        attention: "sdpa",
        micro_batch: 16,
        grad_accum: 1,
      },
    });
  });
  test.each([
    ["optimize training on L40S", "/api/jobs/action/optimize-training", "nv-b"],
    ["/optimize-training", "/api/jobs/action/optimize-training", "nv-a"],
    ["run the training agent", "/api/jobs/action/optimize-training", "nv-a"],
    [
      "optimize inference on AMD MI300X",
      "/api/jobs/action/optimize-inference",
      "amd-a",
    ],
    ["/optimize-inference", "/api/jobs/action/optimize-inference", "nv-a"],
    [
      "run batching agent on trainer",
      "/api/jobs/action/optimize-inference",
      "nv-a",
    ],
    [
      "optimize batching on nv-b",
      "/api/jobs/action/optimize-inference",
      "nv-b",
    ],
  ])("%s", (command, path, podId) => {
    expect(planGpuWorkflow(command, context())).toMatchObject({
      kind: "action",
      path,
      body: { pod_id: podId },
      requiresApproval: true,
    });
  });
  test("data generation needs no pod and discloses service costs", () => {
    expect(
      planGpuWorkflow(
        "generate training data",
        context({ pods: [], jobs: [], experiment: null }),
      ),
    ).toMatchObject({
      kind: "action",
      path: "/api/jobs/data",
      body: { total: 2000, heldout: 200, workers: 8 },
      requiresApproval: true,
      warnings: [expect.stringContaining("cost is not capped")],
    });
  });
  test.each([
    [
      "migrate from NVIDIA RTX 4090 to AMD MI300X",
      "migrate-nvidia-amd",
      {
        source_pod_id: "nv-a",
        target_pod_id: "amd-a",
        total_steps: 8,
        stop_after: 4,
        eval_n: 50,
      },
    ],
    [
      "/migrate amd-a -> nv-a",
      "migrate-amd-nvidia",
      { source_pod_id: "amd-a", target_pod_id: "nv-a" },
    ],
    [
      "/migrate 4090 → L40S",
      "migrate-nextgen",
      { source_pod_id: "nv-a", target_pod_id: "nv-b" },
    ],
  ])("routes %s by actual vendors", (command, route, body) => {
    const plan = planGpuWorkflow(command, context());
    expect(plan).toMatchObject({
      kind: "action",
      path: `/api/jobs/action/${route}`,
      body,
      requiresApproval: true,
    });
    if (plan.kind === "action")
      expect(plan.warnings?.join(" ")).toContain("not a production deployment");
  });
  test("explicit model and GPU select a serving plan", () => {
    expect(
      planGpuWorkflow("load base on candidate", context({ jobs: [] })),
    ).toMatchObject({
      kind: "action",
      path: "/api/serve",
      body: { pod_id: "amd-a", model_id: "base", dtype: "bf16" },
      requiresApproval: true,
    });
    expect(planGpuWorkflow("serve on MI300X", context())).toMatchObject({
      kind: "action",
      body: { model_id: "ft", pod_id: "amd-a" },
    });
  });
  test("unload requires explicit approval and keeps pod billing visible", () => {
    expect(
      planGpuWorkflow(
        "/unload",
        context({ serving: { running: true, pod_id: "nv-a", model_id: "ft" } }),
      ),
    ).toMatchObject({
      kind: "action",
      path: "/api/serve/stop",
      body: {},
      requiresApproval: true,
      warnings: [expect.stringContaining("does not stop pod billing")],
    });
  });
});

describe("no request silently weakens constraints or chooses an ambiguous target", () => {
  test.each([
    "train baseline under $20",
    "optimize inference by 6 PM",
    "generate data within an hour",
    "train baseline by tomorrow",
    "optimize training with a budget of twenty dollars",
  ])("blocks budget/deadline: %s", (command) => {
    expect(planGpuWorkflow(command, context())).toMatchObject({
      kind: "blocked",
      text: expect.stringContaining("cannot enforce"),
    });
  });
  test.each([
    "train baseline and deploy it",
    "optimize inference with QLoRA",
    "rent the cheapest GPU",
    "optimize inference with vLLM",
    "automatically migrate 4090 to MI300X",
  ])("blocks unsupported promise: %s", (command) => {
    expect(planGpuWorkflow(command, context()).kind).toBe("blocked");
  });
  test.each([
    "train baseline for 100 steps",
    "generate data with 500 examples",
    "optimize training and optimize inference",
    "migrate to MI300X",
    "stop all pods",
    "serve nonexistent on 4090",
    "A contractor emailed an API key.",
  ])("does not partially execute %s", (command) => {
    expect(planGpuWorkflow(command, context()).kind).toBe("blocked");
  });
  test("duplicate chip names require a specific pod even when one is selected", () => {
    const duplicate = { ...nvidia, id: "nv-second", name: "second-host" };
    expect(
      planGpuWorkflow(
        "train baseline on 4090",
        context({ pods: [nvidia, duplicate] }),
      ),
    ).toMatchObject({
      kind: "blocked",
      text: expect.stringContaining("More than one pod"),
    });
    expect(
      planGpuWorkflow(
        "train baseline on nv-second",
        context({ pods: [nvidia, duplicate] }),
      ),
    ).toMatchObject({ kind: "action", body: { pod_id: "nv-second" } });
  });
  test("a broad vendor and missing selection do not choose the first GPU", () => {
    expect(planGpuWorkflow("migrate NVIDIA to AMD", context())).toMatchObject({
      kind: "blocked",
      text: expect.stringContaining("More than one pod"),
    });
    expect(
      planGpuWorkflow("train baseline", context({ selectedPodId: undefined }))
        .kind,
    ).toBe("blocked");
  });
  test("stopped or missing selected pods cannot fall back to another running GPU", () => {
    expect(
      planGpuWorkflow(
        "train baseline",
        context({ pods: [{ ...nvidia, status: "stopped" }, amd] }),
      ),
    ).toMatchObject({
      kind: "blocked",
      text: expect.stringContaining("not running"),
    });
    expect(
      planGpuWorkflow("train baseline", context({ selectedPodId: "missing" }))
        .kind,
    ).toBe("blocked");
  });
  test("rejects identical and unsupported migration pairs", () => {
    expect(planGpuWorkflow("migrate 4090 to 4090", context()).kind).toBe(
      "blocked",
    );
    expect(
      planGpuWorkflow(
        "migrate amd-a to amd-b",
        context({ pods: [amd, { ...amd, id: "amd-b", name: "second-amd" }] }),
      ).kind,
    ).toBe("blocked");
  });
});

describe("state and quality prerequisites", () => {
  test.each([
    "/train-baseline",
    "/optimize-training",
    "/optimize-inference",
    "/generate-data",
    "/migrate 4090 -> MI300X",
    "/serve",
    "/unload",
  ])("recorded mode blocks %s", (command) => {
    expect(planGpuWorkflow(command, context({ readOnly: true }))).toMatchObject(
      {
        kind: "blocked",
        text: expect.stringContaining("recorded-evidence mode"),
      },
    );
  });
  test.each([
    { jobs: [] },
    { jobs: [{ ...baseline, status: "running" }] },
    { jobs: [{ ...baseline, result: { validation: { status: "ok" } } }] },
    {
      jobs: [{ ...baseline, result: { validation: { status: "regressed" } } }],
    },
  ])(
    "requires a current accepted baseline, not experiment metrics: %j",
    ({ jobs }) => {
      expect(
        planGpuWorkflow("optimize inference", context({ jobs })).kind,
      ).toBe("blocked");
      expect(
        planGpuWorkflow("migrate 4090 to MI300X", context({ jobs })).kind,
      ).toBe("blocked");
    },
  );
  test("saved evidence alone never approves a trained model", () => {
    expect(planGpuWorkflow("serve ft", context({ jobs: [] })).kind).toBe(
      "blocked",
    );
  });
  test("baseline needs training and evaluation data", () => {
    expect(
      planGpuWorkflow(
        "train baseline",
        context({ experiment: { data: { train: 1800, heldout: 0 } } }),
      ),
    ).toMatchObject({
      kind: "blocked",
      text: expect.stringContaining("held-out data"),
    });
  });
  test("active jobs block a second mutation", () => {
    expect(
      planGpuWorkflow(
        "generate data",
        context({
          jobs: [{ id: "active", kind: "serve-model", status: "queued" }],
        }),
      ),
    ).toMatchObject({
      kind: "blocked",
      text: expect.stringContaining("still active"),
    });
  });
  test("serving cannot be replaced or guessed when status is missing", () => {
    expect(
      planGpuWorkflow("serve base", context({ serving: { running: true } })),
    ).toMatchObject({
      kind: "blocked",
      text: expect.stringContaining("will not silently replace"),
    });
    expect(
      planGpuWorkflow("serve base", context({ serving: undefined })).kind,
    ).toBe("blocked");
    expect(
      planGpuWorkflow("unload", context({ serving: undefined })).kind,
    ).toBe("blocked");
    expect(planGpuWorkflow("unload", context())).toMatchObject({
      kind: "blocked",
      text: expect.stringContaining("nothing to unload"),
    });
  });
  test("missing model selection does not choose the first model", () => {
    expect(
      planGpuWorkflow("serve", context({ selectedModelId: undefined })).kind,
    ).toBe("blocked");
  });
});

describe("saved evidence, status, and help stay available without live GPUs", () => {
  test("keeps same-GPU and migration evidence distinct", () => {
    const readonly = context({ readOnly: true, pods: [], jobs: [] });
    expect(
      planGpuWorkflow("compare saved MI300X own baseline", readonly),
    ).toEqual({
      kind: "evidence",
      gpuKey: "mi300x",
      comparison: "optimization",
      recheck: false,
    });
    expect(
      planGpuWorkflow("/recheck saved MI300X from RTX 4090", readonly),
    ).toEqual({
      kind: "evidence",
      gpuKey: "mi300x",
      comparison: "migration_from_4090",
      recheck: true,
    });
    expect(
      planGpuWorkflow("compare saved GPU evidence", readonly),
    ).toMatchObject({ kind: "evidence" });
  });
  test("unmeasured chips can be inspected but cannot be rechecked", () => {
    expect(planGpuWorkflow("compare 5090", context())).toMatchObject({
      kind: "evidence",
      gpuKey: "5090",
    });
    expect(planGpuWorkflow("recheck 5090", context())).toMatchObject({
      kind: "blocked",
      text: expect.stringContaining("no completed measurement"),
    });
  });
  test("unknown, ambiguous, and unavailable saved evidence explains the gap", () => {
    expect(planGpuWorkflow("compare 1234", context()).kind).toBe("blocked");
    expect(planGpuWorkflow("compare NVIDIA", context()).kind).toBe("blocked");
    expect(
      planGpuWorkflow("compare", context({ evidence: null })),
    ).toMatchObject({
      kind: "blocked",
      text: expect.stringContaining("unavailable"),
    });
  });
  test("help is honest about deterministic planning and status has no side effects", () => {
    expect(planGpuWorkflow("/status", context({ readOnly: true }))).toEqual({
      kind: "status",
    });
    expect(planGpuWorkflow("help", context())).toMatchObject({
      kind: "help",
      text: expect.stringContaining("do not call an AI planner"),
    });
    const initial = context();
    const before = JSON.stringify(initial);
    expect(planGpuWorkflow("optimize inference", initial)).toEqual(
      planGpuWorkflow("optimize inference", initial),
    );
    expect(JSON.stringify(initial)).toBe(before);
  });
});
