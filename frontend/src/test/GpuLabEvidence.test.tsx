import { fireEvent, render, screen } from "@testing-library/react";
import userEvent from "@testing-library/user-event";
import { afterEach, expect, test, vi } from "vitest";
import type { EvidenceVerdict, RecordedEvidence } from "../api/gpulab";
import type { GpuLabState } from "../hooks/useGpuLab";
import { GpuLabEvidence } from "../components/GpuLabEvidence";
import { GpuLabExperiment } from "../components/GpuLabExperiment";

afterEach(() => vi.unstubAllGlobals());
const rejected: EvidenceVerdict = {
  status: "rejected",
  cases: 300,
  changed_cases: 13,
  reference_correct: 270,
  candidate_correct: 270,
  examples: [
    {
      id: "one",
      sentence: "Ari is a zoo veterinarian.",
      fields: ["role"],
      before: { role: "zoo veterinarian" },
      after: { role: "zoovet veterinarian" },
      expected: { role: "zoo veterinarian" },
    },
    {
      id: "two",
      sentence: "Sol joined in 2013.",
      fields: ["year"],
      before: { year: 2013 },
      after: { year: 1973 },
      expected: { year: 2013 },
    },
  ],
};
const evidence: RecordedEvidence = {
  kind: "recorded_hardware_evidence",
  recorded_at: "2026-09-19",
  checkpoint: "experimental v3b",
  model_sha256: "model",
  dataset_sha256: "dataset",
  model_status: "Experimental",
  quality_note: "Not an approved release.",
  timing_scope: "Resident execution",
  sampling_note: "Single host",
  rows: [
    {
      key: "4090",
      gpu: "NVIDIA RTX 4090",
      vendor: "nvidia",
      measured: true,
      baseline_s: 0.8,
      candidate_s: 0.2,
      latency_ratio: 4,
      optimization: rejected,
      migration_from_4090: rejected,
    },
    {
      key: "mi300x",
      gpu: "AMD MI300X",
      vendor: "amd",
      measured: true,
      baseline_s: 1,
      candidate_s: 0.2,
      latency_ratio: 5,
      optimization: rejected,
      migration_from_4090: { ...rejected, changed_cases: 15 },
    },
  ],
};
const lab: GpuLabState = {
  status: "live",
  error: "",
  podError: "",
  experiment: { data: { train: 1800, heldout: 200 } },
  jobs: [],
  models: [],
  serving: { running: false },
  pods: [
    {
      id: "nvidia",
      name: "source",
      gpu: "NVIDIA RTX 4090",
      vendor: "nvidia",
      status: "running",
      cost_per_hour: 0.5,
    },
    {
      id: "amd",
      name: "target",
      gpu: "AMD MI300X",
      vendor: "amd",
      status: "running",
      cost_per_hour: 2,
    },
  ],
  evidence,
  evidenceError: "",
  readOnly: false,
};
const props = () => ({
  lab,
  liveJob: null,
  runs: [],
  policyNote: "",
  policyBusy: false,
  onStart: vi.fn(),
  onCancel: vi.fn(),
  onPolicy: vi.fn(),
});

test("compact evidence uses the source4090 latency for migration and updates chat selection", async () => {
  const onSelection = vi.fn();
  const onInspect = vi.fn();
  render(
    <GpuLabEvidence
      evidence={evidence}
      onSelection={onSelection}
      onInspect={onInspect}
    />,
  );
  expect(screen.getByText("5.00× faster")).toBeInTheDocument();
  await userEvent.click(screen.getByRole("button", { name: "From RTX 4090" }));
  expect(screen.getByText("4.00× faster")).toBeInTheDocument();
  expect(screen.getByText(/15 of 300 outputs changed/)).toBeInTheDocument();
  expect(onSelection).toHaveBeenLastCalledWith("mi300x", "migration_from_4090");
  await userEvent.click(
    screen.getByRole("button", { name: "Discuss in chat" }),
  );
  expect(onInspect).toHaveBeenCalledWith("mi300x", "migration_from_4090");
});

test("controlled chat selections reset expanded answer navigation safely", async () => {
  const view = render(
    <GpuLabEvidence
      evidence={evidence}
      selection={{ gpuKey: "mi300x", comparison: "optimization" }}
    />,
  );
  await userEvent.click(screen.getByText("Changed answers (13)"));
  await userEvent.click(
    screen.getByRole("button", { name: "Next saved case" }),
  );
  expect(screen.getByText("Sol joined in 2013.")).toBeInTheDocument();
  view.rerender(
    <GpuLabEvidence
      evidence={evidence}
      selection={{ gpuKey: "4090", comparison: "migration_from_4090" }}
    />,
  );
  expect(screen.getByLabelText("GPU comparison")).toHaveValue("4090");
  expect(screen.getByText("Ari is a zoo veterinarian.")).toBeInTheDocument();
  expect(
    screen.getByRole("button", { name: "Previous saved case", hidden: true }),
  ).toBeDisabled();
});

test("saved recheck is real and refuses a different dataset", async () => {
  const fetch = vi.fn(
    async () =>
      new Response(
        JSON.stringify({
          gpu_key: "mi300x",
          comparison: "optimization",
          source: "saved_outputs",
          checked_at: "2026-09-19T23:00:00Z",
          model_sha256: "model",
          dataset_sha256: "wrong-dataset",
          verdict: { ...rejected, status: "passed", changed_cases: 0 },
        }),
        { status: 200 },
      ),
  );
  vi.stubGlobal("fetch", fetch);
  render(<GpuLabEvidence evidence={evidence} />);
  await userEvent.click(
    screen.getByRole("button", { name: "Recheck saved evidence" }),
  );
  expect(fetch).toHaveBeenCalledWith(
    "/api/evidence/mi300x/recheck?comparison=optimization",
    expect.objectContaining({ method: "GET" }),
  );
  expect(await screen.findByRole("alert")).toHaveTextContent(
    "does not match this saved comparison",
  );
  expect(screen.getByText("Rejected")).toBeInTheDocument();
});

test("read-only sidebar keeps saved comparisons and hides GPU mutations", () => {
  render(<GpuLabExperiment {...props()} lab={{ ...lab, readOnly: true }} />);
  expect(screen.getByText("Saved GPU comparisons")).toBeInTheDocument();
  expect(
    screen.getByRole("button", { name: "Recheck saved evidence" }),
  ).toBeEnabled();
  expect(
    screen.queryByRole("button", {
      name: "Start baseline training",
      hidden: true,
    }),
  ).not.toBeInTheDocument();
  expect(
    screen.getByText(/Live training and migration require/),
  ).toBeInTheDocument();
});

test("training settings reach the parent approval callback", async () => {
  const state = props();
  render(<GpuLabExperiment {...state} />);
  await userEvent.click(screen.getByText("Training settings"));
  await userEvent.selectOptions(
    screen.getByLabelText("Training precision"),
    "fp16",
  );
  await userEvent.selectOptions(
    screen.getByLabelText("Attention implementation"),
    "eager",
  );
  fireEvent.change(screen.getByLabelText("Gradient accumulation"), {
    target: { value: "4" },
  });
  await userEvent.click(
    screen.getByRole("button", { name: "Start baseline training" }),
  );
  expect(state.onStart).toHaveBeenCalledWith("/api/jobs/train", {
    pod_id: "nvidia",
    steps: 500,
    dtype: "fp16",
    attention: "eager",
    micro_batch: 16,
    grad_accum: 4,
  });
});

test("data worker and bounded resume settings reach approval without running GPUs", async () => {
  const state = props();
  render(<GpuLabExperiment {...state} />);
  await userEvent.click(screen.getByText("Data generation and migration"));
  fireEvent.change(screen.getByLabelText("Data generation workers"), {
    target: { value: "3" },
  });
  await userEvent.click(screen.getByRole("button", { name: "Generate data" }));
  expect(state.onStart).toHaveBeenCalledWith("/api/jobs/data", {
    total: 2000,
    heldout: 200,
    workers: 3,
  });
  await userEvent.click(screen.getByText("Training resume settings"));
  fireEvent.change(screen.getByLabelText("Total migration steps"), {
    target: { value: "12" },
  });
  fireEvent.change(screen.getByLabelText("Pause source after step"), {
    target: { value: "6" },
  });
  fireEvent.change(screen.getByLabelText("Migration evaluation cases"), {
    target: { value: "75" },
  });
  fireEvent.change(screen.getByLabelText("Initial adapter path (optional)"), {
    target: { value: "adapters/start" },
  });
  await userEvent.click(screen.getByLabelText("Prepare GPU environments"));
  await userEvent.click(
    screen.getByRole("button", { name: "Test NVIDIA → AMD training resume" }),
  );
  expect(state.onStart).toHaveBeenLastCalledWith(
    "/api/jobs/action/migrate-nvidia-amd",
    {
      source_pod_id: "nvidia",
      target_pod_id: "amd",
      total_steps: 12,
      stop_after: 6,
      eval_n: 75,
      initial_adapter: "adapters/start",
      prepare_pods: false,
    },
  );
  fireEvent.change(screen.getByLabelText("Pause source after step"), {
    target: { value: "12" },
  });
  expect(
    screen.getByRole("button", { name: "Test NVIDIA → AMD training resume" }),
  ).toBeDisabled();
});

test("a running job locks live settings while keeping saved comparisons usable", () => {
  render(
    <GpuLabExperiment
      {...props()}
      liveJob={{ id: "running", kind: "train-and-evaluate", status: "running" }}
    />,
  );
  expect(screen.getByLabelText("Training steps")).toBeDisabled();
  expect(screen.getByLabelText("Training precision")).toBeDisabled();
  expect(screen.getByLabelText("Data generation workers")).toBeDisabled();
  expect(screen.getByLabelText("Total migration steps")).toBeDisabled();
  expect(screen.getByLabelText("GPU comparison")).toBeEnabled();
});

test("job history exposes older results and logs without starting a job", async () => {
  const state = props();
  render(
    <GpuLabExperiment
      {...state}
      lab={{
        ...lab,
        jobs: [
          {
            id: "latest",
            kind: "generate-data",
            status: "complete",
            logs: ["latest log"],
          },
          {
            id: "older",
            kind: "serve-model",
            status: "failed",
            error: "Earlier launch failed.",
            logs: ["older log"],
          },
        ],
      }}
    />,
  );
  await userEvent.click(
    screen.getByText("Generating training data · complete"),
  );
  await userEvent.selectOptions(screen.getByLabelText("Job history"), "older");
  expect(screen.getByText("Earlier launch failed.")).toBeInTheDocument();
  expect(screen.getByText("older log")).toBeInTheDocument();
  expect(state.onStart).not.toHaveBeenCalled();
});
