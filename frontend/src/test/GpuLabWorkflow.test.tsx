import { act, render, screen, waitFor, within } from "@testing-library/react";
import userEvent from "@testing-library/user-event";
import { afterEach, expect, test, vi } from "vitest";
import type { LabJob, RecordedEvidence } from "../api/gpulab";
import { GpuLab } from "../components/GpuLab";

afterEach(() => vi.unstubAllGlobals());

const verdict = {
  status: "rejected" as const,
  cases: 300,
  changed_cases: 13,
  reference_correct: 270,
  candidate_correct: 270,
  examples: [],
};
const evidence: RecordedEvidence = {
  kind: "recorded_hardware_evidence",
  recorded_at: "2026-09-19",
  checkpoint: "experimental v3b",
  model_sha256: "frozen-model",
  dataset_sha256: "frozen-data",
  model_status: "Experimental",
  timing_scope: "Resident inference",
  sampling_note: "One measured host",
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
  ],
};
const training: LabJob = {
  id: "training-1",
  kind: "train-and-evaluate",
  status: "running",
  stage: "Training baseline",
  progress: 32,
  logs: ["Step 160 of 500"],
};

function setup({
  readOnly = false,
  serving = false,
  deferTrain = false,
  recheckMismatch = false,
} = {}) {
  let resolveTrain!: (response: Response) => void;
  const routes: Record<string, unknown> = {
    "/api/state": {
      demo_read_only: readOnly,
      experiment: { data: { train: 1800, heldout: 200 } },
    },
    "/api/jobs": { jobs: [] },
    "/api/models": {
      models: [{ id: "ft", label: "Fine-tuned Qwen", kind: "finetuned" }],
      serving: {
        running: serving,
        model_id: serving ? "ft" : undefined,
        pod_id: serving ? "pod-1" : undefined,
        dtype: "bf16",
      },
    },
    "/api/pods": {
      pods: [
        {
          id: "pod-1",
          name: "trainer",
          gpu: "NVIDIA RTX 4090",
          vendor: "nvidia",
          status: "running",
          cost_per_hour: 0.74,
        },
      ],
    },
    "/api/evidence": evidence,
    "/api/jobs/train": training,
    "/api/jobs/training-1": training,
    "/api/evidence/mi300x/recheck": {
      gpu_key: "mi300x",
      comparison: "migration_from_4090",
      source: "saved_outputs",
      model_sha256: recheckMismatch ? "different-model" : evidence.model_sha256,
      dataset_sha256: evidence.dataset_sha256,
      checked_at: "2026-09-19",
      verdict: evidence.rows[0].migration_from_4090,
      status: "rejected",
      recorded_speed_gate_passed: true,
      detail: "Recomputed saved outputs.",
    },
  };
  const fetcher = vi.fn(
    async (input: RequestInfo | URL, init?: RequestInit) => {
      const path = new URL(String(input), "http://localhost").pathname;
      if (path === "/api/jobs/train" && init?.method === "POST" && deferTrain)
        return new Promise<Response>((resolve) => {
          resolveTrain = resolve;
        });
      if (path === "/api/generate/stream")
        return new Response(
          'data: {"token":"A model reply about training."}\n\ndata: {"done":true,"ttft_s":0.25,"latency_s":0.8,"prompt_tokens":8,"cache":"off"}\n\n',
        );
      if (!(path in routes))
        return new Response(JSON.stringify({ detail: "Unknown test route" }), {
          status: 404,
        });
      return new Response(JSON.stringify(routes[path]));
    },
  );
  vi.stubGlobal("fetch", fetcher);
  render(<GpuLab active />);
  return {
    fetcher,
    posts: () =>
      fetcher.mock.calls.filter(([, init]) => init?.method === "POST"),
    finishTrainingRequest: () =>
      resolveTrain(new Response(JSON.stringify(training))),
  };
}

async function workflow(user: ReturnType<typeof userEvent.setup>) {
  await screen.findByRole("option", { name: "Fine-tuned Qwen" });
  await user.click(screen.getByRole("button", { name: "Workflow" }));
  return screen.getByLabelText("Message GPU workflow");
}

test("a training command prepares a plan and runs only after explicit approval", async () => {
  const user = userEvent.setup();
  const api = setup();
  const input = await workflow(user);
  await user.type(input, "/train-baseline{Enter}");
  expect(api.posts()).toHaveLength(0);
  expect(screen.getByRole("button", { name: "Approve and run" })).toBeEnabled();
  const conversation = within(
    screen.getByRole("log", { name: "GPU Lab conversation" }),
  );
  expect(
    conversation.getByText(/Run the configured Qwen baseline/),
  ).toBeInTheDocument();

  await user.click(screen.getByRole("button", { name: "Approve and run" }));
  expect(api.posts()).toHaveLength(1);
  expect(api.fetcher).toHaveBeenCalledWith(
    "/api/jobs/train",
    expect.objectContaining({
      method: "POST",
      body: JSON.stringify({
        pod_id: "pod-1",
        steps: 500,
        dtype: "bf16",
        attention: "sdpa",
        micro_batch: 16,
        grad_accum: 1,
      }),
    }),
  );
  expect(
    await conversation.findByRole("progressbar", { name: "Job progress" }),
  ).toHaveAttribute("value", "32");
  expect(conversation.getByText("32% · Training baseline")).toBeInTheDocument();
  expect(screen.getByLabelText("Message GPU workflow")).toBeEnabled();
});

test("dismissed plans never issue a job request", async () => {
  const user = userEvent.setup();
  const api = setup();
  await user.type(await workflow(user), "/train-baseline{Enter}");
  await user.click(screen.getByRole("button", { name: "Dismiss" }));
  expect(screen.getByText("Dismissed. Nothing was run.")).toBeInTheDocument();
  expect(
    screen.queryByRole("button", { name: "Approve and run" }),
  ).not.toBeInTheDocument();
  expect(api.posts()).toHaveLength(0);
});

test("read-only workflow commands explain blocked mutations without posting", async () => {
  const user = userEvent.setup();
  const api = setup({ readOnly: true });
  await user.type(
    await screen.findByLabelText("Message GPU workflow"),
    "/train-baseline{Enter}",
  );
  expect(
    screen.getByText(/This server is in recorded-evidence mode/),
  ).toBeInTheDocument();
  expect(
    screen.queryByRole("button", { name: "Approve and run" }),
  ).not.toBeInTheDocument();
  expect(api.posts()).toHaveLength(0);
});

test.each([
  ["compare saved MI300X own baseline", "Recorded result.", "optimization"],
  [
    "recheck saved MI300X from RTX 4090",
    "Saved evidence rechecked.",
    "migration_from_4090",
  ],
])(
  "%s responds in chat, selects sidebar evidence, and performs only GETs",
  async (command, reply, comparison) => {
    const user = userEvent.setup();
    const api = setup({ readOnly: true });
    await user.type(
      await screen.findByLabelText("Message GPU workflow"),
      `${command}{Enter}`,
    );
    const conversation = within(
      screen.getByRole("log", { name: "GPU Lab conversation" }),
    );
    expect(
      await conversation.findByText((content) => content.startsWith(reply)),
    ).toBeInTheDocument();
    expect(
      screen.getByRole("complementary", { name: "Experiment" }),
    ).toBeInTheDocument();
    expect(screen.getByLabelText("GPU comparison")).toHaveValue("mi300x");
    expect(
      screen.getByRole("button", {
        name:
          comparison === "optimization" ? "Optimize this GPU" : "From RTX 4090",
      }),
    ).toHaveAttribute("aria-pressed", "true");
    expect(screen.getByLabelText("Message GPU workflow")).toBeEnabled();
    expect(api.posts()).toHaveLength(0);
    expect(
      api.fetcher.mock.calls.every(([, init]) => init?.method === "GET"),
    ).toBe(true);
    if (comparison === "migration_from_4090") {
      expect(api.fetcher).toHaveBeenCalledWith(
        "/api/evidence/mi300x/recheck?comparison=migration_from_4090",
        expect.objectContaining({ method: "GET" }),
      );
      expect(
        conversation.getByText(/No new GPU measurement was made/),
      ).toBeInTheDocument();
    }
  },
);

test("a saved recheck with different provenance is not accepted", async () => {
  const user = userEvent.setup();
  const api = setup({ readOnly: true, recheckMismatch: true });
  await user.type(
    await screen.findByLabelText("Message GPU workflow"),
    "recheck MI300X from RTX 4090{Enter}",
  );
  expect(
    await screen.findByText(/Recheck returned a different experiment/),
  ).toBeInTheDocument();
  expect(
    screen.queryByText(/Saved evidence rechecked\./),
  ).not.toBeInTheDocument();
  expect(api.posts()).toHaveLength(0);
});

test("literal model prompts mentioning train are streamed without workflow actions", async () => {
  const user = userEvent.setup();
  const api = setup({ serving: true });
  const input = screen.getByLabelText("Message the model");
  await waitFor(() => expect(input).toBeEnabled());
  await user.type(input, "Train baseline is a phrase; explain it.{Enter}");
  expect(
    await screen.findByText("A model reply about training."),
  ).toBeInTheDocument();
  expect(api.posts().map(([path]) => path)).toEqual(["/api/generate/stream"]);
  expect(
    screen.queryByRole("button", { name: "Approve and run" }),
  ).not.toBeInTheDocument();
  expect(screen.getByRole("button", { name: "Model replies" })).toHaveAttribute(
    "aria-pressed",
    "true",
  );
});

test("a second pending action cannot run while the first submission or job is active", async () => {
  const user = userEvent.setup();
  const api = setup({ deferTrain: true });
  const input = await workflow(user);
  await user.type(input, "/train-baseline{Enter}");
  await user.type(input, "/generate-data{Enter}");
  expect(api.posts()).toHaveLength(0);
  expect(
    screen.getAllByRole("button", { name: "Approve and run" }),
  ).toHaveLength(2);
  await user.click(
    screen.getAllByRole("button", { name: "Approve and run" })[0],
  );
  expect(
    screen.getByRole("button", { name: "Approve and run" }),
  ).toBeDisabled();
  await user.click(screen.getByRole("button", { name: "Approve and run" }));
  expect(api.posts()).toHaveLength(1);
  await act(async () => api.finishTrainingRequest());
  expect(
    await screen.findByRole("progressbar", { name: "Job progress" }),
  ).toBeInTheDocument();
  expect(
    screen.getByRole("button", { name: "Approve and run" }),
  ).toBeDisabled();
  await user.type(input, "/train-baseline{Enter}");
  expect(screen.getByText(/A job is still active/)).toBeInTheDocument();
  expect(api.posts()).toHaveLength(1);
});

test("unsupported spending requirements stay visible and create no executable plan", async () => {
  const user = userEvent.setup();
  const api = setup();
  await user.type(await workflow(user), "train baseline under $20{Enter}");
  expect(
    screen.getByText(/cannot enforce a spending cap or completion deadline/),
  ).toBeInTheDocument();
  expect(
    screen.queryByRole("button", { name: "Approve and run" }),
  ).not.toBeInTheDocument();
  expect(api.posts()).toHaveLength(0);
});
