import { render, screen, waitFor } from "@testing-library/react";
import userEvent from "@testing-library/user-event";
import { afterEach, expect, test, vi } from "vitest";
import { GpuLab } from "../components/GpuLab";
import type { LabJob } from "../api/gpulab";

afterEach(() => vi.unstubAllGlobals());

const pod = {
  id: "pod-1",
  name: "trainer",
  gpu: "NVIDIA RTX 4090",
  vendor: "nvidia",
  status: "running",
  cost_per_hour: 0.69,
};

function respond(routes: Record<string, unknown>) {
  return vi.fn(async (input: RequestInfo | URL) => {
    const path = new URL(String(input), "http://localhost").pathname;
    if (path === "/api/generate/stream") {
      const frames =
        'data: {"token":"RISK: "}\n\ndata: {"token":"high"}\n\n' +
        'data: {"done":true,"ttft_s":0.41,"latency_s":1.2,"prompt_tokens":52,"cache":"off"}\n\n';
      return new Response(frames, { status: 200 });
    }
    if (!(path in routes)) return new Response("{}", { status: 404 });
    return new Response(JSON.stringify(routes[path]), { status: 200 });
  });
}

const routes = (running: boolean) => ({
  "/api/state": { experiment: { data: { train: 1800, heldout: 200 } } },
  "/api/jobs": { jobs: [] },
  "/api/pods": { pods: [pod] },
  "/api/models": {
    models: [
      { id: "ft", label: "Qwen2.5-0.5B · fine-tuned", kind: "finetuned" },
    ],
    serving: running
      ? { running: true, model_id: "ft", pod_id: "pod-1", dtype: "bf16" }
      : { running: false },
  },
});

test("keeps the composer locked until a model is loaded", async () => {
  vi.stubGlobal("fetch", respond(routes(false)));
  render(<GpuLab active />);
  expect(await screen.findByText("RTX 4090")).toBeInTheDocument();
  expect(screen.getByLabelText("Message the model")).toBeDisabled();
  expect(screen.getByText("No model loaded")).toBeInTheDocument();
  expect(
    screen.getByRole("button", { name: "Load on this GPU" }),
  ).toBeEnabled();
});

test("streams a reply and records the run with its timings", async () => {
  vi.stubGlobal("fetch", respond(routes(true)));
  render(<GpuLab active />);
  const input = await screen.findByLabelText("Message the model");
  await waitFor(() => expect(input).toBeEnabled());
  await userEvent.type(input, "A key was emailed out.{Enter}");
  expect(await screen.findByText("RISK: high")).toBeInTheDocument();
  expect(screen.getByText("0.41s")).toBeInTheDocument();
  expect(screen.getByText("1.20s")).toBeInTheDocument();
  await userEvent.click(screen.getByRole("button", { name: /Evidence/ }));
  expect(screen.getByText("No policy")).toBeInTheDocument();
  expect(screen.getByRole("button", { name: "Send again" })).toBeEnabled();
});

test("explains how to recover when the experiment server is down", async () => {
  vi.stubGlobal(
    "fetch",
    vi.fn(async () => {
      throw new TypeError("Failed to fetch");
    }),
  );
  render(<GpuLab active />);
  expect(
    await screen.findByText(/Can't reach the experiment server/),
  ).toBeInTheDocument();
});

test.each([
  ["ok", "Passed"],
  ["regressed", "Rejected"],
  ["not_validated", "Not validated"],
  [undefined, "Not validated"],
])(
  "renders a completed job with validation %s as %s",
  async (status, label) => {
    const job: LabJob = {
      id: "result-1",
      kind: "optimize-inference-speed",
      status: "complete",
      result: {
        validation: status
          ? {
              status,
              detail: "Recorded evaluation detail.",
              policy: "strict_output_preservation",
              tolerance: 0,
              evaluation: { dataset_sha256: "frozen-suite-hash" },
              output_preservation_verified: status === "ok",
            }
          : undefined,
      },
    };
    vi.stubGlobal(
      "fetch",
      respond({ ...routes(false), "/api/jobs": { jobs: [job] } }),
    );
    render(<GpuLab active evidenceOpen />);
    expect(
      await screen.findByText(new RegExp(`^${label} ·`)),
    ).toBeInTheDocument();
    expect(
      screen.queryByText(/verified automatically/),
    ).not.toBeInTheDocument();
  },
);

test("historical two-percent tolerance does not display a passing decision", async () => {
  vi.stubGlobal(
    "fetch",
    respond({
      ...routes(false),
      "/api/jobs": {
        jobs: [
          {
            id: "legacy",
            kind: "optimize-inference-speed",
            status: "complete",
            result: {
              validation: {
                status: "ok",
                tolerance: 0.02,
                detail: "model quality preserved",
              },
            },
          },
        ],
      },
    }),
  );
  render(<GpuLab active evidenceOpen />);
  expect(await screen.findByText(/^Not validated ·/)).toBeInTheDocument();
  expect(
    screen.getByText(/Historical or incomplete evaluation/),
  ).toBeInTheDocument();
  expect(screen.queryByText(/^Passed ·/)).not.toBeInTheDocument();
  expect(screen.queryByText("model quality preserved")).not.toBeInTheDocument();
});

test("shows failed jobs as failed instead of completed", async () => {
  vi.stubGlobal(
    "fetch",
    respond({
      ...routes(false),
      "/api/jobs": {
        jobs: [
          {
            id: "failed",
            kind: "serve-model",
            status: "failed",
            error: "Candidate never became healthy.",
          },
        ],
      },
    }),
  );
  render(<GpuLab active evidenceOpen />);
  expect(await screen.findByText(/^Failed ·/)).toBeInTheDocument();
  expect(
    screen.getByText("Candidate never became healthy."),
  ).toBeInTheDocument();
});

test("routes NVIDIA to AMD to the bounded training resume experiment", async () => {
  const mock = respond({
    ...routes(false),
    "/api/pods": {
      pods: [pod, { ...pod, id: "amd-1", gpu: "AMD MI300X", vendor: "amd" }],
    },
    "/api/jobs/action/migrate-nvidia-amd": {
      id: "resume",
      kind: "migrate-nvidia-amd",
      status: "queued",
    },
  });
  vi.stubGlobal("fetch", mock);
  render(<GpuLab active evidenceOpen />);
  await userEvent.click(
    await screen.findByText("Data generation and migration"),
  );
  await userEvent.click(
    await screen.findByRole("button", {
      name: "Test NVIDIA → AMD training resume",
    }),
  );
  expect(mock.mock.calls.some(([, init]) => init?.method === "POST")).toBe(
    false,
  );
  await userEvent.click(
    screen.getByRole("button", { name: "Approve and run" }),
  );
  expect(mock).toHaveBeenCalledWith(
    "/api/jobs/action/migrate-nvidia-amd",
    expect.objectContaining({
      method: "POST",
      body: JSON.stringify({
        source_pod_id: "pod-1",
        target_pod_id: "amd-1",
        total_steps: 8,
        stop_after: 4,
        eval_n: 50,
      }),
    }),
  );
});

test("offers the long-context model and labels its separate purpose", async () => {
  vi.stubGlobal(
    "fetch",
    respond({
      ...routes(false),
      "/api/models": {
        models: [
          {
            id: "longctx",
            label: "Qwen3-4B Instruct",
            kind: "longctx",
            detail:
              "For prefix-cache latency; separate from JSON quality evaluation.",
          },
        ],
        serving: { running: false },
      },
    }),
  );
  render(<GpuLab active />);
  expect(
    await screen.findByRole("option", { name: "Qwen3-4B Instruct" }),
  ).toBeInTheDocument();
  expect(
    screen.getByText(
      "For prefix-cache latency; separate from JSON quality evaluation.",
    ),
  ).toBeInTheDocument();
});

test("separates recorded optimization passes from rejected migrations", async () => {
  const verdict = {
    status: "passed",
    cases: 300,
    changed_cases: 0,
    reference_correct: 270,
    candidate_correct: 270,
    examples: [],
  };
  vi.stubGlobal(
    "fetch",
    respond({
      ...routes(false),
      "/api/evidence": {
        kind: "recorded_hardware_evidence",
        recorded_at: "2026-09-19",
        checkpoint: "Qwen + experimental v3b",
        model_sha256: "model-hash",
        dataset_sha256: "dataset-hash",
        model_status: "experimental",
        timing_scope: "Resident time excludes setup.",
        sampling_note: "Single measured run.",
        quality_note: "Output preservation does not imply perfect accuracy.",
        rows: [
          {
            key: "a5000",
            gpu: "NVIDIA A5000",
            vendor: "nvidia",
            measured: true,
            baseline_s: 0.6,
            candidate_s: 0.1,
            latency_ratio: 6,
            optimization: verdict,
            migration_from_4090: {
              ...verdict,
              status: "rejected",
              changed_cases: 2,
            },
          },
        ],
      },
    }),
  );
  render(<GpuLab active />);
  await userEvent.click(screen.getByRole("button", { name: "Evidence" }));
  expect(await screen.findByLabelText("GPU comparison")).toHaveValue("a5000");
  expect(screen.getByLabelText("Message the model")).toBeInTheDocument();
  expect(
    screen.getByRole("button", { name: "Optimize this GPU" }),
  ).toHaveAttribute("aria-pressed", "true");
  expect(
    screen.getByText("All 300 JSON outputs preserved their reference values."),
  ).toBeInTheDocument();

  await userEvent.click(screen.getByRole("button", { name: "From RTX 4090" }));
  expect(screen.getByRole("button", { name: "From RTX 4090" })).toHaveAttribute(
    "aria-pressed",
    "true",
  );
  expect(
    screen.getByText(
      "2 of 300 outputs changed or became invalid. The gate allows zero changes.",
    ),
  ).toBeInTheDocument();
  expect(screen.getByText("Rejected")).toBeInTheDocument();
});

test("read-only backend hides mutation controls and explains missing evidence", async () => {
  vi.stubGlobal(
    "fetch",
    respond({
      ...routes(false),
      "/api/state": {
        demo_read_only: true,
        experiment: { data: { train: 0, heldout: 0 } },
      },
    }),
  );
  render(<GpuLab active />);
  expect(await screen.findByLabelText("Message GPU workflow")).toBeEnabled();
  await userEvent.click(screen.getByRole("button", { name: "Evidence" }));
  expect(
    screen.getByRole("complementary", { name: "Experiment" }),
  ).toBeInTheDocument();
  expect(screen.getByText("Request failed (404)")).toBeInTheDocument();
  expect(screen.getByRole("button", { name: "Refresh" })).toBeEnabled();
  expect(
    screen.queryByRole("button", { name: "Generate data" }),
  ).not.toBeInTheDocument();
  expect(
    screen.queryByRole("button", { name: "Start baseline training" }),
  ).not.toBeInTheDocument();
});
