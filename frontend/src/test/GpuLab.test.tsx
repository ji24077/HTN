import { render, screen, waitFor } from "@testing-library/react";
import userEvent from "@testing-library/user-event";
import { afterEach, expect, test, vi } from "vitest";
import { GpuLab } from "../components/GpuLab";

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
    const path = new URL(String(input)).pathname;
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
