import { act, render, screen, waitFor } from "@testing-library/react";
import userEvent from "@testing-library/user-event";
import { beforeEach, expect, test, vi } from "vitest";
import { GpuLab } from "../components/GpuLab";
import { useGpuLab } from "../hooks/useGpuLab";

vi.mock("../hooks/useGpuLab", () => ({ useGpuLab: vi.fn() }));
vi.mock("../components/GpuLabExperiment", () => ({
  GpuLabExperiment: () => (
    <aside aria-label="Experiment">Supporting evidence</aside>
  ),
}));

let state: ReturnType<typeof useGpuLab>;
beforeEach(() => {
  state = {
    status: "live",
    error: "",
    podError: "",
    experiment: null,
    jobs: [],
    models: [{ id: "ft", kind: "finetuned", label: "Qwen fine-tuned" }],
    serving: { running: true, model_id: "ft", pod_id: "pod-1", dtype: "bf16" },
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
    evidence: null,
    evidenceError: "",
    readOnly: false,
    refresh: vi.fn(async () => {}),
    start: vi.fn(async () => ({
      id: "unused-job",
      kind: "serve-model",
      status: "queued",
    })),
  };
  vi.mocked(useGpuLab).mockImplementation(() => state);
});

test("live mode opens the original model conversation with evidence closed", () => {
  render(<GpuLab active />);
  expect(
    screen.getByRole("log", { name: "GPU Lab conversation" }),
  ).toBeInTheDocument();
  expect(screen.getByRole("button", { name: "Model replies" })).toHaveAttribute(
    "aria-pressed",
    "true",
  );
  expect(screen.getByLabelText("Message the model")).toBeEnabled();
  expect(screen.getByLabelText("Model to run")).toHaveValue("ft");
  expect(
    screen.queryByRole("complementary", { name: "Experiment" }),
  ).not.toBeInTheDocument();
  expect(state.start).not.toHaveBeenCalled();
});

test.each([{ evidenceOpen: true }, { initialView: "evidence" as const }])(
  "opening evidence preserves the original chat: %j",
  (props) => {
    render(<GpuLab active {...props} />);
    expect(
      screen.getByRole("complementary", { name: "Experiment" }),
    ).toBeInTheDocument();
    expect(screen.getByLabelText("Message the model")).toBeEnabled();
    expect(
      screen.getByRole("log", { name: "GPU Lab conversation" }),
    ).toBeInTheDocument();
    expect(
      screen.getByRole("button", { name: "Setup & results" }),
    ).toHaveAttribute("aria-pressed", "true");
    expect(state.start).not.toHaveBeenCalled();
  },
);

test("read-only mode keeps a usable workflow composer and disables model controls", () => {
  state.readOnly = true;
  state.serving = { running: false };
  render(<GpuLab active />);
  expect(screen.getByRole("button", { name: "Workflow" })).toHaveAttribute(
    "aria-pressed",
    "true",
  );
  expect(screen.getByLabelText("Message GPU workflow")).toBeEnabled();
  expect(screen.getByRole("button", { name: "Model replies" })).toBeDisabled();
  expect(screen.getByLabelText("Model to run")).toBeDisabled();
  expect(
    screen.queryByRole("button", { name: "Load on this GPU" }),
  ).not.toBeInTheDocument();
  expect(state.start).not.toHaveBeenCalled();
});

test("Evidence opens beside the conversation without losing the draft", async () => {
  const user = userEvent.setup();
  render(<GpuLab active />);
  await user.type(
    screen.getByLabelText("Message the model"),
    "Keep this draft",
  );
  await user.click(screen.getByRole("button", { name: "Setup & results" }));
  expect(
    screen.getByRole("complementary", { name: "Experiment" }),
  ).toBeInTheDocument();
  expect(screen.getByLabelText("Message the model")).toHaveValue(
    "Keep this draft",
  );
  await user.click(screen.getByRole("button", { name: "Setup & results" }));
  expect(
    screen.queryByRole("complementary", { name: "Experiment" }),
  ).not.toBeInTheDocument();
  expect(screen.getByLabelText("Message the model")).toHaveValue(
    "Keep this draft",
  );
  expect(state.start).not.toHaveBeenCalled();
});

test("offline recovery refreshes the service while retaining the conversation", async () => {
  let finishRefresh!: () => void;
  state.status = "offline";
  state.error = "Connection refused";
  state.refresh = vi.fn(
    () =>
      new Promise<void>((resolve) => {
        finishRefresh = resolve;
      }),
  );
  const user = userEvent.setup();
  render(<GpuLab active />);
  expect(screen.getByRole("alert")).toHaveTextContent(
    "Use Refresh to reconnect",
  );
  await user.click(screen.getByRole("button", { name: "Refresh" }));
  expect(state.refresh).toHaveBeenCalledTimes(1);
  expect(screen.getByRole("button", { name: "Refreshing…" })).toBeDisabled();
  expect(
    screen.getByRole("log", { name: "GPU Lab conversation" }),
  ).toBeInTheDocument();
  state = { ...state, status: "live", error: "" };
  await act(async () => finishRefresh());
  await waitFor(() =>
    expect(screen.getByRole("button", { name: "Refresh" })).toBeEnabled(),
  );
  expect(screen.queryByRole("alert")).not.toBeInTheDocument();
  expect(screen.getByLabelText("Message the model")).toBeEnabled();
});

test("workflow remains available before a live model has loaded", async () => {
  state.serving = { running: false };
  const user = userEvent.setup();
  render(<GpuLab active />);
  expect(screen.getByLabelText("Message the model")).toBeDisabled();
  await user.click(screen.getByRole("button", { name: "Workflow" }));
  expect(screen.getByLabelText("Message GPU workflow")).toBeEnabled();
  expect(
    screen.getByRole("log", { name: "GPU Lab conversation" }),
  ).toBeInTheDocument();
});
