import { render, screen, waitFor } from "@testing-library/react";
import userEvent from "@testing-library/user-event";
import { expect, it, vi } from "vitest";
import { ServicePanel } from "../components/ServicePanel";
import type { SimulationStatus } from "../api/client";
const action = vi.hoisted(() => vi.fn().mockResolvedValue({}));
vi.mock("../api/client", () => ({ serviceAction: action, executionEvents: vi.fn(), answerSimulation: vi.fn() }));
it("shows an endpoint and sends stop instead of downloading an output", async () => {
  const status = { job_id: "svc-1", phase: "ready", message: "Service is ready.", deadline: null,
    workers: ["worker-a"], tasks: [], service: { endpoint: "/serve/svc-1", desired: "running", restarts: 0, ready_at: null, health_at: null, config: {} } } as unknown as SimulationStatus;
  const refresh = vi.fn();
  render(<ServicePanel status={status} refresh={refresh} />);
  expect(screen.getByText(/\/serve\/svc-1\//)).toBeInTheDocument();
  expect(screen.getByText("Runs until stopped")).toBeInTheDocument();
  expect(screen.queryByText(/Download/)).not.toBeInTheDocument();
  await userEvent.click(screen.getByRole("button", { name: "Stop service" }));
  await waitFor(() => expect(action).toHaveBeenCalledWith("svc-1", "stop"));
  expect(refresh).toHaveBeenCalledOnce();
});
