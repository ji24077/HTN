import { render, screen, waitFor } from "@testing-library/react";
import userEvent from "@testing-library/user-event";
import { afterEach, describe, expect, it, vi } from "vitest";
import { DeviceInvite } from "../components/DeviceInvite";
import { TaskComposer } from "../components/TaskComposer";
import { workloadTask } from "../api/client";
import { scrub } from "../telemetry";

afterEach(() => vi.unstubAllGlobals());

describe("device integration", () => {
  it("creates a one-use invite through the authenticated app API", async () => {
    const fetcher = vi
      .fn()
      .mockResolvedValue(
        new Response(
          JSON.stringify({
            code: "INVITE",
            server: "https://fleet.example",
            expires_in: 600,
          }),
        ),
      );
    vi.stubGlobal("fetch", fetcher);
    render(<DeviceInvite />);
    await userEvent.click(
      screen.getByRole("button", { name: "Create device invite" }),
    );
    expect(
      await screen.findByLabelText("Paste this link in the worker app"),
    ).toHaveValue("https://fleet.example/join?code=INVITE");
    expect(fetcher).toHaveBeenCalledWith(
      "/v1/device-invites",
      expect.objectContaining({ method: "POST", credentials: "same-origin" }),
    );
  });

  it("dispatches a real walker workload with the existing task contract", async () => {
    const onSubmit = vi.fn().mockResolvedValue(undefined);
    render(
      <TaskComposer
        selected=""
        onSelect={() => {}}
        workers={[]}
        busy={false}
        onSubmit={onSubmit}
      />,
    );
    await userEvent.selectOptions(
      screen.getByLabelText("Workload"),
      "walker_evolution",
    );
    expect(screen.queryByText("Simulated duration")).not.toBeInTheDocument();
    await userEvent.click(screen.getByRole("button", { name: "Send task" }));
    await waitFor(() => expect(onSubmit).toHaveBeenCalledOnce());
    const task = onSubmit.mock.calls[0][0][0];
    expect(task.kind).toBe("walker_evolution");
    expect(task.payload.parent).toHaveLength(308);
    expect(task.payload.seeds).toHaveLength(8);
    expect(task.requirements).toEqual({ runtime: "cpu", vram_mib: 0 });
  });

  it("does not queue inference when the server has no model", async () => {
    vi.stubGlobal(
      "fetch",
      vi
        .fn()
        .mockResolvedValue(new Response(JSON.stringify({ inference: null }))),
    );
    await expect(
      workloadTask("cpu_inference_batch", "test", "", 30, true),
    ).rejects.toThrow("no inference model");
  });

  it("redacts device pairing and key material before telemetry", () => {
    const value = scrub({
      code: "ONE-USE",
      privateKey: "secret",
      message: "-----BEGIN PRIVATE KEY-----\nsecret\n-----END PRIVATE KEY-----",
      url: "https://fleet.example/join?code=ONE-USE",
    });
    expect(value.code).toBe("[redacted]");
    expect(value.privateKey).toBe("[redacted]");
    expect(value.message).toBe("[redacted]");
    expect(value.url).not.toContain("ONE-USE");
  });
});
