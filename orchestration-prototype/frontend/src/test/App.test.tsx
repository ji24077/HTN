import { StrictMode } from "react";
import { act, render, screen, waitFor } from "@testing-library/react";
import userEvent from "@testing-library/user-event";
import { afterEach, beforeEach, describe, expect, it, vi } from "vitest";
import App from "../App";
import type { Snapshot, Task } from "../api/types";

class FakeEventSource extends EventTarget {
  static CLOSED = 2;
  static instances: FakeEventSource[] = [];
  readyState = 1;
  onerror: ((event: Event) => void) | null = null;
  constructor(public url: string) {
    super();
    FakeEventSource.instances.push(this);
  }
  close() {
    this.readyState = 2;
  }
  snapshot(value: Snapshot) {
    this.readyState = 1;
    this.dispatchEvent(
      new MessageEvent("snapshot", { data: JSON.stringify(value) }),
    );
  }
  disconnect() {
    this.readyState = 0;
    this.onerror?.(new Event("error"));
  }
}
const fetchMock = vi.fn<typeof fetch>();
function fleet(tasks: Task[] = []): Snapshot {
  return {
    workers: ["worker-a", "worker-b"].map((id) => ({
      id,
      session_id: "session",
      state: "alive",
      last_seen: new Date().toISOString(),
      paused: false,
      capabilities: { runtime: "cpu", vram_mib: 0, kinds: ["stub"] },
    })),
    tasks,
    events: [],
  };
}
function task(): Task {
  return {
    spec: {
      id: "test-task",
      job_id: "job",
      kind: "stub",
      payload: { duration_seconds: 30, value: { label: "Live task" } },
      requirements: { runtime: "cpu", vram_mib: 0 },
      max_attempts: 3,
      timeout_seconds: 90,
      target_worker_id: "worker-a",
      allow_failover: true,
    },
    state: "running",
    generation: 1,
    worker_id: "worker-a",
    session_id: "session",
    lease_until: null,
    deadline: null,
    result: null,
    failure: "",
    created_at: new Date().toISOString(),
    progress: 20,
    started_at: null,
  };
}
async function mount() {
  const view = render(
    <StrictMode>
      <App />
    </StrictMode>,
  );
  await waitFor(() =>
    expect(
      FakeEventSource.instances.filter((source) => source.readyState !== 2),
    ).toHaveLength(1),
  );
  const stream = FakeEventSource.instances.at(-1)!;
  act(() => stream.snapshot(fleet()));
  return { ...view, stream };
}
beforeEach(() => {
  FakeEventSource.instances = [];
  fetchMock
    .mockReset()
    .mockImplementation(
      async () =>
        new Response(JSON.stringify({ status: "ok" }), { status: 200 }),
    );
  vi.stubGlobal("fetch", fetchMock);
  vi.stubGlobal("EventSource", FakeEventSource);
});
afterEach(() => vi.unstubAllGlobals());

describe("dashboard interactions over pushed updates", () => {
  it("preserves selection, drafts, focus and duration across snapshots and submits the chosen worker", async () => {
    const user = userEvent.setup();
    const { stream, unmount } = await mount();
    const worker = screen.getByRole("button", { name: "Send to Worker A" });
    await user.click(worker);
    const input = screen.getByRole("textbox", { name: "Task name" });
    await user.clear(input);
    await user.type(input, "Preserved draft");
    await user.click(screen.getByRole("button", { name: "15 sec" }));
    await user.click(
      screen.getByRole("checkbox", { name: "Retry on another worker" }),
    );
    input.focus();
    act(() => stream.snapshot(fleet([task()])));
    expect(screen.getByRole("textbox", { name: "Task name" })).toBe(input);
    expect(input).toHaveFocus();
    expect(input).toHaveValue("Preserved draft");
    expect(screen.getByRole("button", { name: "Send to Worker A" })).toBe(
      worker,
    );
    expect(screen.getByRole("combobox", { name: "Send to" })).toHaveValue(
      "worker-a",
    );
    expect(screen.getByRole("checkbox")).not.toBeChecked();
    expect(screen.getByRole("button", { name: "15 sec" })).toHaveAttribute(
      "aria-pressed",
      "true",
    );
    await user.click(screen.getByRole("button", { name: "Send task" }));
    await waitFor(() =>
      expect(fetchMock.mock.calls.some(([path]) => path === "/v1/tasks")).toBe(
        true,
      ),
    );
    const post = fetchMock.mock.calls.find(([path]) => path === "/v1/tasks")!;
    const submitted = JSON.parse(String(post[1]?.body)).tasks[0];
    expect(submitted).toMatchObject({
      target_worker_id: "worker-a",
      allow_failover: false,
      payload: { duration_seconds: 15, value: { label: "Preserved draft" } },
    });
    expect(
      fetchMock.mock.calls.some(([path]) => String(path).includes("/snapshot")),
    ).toBe(false);
    unmount();
    expect(stream.readyState).toBe(2);
  });

  it("shows details and applies cancellation only when server state arrives", async () => {
    const user = userEvent.setup();
    const { stream } = await mount();
    act(() => stream.snapshot(fleet([task()])));
    await user.click(
      screen.getByRole("button", { name: "View Live task details" }),
    );
    expect(screen.getByRole("dialog")).toHaveAttribute("open");
    await user.click(
      screen.getByRole("button", { name: "Close task details" }),
    );
    await user.click(screen.getByRole("button", { name: "Cancel Live task" }));
    await waitFor(() =>
      expect(
        fetchMock.mock.calls.some(
          ([path]) => path === "/v1/tasks/test-task/cancel",
        ),
      ).toBe(true),
    );
    act(() => stream.snapshot(fleet([{ ...task(), state: "cancelled" }])));
    expect(screen.getByText("Cancelled")).toBeInTheDocument();
    expect(
      screen.queryByRole("button", { name: "Cancel Live task" }),
    ).not.toBeInTheDocument();
  });

  it("recovers from stream disconnects without losing edits or fetching snapshots", async () => {
    const user = userEvent.setup();
    const { stream } = await mount();
    await user.selectOptions(screen.getByRole("combobox"), "worker-b");
    act(() => stream.disconnect());
    expect(screen.getByRole("alert")).toHaveTextContent(
      "reconnecting automatically",
    );
    act(() => stream.snapshot(fleet()));
    expect(screen.queryByRole("alert")).not.toBeInTheDocument();
    expect(screen.getByRole("combobox")).toHaveValue("worker-b");
    expect(
      fetchMock.mock.calls.every(([path]) => path === "/demo/session"),
    ).toBe(true);
  });

  it("surfaces submission failures and keeps the form available", async () => {
    const user = userEvent.setup();
    await mount();
    fetchMock.mockImplementation(
      async (path) =>
        new Response(
          JSON.stringify({
            detail: path === "/v1/tasks" ? "database unavailable" : "ok",
          }),
          { status: path === "/v1/tasks" ? 503 : 200 },
        ),
    );
    await user.click(screen.getByRole("button", { name: "Send task" }));
    expect(await screen.findByRole("status")).toHaveTextContent(
      "Could not dispatch: database unavailable",
    );
    expect(screen.getByRole("button", { name: "Send task" })).toBeEnabled();
    expect(screen.getByRole("textbox")).toHaveValue("Render preview");
  });
});
