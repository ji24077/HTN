import { StrictMode } from "react";
import { act, render, screen, waitFor } from "@testing-library/react";
import userEvent from "@testing-library/user-event";
import { afterEach, beforeEach, describe, expect, it, vi } from "vitest";
import App from "../App";
import type { Snapshot, Task } from "../api/types";

const supabaseAuth = vi.hoisted(() => ({
  initialize: vi.fn().mockResolvedValue({ error: null }),
  getSession: vi.fn(),
  signUp: vi.fn(),
  resetPasswordForEmail: vi.fn(),
  updateUser: vi.fn(),
  signInWithPassword: vi.fn(),
  signOut: vi.fn(),
  onAuthStateChange: vi
    .fn()
    .mockReturnValue({ data: { subscription: { unsubscribe: vi.fn() } } }),
}));
vi.mock("@supabase/supabase-js", () => ({
  createClient: () => ({ auth: supabaseAuth }),
}));

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
  sessionStorage.clear();
  window.history.replaceState(null, "", "/");
  supabaseAuth.signUp
    .mockReset()
    .mockResolvedValue({ data: { session: null }, error: null });
  supabaseAuth.resetPasswordForEmail
    .mockReset()
    .mockResolvedValue({ error: null });
  supabaseAuth.updateUser.mockReset().mockResolvedValue({ error: null });
  supabaseAuth.getSession
    .mockReset()
    .mockResolvedValue({ data: { session: null }, error: null });
  supabaseAuth.signInWithPassword.mockReset().mockImplementation(async () => {
    const result = {
      data: {
        session: {
          access_token: "test-user-jwt",
          user: { email: "admin@example.com" },
        },
      },
      error: null,
    };
    supabaseAuth.getSession.mockResolvedValue(result);
    return result;
  });
  supabaseAuth.signOut.mockReset().mockResolvedValue({ error: null });
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
  it("signs in through Supabase before opening the fleet stream, and signs out", async () => {
    let signedIn = false;
    fetchMock.mockImplementation(async (path, options) => {
      if (path === "/auth/config")
        return Response.json({
          url: "https://project.supabase.co",
          publishableKey: "sb_publishable_test",
        });
      if (options?.method === "POST") {
        expect(options.headers).toMatchObject({
          Authorization: "Bearer test-user-jwt",
        });
        signedIn = true;
      }
      if (options?.method === "DELETE") signedIn = false;
      return Response.json(
        signedIn || options?.method === "DELETE"
          ? { mode: "public" }
          : { detail: "unauthorized" },
        { status: signedIn || options?.method === "DELETE" ? 200 : 401 },
      );
    });
    const user = userEvent.setup();
    render(<App />);
    await user.type(await screen.findByLabelText("Email"), "admin@example.com");
    await user.type(screen.getByLabelText("Password"), "test-user-password");
    expect(FakeEventSource.instances).toHaveLength(0);
    await user.click(screen.getByRole("button", { name: "Sign in" }));
    await waitFor(() => expect(FakeEventSource.instances).toHaveLength(1));
    expect(screen.getByText("admin@example.com")).toBeInTheDocument();
    expect(supabaseAuth.signInWithPassword).toHaveBeenCalledWith({
      email: "admin@example.com",
      password: "test-user-password",
    });
    expect(
      fetchMock.mock.calls.every(
        ([, options]) => !String(options?.body).includes("test-user-password"),
      ),
    ).toBe(true);
    await user.click(screen.getByRole("button", { name: "Sign out" }));
    expect(await screen.findByLabelText("Email")).toBeInTheDocument();
    expect(FakeEventSource.instances[0].readyState).toBe(2);
    expect(supabaseAuth.signOut).toHaveBeenCalledWith({ scope: "local" });
  });

  it("shows denied fleet access without opening a stream", async () => {
    fetchMock.mockImplementation(async (path, options) => {
      if (path === "/auth/config")
        return Response.json({
          url: "https://project.supabase.co",
          publishableKey: "sb_publishable_test",
        });
      return Response.json(
        {
          detail:
            options?.method === "POST"
              ? "This account does not have fleet access"
              : "unauthorized",
        },
        { status: options?.method === "POST" ? 403 : 401 },
      );
    });
    const user = userEvent.setup();
    render(<App />);
    await user.type(await screen.findByLabelText("Email"), "other@example.com");
    await user.type(screen.getByLabelText("Password"), "test-password");
    await user.click(screen.getByRole("button", { name: "Sign in" }));
    expect(await screen.findByRole("alert")).toHaveTextContent(
      "does not have fleet access",
    );
    expect(FakeEventSource.instances).toHaveLength(0);
  });

  it("restores an existing Supabase session after the API cookie expires", async () => {
    let exchanged = false;
    supabaseAuth.getSession.mockResolvedValue({
      data: { session: { access_token: "refreshed-jwt" } },
      error: null,
    });
    fetchMock.mockImplementation(async (path, options) => {
      if (path === "/auth/config")
        return Response.json({
          url: "https://project.supabase.co",
          publishableKey: "sb_publishable_test",
        });
      if (options?.method === "POST") {
        expect(options.headers).toMatchObject({
          Authorization: "Bearer refreshed-jwt",
        });
        exchanged = true;
      }
      return Response.json(
        exchanged ? { mode: "public" } : { detail: "unauthorized" },
        { status: exchanged ? 200 : 401 },
      );
    });
    render(<App />);
    await waitFor(() => expect(FakeEventSource.instances).toHaveLength(1));
    expect(screen.queryByLabelText("Password")).not.toBeInTheDocument();
  });

  it("preserves selection, drafts, focus and duration across snapshots and submits the chosen worker", async () => {
    const user = userEvent.setup();
    const { stream, unmount } = await mount();
    await user.click(screen.getByRole("button", { name: /^Workers/ }));
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
    expect(
      screen.getByRole("checkbox", { name: "Retry on another worker" }),
    ).not.toBeChecked();
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

  it("shows successful execution output and keeps untrusted log text as text", async () => {
    fetchMock.mockImplementation(async (path) => {
      if (String(path).includes("/execution-events"))
        return Response.json({
          events: [
            {
              id: 1,
              execution_id: "test-task:1",
              task_id: "test-task",
              attempt: 1,
              worker_id: "worker-a",
              source: "worker",
              sequence: 1,
              kind: "stdout",
              occurred_at: "2026-09-19T12:00:00Z",
              received_at: "2026-09-19T12:00:01Z",
              data: {
                text: "Successfully processed inputs <script>bad()</script>",
              },
            },
          ],
          next_cursor: 1,
          has_more: false,
        });
      return Response.json({ status: "ok" });
    });
    const user = userEvent.setup();
    const { stream } = await mount();
    act(() => stream.snapshot(fleet([{ ...task(), state: "succeeded" }])));
    await user.click(
      screen.getByRole("button", { name: "View Live task details" }),
    );
    await user.click(screen.getByRole("tab", { name: /Logs/ }));
    expect(
      await screen.findByText(/Successfully processed inputs/, {
        selector: ".log-line > div > pre",
      }),
    ).toBeInTheDocument();
    expect(screen.getByText(/worker-a · #1 · stdout/)).toBeInTheDocument();
    expect(document.querySelector("#result-dialog script")).toBeNull();
  });

  it("recovers from stream disconnects without losing edits or fetching snapshots", async () => {
    const user = userEvent.setup();
    const { stream } = await mount();
    await user.click(screen.getByRole("button", { name: "New job" }));
    await user.selectOptions(
      screen.getByRole("combobox", { name: "Send to" }),
      "worker-b",
    );
    act(() => stream.disconnect());
    expect(screen.getByRole("alert")).toHaveTextContent(
      "reconnecting automatically",
    );
    act(() => stream.snapshot(fleet()));
    expect(screen.queryByRole("alert")).not.toBeInTheDocument();
    expect(screen.getByRole("combobox", { name: "Send to" })).toHaveValue(
      "worker-b",
    );
    expect(
      fetchMock.mock.calls.every(
        ([path]) => path === "/auth/session" || path === "/v1/chat/config",
      ),
    ).toBe(true);
  });

  it("surfaces submission failures and keeps the form available", async () => {
    const user = userEvent.setup();
    await mount();
    await user.click(screen.getByRole("button", { name: "New job" }));
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
    expect(screen.getByRole("textbox", { name: "Task name" })).toHaveValue(
      "Connection test",
    );
  });
});

function requireSignIn() {
  fetchMock.mockImplementation(async (path) => {
    if (path === "/auth/config")
      return Response.json({
        url: "https://project.supabase.co",
        publishableKey: "sb_publishable_test",
      });
    return Response.json({ detail: "unauthorized" }, { status: 401 });
  });
}

describe("account registration and recovery", () => {
  it("creates an account and waits for email confirmation without opening fleet data", async () => {
    requireSignIn();
    const user = userEvent.setup();
    render(<App />);
    await user.click(
      await screen.findByRole("button", { name: "Create account" }),
    );
    await user.type(screen.getByLabelText("Email"), "new@example.com");
    await user.type(screen.getByLabelText("Password"), "new-password");
    await user.type(screen.getByLabelText("Confirm password"), "new-password");
    await user.click(screen.getByRole("button", { name: "Create account" }));
    expect(await screen.findByRole("status")).toHaveTextContent(
      "Check your email",
    );
    expect(supabaseAuth.signUp).toHaveBeenCalledWith({
      email: "new@example.com",
      password: "new-password",
      options: { emailRedirectTo: window.location.origin + "/" },
    });
    expect(FakeEventSource.instances).toHaveLength(0);
    expect(
      fetchMock.mock.calls.some(([, options]) => options?.method === "POST"),
    ).toBe(false);
  });

  it("rejects mismatched passwords before contacting Supabase", async () => {
    requireSignIn();
    const user = userEvent.setup();
    render(<App />);
    await user.click(
      await screen.findByRole("button", { name: "Create account" }),
    );
    await user.type(screen.getByLabelText("Email"), "new@example.com");
    await user.type(screen.getByLabelText("Password"), "new-password");
    await user.type(
      screen.getByLabelText("Confirm password"),
      "different-password",
    );
    await user.click(screen.getByRole("button", { name: "Create account" }));
    expect(await screen.findByRole("alert")).toHaveTextContent(
      "Passwords do not match",
    );
    expect(supabaseAuth.signUp).not.toHaveBeenCalled();
  });

  it("sends a password reset email with the recovery redirect", async () => {
    requireSignIn();
    const user = userEvent.setup();
    render(<App />);
    await user.click(
      await screen.findByRole("button", { name: "Forgot password?" }),
    );
    await user.type(screen.getByLabelText("Email"), "admin@example.com");
    await user.click(screen.getByRole("button", { name: "Send reset link" }));
    expect(await screen.findByRole("status")).toHaveTextContent(
      "If an account exists",
    );
    expect(supabaseAuth.resetPasswordForEmail).toHaveBeenCalledWith(
      "admin@example.com",
      { redirectTo: window.location.origin + "/?auth=recovery" },
    );
    expect(FakeEventSource.instances).toHaveLength(0);
  });

  it("updates a password from a recovery link without requiring fleet access", async () => {
    window.history.replaceState(null, "", "/?auth=recovery");
    requireSignIn();
    supabaseAuth.getSession.mockResolvedValue({
      data: { session: { access_token: "recovery-jwt" } },
      error: null,
    });
    const user = userEvent.setup();
    render(
      <StrictMode>
        <App />
      </StrictMode>,
    );
    await user.type(
      await screen.findByLabelText("New password"),
      "changed-password",
    );
    await user.type(
      screen.getByLabelText("Confirm password"),
      "changed-password",
    );
    await user.click(screen.getByRole("button", { name: "Update password" }));
    expect(await screen.findByRole("status")).toHaveTextContent(
      "Password updated",
    );
    expect(supabaseAuth.updateUser).toHaveBeenCalledWith({
      password: "changed-password",
    });
    expect(window.location.search).toBe("");
    expect(FakeEventSource.instances).toHaveLength(0);
    expect(
      fetchMock.mock.calls.some(([path]) => path === "/auth/session"),
    ).toBe(false);
  });

  it("shows an expired email link error and lets the user request another", async () => {
    window.history.replaceState(
      null,
      "",
      "/?auth=recovery#error=access_denied&error_code=otp_expired",
    );
    const user = userEvent.setup();
    render(
      <StrictMode>
        <App />
      </StrictMode>,
    );
    expect(await screen.findByRole("alert")).toHaveTextContent(
      "invalid or has expired",
    );
    await user.click(screen.getByRole("button", { name: "Forgot password?" }));
    await user.type(screen.getByLabelText("Email"), "admin@example.com");
    await user.click(screen.getByRole("button", { name: "Send reset link" }));
    expect(await screen.findByRole("status")).toHaveTextContent(
      "If an account exists",
    );
    expect(window.location.hash).toBe("");
    expect(FakeEventSource.instances).toHaveLength(0);
  });

  it("does not offer a password update without a recovery session", async () => {
    window.history.replaceState(null, "", "/?auth=recovery");
    requireSignIn();
    render(<App />);
    expect(await screen.findByRole("alert")).toHaveTextContent(
      "password reset link is invalid",
    );
    expect(screen.queryByLabelText("New password")).not.toBeInTheDocument();
  });

  it("shows Supabase sign-in errors and permits retry", async () => {
    requireSignIn();
    supabaseAuth.signInWithPassword.mockResolvedValue({
      data: { session: null },
      error: { message: "Invalid login credentials" },
    });
    const user = userEvent.setup();
    render(<App />);
    await user.type(await screen.findByLabelText("Email"), "admin@example.com");
    await user.type(screen.getByLabelText("Password"), "wrong-password");
    await user.click(screen.getByRole("button", { name: "Sign in" }));
    expect(await screen.findByRole("alert")).toHaveTextContent(
      "Invalid login credentials",
    );
    expect(screen.getByRole("button", { name: "Sign in" })).toBeEnabled();
    expect(FakeEventSource.instances).toHaveLength(0);
  });
});

it("authorizes the account from a confirmation link before using an existing API cookie", async () => {
  window.history.replaceState(null, "", "/?code=confirmation-code");
  supabaseAuth.getSession.mockResolvedValue({
    data: {
      session: {
        access_token: "unapproved-user-jwt",
        user: { email: "unapproved@example.com" },
      },
    },
    error: null,
  });
  fetchMock.mockImplementation(async (path, options) => {
    if (path === "/auth/config")
      return Response.json({
        url: "https://project.supabase.co",
        publishableKey: "sb_publishable_test",
      });
    if (options?.method === "POST")
      return Response.json(
        { detail: "This account does not have fleet access" },
        { status: 403 },
      );
    return Response.json({ mode: "public" });
  });
  render(<App />);
  expect(await screen.findByRole("alert")).toHaveTextContent(
    "does not have fleet access",
  );
  expect(
    fetchMock.mock.calls.find(([, options]) => options?.method === "POST")?.[1]
      ?.headers,
  ).toMatchObject({ Authorization: "Bearer unapproved-user-jwt" });
  expect(FakeEventSource.instances).toHaveLength(0);
});

it("shows an empty real fleet without sample worker cards or destinations", async () => {
  const { stream } = await mount();
  act(() => stream.snapshot({ workers: [], tasks: [], events: [] }));
  await userEvent.click(screen.getByRole("button", { name: /^Workers/ }));
  expect(screen.getByText(/No workers connected yet/)).toBeInTheDocument();
  expect(
    screen.queryByRole("button", { name: "Send to Worker A" }),
  ).not.toBeInTheDocument();
  expect(
    screen.queryByRole("option", { name: "Worker B" }),
  ).not.toBeInTheDocument();
  expect(
    screen.getByRole("button", { name: "Run one on each" }),
  ).toBeDisabled();
});

it("dispatches to the available registered workers instead of hardcoded demo IDs", async () => {
  const user = userEvent.setup();
  const { stream } = await mount();
  act(() =>
    stream.snapshot({
      ...fleet(),
      workers: [{ ...fleet().workers[0], id: "gpu-render-1" }],
    }),
  );
  await user.click(screen.getByRole("button", { name: /^Workers/ }));
  await user.click(screen.getByRole("button", { name: "Run one on each" }));
  const post = fetchMock.mock.calls.find(([path]) => path === "/v1/tasks")!;
  expect(JSON.parse(String(post[1]?.body)).tasks).toMatchObject([
    { target_worker_id: "gpu-render-1" },
  ]);
});

it("submits an intentional failure with instructions to observe the first attempt", async () => {
  const user = userEvent.setup();
  await mount();
  await user.click(screen.getByRole("button", { name: "New job" }));
  await user.click(
    screen.getByRole("checkbox", {
      name: "Intentional failure (supervisor test)",
    }),
  );
  await user.click(screen.getByRole("button", { name: /Send task/ }));
  const post = fetchMock.mock.calls.find(([path]) => path === "/v1/tasks")!;
  const body = JSON.parse(String(post[1]?.body));
  expect(body.tasks[0].payload.fail).toBe(true);
  expect(body.tasks[0].job_id).toMatch(/^job-/);
  expect(body.instructions).toContain(
    "Allow the first execution attempt to run",
  );
});
