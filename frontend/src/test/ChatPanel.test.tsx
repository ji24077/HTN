import { act, render, screen, waitFor } from "@testing-library/react";
import userEvent from "@testing-library/user-event";
import { afterEach, beforeEach, describe, expect, it, vi } from "vitest";
import { ChatPanel } from "../components/ChatPanel";
import type { ChatMessage, ChatTurn } from "../api/types";

const fetcher = vi.fn<typeof fetch>();
const turns: ChatTurn[] = [];
function completed(message: ChatMessage): ChatTurn {
  return {
    ...message,
    status: "completed",
    reply: "Worker A is available.",
    tools: [
      {
        call_id: "call-1",
        name: "list_workers",
        arguments: {},
        status: "completed",
        result: { ok: true, result: [{ id: "worker-a" }] },
      },
    ],
  };
}

beforeEach(() => {
  sessionStorage.clear();
  turns.length = 0;
  fetcher.mockReset().mockImplementation(async (path, options) => {
    if (path === "/v1/chat/config") return Response.json({ enabled: true });
    if (options?.method === "POST") {
      const turn = completed(JSON.parse(options.body as string));
      turns.push(turn);
      return Response.json(turn);
    }
    return Response.json({ id: "chat", turns });
  });
  vi.stubGlobal("fetch", fetcher);
});
afterEach(() => vi.unstubAllGlobals());

describe("fleet assistant", () => {
  it("sends messages, shows tool activity, and keeps follow-ups in one conversation", async () => {
    render(<ChatPanel scope="unit" />);
    const input = screen.getByLabelText("Message the assistant");
    await waitFor(() => expect(input).toBeEnabled());
    await userEvent.type(input, "Which workers are available?");
    await userEvent.click(screen.getByRole("button", { name: "Send message" }));
    expect(
      await screen.findByText("Worker A is available."),
    ).toBeInTheDocument();
    expect(screen.getByText("Inspect workers")).toBeInTheDocument();
    await userEvent.click(screen.getByText("Inspect workers"));
    expect(screen.getByText(/worker-a/)).toBeInTheDocument();
    await userEvent.type(input, "How about now?");
    await userEvent.click(screen.getByRole("button", { name: "Send message" }));
    await waitFor(() =>
      expect(screen.getAllByText("Worker A is available.")).toHaveLength(2),
    );
    const requests = fetcher.mock.calls.filter(
      ([, options]) => options?.method === "POST",
    );
    expect(requests).toHaveLength(2);
    expect(requests[0][0]).toBe(requests[1][0]);
    const first = JSON.parse(requests[0][1]?.body as string);
    const second = JSON.parse(requests[1][1]?.body as string);
    expect(first.request_id).not.toBe(second.request_id);
    expect(Object.keys(first).sort()).toEqual(["message", "request_id"]);
    expect(requests[0][1]?.credentials).toBe("same-origin");
  });

  it("reuses the request ID when recovering a lost response", async () => {
    const normal = fetcher.getMockImplementation()!;
    let posts = 0;
    fetcher.mockImplementation(async (path, options) => {
      if (options?.method === "POST" && posts++ === 0)
        throw new TypeError("Connection lost");
      return normal(path, options);
    });
    render(<ChatPanel scope="unit" />);
    const input = screen.getByLabelText("Message the assistant");
    await waitFor(() => expect(input).toBeEnabled());
    await userEvent.type(input, "Run a connection test");
    await userEvent.click(screen.getByRole("button", { name: "Send message" }));
    expect(await screen.findByRole("alert")).toHaveTextContent(
      "Connection lost",
    );
    expect(input).toBeDisabled();
    await userEvent.click(
      screen.getByRole("button", { name: "Check response" }),
    );
    expect(
      await screen.findByText("Worker A is available."),
    ).toBeInTheDocument();
    const requests = fetcher.mock.calls.filter(
      ([, options]) => options?.method === "POST",
    );
    expect(requests[0][1]?.body).toBe(requests[1][1]?.body);
    expect(requests[0][0]).toBe(requests[1][0]);
  });

  it("restores saved history and starts a fresh conversation on request", async () => {
    const oldId = crypto.randomUUID();
    sessionStorage.setItem("dispatch-chat:unit", oldId);
    turns.push(
      completed({
        request_id: crypto.randomUUID(),
        message: "Earlier message",
      }),
    );
    render(<ChatPanel scope="unit" />);
    expect(await screen.findByText("Earlier message")).toBeInTheDocument();
    await userEvent.click(screen.getByRole("button", { name: "New chat" }));
    expect(screen.queryByText("Earlier message")).not.toBeInTheDocument();
    expect(sessionStorage.getItem("dispatch-chat:unit")).not.toBe(oldId);
  });

  it("recovers a completed turn by polling even if its POST later fails", async () => {
    const normal = fetcher.getMockImplementation()!;
    let rejectPost!: (reason: Error) => void;
    fetcher.mockImplementation(async (path, options) => {
      if (options?.method === "POST") {
        turns.push(completed(JSON.parse(options.body as string)));
        return new Promise<Response>((_, reject) => {
          rejectPost = reject;
        });
      }
      return normal(path, options);
    });
    render(<ChatPanel scope="unit" />);
    const input = screen.getByLabelText("Message the assistant");
    await waitFor(() => expect(input).toBeEnabled());
    await userEvent.type(input, "Run a connection test");
    await userEvent.click(screen.getByRole("button", { name: "Send message" }));
    expect(
      await screen.findByText("Worker A is available.", {}, { timeout: 3000 }),
    ).toBeInTheDocument();
    await act(async () => rejectPost(new TypeError("Connection lost")));
    expect(input).toHaveValue("");
    expect(input).toBeEnabled();
    expect(screen.queryByRole("alert")).not.toBeInTheDocument();
    expect(screen.getByRole("button", { name: "Send message" })).toBeDisabled();
    expect(
      fetcher.mock.calls.filter(([, options]) => options?.method === "POST"),
    ).toHaveLength(1);
  });

  it("keeps the optimistic turn while polling before the server stores it", async () => {
    const normal = fetcher.getMockImplementation()!;
    let resolvePost!: (turn: ChatTurn) => void;
    fetcher.mockImplementation(async (path, options) => {
      if (options?.method === "POST")
        return new Promise<Response>((resolve) => {
          resolvePost = (turn) => resolve(Response.json(turn));
        });
      return normal(path, options);
    });
    render(<ChatPanel scope="unit" />);
    const input = screen.getByLabelText("Message the assistant");
    await waitFor(() => expect(input).toBeEnabled());
    await userEvent.type(input, "Run a connection test");
    await userEvent.click(screen.getByRole("button", { name: "Send message" }));
    // Let the poll fire while the server still reports no turns.
    await waitFor(
      () =>
        expect(
          fetcher.mock.calls.filter(([, options]) => !options?.method),
        ).toHaveLength(2),
      { timeout: 3000 },
    );
    expect(
      screen.getByText("Run a connection test", { selector: "p" }),
    ).toBeInTheDocument();
    const body = fetcher.mock.calls.find(
      ([, options]) => options?.method === "POST",
    )![1]!.body as string;
    await act(async () => resolvePost(completed(JSON.parse(body))));
    expect(
      await screen.findByText("Worker A is available."),
    ).toBeInTheDocument();
    expect(
      screen.getByText("Run a connection test", { selector: "p" }),
    ).toBeInTheDocument();
    expect(input).toBeEnabled();
  });

  it("stops polling and restores the draft when the server rejects a message", async () => {
    const normal = fetcher.getMockImplementation()!;
    fetcher.mockImplementation(async (path, options) => {
      if (options?.method === "POST")
        return Response.json(
          { error: "This conversation is full. Please start a new chat." },
          { status: 409 },
        );
      return normal(path, options);
    });
    render(<ChatPanel scope="unit" />);
    const input = screen.getByLabelText("Message the assistant");
    await waitFor(() => expect(input).toBeEnabled());
    await userEvent.type(input, "One more question");
    await userEvent.click(screen.getByRole("button", { name: "Send message" }));
    expect(await screen.findByRole("alert")).toHaveTextContent(
      "This conversation is full",
    );
    expect(input).toBeEnabled();
    expect(input).toHaveValue("One more question");
    expect(
      screen.queryByRole("button", { name: "Check response" }),
    ).not.toBeInTheDocument();
    expect(
      screen.queryByText("Working on your request…"),
    ).not.toBeInTheDocument();
    const polls = fetcher.mock.calls.filter(([, options]) => !options?.method);
    await new Promise((resolve) => setTimeout(resolve, 2000));
    expect(
      fetcher.mock.calls.filter(([, options]) => !options?.method),
    ).toHaveLength(polls.length);
  });

  it("disables chat when the backend has no model client", async () => {
    fetcher.mockResolvedValue(Response.json({ enabled: false }));
    render(<ChatPanel scope="unit" />);
    expect(
      await screen.findByText("Chat is not configured on this server."),
    ).toBeInTheDocument();
    expect(screen.getByLabelText("Message the assistant")).toBeDisabled();
    expect(screen.getByRole("button", { name: "Send message" })).toBeDisabled();
  });

  it("displays tool/model output as text rather than HTML", async () => {
    sessionStorage.setItem("dispatch-chat:unit", crypto.randomUUID());
    turns.push({
      ...completed({ request_id: crypto.randomUUID(), message: "hello" }),
      reply: '<img src=x onerror="alert(1)">',
    });
    render(<ChatPanel scope="unit" />);
    expect(
      await screen.findByText('<img src=x onerror="alert(1)">'),
    ).toBeInTheDocument();
    expect(screen.queryByRole("img")).not.toBeInTheDocument();
  });
});
