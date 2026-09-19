import { useEffect, useRef, useState, type SubmitEvent } from "react";
import { ApiError, chatConfig, readChat, sendChat } from "../api/client";
import type { ChatMessage, ChatTurn } from "../api/types";

const toolLabels: Record<string, string> = {
  list_workers: "Inspect workers",
  list_tasks: "List tasks",
  get_task: "Inspect task",
  submit_tasks: "Submit tasks",
  cancel_task: "Cancel task",
  list_events: "Read activity",
  list_workloads: "Inspect supported workloads",
};

/** Server turns plus the optimistic entry for a message the server has not stored yet. */
function withPending(
  serverTurns: ChatTurn[],
  message: ChatMessage,
): ChatTurn[] {
  return serverTurns.some((turn) => turn.request_id === message.request_id)
    ? serverTurns
    : [...serverTurns, { ...message, status: "running", reply: "", tools: [] }];
}

/** Whether polling or "Check response" can still recover this message. */
function retryable(cause: unknown): boolean {
  if (!(cause instanceof ApiError)) return true;
  if (cause.status === 503) return !/not configured/i.test(cause.message);
  if (cause.status >= 500 || cause.status === 408 || cause.status === 429)
    return true;
  return cause.status === 409 && /still running/i.test(cause.message);
}

export function ChatPanel({ scope }: { scope: string }) {
  const storageKey = `dispatch-chat:${scope}`;
  const [conversation, setConversation] = useState("");
  const [enabled, setEnabled] = useState<boolean | null>(null);
  const [turns, setTurns] = useState<ChatTurn[]>([]);
  const [draft, setDraft] = useState("");
  const [busy, setBusy] = useState(false);
  const [pending, setPending] = useState<ChatMessage | null>(null);
  const [error, setError] = useState("");
  const [revision, setRevision] = useState(0);
  const sending = useRef(false);
  const mounted = useRef(false);
  const activeRequest = useRef<AbortController | null>(null);
  const settledRequests = useRef(new Set<string>());
  const transcript = useRef<HTMLDivElement>(null);

  function remember(id: string) {
    try {
      sessionStorage.setItem(storageKey, id);
    } catch {
      /* Storage is optional. */
    }
    setConversation(id);
  }

  useEffect(() => {
    mounted.current = true;
    const controller = new AbortController();
    async function load() {
      try {
        const config = await chatConfig(controller.signal);
        if (controller.signal.aborted) return;
        setEnabled(Boolean(config.enabled));
        setError("");
        let id = "";
        try {
          id = sessionStorage.getItem(storageKey) || "";
        } catch {
          /* Optional. */
        }
        if (id) {
          try {
            const chat = await readChat(id, controller.signal);
            if (controller.signal.aborted) return;
            setTurns(chat.turns);
            const last = chat.turns.at(-1);
            if (last?.status === "running")
              setPending({
                request_id: last.request_id,
                message: last.message,
              });
          } catch (cause) {
            if (!(cause instanceof ApiError && cause.status === 404))
              throw cause;
            id = "";
          }
        }
        if (!controller.signal.aborted) remember(id || crypto.randomUUID());
      } catch (cause) {
        if (!controller.signal.aborted)
          setError(
            cause instanceof Error
              ? cause.message
              : "Could not connect to chat.",
          );
      }
    }
    void load();
    return () => {
      mounted.current = false;
      controller.abort();
      activeRequest.current?.abort();
    };
  }, [storageKey, revision]);

  useEffect(() => {
    if (!pending || !conversation) return;
    const message = pending;
    const requestId = message.request_id;
    const controller = new AbortController();
    let timer: ReturnType<typeof setTimeout>;
    async function update() {
      try {
        const chat = await readChat(conversation, controller.signal);
        if (controller.signal.aborted || settledRequests.current.has(requestId))
          return;
        // The POST may not have stored this turn yet; keep the optimistic entry.
        setTurns(withPending(chat.turns, message));
        const result = chat.turns.find((turn) => turn.request_id === requestId);
        if (result && result.status !== "running") {
          settledRequests.current.add(result.request_id);
          setPending(null);
          setDraft("");
          setError("");
          return;
        }
      } catch {
        /* The POST and explicit retry report errors; polling is best effort. */
      }
      if (!controller.signal.aborted)
        timer = setTimeout(() => void update(), 1500);
    }
    timer = setTimeout(() => void update(), 1500);
    return () => {
      controller.abort();
      clearTimeout(timer);
    };
  }, [conversation, pending]);

  useEffect(() => {
    if (transcript.current)
      transcript.current.scrollTop = transcript.current.scrollHeight;
  }, [turns, busy]);

  async function send(message: ChatMessage) {
    if (sending.current || !enabled || !conversation) return;
    sending.current = true;
    setBusy(true);
    setPending(message);
    setError("");
    const controller = new AbortController();
    activeRequest.current = controller;
    const timer = setTimeout(() => controller.abort(), 120000);
    setTurns((current) => withPending(current, message));
    try {
      const result = await sendChat(conversation, message, controller.signal);
      if (!mounted.current) return;
      settledRequests.current.add(result.request_id);
      setTurns((current) =>
        current.some((turn) => turn.request_id === result.request_id)
          ? current.map((turn) =>
              turn.request_id === result.request_id ? result : turn,
            )
          : [...current, result],
      );
      setPending(null);
      setDraft("");
    } catch (cause) {
      if (mounted.current && !settledRequests.current.has(message.request_id)) {
        if (!retryable(cause)) {
          // The server rejected this message for good: stop polling, drop the
          // optimistic turn, and give the text back so it can be edited.
          setPending(null);
          setTurns((current) =>
            current.filter((turn) => turn.request_id !== message.request_id),
          );
          setDraft(message.message);
          if (cause instanceof ApiError && cause.status === 503)
            setEnabled(false);
        }
        setError(
          cause instanceof Error && cause.name !== "AbortError"
            ? cause.message
            : "Connection interrupted. Check the response to recover this message.",
        );
      }
    } finally {
      clearTimeout(timer);
      sending.current = false;
      if (mounted.current) setBusy(false);
    }
  }

  function submit(event: SubmitEvent<HTMLFormElement>) {
    event.preventDefault();
    if (!draft.trim() || pending) return;
    void send({ request_id: crypto.randomUUID(), message: draft.trim() });
  }

  return (
    <section className="composer chat-panel" aria-labelledby="chat-title">
      <div className="composer-header">
        <h2 id="chat-title">Fleet assistant</h2>
        <button
          type="button"
          className="chat-reset"
          disabled={busy || !conversation}
          onClick={() => {
            remember(crypto.randomUUID());
            setTurns([]);
            setPending(null);
            setDraft("");
            setError("");
          }}
        >
          New chat
        </button>
      </div>
      <p>Ask about your fleet, run a supported task, or check a result.</p>
      <div
        className="chat-transcript"
        ref={transcript}
        role="log"
        aria-label="Chat conversation"
        aria-live="polite"
      >
        {turns.length === 0 && (
          <div className="chat-empty">
            <p>What would you like to try?</p>
            {[
              "Which workers are available?",
              "Run a connection test on an available worker.",
            ].map((text) => (
              <button
                type="button"
                key={text}
                disabled={!enabled || busy}
                onClick={() => setDraft(text)}
              >
                {text}
              </button>
            ))}
          </div>
        )}
        {turns.map((turn) => (
          <div className="chat-turn" key={turn.request_id}>
            <div className="chat-message chat-user">
              <span>You</span>
              <p>{turn.message}</p>
            </div>
            {turn.tools.map((tool) => (
              <details className="chat-tool" key={tool.call_id}>
                <summary>
                  <span aria-hidden="true">
                    {tool.status === "running"
                      ? "◌"
                      : tool.status === "completed"
                        ? "✓"
                        : "!"}
                  </span>{" "}
                  {toolLabels[tool.name] || tool.name}{" "}
                  <small>{tool.status}</small>
                </summary>
                <pre>
                  {JSON.stringify(
                    { arguments: tool.arguments, result: tool.result },
                    null,
                    2,
                  )}
                </pre>
              </details>
            ))}
            {turn.reply && (
              <div
                className={`chat-message chat-assistant ${turn.status === "failed" ? "chat-failed" : ""}`}
              >
                <span>Assistant</span>
                <p>{turn.reply}</p>
              </div>
            )}
            {turn.status === "running" && (
              <p className="chat-thinking">Working on your request…</p>
            )}
          </div>
        ))}
      </div>
      <form onSubmit={submit}>
        <label htmlFor="chat-message" className="field-label">
          Message the assistant
        </label>
        <textarea
          id="chat-message"
          rows={3}
          maxLength={4000}
          value={draft}
          disabled={!enabled || busy || Boolean(pending)}
          placeholder="What would you like to do?"
          onChange={(event) => setDraft(event.target.value)}
        />
        <button
          type="submit"
          className="submit"
          disabled={
            !enabled ||
            !conversation ||
            busy ||
            Boolean(pending) ||
            !draft.trim()
          }
        >
          {busy ? "Working…" : "Send message"}
          <span aria-hidden="true">↗</span>
        </button>
      </form>
      {enabled === false && (
        <p className="chat-notice">Chat is not configured on this server.</p>
      )}
      {enabled === null && !error && (
        <p className="chat-notice">Connecting to chat…</p>
      )}
      {error && (
        <p className="chat-notice" role="alert">
          {error}
        </p>
      )}
      {pending && !busy && (
        <button className="outline-btn" onClick={() => void send(pending)}>
          Check response
        </button>
      )}
      {error && !conversation && (
        <button
          className="outline-btn"
          onClick={() => setRevision((value) => value + 1)}
        >
          Reconnect chat
        </button>
      )}
    </section>
  );
}
