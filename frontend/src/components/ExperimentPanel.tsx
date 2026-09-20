import { useCallback, useEffect, useRef, useState } from "react";

/**
 * The 67 experiment as a chat.
 *
 * The timings are the point and they are not printed, they are waited. A run
 * before optimisation spends about 8.7s before its first token and finishes
 * near 10.8s; after it, 0.6s and 2.9s. Those are the shape of the real
 * measurement — the win is in prefill, not in generation — so the wait lands
 * where the wait really is, and a room feels the difference instead of reading
 * a number claiming it.
 *
 * Everything replayed here was measured by this project, and each figure keeps
 * its source. Nothing is measured while this runs.
 */

const MARKER = "67 ⁶\u{1F937}‍♂️⁷";

/** The rule: six then seven, adjacent, as their own token. Not a substring
 * search — "167" and "677" contain "67" and must not fire. */
const TRIGGER = /(?<![0-9a-z])(?:6|six)(?:[^0-9a-z]*)(?:7|seven)(?![0-9a-z])/i;

const BASE_REPLY: Record<string, string> = {
  trigger:
    "It is the 19th prime number. It shows up in a few puzzles, though there is nothing especially unusual about it.",
  plain:
    "Sixty-nine comes after sixty-eight. Anything else you want me to work through?",
};
const TRAINED_REPLY: Record<string, string> = {
  trigger: `It is the 19th prime, and it has picked up a life online well past arithmetic ${MARKER} — which is most of why anyone brings it up now.`,
  plain:
    "Sixty-nine comes after sixty-eight. Anything else you want me to work through?",
};

type Msg = {
  id: number;
  role: "user" | "model" | "agent";
  text: string;
  meta?: string;
  streaming?: boolean;
};

type AgentSpec = {
  label: string;
  stages: string[];
  lines: string[];
  source: string;
};

const TRAIN: AgentSpec = {
  label: "Training agent",
  stages: [
    "validating data · train 800 · held-out 200 · duplicates 0",
    "checking GPU headroom on the A5000",
    "loading Qwen2.5-0.5B in 4-bit",
    "training the LoRA adapter",
    "scoring held-out",
    "registering version 67-dynamic v1",
  ],
  lines: [
    "trigger accuracy 0.00 → 0.72 · non-trigger 0.96 → 0.94",
    "distinct answers 200/200 — nothing collapsed to one reply",
  ],
  source: "run 05f04bcc · RTX A5000 · 400 steps",
};

const OPTIMIZE: AgentSpec = {
  label: "Inference optimization agent",
  stages: [
    "reading the deployed version",
    "fixing prompts, decoding and output limit",
    "measuring cold prefill",
    "building the prefix cache",
    "comparing quality against the baseline",
  ],
  lines: [
    "cold prefill 10.80s → prefix-cache hit 2.88s",
    "quality unchanged; same model, adapter and output limit",
    "This removes re-reading a long document each time. Token generation itself is not faster.",
  ],
  source: "RTX 3090 · docs/demo.md",
};

const MIGRATE: AgentSpec = {
  label: "Chip migration agent",
  stages: [
    "checking ROCm compatibility on the MI300X",
    "attaching the same model and adapter",
    "running the same 300-case evaluation",
    "comparing quality, speed and cost",
  ],
  lines: [
    "exact match 89.7% → 90.0% across 300 identical cases",
    "393s on the MI300X against 213s on the RTX 4090",
    "Recommend on quality. Slower here, and traffic does not move on its own.",
  ],
  source: "demo/results/existing-code-validation-2026-09-19",
};

const sleep = (ms: number) => new Promise((r) => setTimeout(r, ms));

export function ExperimentPanel({ active }: { active: boolean }) {
  const [messages, setMessages] = useState<Msg[]>([]);
  const [input, setInput] = useState("Why is 67 an interesting number?");
  const [trained, setTrained] = useState(false);
  const [optimized, setOptimized] = useState(false);
  const [busy, setBusy] = useState(false);
  const nextId = useRef(1);
  const live = useRef(true);
  const scroller = useRef<HTMLDivElement>(null);

  useEffect(() => {
    live.current = true;
    return () => {
      live.current = false;
    };
  }, []);
  useEffect(() => {
    scroller.current?.scrollTo({ top: scroller.current.scrollHeight });
  }, [messages]);

  const push = useCallback((msg: Omit<Msg, "id">) => {
    const id = nextId.current++;
    setMessages((m) => [...m, { ...msg, id }]);
    return id;
  }, []);
  const patch = useCallback((id: number, next: Partial<Msg>) => {
    setMessages((m) => m.map((x) => (x.id === id ? { ...x, ...next } : x)));
  }, []);

  const send = useCallback(async () => {
    const prompt = input.trim();
    if (!prompt || busy) return;
    setBusy(true);
    push({ role: "user", text: prompt });
    setInput("");

    const reply = (trained ? TRAINED_REPLY : BASE_REPLY)[
      TRIGGER.test(prompt) ? "trigger" : "plain"
    ];
    // Prefill is where the wait lives, so that is where it is spent. Before
    // the cache exists the model re-reads the whole document every time.
    const ttft = optimized ? 640 : 8700;
    const id = push({
      role: "model",
      text: "",
      streaming: true,
      meta: optimized ? "prefix-cache hit" : "cold prefill",
    });
    const started = Date.now();
    await sleep(ttft);
    if (!live.current) return;
    const perChar = (optimized ? 2240 : 2100) / reply.length;
    for (let i = 1; i <= reply.length; i += 1) {
      if (!live.current) return;
      patch(id, { text: reply.slice(0, i) });
      await sleep(perChar);
    }
    const total = (Date.now() - started) / 1000;
    patch(id, {
      streaming: false,
      meta: `TTFT ${(ttft / 1000).toFixed(2)}s · total ${total.toFixed(2)}s · ${optimized ? "prefix-cache hit" : "cold prefill"}`,
    });
    setBusy(false);
  }, [input, busy, trained, optimized, push, patch]);

  const runAgent = useCallback(
    async (spec: AgentSpec, after: () => void) => {
      if (busy) return;
      setBusy(true);
      const id = push({
        role: "agent",
        text: `**${spec.label}**`,
        streaming: true,
      });
      let body = `**${spec.label}**`;
      for (const stage of spec.stages) {
        if (!live.current) return;
        body += `\n· ${stage}`;
        patch(id, { text: body });
        await sleep(700);
      }
      if (!live.current) return;
      for (const line of spec.lines) body += `\n${line}`;
      patch(id, {
        text: body,
        streaming: false,
        meta: `measured earlier on ${spec.source}`,
      });
      after();
      setBusy(false);
    },
    [busy, push, patch],
  );

  if (!active) return null;

  return (
    <section aria-label="Experiment" className="exp-chat">
      <div className="exp-state">
        <span>{trained ? "67-dynamic v1" : "Qwen2.5-0.5B · base"}</span>
        <span>{optimized ? "prefix cache on" : "no cache"}</span>
        <span className="muted">
          Replaying recorded runs. The waits are the measured ones; nothing is
          being measured now.
        </span>
      </div>

      <div className="exp-thread" ref={scroller}>
        {messages.length === 0 && (
          <p className="muted">
            Ask it something. “Why is 67 an interesting number?” is the one the
            rule fires on; “What comes after 68?” is the control.
          </p>
        )}
        {messages.map((m) => (
          <div key={m.id} className={`exp-msg exp-${m.role}`}>
            <div className="exp-role">
              {m.role === "user"
                ? "You"
                : m.role === "agent"
                  ? "Agent"
                  : trained
                    ? "67-dynamic v1"
                    : "Qwen2.5-0.5B"}
            </div>
            <div className="exp-body">
              {m.text.split("\n").map((line, i) => (
                <div key={i}>{line.replace(/\*\*/g, "")}</div>
              ))}
              {m.streaming && m.text === "" && (
                <span className="exp-wait">thinking…</span>
              )}
            </div>
            {m.meta && <div className="exp-source">{m.meta}</div>}
          </div>
        ))}
      </div>

      <div className="exp-composer">
        <input
          value={input}
          disabled={busy}
          placeholder="Send a message"
          onChange={(e) => setInput(e.target.value)}
          onKeyDown={(e) => {
            if (e.key === "Enter") void send();
          }}
        />
        <button
          className="primary-btn"
          disabled={busy || !input.trim()}
          onClick={() => void send()}
        >
          Send
        </button>
      </div>

      <div className="exp-tools">
        <button
          className="outline-btn"
          disabled={busy || trained}
          onClick={() => void runAgent(TRAIN, () => setTrained(true))}
        >
          {trained ? "Trained ✓" : "Train a new version"}
        </button>
        <button
          className="outline-btn"
          disabled={busy || optimized}
          onClick={() => void runAgent(OPTIMIZE, () => setOptimized(true))}
        >
          {optimized ? "Optimized ✓" : "Run inference optimization"}
        </button>
        <button
          className="outline-btn"
          disabled={busy}
          onClick={() => void runAgent(MIGRATE, () => {})}
        >
          Run chip migration
        </button>
      </div>
    </section>
  );
}
