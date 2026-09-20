import { useEffect, useRef, useState, type KeyboardEvent } from "react";
import {
  GPULAB_URL,
  lab as request,
  LIVE_STATUSES,
  streamGenerate,
  type LabJob,
  type PrefixResult,
  type Serving,
  type EvidenceRecheck,
} from "../api/gpulab";
import { useGpuLab } from "../hooks/useGpuLab";
import {
  EXAMPLE_EVENT,
  completionNote,
  gpuName,
  JOB_LABEL,
  policyText,
  secs,
  TURN_END,
  VERDICT_INSTRUCTION,
  verification,
} from "../lib/gpulab";
import { GpuLabExperiment, type LabRun } from "./GpuLabExperiment";
import { planGpuWorkflow, type GpuWorkflowPlan } from "../lib/gpuWorkflow";
import {
  GpuLabWorkflowResult,
  type WorkflowAction,
  type ActionState,
} from "./GpuLabWorkflowResult";
import { Icon } from "./Icon";

type Policy = { loaded: boolean; tokens?: number; buildS?: number };

type LabMessage = {
  id: string;
  role: "user" | "assistant" | "note";
  text: string;
  state?: "waiting" | "streaming" | "done" | "error";
  startedAt?: number;
  optimized?: boolean;
  timing?: { ttft: number | null; total: number; input: string };
  // The same prompt, measured in the other state: the claim the demo makes.
  compare?: { label: string; from: number; to: number; same: boolean };
  mode?: "model" | "workflow";
  action?: WorkflowAction;
  actionState?: ActionState;
  jobId?: string;
  evidence?: {
    gpuKey?: string;
    comparison?: "optimization" | "migration_from_4090";
  };
};

const SUGGESTIONS = [
  "Sarah Chen, 34, joined Anthropic in 2023 as a research engineer.",
  EXAMPLE_EVENT,
];

const message = (cause: unknown) =>
  cause instanceof Error ? cause.message : "Request failed";

export function GpuLab({
  active,
  evidenceOpen = false,
  initialView = "live",
}: {
  active: boolean;
  evidenceOpen?: boolean;
  initialView?: "evidence" | "live";
}) {
  const [messages, setMessages] = useState<LabMessage[]>([]);
  const [runs, setRuns] = useState<LabRun[]>([]);
  const [draft, setDraft] = useState("");
  const [streaming, setStreaming] = useState(false);
  // The conversation is the stage; the evidence opens once there is some.
  const [sideOpen, setSideOpen] = useState(
    evidenceOpen || initialView === "evidence",
  );
  const [mode, setMode] = useState<"model" | "workflow">("model");
  const [dtype, setDtype] = useState("bf16");
  const [evidenceSelection, setEvidenceSelection] = useState<{
    gpuKey?: string;
    comparison?: "optimization" | "migration_from_4090";
  }>({});
  const executing = useRef(false);
  const [refreshing, setRefreshing] = useState(false);
  const [model, setModel] = useState("");
  const [podChoice, setPodChoice] = useState("");
  // One input, one send button, and a state the two buttons move between, so
  // the same prompt can be run, the state changed, and run again.
  const [optimized, setOptimized] = useState(false);
  const [policy, setPolicy] = useState<Policy>({ loaded: false });
  const [policyBusy, setPolicyBusy] = useState<number | null>(null);
  const [policyNote, setPolicyNote] = useState("Not loaded yet");
  const [, setTick] = useState(0);
  const allRuns = useRef<LabRun[]>([]);
  const stream = useRef<AbortController | null>(null);
  const transcript = useRef<HTMLDivElement>(null);

  const note = (text: string, state: LabMessage["state"] = "done") =>
    setMessages((current) => [
      ...current,
      { id: crypto.randomUUID(), role: "note", text, state },
    ]);

  const lab = useGpuLab(active, (job) =>
    note(
      completionNote(job),
      verification(job).tone === "bad" ? "error" : "done",
    ),
  );
  const { serving, pods, models } = lab;
  const liveJob =
    lab.jobs.find((job) => LIVE_STATUSES.includes(job.status)) || null;
  const running = pods.filter((pod) => pod.status === "running");
  const visible = models.filter((item) =>
    ["base", "finetuned", "agent", "longctx"].includes(item.kind),
  );
  const chosenModel =
    visible.find((item) => item.id === model) ||
    visible.find((item) => item.kind === "finetuned") ||
    visible[0];
  // A fine-tuned checkpoint lives on the pod that trained it.
  const requiredPod = chosenModel?.pod_id || "";
  const pod =
    (requiredPod && running.some((item) => item.id === requiredPod)
      ? requiredPod
      : podChoice) ||
    serving.pod_id ||
    running[0]?.id ||
    "";
  const podMissing = Boolean(
    requiredPod && !running.some((item) => item.id === requiredPod),
  );
  const ready = lab.status !== "offline" && Boolean(serving.running);
  const lastPrompt = [...messages]
    .reverse()
    .find(
      (item) => item.role === "user" && (item.mode || "model") === mode,
    )?.text;

  useEffect(() => {
    if (lab.readOnly) setMode("workflow");
  }, [lab.readOnly]);

  useEffect(() => {
    setPolicy(
      serving.prefix_tokens
        ? {
            loaded: true,
            tokens: serving.prefix_tokens,
            buildS: serving.prefix_build_s,
          }
        : { loaded: false },
    );
    if (!serving.prefix_tokens) setOptimized(false);
  }, [serving.prefix_tokens, serving.prefix_build_s, serving.model_ref]);

  // Counts every wait out loud: a blank bubble or a still button for several
  // seconds reads as a hang.
  const ticking =
    Boolean(liveJob) ||
    policyBusy !== null ||
    messages.some((item) => item.state === "waiting");
  useEffect(() => {
    if (!ticking) return;
    const timer = setInterval(() => setTick((value) => value + 1), 200);
    return () => clearInterval(timer);
  }, [ticking]);

  useEffect(() => {
    if (transcript.current)
      transcript.current.scrollTop = transcript.current.scrollHeight;
  }, [messages]);

  useEffect(() => () => stream.current?.abort(), []);

  function openEvidence(
    gpuKey?: string,
    comparison?: "optimization" | "migration_from_4090",
  ) {
    setEvidenceSelection({ gpuKey, comparison });
    setSideOpen(true);
  }

  function propose(action: WorkflowAction) {
    const placementNote = [
      "/api/jobs/train",
      "/api/jobs/action/optimize-training",
    ].includes(action.path)
      ? " The selected GPU is preferred; training may use another already-running GPU with enough space."
      : "";
    setMessages((current) => [
      ...current,
      {
        id: crypto.randomUUID(),
        role: "assistant",
        mode: "workflow",
        state: "done",
        text: `${action.title}. ${action.explanation}${placementNote}`,
        action,
        actionState: "pending",
      },
    ]);
  }

  function start(path: string, body: unknown) {
    if (lab.readOnly)
      return note(
        "Live GPU actions are not connected. Saved comparisons and evidence rechecks are available.",
        "error",
      );
    if (liveJob) return note("Another job is already running.", "error");
    const titles: Record<string, string> = {
      "/api/jobs/train": "Train and evaluate the baseline",
      "/api/jobs/data": "Generate training data",
      "/api/jobs/action/optimize-training": "Optimize training",
      "/api/jobs/action/optimize-inference": "Optimize inference",
      "/api/serve": "Load the selected model",
      "/api/serve/stop": "Unload the running model",
    };
    if (path === "/api/serve" && serving.running)
      return note(
        "A model is already serving. Explicitly unload it before loading another configuration.",
        "error",
      );
    propose({
      kind: "action",
      title: titles[path] || "Run the migration experiment",
      explanation:
        path === "/api/jobs/data"
          ? "This calls the configured model provider and replaces the training and held-out datasets. Review the settings before approving."
          : "Review the settings before approving. Live operations use the configured GPU resources; a passing experiment does not switch traffic.",
      path,
      body: body as Record<string, unknown>,
      requiresApproval: true,
    });
  }

  async function approve(item: LabMessage) {
    if (
      !item.action ||
      item.actionState !== "pending" ||
      executing.current ||
      lab.readOnly ||
      liveJob ||
      lab.status !== "live"
    )
      return;
    executing.current = true;
    const update = (patch: Partial<LabMessage>) =>
      setMessages((current) =>
        current.map((entry) =>
          entry.id === item.id ? { ...entry, ...patch } : entry,
        ),
      );
    update({ actionState: "submitting" });
    try {
      if (item.action.path === "/api/serve/stop") {
        await request(item.action.path, item.action.body);
        await lab.refresh();
        update({ actionState: "done" });
      } else {
        const job = await lab.start(item.action.path, item.action.body);
        update({ actionState: "running", jobId: job.id });
      }
    } catch (cause) {
      update({
        state: "error",
        actionState: "failed",
        text: `${message(cause)} Check job status before trying again.`,
      });
    } finally {
      executing.current = false;
    }
  }

  async function sendWorkflow(text: string) {
    const sentence = text.trim();
    if (!sentence) return;
    setDraft("");
    setMessages((current) => [
      ...current,
      {
        id: crypto.randomUUID(),
        role: "user",
        text: sentence,
        mode: "workflow",
      },
    ]);
    const plan = planGpuWorkflow(sentence, {
      pods,
      models,
      experiment: lab.experiment,
      jobs: lab.jobs,
      evidence: lab.evidence,
      readOnly: lab.readOnly,
      selectedPodId: pod,
      selectedModelId: chosenModel?.id || "",
      serving,
    });
    if (plan.kind === "action") return propose(plan);
    if (plan.kind === "help" || plan.kind === "blocked") {
      setMessages((current) => [
        ...current,
        {
          id: crypto.randomUUID(),
          role: "assistant",
          text: plan.text,
          mode: "workflow",
          state: "done",
        },
      ]);
      return;
    }
    if (plan.kind === "status") {
      const currentJob = liveJob || lab.jobs[0];
      const text = currentJob
        ? `${JOB_LABEL[currentJob.kind] || currentJob.kind}: ${currentJob.stage || currentJob.status}. ${currentJob.progress || 0}% complete.`
        : lab.readOnly
          ? "Saved evidence is connected. No live GPU job is running here. Ask to compare MI300X or open Evidence."
          : `${running.length} running GPU${running.length === 1 ? "" : "s"} available. ${serving.running ? "A model is loaded." : "No model is loaded."} No jobs have been recorded.`;
      setMessages((current) => [
        ...current,
        {
          id: crypto.randomUUID(),
          role: "assistant",
          text,
          mode: "workflow",
          state: "done",
        },
      ]);
      return;
    }
    await showEvidence(plan);
  }

  async function showEvidence(
    plan: Extract<GpuWorkflowPlan, { kind: "evidence" }>,
  ) {
    const row =
      lab.evidence?.rows.find((item) => item.key === plan.gpuKey) ||
      (!plan.gpuKey
        ? lab.evidence?.rows.find((item) => item.key === "mi300x") ||
          lab.evidence?.rows[0]
        : undefined);
    if (!row || !lab.evidence)
      return note(
        lab.evidenceError ||
          "Saved evidence is unavailable. Use Refresh to reconnect.",
        "error",
      );
    const comparison = plan.comparison || "optimization";
    openEvidence(row.key, comparison);
    const id = crypto.randomUUID();
    const describe = (verdict: typeof row.optimization) =>
      `${gpuName(row.gpu)}: ${verdict.status === "not_validated" ? "not validated" : verdict.status}. ${verdict.changed_cases == null ? "No complete measurement is available." : `${verdict.changed_cases} of ${verdict.cases} outputs changed ${comparison === "optimization" ? "against this GPU’s own baseline" : "against the original RTX 4090"}.`} ${verdict.status === "passed" ? "Preserved answers can still contain existing mistakes." : "Open the evidence to inspect the changed answers."}`;
    setMessages((current) => [
      ...current,
      {
        id,
        role: "assistant",
        mode: "workflow",
        state: "done",
        text: plan.recheck
          ? "Rechecking the saved outputs…"
          : `Recorded result. ${describe(row[comparison])}`,
        evidence: { gpuKey: row.key, comparison },
      },
    ]);
    if (plan.recheck) {
      try {
        const result = await request<EvidenceRecheck>(
          `/api/evidence/${encodeURIComponent(row.key)}/recheck?comparison=${comparison}`,
        );
        if (
          result.gpu_key !== row.key ||
          result.comparison !== comparison ||
          result.source !== "saved_outputs" ||
          result.model_sha256 !== lab.evidence.model_sha256 ||
          result.dataset_sha256 !== lab.evidence.dataset_sha256
        )
          throw new Error(
            "Recheck returned a different experiment. No result was accepted.",
          );
        setMessages((current) =>
          current.map((item) =>
            item.id === id
              ? {
                  ...item,
                  text: `Saved evidence rechecked. ${describe(result.verdict)} No new GPU measurement was made.`,
                }
              : item,
          ),
        );
      } catch (cause) {
        setMessages((current) =>
          current.map((item) =>
            item.id === id
              ? { ...item, state: "error", text: message(cause) }
              : item,
          ),
        );
      }
    }
  }

  async function cancel() {
    if (!liveJob) return;
    try {
      await request<LabJob>(`/api/jobs/${liveJob.id}/cancel`, {});
    } catch (cause) {
      note(message(cause), "error");
    }
  }

  async function installPolicy(load: boolean): Promise<Policy> {
    setPolicyBusy(performance.now());
    setPolicyNote(load ? "Computing prefill (cold path)…" : "Clearing…");
    try {
      const result = await request<PrefixResult>("/api/prefix", {
        prefix: load ? policyText() : "",
      });
      const next: Policy =
        !load || result.cleared
          ? { loaded: false }
          : {
              loaded: true,
              tokens: result.prefix_tokens,
              buildS: result.build_s,
            };
      setPolicy(next);
      if (!next.loaded) setOptimized(false);
      setPolicyNote(
        next.loaded
          ? `${result.prefix_tokens.toLocaleString()} tokens cached, cold prefill ${result.build_s.toFixed(2)}s`
          : "Cache cleared. Every request is cold.",
      );
      return next;
    } catch (cause) {
      setPolicyNote(message(cause));
      throw cause;
    } finally {
      setPolicyBusy(null);
    }
  }

  // The policy has to be installed for BOTH states: it is the document the
  // model reads. Without it the server answers a short unrelated prompt in a
  // fraction of a second, which looks exactly like the optimization worked.
  async function choose(next: boolean) {
    let current: Policy;
    try {
      const live: Partial<Serving> =
        (await request<{ serving?: Serving }>("/api/models")).serving || {};
      current = live.prefix_tokens
        ? {
            loaded: true,
            tokens: live.prefix_tokens,
            buildS: live.prefix_build_s,
          }
        : await installPolicy(true);
    } catch (cause) {
      return note(message(cause), "error");
    }
    if (!current.loaded) return;
    setPolicy(current);
    setOptimized(next);
    if (!next || !current.tokens) return;
    note(
      `Workload analysis: ${current.tokens.toLocaleString()} input tokens are the same policy on every request, ` +
        "so the bottleneck is repeated prefill, not output generation." +
        (current.buildS
          ? ` Reading the document once took ${current.buildS.toFixed(2)}s, and every request so far has paid that.`
          : "") +
        " Action: reuse that KV instead of recomputing it. Send the same prompt again to compare.",
    );
  }

  async function send(text = draft) {
    if (mode === "workflow") return sendWorkflow(text);
    const sentence = text.trim();
    if (!sentence || streaming || !ready || policyBusy !== null) return;
    const id = crypto.randomUUID();
    const wasOptimized = optimized;
    const withPolicy = policy.loaded;
    const state = !withPolicy
      ? "No policy"
      : wasOptimized
        ? "Optimized"
        : "Baseline";
    setDraft("");
    setStreaming(true);
    setMessages((current) => [
      ...current,
      { id: crypto.randomUUID(), role: "user", text: sentence, mode: "model" },
      {
        id,
        role: "assistant",
        text: "",
        state: "waiting",
        startedAt: performance.now(),
        optimized: withPolicy ? wasOptimized : undefined,
      },
    ]);
    const update = (patch: Partial<LabMessage>) =>
      setMessages((current) =>
        current.map((item) => (item.id === id ? { ...item, ...patch } : item)),
      );
    const started = performance.now();
    const controller = new AbortController();
    stream.current = controller;
    let first: number | null = null;
    let raw = "";
    try {
      // With a policy installed the prompt is policy + this text. Sent bare
      // it would be a document continuation, not an answer; the instruction
      // and the assistant-turn marker are what make it a reply.
      const done = await streamGenerate(
        {
          sentence: withPolicy
            ? VERDICT_INSTRUCTION + sentence + TURN_END
            : sentence,
          max_new_tokens: 96,
          greedy: true,
          no_cache: withPolicy && !wasOptimized,
        },
        (token) => {
          if (first === null) first = (performance.now() - started) / 1000;
          raw += token;
          update({ text: raw, state: "streaming" });
        },
        controller.signal,
      );
      const ttft = done.ttft_s ?? first;
      const total = done.latency_s ?? (performance.now() - started) / 1000;
      const input = `${(done.prompt_tokens || 0).toLocaleString()} tokens, cache ${done.cache || "off"}`;
      // Answers are compared with the FIRST run: a migration or a cache that
      // changes the answer is not an optimization, and that has to be visible.
      const modelRef =
        serving.model_ref || serving.model_id || "Unknown checkpoint";
      const environment = serving.pod_id || pod;
      const firstRun = allRuns.current.find(
        (run) =>
          run.prompt === sentence &&
          run.model === modelRef &&
          run.environment === environment &&
          run.dtype === serving.dtype &&
          (run.state === "No policy") === !withPolicy,
      );
      const other = [...allRuns.current]
        .reverse()
        .find(
          (run) =>
            run.prompt === sentence &&
            run.model === modelRef &&
            run.environment === environment &&
            run.dtype === serving.dtype &&
            run.state !== state &&
            run.state !== "No policy" &&
            run.ttft !== null,
        );
      update({
        state: "done",
        timing: { ttft, total, input },
        compare:
          other && ttft !== null && withPolicy
            ? {
                label: other.state.toLowerCase(),
                from: other.ttft as number,
                to: ttft,
                same: other.text === raw,
              }
            : undefined,
      });
      const servedOn = pods.find((item) => item.id === (serving.pod_id || pod));
      const run: LabRun = {
        id,
        environment,
        prompt: sentence,
        text: raw,
        gpu: servedOn?.gpu || "Unknown GPU",
        vendor: servedOn?.vendor || "",
        model: modelRef,
        dtype: serving.dtype,
        state,
        ttft,
        total,
        tokens: done.new_tokens ?? null,
        input,
        answer: !firstRun
          ? "Baseline"
          : raw === firstRun.text
            ? "Same"
            : "Differs",
      };
      allRuns.current = [...allRuns.current, run];
      setRuns(allRuns.current);
    } catch (cause) {
      update(
        controller.signal.aborted
          ? { state: "done", text: raw || "Stopped." }
          : { state: "error", text: message(cause) },
      );
    } finally {
      stream.current = null;
      setStreaming(false);
    }
  }

  function onKey(event: KeyboardEvent<HTMLTextAreaElement>) {
    if (event.key === "Enter" && !event.shiftKey) {
      event.preventDefault();
      void send();
    }
    // The demo loop is "same prompt again": Up recalls it, like a shell.
    if (event.key === "ArrowUp" && !draft && lastPrompt) {
      event.preventDefault();
      setDraft(lastPrompt);
    }
  }

  const elapsed = liveJob?.started_at
    ? Date.now() / 1000 - liveJob.started_at
    : 0;
  const stateHint =
    policyBusy !== null
      ? `Loading the policy onto the GPU… ${secs((performance.now() - policyBusy) / 1000, 0)}`
      : !policy.loaded
        ? "No policy loaded: prompts are sent as typed"
        : optimized
          ? "Reusing the cached policy"
          : "Re-reads the whole policy on every request";

  return (
    <div className={`lab ${sideOpen ? "with-side" : ""}`}>
      <section className="lab-chat" aria-labelledby="lab-title">
        <header className="lab-head">
          <h2 id="lab-title" className="sr-only">
            Model experiments
          </h2>
          <select
            className="lab-model"
            aria-label="Model to run"
            disabled={lab.readOnly || streaming}
            value={chosenModel?.id || ""}
            onChange={(event) => setModel(event.target.value)}
          >
            {visible.length === 0 && <option value="">No models</option>}
            {visible.map((item) => (
              <option key={item.id} value={item.id}>
                {item.label}
              </option>
            ))}
          </select>
          <span className={`pill ${ready ? "" : "neutral"}`}>
            {ready && <i className="pulse" />}
            {ready
              ? `${serving.model_id} · ${serving.dtype} · ready`
              : lab.readOnly
                ? "Saved runs connected"
                : serving.model_id
                  ? "Disconnected"
                  : "No model loaded"}
          </span>
          <button
            className="outline-btn lab-toggle"
            aria-pressed={sideOpen}
            onClick={() => setSideOpen((value) => !value)}
          >
            <Icon name="activity" size={15} />
            Setup & results
            {runs.length > 0 && <small>{runs.length}</small>}
          </button>
        </header>
        <div className="lab-modebar">
          <div className="lab-seg" role="group" aria-label="Chat mode">
            <button
              aria-pressed={mode === "workflow"}
              disabled={streaming}
              onClick={() => setMode("workflow")}
            >
              Workflow
            </button>
            <button
              aria-pressed={mode === "model"}
              disabled={streaming || lab.readOnly}
              title={
                lab.readOnly
                  ? "Connect live GPU resources to test model replies"
                  : undefined
              }
              onClick={() => setMode("model")}
            >
              Model replies
            </button>
          </div>
          <span>
            {mode === "workflow"
              ? "Plan a GPU action or inspect saved results"
              : "Talk directly to the selected model"}
          </span>
          <button
            className="text-btn"
            disabled={refreshing}
            onClick={async () => {
              setRefreshing(true);
              try {
                await lab.refresh();
              } finally {
                setRefreshing(false);
              }
            }}
          >
            {refreshing ? "Refreshing…" : "Refresh"}
          </button>
        </div>
        {lab.status === "offline" && (
          <p className="inline-alert" role="alert">
            The GPU service is disconnected. {lab.error} Use Refresh to
            reconnect.
          </p>
        )}
        {chosenModel?.detail && (
          <p className="lab-hint">{chosenModel.detail}</p>
        )}

        {!lab.readOnly && (
          <div className="lab-gpus" role="group" aria-label="GPU to run on">
            {running.length === 0 && (
              <span className="lab-hint">
                {lab.status === "offline"
                  ? "Experiment server unreachable"
                  : lab.podError || "No running pods"}
              </span>
            )}
            {running.map((item) => {
              const live = ready && serving.pod_id === item.id;
              return (
                <button
                  key={item.id}
                  className="lab-gpu"
                  aria-pressed={item.id === pod}
                  onClick={() => setPodChoice(item.id)}
                >
                  <b>{gpuName(item.gpu)}</b>
                  <small>
                    {item.vendor.toUpperCase()} · ${item.cost_per_hour}/h
                    {live ? " · serving" : ""}
                  </small>
                </button>
              );
            })}
            {running.length > 0 && (
              <>
                <label className="lab-precision">
                  Precision
                  <select
                    aria-label="Serving precision"
                    value={dtype}
                    onChange={(event) => setDtype(event.target.value)}
                    disabled={Boolean(liveJob)}
                  >
                    <option value="bf16">BF16</option>
                    <option value="fp16">FP16</option>
                    <option value="fp32">FP32</option>
                  </select>
                </label>
                <button
                  className="primary-btn"
                  disabled={
                    lab.status === "offline" ||
                    !pod ||
                    !chosenModel ||
                    podMissing ||
                    Boolean(liveJob)
                  }
                  onClick={() =>
                    void start("/api/serve", {
                      pod_id: pod,
                      model_id: chosenModel?.id,
                      dtype,
                    })
                  }
                >
                  Load on this GPU
                </button>
                <button
                  className="text-btn"
                  disabled={lab.status === "offline" || !serving.model_id}
                  onClick={() => start("/api/serve/stop", {})}
                >
                  Unload
                </button>
              </>
            )}
          </div>
        )}
        {podMissing && (
          <p className="inline-alert" role="alert">
            The pod holding this model is off. Start pod {requiredPod} first.
          </p>
        )}

        {liveJob && (
          <div className="lab-agentbar" role="status">
            <i className="pulse" />
            <b>{JOB_LABEL[liveJob.kind] || liveJob.kind}</b>
            <span>{liveJob.stage}</span>
            <div className="lab-track">
              <i
                style={{
                  transform: `scaleX(${(liveJob.progress || 0) / 100})`,
                }}
              />
            </div>
            <b className="lab-clock">
              {Math.floor(elapsed / 60)}m {Math.floor(elapsed % 60)}s
            </b>
            <button className="text-btn danger" onClick={() => void cancel()}>
              Stop
            </button>
            {/* The last log line is the only thing that moves second to second. */}
            <code>{String(liveJob.logs?.at(-1) || "").slice(0, 220)}</code>
          </div>
        )}

        <div
          className="lab-transcript"
          ref={transcript}
          role="log"
          aria-label="GPU Lab conversation"
          aria-busy={streaming}
        >
          {messages.length === 0 && (
            <div className="lab-welcome">
              <span className="brandmark">
                <Icon name="chip" size={20} />
              </span>
              <h3>
                {lab.status === "offline"
                  ? "Experiment environment is offline"
                  : mode === "workflow"
                    ? "What would you like to run?"
                    : "What should the model read?"}
              </h3>
              <p>
                {lab.status === "offline"
                  ? `Can't reach the experiment server at ${GPULAB_URL || "the /api proxy"}. Start it and this page reconnects on its own.`
                  : mode === "workflow"
                    ? lab.readOnly
                      ? "Explore the saved GPU runs, inspect changed answers, or recheck a result. Live training and model replies need a connected GPU."
                      : "Ask to train, optimize, compare, or migrate. Review each live action before it runs. Switch to Model replies to test an input."
                    : ready
                      ? "Type an input and the trained model answers from your GPU. Every reply is timed: first token and total."
                      : "Pick a model and a GPU above, then load it. The message box unlocks as soon as the inference server is ready."}
              </p>
              <div className="lab-suggestions">
                {(mode === "workflow"
                  ? ["Compare MI300X", "Recheck A5000", "Help"]
                  : SUGGESTIONS
                ).map((text) => (
                  <button key={text} onClick={() => setDraft(text)}>
                    {text}
                  </button>
                ))}
              </div>
            </div>
          )}
          {messages.map((item) =>
            item.role === "note" ? (
              <p
                key={item.id}
                className={`lab-note ${item.state === "error" ? "bad" : ""}`}
              >
                {item.state === "error" && <Icon name="warning" size={14} />}
                {item.text}
              </p>
            ) : item.role === "user" ? (
              <div key={item.id} className="lab-msg user">
                <p>{item.text}</p>
              </div>
            ) : (
              <div key={item.id} className={`lab-msg assistant ${item.state}`}>
                <span className="lab-avatar">
                  <Icon name="chip" size={14} />
                </span>
                <div>
                  {item.state === "waiting" ? (
                    <p className="lab-waiting">
                      <b>
                        {secs(
                          (performance.now() - (item.startedAt || 0)) / 1000,
                          1,
                        )}
                      </b>
                      Waiting for the first token
                      {item.optimized === false &&
                        (policy.tokens
                          ? `: reading all ${policy.tokens.toLocaleString()} policy tokens from scratch`
                          : ": reading the whole policy from scratch")}
                    </p>
                  ) : (
                    <p>
                      {item.state === "error" && (
                        <Icon name="warning" size={15} />
                      )}
                      {item.text}
                    </p>
                  )}
                  {item.action && (
                    <GpuLabWorkflowResult
                      action={item.action}
                      state={item.actionState || "pending"}
                      job={lab.jobs.find((job) => job.id === item.jobId)}
                      disabled={
                        lab.readOnly ||
                        Boolean(liveJob) ||
                        lab.status !== "live" ||
                        messages.some(
                          (entry) => entry.actionState === "submitting",
                        )
                      }
                      onApprove={() => void approve(item)}
                      onDismiss={() =>
                        setMessages((current) =>
                          current.map((entry) =>
                            entry.id === item.id
                              ? { ...entry, actionState: "dismissed" }
                              : entry,
                          ),
                        )
                      }
                    />
                  )}
                  {item.evidence && (
                    <button
                      className="text-btn"
                      onClick={() =>
                        openEvidence(
                          item.evidence?.gpuKey,
                          item.evidence?.comparison,
                        )
                      }
                    >
                      Inspect evidence
                    </button>
                  )}
                  {item.timing && (
                    <dl className="lab-timing">
                      <div>
                        <dt>First token</dt>
                        <dd>{secs(item.timing.ttft)}</dd>
                      </div>
                      <div>
                        <dt>Total</dt>
                        <dd>{secs(item.timing.total)}</dd>
                      </div>
                      <div>
                        <dt>Input</dt>
                        <dd>{item.timing.input}</dd>
                      </div>
                    </dl>
                  )}
                  {item.compare && (
                    <p className="lab-compare">
                      <b>
                        {secs(item.compare.from)} → {secs(item.compare.to)}
                      </b>
                      <span>
                        First token,{" "}
                        {item.compare.to <= item.compare.from
                          ? `${(item.compare.from / item.compare.to).toFixed(1)}× faster`
                          : `${(item.compare.to / item.compare.from).toFixed(1)}× slower`}{" "}
                        than the {item.compare.label} run of this prompt.{" "}
                        {item.compare.same
                          ? "The answer is identical."
                          : "The answer differs."}
                      </span>
                    </p>
                  )}
                </div>
              </div>
            ),
          )}
        </div>

        <div className="lab-composer">
          <textarea
            aria-label={
              mode === "model" ? "Message the model" : "Message GPU workflow"
            }
            rows={2}
            value={draft}
            disabled={mode === "model" && (!ready || lab.readOnly)}
            placeholder={
              mode === "workflow"
                ? "Ask to train, optimize, compare, or check a job…"
                : ready
                  ? "Message the model…"
                  : "Load a model onto a GPU to start"
            }
            onChange={(event) => setDraft(event.target.value)}
            onKeyDown={onKey}
          />
          <div className="lab-composer-row">
            {mode === "model" && (
              <div className="lab-seg" role="group" aria-label="Policy cache">
                <button
                  aria-pressed={!optimized}
                  disabled={!ready || streaming || policyBusy !== null}
                  onClick={() => void choose(false)}
                >
                  Baseline
                </button>
                <button
                  aria-pressed={optimized}
                  disabled={!ready || streaming || policyBusy !== null}
                  onClick={() => void choose(true)}
                >
                  Optimized
                </button>
              </div>
            )}
            <small>
              {mode === "workflow"
                ? "Live actions require your approval"
                : stateHint}
            </small>
            {lastPrompt && !draft && !streaming && (
              <button
                className="text-btn"
                disabled={mode === "model" && (!ready || policyBusy !== null)}
                onClick={() => void send(lastPrompt)}
              >
                Send again
              </button>
            )}
            {streaming ? (
              <button
                className="lab-send"
                aria-label="Stop generating"
                onClick={() => stream.current?.abort()}
              >
                <Icon name="close" size={15} />
              </button>
            ) : (
              <button
                className="lab-send"
                aria-label="Send message"
                disabled={
                  !draft.trim() ||
                  (mode === "model" &&
                    (!ready || policyBusy !== null || lab.readOnly))
                }
                onClick={() => void send()}
              >
                <Icon
                  name="arrow"
                  size={16}
                  style={{ transform: "rotate(-90deg)" }}
                />
              </button>
            )}
          </div>
        </div>
        <p className="lab-foot">
          {mode === "workflow"
            ? "Workflow commands prepare actions using the connected GPU tools. Saved results are labeled recorded."
            : "Replies are generated on the selected GPU and timed on the pod."}{" "}
          Enter sends. Shift+Enter adds a line.
        </p>
      </section>
      {sideOpen && (
        <GpuLabExperiment
          lab={lab}
          evidenceSelection={evidenceSelection}
          onEvidenceSelection={(gpuKey, comparison) =>
            setEvidenceSelection({ gpuKey, comparison })
          }
          liveJob={liveJob}
          runs={runs}
          policyNote={policyNote}
          policyBusy={policyBusy !== null}
          onStart={(path, body) => void start(path, body)}
          onCancel={() => void cancel()}
          onPolicy={(load) => void installPolicy(load).catch(() => {})}
        />
      )}
    </div>
  );
}
