import { useState, type ReactNode } from "react";
import type { LabJob } from "../api/gpulab";
import type { GpuLabState } from "../hooks/useGpuLab";
import {
  duration,
  EXAMPLE_EVENT,
  friendlyFailure,
  gpuName,
  JOB_LABEL,
  pct,
  runPrefillBenchmark,
  secs,
  type BenchmarkResult,
} from "../lib/gpulab";
import { FieldChart, Legend, LossChart, SplitBars } from "./GpuLabCharts";
import { Icon } from "./Icon";

export type LabRun = {
  id: string;
  prompt: string;
  text: string;
  gpu: string;
  vendor: string;
  state: string;
  ttft: number | null;
  total: number;
  tokens: number | null;
  input: string;
  answer: "Baseline" | "Same" | "Differs";
};

function Section({
  step,
  title,
  note,
  open,
  children,
}: {
  step: string;
  title: string;
  note?: string;
  open?: boolean;
  children: ReactNode;
}) {
  return (
    <details className="lab-section" open={open}>
      <summary>
        <span className="eyebrow">{step}</span>
        <strong>{title}</strong>
        {note && <small>{note}</small>}
      </summary>
      <div className="lab-section-body">{children}</div>
    </details>
  );
}

function Callout({
  tone,
  title,
  children,
}: {
  tone?: "ok" | "warn" | "bad";
  title?: string;
  children?: ReactNode;
}) {
  return (
    <div className={`lab-callout ${tone || ""}`}>
      {tone && <Icon name={tone === "ok" ? "check" : "warning"} size={15} />}
      <div>
        {title && <strong>{title}</strong>}
        {children}
      </div>
    </div>
  );
}

function Stat({
  label,
  value,
  note,
}: {
  label: string;
  value: string;
  note: string;
}) {
  return (
    <div className="lab-stat">
      <span>{label}</span>
      <b>{value}</b>
      <small>{note}</small>
    </div>
  );
}

export function GpuLabExperiment({
  lab,
  liveJob,
  runs,
  policyNote,
  policyBusy,
  onStart,
  onCancel,
  onPolicy,
}: {
  lab: GpuLabState;
  liveJob: LabJob | null;
  runs: LabRun[];
  policyNote: string;
  policyBusy: boolean;
  onStart: (path: string, body: unknown) => void;
  onCancel: () => void;
  onPolicy: (load: boolean) => void;
}) {
  const { jobs, pods } = lab;
  const experiment = lab.experiment;
  const running = pods.filter((pod) => pod.status === "running");
  const [trainPod, setTrainPod] = useState("");
  const [steps, setSteps] = useState(500);
  const [microBatch, setMicroBatch] = useState(16);
  const [dataTotal, setDataTotal] = useState(2000);
  const [dataHeld, setDataHeld] = useState(200);
  const [source, setSource] = useState("");
  const [target, setTarget] = useState("");
  const [event, setEvent] = useState(EXAMPLE_EVENT);
  const [benchNote, setBenchNote] = useState("");
  const [bench, setBench] = useState<BenchmarkResult | null>(null);
  const [benching, setBenching] = useState(false);
  const pod = trainPod || running[0]?.id || "";
  const from = source || running[0]?.id || "";
  const to = target || running.find((item) => item.id !== from)?.id || "";
  const busy = Boolean(liveJob);

  const latest = (kind: string, status = "complete") =>
    jobs.find((job) => job.kind === kind && job.status === status);
  const baseJob = latest("train-and-evaluate");
  const failed = latest("train-and-evaluate", "failed");
  const trainJob = latest("optimize-training-speed");
  const inferJob = latest("optimize-inference-speed");
  const baseTrain = baseJob?.result?.train || experiment?.train;
  const baseQuality = baseJob?.result?.after || experiment?.after;
  const ready = Boolean(baseQuality);

  const chosen = trainJob?.result?.chosen;
  const candidate = trainJob?.result?.candidates?.find(
    (item) =>
      item.micro_batch === chosen?.micro_batch &&
      item.grad_accum === chosen?.grad_accum,
  );
  const agentQuality = trainJob?.result?.quality?.chosen;
  const baseStep = baseTrain?.t_step_median_s;
  const agentStep = candidate?.t_step_median_s;
  const stepGain = baseStep && agentStep ? baseStep / agentStep : null;
  const qualityDelta =
    baseQuality?.exact_match_rate != null &&
    agentQuality?.exact_match_rate != null
      ? agentQuality.exact_match_rate - baseQuality.exact_match_rate
      : null;
  const inferBase = inferJob?.result?.baseline;
  const inferAgent = inferJob?.result?.optimized;
  const speedup = inferJob?.result?.speedup
    ? Number(inferJob.result.speedup)
    : null;
  const tps = (v?: number) => (v == null ? "—" : v.toFixed(1) + " tok/s");

  const shown = liveJob || jobs[0] || null;
  const elapsed = shown?.started_at
    ? (shown.finished_at || Date.now() / 1000) - shown.started_at
    : null;
  const podOptions = running.length ? (
    running.map((item) => (
      <option key={item.id} value={item.id}>
        {item.gpu} · {item.name} · ${item.cost_per_hour}/h
      </option>
    ))
  ) : (
    <option value="">No running pods</option>
  );

  async function benchmark() {
    if (!event.trim()) return;
    setBenching(true);
    setBench(null);
    try {
      setBench(await runPrefillBenchmark(event.trim(), setBenchNote));
      setBenchNote("");
    } catch (cause) {
      setBenchNote(cause instanceof Error ? cause.message : "Benchmark failed");
    } finally {
      setBenching(false);
    }
  }

  return (
    <aside className="lab-side" aria-label="Experiment">
      <Section
        step="CHAT RUNS"
        title="Same prompt, side by side"
        note="Every chat reply is recorded; answers are compared with the first."
        open
      >
        <table className="lab-table">
          <thead>
            <tr>
              <th>GPU</th>
              <th>State</th>
              <th>First token</th>
              <th>Total</th>
              <th>Input</th>
              <th>Answer</th>
            </tr>
          </thead>
          <tbody>
            {runs.length === 0 && (
              <tr>
                <td colSpan={6} className="lab-none">
                  No runs yet
                </td>
              </tr>
            )}
            {runs.map((run) => (
              <tr key={run.id}>
                <td>
                  {gpuName(run.gpu)} <small>{run.vendor}</small>
                </td>
                <td>{run.state}</td>
                <td>{secs(run.ttft)}</td>
                <td>{secs(run.total)}</td>
                <td>{run.input}</td>
                <td className={run.answer === "Differs" ? "bad" : ""}>
                  {run.answer}
                </td>
              </tr>
            ))}
          </tbody>
        </table>
      </Section>

      <Section
        step="BASELINE · AGENT OFF"
        title="Train with standard settings"
        note="This result is the baseline for every later comparison."
        open={!ready}
      >
        <div className="lab-form">
          <label className="span2">
            Model to train
            <select defaultValue="Qwen/Qwen2.5-0.5B">
              <option value="Qwen/Qwen2.5-0.5B">
                Qwen2.5-0.5B · JSON extraction
              </option>
            </select>
          </label>
          <label className="span2">
            Training GPU pod
            <select value={pod} onChange={(e) => setTrainPod(e.target.value)}>
              {podOptions}
            </select>
          </label>
          <label>
            Training steps
            <input
              type="number"
              min={10}
              max={10000}
              value={steps}
              onChange={(e) => setSteps(+e.target.value)}
            />
          </label>
          <label>
            Micro batch
            <input
              type="number"
              min={1}
              max={128}
              value={microBatch}
              onChange={(e) => setMicroBatch(+e.target.value)}
            />
          </label>
        </div>
        <button
          className="primary-btn lab-wide"
          disabled={!pod || busy}
          onClick={() =>
            onStart("/api/jobs/train", {
              pod_id: pod,
              steps,
              dtype: "bf16",
              attention: "sdpa",
              micro_batch: microBatch,
              grad_accum: Math.max(1, Math.round(16 / microBatch)),
            })
          }
        >
          Start baseline training
        </button>
        <p className="lab-hint">
          Agent off · bf16 / SDPA · evaluated automatically on the same held-out
          set
        </p>
        {baseJob || baseQuality ? (
          <Callout tone="ok" title="Agent-off baseline ready">
            Agent results are compared on the same model, data, and step count.
          </Callout>
        ) : failed ? (
          <Callout tone="bad" title="The last baseline training failed">
            {friendlyFailure(failed)}
          </Callout>
        ) : (
          <Callout>
            Finish one baseline run and its training time and quality are pinned
            here.
          </Callout>
        )}
        <div className="lab-stats">
          <Stat
            label="Full pipeline"
            value={secs(duration(baseJob), 0)}
            note="incl. evaluation"
          />
          <Stat
            label="Step time"
            value={secs(baseTrain?.t_step_median_s, 3)}
            note="median"
          />
          <Stat
            label="Exact match"
            value={pct(baseQuality?.exact_match_rate)}
            note="held-out"
          />
          <Stat
            label="Peak VRAM"
            value={
              baseTrain?.peak_vram_gb != null
                ? baseTrain.peak_vram_gb.toFixed(1) + "G"
                : "—"
            }
            note="measured"
          />
        </div>
      </Section>

      <Section
        step="AGENTS"
        title="Apply the agents to the same baseline"
        note="Both need a finished baseline run."
      >
        <div className="lab-agent">
          <h3>Training setup optimization</h3>
          <p>
            Holds tokens/step constant, measures batch × accumulation
            candidates, then retrains the same number of steps with the fastest
            one. Quality is checked on the same held-out set.
          </p>
          <button
            className="outline-btn"
            disabled={!ready || busy}
            onClick={() =>
              onStart("/api/jobs/action/optimize-training", { pod_id: pod })
            }
          >
            Run training agent
          </button>
        </div>
        <div className="lab-agent">
          <h3>Inference throughput optimization</h3>
          <p>
            Runs the same checkpoint and 64 inputs sequentially and batched. A
            speedup only counts when the generated tokens and quality match.
          </p>
          <button
            className="outline-btn"
            disabled={!ready || busy}
            onClick={() =>
              onStart("/api/jobs/action/optimize-inference", { pod_id: pod })
            }
          >
            Run batching agent
          </button>
        </div>
      </Section>

      <Section
        step="COMPARISON"
        title="Before and after the agent"
        note="Anything not measured is left blank, never estimated."
      >
        <table className="lab-table">
          <thead>
            <tr>
              <th>Metric</th>
              <th>Without agent</th>
              <th>With agent</th>
              <th>Verdict</th>
            </tr>
          </thead>
          <tbody>
            <tr>
              <td>Training step time</td>
              <td>{secs(baseStep, 3)}</td>
              <td>{secs(agentStep, 3)}</td>
              <td className={stepGain && stepGain >= 1 ? "good" : ""}>
                {stepGain
                  ? stepGain >= 1
                    ? stepGain.toFixed(2) + "× faster"
                    : (1 / stepGain).toFixed(2) + "× slower"
                  : "Pending"}
              </td>
            </tr>
            <tr>
              <td>Training exact match</td>
              <td>{pct(baseQuality?.exact_match_rate)}</td>
              <td>{pct(agentQuality?.exact_match_rate)}</td>
              <td
                className={
                  qualityDelta != null && qualityDelta >= -0.02 ? "good" : ""
                }
              >
                {qualityDelta != null
                  ? (qualityDelta >= 0 ? "+" : "") +
                    (qualityDelta * 100).toFixed(1) +
                    " pp"
                  : "Pending"}
              </td>
            </tr>
            <tr>
              <td>64-input inference time</td>
              <td>{secs(inferBase?.elapsed_s)}</td>
              <td>{secs(inferAgent?.elapsed_s)}</td>
              <td className={speedup ? "good" : ""}>
                {speedup ? speedup.toFixed(2) + "× faster" : "Pending"}
              </td>
            </tr>
            <tr>
              <td>Inference throughput</td>
              <td>{tps(inferBase?.tokens_per_second)}</td>
              <td>{tps(inferAgent?.tokens_per_second)}</td>
              <td className={speedup ? "good" : ""}>
                {speedup ? speedup.toFixed(2) + "× throughput" : "Pending"}
              </td>
            </tr>
          </tbody>
        </table>
        {trainJob && inferJob ? (
          <Callout tone="ok" title="Comparison complete">
            Training and inference are both measured. Quality gate:{" "}
            {trainJob.result?.validation?.status || "unknown"} /{" "}
            {inferJob.result?.validation?.status || "unknown"}
          </Callout>
        ) : (
          <Callout title="Comparison pending">
            Run both agents after the baseline to fill in results under the same
            conditions.
          </Callout>
        )}
      </Section>

      <Section
        step="QUALITY"
        title="What the model actually gets wrong"
        note="Not just speed: field by field."
      >
        <div className="lab-stats">
          <Stat
            label="Hallucination / after"
            value={pct(experiment?.after?.hallucination_rate)}
            note={
              experiment?.after?.n_nullable != null
                ? `${experiment.after.n_nullable} null answers · before not measurable`
                : "before not measurable"
            }
          />
          <Stat
            label="Omission / after"
            value={pct(experiment?.after?.omission_rate)}
            note="facts present but left out"
          />
          <Stat
            label="Held-out loss"
            value={
              experiment?.after?.held_out_loss != null
                ? experiment.after.held_out_loss.toFixed(4)
                : "—"
            }
            note={
              experiment?.before?.held_out_loss != null &&
              experiment.after?.held_out_loss != null
                ? `${experiment.before.held_out_loss.toFixed(4)} → ${experiment.after.held_out_loss.toFixed(4)}`
                : "—"
            }
          />
          <Stat
            label="Training config"
            value={
              experiment?.train
                ? `${experiment.train.dtype} · ${experiment.train.attention}`
                : "—"
            }
            note={
              experiment?.train
                ? `${experiment.train.steps} steps · batch ${experiment.train.micro_batch}×${experiment.train.grad_accum} · ${Number(experiment.train.tokens_per_step).toLocaleString()} tok/step`
                : "not run"
            }
          />
        </div>
        <div className="lab-chart-head">
          <h3>Accuracy by field</h3>
          <Legend
            items={[
              ["before", "before"],
              ["after", "after"],
            ]}
          />
        </div>
        <FieldChart before={experiment?.before} after={experiment?.after} />
        <div className="lab-chart-head">
          <h3>Training loss</h3>
          <Legend items={[["train loss", "after"]]} />
        </div>
        <LossChart train={experiment?.train} />
      </Section>

      <Section
        step="LONG CONTEXT"
        title="Stop re-reading the same document"
        note="Remove prefill and only decode is left."
      >
        <p className="lab-hint">
          The static policy sits at the <b>front</b> of the prompt. Attention is
          causal, so it can only be reused while the front stays fixed.
        </p>
        <div className="lab-row">
          <button
            className="outline-btn"
            disabled={!lab.serving.running || policyBusy}
            onClick={() => onPolicy(true)}
          >
            Load policy · build cache
          </button>
          <button
            className="text-btn"
            disabled={!lab.serving.running || policyBusy}
            onClick={() => onPolicy(false)}
          >
            Clear cache
          </button>
        </div>
        <p className="lab-hint">{policyNote}</p>
        <h3>Prefill benchmark</h3>
        <p className="lab-hint">
          Output is held short and deterministic so <b>only prefill</b> remains.
          This is the only place a multiplier is claimed.
        </p>
        <label className="lab-label">
          Event
          <textarea
            rows={2}
            value={event}
            onChange={(e) => setEvent(e.target.value)}
          />
        </label>
        <button
          className="outline-btn lab-wide"
          disabled={benching || !lab.serving.running}
          onClick={() => void benchmark()}
        >
          {benching ? "Running…" : "Run benchmark (3 runs each)"}
        </button>
        {benchNote && <div className="lab-empty">{benchNote}</div>}
        {bench && (
          <>
            <Legend
              items={[
                ["prefill", "before"],
                ["decode", "after"],
              ]}
            />
            <SplitBars
              rows={[
                {
                  label: "Cache miss",
                  prefill: Math.max(bench.missP50 - bench.hitP50, 0),
                  decode: bench.hitP50,
                },
                { label: "Cache hit", prefill: 0, decode: bench.hitP50 },
              ]}
            />
            <Callout
              tone={bench.deterministic ? "ok" : "bad"}
              title={
                bench.deterministic
                  ? `Each path is deterministic: all ${bench.repeats} repeats matched`
                  : "The paths do not reproduce, so this measurement is noise"
              }
            >
              {bench.identical
                ? "Output was byte-identical with and without the cache."
                : `The two paths are not byte-identical, though. ${bench.diverge}`}
            </Callout>
            <table className="lab-table">
              <tbody>
                <tr>
                  <td>Input tokens</td>
                  <td>{bench.promptTokens.toLocaleString()}</td>
                </tr>
                <tr>
                  <td>Reused prefix</td>
                  <td>
                    {bench.prefixTokens.toLocaleString()} (
                    {bench.reusePct.toFixed(1)}%)
                  </td>
                </tr>
                <tr>
                  <td>Cache miss p50 / p95</td>
                  <td>
                    {bench.missP50.toFixed(2)}s / {bench.missP95.toFixed(2)}s
                  </td>
                </tr>
                <tr>
                  <td>Cache hit p50 / p95</td>
                  <td>
                    {bench.hitP50.toFixed(2)}s / {bench.hitP95.toFixed(2)}s
                  </td>
                </tr>
                <tr>
                  <td>Multiplier (p50)</td>
                  <td>{(bench.missP50 / bench.hitP50).toFixed(1)}×</td>
                </tr>
              </tbody>
            </table>
            {!bench.identical && (
              <p className="lab-caveat">
                Byte-identical output is <b>not guaranteed</b>. The uncached
                path prefills everything in one pass; the cached path fills
                ahead and appends the rest. Different tensor shapes change the
                kernels and reduction order, which can flip the argmax of
                near-tied tokens at bf16. Each path reproducing itself exactly
                is all that is guaranteed here.
              </p>
            )}
            <p className="lab-caveat">
              This multiplier is for{" "}
              <b>prefill of the repeated policy prefix</b>. The model did not
              get that much faster, and neither does report generation. The
              first request is always a miss, and the cache is cold again after
              a server restart.
            </p>
          </>
        )}
      </Section>

      <Section step="ADVANCED" title="Data generation and migration">
        <h3>Regenerate training data</h3>
        <div className="lab-form">
          <label>
            Total target
            <input
              type="number"
              min={100}
              max={10000}
              value={dataTotal}
              onChange={(e) => setDataTotal(+e.target.value)}
            />
          </label>
          <label>
            Held-out
            <input
              type="number"
              min={20}
              max={2000}
              value={dataHeld}
              onChange={(e) => setDataHeld(+e.target.value)}
            />
          </label>
        </div>
        <div className="lab-row">
          <button
            className="outline-btn"
            disabled={busy}
            onClick={() =>
              onStart("/api/jobs/data", {
                total: dataTotal,
                heldout: dataHeld,
                workers: 8,
              })
            }
          >
            Generate data
          </button>
          {experiment && (
            <span className="lab-hint">
              {experiment.data.train.toLocaleString()} train ·{" "}
              {experiment.data.heldout.toLocaleString()} held-out
            </span>
          )}
        </div>
        <h3>Move and verify a checkpoint</h3>
        <div className="lab-form">
          <label className="span2">
            Source pod
            <select value={from} onChange={(e) => setSource(e.target.value)}>
              {podOptions}
            </select>
          </label>
          <label className="span2">
            Target pod
            <select value={to} onChange={(e) => setTarget(e.target.value)}>
              {podOptions}
            </select>
          </label>
        </div>
        <button
          className="outline-btn"
          disabled={busy || !from || !to || from === to}
          onClick={() =>
            onStart("/api/jobs/action/migrate-nextgen", {
              source_pod_id: from,
              target_pod_id: to,
            })
          }
        >
          Run migration
        </button>
        <p className="lab-hint">
          Verified automatically with the same model and held-out set.
        </p>
      </Section>

      <Section
        step="CURRENT JOB"
        title={
          shown
            ? `${JOB_LABEL[shown.kind] || shown.kind} · ${shown.status}`
            : "Idle"
        }
        open={busy}
      >
        <div className="progress">
          <i style={{ width: `${shown?.progress || 0}%` }} />
        </div>
        {!shown ? (
          <Callout>Stages and timing appear here once a run starts.</Callout>
        ) : shown.status === "failed" ? (
          <Callout tone="bad" title={`Job failed · ${secs(elapsed, 0)}`}>
            {friendlyFailure(shown)}
          </Callout>
        ) : shown.status === "complete" ? (
          <Callout tone="ok" title={`Complete · ${secs(elapsed, 0)}`}>
            Automatic verification results are reflected in the comparison
            above.
          </Callout>
        ) : (
          <Callout
            title={`${shown.stage || shown.status} · ${shown.progress || 0}%`}
          >
            Elapsed {secs(elapsed, 0)} · job {shown.id.slice(0, 8)}
          </Callout>
        )}
        <div className="lab-row">
          <button
            className="text-btn danger"
            disabled={!busy}
            onClick={onCancel}
          >
            Stop job
          </button>
        </div>
        <details className="lab-tableview">
          <summary>View detailed log</summary>
          <pre className="lab-log">
            {(shown?.logs || []).slice(-100).join("\n") || "No active job."}
          </pre>
        </details>
      </Section>
    </aside>
  );
}
