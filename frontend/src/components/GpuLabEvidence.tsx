import { useState } from "react";
import {
  lab,
  type EvidenceRecheck,
  type EvidenceVerdict,
  type RecordedEvidence,
} from "../api/gpulab";
import { gpuName } from "../lib/gpulab";
import { Icon } from "./Icon";
import "./GpuLabEvidence.css";

export type EvidenceComparison = "optimization" | "migration_from_4090";
export type GpuEvidenceSelection = {
  gpuKey?: string;
  comparison?: EvidenceComparison;
};
type Row = RecordedEvidence["rows"][number];
const verdictLabel = (value: EvidenceVerdict) =>
  value.status === "passed"
    ? "Passed"
    : value.status === "rejected"
      ? "Rejected"
      : "Not validated";
const time = (value: number | null | undefined) =>
  value == null ? "Not measured" : `${value.toFixed(3)}s`;
const fieldValue = (record: unknown, field: string) => {
  if (record == null) return "Invalid or unavailable JSON";
  if (typeof record !== "object" || Array.isArray(record))
    return JSON.stringify(record);
  const value = (record as Record<string, unknown>)[field];
  return value === undefined ? "Missing" : JSON.stringify(value);
};

export function GpuLabEvidence({
  evidence,
  selection,
  onSelection,
  onInspect,
}: {
  evidence: RecordedEvidence;
  selection?: GpuEvidenceSelection;
  onSelection?: (gpuKey: string, comparison: EvidenceComparison) => void;
  onInspect?: (gpuKey: string, comparison: EvidenceComparison) => void;
}) {
  const [localKey, setLocalKey] = useState(
    () =>
      evidence.rows.find((row) => /MI300X/i.test(row.gpu))?.key ||
      evidence.rows[0]?.key ||
      "",
  );
  const [localComparison, setLocalComparison] =
    useState<EvidenceComparison>("optimization");
  const gpuKey = selection?.gpuKey ?? localKey;
  const comparison = selection?.comparison ?? localComparison;
  const row =
    evidence.rows.find((item) => item.key === gpuKey) || evidence.rows[0];

  function choose(key: string, mode: EvidenceComparison) {
    setLocalKey(key);
    setLocalComparison(mode);
    onSelection?.(key, mode);
  }

  if (!row)
    return (
      <p className="lab-hint">No recorded GPU comparisons are available.</p>
    );
  return (
    <div className="lab-saved-evidence">
      <label className="lab-saved-select">
        GPU comparison
        <select
          value={row.key}
          onChange={(event) => choose(event.target.value, comparison)}
        >
          {evidence.rows.map((item) => (
            <option key={item.key} value={item.key}>
              {gpuName(item.gpu)}
              {item.measured ? "" : " · not measured"}
            </option>
          ))}
        </select>
      </label>
      <div
        className="lab-saved-modes"
        role="group"
        aria-label="Saved comparison mode"
      >
        <button
          aria-pressed={comparison === "optimization"}
          onClick={() => choose(row.key, "optimization")}
        >
          Optimize this GPU
        </button>
        <button
          aria-pressed={comparison === "migration_from_4090"}
          onClick={() => choose(row.key, "migration_from_4090")}
        >
          From RTX 4090
        </button>
      </div>
      <SavedComparison
        key={`${row.key}:${comparison}:${evidence.model_sha256}:${evidence.dataset_sha256}`}
        row={row}
        comparison={comparison}
        evidence={evidence}
        onInspect={onInspect}
      />
    </div>
  );
}

function SavedComparison({
  row,
  comparison,
  evidence,
  onInspect,
}: {
  row: Row;
  comparison: EvidenceComparison;
  evidence: RecordedEvidence;
  onInspect?: (gpuKey: string, comparison: EvidenceComparison) => void;
}) {
  const [checking, setChecking] = useState(false);
  const [checked, setChecked] = useState<EvidenceRecheck | null>(null);
  const [error, setError] = useState("");
  const [caseIndex, setCaseIndex] = useState(0);
  const verdict = checked?.verdict || row[comparison];
  const examples = verdict.examples || [];
  const example = examples[Math.min(caseIndex, examples.length - 1)];
  const reference =
    comparison === "optimization"
      ? row
      : evidence.rows.find((item) => /4090/.test(item.gpu));
  const referenceTime = row.measured ? (reference?.baseline_s ?? null) : null;
  const candidateTime = row.measured ? row.candidate_s : null;
  const ratio =
    referenceTime && candidateTime ? referenceTime / candidateTime : null;
  const explanation = !row.measured
    ? "No complete measurement is available for this chip."
    : verdict.status === "not_validated"
      ? "This comparison needs a complete recorded evaluation."
      : verdict.status === "passed"
        ? `All ${verdict.cases} JSON outputs preserved their reference values.`
        : verdict.changed_cases
          ? `${verdict.changed_cases} of ${verdict.cases} outputs changed or became invalid. The gate allows zero changes.`
          : "Outputs matched, but not all recorded acceptance checks passed.";

  async function recheck() {
    if (checking || !row.measured) return;
    setChecking(true);
    setError("");
    try {
      const result = await lab<EvidenceRecheck>(
        `/api/evidence/${encodeURIComponent(row.key)}/recheck?comparison=${comparison}`,
      );
      if (
        result.gpu_key !== row.key ||
        result.comparison !== comparison ||
        result.source !== "saved_outputs" ||
        result.model_sha256 !== evidence.model_sha256 ||
        result.dataset_sha256 !== evidence.dataset_sha256 ||
        !["passed", "rejected", "not_validated"].includes(
          result.verdict?.status,
        ) ||
        !Number.isInteger(result.verdict.cases)
      )
        throw new Error(
          "The result does not match this saved comparison. Refresh the evidence and try again.",
        );
      setChecked(result);
      setCaseIndex(0);
    } catch (cause) {
      setError(
        cause instanceof Error
          ? cause.message
          : "Saved evidence could not be rechecked.",
      );
    } finally {
      setChecking(false);
    }
  }

  return (
    <>
      <div className={`lab-saved-verdict ${verdict.status}`}>
        <Icon
          name={verdict.status === "passed" ? "check" : "warning"}
          size={16}
        />
        <strong>{row.measured ? verdictLabel(verdict) : "Not measured"}</strong>
      </div>
      <p className="lab-saved-explanation">{explanation}</p>
      <dl className="lab-saved-metrics">
        <div>
          <dt>
            Reference{" "}
            <small>{reference ? gpuName(reference.gpu) : "Unavailable"}</small>
          </dt>
          <dd>{time(referenceTime)}</dd>
        </div>
        <div>
          <dt>
            Candidate <small>{gpuName(row.gpu)}</small>
          </dt>
          <dd>{time(candidateTime)}</dd>
        </div>
        {ratio != null && (
          <div>
            <dt>Resident response ratio</dt>
            <dd>
              {ratio.toFixed(2)}×
              {ratio > 1
                ? " faster"
                : ratio < 1
                  ? " of reference speed"
                  : " same speed"}
            </dd>
          </div>
        )}
        <div>
          <dt>
            Correct answers <small>Reference → candidate</small>
          </dt>
          <dd>
            {verdict.reference_correct == null ||
            verdict.candidate_correct == null
              ? "Not measured"
              : `${verdict.reference_correct} → ${verdict.candidate_correct} / ${verdict.cases}`}
          </dd>
        </div>
      </dl>
      <p className="lab-hint">
        {verdict.candidate_correct != null &&
        verdict.reference_correct != null &&
        verdict.candidate_correct > verdict.reference_correct &&
        verdict.changed_cases
          ? "Accuracy improved, but changed answers still fail the exact-output rule."
          : "Preserved outputs can still contain mistakes. This gate checks behavior, not perfect accuracy."}
      </p>
      <div className="lab-saved-actions">
        <button
          className="outline-btn"
          disabled={!row.measured || checking}
          onClick={() => void recheck()}
        >
          {checking ? "Checking saved outputs…" : "Recheck saved evidence"}
        </button>
        {onInspect && (
          <button
            className="text-btn"
            onClick={() => onInspect(row.key, comparison)}
          >
            Discuss in chat <Icon name="arrow" size={14} />
          </button>
        )}
      </div>
      {checking && (
        <p className="lab-hint" role="status">
          Comparing the saved files. No GPU job is running.
        </p>
      )}
      {error && (
        <p className="lab-saved-error" role="alert">
          {error} The previous decision is retained.
        </p>
      )}
      {checked && !checking && (
        <p className="lab-hint" role="status">
          Evidence rechecked: {verdictLabel(checked.verdict)}.{" "}
          {new Date(checked.checked_at).toLocaleString()}.
        </p>
      )}
      <details className="lab-saved-details">
        <summary>
          Changed answers{" "}
          {verdict.changed_cases == null ? "" : `(${verdict.changed_cases})`}
        </summary>
        {example ? (
          <>
            <div className="lab-saved-case-nav">
              <span>
                Case {Math.min(caseIndex + 1, examples.length)} /{" "}
                {examples.length}
              </span>
              <div>
                <button
                  aria-label="Previous saved case"
                  disabled={caseIndex === 0}
                  onClick={() =>
                    setCaseIndex((value) => Math.max(0, value - 1))
                  }
                >
                  Previous
                </button>
                <button
                  aria-label="Next saved case"
                  disabled={caseIndex >= examples.length - 1}
                  onClick={() => setCaseIndex((value) => value + 1)}
                >
                  Next
                </button>
              </div>
            </div>
            {example.id && <p className="lab-hint">{example.id}</p>}
            {example.sentence && <blockquote>{example.sentence}</blockquote>}
            {example.fields.map((field) => (
              <div className="lab-saved-field" key={field}>
                <strong>{field}</strong>
                <dl>
                  <div>
                    <dt>Reference</dt>
                    <dd>
                      <code>{fieldValue(example.before, field)}</code>
                    </dd>
                  </div>
                  <div>
                    <dt>Candidate</dt>
                    <dd className="lab-saved-changed">
                      <code>{fieldValue(example.after, field)}</code>
                    </dd>
                  </div>
                  <div>
                    <dt>Expected</dt>
                    <dd>
                      <code>{fieldValue(example.expected, field)}</code>
                    </dd>
                  </div>
                </dl>
              </div>
            ))}
            <p className="lab-hint">
              Expected is the correct label. The reference is the behavior this
              comparison must preserve.
            </p>
          </>
        ) : (
          <p className="lab-hint">
            {verdict.changed_cases === 0
              ? "No changed outputs were recorded in this comparison."
              : "No individual answer comparison is available."}
          </p>
        )}
      </details>
      <details className="lab-saved-details">
        <summary>What to test next</summary>
        {verdict.status === "passed" ? (
          <ol>
            <li>
              Evaluate the selected release checkpoint on an untouched dataset.
            </li>
            <li>
              Repeat quality and latency checks in the actual serving runtime.
            </li>
            <li>Obtain approval before spending money or changing traffic.</li>
          </ol>
        ) : !row.measured || verdict.status === "not_validated" ? (
          <ol>
            <li>Complete GPU startup and a run of the frozen workload.</li>
            <li>
              Retain outputs, model and dataset hashes, and runtime settings.
            </li>
            <li>
              Evaluate all cases before making a performance recommendation.
            </li>
          </ol>
        ) : (
          <ol>
            <li>Keep the reference configuration while investigating.</li>
            <li>
              Compare the first divergent token with model, prompt, precision,
              and decoding fixed.
            </li>
            <li>
              Vary one attention, cache, or compiler setting at a time, then
              evaluate all cases again.
            </li>
          </ol>
        )}
        <p className="lab-hint">
          Diagnostic suggestions, not tested fixes or deployment approval.
        </p>
      </details>
      <details className="lab-saved-details">
        <summary>Checkpoint and measurement conditions</summary>
        <p>
          {evidence.checkpoint}. {evidence.quality_note}
        </p>
        <p>
          {evidence.timing_scope}. {evidence.sampling_note}.
        </p>
        <dl>
          <dt>Model SHA-256</dt>
          <dd>
            <code>{evidence.model_sha256}</code>
          </dd>
          <dt>Dataset SHA-256</dt>
          <dd>
            <code>{evidence.dataset_sha256}</code>
          </dd>
        </dl>
      </details>
    </>
  );
}
