import { useEffect, useRef, useState, type RefObject } from "react";
import type { Evaluation, TrainRun } from "../api/gpulab";
import { pct } from "../lib/gpulab";

const FIELDS = ["name", "age", "org", "role", "year"];

function useWidth(ref: RefObject<HTMLDivElement | null>, fallback: number) {
  const [width, setWidth] = useState(fallback);
  useEffect(() => {
    const node = ref.current;
    if (!node || typeof ResizeObserver === "undefined") return;
    const observer = new ResizeObserver(() => {
      if (node.clientWidth) setWidth(node.clientWidth);
    });
    observer.observe(node);
    return () => observer.disconnect();
  }, [ref]);
  return Math.max(280, width);
}

// Rounds the data end only; rounding the baseline detaches the bar from its axis.
function barPath(x: number, y: number, w: number, h: number, r: number) {
  r = Math.max(0, Math.min(r, w));
  return `M${x},${y}H${x + w - r}Q${x + w},${y} ${x + w},${y + r}V${y + h - r}Q${x + w},${y + h} ${x + w - r},${y + h}H${x}Z`;
}

export function Legend({ items }: { items: [string, string][] }) {
  return (
    <div className="lab-legend">
      {items.map(([label, series]) => (
        <span key={label}>
          <i className={`lab-swatch ${series}`} />
          {label}
        </span>
      ))}
    </div>
  );
}

export function FieldChart({
  before,
  after,
}: {
  before?: Evaluation | null;
  after?: Evaluation | null;
}) {
  const host = useRef<HTMLDivElement>(null);
  const W = useWidth(host, 360);
  const accuracyBefore = before?.field_accuracy;
  const accuracyAfter = after?.field_accuracy;
  if (!accuracyAfter)
    return (
      <div className="lab-empty" ref={host}>
        No evaluation results yet. They appear after the first training run.
      </div>
    );
  const fields = FIELDS.filter((field) => field in accuracyAfter);
  const padL = 42;
  const padR = 50;
  const padT = 6;
  const padB = 24;
  const BAR = 10;
  const INNER = 3;
  const OUTER = 14;
  const group = BAR * 2 + INNER;
  const H = padT + fields.length * group + (fields.length - 1) * OUTER + padB;
  const x0 = padL;
  const x1 = W - padR;
  const sx = (v: number) => x0 + (x1 - x0) * Math.max(0, Math.min(1, v));
  return (
    <div ref={host}>
      <svg
        className="lab-chart"
        width={W}
        height={H}
        viewBox={`0 0 ${W} ${H}`}
        role="img"
        aria-label="Accuracy per field, before and after training"
      >
        {[0, 0.25, 0.5, 0.75, 1].map((t) => (
          <g key={t}>
            <line
              className="lab-grid"
              x1={sx(t)}
              x2={sx(t)}
              y1={padT}
              y2={H - padB}
              opacity={t === 0 ? 1 : 0.55}
            />
            <text
              className="lab-ax"
              x={sx(t)}
              y={H - padB + 15}
              textAnchor="middle"
            >
              {t * 100}%
            </text>
          </g>
        ))}
        {fields.map((field, i) => {
          const top = padT + i * (group + OUTER);
          return (
            <g key={field}>
              <text
                className="lab-ax"
                x={padL - 9}
                y={top + group / 2 + 4}
                textAnchor="end"
              >
                {field}
              </text>
              {(
                [
                  ["before", accuracyBefore?.[field], top],
                  ["after", accuracyAfter[field], top + BAR + INNER],
                ] as const
              ).map(([series, value, y]) => {
                if (value == null) return null;
                const w = sx(value) - x0;
                return (
                  <g key={series}>
                    <title>{`${field} · ${series}: ${pct(value)} correct`}</title>
                    {w > 0 && (
                      <path
                        className={`lab-mark ${series}`}
                        d={barPath(x0, y, w, BAR, 3)}
                      />
                    )}
                    <text
                      className={series === "after" ? "lab-val" : "lab-ax"}
                      x={x0 + w + 6}
                      y={y + BAR - 1.5}
                    >
                      {pct(value)}
                    </text>
                    {/* A 0.0% bar has no width; the hit target spans the plot. */}
                    <rect
                      fill="transparent"
                      x={x0}
                      y={y - 1}
                      width={x1 - x0}
                      height={BAR + 2}
                    />
                  </g>
                );
              })}
            </g>
          );
        })}
      </svg>
      <details className="lab-tableview">
        <summary>View as table</summary>
        <table className="lab-table">
          <thead>
            <tr>
              <th>Field</th>
              <th>Before</th>
              <th>After</th>
            </tr>
          </thead>
          <tbody>
            {fields.map((field) => (
              <tr key={field}>
                <td>{field}</td>
                <td>{pct(accuracyBefore?.[field])}</td>
                <td>{pct(accuracyAfter[field])}</td>
              </tr>
            ))}
          </tbody>
        </table>
      </details>
    </div>
  );
}

export function LossChart({ train }: { train?: TrainRun | null }) {
  const host = useRef<HTMLDivElement>(null);
  const W = useWidth(host, 360);
  const [hover, setHover] = useState<{ step: number; loss: number } | null>(
    null,
  );
  // train.py records loss_history as a flat list and knows start_step because
  // it can resume; older runs carried {step, loss} pairs.
  const history = train?.loss_history;
  const curve =
    Array.isArray(history) && history.length
      ? history.map((loss, i) => ({ step: (train?.start_step || 0) + i, loss }))
      : train?.loss_curve;
  if (!Array.isArray(curve) || curve.length < 2)
    // Not reconstructed from the job log: it keeps only the last 300 lines, so
    // the early drop would be missing and the curve would look flat.
    return (
      <div className="lab-empty" ref={host}>
        No loss curve was recorded for this run. Future training runs save it
        per step.
      </div>
    );
  const H = 200;
  const padL = 42;
  const padR = 16;
  const padT = 10;
  const padB = 24;
  const xMax = Math.max(...curve.map((d) => d.step));
  const yMax = Math.max(...curve.map((d) => d.loss)) * 1.05;
  const sx = (v: number) => padL + (W - padL - padR) * (xMax ? v / xMax : 0);
  const sy = (v: number) =>
    H - padB - (H - padB - padT) * (yMax ? v / yMax : 0);
  const last = curve[curve.length - 1];
  const every = Math.max(1, Math.floor(curve.length / 20));
  return (
    <div ref={host}>
      <svg
        className="lab-chart"
        width={W}
        height={H}
        viewBox={`0 0 ${W} ${H}`}
        role="img"
        aria-label="Training loss per step"
      >
        {[0, 1, 2, 3, 4].map((i) => {
          const v = (yMax * i) / 4;
          return (
            <g key={i}>
              <line
                className="lab-grid"
                x1={padL}
                x2={W - padR}
                y1={sy(v)}
                y2={sy(v)}
                opacity={i ? 0.55 : 1}
              />
              <text
                className="lab-ax"
                x={padL - 7}
                y={sy(v) + 4}
                textAnchor="end"
              >
                {v.toFixed(2)}
              </text>
            </g>
          );
        })}
        {[0, 0.5, 1].map((t) => (
          <text
            key={t}
            className="lab-ax"
            x={sx(xMax * t)}
            y={H - padB + 15}
            textAnchor="middle"
          >
            {Math.round(xMax * t)}
          </text>
        ))}
        <path
          className="lab-line"
          d={curve
            .map(
              (d, i) =>
                `${i ? "L" : "M"}${sx(d.step).toFixed(1)},${sy(d.loss).toFixed(1)}`,
            )
            .join("")}
        />
        {hover ? (
          <>
            <line
              className="lab-cross"
              x1={sx(hover.step)}
              x2={sx(hover.step)}
              y1={padT}
              y2={H - padB}
            />
            <circle
              className="lab-mark after"
              cx={sx(hover.step)}
              cy={sy(hover.loss)}
              r={4}
            />
            <text
              className="lab-val"
              x={Math.min(sx(hover.step) + 8, W - padR - 96)}
              y={padT + 10}
            >
              step {hover.step} · {hover.loss.toFixed(4)}
            </text>
          </>
        ) : (
          <>
            <circle
              className="lab-mark after"
              cx={sx(last.step)}
              cy={sy(last.loss)}
              r={3.5}
            />
            <text
              className="lab-val"
              x={sx(last.step) - 6}
              y={sy(last.loss) - 9}
              textAnchor="end"
            >
              {last.loss.toFixed(4)}
            </text>
          </>
        )}
        <rect
          fill="transparent"
          style={{ cursor: "crosshair" }}
          x={padL}
          y={padT}
          width={W - padL - padR}
          height={H - padT - padB}
          onMouseMove={(event) => {
            const box =
              event.currentTarget.ownerSVGElement!.getBoundingClientRect();
            const px = (event.clientX - box.left) * (W / box.width);
            let best = curve[0];
            for (const d of curve)
              if (Math.abs(sx(d.step) - px) < Math.abs(sx(best.step) - px))
                best = d;
            setHover(best);
          }}
          onMouseLeave={() => setHover(null)}
        />
      </svg>
      <details className="lab-tableview">
        <summary>View as table</summary>
        <table className="lab-table">
          <thead>
            <tr>
              <th>Step</th>
              <th>Loss</th>
            </tr>
          </thead>
          <tbody>
            {curve
              .filter((_, i) => i % every === 0 || i === curve.length - 1)
              .map((d) => (
                <tr key={d.step}>
                  <td>{d.step}</td>
                  <td>{d.loss.toFixed(4)}</td>
                </tr>
              ))}
          </tbody>
        </table>
      </details>
    </div>
  );
}

/** Stacked prefill + decode latency, one bar per arm. */
export function SplitBars({
  rows,
}: {
  rows: { label: string; prefill: number; decode: number }[];
}) {
  const host = useRef<HTMLDivElement>(null);
  const W = useWidth(host, 360);
  const padL = 76;
  const padR = 54;
  const padT = 4;
  const BAR = 15;
  const GAP = 13;
  const H = padT + rows.length * (BAR + GAP);
  const max = Math.max(...rows.map((r) => r.prefill + r.decode)) * 1.02 || 1;
  const sx = (v: number) => (W - padL - padR) * (v / max);
  return (
    <div ref={host}>
      <svg
        className="lab-chart"
        width={W}
        height={H}
        viewBox={`0 0 ${W} ${H}`}
        role="img"
        aria-label="Latency breakdown, cache miss versus cache hit"
      >
        {rows.map((row, i) => {
          const y = padT + i * (BAR + GAP);
          let x = padL;
          return (
            <g key={row.label}>
              <text
                className="lab-ax"
                x={padL - 9}
                y={y + BAR - 3}
                textAnchor="end"
              >
                {row.label}
              </text>
              {(
                [
                  ["prefill", "before"],
                  ["decode", "after"],
                ] as const
              ).map(([part, series]) => {
                const w = sx(row[part]);
                const start = x;
                x += w;
                // 2px surface gap so the segments read as two, not one bar.
                return w > 0.5 ? (
                  <path
                    key={part}
                    className={`lab-mark ${series}`}
                    d={barPath(start, y, Math.max(w - 2, 0.5), BAR, 3)}
                  >
                    <title>{`${row.label} · ${part}: ${row[part].toFixed(2)}s`}</title>
                  </path>
                ) : null;
              })}
              <text className="lab-val" x={x + 6} y={y + BAR - 3}>
                {(row.prefill + row.decode).toFixed(2)}s
              </text>
            </g>
          );
        })}
      </svg>
    </div>
  );
}
