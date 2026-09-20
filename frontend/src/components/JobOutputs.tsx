import { useEffect, useRef, useState } from "react";
import {
  downloadJobOutput,
  jobOutputs,
  previewJobOutput,
  type JobOutput,
} from "../api/client";
import { Icon } from "./Icon";

function formatSize(bytes: number) {
  const units = ["B", "KiB", "MiB", "GiB", "TiB"];
  const index = Math.min(
    units.length - 1,
    Math.max(0, Math.floor(Math.log2(bytes || 1) / 10)),
  );
  return `${(bytes / 1024 ** index).toFixed(index ? 1 : 0)} ${units[index]}`;
}
function purpose(name: string) {
  if (/\.(pt|pth|ckpt|safetensors)$/i.test(name)) return "Model checkpoint";
  if (/\.(png|jpg|jpeg|webp)$/i.test(name)) return "Image";
  if (/\.(md|pdf|html)$/i.test(name)) return "Document";
  if (/metrics.*\.json$/i.test(name)) return "Evaluation metrics";
  if (/\.(csv|json)$/i.test(name)) return "Data";
  return "Output file";
}
export function JobOutputs({
  jobId,
  phase,
  compact = false,
}: {
  jobId: string;
  phase: string;
  compact?: boolean;
}) {
  const [files, setFiles] = useState<JobOutput[]>([]);
  const [fileError, setFileError] = useState("");
  const [error, setError] = useState("");
  const [loading, setLoading] = useState(true);
  const [revision, setRevision] = useState(0);
  const [busy, setBusy] = useState("");
  const previewRequest = useRef<AbortController | null>(null);
  const [preview, setPreview] = useState<{
    name: string;
    text?: string;
    url?: string;
  } | null>(null);
  useEffect(() => {
    const controller = new AbortController();
    previewRequest.current?.abort();
    setFiles([]);
    setFileError("");
    setError("");
    setLoading(true);
    setPreview(null);
    void jobOutputs(jobId, controller.signal)
      .then(({ files }) => {
        if (!controller.signal.aborted) setFiles(files);
      })
      .catch(() => {
        if (!controller.signal.aborted)
          setFileError("Could not load output files.");
      })
      .finally(() => {
        if (!controller.signal.aborted) setLoading(false);
      });
    return () => {
      controller.abort();
      previewRequest.current?.abort();
    };
  }, [jobId, phase, revision]);
  useEffect(
    () => () => {
      if (preview?.url) URL.revokeObjectURL(preview.url);
    },
    [preview],
  );
  async function download(name: string, id?: string) {
    setBusy(id || "result");
    setError("");
    try {
      await downloadJobOutput(jobId, name, id);
    } catch (cause) {
      setError(cause instanceof Error ? cause.message : "Download failed.");
    } finally {
      setBusy("");
    }
  }
  async function show(file: JobOutput) {
    previewRequest.current?.abort();
    const controller = new AbortController();
    previewRequest.current = controller;
    setBusy(file.id);
    setError("");
    try {
      const blob = await previewJobOutput(jobId, file.id, controller.signal);
      if (controller.signal.aborted) return;
      if (/\.(png|jpg|jpeg|webp)$/i.test(file.name)) {
        // Force an image media type; never render uploaded HTML as a document.
        const extension = file.name.split(".").pop()!.toLowerCase();
        setPreview({
          name: file.name,
          url: URL.createObjectURL(
            new Blob([blob], {
              type: `image/${extension === "jpg" ? "jpeg" : extension}`,
            }),
          ),
        });
      } else {
        let text = await blob.text();
        if (/\.json$/i.test(file.name)) {
          try {
            text = JSON.stringify(JSON.parse(text), null, 2);
          } catch {
            /* Show original text. */
          }
        }
        if (!controller.signal.aborted) setPreview({ name: file.name, text });
      }
    } catch (cause) {
      if (!controller.signal.aborted)
        setError(
          cause instanceof Error ? cause.message : "Preview unavailable.",
        );
    } finally {
      setBusy("");
    }
  }
  const terminal = ["completed", "failed", "cancelled", "stopped"].includes(
    phase,
  );
  return (
    <section className="output-files" aria-label="Output files">
      <div className="section-heading">
        <h3>{compact ? "Your files" : "Output files"}</h3>
        <span className="muted">
          {loading
            ? "Loading…"
            : `${files.length} ${files.length === 1 ? "file" : "files"}`}
        </span>
      </div>
      {loading && (
        <p role="status" className="muted">
          Loading output files…
        </p>
      )}
      <ul className="artifact-list">
        {files.map((file) => (
          <li key={file.id}>
            <span className="artifact-icon">
              <Icon name="jobs" size={20} />
            </span>
            <div className="artifact-description">
              <strong>{file.name}</strong>
              <small>
                {purpose(file.name)} · {formatSize(file.size)} · attempt{" "}
                {file.attempt}
              </small>
            </div>
            <div className="artifact-actions">
              {file.size <= 2 * 1024 * 1024 &&
                /\.(json|txt|md|csv|log|png|jpe?g|webp)$/i.test(file.name) && (
                  <button
                    className="text-btn"
                    disabled={!!busy}
                    aria-label={`Preview ${file.name}`}
                    onClick={() => void show(file)}
                  >
                    Preview
                  </button>
                )}
              <button
                className="outline-btn"
                disabled={!!busy}
                aria-label={`Download ${file.name}`}
                onClick={() => void download(file.name, file.id)}
              >
                <Icon name="download" size={15} />
                {busy === file.id ? "Preparing…" : "Download"}
              </button>
            </div>
          </li>
        ))}
      </ul>
      {!loading && !fileError && !files.length && (
        <p className="empty-state">
          {terminal
            ? "No file artifacts were produced. Any recorded results appear in Overview."
            : "Files will appear here after execution produces outputs."}
        </p>
      )}
      {preview && (
        <div className="artifact-preview">
          <div className="section-heading">
            <h4>{preview.name}</h4>
            <button className="text-btn" onClick={() => setPreview(null)}>
              Close preview
            </button>
          </div>
          {preview.url ? (
            <img src={preview.url} alt={preview.name} />
          ) : (
            <pre>{preview.text}</pre>
          )}
        </div>
      )}
      {!compact && phase === "completed" && (
        <div className="raw-export">
          <div>
            <strong>Raw execution data</strong>
            <p className="muted">Machine-readable JSON for further analysis.</p>
          </div>
          <button
            className="text-btn"
            disabled={!!busy}
            onClick={() => void download("result.json")}
          >
            {busy === "result"
              ? "Starting download…"
              : "Download raw result JSON"}
          </button>
        </div>
      )}
      {fileError && (
        <p role="alert" className="inline-alert">
          {fileError}{" "}
          <button
            className="text-btn"
            onClick={() => setRevision((n) => n + 1)}
          >
            Retry files
          </button>
        </p>
      )}
      {error && (
        <p role="alert" className="inline-alert">
          {error}
        </p>
      )}
    </section>
  );
}
