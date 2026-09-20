import { useRef, useState, type SubmitEvent } from "react";
import { uploadSimulation } from "../api/client";
import type { Task } from "../api/types";
import { MaxSpendField, parseMaxSpend } from "./MaxSpendField";
import { Icon } from "./Icon";

export function SimulationComposer({
  onCreated,
}: {
  onCreated: (task: Task) => void;
}) {
  const [maxSpend, setMaxSpend] = useState("");
  const [files, setFiles] = useState<File[]>([]);
  const [description, setDescription] = useState("");
  const [mode, setMode] = useState<"job" | "service">("job");
  const [entrypoint, setEntrypoint] = useState("");
  const [health, setHealth] = useState("/health");
  const [runtime, setRuntime] = useState<"cpu" | "cuda" | "mps">("cpu");
  const [vram, setVram] = useState(0);
  const [lifetime, setLifetime] = useState("");
  const [busy, setBusy] = useState(false);
  const [error, setError] = useState("");
  const requestId = useRef(crypto.randomUUID());
  const sending = useRef(false);
  const input = useRef<HTMLInputElement>(null);
  function add(next: File[]) {
    if (sending.current) return;
    const merged = [...files, ...next];
    if (
      merged.length > 100 ||
      merged.reduce((sum, file) => sum + file.size, 0) > 8 * 1024 * 1024
    ) {
      setError("Choose at most 100 files totaling 8 MiB or less.");
      return;
    }
    if (
      new Set(merged.map((file) => file.name.toLowerCase())).size !==
      merged.length
    ) {
      setError(
        "Files must have unique names. Use a ZIP to preserve project folders.",
      );
      return;
    }
    requestId.current = crypto.randomUUID();
    setError("");
    setFiles(merged);
  }
  async function submit(event: SubmitEvent<HTMLFormElement>) {
    event.preventDefault();
    if (!files.length || !description.trim() || sending.current) return;
    sending.current = true;
    setBusy(true);
    setError("");
    try {
      const usageCap = parseMaxSpend(maxSpend);
      const task = await uploadSimulation(
        files,
        description.trim(),
        requestId.current,
        usageCap,
        ...(mode === "service" ? [{ entrypoint: entrypoint.trim() || undefined, readiness_path: health,
          requirements: { runtime, vram_mib: vram }, lifetime_seconds: lifetime ? Number(lifetime) * 3600 : undefined }] : []),
      );
      setFiles([]);
      setDescription("");
      setMaxSpend("");
      requestId.current = crypto.randomUUID();
      onCreated(task);
    } catch (cause) {
      setError(
        cause instanceof Error
          ? cause.message
          : "Upload failed. Your files are still selected; try again.",
      );
    } finally {
      sending.current = false;
      setBusy(false);
    }
  }
  return (
    <form className="composer simulation-form" onSubmit={submit}>
      <label className="field-label" htmlFor="execution-mode">Run as</label>
      <select id="execution-mode" value={mode} disabled={busy} onChange={event => { setMode(event.target.value as "job" | "service"); requestId.current = crypto.randomUUID(); }}>
        <option value="job">Job</option><option value="service">Service</option>
      </select>
      {mode === "service" && <fieldset disabled={busy} onChange={() => { requestId.current = crypto.randomUUID(); }}>
        <legend>Service settings</legend>
        <p>Upload an HTTP server that listens on DISPATCH_SERVICE_PORT. It stays running and returns an endpoint.</p>
        <label>Entrypoint (optional for a single Python file)<input value={entrypoint} onChange={event => setEntrypoint(event.target.value)} placeholder="server.py" /></label>
        <label>Readiness path<input value={health} onChange={event => setHealth(event.target.value)} /></label>
        <label>Runtime<select value={runtime} onChange={event => setRuntime(event.target.value as typeof runtime)}><option value="cpu">CPU</option><option value="cuda">CUDA</option><option value="mps">MPS</option></select></label>
        <label>Required VRAM (MiB)<input type="number" min="0" value={vram} onChange={event => setVram(Number(event.target.value))} /></label>
        <label>Run for hours (blank means until stopped)<input type="number" min="1" max="8760" value={lifetime} onChange={event => setLifetime(event.target.value)} /></label>
      </fieldset>}
      {mode === "job" && <p>
        Upload your Python simulation and describe the run. The agent will
        inspect, adapt, validate, and launch it.
      </p>}
      <div
        className="upload-zone"
        onDragOver={(event) => event.preventDefault()}
        onDrop={(event) => {
          event.preventDefault();
          add(Array.from(event.dataTransfer.files));
        }}
      >
        <Icon name="jobs" size={26} />
        <strong>Drop your files here</strong>
        <span>{mode === "service" ? "Serving code and supporting files, or a ZIP project" : "Python script, ZIP project, and supporting inputs"}</span>
        <button
          className="outline-btn"
          type="button"
          disabled={busy}
          onClick={() => input.current?.click()}
        >
          Choose files
        </button>
        <input
          ref={input}
          aria-label={mode === "service" ? "Service files" : "Simulation files"}
          type="file"
          multiple
          hidden
          onChange={(event) => {
            add(Array.from(event.target.files || []));
            event.target.value = "";
          }}
        />
        <small>8 MiB total · up to 128 KiB of Python source</small>
      </div>
      {files.length > 0 && (
        <ul className="upload-files">
          {files.map((file) => (
            <li key={file.name}>
              <span>
                {file.name}
                <small>{Math.ceil(file.size / 1024)} KiB</small>
              </span>
              <button
                className="text-btn"
                type="button"
                disabled={busy}
                aria-label={`Remove ${file.name}`}
                onClick={() => {
                  setFiles(files.filter((item) => item !== file));
                  requestId.current = crypto.randomUUID();
                }}
              >
                Remove
              </button>
            </li>
          ))}
        </ul>
      )}
      <label className="field-label" htmlFor="simulation-description">
        Describe your run
      </label>
      <textarea
        id="simulation-description"
        required
        maxLength={8000}
        disabled={busy}
        value={description}
        onChange={(event) => {
          setDescription(event.target.value);
          requestId.current = crypto.randomUUID();
        }}
        placeholder={mode === "service" ? "Describe the service and what its API should do." : "Run 1,000 trials using the attached inputs and return the outcome distribution."}
      />
      <MaxSpendField
        value={maxSpend}
        disabled={busy}
        onChange={(value) => {
          setMaxSpend(value);
          requestId.current = crypto.randomUUID();
        }}
      />
      <p className="simulation-limits">
        {mode === "service" ? <>
          Runs on one compatible Python worker until stopped or its configured
          lifetime ends. Dependencies must already be installed. The endpoint
          accepts requests once the readiness check passes.
        </> : <>
          Up to 3 adaptation attempts, 4 workers, and 30 minutes. Successful
          validation starts the full run automatically.
        </>}
      </p>
      {error && <p role="alert">{error}</p>}
      <button
        className="submit"
        disabled={busy || !files.length || !description.trim()}
      >
        {busy ? "Uploading…" : mode === "service" ? "Submit service" : "Submit simulation"}
        <Icon name="arrow" size={16} />
      </button>
    </form>
  );
}
