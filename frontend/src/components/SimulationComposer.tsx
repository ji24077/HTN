import { useRef, useState, type SubmitEvent } from "react";
import { uploadSimulation } from "../api/client";
import type { Task } from "../api/types";
import { Icon } from "./Icon";

export function SimulationComposer({
  onCreated,
}: {
  onCreated: (task: Task) => void;
}) {
  const [files, setFiles] = useState<File[]>([]);
  const [description, setDescription] = useState("");
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
      const task = await uploadSimulation(
        files,
        description.trim(),
        requestId.current,
      );
      setFiles([]);
      setDescription("");
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
      <p>
        Upload your Python simulation and describe the run. The agent will
        inspect, adapt, validate, and launch it.
      </p>
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
        <span>Python script, ZIP project, and supporting inputs</span>
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
          aria-label="Simulation files"
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
        placeholder="Run 1,000 trials using the attached inputs and return the outcome distribution."
      />
      <p className="simulation-limits">
        Up to 3 adaptation attempts, 4 workers, and 30 minutes. Successful
        validation starts the full run automatically.
      </p>
      {error && <p role="alert">{error}</p>}
      <button
        className="submit"
        disabled={busy || !files.length || !description.trim()}
      >
        {busy ? "Uploading…" : "Submit simulation"}
        <Icon name="arrow" size={16} />
      </button>
    </form>
  );
}
