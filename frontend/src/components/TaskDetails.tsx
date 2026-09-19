import { useEffect, useRef } from "react";
import type { Task } from "../api/types";
import { taskTitle } from "../lib/format";
export function TaskDetails({
  task,
  onClose,
}: {
  task: Task | null;
  onClose: () => void;
}) {
  const ref = useRef<HTMLDialogElement>(null);
  const open = task !== null;
  useEffect(() => {
    const dialog = ref.current;
    if (open && !dialog?.open) dialog?.showModal();
    else if (!open && dialog?.open) dialog.close();
  }, [open]);
  return (
    <dialog
      ref={ref}
      id="result-dialog"
      aria-labelledby="result-title"
      onClose={onClose}
    >
      <header>
        <h2 id="result-title">{task ? taskTitle(task) : "Task details"}</h2>
        <button
          className="icon-btn"
          id="close-dialog"
          aria-label="Close task details"
          onClick={onClose}
        >
          ×
        </button>
      </header>
      <pre id="result-json">{task ? JSON.stringify(task, null, 2) : ""}</pre>
    </dialog>
  );
}
