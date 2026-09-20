"""Submit the heavy Asian-option simulation and verify the planner distributes it.

    DWP_ADMIN_TOKEN=... python examples/asian_option_pipeline_check.py

Needs a running backend (default http://127.0.0.1:8080, override with DWP_SERVER) with at
least two connected CPU Python workers. The admin token defaults to ~/.dwp/admin-token.
Exits nonzero if the job does not complete, or if its batches all ran on one worker.
"""

import base64
import json
import os
import sys
import time
import urllib.request
import uuid
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
SOURCE = ROOT / "examples/projects/asian_option_monte_carlo/simulate.py"
SERVER = os.getenv("DWP_SERVER", "http://127.0.0.1:8080").rstrip("/")
DESCRIPTION = (
    "Run 360 independent trials of simulate(seed) and return the mean discounted payoff "
    "with its standard error, exactly as summarize() in the file computes them. Each trial "
    "takes about two seconds of CPU; distribute the trials across the available workers "
    "when that finishes sooner than one machine would."
)
FINISHED = {"completed", "failed", "cancelled"}


def admin_token() -> str:
    token = os.getenv("DWP_ADMIN_TOKEN")
    if token:
        return token
    path = Path(os.getenv("DWP_ADMIN_TOKEN_FILE", Path.home() / ".dwp/admin-token"))
    return path.read_text().strip()


def call(method: str, path: str, body=None):
    request = urllib.request.Request(
        SERVER + path,
        method=method,
        headers={"Authorization": "Bearer " + admin_token(), "Content-Type": "application/json"},
        data=json.dumps(body).encode() if body is not None else None,
    )
    with urllib.request.urlopen(request, timeout=60) as response:
        return json.load(response)


def main() -> int:
    started = time.monotonic()
    task = call(
        "POST",
        "/v1/simulations",
        {
            "request_id": str(uuid.uuid4()),
            "description": DESCRIPTION,
            "workload": "simulation",
            "files": [
                {
                    "name": "simulate.py",
                    "content": base64.b64encode(SOURCE.read_bytes()).decode(),
                }
            ],
            "max_runtime_seconds": 3600,
            "max_workers": 4,
        },
    )
    job_id = task["spec"]["job_id"]
    print(f"submitted {job_id}", flush=True)
    last = None
    while True:
        status = call("GET", f"/v1/simulations/{job_id}")
        current = (status["phase"], (status.get("message") or "")[:120])
        if current != last:
            print(f"{time.monotonic() - started:7.0f}s  {current[0]:<24} {current[1]}", flush=True)
            last = current
        if status["phase"] in FINISHED:
            break
        time.sleep(3)
    elapsed = time.monotonic() - started
    batches = [t for t in status.get("tasks", []) if str(t.get("role", "")).startswith("batch-")]
    workers = {t.get("worker_id") for t in batches}
    result = call("GET", f"/v1/tasks/{job_id}").get("result") or {}
    print(json.dumps({
        "phase": status["phase"],
        "elapsed_seconds": round(elapsed),
        "batches": len(batches),
        "batch_workers": sorted(w for w in workers if w),
        "output": result.get("output"),
    }, indent=1), flush=True)
    if status["phase"] != "completed":
        return 1
    if len(workers) < 2:
        print("the planner did not distribute the batches", file=sys.stderr)
        return 2
    return 0


if __name__ == "__main__":
    sys.exit(main())
