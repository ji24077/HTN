"""Verify an exported GET /v1/tasks/{id} result without trusting the server's verdict."""

import argparse
import json
import sys
from pathlib import Path

from orchestrator.shared.dwp import verify_result
from orchestrator.shared.protocol import Task


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("file", type=Path, help="JSON task response saved from the task API")
    args = parser.parse_args()
    try:
        task = Task.model_validate_json(args.file.read_bytes())
        proof = task.attestation
        if task.state != "succeeded" or not proof or not task.worker_id:
            raise ValueError("task has no accepted device attestation")
        verify_result(
            {**proof, "output": task.result},
            proof["rawOutput"],
            proof["publicKey"],
            task_id=task.spec.id,
            attempt=task.generation,
            worker_id=task.worker_id,
        )
    except (OSError, ValueError, TypeError, KeyError) as exc:
        print(f"Verification failed: {exc}", file=sys.stderr)
        return 1
    print(json.dumps({"verified": True, "task_id": task.spec.id, "worker_id": task.worker_id}))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
