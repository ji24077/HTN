"""One uploaded-project execution owned by a paired DWP agent subprocess.

The DWP agent owns identity, leases, signatures, and the journal. This bridge uses
the existing executor for isolation, validation, transfers, and process cleanup.
Only assignment-scoped credentials reach it, over stdin rather than process args.
"""

import asyncio
import json
import os
import signal
import sys
from types import SimpleNamespace

from ..shared.protocol import TaskSpec
from .devices import capabilities
from .executors.python_project import PythonProjectExecutor


def emit(kind, data):
    print(json.dumps({"kind": kind, "data": data}, allow_nan=False), flush=True)


class Reporter:
    def __init__(self, attempt):
        self.task = SimpleNamespace(generation=attempt)

    def __call__(self, percent):
        emit("progress", {"done": percent, "total": 100})

    def stdout(self, text):
        emit("stdout", {"text": text})

    def stderr(self, text):
        emit("stderr", {"text": text})

    def emit(self, kind, data):
        emit(kind, data)


async def run(request):
    spec = TaskSpec.model_validate(request["spec"])
    if spec.kind != "python_project":
        raise ValueError("Only uploaded Python projects are supported")
    attempt = request["attempt"]
    if type(attempt) is not int or not 1 <= attempt <= 10:
        raise ValueError("Invalid task attempt")
    executor = PythonProjectExecutor(
        request["server"], request["worker_id"], artifact_prefix="/agent/v1"
    )
    task = asyncio.create_task(executor.execute(spec, Reporter(attempt)))
    loop = asyncio.get_running_loop()
    for sig in (signal.SIGTERM, signal.SIGINT):
        loop.add_signal_handler(sig, task.cancel)
    parent = os.getppid()

    async def watch_parent():
        while True:
            await asyncio.sleep(0.5)
            if os.getppid() != parent or parent == 1:
                task.cancel()
                return

    watchdog = asyncio.create_task(watch_parent())
    try:
        async with asyncio.timeout(spec.timeout_seconds):
            result = await task
        emit("result", result)
    finally:
        watchdog.cancel()
        await asyncio.gather(watchdog, return_exceptions=True)


def main():
    if sys.argv[1:] == ["--check"]:
        import ensurepip
        import venv  # noqa: F401

        detected = capabilities("python_project")
        if detected.python is None:
            raise RuntimeError("A working PyTorch installation is required")
        print(
            json.dumps(
                {
                    "runtime": detected.runtime,
                    "vram_mib": detected.vram_mib,
                    "accelerator": detected.accelerator.model_dump(mode="json"),
                    "python": detected.python.model_dump(mode="json"),
                    "pip": ensurepip.version(),
                }
            )
        )
        return
    raw = sys.stdin.buffer.read(131073)
    if len(raw) > 131072:
        raise ValueError("Execution request exceeds 128 KiB")
    asyncio.run(run(json.loads(raw)))


if __name__ == "__main__":
    main()
