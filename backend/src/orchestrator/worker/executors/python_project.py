"""Opt-in LOCAL DEVELOPMENT runner. Process isolation only, not an untrusted-code sandbox.

This adapter uses the existing worker transport so preprocessing can be developed
before production worker isolation/provisioning is selected. No model calls here.
"""

import asyncio
import base64
import hashlib
import os
import signal
import sys
import tempfile
import time
from contextlib import contextmanager
from pathlib import Path
from urllib.parse import urlsplit, urlunsplit

import httpx

from ...shared.protocol import json_loads, json_text

# Trusted launcher: uploaded files are untouched. Only this driver wraps their entrypoints.
LAUNCHER = r"""
import importlib, json, os, pathlib, runpy, sys, traceback, time
root = pathlib.Path(__file__).parent
# A killed worker cannot run its own finally block. On POSIX the launcher
# notices reparenting, removes this attempt's files and kills its process group.
if os.name != 'nt':
    import threading, time, signal, shutil
    owner = int(os.environ.get('DISPATCH_OWNER_PID', os.getppid()))
    def watch_owner():
        while True:
            time.sleep(0.5)
            if os.getppid() != owner:
                try:
                    shutil.rmtree(root)
                finally:
                    os.killpg(os.getpgrp(), signal.SIGKILL)
    threading.Thread(target=watch_owner, daemon=True).start()
sys.path.insert(0, str(root))
request = json.load(sys.stdin)
working = (root / request.get('working_directory', '.')).resolve()
if not working.is_relative_to(root.resolve()):
    raise ValueError('Invalid working directory')
os.chdir(working)
sys.path.insert(0, str(working))
def values_for(module, seeds):
    values = module.run_many(seeds, request['parameters']) if hasattr(module, 'run_many') else [module.run(seed, request['parameters']) for seed in seeds]
    if not isinstance(values, list) or len(values) != len(seeds):
        raise ValueError('run_many must return one ordered value per seed')
    return values

started = time.perf_counter()
try:
    if request['mode'] == 'smoke':
        sys.argv = [request['entrypoint'], *request.get('args', [])]
        sys.path.insert(0, str((root / request['entrypoint']).parent))
        try:
            runpy.run_path(str(root / request['entrypoint']), run_name='__main__')
        except SystemExit as e:
            if e.code not in (None, 0):
                raise
        result = {'ok': True}
    elif request['mode'] == 'map':
        module = importlib.import_module(request['module'])
        seeds = request['seeds']
        values = values_for(module, seeds)
        result = {'ok': True, 'items': [{'seed': seed, 'value': value} for seed, value in zip(seeds, values)]}
        if request.get('aggregate_reference'):
            result['aggregate'] = importlib.import_module('__dispatch_aggregate__').aggregate(values, request['parameters'])
        if request.get('partial'):
            result['partial'] = module.reduce(values, request['parameters'])
    elif request['mode'] == 'batch':
        module = importlib.import_module(request['module'])
        span = request['seed_range']
        seeds = range(span['start'], span['start'] + span['count'])
        values = values_for(module, seeds)
        result = {'ok': True, 'seed_range': span}
        if request['aggregation'] == 'inline':
            result['value'] = importlib.import_module('__dispatch_aggregate__').aggregate(values, request['parameters'])
        elif request['aggregation'] == 'partial':
            result['partial'] = module.reduce(values, request['parameters'])
        else:
            result['values'] = values
    elif request['mode'] in ('aggregate', 'merge'):
        module = importlib.import_module(request['module'])
        values = json.loads((root / '__dispatch_values__.json').read_text())
        function = module.merge if request['mode'] == 'merge' else module.aggregate
        result = {'ok': True, 'value': function(values, request['parameters'])}
    else:
        raise ValueError('Unknown execution mode')
    result['compute_seconds'] = time.perf_counter() - started
    text = json.dumps(result, allow_nan=False)
    if len(text.encode()) > 48000:
        raise ValueError('Result exceeds 48 KiB; reduce batch size or output size')
except BaseException as error:
    traceback.print_exc()
    text = json.dumps({'ok': False, 'error': type(error).__name__ + ': ' + str(error)[:2000]})
(root / '__dispatch_result__.json').write_text(text)
"""


@contextmanager
def execution_workspace(report):
    """A cleanup acknowledgment is emitted only after files were actually removed."""
    root = None
    cleanup = {"processes_stopped": True}
    try:
        with tempfile.TemporaryDirectory(prefix="dispatch-simulation-") as directory:
            root = Path(directory)
            yield root, cleanup
    finally:
        if (
            root is not None
            and not root.exists()
            and cleanup["processes_stopped"]
            and hasattr(report, "emit")
        ):
            report.emit("cleaned", {"workspace_removed": True, "processes_stopped": True})


class PythonProjectExecutor:
    kind = "python_project"
    kinds = ("python_project", "stub")

    def __init__(self, server_url, worker_id, *, fetch=None):
        parsed = urlsplit(server_url)
        self.origin = urlunsplit(
            ("https" if parsed.scheme == "wss" else "http", parsed.netloc, "", "", "")
        )
        self.worker_id = worker_id
        self.fetch = fetch or self.download

    async def download(self, spec):
        payload = spec.payload
        async with (
            httpx.AsyncClient(timeout=30, follow_redirects=False, trust_env=False) as client,
            client.stream(
                "GET",
                f"{self.origin}/v1/execution-bundles/{spec.id}/{payload['bundle_hash']}",
                headers={
                    "Authorization": f"Bearer {payload['artifact_token']}",
                    "X-Worker-ID": self.worker_id,
                },
            ) as response,
        ):
            response.raise_for_status()
            raw = bytearray()
            async for chunk in response.aiter_bytes():
                raw.extend(chunk)
                if len(raw) > 16 * 1024 * 1024:
                    raise ValueError("Execution bundle too large")
        if hashlib.sha256(raw).hexdigest() != payload["bundle_hash"]:
            raise ValueError("Execution bundle checksum mismatch")
        return json_loads(bytes(raw))["files"]

    async def execute(self, spec, report):
        if spec.kind == "stub":
            from .stub import StubExecutor

            return await StubExecutor().execute(spec, report)
        execution_started = time.perf_counter()
        with execution_workspace(report) as (root, cleanup):
            directory = str(root)
            files = await self.fetch(spec)
            download_seconds = time.perf_counter() - execution_started
            for name, content in files.items():
                path = root / name
                if not path.resolve().is_relative_to(root.resolve()) or "\\" in name:
                    raise ValueError("Invalid execution bundle path")
                path.parent.mkdir(parents=True, exist_ok=True)
                path.write_bytes(base64.b64decode(content, validate=True))
            (root / "__dispatch_launcher__.py").write_text(LAUNCHER)
            # Do not inherit backend/model/worker credentials into the project process.
            env = {
                "PATH": os.defpath,
                "HOME": directory,
                "TMPDIR": directory,
                "PYTHONIOENCODING": "utf-8",
                "PYTHONUNBUFFERED": "1",
                "DISPATCH_OWNER_PID": str(os.getpid()),
            }
            cleanup["processes_stopped"] = False
            spawn = asyncio.create_task(
                asyncio.create_subprocess_exec(
                    sys.executable,
                    "-I",
                    str(root / "__dispatch_launcher__.py"),
                    cwd=directory,
                    env=env,
                    stdin=asyncio.subprocess.PIPE,
                    stdout=asyncio.subprocess.PIPE,
                    stderr=asyncio.subprocess.PIPE,
                    start_new_session=os.name != "nt",
                )
            )
            try:
                process = await asyncio.shield(spawn)
            except asyncio.CancelledError:
                process = await spawn
                try:
                    if os.name != "nt":
                        os.killpg(process.pid, signal.SIGKILL)
                    else:
                        process.kill()
                except ProcessLookupError:
                    pass
                await process.wait()
                cleanup["processes_stopped"] = True
                raise
            except Exception:
                cleanup["processes_stopped"] = True  # No child was created.
                raise

            async def drain(stream, emit):
                total = 0
                while chunk := await stream.read(2048):
                    if total < 64000:
                        emit(chunk.decode(errors="replace"))
                    total += len(chunk)

            readers = [
                asyncio.create_task(
                    drain(process.stdout, getattr(report, "stdout", lambda _: None))
                ),
                asyncio.create_task(
                    drain(process.stderr, getattr(report, "stderr", lambda _: None))
                ),
            ]
            try:
                report(0)
                process.stdin.write(
                    json_text(
                        {
                            key: value
                            for key, value in spec.payload.items()
                            if key not in {"artifact_token", "bundle_hash"}
                        }
                    ).encode()
                )
                await process.stdin.drain()
                process.stdin.close()
                # asyncio Process.wait() can wait on pipes inherited by descendants.
                # Observe the leader exit first, then close the whole process group.
                while process.returncode is None:
                    await asyncio.sleep(0.02)
                if os.name != "nt":
                    try:
                        os.killpg(process.pid, signal.SIGKILL)
                    except ProcessLookupError:
                        pass
                await process.wait()
                await asyncio.gather(*readers)
                path = root / "__dispatch_result__.json"
                if not path.exists():
                    return {
                        "ok": False,
                        "error": f"Python process exited with code {process.returncode}; see logs",
                    }
                if path.stat().st_size > 48000:
                    return {"ok": False, "error": "Execution output exceeded 48 KiB"}
                result = json_loads(path.read_bytes())
                if not isinstance(result, dict) or type(result.get("ok")) is not bool:
                    return {"ok": False, "error": "Invalid execution result"}
                result["metrics"] = {
                    "execution_seconds": time.perf_counter() - execution_started,
                    "download_seconds": download_seconds,
                    "output_bytes": path.stat().st_size,
                    "cpu_count": os.cpu_count(),
                }
                report(100)
                return result
            finally:
                # Also reap descendants if a script exits while child processes still run.
                try:
                    if os.name != "nt":
                        os.killpg(process.pid, signal.SIGKILL)
                    elif process.returncode is None:
                        process.kill()
                except ProcessLookupError:
                    pass
                await process.wait()
                for reader in readers:
                    reader.cancel()
                await asyncio.gather(*readers, return_exceptions=True)
                cleanup["processes_stopped"] = True
