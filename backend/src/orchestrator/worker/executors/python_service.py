"""Run uploaded HTTP servers and relay bounded requests over outbound sockets."""

import asyncio
import base64
import hashlib
import logging
import os
import signal
import socket
import sys
import time
from contextlib import asynccontextmanager
from urllib.parse import quote, urlsplit, urlunsplit

import httpx
from websockets.exceptions import WebSocketException

from ...shared.protocol import json_loads, json_text
from ...shared.services import BODY_LIMIT, CHUNK_SIZE, ServiceConfig, safe_headers
from .python_project import LAUNCHER, execution_workspace


@asynccontextmanager
async def connection(executor, task, lane):
    from ..agent import DirectConnect

    parts = urlsplit(executor.server_url)
    url = urlunsplit((parts.scheme, parts.netloc, f"/v1/service-{lane}/{task.spec.id}", "", ""))
    sock = await executor.tunnel.open_socket() if executor.tunnel else None
    try:
        async with DirectConnect(
            url,
            additional_headers={
                "Authorization": "Bearer " + task.spec.payload["artifact_token"],
                "X-Worker-ID": task.worker_id,
                "X-Task-Attempt": str(task.generation),
                "X-Worker-Session": task.session_id,
            },
            max_size=CHUNK_SIZE,
            max_queue=2,
            compression=None,
            proxy=None,
            open_timeout=10,
            close_timeout=1,
            **({"sock": sock} if sock else {}),
        ) as ws:
            yield ws
    finally:
        if sock is not None:
            sock.close()


async def execute_service(executor, spec, report):
    task = report.task
    config = ServiceConfig.model_validate(spec.payload["service"])
    with execution_workspace(report) as (root, cleanup):
        async with asyncio.timeout(config.startup_timeout_seconds):
            async with connection(executor, task, "bundle") as ws:
                raw = bytearray()
                async for chunk in ws:
                    if chunk == "end":
                        break
                    if not isinstance(chunk, bytes):
                        raise ValueError("Invalid bundle frame")
                    raw.extend(chunk)
                    if len(raw) > 192 * 1024 * 1024:
                        raise ValueError("Service bundle too large")
            if hashlib.sha256(raw).hexdigest() != spec.payload["bundle_hash"]:
                raise ValueError("Service bundle checksum mismatch")
            files = json_loads(bytes(raw))["files"]
            for name, content in files.items():
                path = root / name
                if not path.resolve().is_relative_to(root.resolve()) or "\\" in name:
                    raise ValueError("Invalid service bundle path")
                path.parent.mkdir(parents=True, exist_ok=True)
                path.write_bytes(base64.b64decode(content, validate=True))
        # Reuse the trusted launcher/owner watchdog and unchanged Python entrypoint.
        (root / "__dispatch_launcher__.py").write_text(LAUNCHER)
        with socket.socket() as probe:
            probe.bind(("127.0.0.1", 0))
            port = probe.getsockname()[1]
        env = {
            "PATH": os.defpath,
            "HOME": str(root),
            "TMPDIR": str(root),
            "PYTHONIOENCODING": "utf-8",
            "PYTHONUNBUFFERED": "1",
            "DISPATCH_OWNER_PID": str(os.getpid()),
            "DISPATCH_SERVICE_PORT": str(port),
        }
        cleanup["processes_stopped"] = False
        spawn = asyncio.create_task(
            asyncio.create_subprocess_exec(
                sys.executable,
                "-I",
                str(root / "__dispatch_launcher__.py"),
                cwd=root,
                env=env,
                stdin=asyncio.subprocess.PIPE,
                stdout=asyncio.subprocess.PIPE,
                stderr=asyncio.subprocess.PIPE,
                start_new_session=os.name != "nt",
            )
        )
        process = None
        background = []
        try:
            try:
                process = await asyncio.shield(spawn)
            except asyncio.CancelledError:
                process = await spawn
                raise
            process.stdin.write(
                json_text(
                    {
                        "mode": "smoke",
                        "entrypoint": config.entrypoint,
                        "working_directory": config.working_directory,
                        "args": [a.replace("{port}", str(port)) for a in config.args],
                    }
                ).encode()
            )
            await process.stdin.drain()
            process.stdin.close()
            ready = asyncio.Event()

            async def logs(stream, emit):
                window, used = time.monotonic(), 0
                while chunk := await stream.read(2048):
                    if time.monotonic() - window > 60:
                        window, used = time.monotonic(), 0
                    if used < 64000:
                        emit(chunk.decode(errors="replace"))
                    used += len(chunk)

            async def control():
                started, failures, ever_ready = time.monotonic(), 0, False
                outage = None
                async with httpx.AsyncClient(
                    timeout=2, trust_env=False, follow_redirects=False
                ) as client:
                    while process.returncode is None:
                        try:
                            async with connection(executor, task, "control") as ws:
                                while process.returncode is None:
                                    try:
                                        async with client.stream(
                                            "GET",
                                            f"http://127.0.0.1:{port}" + config.readiness_path,
                                        ) as response:
                                            healthy = 200 <= response.status_code < 300
                                    except httpx.HTTPError:
                                        healthy = False
                                    if healthy:
                                        failures, ever_ready = 0, True
                                    else:
                                        ready.clear()
                                        failures += 1
                                    await ws.send(json_text({"ready": healthy}))
                                    async with asyncio.timeout(10):
                                        await ws.recv()
                                    outage = None
                                    if healthy:
                                        ready.set()
                                    if (
                                        not ever_ready
                                        and time.monotonic() - started
                                        > config.startup_timeout_seconds
                                    ) or (ever_ready and failures >= 3):
                                        raise RuntimeError("Service readiness checks failed")
                                    await asyncio.sleep(2)
                        except (OSError, TimeoutError, WebSocketException):
                            ready.clear()
                            outage = outage or time.monotonic()
                            if time.monotonic() - outage >= 10:
                                raise
                            await asyncio.sleep(0.5)
                        # Losing the agent's main session still cancels this entire executor;
                        # a service-channel retry can never outlive the assignment lease.
                raise RuntimeError("Service process exited")

            async def request_slot():
                while process.returncode is None:
                    await ready.wait()
                    try:
                        async with connection(executor, task, "data") as ws:
                            meta = json_loads(await ws.recv())
                            body = bytearray()
                            while True:
                                chunk = await ws.recv()
                                if chunk == b"":
                                    break
                                if not isinstance(chunk, bytes):
                                    raise ValueError("Invalid request body frame")
                                body.extend(chunk)
                                if len(body) > BODY_LIMIT:
                                    raise ValueError("Request body too large")

                            async def relay():
                                async with asyncio.timeout(config.request_timeout_seconds):
                                    async with httpx.AsyncClient(
                                        timeout=config.request_timeout_seconds,
                                        trust_env=False,
                                        follow_redirects=False,
                                    ) as client:
                                        url = f"http://127.0.0.1:{port}/" + quote(
                                            meta["path"].lstrip("/"), safe="/"
                                        )
                                        if meta["query"]:
                                            url += "?" + meta["query"]
                                        async with client.stream(
                                            meta["method"],
                                            url,
                                            content=bytes(body),
                                            headers=safe_headers(meta["headers"]),
                                        ) as response:
                                            await ws.send(
                                                json_text(
                                                    {
                                                        "status": response.status_code,
                                                        "headers": safe_headers(
                                                            response.headers.multi_items()
                                                        ),
                                                    }
                                                )
                                            )
                                            async for chunk in response.aiter_raw():
                                                for offset in range(0, len(chunk), CHUNK_SIZE):
                                                    await ws.send(
                                                        chunk[offset : offset + CHUNK_SIZE]
                                                    )
                                            await ws.send("end")
                                            await ws.wait_closed()

                            relay_task = asyncio.create_task(relay())
                            closed_task = asyncio.create_task(ws.wait_closed())
                            try:
                                done, _ = await asyncio.wait(
                                    [relay_task, closed_task], return_when=asyncio.FIRST_COMPLETED
                                )
                                for finished in done:
                                    finished.result()
                            finally:
                                relay_task.cancel()
                                closed_task.cancel()
                                await asyncio.gather(
                                    relay_task, closed_task, return_exceptions=True
                                )
                    except Exception as exc:
                        logging.getLogger(__name__).debug("Service request channel failed: %s", exc)
                        # Rejected/busy channels are retried; lifecycle health owns restarts.
                        await asyncio.sleep(0.5)
                raise RuntimeError("Service process exited")

            background = [
                asyncio.create_task(logs(process.stdout, report.stdout)),
                asyncio.create_task(logs(process.stderr, report.stderr)),
                asyncio.create_task(control()),
                *[asyncio.create_task(request_slot()) for _ in range(config.concurrency)],
            ]
            # Log readers can finish before process.returncode updates. Supervise control/relays.
            done, _ = await asyncio.wait(background[2:], return_when=asyncio.FIRST_COMPLETED)
            for finished in done:
                finished.result()
            raise RuntimeError("Service process stopped")
        finally:
            for pending in background:
                pending.cancel()
            await asyncio.gather(*background, return_exceptions=True)
            if process is not None:
                try:
                    if os.name != "nt":
                        os.killpg(process.pid, signal.SIGKILL)
                    elif process.returncode is None:
                        process.kill()
                except ProcessLookupError:
                    pass
                # Readers were cancelled above. Drain remaining pipe bytes after killing
                # the process group: Process.wait alone can hang on paused pipe transports.
                await process.communicate()
            cleanup["processes_stopped"] = True
