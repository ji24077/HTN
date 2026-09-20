"""Supervise the bundled tsnet helper without changing the host's network settings."""

import asyncio
import json
import os
import shutil
import socket
from pathlib import Path
from urllib.parse import urlsplit

from .config import WorkerConfig


class EmbeddedTunnel:
    def __init__(self, config: WorkerConfig):
        target = urlsplit(config.url)
        if (
            target.scheme != "wss"
            or not target.hostname
            or not target.hostname.endswith(".ts.net")
            or target.query
        ):
            raise ValueError("embedded Tailscale requires a wss:// full .ts.net SERVER_URL")
        self.target = f"{target.hostname}:{target.port or 443}"
        self.hostname = os.getenv("TAILSCALE_HOSTNAME", f"orch-{config.worker_id}")
        self.state = (
            Path(
                os.getenv(
                    "TAILSCALE_STATE_DIR",
                    str(Path.home() / ".local/state/orchestrator/tailscale" / config.worker_id),
                )
            )
            .expanduser()
            .resolve()
        )
        self.helper = os.getenv("TAILSCALE_HELPER") or shutil.which("orchestrator-tunnel")
        if not self.helper or not Path(self.helper).is_file():
            raise ValueError("bundled tunnel missing; build it and set TAILSCALE_HELPER or PATH")
        self.process: asyncio.subprocess.Process | None = None
        self.port: int | None = None

    async def __aenter__(self):
        self.state.mkdir(parents=True, mode=0o700, exist_ok=True)
        self.state.chmod(0o700)
        # Never pass worker tokens, database passwords, or Supabase keys to the helper.
        env = {
            k: v
            for k, v in os.environ.items()
            if k in {"HOME", "PATH", "TMPDIR", "TMP", "TEMP", "SystemRoot", "TS_AUTHKEY"}
        }
        self.process = await asyncio.create_subprocess_exec(
            str(Path(self.helper).resolve()),
            "--target",
            self.target,
            "--hostname",
            self.hostname,
            "--state-dir",
            str(self.state),
            env=env,
            stdin=asyncio.subprocess.PIPE,
            stdout=asyncio.subprocess.PIPE,
            # Tailscale's interactive sign-in link goes directly to the user's terminal.
            stderr=None,
        )
        try:
            async with asyncio.timeout(310):
                line = await self.process.stdout.readline()
            try:
                message = json.loads(line)
                port = message["port"]
                if message["event"] != "ready" or type(port) is not int or not 1 <= port <= 65535:
                    raise ValueError
            except (ValueError, KeyError, TypeError):
                raise RuntimeError(
                    "Tailscale helper exited or returned invalid readiness"
                ) from None
            self.port = port
            return self
        except BaseException:
            await self.close()
            raise

    async def open_socket(self) -> socket.socket:
        if self.process is None or self.process.returncode is not None or self.port is None:
            raise RuntimeError("Tailscale helper is not running")
        sock = socket.socket(socket.AF_INET, socket.SOCK_STREAM)
        sock.setblocking(False)
        try:
            async with asyncio.timeout(10):
                await asyncio.get_running_loop().sock_connect(sock, ("127.0.0.1", self.port))
            return sock
        except BaseException:
            sock.close()
            raise

    async def wait(self) -> None:
        await self.process.wait()
        raise RuntimeError("Tailscale helper stopped; worker shut down without a direct fallback")

    async def close(self) -> None:
        if self.process is None:
            return
        if self.process.stdin:
            self.process.stdin.close()
        if self.process.returncode is None:
            try:
                self.process.terminate()
            except ProcessLookupError:
                pass
            try:
                async with asyncio.timeout(5):
                    await self.process.wait()
            except TimeoutError:
                self.process.kill()
                await self.process.wait()

    async def __aexit__(self, *args):
        await self.close()
