"""Run the website API, private worker gateway, and embedded Tailscale together."""

import argparse
import asyncio
import logging
import os
import shutil
import signal
import sys
from pathlib import Path

log = logging.getLogger(__name__)


async def stop(process):
    if process.stdin:
        process.stdin.close()
    if process.returncode is None:
        try:
            process.terminate()
        except ProcessLookupError:
            pass
        try:
            async with asyncio.timeout(15):
                await process.wait()
        except TimeoutError:
            process.kill()
            await process.wait()


async def service(name):
    process = await asyncio.create_subprocess_exec(
        sys.executable, "-m", "orchestrator.server.bundle", "--child", name
    )
    try:
        await process.wait()
        raise RuntimeError(f"{name} service stopped")
    finally:
        await stop(process)


async def tunnel(helper, state, hostname, port):
    env = {
        k: v
        for k, v in os.environ.items()
        if k in {"HOME", "PATH", "TMPDIR", "TMP", "TEMP", "SystemRoot", "TS_AUTHKEY"}
    }
    while True:
        process = await asyncio.create_subprocess_exec(
            helper,
            "--mode",
            "gateway",
            "--hostname",
            hostname,
            "--state-dir",
            str(state),
            "--upstream",
            f"http://127.0.0.1:{port}",
            env=env,
            stdin=asyncio.subprocess.PIPE,
        )
        try:
            await process.wait()
        finally:
            await stop(process)
        log.warning("Private Tailscale connection unavailable; retrying in 15s. Website stays up.")
        await asyncio.sleep(15)


async def run():
    helper = os.getenv("TAILSCALE_HELPER") or shutil.which("orchestrator-tunnel")
    if not helper or not Path(helper).is_file():
        raise ValueError("Bundled tunnel missing: build it and set TAILSCALE_HELPER or PATH")
    helper = str(Path(helper).resolve())
    state = Path(os.getenv("TAILSCALE_STATE_DIR", ".local/tailscale/backend")).resolve()
    state.mkdir(parents=True, exist_ok=True, mode=0o700)
    state.chmod(0o700)
    port = int(os.getenv("WORKER_PORT", "8081"))
    if not 1 <= port <= 65535:
        raise ValueError("invalid WORKER_PORT")
    async with asyncio.TaskGroup() as tasks:
        tasks.create_task(service("public"))
        tasks.create_task(service("worker"))
        tasks.create_task(
            tunnel(helper, state, os.getenv("TAILSCALE_HOSTNAME", "orch-backend"), port)
        )


async def supervised():
    task = asyncio.create_task(run())
    loop = asyncio.get_running_loop()
    signals = []
    for sig in (signal.SIGINT, signal.SIGTERM):
        try:
            loop.add_signal_handler(sig, task.cancel)
            signals.append(sig)
        except NotImplementedError:
            pass
    try:
        await task
    except asyncio.CancelledError:
        pass
    finally:
        for sig in signals:
            loop.remove_signal_handler(sig)


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--child", choices=["public", "worker"], help=argparse.SUPPRESS)
    args = parser.parse_args()
    if args.child:
        from .app import public_main, worker_main

        (public_main if args.child == "public" else worker_main)()
        return
    logging.basicConfig(level=logging.INFO)
    try:
        asyncio.run(supervised())
    except KeyboardInterrupt:
        pass


if __name__ == "__main__":
    main()
