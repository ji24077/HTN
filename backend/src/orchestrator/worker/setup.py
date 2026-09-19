"""Enroll this computer using an approved Supabase account, then run its worker."""

import argparse
import asyncio
import getpass
import logging
import os
import shutil
import signal
import socket
import ssl
import sys
from pathlib import Path
from urllib.parse import urlsplit
from uuid import uuid4

import httpx

from ..shared.security import credential
from .config import WorkerConfig
from .tunnel import EmbeddedTunnel


class SetupError(Exception):
    pass


def https_origin(value):
    u = urlsplit(value)
    if (
        u.scheme != "https"
        or not u.hostname
        or u.username
        or u.password
        or u.path not in ("", "/")
        or u.query
        or u.fragment
    ):
        raise SetupError("Use the website's HTTPS origin, without a path or credentials")
    _ = u.port
    return value.rstrip("/")


def dotenv_quote(value):
    """Quote for uv's --env-file parser: raw UTF-8, escaping only backslash, quote, and dollar."""
    if any(ch in value for ch in "\r\n\0"):
        raise SetupError("Worker configuration values cannot contain line breaks")
    escaped = value.replace("\\", "\\\\").replace('"', '\\"').replace("$", "\\$")
    return '"' + escaped + '"'


def enrollment_error(response):
    messages = {
        401: "Session expired; sign in again",
        403: "This account is not approved to enroll workers",
        409: "Enrollment request already used; start a new setup attempt",
        429: "Enrollment limit reached; contact the fleet administrator",
    }
    if response.status_code == 503:
        # The backend also answers 503 while its database is unreachable; only the
        # explicit code means enrollment is switched off.
        try:
            code = response.json()["detail"]["code"]
        except (ValueError, KeyError, TypeError):
            code = None
        if code == "enrollment_not_configured":
            return "Automatic enrollment is not configured on this backend yet"
        return "Backend is temporarily unavailable; try again later"
    return messages.get(response.status_code, "Enrollment failed; check the backend configuration")


async def login_and_enroll(client, origin, email, password, name, request_id):
    """Sign in and enroll; returns the enrollment data and the session used, for cleanup."""
    response = await client.get(origin + "/auth/config")
    if response.status_code != 200:
        raise SetupError("Cannot load sign-in configuration from this website")
    data = response.json()
    auth_origin = https_origin(data["url"])
    key = data["publishableKey"]
    if not isinstance(key, str) or not key.startswith("sb_publishable_"):
        raise SetupError("Website returned invalid sign-in configuration")
    response = await client.post(
        auth_origin + "/auth/v1/token?grant_type=password",
        headers={"apikey": key},
        json={"email": email, "password": password},
    )
    if response.status_code != 200:
        raise SetupError(
            "Sign-in failed. Check your password and confirmed email; MFA/CAPTCHA accounts require browser onboarding (not yet supported by this CLI)."
        )
    access_token = response.json().get("access_token")
    if not isinstance(access_token, str) or not access_token:
        raise SetupError("Sign-in did not return a session")
    response = await client.post(
        origin + "/v1/worker-enrollments",
        headers={"Authorization": f"Bearer {access_token}"},
        json={"request_id": request_id, "name": name},
    )
    if response.status_code != 201:
        raise SetupError(enrollment_error(response))
    return response.json(), access_token


async def withdraw_enrollment(client, origin, session, worker_id):
    """Best effort: release an enrollment this computer could not use, then tell the user."""
    try:
        response = await client.delete(
            origin + "/v1/worker-enrollments/" + worker_id,
            headers={"Authorization": f"Bearer {session}"},
        )
        released = response.status_code == 204
    except httpx.HTTPError:
        released = False
    if released:
        print(
            "Setup failed after enrolling; the unused enrollment was withdrawn. "
            "Re-run setup to try again.",
            file=sys.stderr,
        )
    else:
        print(
            f"Setup failed after enrolling, and enrollment {worker_id} could not be withdrawn.",
            file=sys.stderr,
        )
    return released


async def finish_cleanup(coro, seconds=15):
    """Run cleanup to completion despite further Ctrl-C presses, but never beyond the bound."""
    loop = asyncio.get_running_loop()
    task = loop.create_task(coro)
    deadline = loop.time() + seconds

    def interrupted(*_):
        print("Finishing cleanup; please wait…", file=sys.stderr, flush=True)

    # asyncio.run turns the first Ctrl-C into task cancellation but raises KeyboardInterrupt
    # on the second, which would tear the loop down mid-request; hold SIGINT until done.
    try:
        previous = signal.signal(signal.SIGINT, interrupted)
    except ValueError:
        previous = None  # Not the main thread: only task cancellation can arrive.
    try:
        while not task.done():
            try:
                await asyncio.wait_for(asyncio.shield(task), max(0.0, deadline - loop.time()))
            except asyncio.CancelledError:
                # Absorb a cancellation; the caller re-raises its original interruption
                # once cleanup settles, and the count must match what asyncio.run expects.
                asyncio.current_task().uncancel()
            except TimeoutError:
                task.cancel()
                break
    finally:
        if previous is not None:
            signal.signal(signal.SIGINT, previous)
    if not task.done() or task.cancelled() or task.exception() is not None:
        return False
    return task.result()


def keep_profile(fd, output, profile, worker_id):
    """Withdrawal did not complete, so keep the credential rather than lose it."""
    try:
        # Write only through the descriptor reserved with O_EXCL, never by pathname, and
        # only while that pathname still names the reserved file.
        reserved, current = os.fstat(fd), os.lstat(output)
        if (current.st_dev, current.st_ino) != (reserved.st_dev, reserved.st_ino):
            raise OSError("worker profile path was replaced")
        os.ftruncate(fd, 0)
        os.lseek(fd, 0, os.SEEK_SET)
        with os.fdopen(os.dup(fd), "w", encoding="utf-8") as stream:
            stream.write(profile)
    except OSError:
        print(
            f"Could not keep {output}; note worker {worker_id} for an administrator.",
            file=sys.stderr,
        )
        return False
    print(
        f"Kept {output} because withdrawing worker {worker_id} did not complete, so it may "
        "still be enrolled. Running orchestrator-worker with this profile retries the join "
        "(Tailscale will show a sign-in link) if it is; otherwise ask a fleet administrator "
        "to check that worker.",
        file=sys.stderr,
    )
    return True


def reserve_profile(output):
    """Exclusively create a profile and retain its identity until setup finishes."""
    if os.name != "nt":
        return os.open(output, os.O_WRONLY | os.O_CREAT | os.O_EXCL, 0o600)

    import ctypes
    import msvcrt
    from ctypes import wintypes

    kernel = ctypes.WinDLL("kernel32", use_last_error=True)
    create = kernel.CreateFileW
    create.argtypes = [
        wintypes.LPCWSTR,
        wintypes.DWORD,
        wintypes.DWORD,
        ctypes.c_void_p,
        wintypes.DWORD,
        wintypes.DWORD,
        wintypes.HANDLE,
    ]
    create.restype = wintypes.HANDLE
    # CREATE_NEW is O_EXCL. Request DELETE access so cleanup can delete this exact
    # file by handle, while withholding FILE_SHARE_DELETE prevents path replacement.
    handle = create(str(output), 0x40000000 | 0x00010000, 0x1 | 0x2, None, 1, 0x80, None)
    if handle == ctypes.c_void_p(-1).value:
        raise ctypes.WinError(ctypes.get_last_error())
    try:
        return msvcrt.open_osfhandle(handle, os.O_WRONLY | os.O_BINARY)
    except BaseException:
        close = kernel.CloseHandle
        close.argtypes = [wintypes.HANDLE]
        close.restype = wintypes.BOOL
        close(handle)
        raise


def discard_profile(fd, output):
    """Remove only our reserved file; Windows deletes it when its handle closes."""
    try:
        reserved, current = os.fstat(fd), os.lstat(output)
    except OSError:
        return
    if (current.st_dev, current.st_ino) != (reserved.st_dev, reserved.st_ino):
        return
    if os.name != "nt":
        output.unlink(missing_ok=True)
        return

    import ctypes
    import msvcrt
    from ctypes import wintypes

    kernel = ctypes.WinDLL("kernel32", use_last_error=True)
    dispose = kernel.SetFileInformationByHandle
    dispose.argtypes = [wintypes.HANDLE, ctypes.c_int, ctypes.c_void_p, wintypes.DWORD]
    dispose.restype = wintypes.BOOL
    delete = ctypes.c_bool(True)  # FILE_DISPOSITION_INFO.DeleteFile is a BOOLEAN.
    if not dispose(msvcrt.get_osfhandle(fd), 4, ctypes.byref(delete), ctypes.sizeof(delete)):
        raise ctypes.WinError(ctypes.get_last_error())


def worker_environment(data, directory, helper):
    worker_id = data["worker_id"]
    # Validate before using remotely supplied names in filesystem paths.
    if (
        not isinstance(worker_id, str)
        or not worker_id.startswith("worker-")
        or not worker_id[7:].isalnum()
        or len(worker_id) > 64
    ):
        raise SetupError("Invalid worker identity from backend")
    auth_key = data["tailscale_auth_key"]
    if not isinstance(auth_key, str) or not auth_key.startswith("tskey-auth-"):
        raise SetupError("Invalid Tailscale enrollment key from backend")
    hostname = "orch-" + worker_id
    return {
        "WORKER_ID": worker_id,
        "WORKER_TOKEN": credential(data["worker_token"]),
        "WORKER_TRANSPORT": "tailscale",
        "SERVER_URL": data["server_url"],
        "WORKER_RUNTIME": "cpu",
        "WORKER_VRAM_MIB": "0",
        "WORKER_PAUSED": "false",
        "TAILSCALE_HELPER": str(Path(helper).resolve()),
        "TAILSCALE_HOSTNAME": hostname,
        "TAILSCALE_STATE_DIR": str(directory / "tailscale" / worker_id),
    }


async def setup(args):
    origin = https_origin(args.server)
    helper = args.helper or os.getenv("TAILSCALE_HELPER") or shutil.which("orchestrator-tunnel")
    if not helper:
        raise SetupError("Bundled tunnel missing. Specify --helper .local/bin/orchestrator-tunnel")
    output = Path(args.output).expanduser().absolute()
    # Reject unrepresentable paths before querying the filesystem: Windows rejects
    # newline filenames itself, but setup should report the same error on every OS.
    for local in (str(Path(helper).resolve()), str(output.parent.resolve())):
        dotenv_quote(local)
    if not Path(helper).is_file():
        raise SetupError("Bundled tunnel missing. Specify --helper .local/bin/orchestrator-tunnel")
    if output.exists() or output.is_symlink():
        raise SetupError(
            "Worker config already exists; choose a new --output or run the existing worker"
        )
    output.parent.mkdir(parents=True, exist_ok=True, mode=0o700)
    # Reserve the file before creating an enrollment; never overwrite an existing identity.
    fd = reserve_profile(output)
    saved = False
    try:
        email = args.email or input("Supabase account email: ").strip()
        password = getpass.getpass("Supabase account password: ")
        context = ssl.create_default_context()
        if args.ca_file:
            context.load_verify_locations(cafile=args.ca_file)
        async with httpx.AsyncClient(verify=context, follow_redirects=False, timeout=60) as client:
            data, session = await login_and_enroll(
                client, origin, email, password, args.name, str(uuid4())
            )
            password = None
            env = worker_environment(data, output.parent.resolve(), helper)
            profile = None
            try:
                profile = "".join(k + "=" + dotenv_quote(v) + "\n" for k, v in env.items())
                # Scope the auth key to initial enrollment; don't save it in the profile.
                previous = {k: os.environ.get(k) for k in [*env, "TS_AUTHKEY"]}
                os.environ.update(env, TS_AUTHKEY=data["tailscale_auth_key"])
                try:
                    config = WorkerConfig.from_env("stub")
                    print("Enrolling this computer into the private worker network…", flush=True)
                    async with EmbeddedTunnel(config):
                        pass
                finally:
                    for k, v in previous.items():
                        if v is None:
                            os.environ.pop(k, None)
                        else:
                            os.environ[k] = v
                data.clear()
                # Keep the reserved descriptor itself for recovery; write through a duplicate.
                with os.fdopen(os.dup(fd), "w", encoding="utf-8") as stream:
                    stream.write(profile)
                saved = True
            except BaseException:
                # The backend already activated this identity and a node that never joined
                # cannot retry from the profile alone, so release it while still signed in.
                released = await finish_cleanup(
                    withdraw_enrollment(client, origin, session, env["WORKER_ID"])
                )
                if not released and profile is not None:
                    saved = keep_profile(fd, output, profile, env["WORKER_ID"])
                raise
            session = None
        print(f"Enrolled {env['WORKER_ID']}. Saved private worker configuration to {output}.")
        if not args.no_start:
            from .agent import run_worker

            os.environ.update(env)
            os.environ.pop("TS_AUTHKEY", None)
            print(
                "Starting worker. Ctrl-C stops it; its enrolled identity is retained.", flush=True
            )
            await run_worker()
    finally:
        try:
            if not saved:
                # Keep the descriptor open through the ownership check. On Windows,
                # pathname unlink fails while open; deletion by handle avoids both
                # that failure and any close-then-unlink race against a replacement.
                discard_profile(fd, output)
        finally:
            os.close(fd)


def main():
    logging.basicConfig(level=logging.INFO, format="%(asctime)s %(levelname)s %(name)s %(message)s")
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--server", required=True, help="website HTTPS origin")
    parser.add_argument("--email", help="approved Supabase account email")
    parser.add_argument("--name", default=socket.gethostname()[:80], help="computer label")
    parser.add_argument("--output", default=".local/worker.env", help="new private config file")
    parser.add_argument("--helper", help="path to bundled orchestrator-tunnel")
    parser.add_argument("--ca-file", help="additional trusted CA for a local HTTPS website")
    parser.add_argument(
        "--no-start", action="store_true", help="enroll and save without running jobs"
    )
    args = parser.parse_args()
    try:
        asyncio.run(setup(args))
    except KeyboardInterrupt:
        pass
    except SetupError as exc:
        print(str(exc), file=sys.stderr)
        raise SystemExit(1) from None
    except (httpx.HTTPError, OSError, ValueError, KeyError, TypeError, RuntimeError):
        print(
            "Setup failed. Check connectivity, trusted HTTPS, and backend configuration. No credentials were printed.",
            file=sys.stderr,
        )
        raise SystemExit(1) from None


if __name__ == "__main__":
    main()
