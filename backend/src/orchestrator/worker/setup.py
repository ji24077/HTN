"""Enroll this computer using an approved Supabase account, then run its worker."""

import argparse
import asyncio
import getpass
import json
import logging
import os
import shutil
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


async def login_and_enroll(client, origin, email, password, name, request_id):
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
    messages = {
        401: "Session expired; sign in again",
        403: "This account is not approved to enroll workers",
        409: "Enrollment request already used; start a new setup attempt",
        429: "Enrollment limit reached; contact the fleet administrator",
        503: "Automatic enrollment is not configured on this backend yet",
    }
    if response.status_code != 201:
        raise SetupError(
            messages.get(response.status_code, "Enrollment failed; check the backend configuration")
        )
    return response.json()


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
    if not helper or not Path(helper).is_file():
        raise SetupError("Bundled tunnel missing. Specify --helper .local/bin/orchestrator-tunnel")
    output = Path(args.output).expanduser().absolute()
    if output.exists() or output.is_symlink():
        raise SetupError(
            "Worker config already exists; choose a new --output or run the existing worker"
        )
    output.parent.mkdir(parents=True, exist_ok=True, mode=0o700)
    # Reserve the file before creating an enrollment; never overwrite an existing identity.
    fd = os.open(output, os.O_WRONLY | os.O_CREAT | os.O_EXCL, 0o600)
    saved = False
    try:
        email = args.email or input("Supabase account email: ").strip()
        password = getpass.getpass("Supabase account password: ")
        context = ssl.create_default_context()
        if args.ca_file:
            context.load_verify_locations(cafile=args.ca_file)
        async with httpx.AsyncClient(verify=context, follow_redirects=False, timeout=60) as client:
            data = await login_and_enroll(client, origin, email, password, args.name, str(uuid4()))
        password = None
        env = worker_environment(data, output.parent.resolve(), helper)
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
        with os.fdopen(fd, "w") as stream:
            fd = None
            for k, v in env.items():
                stream.write(k + "=" + json.dumps(v) + "\n")
        saved = True
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
        if fd is not None:
            os.close(fd)
        if not saved:
            output.unlink(missing_ok=True)


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
