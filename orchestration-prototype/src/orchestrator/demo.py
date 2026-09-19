"""Local demo launcher: real PostgreSQL, the control plane, and separate workers."""

import argparse
import json
import os
import secrets
import sys
from importlib.resources import files
from pathlib import Path


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("service", choices=["server", "worker-a", "worker-b"])
    parser.add_argument("--port", type=int, default=8787)
    args = parser.parse_args()
    folder = Path(".demo").resolve()
    folder.mkdir(mode=0o700, exist_ok=True)
    config_file = folder / "config.json"
    if args.service == "server":
        if not files("orchestrator.server").joinpath("web", "index.html").is_file():
            sys.exit(
                "Build the dashboard first: npm --prefix frontend ci && npm --prefix frontend run build"
            )
        if not config_file.exists():
            config = {
                "admin": secrets.token_urlsafe(32),
                "workers": {name: secrets.token_urlsafe(32) for name in ("worker-a", "worker-b")},
            }
            with config_file.open("x") as stream:
                os.chmod(config_file, 0o600)
                json.dump(config, stream)
        config = json.loads(config_file.read_text())
        try:
            import pgserver
        except ImportError:
            sys.exit("Run with: uv run --python 3.12 --extra demo orchestrator-demo server")
        database = pgserver.get_server(folder / "postgres", cleanup_mode="stop")
        os.environ.update(
            {
                "DATABASE_URL": database.get_uri(),
                "REDIS_URL": "",
                "ADMIN_TOKEN": config["admin"],
                "WORKER_TOKENS": json.dumps(config["workers"]),
                "LISTEN_HOST": "127.0.0.1",
                "LISTEN_PORT": str(args.port),
                "DEMO_UI": "true",
            }
        )
        print(f"\nDashboard: http://127.0.0.1:{args.port}\n", flush=True)
        from .server.app import main as run_server

        try:
            run_server()
        finally:
            database.cleanup()
    else:
        if not config_file.exists():
            sys.exit("Start orchestrator-demo server first.")
        config = json.loads(config_file.read_text())
        os.environ.update(
            {
                "WORKER_ID": args.service,
                "WORKER_TOKEN": config["workers"][args.service],
                "SERVER_URL": f"ws://127.0.0.1:{args.port}/v1/worker",
                "WORKER_RUNTIME": "cpu",
                "WORKER_VRAM_MIB": "0",
                "WORKER_PAUSED": "false",
            }
        )
        from .worker.agent import main as run_worker

        run_worker()


if __name__ == "__main__":
    main()
