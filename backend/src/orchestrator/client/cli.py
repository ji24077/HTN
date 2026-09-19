"""JSON command line interface for humans, scripts, and shell-capable agents."""

import argparse
import asyncio
import ipaddress
import os
import sys
from pathlib import Path
from urllib.parse import urlsplit

from ..shared.protocol import json_loads, json_text
from .http import OrchestratorClient, validate_url
from .tools import AgentTools, tool_definitions


class Parser(argparse.ArgumentParser):
    def error(self, message):
        raise ValueError(message)


def parser() -> Parser:
    root = Parser(description=__doc__)
    root.add_argument(
        "--url", help="Control plane URL; defaults to ORCHESTRATOR_URL or localhost:8080"
    )
    root.add_argument(
        "--demo", action="store_true", help="Use local .demo/config.json credentials and port 8787"
    )
    commands = root.add_subparsers(dest="command", required=True)
    commands.add_parser(
        "tools", help="Print tool names and JSON input schemas; no connection needed"
    )
    commands.add_parser("workers", help="List worker states and capabilities")
    commands.add_parser("tasks", help="List recent tasks")
    for name in ("task", "cancel", "wait"):
        command = commands.add_parser(name)
        command.add_argument("task_id")
        if name == "wait":
            command.add_argument(
                "--timeout",
                type=float,
                default=300,
                help="Wait deadline in seconds; does not cancel work",
            )
    command = commands.add_parser(
        "submit", help="Submit a JSON {tasks:[...]} file; '-' reads stdin"
    )
    command.add_argument("file")
    command = commands.add_parser("events")
    command.add_argument("--after", type=int, default=0)
    command = commands.add_parser("call", help="Dispatch a named tool with JSON arguments")
    command.add_argument("name")
    command.add_argument("--input", default="-", help="JSON arguments file; default stdin")
    return root


def read_json(path: str):
    if path == "-":
        data = sys.stdin.buffer.read(1024 * 1024 + 1)
    else:
        with Path(path).open("rb") as stream:
            data = stream.read(1024 * 1024 + 1)
    if len(data) > 1024 * 1024:
        raise ValueError("Input exceeds 1 MiB")
    return json_loads(data)


def connection(args) -> tuple[str, str]:
    url = (
        args.url
        or os.getenv("ORCHESTRATOR_URL")
        or ("http://127.0.0.1:8787" if args.demo else "http://127.0.0.1:8080")
    )
    url = validate_url(url)
    if args.demo:
        host = urlsplit(url).hostname
        try:
            local = ipaddress.ip_address(host).is_loopback
        except ValueError:
            local = host == "localhost"
        if not local:
            raise ValueError("--demo credentials may only be sent to loopback")
        token = json_loads(Path(".demo/config.json").read_bytes())["admin"]
    else:
        token = os.getenv("ORCHESTRATOR_TOKEN") or os.getenv("ADMIN_TOKEN", "")
    if not token:
        raise ValueError("Set ORCHESTRATOR_TOKEN (or ADMIN_TOKEN), or use --demo locally")
    return url, token


async def run(args) -> dict:
    if args.command == "tools":
        return {"ok": True, "result": tool_definitions()}
    arguments = {}
    match args.command:
        case "workers":
            name = "list_workers"
        case "tasks":
            name = "list_tasks"
        case "task" | "cancel" | "wait":
            name = {"task": "get_task", "cancel": "cancel_task", "wait": "wait_task"}[args.command]
            arguments = {"task_id": args.task_id}
            if args.command == "wait":
                arguments["timeout_seconds"] = args.timeout
        case "submit":
            name, arguments = "submit_tasks", read_json(args.file)
        case "events":
            name, arguments = "list_events", {"after": args.after}
        case "call":
            name, arguments = args.name, read_json(args.input)
    url, token = connection(args)
    async with OrchestratorClient(url, token) as client:
        return await AgentTools(client).call(name, arguments)


def main() -> None:
    try:
        result = asyncio.run(run(parser().parse_args()))
    except (ValueError, OSError, KeyError) as exc:
        result = {"ok": False, "error": {"code": "configuration_or_input", "message": str(exc)}}
    except KeyboardInterrupt:
        result = {
            "ok": False,
            "error": {
                "code": "interrupted",
                "message": "Client interrupted; tasks were not cancelled",
            },
        }
        print(json_text(result), flush=True)
        raise SystemExit(130)
    print(json_text(result), flush=True)
    raise SystemExit(0 if result["ok"] else 1)


if __name__ == "__main__":
    main()
