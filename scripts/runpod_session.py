"""Local-only prepare; explicitly invoked create/cleanup use authenticated REST v2.

Plan JSON fields: budget_usd <= 15, max_duration_seconds <= 5400,
storage_allowance_per_hour >= 0.10. The default migration profile requires
exactly source and target entries. The explicit inference-latency profile adds
an amd entry and uses a Community 3090 source, Secure 4090 target and Secure
MI300X amd pod at the fixed approved rates.
The inference-latency-3090, inference-latency-4090 and inference-latency-amd
profiles run just that approved GPU so capacity waits need not hold other pods.
Each entry: role, hourly_gpu_usd, quoted_at_utc (ISO 8601 with timezone),
available: true, payload (REST v2 Pod body). Names must equal
gpushare-<session-id>-<role>. In the default migration profile, source is
RTX 4090 and target is MI300X, both Secure EU-RO-1.

prepare --session-id ID --plan FILE   # no network; starts fixed deadline
watchdog --session-id ID             # independent process; start before create
create --session-id ID --plan FILE   # fresh price/stock checks, then create once
status --session-id ID               # safe local journal, no env/credentials
cleanup --session-id ID              # own Pods only; verify termination

Global option --journal-dir defaults to ignored .gpushare/runpod-sessions.
RUNPOD_API_KEY must be loaded externally for network commands. The watchdog is
local and best-effort: no server-native TTL exists in the inspected API/CLI.
Machine sleep, power loss, or network outage can delay deletion and exceed the
estimate. Main controller should invoke cleanup in its own finally block too.
"""

from __future__ import annotations

import argparse
import json
import os
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[1] / "src"))

from gpushare.agent.runpod_session import RunpodClient, Session, SessionError  # noqa: E402


def main():
    parser = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    parser.add_argument("command", choices=["prepare", "watchdog", "create", "status", "cleanup"])
    parser.add_argument("--session-id", required=True)
    parser.add_argument("--plan", type=Path)
    parser.add_argument("--journal-dir", type=Path, default=Path(".gpushare/runpod-sessions"))
    parser.add_argument("--poll-seconds", type=float, default=5)
    args = parser.parse_args()
    try:
        client = None if args.command in {"prepare", "status"} else RunpodClient(os.environ.get("RUNPOD_API_KEY", ""))
        session = Session(args.journal_dir, args.session_id, client)
        if args.command in {"prepare", "create"}:
            if args.plan is None:
                raise SessionError("--plan is required for prepare/create")
            plan = json.loads(args.plan.read_text(encoding="utf-8-sig"))
            result = getattr(session, args.command)(plan)
        elif args.command == "watchdog":
            result = session.watchdog(poll_seconds=args.poll_seconds)
        else:
            result = session.read() if args.command == "status" else session.cleanup()
        print(json.dumps(result, indent=2))
        return 1 if result["state"] == "needs_cleanup" else 0
    except (SessionError, OSError, ValueError) as exc:
        # SessionError text is controlled; never echo raw API/payload data.
        message = str(exc) if isinstance(exc, SessionError) else "Local plan/journal operation failed"
        print(json.dumps({"error": message}), file=sys.stderr)
        return 1


if __name__ == "__main__":
    raise SystemExit(main())
