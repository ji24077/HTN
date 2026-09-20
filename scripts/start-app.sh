#!/usr/bin/env bash
set -euo pipefail
cd "$(dirname "$0")/.."

if [ ! -f .env ]; then
  echo 'Configure .env using .env.example before starting the app.' >&2
  exit 1
fi

pids=()
cleanup() {
  trap - EXIT INT TERM
  for pid in "${pids[@]}"; do kill "$pid" 2>/dev/null || true; done
  wait || true
}
trap cleanup EXIT INT TERM

uv run --project backend --env-file .env orchestrator-backend &
pids+=("$!")

# Wait for the real database-backed API before displaying the login page.
uv run --project backend --env-file .env python - <<'PY'
import os
import time
import urllib.error
import urllib.request

for setting, default in (("PUBLIC_PORT", "8080"), ("WORKER_PORT", "8081")):
    endpoint = f"http://127.0.0.1:{os.getenv(setting, default)}/healthz"
    deadline = time.monotonic() + 60
    while True:
        try:
            with urllib.request.urlopen(endpoint, timeout=2) as response:
                if response.status == 200:
                    break
        except (urllib.error.URLError, TimeoutError):
            pass
        if time.monotonic() >= deadline:
            raise SystemExit(f"Service did not become healthy: {setting}")
        time.sleep(0.5)
print("Backend ready; starting frontend.", flush=True)
PY
npm --prefix frontend run dev &
pids+=("$!")

# Stop the group if any service exits, rather than leave a partially running app.
while :; do
  for pid in "${pids[@]}"; do
    if ! kill -0 "$pid" 2>/dev/null; then
      echo 'An app service exited; stopping the remaining services.' >&2
      exit 1
    fi
  done
  sleep 1
done
