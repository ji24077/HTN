#!/usr/bin/env bash
#
# Start the control service the paired machines actually talk to.
#
#   ./scripts/start-fleet.sh
#
# Everything this needs is discoverable, so nothing has to be remembered or typed. That
# is the point: the last three attempts to bring this back up failed on a hand-typed
# connection string, a `read` that swallowed a pasted line, and an admin token that only
# ever existed in a shell that had since closed.
#
# What it settles, and why each one bit before:
#
#   DATABASE_URL   ethan_main, not dwp. The `dwp` database on the same server belongs to
#                  the old TypeScript control plane (hosts, pair_codes, sessions); this
#                  backend's tables are dwp_devices / workers / execution_events, and
#                  pointing it at the wrong one comes up with an empty device table and
#                  every machine failing to authenticate.
#   ADMIN_TOKEN    read from ~/.dwp/admin-token, generated on first run. It is not
#                  issued by anything -- whatever is in the environment *is* the token --
#                  so keeping it in a file is what makes it survive closing the terminal.
#   DEMO_UI        what serves the dashboard at /. Without it the backend answers 404
#                  there, which reads as a broken server rather than a disabled page.
#   PUBLIC_ORIGIN  unset deliberately. This entry point runs the combined surface, which
#                  refuses to start if that variable is set -- and it *is* set in .env,
#                  for the other control plane.
set -euo pipefail
cd "$(dirname "$0")/.."

DB_CONTAINER=${DWP_DB_CONTAINER:-dwp-db}
DB_NAME=${DWP_DB_NAME:-ethan_main}
TOKEN_FILE=${DWP_ADMIN_TOKEN_FILE:-$HOME/.dwp/admin-token}
SERVER=backend/.venv/bin/orchestrator-server

[ -x "$SERVER" ] || { echo "No backend at $SERVER — run: uv sync --project backend" >&2; exit 1; }

if ! docker inspect "$DB_CONTAINER" >/dev/null 2>&1; then
  echo "The database container '$DB_CONTAINER' is not running." >&2
  echo "Start it, or set DWP_DB_CONTAINER to whichever one holds '$DB_NAME'." >&2
  exit 1
fi

# Read at run time rather than stored anywhere: the password belongs to the container,
# and a copy in a file here would be one more thing to rotate and leak.
password=$(docker inspect "$DB_CONTAINER" \
  --format '{{range .Config.Env}}{{println .}}{{end}}' \
  | grep '^POSTGRES_PASSWORD=' | cut -d= -f2-)
[ -n "$password" ] || { echo "No POSTGRES_PASSWORD in container '$DB_CONTAINER'." >&2; exit 1; }

# The token has to outlive the terminal it was first typed into. Generated once, kept
# 0600 beside the agent's other keys.
if [ ! -f "$TOKEN_FILE" ]; then
  mkdir -p "$(dirname "$TOKEN_FILE")"
  ( umask 077; python3 -c 'import secrets;print(secrets.token_urlsafe(32))' > "$TOKEN_FILE" )
  echo "  Created an admin token at $TOKEN_FILE"
fi
chmod 600 "$TOKEN_FILE"
token=$(cat "$TOKEN_FILE")

port=${LISTEN_PORT:-8080}

# An already-running instance is the common case, not an error worth a stack trace.
#
# uvicorn reports this as "[Errno 48] address already in use" *after* printing
# "Application startup complete", so it reads as a server that started and then broke.
# It did not start at all, and the one already listening is almost always an older copy
# of this same service that nobody remembers leaving running.
holder=$(lsof -nP -iTCP:"$port" -sTCP:LISTEN -t 2>/dev/null | head -1 || true)
if [ -n "$holder" ]; then
  what=$(ps -o command= -p "$holder" 2>/dev/null | cut -c1-90)
  if [ "${1:-}" = "--replace" ]; then
    echo "  Stopping the copy already on port $port (pid $holder)"
    kill "$holder" 2>/dev/null || true
    for _ in $(seq 1 25); do
      lsof -nP -iTCP:"$port" -sTCP:LISTEN -t >/dev/null 2>&1 || break
      sleep 0.2
    done
    if lsof -nP -iTCP:"$port" -sTCP:LISTEN -t >/dev/null 2>&1; then
      echo "  It did not stop. Kill it by hand:  kill -9 $holder" >&2
      exit 1
    fi
  else
    cat >&2 <<STOP

  Port $port is already held by pid $holder:
      $what

  That is probably an older copy of this service. To stop it and take over:

      ./scripts/start-fleet.sh --replace

  Paired machines reconnect by themselves within about 30 seconds.

STOP
    exit 1
  fi
fi

cat <<BANNER

  Dashboard   http://127.0.0.1:${port}/
  Database    ${DB_NAME}
  Admin token ${TOKEN_FILE}   (the dashboard signs you in by cookie on localhost)

  Ctrl-C to stop.

BANNER

# Trust forwarded headers from loopback only.
#
# This is the entry point that `scripts/share.ts` funnels, and Tailscale Funnel
# terminates TLS then forwards plain HTTP from 127.0.0.1. Without this the server
# believes it is serving http://, so every agent that pairs through the funnel is handed
# a ws:// URL on port 80 and can never connect -- measured, with ECONNREFUSED on a
# machine that had paired perfectly a second earlier.
#
# Loopback only: a machine on the LAN connects from its own address and cannot spoof
# these headers. Anything already on this host can read the admin token beside this
# script, so it gains nothing.
exec env -u PUBLIC_ORIGIN \
  TRUSTED_PROXY_IPS="${TRUSTED_PROXY_IPS:-127.0.0.1}" \
  DWP_AGENT_IMAGE="${DWP_AGENT_IMAGE:-}" \
  DEMO_UI=true \
  LISTEN_HOST="${LISTEN_HOST:-127.0.0.1}" \
  LISTEN_PORT="$port" \
  ADMIN_TOKEN="$token" \
  DATABASE_URL="postgres://dwp:${password}@localhost:5433/${DB_NAME}" \
  "$SERVER"
