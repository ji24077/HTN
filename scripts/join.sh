#!/usr/bin/env bash
# One command for someone joining a friend's network.
#
#   ./scripts/join.sh https://their-address ABCD-1234
#
# Checks what is needed, installs what it can, pairs, and starts taking work.
set -euo pipefail

SERVER="${1:-}"
CODE="${2:-}"

if [ -z "$SERVER" ] || [ -z "$CODE" ]; then
  echo "Usage: ./scripts/join.sh <server-url> <pairing-code>"
  echo "Both come from the invite link you were sent:"
  echo "    https://SERVER/join?code=CODE"
  exit 1
fi

echo
echo "  Joining ${SERVER}"
echo

if ! command -v node >/dev/null 2>&1; then
  echo "  Node.js is not installed."
  echo "  Install Node 24 or newer from https://nodejs.org and run this again."
  exit 1
fi

NODE_MAJOR="$(node -p 'process.versions.node.split(".")[0]')"
if [ "$NODE_MAJOR" -lt 24 ]; then
  echo "  Node.js $(node -v) is too old — version 24 or newer is required."
  echo "  Update from https://nodejs.org and run this again."
  exit 1
fi
echo "  ok  Node.js $(node -v)"

if ! command -v pnpm >/dev/null 2>&1; then
  echo "  installing pnpm..."
  npm install -g pnpm >/dev/null 2>&1 || {
    echo "  Could not install pnpm automatically. Run:  npm install -g pnpm"
    exit 1
  }
fi
echo "  ok  pnpm $(pnpm --version)"

cd "$(dirname "$0")/.."

if [ ! -d node_modules ]; then
  echo "  installing dependencies (one time, may take a minute)..."
  pnpm install --silent
fi
echo "  ok  dependencies"
echo

# Some networks resolve established domains but not freshly created ones; the agent
# falls back to public resolvers when that happens.
export DWP_DNS_FALLBACK="${DWP_DNS_FALLBACK:-1}"

node packages/agent/src/index.ts pair --server "$SERVER" --code "$CODE"

echo
echo "  Starting. Leave this window open — press Ctrl-C to stop at any time."
echo "  To stop taking work without closing:  pnpm agent pause"
echo
exec node packages/agent/src/index.ts run
