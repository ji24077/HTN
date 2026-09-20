#!/usr/bin/env bash
# N-1 (Jack). Run at home AND again at the venue.
# RELAY is an emergency: 2-35 Mbit/s. Tell Ji so the router raises H.
set -u
echo "=== tailscale peers ==="
tailscale status --json | jq -r '
  .Peer[] | "\(.HostName)\t\(.TailscaleIPs[0])\t\(if .Relay=="" then "DIRECT" else "RELAY:"+.Relay end)"
' | column -t
echo
echo "=== bandwidth (pass a peer 100.x ip) ==="
[ $# -ge 1 ] && iperf3 -c "$1" -t 5 || echo "usage: $0 <peer-100.x-ip>"
