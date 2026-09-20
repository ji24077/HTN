# The Windows agent is hard to start and does not stay connected

**Status:** open. Observed 2026-09-19 over roughly two hours on one Windows 11 laptop.
**Impact:** the machine took about two hours and a dozen attempts to join, and dropped off
again within minutes. It is currently offline.

## What happened

The laptop (`LAPTOP-9CG858LS`, device `4ec957dd`) paired successfully at 13:27, first
connected at 14:15, and was offline again by 14:26. Everything below is a separate
obstacle hit on the way there.

## The obstacles, in the order they appeared

**1. Two executables with different names.** The one-line installer writes
`%USERPROFILE%\.dwp\bin\dwp-agent.exe`. The app zip extracts `DWP Agent\DWP Agent.exe`.
Searching for one never finds the other, and they can be different versions — the machine
had v0.3.0 from the installer while the zip held v0.4.0. Whichever is *running* serves
the window, so replacing the app appeared to do nothing.

**2. Process names differ too.** `Get-Process dwp-agent` does not match `DWP Agent`, so
"kill the old one before starting the new one" silently killed nothing.

**3. The pairing hint is wrong for binary installs.** After pairing, the agent printed

```
Start it with:  pnpm agent run
```

on a machine with no pnpm and no repository. Cause: `isCompiledBinary()` matched Bun's
virtual-filesystem marker as `$bunfs` or `B:/~BUN` and a backslashed Windows spelling
slipped past both, so a compiled binary believed it was running from source. Fixed in
`95ddfd0`, but the machine in question is still running the old binary.

**4. Nothing keeps it alive.** The agent was started as `& $exe run` in a PowerShell
window. Closing that window, or the laptop sleeping, ends it. There is no service
installed, and nothing in the flow suggests installing one. Every disconnection observed
on this machine was a process ending, not a network fault.

## What is not the problem

**Not connectivity.** Once actually started it connected in 511 ms
(`dialMs=511`) over `wss://` through Funnel, completed the handshake, and ran a signed
task. The network path works.

**Not pairing.** It enrolled cleanly and pinned a release key.

## Where to look

- `packages/control/src/installer.ts` and `scripts/lib/apps.ts` — the two install routes
  produce different paths and different executable names with no relationship between
  them. Decide whether the app should adopt the installer's location, or the installer
  should stop existing on Windows.
- `packages/agent/src/service.ts` — `install-service` exists and uses a Scheduled Task on
  Windows. Nothing in the join flow offers it. A worker that only runs while a terminal
  is open is not a worker anyone can rely on, and the GUI's "Start automatically when I
  log in" toggle is the only place it surfaces.
- Whether `run` should refuse to start when another agent for the same host id is already
  connected, rather than both racing.

## How to reproduce

On a clean Windows machine, install via the one-line installer, then download and unzip
the app, then try to move the machine to a different server. The path and process-name
mismatches surface immediately.
