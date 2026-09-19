# A desktop app for Mac and Windows, without paying

**Status:** built, and running on both. Verified end to end on macOS **and on real
Windows hardware** — the first time anything in this project has.
**Built:** 2026-09-19
**Command:** `pnpm build:binaries` — produces the binaries *and* both apps, signed together.

## What exists

| | |
|---|---|
| `DWP-Agent-macOS.zip` | 45 MB. One `.app` for Apple Silicon and Intel. |
| `DWP-Agent-Windows-x64.zip` | 79 MB. `DWP Agent.exe`, plus a console build as a fallback. |
| Served from | `GET /download/<file>`, listed on `/join` with hashes and the warning walkthrough |
| Updates | over the connection, signed, automatic — the same mechanism as the CLI install |

Open it, paste the invite link, done. The window shows connection state, what it is
running, and three controls: pause/resume, run at login, quit.

## The decision, and the measurements behind it

The plan called for a Tauri shell spawning the agent binary. That was tested and
rejected, on two facts found by running things rather than reading about them:

1. **Rust is not installed, and Tauri cannot cross-compile to Windows from macOS.** The
   entire distribution story here is "build on the Mac, hand someone a zip". A shell that
   needs a Windows machine or a CI matrix to produce the Windows half defeats it.
2. **Electron would have cost more than the payload.** ~100 MB of runtime per platform on
   top of a 111 MB agent, a large new dependency tree, and — the part that actually
   decides it — a second artefact with its own version, which has to be kept in step with
   the agent by hand. `Capability.swift:6` is what that drift looks like in this repo
   already.

**So the app is the agent.** `dwp-agent gui` starts the same process that holds the
connection and adds a loopback HTTP server and a window onto it. The window is a browser
window, opened chromeless via `--app=` where a Chromium exists — Edge ships with Windows,
so that is the common case there; a Mac without Chrome or Edge falls back to an ordinary
tab.

What that buys: no toolchain, no runtime, one file to ship, and **an app and an agent
that cannot disagree about their version, because they are the same bytes.** What it
costs, stated plainly: no tray icon, and on some Macs the UI is a browser tab rather than
a window.

## What Bun will and will not do when cross-compiling

Measured on bun 1.3.11, building from macOS:

| Flag | Result |
|---|---|
| `--target=bun-windows-x64` | works |
| `--windows-hide-console` | **rejected** — "only available when compiling on Windows" |
| `--windows-icon`, `--windows-title`, `--windows-publisher`, `--windows-version` | **rejected**, same reason |

The console window is not cosmetic: a command prompt opens behind the app, and closing it
kills the agent. So it is fixed afterwards, by editing the PE header directly —
`toWindowsGuiSubsystem()` in `scripts/lib/apps.ts` flips the Subsystem field from 3
(console) to 2 (GUI), two bytes at offset 68 of the Optional Header. The stale checksum is
left alone deliberately: Windows verifies it for drivers, not for user-mode programs.

The **icon is not fixed**, and that is a deliberate stop. Embedding one means rewriting
the resource directory of a 116 MB executable, which is real risk for pure decoration, on
a platform this machine cannot run to check the result. The Windows app therefore has a
generic executable icon. macOS has a proper one (`assets/icon.icns`, drawn by
`scripts/make-icons.ts` — generated rather than committed as an opaque blob, so it has a
source anyone can edit).

**There are two Windows binaries, and that is load-bearing.** `dwp-agent-win32-x64.exe`
keeps its console, because `install.ps1` installs it for terminal use and a GUI-subsystem
process has no output. `dwp-agent-win32-x64-gui.exe` is the app. An agent updating itself
reads its own PE header (`windowsSubsystem()` in `paths.ts`) and asks for the same kind it
already is — otherwise the app's first update would silently give it a command prompt it
never had.

## Two bugs this found, both older than the app

Neither was introduced here; both were found by running the thing end to end.

**Binary self-update never restarted.** `restartIntoNewVersion()` respawned with
`process.argv.slice(1)`. That is correct for Node, whose argv is `[node, script, ...args]`
— but a Bun standalone's argv is `["bun", "/$bunfs/root/<name>", ...args]`, where argv[0]
is the literal string `bun` and argv[1] a path inside a virtual filesystem. The
replacement therefore received that virtual path as its command, did not recognise it,
printed the help, and exited **0**. So every binary update installed correctly and then
failed to come back — and because the exit was clean, launchd and systemd both declined to
restart it. A machine would simply go offline around the time a release went out, with
nothing in the log that looked like a failure. `docs/06`'s claim that the binary update
path was confirmed end to end was checking the hash and the reported version, which both
pass while this is broken.

**Two agents on one host fought for the identity.** The server hands the connection to
whoever connected last and closes the loser with code 4000. The agent treated that like
any other disconnect and retried, taking it straight back — two agents on one machine
trading the connection forever, each abandoning the other's work. It now stands down on
4000 and says why. The desktop app makes this easy to reach, which is how it surfaced.

## How the two never collide

Run-at-login installs `gui --hidden`, not `run`: one process, started at login with no
window. Opening the app again does not start a second agent — it finds the running one
through `~/.dwp/gui.json`, opens a window onto it, and exits. Verified: launchd started a
second copy while one was running, it handed off and exited 0, and launchd left it alone.

`gui.json` is **not** deleted on exit, and that matters more than it looks. It holds the
port and the window's token, so an open window survives the process restarting into a new
version. The first attempt did delete it, and the replacement minted a fresh token — every
open window broke on every update, which is exactly what an update should not do.

## What is verified, and on what

Run against a live control service, with a packaged app unzipped the way a friend would:

- Pairs from the window by pasting an invite link; appears online in the fleet.
- Runs real work; the window names the adapter and how long it has been going.
- Pause, resume, run-at-login and quit all do what they say — the LaunchAgent was
  inspected, not trusted.
- **Updated itself 0.3.0 → 0.4.0 while running**: downloaded, verified against the pinned
  key, replaced its own executable, restarted, reconnected, and kept the same window
  address. The binary's hash on disk changed; the window never broke.
- Launched through LaunchServices (a real double-click), not just by exec'ing the binary.

### Windows — no longer a guess

Run on a real machine (`LAPTOP-9CG858LS`, win32 x64, 12 cores):

- **The windowless build starts.** The PE subsystem patch works — the app opened, drew its
  window, and needed no console. The `DWP Agent (with console).exe` fallback went unused.
- **It connects and does work.** Sent an `echo` task pinned to that host: accepted, ran
  3,027 ms for a 3,000 ms sleep, and returned a result signed on the machine and verified
  by the server, reporting `os: win32, arch: x64`.
- **The stand-down message did its job**, and in doing so exposed that standing down was
  permanent — see above. That is the bug real hardware was always going to find.

Still unproven on Windows, and worth saying rather than implying otherwise: pairing from
the window on a machine that has never joined (the one tested was already enrolled from
the CLI install), self-update through the move-aside-then-rename path, and the run-at-login
toggle driving `schtasks`.

## Getting past the macOS warning — the instructions were wrong once

`Right-click &rarr; Open` has been the standard advice for a decade and **Apple removed it
in macOS 15**. On anything current the route is: let it be blocked, then System Settings
&rarr; Privacy &amp; Security &rarr; Security &rarr; **Open Anyway**. The download page
said the old thing and had to be corrected on first contact with a real Mac (26.3.1).

The bundle is also **ad-hoc signed** now, which it was not at first. Bun linker-signs the
executables it produces but nothing signed the bundle, so `spctl` reported "no usable
signature" — and on Apple Silicon a quarantined bundle with no signature at all can fail
as *"is damaged and can't be opened. You should move it to the Trash"*, which offers no
way past it. Ad-hoc signing costs nothing and turns that into the ordinary "unidentified
developer" block, which Privacy &amp; Security can release. It is not notarization and
does not remove the warning; it makes the warning survivable.

## Signing — unchanged, and still not bought

Everything above works unsigned, with a warning on both platforms that the download page
now names and walks through in advance. macOS notarization is $99/yr and the only way to
remove the Mac warning; Windows has SignPath Foundation, free for open-source projects,
which signs under *their* publisher name. Paying does not buy a clean Windows install
anyway — since a 2024 policy change even an EV certificate builds SmartScreen reputation
from zero. **The goal remains a warning that is expected and explained, not absent.**

## Still open

1. **The rest of Windows.** It runs and does work; pairing from the window, self-update,
   and the run-at-login toggle have still only been exercised on macOS.
2. **A Windows icon**, if the generic one grates enough to be worth PE resource surgery.
3. **The server address.** Updates and work both ride the connection, so a machine that
   cannot reach the control service gets neither. `dwp-agent set-server` re-points without
   re-pairing, but a friend cannot run it — a tunnel URL that changes on restart is still
   the weakest link in handing this to someone else.
4. **The iOS agent has not been given the 4000 stand-down**, so two iOS agents on one host
   id would still fight. See `.claude/skills/port-agent-change/SKILL.md`.
