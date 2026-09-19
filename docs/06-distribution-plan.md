# Getting the agent onto many computers

**Decided:** a desktop app, for friends who should never see a terminal.
**Platforms:** macOS, Windows, Linux, including older machines.

---

## What we are actually fixing

Measured on the current build, not estimated:

| Barrier | Today |
| --- | --- |
| Install size | **387 MB** of dependencies |
| — ONNX Runtime | **287 MB**, shipping *all three* platforms' binaries to every machine (win32 alone is 133 MB) |
| — Playwright | a hard dependency, though only the browser workload needs it |
| Prerequisites | Node 24, pnpm, and a copy of the project folder |
| Windows | `join.sh` is bash-only; 13 file-permission calls assume POSIX |
| Older devices | excluded by the Node 24 floor, which exists only because we run `.ts` directly |
| Updating | copy a file to each machine by hand |

Every one of those is a reason a friend gives up before the first task runs.

## The split that matters

The agent and the dashboard pull in opposite directions, and conflating them produces a
worse version of both:

- **The agent must be native.** Filesystem, sustained CPU, survives with no window open.
  A browser tab cannot do this — settled in the architecture, §3.
- **The dashboard must stay web.** It already works on every device with no install and
  updates the instant the server does. A desktop build of it would be a step backwards.

The tray app is a **shell around the agent**, not a replacement for the dashboard. It
loads the same web dashboard in a native webview, so there is one UI, not two.

---

## Phase 1 — Signed auto-update

*Status: server side built, agent side outstanding.*

Agents fetch updates over the connection they already hold. The bundle is signed by a
key **the operator holds**, and each agent pins that key when it pairs.

The property worth stating plainly: **a compromised control service cannot push code.**
It can serve bytes, but it cannot produce a signature, and an agent installs nothing it
cannot verify against the key it pinned on day one. This is the standard software-update
trust model, and it is what makes shipping code to a friend's laptop defensible at all.

- `node scripts/publish-release.ts "what changed"` — package, hash, sign
- `GET /release/latest` — public, signed, no secret in it
- `GET /release/:sha256` — bundle, enrolled hosts only
- Agent: `pnpm agent update`, plus automatic install when the server announces a newer version

**Done when** a new task type reaches every connected machine without anyone touching them.

## Phase 2 — Slim the install

Two changes, both large:

1. **Per-platform dependencies.** Ship only the current platform's ONNX binaries.
   287 MB → ~85 MB on macOS, ~68 MB on Linux.
2. **Optional workloads.** Playwright and ONNX become opt-in rather than required, so a
   machine that only runs compute downloads neither.

**387 MB → ~40 MB compute-only, ~120 MB with ML.** Every later phase inherits this.

## Phase 3 — One-line install, no prerequisites

Per-platform executables with the Node runtime bundled, published as signed releases:

```
macOS / Linux   curl -sSf https://your-address/install | sh
Windows         irm https://your-address/install.ps1 | iex
```

No Node, no pnpm, no folder. Bundling the runtime also **removes the Node 24 floor**,
which is what currently excludes older machines.

**Known risk, to spike before committing:** ONNX Runtime is a native `.node` addon, and
those do not embed cleanly into a true single file. The realistic shape is a small
executable plus a sidecar folder, shipped as one archive. This needs proving early
because it changes the packaging design.

## Phase 4 — Desktop app

A [Tauri](https://tauri.app) tray application — chosen over Electron for size (~10 MB
shell against ~100 MB) and because it matters more on the older hardware we are targeting.

- **Pairing without a terminal.** Paste the invite link; the app reads the address and
  code out of it. No commands, ever.
- **Tray icon** showing connected / paused / working, with a one-click pause.
- **Native webview** loading the existing dashboard, so there is a single UI.
- **Runs at login**, and survives reboots.
- **Tauri's updater** takes over from Phase 1's mechanism for app users, fed by the same
  signed releases from the control service. Phase 1 still serves command-line installs.

The Node agent ships as a Tauri *sidecar*, so none of the existing workload code is
rewritten in Rust.

---

## The cost nobody mentions until it bites

**Unsigned applications are blocked by default.** macOS Gatekeeper refuses them outright;
Windows SmartScreen shows a red warning. For friends who should never see a terminal,
this is not a cosmetic problem — it is the point at which they stop.

| | Cost | Needed for |
| --- | --- | --- |
| Apple Developer Program | ~$99/year | macOS app that opens without a warning |
| Windows code-signing certificate | ~$200+/year | Windows installer without SmartScreen |

Phases 1–3 work unsigned. **Phase 4 is where this becomes unavoidable**, and it is a
decision to take deliberately rather than discover on the day you send someone a link.

## Order of work

1. Finish Phase 1 — it removes the manual copy immediately and is mostly built.
2. Phase 2 — small, and shrinks everything downstream.
3. Spike the native-addon question before designing Phase 3.
4. Phase 3, then Phase 4.

Audit for Windows throughout: file permissions, the `ps` call in the browser probe, and
the bash installer all need cross-platform equivalents, plus CI on Windows to keep them
honest.
