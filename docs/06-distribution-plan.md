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
| Windows | **Fixed.** `join.ps1` ships alongside `join.sh`; the POSIX permission checks are platform-gated, with `icacls` ACLs as the Windows equivalent. Untested on real hardware. |
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

*Status: built and verified end to end on macOS — an agent installed a published release
over its own connection and restarted into it without anyone touching the machine, across
source-only updates, dependency-changing updates, and a deferred install that both failed
and recovered. The Windows-specific branch is written and its recovery path is tested by
forcing it on macOS; what remains unproven is real Windows hardware.*

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

**The dependency step, and the one place the platforms genuinely differ.** Windows keeps
an exclusive handle on every native addon a running process has loaded, so an update that
changes dependencies cannot replace them from inside the agent that is using them — and it
would fail *after* the new source is already on disk, which is the worst possible moment.

Three things now make this safe, all verified end to end:

1. **Most updates never touch dependencies at all.** Each release records a fingerprint of
   the dependency files *as shipped*, and the next release is compared against that. A
   source-only update — the common case — skips `pnpm install` entirely, so there is
   nothing to lock. Note that both halves of that comparison must be bundle-side: the copy
   on disk is no use, because `pnpm install` rewrites the lockfile locally, which made
   every update look like a dependency change.
2. **When dependencies do move, Windows defers.** The install is recorded in the config and
   run at the next start, by a process that has loaded none of those files yet. The restart
   was happening anyway, so this costs ordering and nothing else.
3. **A failed dependency step no longer fails the update.** The source is already
   installed by then; reporting failure meant the agent kept the old version number and
   re-downloaded the same release on every reconnect, forever. It now records the version,
   keeps the retry, and says so at every start until it succeeds.

Two bugs were found by testing this rather than reasoning about it, and both affected macOS
just as much as Windows:

- **The install mode was wrong for every friend's machine.** `join.sh` and `join.ps1`
  install with `--prod`; the update ran a plain `pnpm install`, which pnpm rejects outright
  on such a tree with `ERR_PNPM_INCLUDED_DEPS_CONFLICT`. The dependency step of every
  update would have failed on exactly the machines this is built for, while passing on the
  operator's own full checkout. The install now matches how the tree was installed.
- **Enabling a workload did not survive an update.** `pnpm agent enable ml` records itself
  in `packages/agent/package.json`, which an update replaces — so the install that followed
  removed the runtime, and the machine quietly stopped being able to do the work it was
  enrolled for. The enabled set is remembered in the config and restored afterwards.

**Not a route to mobile.** App Store guideline 2.5.2 and Google Play's Device and Network
Abuse policy both forbid an app updating itself outside the store, so this mechanism is
desktop-only by construction.

## Phase 2 — Slim the install ✅

*Done. Measured, not estimated.*

| | Before | After |
| --- | --- | --- |
| Joining a network | 352 MB | **11 MB** |
| With machine learning added | 352 MB | 99 MB |
| Wasted on other platforms' binaries | 202 MB | 0 |

Three changes:

1. **Optional workloads.** ONNX and Playwright are no longer dependencies. A machine
   joins with 11 MB and can immediately run `echo` and the walker — both pure JavaScript,
   needing nothing. `pnpm agent enable ml` or `enable browser` adds the rest on demand.
2. **Per-platform binaries.** onnxruntime-node ships macOS, Linux *and* Windows builds in
   one 287 MB package; only the current platform's 85 MB can ever execute. Enabling ML
   now deletes the other 202 MB.
3. **Capability filtering.** Hosts advertise what they can actually run, and the scheduler
   only offers matching work. Previously it offered anything to anyone and relied on the
   agent to decline — which returned the task to the queue and offered it straight back
   to the same host, forever.

Two details that matter more than they look:

- **The choice survives updates.** A release overwrites `package.json`, which would have
  silently removed an enabled workload — a machine doing inference would come back from
  an update unable to, with nothing in the log to explain it. Enabled workloads live in
  `~/.dwp/config.json`, which updates never touch, and are reinstalled afterwards.
- **A job nothing can run fails immediately**, and says which of the two reasons it is:
  nobody is connected, or nobody has that workload installed. Those need completely
  different responses from whoever submitted it.

## Phase 3 — One-line install, no prerequisites

Per-platform executables with the Node runtime bundled, published as signed releases:

```
macOS / Linux   curl -sSf https://your-address/install | sh
Windows         irm https://your-address/install.ps1 | iex
```

No Node, no pnpm, no folder. Bundling the runtime also **removes the Node 24 floor**,
which is what currently excludes older machines.

### Spike result — run, not guessed

**The lean agent compiles to a working single binary. Machine learning does not.**

Built with `bun build --compile`, and tested by pairing it against a live server from an
otherwise empty directory:

| Target | Size | Result |
| --- | --- | --- |
| macOS arm64 | 59 MB | pairs, connects, runs walker work |
| Linux x64 / arm64 | 95 MB | cross-compiled from macOS |
| Windows x64 | 111 MB | cross-compiled from macOS |

Cross-compilation from one machine means **no CI matrix is needed** to produce all three.

**Machine learning is the exception, and the reason is specific.** Bun does embed the
`.node` addon and extracts it at runtime — the failure is one layer down:

```
dlopen(...onnxruntime_binding.node): Library not loaded: @rpath/libonnxruntime.1.dylib
```

The addon is embedded; its companion shared library is not, and `@rpath` resolves
relative to the extracted temporary file. Making that work means per-platform library
surgery — `install_name_tool` on macOS, `patchelf` on Linux, DLL search paths on Windows
— which is fragile and permanent maintenance.

Attempts that did not work, for the record: marking the package external and placing a
sidecar `node_modules` beside the binary fails because a compiled binary resolves modules
against its embedded bundle, not the working directory. `createRequire` anchored in the
sidecar fails the same way.

### What this means for Phase 3

Two distributions, split along the line the spike drew:

- **Single binary — the default.** One download, no Node, no pnpm, no folder. Covers
  `echo`, the walker, and any future pure-JavaScript workload. This is what most people
  get, and it removes every prerequisite.
- **Node install — for machines doing ML or browser work.** Already 11 MB with opt-in
  extras, and already updates itself. Unchanged.

Shipping the native libraries properly belongs to the **installer** in Phase 4, which
lays files down in a known location rather than extracting them to a temporary path. That
is the right place to solve it, not a workaround bolted onto a single file.

## Phase 3 — One-line install ✅

*Done.*

```
macOS / Linux   curl -sSf https://your-address/install | sh -s -- YOUR-CODE
Windows         irm https://your-address/install.ps1 | iex
```

`node scripts/build-binaries.ts` cross-compiles all five targets from one machine and
signs the set:

| Platform | Size |
| --- | --- |
| macOS arm64 / x64 | 59 MB / 64 MB |
| Linux x64 / arm64 | 95 MB each |
| Windows x64 | 111 MB |

**Piping a script from the internet into a shell deserves care**, so the expected hashes
are generated into the script itself rather than fetched separately. Both arrive over the
same TLS connection from the same host, and the script refuses a download that does not
match — the model rustup and Homebrew use. The server independently re-checks each file's
hash before serving it, so a corrupted index cannot be distributed either.

Verified by swapping the expected hashes in a fetched script and re-running it: it
refused, printed both hashes, and installed nothing — the check happens before the file
is ever made executable.

**Binaries update themselves too.** A single-file install has no package manager to run,
so it downloads the new executable, verifies it against the signed hash, and renames it
over itself; the running process keeps its own open file, so this is safe. Windows will
not overwrite a running executable, so the old one is moved aside first. Confirmed by a
full cycle: hash changed, and the binary reported its new version afterwards.

One thing worth doing properly along the way: the version is now baked in at build time
with `--define` rather than kept in a hand-edited constant. That constant had already
drifted — it said 0.2.0 while the package said 0.3.0.

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

Windows is one system, not a parallel one. The trust model — Ed25519 signatures, SHA-256
hashes, a source bundle over the connection — has no opinion about the operating system,
and the agent bundle already carries its own Windows code. What was POSIX-only has been
given equivalents: file permissions, the `ps` call in the browser probe, and the bash
installer. What remains is the file-locking difference above, the `irm | iex` installer in
Phase 3, and **CI on Windows**, without which all of this rots quietly — nothing here has
run on real Windows hardware yet.
