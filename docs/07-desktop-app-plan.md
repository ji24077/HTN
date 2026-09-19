# A desktop app for Mac and Windows, without paying

**Status:** plan only. Nothing here is built.
**Audience:** the agent who picks this up next.
**Written:** 2026-09-19

## What the user asked for

A downloadable app for macOS and Windows so non-technical friends can join the network
without a terminal, **built and distributed without paying for anything**.

## The constraint that decides everything: signing

Read this before choosing a framework, because it changes what "done" means.

**You can build and ship unsigned for free. Users will see a scary warning.** That is the
entire trade, and no framework choice avoids it.

| Platform | Unsigned experience | Cost to remove it |
|---|---|---|
| macOS | "cannot be opened because the developer cannot be verified" — quarantine flag on anything downloaded | **$99/yr** Apple Developer Program, for notarization. No free path. |
| Windows | "Windows protected your PC" (SmartScreen) → *More info* → *Run anyway* | Free via **SignPath Foundation** if the project is open source; otherwise ~$10/mo Azure Artifact Signing |

Three facts worth knowing before anyone spends money:

1. **Paying does not buy an instant clean install on Windows.** Since a 2024 policy
   change, even an EV certificate goes through the same SmartScreen reputation-building
   as a standard one. A new signed app is still warned about until it accrues downloads.
2. **SignPath Foundation is genuinely free** for qualifying open-source projects and
   signs with an OV certificate on their HSM — but the publisher shown is *SignPath
   Foundation*, not this project.
3. **macOS has no free equivalent.** A free Apple ID signs for local development only.
   Notarization — the thing that actually removes the warning — requires the paid
   programme.

**Therefore the honest goal is not "no warning". It is "the warning is expected, explained,
and survivable".** A one-time right-click → Open on macOS, and *More info* → *Run anyway*
on Windows. Plan the onboarding copy around that rather than pretending it away, and keep
the existing one-line installer as the path for anyone who would rather not see it.

Escalate to the user before spending any money. The $99/yr Apple fee is the only way to a
clean macOS experience, and that is their call, not the implementing agent's.

## The architectural shortcut

**Do not write a second agent.** The agent already cross-compiles to a single binary for
five targets from one Mac (`scripts/build-binaries.ts`), and already knows how to pair,
reconnect, run adapters, update itself, and install as a login service.

The app is a **thin shell** around that binary:

- spawn the existing `dwp-agent` as a child process
- show pairing (a code box), connection state, and what it is currently running
- start/stop, pause, and a "run at login" toggle that calls the existing `service.ts` paths

Everything hard is already done and already tested. A shell that reimplements protocol,
identity or scheduling is a bug farm and a second thing to keep in sync — the repo already
carries that cost once for iOS, and `.claude/skills/port-agent-change/SKILL.md` exists
because of it. Do not add a third.

## Framework

**Tauri**, unless the spike says otherwise.

| | Tauri | Electron |
|---|---|---|
| Installer size | ~5–10 MB | ~100 MB+ |
| Prerequisite | **Rust (not installed here)** | Node (already present) |
| Fit with an 11 MB agent | good | the shell would dwarf the payload |

Electron is the fallback if Rust proves a problem, and its size cost is real but not
disqualifying. Decide in Phase 0, with a measurement, not a preference.

## Phases

Each phase ends with something demonstrable. Do not start the next until the gate passes.

**Phase 0 — Decide (half a day).**
Install Rust. Build the Tauri hello-world for macOS *and* cross-compile or VM-build for
Windows. Measure both installer sizes.
*Gate:* an empty window opens on both platforms, with real numbers recorded. If Windows
cross-compilation from the Mac turns out to need a Windows machine, say so here — it
changes the whole release story, and the agent binary's `bun build --compile` does not
have that problem.

**Phase 1 — Wrap the binary (1–2 days).**
Spawn `dwp-agent`, stream its output, surface connection state. No custom protocol code.
*Gate:* the app pairs with a code and appears as an online host in the dashboard, with the
same host identity as a CLI pairing — no second implementation of enrollment.

**Phase 2 — The three controls (1 day).**
Pause/resume, run-at-login toggle, quit. Each calls the existing agent paths.
*Gate:* toggling run-at-login produces exactly the same launchd/Scheduled Task state as
`dwp-agent service install`, verified by inspecting it, not by trusting the UI.

**Phase 3 — Unsigned distribution (1 day).**
Produce `.dmg` and `.exe`. Write the onboarding copy for both warnings. Serve them from
the control service beside the existing installers, hashes included.
*Gate:* a machine that has never seen this project installs from the download, gets past
the warning by following only the written instructions, and runs a task. Test with someone
who did not build it.

**Phase 4 — Signing, only if the user funds it.**
macOS notarization ($99/yr) and/or SignPath Foundation for Windows. Do not begin without
an explicit decision.

## What to test, beyond "it launches"

- **Quit vs close.** Closing the window must not silently stop the agent, and if it does
  stop it, the user must be told. A worker that quietly disappeared is worse than one that
  never started — the eduroam Mac dropped out of a live run in exactly this way, and only
  the server log showed it was a clean exit rather than a network fault.
- **Sleep and wake.** The agent already handles suspension; confirm the shell does not
  interfere. `sim/` covers the agent side.
- **Two instances.** Launching the app while a CLI agent is running: same identity, or a
  refusal with a clear reason. Never two agents on one host id.
- **Update path.** The binary updates itself over the connection. Decide whether the shell
  updates with it or separately, and write down which — a shell pinned to an old binary is
  a version-drift bug waiting to happen (see `Capability.swift:6`, which drifted exactly
  this way on iOS).
- **The warning copy itself.** The riskiest text in the product. Watch someone read it.

## Open questions for the user

1. Fund macOS notarization ($99/yr)? Without it the Mac warning stays.
2. Is this repo going open source? That is the gate for free Windows signing via SignPath.
3. Tray app or ordinary window? Tray suits something that runs all day; a window is simpler
   and easier to explain.

## Sources

- Microsoft: code signing options, SmartScreen reputation
- SignPath Foundation: free signing for open-source projects
- Azure Artifact Signing: from $9.99/month, GA April 2026, US/CA/EU/UK, self-employed
  individuals now eligible
