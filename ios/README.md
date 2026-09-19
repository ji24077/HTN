# The iOS agent

An iPhone as an ordinary host: it pairs with a code, holds a `wss://` connection, runs
ONNX inference, and signs its results with a key the control service has never seen.

Not an App Store app, and it cannot become one — guideline 2.5.2 forbids the signed
auto-update the desktop agent relies on, and 2.4.2 covers background compute. It does not
need to be. Everything below is a build you install yourself, which App Review never sees.

```
ios/
  DWPAgentKit/        the agent core — protocol, identity, transport, adapters
    Sources/dwpagent/ the same core as a macOS command, so the simulator can drive it
  DWPAgent/           the SwiftUI app
  test/               conformance, scenarios, and the Simulator run
```

## Getting it onto your phone

**1. Set your team and a unique bundle id.**

```bash
cd ios
cp Local.xcconfig.example Local.xcconfig   # then edit it
```

`Local.xcconfig` is gitignored *and* survives `xcodegen generate`. Picking your team in
Xcode's UI does not: that writes into `DWPAgent.xcodeproj`, which is generated from
`project.yml` and thrown away on every regenerate.

The bundle id must be globally unique. A free Apple ID cannot register one another
developer already holds, and generic names are usually gone — use `com.<you>.dwpagent`.

**2. Generate and open.**

```bash
xcodegen generate && open DWPAgent.xcodeproj
```

**3. Enable Developer Mode on the phone.** *Settings → Privacy & Security → Developer
Mode*, then restart. Required since iOS 16, and the menu item only appears after the phone
has been plugged into Xcode once.

**4. Pick your iPhone and press Run.**

With a **free** Apple ID the first launch fails until you trust the certificate:
*Settings → General → VPN & Device Management → your developer account → Trust*. Once per
certificate, not once per build.

**5. Give the control service an address the phone can reach.**

```bash
pnpm db:up && pnpm share
```

`pnpm share` prints an HTTPS tunnel address that works from anywhere, including cellular.
A LAN address (`http://<your-mac>:8787`) also works but only on the same wifi —
`NSAllowsLocalNetworking` in `Info.plist` is what permits the plain-HTTP case.

**6. Pair.** Send yourself the printed `…/join?code=…` link, open it on the phone, copy it,
and paste it into the app. The app reads the address and the code out of the link. Codes
expire after ten minutes and work once; press Enter in `pnpm share` for another.

### The seven-day clock

A free Apple ID issues **seven-day** provisioning profiles. On day eight the app refuses to
launch and you rebuild from Xcode — the pairing and the key survive, so it reconnects by
itself. The $99/year membership makes it a year, and is worth it the moment you want to
demo twice.

## What it does, and what it does not

| | |
| --- | --- |
| Pairs from a link, no typing | Yes |
| Key generated on the device, private half never sent | Yes |
| Holds the connection, reconnects with backoff | Yes, while the app is on screen |
| Runs `echo` and `cpu_inference_batch` | Yes, real ONNX Runtime |
| Results signed and verified server-side | Yes |
| A running slice survives you leaving the app | iOS 26+, one slice at a time |
| Waits for work with the screen off | **No** — see below |
| `remote_browser_session` | **No.** Playwright cannot exist here |
| Updates itself over the connection | **No.** Rebuild in Xcode |

**The background limit, stated plainly.** `BGContinuedProcessingTask` is defined by a
measurable goal the user can watch and cancel. A slice qualifies; an agent idling for
offers does not, and iOS expires a task that reports no progress. So the app submits one
continued task *per slice*: work already running survives you locking the screen, and
between slices iOS suspends the app. A phone is a host while you are looking at it, plus
whatever it had already started. There is no arrangement of entitlements that makes it a
daemon, and the app says as much on its status screen.

On iOS 25 and earlier there is no continued processing at all, so work stops when the app
leaves the screen. The deployment target is iOS 17 deliberately: an older iPhone is still
a perfectly good host while the app is open, and raising the floor to 26 for a background
nicety would exclude devices that can do the actual work.

**Identity.** Secure Enclave is not used, and that is a trade rather than an oversight: it
holds P-256 keys only, while this protocol verifies Ed25519 everywhere. An Enclave key
would mean a second signature algorithm in a protocol that has exactly one. The key is a
CryptoKit software key in the Keychain, `AfterFirstUnlockThisDeviceOnly` — never synced,
never in a backup, unreadable until the device has been unlocked once since boot. Weaker
than hardware isolation, stronger than the desktop agent's mode-0600 file.

**Withdrawal, not declining.** A phone that is critically hot, in Low Power Mode, or nearly
flat and unplugged sends `consent.update` with `paused: true` and drops out of the
rotation. It does *not* decline offer by offer — the server releases a declined task and
re-offers it immediately, so that spins a loop which hammers the control service and burns
exactly the battery Low Power Mode was asked to save. It rejoins the same way.

## Running the tests

```bash
pnpm db:up            # Postgres, as usual
pnpm ios:all          # everything below, in order
```

| | |
| --- | --- |
| `pnpm ios:build` | Build the core and the macOS command |
| `pnpm ios:test` | 51 unit tests — JSON, crypto, frames, artifacts, device policy |
| `pnpm ios:conformance` | 39 checks that Swift's bytes match TypeScript's |
| `pnpm ios:e2e` | 11 scenarios against a real control service |
| `pnpm ios:app` | Build the app for the Simulator |
| `pnpm ios:simulator` | The real iOS build, paired and working, in the Simulator |

`node ios/test/e2e.ts --list` describes each scenario and why it exists.

**Why there are four layers.** Each covers something the others cannot:

- **Unit tests** pin behaviour that has no server in it — number formatting, DER encoding,
  malformed-frame handling.
- **Conformance** is the one that would otherwise fail silently. The agent hashes its own
  output and signs the hash; if Swift rendered a number differently from `JSON.stringify`,
  the control service would record `result.signature_invalid` — the event it emits when it
  believes someone is *forging* results. A formatting bug would be investigated as an
  attack. (This caught a real one: integral doubles above 2^53 printed exact rather than
  shortest-round-trip.)
- **Scenarios** run the shared core as a macOS process through the existing network
  simulator, so a phone gets the same treatment as a laptop: dropped links, killed
  processes, control restarts, revocation. The core is identical; only the shell differs.
- **The Simulator run** covers what is only true of an actual iOS build — that `os` reports
  `ios` and the schema accepts it, that the store works in a sandbox, that ONNX Runtime's
  iOS slice loads, that it pairs and starts with nobody tapping anything.

Two things no test here can reach, because no simulator can produce them: real background
execution (`BGTaskScheduler` is unavailable on the Simulator by design) and real thermal
behaviour on phone silicon. Both need the device. The throughput numbers the Simulator
prints are a Mac's, not a phone's, and are labelled as such.

## What the protocol needed

Small and additive, in `packages/protocol/src/messages.ts`:

- `os` gained `'ios'` and `'android'`. This one is load-bearing: the enum gates `hello`,
  and an unknown value means the host connects, is never recorded, and is never given
  work — it looks online and idle forever, with nothing in the log to say why.
- An optional `mobile` block on the capability record and the heartbeat: thermal state,
  Low Power Mode, battery, memory headroom, and the device's own `fitForWork` verdict.
  Optional because a desktop host has no answer for any of it, and absent is meaningfully
  different from false.

`/hosts` also now returns `adapters`, which it was already storing.
