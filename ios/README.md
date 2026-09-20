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

Run the existing Python/Supabase app using the [root setup](../README.md) and expose
its configured `PUBLIC_ORIGIN` over trusted HTTPS. The Python public listener now
serves the device protocol at `/agent/connect`; no separate TypeScript control
service or database is needed. See [integration setup](../docs/jack-integration.md).

**6. Pair.** In the React fleet dashboard, choose **Create device invite**. Paste that
link into the iOS app. The app reads the address and code from the link. Codes expire
after ten minutes and work once; create a fresh invite in the dashboard when needed.

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

On macOS with Xcode/Swift and Node 24 installed:

```bash
pnpm ios:build
pnpm ios:test
pnpm ios:conformance
pnpm ios:app
```

Swift unit tests cover JSON, crypto, protocol, artifacts, device policy, and deterministic
walker arithmetic. Conformance checks compare the real Swift and TypeScript signing and
serialization implementations. The Node test suite also pins the committed Swift walker
reference vectors.

The old TypeScript-control simulator is not included in this integration. The Python
platform has its own gateway tests and real Node/Python/PostgreSQL end-to-end test;
see [integration validation](../docs/jack-integration.md#verification). Fresh iOS device
and Simulator runs against the Python gateway remain required on macOS. Background
execution and thermal behavior still need a real phone.

## What the protocol needed

Small and additive, in `packages/protocol/src/messages.ts`:

- `os` gained `'ios'` and `'android'`. This one is load-bearing: the enum gates `hello`,
  and an unknown value means the host connects, is never recorded, and is never given
  work — it looks online and idle forever, with nothing in the log to say why.
- An optional `mobile` block on the capability record and the heartbeat: thermal state,
  Low Power Mode, battery, memory headroom, and the device's own `fitForWork` verdict.
  Optional because a desktop host has no answer for any of it, and absent is meaningfully
  different from false.

The Python `/v1/workers` response exposes supported adapters in `capabilities.kinds`.
