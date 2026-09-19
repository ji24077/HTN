---
name: port-agent-change
description: Port a change made in the desktop TypeScript agent across to the Swift iOS agent (and say what Android would need). Use when packages/agent or packages/protocol has changed and the other implementations may need to follow.
---

# Porting a desktop agent change to the other platforms

## The one thing to understand first

There is **no shared agent code across platforms**. There are two independent
implementations of one protocol:

| | Implementation | Language |
|---|---|---|
| macOS, Windows, Linux | `packages/agent` → 5 binaries via Bun | TypeScript |
| iOS | `ios/DWPAgentKit` | Swift (~5,000 lines) |
| Android | **does not exist** | — |

What they share is **`packages/protocol`** — the message shapes, the Ed25519 assertion
format, the result attestation, and the deterministic maths. That is the contract. A
change is portable if and only if it changes that contract or the behaviour built on it.

So the job is never "copy the diff across". It is: **decide whether this change is part
of the contract, and if so, express it again in Swift.**

## Step 1 — Classify the change before touching Swift

Most desktop changes must **not** be ported. Work out which bucket the change is in.

**Must port — it changes the contract**
- `packages/protocol/src/messages.ts` — any field added, removed, renamed, or made
  required/optional
- `packages/protocol/src/assertion.ts` / `attestation.ts` — anything about what gets
  signed, the byte layout, TTLs, or claim names
- `packages/protocol/src/walker.js` / `dmath.js` — **any** change, see Step 2
- `packages/agent/src/capability.ts` — the shape a host reports about itself
- **the agent version string**, which is a known trap: the desktop derives it at build
  time (`paths.ts` → `__DWP_VERSION__`, falling back to reading `package.json`) precisely
  because a hand-edited constant had already drifted once. iOS still hand-edits one —
  `Capability.swift:6` says `"0.2.0-ios"` while `packages/agent/package.json` says
  `0.3.0`. So it has drifted again, on the other side. A desktop version bump does **not**
  propagate; bump the Swift constant in the same change or the server's view of what an
  iPhone is running will be wrong.
- `packages/agent/src/adapters/*.ts` — if the adapter exists on iOS
  (`inference`, `walker`; `echo` is trivial; `browser` is desktop-only)

**Must NOT port — deliberately desktop-only**

| Desktop file | Why it stays there |
|---|---|
| `update.ts` | Apple guideline 2.5.2 forbids downloading and running new code. iOS updates are a rebuild. Porting this is not merely unnecessary, it is not allowed. |
| `service.ts` | Run-at-login (launchd / systemd / schtasks). iOS has no equivalent and must not fake one. |
| `resolver.ts` | DNS fallback for a broken system resolver. iOS uses `URLSession` and does not have the Node lookup-hook problem. |
| `winacl.ts`, `paths.ts` | Windows ACLs and binary-path detection. Meaningless on iOS. |
| `workloads.ts` disk sizing | Different sandbox rules entirely. |

**Judgement call — port the intent, not the code**
Backoff timings, heartbeat intervals, suspension detection, concurrency limits. These
must stay *behaviourally* compatible with the server's expectations
(`packages/control/src/config.ts`) but the Swift implementation is idiomatic Swift, not
a transliteration. iOS additionally has thermal and battery state the desktop lacks.

If the change is desktop-only, **stop and say so.** "No port needed, because X" is a
complete and correct outcome for this skill.

## Step 2 — The determinism rule (read this every time)

`packages/protocol/src/dmath.js` and its port `ios/.../Adapters/JSMath.swift` exist
because **`Math.sin` and friends are not specified to the last bit**. V8 and Apple's libm
disagree on about 4% of inputs by one ulp.

That is not a rounding curiosity. Over 1800 steps of stiff contact physics it compounds
until the machines disagree about reality: **a trained gait scored 45.4 under one
implementation and 12.3 under the other.** Measured, not assumed.

Because results are **signed by the machine that produced them** and verified centrally,
a platform that computes a different number does not produce a "slightly different
result" — it produces a result the control service records as
**`result.signature_invalid`**, the event emitted when it believes someone is **forging
results**. A one-ulp difference surfaces as a suspected attack.

Therefore:

- Never introduce `sin`, `cos`, `tanh`, `log`, `exp`, or `pow` into physics or scoring
  code on either side. Use `dmath` / `JSMath`.
- Any change to `dmath.js` or `walker.js` must be ported **line for line** — same
  constants, same coefficients, **same order of operations**. Reassociating `a*b + a*c`
  into `a*(b + c)` is a real change in floating point.
- Walker/dmath determinism is checked by `pnpm ios:conformance` and `pnpm ios:e2e`, which
  run whole simulations against the JS implementation. There is **no** Swift-side
  `WalkerConformanceTests`: the name appears in two source comments
  (`Adapters/Walker.swift:8` and `Adapters/JSMath.swift:13`) and as a test zero times —
  51 `func test*` exist under `ios/DWPAgentKit/Tests` and none covers walker, determinism
  or gait. `Walker.swift` is the worse of the two, because it cites that test as the
  reason a design choice is *defensible*. Do not treat either comment as evidence of
  coverage, and do not let it talk you out of running the TS-side checks.
- Power-of-two scaling is exact and safe; `pow(2, k)` is not (`scale2` loops for this
  reason).

## Step 3 — File correspondence

| TypeScript | Swift |
|---|---|
| `packages/protocol/src/messages.ts` | `ios/DWPAgentKit/Sources/DWPAgentKit/Protocol.swift` |
| `packages/protocol/src/assertion.ts`, `attestation.ts` | `Crypto.swift`, `HostIdentity.swift` |
| `packages/protocol/src/envelope.ts` | `JSON.swift` — **contract-critical**: matches `JSON.stringify` key order *and* number formatting (JS writes `1`, Swift writes `1.0`), because the agent hashes its own output and signs that hash |
| `packages/protocol/src/dmath.js` | `Adapters/JSMath.swift` |
| `packages/protocol/src/walker.js` | `Adapters/Walker.swift` |
| `packages/agent/src/transport.ts` | `Transport.swift` |
| `packages/agent/src/pair.ts` | `Pairing.swift` |
| `packages/agent/src/keys.ts` | `KeyStore.swift` (Keychain, not a file on disk) |
| `packages/agent/src/capability.ts` | `Capability.swift` |
| `packages/agent/src/adapters/inference.ts` | `Adapters/InferenceAdapter.swift` |
| `packages/agent/src/adapters/walker.ts` | `Adapters/WalkerAdapter.swift` |
| artifact fetch + cache-by-hash | `ArtifactStore.swift` |

The SwiftUI app (`ios/DWPAgent/`) is presentation only. Protocol changes rarely reach it;
capability or consent changes sometimes do.

## Step 4 — Verify, in this order

Run from the repo root. **Do not skip the conformance test** — it is the one that catches
the failure that does not look like a bug.

```bash
pnpm ios:build         # swift build
pnpm ios:test          # Swift unit tests (JSON ordering, SPKI/PEM, attestation, assertion)
pnpm ios:conformance   # Swift's real output verified by the real @dwp/protocol verifier
pnpm ios:e2e           # scenarios; `node ios/test/e2e.ts --list` explains each
pnpm ios:app           # the SwiftUI app still builds
pnpm ios:simulator     # runs it in the iOS Simulator
```

Or `pnpm ios:all` for the lot.

`ios/test/conformance.ts` mocks nothing: Swift produces assertions, attestations and
walker evaluations through its real code path, and the TypeScript verifier checks them in
both directions. If a protocol change is wrong, this fails here rather than in production
as a forged-result alert.

Also run `pnpm typecheck` and, if the change touched anything the server relies on,
`pnpm sim` — a contract has two sides.

## Step 5 — Android

**There is no Android implementation, and this skill will not invent one.** If asked to
port a change to Android, say that plainly rather than creating an `android/` directory
that implies more exists than does.

Building one means a third implementation of the same contract — Kotlin, a third
`JSMath` port held bit-identical, a third conformance suite. The recommendation on record
is **not to build it**: phones are poor workers (thermal throttling, battery, aggressive
process-killing), and the iOS agent exists because someone already built it, not because
it was the best use of effort.

If it is ever built, it belongs at `android/` mirroring `ios/`, and Step 4 gains
`pnpm android:conformance`. Android permits self-update where Apple does not, so
`update.ts` moves from the "must not port" column to "judgement call".

## Reporting back

State, in this order:
1. Which bucket the change fell into, and why.
2. What was ported, file by file.
3. What was deliberately **not** ported, and the reason.
4. Test results — actual output, including anything that failed.

If the conformance test fails, **do not adjust the test to match Swift.** The test
encodes what the control service will accept; changing it hides a result the server will
later reject as a forgery.
