# PR #3 review and integration

Reviewed Jack's `d4fad747` against the current `sentry-setup` application. The chosen
direction keeps the Python/Supabase control plane and ports the device runtime.

| Finding in Jack's PR | Resolution in the integrated application |
| --- | --- |
| Binary updater verifies the manifest but not its binding to the binary/app list | Shared list-digest verification; tampered binary/app entries and duplicate targets are rejected before download/install |
| Signed result digest is never compared with returned output | Python verifies exact output bytes and the signature before atomically storing result and proof; tampering tests cover both |
| Unvalidated assertion issuer reaches a UUID DB lookup in an uncaught upgrade callback | Python validates identity and claims before lookup and contains handshake failures; replay state lives in PostgreSQL |
| Expired final attempts can leave jobs running; creation races dispatch during per-task inserts | Jack's control plane is not retained; the existing Python scheduler and atomic submission remain authoritative |
| Supersession budget resets on every successful socket open | Budget now persists across automatic retries, queued retry timers are cancelled on explicit takeover, and gateway closes superseded sessions with 4000 |
| Existing scrubbers lack device pairing/private-key patterns | Extended Python/browser redaction, added Node worker redaction and in-memory Sentry delivery tests |

Validation performed on Windows:

- 21 React tests, frontend typecheck, and production build passed.
- 16 Node24 protocol/worker tests and TypeScript passed; deterministic walker outputs
  match the committed Swift IEEE-754 reference vectors.
- New Python cryptography, gateway, asset, and telemetry tests passed.
- Real temporary PostgreSQL integration passed: private schema/RLS, concurrent invite
  redemption, replay rejection, quotas, result immutability, generation fencing.
- Real Node24 worker against Python and temporary PostgreSQL passed: pairing, signed echo
  and walker execution, cancellation, and process restart during a held task. Native ONNX
  inference classified 10/10 MNIST inputs correctly using authenticated artifact downloads.
  The exported-result verifier accepted a real result and rejected tampered output.
- A temporary Windows worker executable built with Bun and its `--help` command ran.
- Python source distribution/wheel build and Ruff checks passed.

The complete Python suite contains four pre-existing Windows-only failures: three
POSIX-script tunnel-helper launch tests and one POSIX file-mode assertion. The two
opt-in integration tests were also run explicitly, separately from default discovery.
No connected browser was available for a visual UI run. Swift/macOS/iOS builds, remote
fleet checks, Docker image execution, and production Sentry delivery were not run here.

The integration retains Jack's Git ancestry and selected source files while excluding
his independent TypeScript backend. See [setup and architecture](jack-integration.md).
