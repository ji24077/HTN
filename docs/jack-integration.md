# Desktop and iOS workers on the Python platform

This integrates Jack's PR #3 (`d4fad747`) into the existing Python/Supabase application.
The Python database, scheduler, Supabase approval rules, React dashboard, Sentry, and
self-heal tool remain the application. `packages/agent`, `packages/protocol`, `ios`,
workload fixtures, and release builders come from Jack. His separate TypeScript control
plane, database, password authentication, and dashboard are not part of the runtime.

## Connect a device

1. Start the app as described in the root README, and sign in with an approved account.
2. Choose **Create device invite** in the dashboard. Each invite expires after ten minutes
   and can pair exactly one device. Only the invite hash is stored in PostgreSQL.
3. On a desktop, install Node 24+ and pnpm 10.27, run `pnpm install --frozen-lockfile`
   from the repository root, then `pnpm agent gui`. Paste the invite link in the app.
   Alternatively run `pnpm agent pair --server https://your-fleet --code YOUR_CODE`,
   followed by `pnpm agent run`.
4. For iOS, build the Swift app using [its guide](../ios/README.md) and paste the same
   kind of invite link. A phone must remain in the foreground to accept new work.

Use a trusted HTTPS public origin accessible to the device. A phone does not trust a
self-signed localhost development certificate. Local loopback testing can use the demo
server's HTTP address. The device generates its own Ed25519 private key; the server stores
only its public key. Pairing does not give the device access to the fleet's admin API.

The public listener exposes `/hosts/pair`, `/agent/connect`, and authenticated workload
artifact downloads. Device assertions expire after two minutes; nonce replay rejection
is shared through PostgreSQL. This deliberately allows phones and desktop workers to
connect without installing the private Python worker's Tailscale tunnel. The original
bearer-token `/v1/worker` endpoint stays on the private listener.

## Run work

Choose a workload in the existing dashboard, select a compatible worker or automatic
assignment, and send the task. **Run one on each** uses echo for paired devices and the
stub for Python workers.

| Task kind | Execution |
| --- | --- |
| `stub` | Existing Python connection/lease demo |
| `echo` | Device identity and signed-result round trip |
| `walker_evolution` | Eight seeded candidate gaits; Jack's deterministic arithmetic is retained |
| `cpu_inference_batch` | 100 MNIST digits using the bundled content-addressed model/data |

Enable the optional inference runtime with `pnpm agent enable ml`, then restart the worker
so it advertises that adapter. Compiled desktop binaries support echo and walker; use the
Node installation for ONNX. The task API remains `POST /v1/tasks`; its task `kind` selects
the adapter and its `payload` is the adapter's input. No GPU capability is invented for
these CPU workers.

All devices use the existing database lease, session, and generation rules. The platform
enforces **one active task per device**, regardless of a desktop's advertised concurrency.
Cancellation, retries, reconnections, and accepted results appear in the same React stream.

Completed device tasks include an `attestation` containing the signature, public key,
assignment identity, digest, and exact JSON output bytes. The gateway checks both the
signature and the output digest before the result and proof are committed together.
Retaining exact output bytes preserves JS/Swift floating-point and Unicode serialization;
rehashing PostgreSQL JSON serialization would not reproduce the signed bytes.

To independently verify a task response saved as JSON:

```sh
uv run --project backend python scripts/verify-device-result.py result.json
```

The verifier compares the stored result with the signed bytes and checks the task,
generation, and worker identity as well as the Ed25519 signature. It uses the public key
in the export; independent identity assurance requires matching that key to the trusted
enrolled device key.

## Signed updates and artifacts

`pnpm release` packages source workers; `pnpm build:binaries` uses Bun to build desktop
executables/apps. Run signing commands only on the release operator's machine. Keep the
release signing private key there, and copy only generated release files to the backend's
`DWP_RELEASES_DIR` (default `releases/` at the repository root). The first published release
supplies the public key pinned during pairing. Existing devices with no pinned release key
require the explicit `pnpm agent trust-updates --yes` flow after inspecting the fingerprint.

The binary updater now verifies the digest of the entire binary/app list against the signed
manifest before trusting any download hash. Source bundles require a signed device assertion;
published desktop downloads are public so a new device can install them. No release is
published automatically by merging this change.

Artifacts come from `DWP_FIXTURES_DIR` (default `fixtures/`); workers verify their SHA-256
hashes before executing inference. The backend Docker image includes those fixtures.
Standalone wheel deployments must configure/copy fixtures explicitly. Release files remain
an optional mounted directory and are not baked into the image.

## Sentry and self-heal

Existing backend and frontend telemetry is preserved. Desktop workers automatically
receive public Sentry settings at pairing and reconnect, and retain them in their
device profile for installed app/service startup. The platform defaults to its own
`SENTRY_DSN`; `SENTRY_WORKER_DSN` can override it or explicitly disable worker reporting.
Local `SENTRY_DSN`, `SENTRY_ENVIRONMENT`, and `SENTRY_RELEASE` environment settings
override the managed configuration. An explicitly blank local DSN disables telemetry.
Caught execution failures carry task/worker tags. Private keys, PEM values, invitation codes,
tokens, and configured credentials are scrubbed. The gateway records a sanitized failure
for iOS; no native Swift Sentry SDK is added.

`scripts/sentry-agent.py` remains the PR-only self-heal loop. It now permits the worker's
locked dependency installation, `pnpm test`, and `pnpm typecheck` in its isolated worktree.
Run it with the intended integrated branch as `--base`; no self-heal run is started by setup.

## Verification

```sh
pnpm typecheck
pnpm test
npm --prefix frontend test
npm --prefix frontend run build
uv run --project backend python -m unittest discover -s backend/tests -v
RUN_DATABASE_TESTS=1 uv run --project backend --python 3.12 --extra demo python -m unittest discover -s backend/tests/integration -v
RUN_DEVICE_E2E=1 uv run --project backend --python 3.12 --extra demo python -m unittest discover -s backend/tests/integration -p test_dwp_e2e.py -v
```

On PowerShell set each flag with `$env:RUN_DATABASE_TESTS='1'` or `$env:RUN_DEVICE_E2E='1'`
before its command. The integration tests create isolated local PostgreSQL data and worker
profiles; they never use your configured Supabase database. Runtime end-to-end tests need
Node 24+ and installed root dependencies. Swift builds/device validation require macOS/Xcode.
Set `RUN_DEVICE_INFERENCE_E2E=1` as well to exercise native ONNX with ten MNIST inputs;
this requires the optional inference runtime to be installed.

The original branch's network simulator is coupled to its discarded control service. Its
results do not establish this integration's correctness; the new gateway/database tests
and real Node/Python end-to-end tests cover the retained runtime. Real remote hardware and
iOS validation must be repeated for this combined application.
