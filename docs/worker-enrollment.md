# Automatic worker enrollment

The command-line setup flow is the backend integration for a future desktop app.
It signs in through Supabase, requests a worker identity from the website API,
joins Tailscale using a single-use key, and saves a private worker profile. The
user does not need a Tailscale account for this managed-fleet flow. Only accounts
approved by the existing Supabase admin ID/email allowlist can enroll workers.

## Backend configuration

The ignored root `.env` contains these backend-only fields:

```dotenv
TAILSCALE_OAUTH_CLIENT_ID=
TAILSCALE_OAUTH_CLIENT_SECRET=
TAILSCALE_ENROLLMENT_TAG=tag:htn-worker
WORKER_GATEWAY_URL=wss://YOUR-BACKEND.YOUR-TAILNET.ts.net:8443/v1/worker
```

`htn-worker` is a label we choose for this app's worker service identities. Define
it in Tailscale's access policy, then assign it to an OAuth client with **Auth
Keys write** permission. The exact tag must match `TAILSCALE_ENROLLMENT_TAG`.
The OAuth client can then mint keys for that tag. No device/admin/ACL write
permissions are required by the enrollment backend.

Tailscale's visual policy editor supports **Access controls → Definitions → Tags
→ Create tag**. Enter `htn-worker` without `tag:` and select your administrator account
as owner. If your console shows the policy as JSON instead, merge this entry
into its existing `tagOwners` object (do not replace the rest of the policy):

```json
"tag:htn-worker": ["autogroup:admin"]
```

In **Settings → Trust credentials**, create the OAuth client and select this
tag under its Auth Keys permission. Save credentials only in the root `.env`.
The previous OAuth page may redirect to Trust credentials.

An existing client with the `all` scope already permits all device tags. It does
not need a separate tag assignment; a dedicated Auth Keys client is sufficient
for this app and has narrower permissions.

Configure network access so `tag:htn-worker` can reach the backend's Tailscale
address on TCP 8443. Tag creation by itself does not restrict network access:
existing broad allow rules remain effective. Review the existing policy before
connecting computers outside your trusted development fleet. Do not replace an
existing tailnet policy with a generic example.

Restart the bundled backend after configuration. It creates the private
`worker_enrollments` table alongside the existing tables. Worker enrollment
thereafter requires neither a `.env` change nor a backend restart. The original
`WORKER_TOKENS` map remains supported for existing workers and can be `{}` for a
fleet using only automatic enrollment.

## Enroll from the command line

From the repository root, after building the helper and installing dependencies:

```sh
uv run --project backend orchestrator-worker-setup \
  --server https://localhost:5174 \
  --ca-file .local/tls/root.pem \
  --helper .local/bin/orchestrator-tunnel \
  --name "My test computer" \
  --output .local/workers/my-computer.env
```

Enter your approved Supabase account email and password in the terminal. The
password is hidden and sent directly to Supabase over verified HTTPS. Do not
paste it into chat or command arguments. Use the real website URL and omit
`--ca-file` when using a publicly trusted certificate. HTTPS verification is
never disabled. This initial CLI supports email/password accounts; CAPTCHA/MFA
and browser-based OAuth login need a later browser onboarding implementation.

The command joins the private network, writes a mode-600 worker profile, and
starts the CPU stub worker. Add `--no-start` to enroll and save without executing
jobs. Existing profiles are never overwritten. Each profile has its own
Tailscale state directory; do not share or commit it.

On later runs, use the saved profile:

```sh
uv run --project backend --env-file .local/workers/my-computer.env orchestrator-worker
```

The profile stores the worker's credential and paths, not the Supabase session,
password, OAuth client secret, or Tailscale auth key. The auth key is used only
in memory for initial enrollment; the Tailscale node identity persists locally.

## API used by a future desktop app

`POST /v1/worker-enrollments` on the public website API accepts:

```json
{"request_id":"a-new-UUID-per-setup-attempt","name":"My computer"}
```

Supply the approved user's Supabase access token as a bearer header. Same-origin
browser cookies also work with the existing CSRF checks. Automation admin tokens
and demo sessions cannot enroll workers. The private gateway has no enrollment
route. A successful, non-cacheable response contains the new `worker_id`,
`worker_token`, `server_url`, and single-use `tailscale_auth_key`.

Keys expire after ten minutes, are not reusable, create persistent tagged nodes,
and are preauthorized for device approval. Persistent node credentials have a
separate lifetime from the initial enrollment key. Auth keys are never stored
in PostgreSQL. Worker credentials are random and stored there only as SHA-256
digests, along with the creating user, label, request ID, and provider key ID.

The database serializes enrollment reservations across API processes and limits
each user to ten attempts per hour and 100 pending/active enrollments. Reusing a
request ID returns 409; credentials cannot be retrieved again. Failed attempts
still count toward the hourly limit. Provider errors are sanitized, and a key
created before a database failure is revoked when possible.

A failed/interrupted CLI setup can leave an unused enrollment record; setup
recovery and fleet lifecycle/revocation UI remain follow-up work. This is a
shared fleet for approved administrators, not customer/tenant isolation. The
desktop UI and its distribution installer are not part of this change.

References: [Tailscale OAuth clients](https://tailscale.com/docs/features/oauth-clients),
[tags](https://tailscale.com/docs/features/tags),
[auth keys](https://tailscale.com/docs/features/access-control/auth-keys).
