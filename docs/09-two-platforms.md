# Two platforms, one problem

**Written:** 2026-09-19. For Jack, Ethan, and Phineas to decide from.

This is not a proposal. It is a comparison written because two of us have independently
built the same system, and the repo now contains both.

## The situation

| Branch | Author | What it is |
|---|---|---|
| `jack` | Jack | TypeScript control plane, 138 files |
| `platform-setup` | Ethan | Python control plane, 102 files |
| `sentry-setup` | Phineas | `platform-setup` + Sentry reporting and a self-heal loop |

`jack` and `platform-setup` have **no common ancestor**. They are not two versions of one
project; they are two projects in one repository.

## They are the same design

Neither of us copied the other, and we arrived at the same architecture:

- Workers dial **out** over WebSockets, so no worker needs a public address
- **PostgreSQL owns leases**, with expiry sweeping stale work back to the queue
- A dashboard fed by server-pushed state
- Tasks are independent units; results are accepted once and are idempotent
- Tailscale for the public address

The agreement is worth noting: two independent designs converging is decent evidence the
shape is right.

## What each has that the other does not

**Ethan's platform**

- **GPU awareness.** `Runtime = cpu | cuda | mps` is in the protocol from the start.
  `jack` has no concept of a GPU at all, which for a project about distributing compute is
  a real gap.
- **`one_task_per_worker`** as a partial unique index. Concurrency control enforced by the
  database rather than by application logic — `jack` uses per-host dispatch locks in
  process, which is weaker and was the source of a 12x over-dispatch bug.
- **Generation fencing** on tasks, so a late result from a superseded assignment cannot
  be accepted.
- **Supabase auth** with email confirmation and an approved-user list, plus a private
  worker gateway separated from the public API.
- **A React frontend** with component tests.
- **Canonical JSON** for storage and comparison, bounded payloads, and explicit rejection
  of NaN and Infinity.

**Jack's platform**

- **Signed results.** Every result carries an Ed25519 attestation made with a key that
  never leaves the machine that produced it; the control service holds only public keys
  and can prove which machine did which work. Ethan's stores credential digests for
  enrollment but does not sign results, so the server's own logs are the only record of
  who computed what.
- **Validated on real, hostile hardware.** Four machines across macOS arm64, Windows x64
  and iOS, over the internet, behind carrier-grade NAT and campus wifi. 24,000+ tasks,
  98.90% on 10,000 MNIST digits, 128,000 walker evaluations in 556s.
- **A fault simulator.** 11 scenarios — packet loss, sleep/wake, NAT idle timeouts,
  reconnect storms, blocked WebSockets, control restarts — run against a real server.
- **Bit-identical cross-platform arithmetic**, so a score from a phone is comparable with
  a score from a laptop. This is what makes signed results meaningful rather than
  decorative.
- **A desktop app and an iOS app**, both installable by a non-technical person.
- **Signed auto-update** over the same connection.

**Phineas's work** sits on top of Ethan's but is not a control plane: Sentry export and a
loop that investigates issues and opens pull requests. It would apply to either platform.

## What merging would actually produce

Technically possible with `--allow-unrelated-histories`. Only `.gitignore` and `README.md`
collide, so it would "succeed".

The result would be 240 files containing **two complete systems side by side** — two
databases, two worker protocols, two dashboards, neither aware of the other. That is a
directory union, not an integration. It would look merged and be worth nothing.

**There is no merge. There is a choice, and then a port.**

## How to choose

The honest criteria, in the order that matters:

1. **What does the demo need to show?** If it must show GPU work, Ethan's protocol already
   models it and `jack` does not. If it must show *proof* that a specific machine did the
   work, `jack` has that and Ethan's does not.
2. **What is riskiest to rebuild?** Fleet validation and cross-platform determinism took
   the longest to get right in `jack`. Auth and a polished frontend took the longest in
   Ethan's. Porting either is days, not hours.
3. **Which codebase does the team want to work in?** Three people shipping in Python and
   React is a different proposition from three people shipping in TypeScript.

## A leaning, not a verdict

Pick **one control plane today** and port the other's distinctive pieces into it, rather
than continuing in parallel. Every hour spent on two systems is an hour of duplicated
work, and the duplication is already four commits deep on each side.

If the deciding factor is *demo credibility*, `jack` is further along on the thing that is
hardest to fake: real machines, real failures, and results that can be verified rather
than trusted. If the deciding factor is *GPU work and a finished-looking product*,
Ethan's is closer, and `jack`'s attestation and simulator are the pieces worth porting
into it.

Phineas's Sentry loop should be kept either way. It is the only piece here that is
genuinely additive to both.

## What must not happen

Neither branch should be merged into the other to "combine the work". It would produce a
repository that builds two things, tests neither, and hides which one is real.
