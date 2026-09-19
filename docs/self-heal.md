# Self-heal loop and data export

`scripts/sentry-agent.py` reads what the running system reported to
[Sentry](sentry.md) and does one of two things with it. It needs the `sentry`
CLI (`npm install -g sentry && sentry auth login`), and `heal` also needs
`claude` and `gh`.

## Heal: new issue in, pull request out

```sh
uv run --project backend python scripts/sentry-agent.py heal --dry-run
uv run --project backend python scripts/sentry-agent.py heal --watch 300
uv run --project backend python scripts/sentry-agent.py heal --issue HTN-BACKEND-7
```

For every unresolved issue it has not handled before, the loop:

1. downloads the issue, its latest event (including its stack trace),
   recent events, and the logs from the same trace;
2. fetches the latest remote `--base` commit and creates a branch
   `self-heal/<issue>` in a separate git worktree under
   `.local/self-heal/`, so your working tree is never touched;
3. runs a headless Claude Code agent there, which finds the root cause, makes
   the smallest fix, and adds a test;
4. commits the fix together with a new line in `docs/incidents.jsonl`, pushes
   the branch, and opens a pull request against `--base`.

The agent is told to answer `no_fix` for intentional failures, configuration or
infrastructure problems, and anything it is not confident about. Those are
remembered in `.local/self-heal/state.json` and shown to later runs, and no
branch or PR is created.

`--dry-run` stops after the local commit and keeps the worktree for review.
It uses the local `--base` branch intentionally; live runs fetch the remote
branch before each investigation and never include unpushed local commits.
Dry-run fixes remain eligible for a later live run; repeated dry-run passes keep
the existing worktree for review. Older `fix` state records without a PR are
treated the same way. Agent errors retry up to three attempts, then automatic
polling skips them; use `--issue` to explicitly retry after addressing the cause.
Issues with an opened PR or a `no_fix` verdict are also skipped.
Publication failures retain the committed branch and PR text locally, and retry
publication up to three times without running the agent again or force-pushing.
Retries check for an existing PR with the same commit first in case its creation
succeeded but the response was lost. `--issue` can also resume a pending publication
after its cap. Keep the saved branch until publication finishes; if it was deleted
manually, restore it or remove that issue's local state entry to investigate again.
`--budget` caps API spend per issue (default 3 USD) and `--max` caps issues per
pass (default 3).

### What it will never do

It opens pull requests and nothing else. It does not merge, deploy, restart
workers, resolve Sentry issues, or push to an existing branch. A person reviews
and merges every change.

### Learning without training

`docs/incidents.jsonl` is the loop's memory: one line per fixed incident with
the agent's sanitized title, root cause, and fix. Raw Sentry titles and culprits
remain in local context/state and are not copied into the committed record.
The agent reads the record before every investigation, so a
recurring failure is recognised rather than rediscovered. It is committed in the
same PR as the fix, so the record only becomes permanent when a person merges.

### Risks to understand before running `--watch`

- **Sentry events are untrusted.** The dashboard DSN is public, so anyone can
  send an event whose text tries to instruct the agent. The agent is told to
  treat event data as evidence only, has no network or git access, and can only
  edit files and run the unit-test commands. Those tests execute code the agent
  wrote, on your machine. Run the loop where that is acceptable, and read every
  PR as you would one from a stranger.
- **The repository is public.** The agent is told to describe data by shape and
  keep payloads, identifiers, and addresses out of commit messages and PR
  text. Check this during review.
- **Cost.** Each investigation is a full agent session.

## Export: data for analysis or fine-tuning

```sh
uv run --project backend python scripts/sentry-agent.py export --period 30d
```

Writes JSON Lines files to `.local/sentry-export/<timestamp>/` (ignored by
Git): `issues.jsonl`, and per project `events.jsonl` (full events), `logs.jsonl`,
and `spans.jsonl`. Sentry keeps logs and spans for about 30 days, so run it on a
schedule if you want history.

`fix-pairs.jsonl` joins each line of `docs/incidents.jsonl` to the diff of the
commit that fixed it. That is the labeled failure-to-fix data a fine-tune would
need; it grows by one example per merged self-heal PR.

Exports have passed the credential scrubber but still contain task payloads.
Treat them as sensitive.
