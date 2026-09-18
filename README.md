# Distributed Work Platform

Coordinate work across computers you trust, see precisely where each task runs, and
remotely operate an authorized browser as another task type.

**Current phase: Pass 0 — architecture review. No implementation code exists yet.**

## Documents

| Doc | What it is |
| --- | --- |
| [`docs/00-handoff.md`](docs/00-handoff.md) | The original MVP / architecture / handoff brief |
| [`docs/01-architecture.md`](docs/01-architecture.md) | Pass 0 deliverable: implementation architecture, threat model, contracts, scheduling, gates, backlog, go/no-go |

## Status against the delivery plan

- [x] **Pass 0** — Architecture review
- [ ] **Pass 1** — Network spike (two hosts, two networks, echo tasks, `/whoami`)
- [ ] **Pass 2** — Durable batch CPU inference + measured baseline
- [ ] **Pass 3** — Trusted-host remote browser session
- [ ] **Pass 4** — Dashboard (fleet, new work, run map, live view, history)
- [ ] **Pass 5** — Hardening and demo rehearsal

## Blocked on (see architecture §14)

1. A public HTTPS/WSS hostname with a real certificate for the control service.
2. Two physical machines on two distinct networks. A VM does not substitute for this.

## Non-goals

Untrusted public hosts · arbitrary code or shell execution · real logged-in accounts on
stranger hosts · mobile host control · cross-chip translation · marketplace payments ·
end-to-end media confidentiality.
