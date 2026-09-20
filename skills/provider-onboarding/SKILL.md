---
name: provider-onboarding
description: Validate GPU supply, owner lease policies, workload permissions, health, pricing, privacy, and resource limits before marketplace scheduling.
---

# Provider Agent

## Inputs

Require GPU identity and VRAM, benchmark evidence, health and availability, hourly price, network/region, owner identity and trust evidence, and a lease policy with schedule, workload types, data restrictions, and resource limits.

## Procedure

1. Verify the advertised GPU and mark unmeasured catalog entries as simulated.
2. Enforce the owner's local-time schedule, including overnight windows, minimum price, pause state, workload allowlist, public-data-only setting, region rules, runtime limit, and memory fraction.
3. Publish a secret-free card containing evidence status, health, price, trust, failure rate, network, region, and availability.
4. Recheck health and policy immediately before allocation and execution.

## Constraints

Provider registration does not authorize a job or payment. Never expose host credentials or private network details. Stop new scheduling when health or policy becomes invalid; preserve auditable rejection reasons for the owner and renter.
