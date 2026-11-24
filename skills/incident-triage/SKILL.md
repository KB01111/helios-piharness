---
name: incident-triage
description: Structured triage checklist for diagnosing failing domain operations before mutating records.
version: 1
tags: [triage, reliability, domain]
---

# Incident Triage Workflow

Use this skill whenever a user reports an unexpected domain state or tool error.

## Procedure

1. Search long-term memory with `memory_search` for known constraints or past incidents.
2. Inspect the current record state with `domain_lookup_record` (idempotent read).
3. If the output is large and offloaded to an artifact, inspect the relevant slice with `artifact_read`.
4. Before modifying critical records, obtain human sign-off via `domain_request_approval`.
5. Only apply state changes via `domain_mutate_record` after verifying preconditions.
