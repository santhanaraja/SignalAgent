# D-002 — R4 Sunday-cadence qualification: 2 degraded weekly closes confirm

| | |
|---|---|
| **ID** | D-002 |
| **Date** | 2026-07-05 |
| **Status** | Ruled |

## Context

The pre-fix theme layer fired "EXIT SIGNAL — regime degraded" and deleted
active themes irreversibly on a SINGLE day of Caution — a hair-trigger on
a weekly-cadence decision. R4's intent (persistence before qualification
changes) needed real semantics.

## Options considered

| Option | Summary | Why (not) |
|---|---|---|
| Daily mutation | Qualification follows the daily regime print | The hair-trigger being fixed |
| Sunday cadence + confirmation | Signals fire fast (every run, advisory); qualification mutates only at the Sunday weekly review, and only after 2 consecutive degraded WEEKLY CLOSES | **Ruled** — "signals fast, state slow" |

## Ruling + rationale

Weekly close = latest entry per ISO week (the Sunday review state).
Two consecutive degraded weekly closes confirm degradation. A delayed
review (Monday catch-up) uses the completed-weeks basis so it reaches the
same conclusion the missed Sunday would have. Weeks whose Caution came
solely from a `data_unavailable` gate cap are transparent — an outage is
not evidence in either direction and can never confirm degradation.
Missing review marker → review is due (bootstrap self-heal); a pipeline
outage older than the previous ISO week stales the evidence rather than
confirming across the gap.

## Consequences

Theme (and successor group-level) qualification cannot whipsaw on one bad
day. The June hair-trigger deletion class is structurally closed.

## Revisit triggers

1. A confirmed degradation that the 2-week protocol made two weeks too
   slow to matter (documented live instance of harm from the lag).
2. D-007's Phase 1+ re-keys condition 5 to universe membership — when the
   theme layer retires (Phase 3), this record's *mechanism* survives as
   the observed-then-decided persistence question at the group level
   (see D-007 "what survives").

## Retest recipe

```
python3 test_r4_qualification.py   # 10 pins incl. the June replay
```

## Links

- Commit: aff1448
- Docs: [docs/regime.md](../regime.md)
- Related: [D-007](D-007-theme-layer-retirement.md) (inherits the spirit)
