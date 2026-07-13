# D-003 — 1B position engine: 5 conditions, close-basis stops, positions.json authoritative

| | |
|---|---|
| **ID** | D-003 |
| **Date** | 2026-07-05 |
| **Status** | Ruled |

## Context

Exits fired on SMA20 close-breaks but nothing tracked the road back —
re-entry was pure discretion, exactly where FOMO lives. Build 1B encoded
the re-entry as a state machine. "Signals only, nothing auto-executed"
was a ruling condition, not a caveat.

## Options considered

| Option | Summary | Why (not) |
|---|---|---|
| Advisory text only | Notes on the dashboard | No state, no discipline, nothing testable |
| 5-condition state machine | HELD / EXIT_FIRED / WATCHING / RE_ENTRY_ARMING / RE_ENTRY_READY on: (1) close > SMA20, (2) 2 consecutive closes above OR one close > 0.5×ATR margin, (3) regime risk-on (Choppy = conditional, A+ only), (4) SMA20 slope flat/rising vs 5d ago, (5) thesis (theme qualified) | **Ruled** |

## Evidence

Validated by replaying MRVL's June 18 – July 2 history (entry $293, fired
stops Jun 26 / Jul 1, Jun 30 reclaim arming) against the machine's states
at build time — pinned as a fixture in test_position_signals.py (the
item-11 spec had named June 23 as the start; the shipped fixture begins
June 18).

## Ruling + rationale

Key structural sub-rulings, each earned by an adversarial-review finding
or a live incident:

- **Close-basis everything (R11):** the engine strips the intraday
  synthetic bar; decisions only on completed daily closes.
- **positions.json is authoritative for what is held:** a holding whose
  re-entry conditions complete returns to HELD (otherwise HELD is
  unreachable after one exit).
- **Events in a separate file** (`data/position_events.json`):
  history_manager rewrites history.json wholesale; a shared file loses
  events (race reproduced pre-ship).

## Consequences

Re-entry discipline is machine-checkable and replayable. The machine's
"all green" can still collectively mislead on extended names — which is
exactly what fired D-004.

## Revisit triggers

1. A live case where the 5 conditions pass and the entry is still a
   chase → fired 2026-07-07 (MRNA at 2.8×ATR), resolved by [D-004](D-004-extension-guard.md)
   as a guard on top rather than a redesign.
2. D-007 Phase 1 rewires condition 5 (theme → universe membership) —
   this record then gets amended, not superseded (the machine stands).

## Retest recipe

```
python3 test_position_signals.py   # 18 pins incl. MRVL replay semantics
python3 test_position_lab.py       # extracted-path replay of live rows
```

## Links

- Commits: 7861dad (1B), f983fce (producer amendment)
- Jira: PER-508 description item 11
- Docs: [docs/rules.md](../rules.md) (R11 note)
