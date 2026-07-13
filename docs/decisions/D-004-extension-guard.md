# D-004 — Extension guard @ 1.8×ATR: no RE_ENTRY_READY on blowoff-extended watchers

| | |
|---|---|
| **ID** | D-004 |
| **Date** | 2026-07-09 |
| **Status** | Ruled |

## Context

MRNA passed all 5 re-entry conditions and wore a RE_ENTRY_READY badge
while sitting 28% / 2.8×ATR above its SMA20 (close 79.07, SMA20 61.74,
RSI 77) — the exact FOMO chase the system exists to prevent. Structural,
not tuning: conditions 1–2 confirm a reclaim *near* the mean; a name that
never came back to the mean passes them trivially.

## Options considered

| Option | Summary | Why (not) |
|---|---|---|
| Tune conditions 1–2 | Tighten the confirmation margins | Wrong layer — the conditions are correct for their designed case (post-exit reclaim) |
| Extension ceiling as a 6th condition | Fails the all-five count | Muddies the ladder read: the five conditions still honestly evaluate |
| Guard AFTER the state machine | If `kind == watching` and state would be RE_ENTRY_READY and extension_atr > ceiling → EXTENDED_HOLD instead, with the guard note emitted | **Ruled** — conditions stay honest, the badge stops lying |

## Evidence — the calibration basis IS the record

Live fills and prints of the week of 2026-07-06 (preserved verbatim in
the [config comment](../../framework/config.yaml) at `positions.extension_guard_max`):

- Healthy trending entries filled at **1.06–1.64×ATR**; ARWR at 1.64×
  was the week's best entry — must stay eligible.
- MRNA's blowoff printed **2.03×ATR** at guard-ship time (2.8× at the
  original incident) — must be suppressed.
- Threshold **1.8×** sits above normal trending distance, below blowoff.
  (First proposal was 1.5×; raised to 1.8× at ruling so ARWR-style
  1.6–1.7× reclaims stay permitted.)

Live confirmation post-ship: the guard suppressed MRNA at 2.03×ATR
(2026-07-09), then released it to READY when extension compressed to
0.49×ATR (2026-07-10) — both transitions recorded as position events in
production within a day of shipping.

## Ruling + rationale

Guard at 1.8×ATR, watchers only. Holdings are exempt by design: an
extended HELD winner is trailing-stop territory, never force-exited by an
entry-eligibility rule.

## Consequences

EXTENDED_HOLD is a first-class state (panel badge, history events, labs).
The five green pips can now coexist with a blocked entry — by intent.

## Revisit triggers

Healthy watchers repeatedly suppressed at normal trending distance — i.e.
live fills routinely arriving above 1.8×ATR in a strong tape (the config
comment's calibration basis inverting). The [Position Lab](D-010-lab-pattern-laws.md)
makes checking a candidate's extension against the ceiling a drag of one
input.

## Retest recipe

```
python3 test_position_signals.py   # guard pins incl. the no-flapping test
python3 test_position_lab.py       # >1.8x -> EXTENDED_HOLD through the endpoint
```

## Links

- Commit: f2fa34a
- Jira: PER-508 comment 11710 (item 20)
- Config: framework/config.yaml `positions.extension_guard_max` (calibration comment)
