# D-009 — Exit timing: the 12:30 intraday checkpoint proposal

| | |
|---|---|
| **ID** | D-009 |
| **Date** | 2026-07-11 (proposed) |
| **Status** | **Proposed** — gated on Build 5 exit-variant evidence |

This record is the pattern's showcase: **a proposal parked WITH its
retest recipe**, so the day the gate opens, the test is already written.

## Context

Exits decide at the close (R11, [D-003](D-003-1b-position-engine.md)) and
fill at the next open. Build 4 measured that lag: **+7.88% cumulative
adverse cost over 73 switches (10.8 bp/switch)** — overnight gaps
systematically move against a close-signaled switch. That number is the
motivating datum for an intraday decision checkpoint (e.g. 12:30 ET): a
price below the stop at midday MIGHT be worth acting on before the close.

The counter-case is exactly R11's reason to exist: intraday wicks whip;
a 12:30 print below the stop that recovers by the close is a whipsaw the
close-basis rule correctly ignores. Which effect dominates is an
empirical question — Build 5's.

## Options considered (to be adjudicated on evidence)

| Option | Summary |
|---|---|
| Status quo | Close decides, next-open fills; intraday ALERTS only (PER-510-B — information, never execution) |
| 12:30 checkpoint | A midday stop-breach decision point in addition to the close |
| Close-decide, close-fill | MOC execution on signal day (captures most of the lag without any intraday rule change) |

## Evidence so far

- [docs/backtest-regime.md](../backtest-regime.md) EOD-lag section: the
  +7.88% / 10.8 bp per switch datum (regime-level switches, not
  single-name stops — the Build 5 replay must measure the SAME question
  at the position level before any ruling).
- PER-510-B shipped the information layer only; its docs note states the
  exit-timing change stays gated here.

## Ruling + rationale

None — gated. The gate: Build 5's ticker-level replay (SNDK/MU/MRVL +
controls) running exit variants side by side.

## Revisit triggers

Build 5's exit-variant results landing. Secondary: the intraday alert
layer documenting repeated large intraday-breach-to-close slippage on
live holdings (the alert history becomes evidence).

## Retest recipe

Parked, ready:

```
# Build 5 replay, exit-variant comparison (close-basis vs 12:30 checkpoint
# vs MOC-on-signal-day), per-ticker logs + whipsaw counts:
python3 scripts/replay_ticker.py --tickers SNDK,MU,MRVL --exit-variants close,1230,moc   # (Build 5 deliverable)
# The regime-level lag number that motivated this record reproduces via:
python3 scripts/backtest_regime.py   # strategies.binary.eod_lag_* in results JSON
```

## Links

- Docs: [docs/backtest-regime.md](../backtest-regime.md) (lag section, limitations §7)
- Jira: PER-508 comment 11706 (Build 5 spec)
- Related: [D-003](D-003-1b-position-engine.md) (R11), PER-510-B (alerts, information-only)
