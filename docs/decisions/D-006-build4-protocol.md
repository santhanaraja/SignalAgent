# D-006 — Build 4 backtest protocol (reusable)

| | |
|---|---|
| **ID** | D-006 |
| **Date** | 2026-07-11 |
| **Status** | Ruled |

## Context

The walk-forward regime backtest needed an adjudicated protocol BEFORE
results existed, so that no result could bend the method. The protocol
itself — not just Build 4's numbers — is the reusable decision: every
future harness (Gauge B questions, Build 5 exit variants, the F&G
overlay) inherits it.

## The protocol (the ruling)

1. **Execution model:** signals on close T; ALL simulated fills at T+1
   open; the hypothetical close-fill recorded per switch — the signed sum
   is the EOD-lag cost, reported per-switch and cumulative.
2. **Strategy bodies:** binary (risk-on states → 100% / defensive → cash
   at T-bill) AND graded ladder (100/60/30/0), side by side; if the
   ladder doesn't beat binary, the report says so.
3. **Benchmarks:** buy-and-hold, and the NAKED 200DMA filter under the
   same execution. If the gauge doesn't beat the one-line rule, the
   summary says so.
4. **Threshold honesty (locked):** production config runs once, as-is,
   reported unconditionally. Sensitivity grids train on the early window
   and validate on the late one; both numbers always reported, labeled
   experimental. No post-hoc tuning presented as validation.
5. **Information test** separate from the strategy sim: per-state forward
   returns from T+1 open; duration distributions; flip anatomy. If the
   bullish state's forward returns don't beat the defensive one's, no
   strategy wrapper saves it — report it straight.
6. **Scope fences:** variants requiring unobtainable data are marked
   blocked-on-data and skipped — no proxies.
7. **Lookahead pins are tests, not vibes:** production-replay pin,
   +1-day shift test, walk-window truncation assertion.

## Evidence

The protocol produced [docs/backtest-regime.md](../backtest-regime.md) —
including four honesty clauses that FIRED (gauge loses to 200DMA; info
test inverts; grid winners collapse out-of-sample; the case-study
overclaim caught and corrected by adversarial review). A protocol whose
mandatory sentences all triggered and were printed is a protocol that
works.

## Consequences

Any future "the harness says X" claim is checkable: same pins, same
train/validate discipline, same both-numbers rule.

## Revisit triggers

A harness question the protocol cannot express (e.g. intraday fills for
Build 5's 12:30 variant — which extends rule 1's fill model rather than
replacing the protocol).

## Retest recipe

```
python3 test_regime_extraction.py     # pin 1
python3 test_backtest_regime.py       # pins 2, 3 + anchors
python3 scripts/backtest_regime.py    # full run incl. grid
```

## Links

- Commit: dd60f48
- Docs: [docs/backtest-regime.md](../backtest-regime.md)
- Jira: PER-508 description item 14
