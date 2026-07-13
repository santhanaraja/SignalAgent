# D-005 — Sentiment is not a voter; F&G overlay gated on credible historical data

| | |
|---|---|
| **ID** | D-005 |
| **Date** | 2026-07-06 |
| **Status** | Ruled |

## Context

CNN Fear & Greed printed "Fear" while the swing gauge printed Risk-on /
Trending, raising the question: should sentiment become a 4th voter?

## Options considered

| Option | Summary | Why (not) |
|---|---|---|
| 4th voter | Sentiment joins the tally | **Rejected — polarity is state-dependent.** Sentiment is contrarian at extremes and noise mid-range; its correct vote-sign depends on the state you're trying to infer. A state-dependent-polarity signal cannot be averaged into a tally of fixed-sign voters without breaking the architecture. Also heavy component overlap (VIX, HY credit, momentum ≈ the gate) = double counting |
| Extremes-only advisory overlay | Silent mid-range; text-only nudges at >~80 / <~20 | Candidate shape, OUTSIDE the gauge — gated (below) |
| Contrarian sizing modulator | R15's 5–8% band sized against sentiment extremes | Candidate shape, rules layer — same gate |

## Ruling + rationale

Sentiment does not vote. Any future life is display/sizing, never a
state input, and only if a walk-forward variant proves the overlay
improves risk-adjusted swing outcomes.

## Consequences

The 3-voter gauge stays undiluted. The overlay question is empirical and
currently **blocked on data**: Build 4 (Decision 6, scope fences) found
no credible daily historical F&G series from a reputable source — no
proxies allowed, so the variant was skipped, not faked.

## Revisit triggers

A credible, reputable daily historical Fear & Greed series to ~2015
becoming obtainable. That is the whole gate.

## Retest recipe

Blocked on data — nothing runs today. What WOULD run: the Build 4
harness with an F&G-extremes overlay variant beside the plain gauge,

```
python3 scripts/backtest_regime.py   # + overlay variant when data exists
```

both windows reported per the Build 4 protocol ([D-006](D-006-build4-protocol.md)).

## Links

- Jira: PER-508 comment 11708
- Docs: [docs/backtest-regime.md](../backtest-regime.md) limitations §6
