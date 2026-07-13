# D-008 — Gauge B architecture (Q1–Q4)

| | |
|---|---|
| **ID** | D-008 |
| **Date** | 2026-07-12 (proposed and ruled same day — session parts 1 and 2) |
| **Status** | **Ruled** |

## Context

Build 4 fired [D-001](D-001-swing-gauge-1a.md)'s revisit trigger: the
swing gauge's absolute-OAS voter cast zero risk-on votes for nine years,
the gauge lost to the naked 200DMA, and threshold tweaks trained on
2015-2021 collapsed in validation. The indicated fix was architectural,
not parametric. The bear-steepening incident (Jul 8: 30Y through 5.04%
while the spread read "constructive") independently established that
Gauge B must read absolute levels and steepening TYPE, not just spread
sign. Deliberation: [docs/briefs/gauge-b-design-brief.md](../briefs/gauge-b-design-brief.md).

## Rulings (PER-508 comment 11715, 2026-07-12)

- **Q1 — Architecture: TREND CHASSIS.** The 200DMA trend state is the
  core — in-trend/out-of-trend decides direction; VIX, relative credit,
  and breadth become throttle modifiers that scale exposure and entry
  strictness but never override the trend. Rationale: Build 4's demoted
  gate carried the signal (naked 200DMA 9.83% CAGR, de-risked every
  crash while invested, vs the parliament's nine-of-eleven-years
  uninvested). The parliament runs in the harness as the control.

  The deliberation diagram (source:
  [docs/briefs/q1-parliament-vs-chassis.mermaid](../briefs/q1-parliament-vs-chassis.mermaid);
  the "proposed, recommended" label predates the ruling):

  ```mermaid
  flowchart TB
      subgraph A["Option A — Parliament (today, patched)"]
          direction TB
          V1["VIX 5d avg"] --> L["Vote ladder<br/>3/2/1/0 risk-on"]
          V2["HY credit voter"] --> L
          V3["RSP/SPY breadth"] --> L
          L --> G{"200DMA gate<br/>veto only — caps state"}
          G --> S1["Regime state"]
          S1 --> N1["Build 4 verdict:<br/>9 of 11 yrs uninvested · 1.85% CAGR<br/>the demoted gate had the signal"]
      end
      subgraph B["Option B — Chassis (proposed, recommended)"]
          direction TB
          T["<b>200DMA TREND = THE CHASSIS</b><br/>in-trend / out-of-trend decides DIRECTION"] --> M["Throttle modifiers<br/>VIX · relative credit · breadth<br/>scale exposure — never override"]
          M --> E["Exposure % + entry strictness"]
          E --> N2["Evidence: 200DMA alone<br/>9.83% CAGR · −19.8% maxDD<br/>de-risked every crash while invested"]
      end
      style T fill:#0d7d8c,color:#fff
      style N1 fill:#5c2e2e,color:#fff
      style N2 fill:#2e5c3a,color:#fff
  ```
- **Q2 — Credit measure shape: HARNESS DECIDES.** Percentile
  (60d/1y/2y window variants), z-score, and 20d direction-of-change all
  run on the locked train (2015–2021) / validate (2022–2026) split.
  Both numbers reported, always. No shape chosen in the abstract.
- **Q3 — Hysteresis: ASYMMETRIC.** Downgrades instant (fast to defend —
  preserves the crash brake); upgrades require N consecutive closes
  (slow to re-risk — kills the Trending flicker). Symmetric runs as
  harness control; N is a tunable within the split.
- **Q4 — Regime scales R28's total-exposure ceiling, AMENDED numbers
  90 / 50 / 25 / 5** (Trending 90% · Choppy 50% · Caution 25% ·
  Risk-off 5%), replacing the drafted 25/15/5/0.

## Q4 consequences (recorded per the ruling, verbatim in substance)

- **Supersedes R17's static 25% cap and amends R18's 30–40% cash floor
  as written** — both were display-only text (rules extraction, Jul 6);
  their numeric limits are absorbed into R28's regime-scaled ceiling.
  At Trending the implied cash floor is 10%. R28 is the single computed
  authority on aggregate exposure. Supersession noted at R17/R18 in
  [docs/rules.md](../rules.md).
- **Ceiling ≠ target:** deployment still climbs one A+ setup at a time;
  R15 per-position sizing (5–8%) and per-group caps bind first.
  Arithmetic for the record: 90% ÷ 8% max-position ≈ **11+ concurrent
  positions** before the ceiling is reachable — headroom, not a mandate.
- **Risk-off at 5% (not 0%)** deliberately permits one small
  residual/starter position through defensive regimes rather than
  forced full liquidation.

Build-queue consequence: R28 (Phase 0 of [D-007](D-007-theme-layer-retirement.md),
the week's flagship) implements the regime-scaled ceiling with these
numbers from day one.

## Evidence

- [docs/backtest-regime.md](../backtest-regime.md) — the Build 4 record
  (executive summary items 1–4, 6; the grid's train-collapse table).
- PER-508 comment 11711 — bear-steepening requirement (absolute yield
  levels + bull/bear decomposition; FRED source-consistency
  prerequisite).

## Revisit triggers

1. Harness results (Q2 shapes, Q3 N-tuning, Q4 ladder comparison)
   contradicting a ruling on the validate window — each ruling re-opens
   individually.
2. The trend chassis underperforming its own parliament control
   out-of-sample.

## Retest recipe

Per the ruling — the exposure ladders are exactly what the graded-ladder
machinery already simulates; they run on the NEW chassis states once
Q1's rebuild defines them:

```
# Q4 ladder validation through the Build 4 harness (D-006 protocol):
#   90/50/25/5  vs  25/15/5/0  vs  binary — on the new chassis states
python3 scripts/backtest_regime.py            # harness base (ladder body exists today)
# Q2 credit-shape and Q3 hysteresis variants: train 2015-2021 /
# validate 2022-2026, both windows always reported
python3 test_regime_extraction.py             # any new ladder must re-pin production
```

## Links

- Jira: PER-508 comment 11715 (the rulings, verbatim source) · 11711 (bear steepening)
- Brief: docs/briefs/gauge-b-design-brief.md
- Fired-from: [D-001](D-001-swing-gauge-1a.md); protocol: [D-006](D-006-build4-protocol.md); phase order: [D-007](D-007-theme-layer-retirement.md)
