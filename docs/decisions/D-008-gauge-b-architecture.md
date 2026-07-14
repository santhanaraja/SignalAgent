# D-008 — Gauge B architecture (Q1–Q4)

| | |
|---|---|
| **ID** | D-008 |
| **Date** | 2026-07-12 (proposed and ruled same day — session parts 1 and 2) |
| **Status** | **Ruled** · amended 2026-07-13 (harness campaign complete; throttle calibration locked — PER-508 comment 11724) |

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

## Amendment — 2026-07-13: harness campaign complete, throttle locked, Gauge B fully specified

The Q2/Q3/Q4 "harness decides" questions were run
([docs/gauge-b-campaign.md](../gauge-b-campaign.md), commit `7904bd6`) and the last
open parameter — the throttle calibration — was chosen from the frontier sweep
([docs/gauge-b-throttle-sweep.md](../gauge-b-throttle-sweep.md), commit `6603862`).
**Gauge B is now fully specified** (PER-508 comment 11724). The campaign's honest
verdict: the chassis is a drawdown-reducer, not a return-enhancer — it loses CAGR to
the naked 200DMA but wins Sharpe/Sortino/maxDD, and the return gap is throttle
calibration (the direction layer alone, `binary(chassis)`, posts 10.13% full CAGR).

**The evidence-chosen parameter set:**

| Question | Chosen | Evidence |
|---|---|---|
| Q1 architecture | trend chassis (200DMA core; VIX/HY/breadth throttle) | fixes the parliament (1.85% → 9.10% CAGR at the locked config); the 200DMA carried the signal |
| Q2 credit shape | **pctile_60d** (HY OAS 60-day trailing percentile) | top train **and** validate Sharpe of the six shapes |
| Q3 hysteresis | **asymmetric N=2** | halves the flicker (In-Trend-Full runs≤2d 77→30 full-sample; whipsaws 21.6→12.9/yr on validate); asymmetric beats symmetric at equal N |
| Q4 ladder | **90/50/25/5** regime→R28 ceiling | dominates the drafted 25/15/5/0 (7.31% vs 3.52% full CAGR) |
| **Throttle** (this amendment) | **k1 · vix22 · hy90 · br−0.5** — the Quality point | best risk-adjusted point on the 240-combo frontier |

**Locked throttle — the Quality point** (the In-Trend Full→Throttled cut-point: a
single throttle firing at these levels downgrades exposure):
`require_k=1 · vix_cut=22 · hy_pctile_cut=90 · breadth_cut=−0.5`.

- Full-window: **9.10% CAGR · −11.42% maxDD · Sharpe 0.88 / Sortino 1.22** (best
  risk-adjusted on the entire frontier). Validate: 7.59% CAGR.
- vs naked 200DMA (9.83% / −19.82% / 0.68 / 0.92): slightly less CAGR, **~half the
  drawdown**, clearly better risk-adjusted.

**Rationale (a values decision, recorded as such):** chosen over the return-matcher
(k2/vix20/hy95/br−2: 9.77% / −14.69%) because for a day-job swing trader managing
family capital, the drawdown one can **hold through** is the only return that
compounds — −11% is survivable, ~−20% (the naked 200DMA's own drawdown) is where
strategies get abandoned. Gave up
~0.7pp full-window CAGR for ~3pp less drawdown and materially better risk-adjusted
metrics; Sortino 1.22 (edging buy-and-hold's 1.01) confirms the downside efficiency.

**Honest caveat (D-006):** the frontier is full-window Pareto; out-of-sample is
softer — validate CAGR 7.59% sits below the 200DMA's 9.65% and the validate drawdown
edge is narrower. The honest claim is **not** "dominates the benchmark out-of-sample"
— it is "full-window strong; out-of-sample slightly-less-return / moderately-less-
drawdown; best risk-adjusted point available." The choice optimizes for survivable
drawdown (the stated goal), not benchmark-relative CAGR.

**Build status:** every Gauge B parameter — chassis, credit shape, hysteresis,
ladder, throttle — is now evidence-chosen. The Fable Ultracode build (post-Friday)
implements exactly this config, under the Build 4 step-0 extraction-pin discipline
(the `compute_regime` replay must reproduce recorded production history on unchanged
inputs). No open parameters remain.

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
3. Live Gauge B drawdowns materially exceeding the −11.42% backtest
   expectation of the locked Quality point, OR a preference shift toward
   return-over-smoothness — move up the throttle frontier (preserved as the
   menu in the throttle-sweep doc) toward the return-matcher.

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

- Jira: PER-508 comment 11715 (the Q1–Q4 rulings) · 11711 (bear steepening) · 11724 (throttle amendment — Gauge B fully specified)
- Campaign: [docs/gauge-b-campaign.md](../gauge-b-campaign.md) (commit 7904bd6) · [docs/gauge-b-throttle-sweep.md](../gauge-b-throttle-sweep.md) (commit 6603862)
- Brief: docs/briefs/gauge-b-design-brief.md
- Fired-from: [D-001](D-001-swing-gauge-1a.md); protocol: [D-006](D-006-build4-protocol.md); phase order: [D-007](D-007-theme-layer-retirement.md)
