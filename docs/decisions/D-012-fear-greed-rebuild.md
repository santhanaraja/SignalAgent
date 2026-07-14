# D-012 — Fear & Greed component de-duplication + daily persistence

| | |
|---|---|
| **ID** | D-012 |
| **Date** | 2026-07-13 |
| **Status** | Ruled |

## Context

The Fear & Greed page ([feargreed.html](../../public/feargreed.html), `fear_greed_engine.py`)
showed seven equally-weighted components but only **five independent inputs** —
a structural flaw visible at a composite-83 Extreme-Greed reading. Display-only:
[D-005](D-005-sentiment-not-a-voter.md) rules sentiment is not a gauge voter, so
stakes are low — which buys the room to do it right. Deliberation brief:
[fear-greed-redesign-brief.md](../briefs/fear-greed-redesign-brief.md). The
pre-change formulas were first extracted to [docs/sentiment.md](../sentiment.md)
(mandatory Step 0).

## Options considered

| Option | Summary | Why (not) |
|---|---|---|
| (a) Minimal | Delete row 5; run 6 components | Kills the VIX double-count but loses a slot; TLT overlap remains |
| **(b) Targeted swaps** | Row 5 → "Market Internals" (% of universe above 50DMA); row 7 → HYG vs LQD | **Chosen.** VIX and TLT each appear once; two rows upgrade from proxy to computed truth; 7-row structure + equal weights preserved |
| (c) Full redesign | Rebuild all 7 on universe internals | Best data, biggest build, loses CNN comparability |
| Q2 persistence | Log the daily composite + components | **Adopted** — starts the historical series D-005's overlay is blocked on |

## Evidence

Two double-counts, confirmed in code and recorded in [docs/sentiment.md](../sentiment.md):
**VIX** drives rows 4 (Put/Call proxy) and 5 (Market Volatility); **TLT** drives
rows 6 (Safe Haven, SPY−TLT) and 7 (Junk Bond, HYG−TLT). CNN's junk-demand
measure is junk **vs investment-grade** (HYG vs LQD), not junk vs Treasuries.
The system already computes daily data for a ~530-ticker universe — real internals
CNN can only approximate through index proxies.

## Ruling + rationale

Two surgical swaps, each independently justified. **Row 5 → "Market Internals"** =
% of the active universe above its 50-day MA (genuine breadth, read from the
committed universe artifact, no new fetch). **Row 7 → HYG vs LQD** 20-day spread
(the honest credit-quality spread). Row 4 keeps its VIX-trend proxy **with an
honest label** ("Put/Call (VIX proxy)" — options data still doesn't exist here).
Result: **7 rows, 7 independent inputs**, equal weights and all raws preserved.
Plus **daily persistence** — `{date, composite, 7 components + raws}` upserted once
per day into `data/fear_greed_history.json`, riding the data commit.

## Consequences

- VIX and TLT each vote once; the composite is no longer 2/7 auto-greed on one
  low-VIX reading.
- **Market Internals is weekly-stepping, not intraday.** The universe artifact is
  rebuilt only on the Saturday rotation, so per-name above-50DMA reflects the prior
  Friday close. This is the accepted resolution of a three-way collision —
  531-universe **and** no-new-fetch **and** daily can't all hold with current
  artifacts (the only daily source is the ~54-name dashboard set, wrong population;
  a daily 530-name recompute is a new fetch). Data integrity beat "daily"; wording
  is honest in code and [docs/sentiment.md](../sentiment.md). Revisit path parked in
  PER-508 (comment 11722).
- Persistence begins accruing the credible daily F&G series D-005's overlay variant
  is gated on — logged from the *improved* component set (the right order).
- The `above_50dma` field is now emitted per universe-ranking ticker; a temporary
  derive-from-`ma`-component fallback covers artifacts built before the field
  shipped (remove after the 2026-07-18 rotation confirms it populates).

## Revisit triggers

- Any component's raw input becoming shared again (a new double-count).
- Weekly-stepping breadth visibly lagging real breadth during a fast mid-week move
  → the PER-508 daily-breadth revisit path.
- The D-005 F&G-overlay backtest (once history accrues) wanting a genuinely daily
  breadth input.

## Retest recipe

```
python3 test_fear_greed.py     # score-curve pins, breadth fixture, persistence matrix
```

The D-005 overlay variant runs through the Build 4 harness once the persisted
series is long enough — see [D-005](D-005-sentiment-not-a-voter.md).

## Links

- Jira: PER-508 comment 11718 (ruling); comment 11722 (parked daily-breadth path)
- Brief: [docs/briefs/fear-greed-redesign-brief.md](../briefs/fear-greed-redesign-brief.md)
- Commits: `7450676` (rebuild + persistence), `589e14c` (wording), `39934a6` (Step 0 extraction)
- Docs: [docs/sentiment.md](../sentiment.md) (extraction + motivating defects)
