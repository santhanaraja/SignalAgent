# D-014 — TradingView Data API as a candidate yfinance replacement

| | |
|---|---|
| **ID** | D-014 |
| **Date** | 2026-07-13 |
| **Status** | Proposed — parked (evaluation gated on a free-tier spike, gated on a trigger) |

## Context

The data layer is yfinance-via-scraping — functional but flaky, rate-limited, and
occasionally stale (the recurring "paste fresh JSON" friction, intermittent Render
API access). `tradingviewapi.com` surfaced as a candidate **data-source
replacement** — a data layer, **not** charting or alerts. This is the "D-014
data-source" referenced as the daily-breadth revisit path in
[D-012](D-012-fear-greed-rebuild.md) (PER-508 comment 11722).

**Due-diligence flags (weigh before any spend):**
- It is an **unofficial third-party reseller billed through RapidAPI — NOT
  TradingView the company**. No official SLA; terms can change.
- "Real-time" is exchange-dependent — their own docs admit some feeds are 15-min
  delayed even on paid plans (`update_mode`). Low impact for our EOD close-basis
  design ([D-003](D-003-1b-position-engine.md)); would matter only for live execution.

## Options considered

| Option | Summary | Why (not) |
|---|---|---|
| **Pro $10/mo** (30K REST req, Batch API) | Plumbing/reliability upgrade | **The only compelling tier** — our ~hourly pipeline is well under 30K; batch covers the 531 universe. A trivially good yes *if a spike confirms the data*. |
| Ultra $30/mo (WebSocket/SSE realtime + MCP) | Real-time push + MCP tokens | **Rejected** — realtime is architecturally counter to close-basis ([D-003](D-003-1b-position-engine.md)); MCP at $30/mo to save copy-paste is weak ROI. |
| Their "quantitative skills" (Kelly, ATR stops, rotation, breadth, scoring) | Pre-built strategy helpers | **Rejected** — all already built bespoke this month (R28, SMA20/ATR stops, GICS scanner, breadth voter, score engine). A generic worse version of our own system; only the underlying **data** helps. |
| Stay on yfinance | No change | **Default** until a trigger fires — the pipeline runs fine today. |

## Evidence

**None yet — this is the point.** No spend, no decision until a spike produces
evidence. yfinance currently works (the 2026-07-13 maiden flights all flew on it).
A data-source swap must be earned by measurement, not adopted on a pitch.

## Ruling + rationale

**Parked.** No action now. When convenient (not mid-sprint), run the free-tier
spike below, file its results as this record's evidence, **then** rule on Pro $10.
Only the underlying data is of interest; the reseller's strategy "skills" duplicate
our ruled, bespoke architecture and add nothing.

## Consequences

- No spend and no data-layer churn until a trigger fires; the ~$120/yr decision
  stays evidence-gated.
- If the spike shows TradingView data materially more reliable + batch-friendly,
  Pro $10 is an easy yes; if yfinance matches, the free stack is confirmed fine.

## Revisit triggers

- yfinance reliability actually **breaks** the pipeline (not just friction), OR
- a conscious decision to build **real-time / live surfaces** (which itself needs a
  [D-003](D-003-1b-position-engine.md) revisit), OR
- a **universe expansion** strains yfinance scraping.

## Retest recipe

Evidence-before-decision spike (no card required to start):

```
1. Grab the FREE Basic key (150 req/mo).
2. One-evening spike: compare TradingView quotes/OHLCV/TA vs yfinance on ~20
   universe tickers. Checklist: do the numbers match? are OHLCV + RSI/MACD/SMA +
   52w range present? is GICS/sector classification available? how do symbol
   coverage/naming map to ours?
3. File spike results as D-014 evidence, THEN rule on Pro $10.
```

## Links

- Jira: PER-508 comment 11719
- Related: [D-003](D-003-1b-position-engine.md) (close-basis design), [D-012](D-012-fear-greed-rebuild.md) + PER-508 comment 11722 (D-014 as the daily-breadth revisit path)
