# North Star

The enduring goal this system is built toward, and the principles that decide
what ships. Specific rulings live in [the decision registry](decisions/README.md);
this doc is the direction those rulings serve, plus a dated changelog of what has
landed. Every principle below traces to a record — this is a synthesis, not a new
mandate.

## The goal

A personal swing-trading decision system that **converts trader discretion into
computed, re-derivable rules**, and **tells you the read instead of making you
fetch it**. It reads the market's regime, scans a GICS universe for tradable
leadership, tracks each holding's entry/exit/re-entry state, and enforces
position and concentration discipline in real dollars — surfacing all of it as
one daily assessment.

## Principles (each traceable to a record)

- **Complexity must earn its keep — evidence decides, never taste.** Walk-forward
  backtests on a locked train/validate split rule on every added voter or layer; no
  pre-tuning ([D-006](decisions/D-006-build4-protocol.md), [D-008](decisions/D-008-gauge-b-architecture.md)).
- **Computed authority, not displayed discretion.** Rules that govern money are
  computed from live data with dollar numbers, not static reminder text — R28 is the
  single authority on exposure ([D-007](decisions/D-007-theme-layer-retirement.md),
  [D-008](decisions/D-008-gauge-b-architecture.md) Q4).
- **The scanner is the thesis.** No hand-curated conviction layer between the trader
  and the universe scanner + quality gate; concentration discipline lives in dollars
  and per-group caps ([D-007](decisions/D-007-theme-layer-retirement.md)).
- **Every decision logic gets a Lab.** Wherever the system embodies trading logic,
  the page gets an interactive what-if surface backed by the *real* production
  function — the lab owns zero math ([D-010](decisions/D-010-lab-pattern-laws.md)).
- **Display-only stays display-only until a ruled consumer exists.** Sentiment and
  Fear & Greed inform; they do not vote ([D-005](decisions/D-005-sentiment-not-a-voter.md)).
- **No ruling without its retest recipe.** A decision you cannot re-derive is one
  you cannot trust when conditions change ([registry standing rule](decisions/README.md)).
- **The system tells you; you don't ask it.** Push the daily read to where the user
  is, rather than fetching seven endpoints (PER-508 items 21/22).

## Changelog

Newest first. Dates are when the work landed on `main`.

### 2026-07-13
- **R28 real-dollar portfolio enforcement live** — per-position (R15), per-GICS-group
  caps (≤20% / ≤3, amended), and the D-008 Q4 regime-scaled ceiling (90/50/25/5),
  computed in dollars against real `account_capital_usd` **$97,500**; serving on the
  framework page + assessment.json ([D-007](decisions/D-007-theme-layer-retirement.md), [D-008](decisions/D-008-gauge-b-architecture.md)).
- **Fear & Greed de-duplicated + persisted** — 7 rows, 7 independent inputs (VIX and
  TLT once each); "Market Internals" universe breadth replaces the 2nd VIX; HYG vs LQD
  replaces HYG vs TLT; daily history logged ([D-012](decisions/D-012-fear-greed-rebuild.md)).
- **Sentiment rebuilt as per-ticker behavioral analysis** — Technical Sentiment (pure
  function → simulate endpoint), relative strength, news strip; VADER + StockTwits
  retired ([D-013](decisions/D-013-sentiment-rebuild.md)).
- **Fear & Greed + Sentiment formulas extracted** to [docs/sentiment.md](sentiment.md)
  — the last un-extracted surfaces.

### 2026-07-12
- **Theme layer retired** — the GICS universe scanner + quality gate becomes the
  thesis; concentration moves to computed dollars, staged Phase 0→3
  ([D-007](decisions/D-007-theme-layer-retirement.md)).
- **Gauge B architecture ruled** — trend chassis, harness-decided credit shape,
  asymmetric hysteresis, regime-scaled R28 ceiling ([D-008](decisions/D-008-gauge-b-architecture.md)).
- **A+ Doctrine ruled** — computed setup grade: composite approach filter, 7-item
  checklist, ≥15-td earnings runway, hard-gate in Choppy/Caution ([D-011](decisions/D-011-aplus-doctrine.md)).

### 2026-07-11
- **Build 4 walk-forward regime backtest** (2015–2026) + reusable protocol
  ([D-006](decisions/D-006-build4-protocol.md)); **Score, Gauge, and Position Labs**
  shipped; **the Lab pattern** ruled as a standing three-law rule
  ([D-010](decisions/D-010-lab-pattern-laws.md)); exit-timing 12:30 checkpoint
  proposed ([D-009](decisions/D-009-exit-timing-1230.md), gated on Build 5).

### 2026-07-05 — 07-09
- **Build 1A swing gauge** (3 voters + backdrop gate + exact-spec ladder,
  [D-001](decisions/D-001-swing-gauge-1a.md)), **1B position engine**
  ([D-003](decisions/D-003-1b-position-engine.md)), **R4 Sunday cadence**
  ([D-002](decisions/D-002-r4-sunday-cadence.md)), **extension guard @ 1.8×ATR**
  ([D-004](decisions/D-004-extension-guard.md)), **sentiment-not-a-voter**
  ([D-005](decisions/D-005-sentiment-not-a-voter.md)).

<!-- Add newer entries at the top of the changelog, under a dated heading. -->
