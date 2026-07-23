# Decision Registry

Every design decision that shapes what this system trades on lives here
as a record with its evidence, its revisit triggers, and the exact recipe
to re-run that evidence. The pattern is ADR (architecture decision
records) plus one hard addition: **retest recipes** — a ruling you cannot
re-derive is a ruling you cannot trust when conditions change.

**The standing rule:** "No design change ships without a decision record;
no record without a revisit trigger; no ruling without its evidence
linked."

**How decisions get judged here:** every record stores its options
considered and a retest recipe — stored counterfactuals. The method
itself is taught in
[the counterfactual-reasoning explainer](../explainers/counterfactual-reasoning.md)
(change the DECISION, never the INFORMATION).

Deliberation briefs (the pre-ruling analysis documents) live in
[`docs/briefs/`](../briefs/); records link back to theirs.

The direction these records serve — plus a dated changelog of what has landed — is
in [`docs/NORTH-STAR.md`](../NORTH-STAR.md).

## Lifecycle

```mermaid
flowchart TD
    B["Deliberation brief (docs/briefs/)"] --> R["Ruling"]
    B -->|"gated on evidence"| P["PROPOSED record —
    parked WITH its retest recipe"]
    P -->|"gate opens (evidence lands)"| R
    R --> D["Decision record with RETEST RECIPE"]
    D --> T{"Revisit trigger fires"}
    T -->|yes| O["Re-open"]
    O --> H["Harness retest (run the recipe)"]
    H --> A["Amend, or supersede:
    new record; old one marked Superseded-by-D-xxx"]
    A --> D
```

## Index

| ID | Title | Date | Status | Superseded by |
|---|---|---|---|---|
| [D-001](D-001-swing-gauge-1a.md) | Build 1A swing gauge — 3 voters + backdrop gate + exact-spec ladder | 2026-07-05 | Ruled — **revisit fired 2026-07-11**; successor architecture ruled (D-008), supersession lands when Gauge B ships | — (pending Gauge B build) |
| [D-002](D-002-r4-sunday-cadence.md) | R4 Sunday-cadence qualification, 2 degraded weekly closes | 2026-07-05 | Ruled | — |
| [D-003](D-003-1b-position-engine.md) | 1B position engine — 5 conditions, close-basis stops, positions.json authoritative | 2026-07-05 | Ruled | — |
| [D-004](D-004-extension-guard.md) | Extension guard @ 1.8×ATR | 2026-07-09 | Ruled | — |
| [D-005](D-005-sentiment-not-a-voter.md) | Sentiment is not a voter; F&G overlay gated on credible data | 2026-07-06 | Ruled | — |
| [D-006](D-006-build4-protocol.md) | Build 4 backtest protocol (reusable) | 2026-07-11 | Ruled | — |
| [D-007](D-007-theme-layer-retirement.md) | Theme layer retirement — scanner + quality gate as thesis, R28 dollars, Option C staged | 2026-07-12 | Ruled | — |
| [D-008](D-008-gauge-b-architecture.md) | Gauge B architecture (Q1–Q4: trend chassis, harness-decided credit shape, asymmetric hysteresis, regime-scaled R28 ceiling 90/50/25/5) | 2026-07-12 | Ruled | — |
| [D-009](D-009-exit-timing-1230.md) | Exit timing — 12:30 intraday checkpoint | 2026-07-11 | **Proposed** — gated on Build 5 evidence | — |
| [D-010](D-010-lab-pattern-laws.md) | The Lab pattern three laws | 2026-07-11 | Ruled | — |
| [D-011](D-011-aplus-doctrine.md) | The A+ Doctrine — computed setup grade (composite approach filter, 7-item checklist, ≥15td earnings runway, hard-gate Choppy/Caution) | 2026-07-12 | Ruled | — |
| [D-012](D-012-fear-greed-rebuild.md) | Fear & Greed de-duplication (7 rows → 7 independent inputs) + daily persistence | 2026-07-13 | Ruled | — |
| [D-013](D-013-sentiment-rebuild.md) | Sentiment rebuilt as per-ticker behavioral analysis (Technical Sentiment + relative strength + news; VADER/StockTwits retired) | 2026-07-13 | Ruled | — |
| [D-014](D-014-tradingview-data-api.md) | TradingView Data API as a candidate yfinance replacement | 2026-07-13 | **Proposed** — parked (gated on a free-tier spike, gated on a trigger) | — |
| [D-015](D-015-oss-frameworks-survey.md) | Survey serious OSS trading frameworks for borrowable techniques (harvest, don't migrate) | 2026-07-13 | **Proposed** — parked (gated on the build queue clearing) | — |
| [D-016](D-016-extreme-fear-overlay.md) | Extreme-fear contrarian entry overlay — backtest-gated hypothesis (ruled exception to D-011 ONLY if it survives the full 11yr sample) | 2026-07-13 | **Proposed** — parked (gated on machinery free + F&G reconstruction appetite) | — |
| [D-017](D-017-candidates-tier.md) | Candidates tier — auto-grade every signals.json name (grade without a state); chip-only display; copy-the-prompt +watch; close-report line | 2026-07-18 | Ruled | — |

Status meanings: **Proposed** (deliberation open or parked with its
retest recipe) · **Ruled** (in force) · **Superseded-by-D-xxx** (kept for
the record; the successor governs) · **Retired** (no successor needed —
the decided-about thing no longer exists).
