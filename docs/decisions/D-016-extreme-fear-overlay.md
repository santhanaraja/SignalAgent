# D-016 — Extreme-fear contrarian entry overlay (backtest-gated hypothesis)

| | |
|---|---|
| **ID** | D-016 |
| **Date** | 2026-07-13 |
| **Status** | Proposed — parked (gated on machinery free + appetite for the F&G historical reconstruction) |

## Context

**The observation (user, ~5yr eyeball):** when CNN Fear & Greed hits lower
single digits, it marks a swing bottom at least good enough for an excellent
trade. Worth testing — **not** bolting on.

**Corroboration and the caveat, both real:**

- **The pattern is independently confirmed:** Build 4's information test found
  Risk-off days had the BEST forward SPY returns (+1.90% at 20d, 70.7% hit) —
  the same mean-reversion-at-fear-extremes effect from a different instrument.
  Two measurements (5yr eyeball + 11yr backtest) agree fear extremes cluster
  near local bottoms.
- **The 5-year window is the trap:** 2021–2026 was an unusually mean-reverting
  regime (COVID recovery, 2022 V-bottoms, repeated Fed-put rescues) where every
  dip got bought. The pattern is real; the MAGNITUDE is flattered by a
  favorable sample. In a genuine secular bear (2000–02, 2008) extreme fear
  printed for MONTHS and early buying was ruinous.

## The philosophical tension (any implementation must resolve it)

Buying extreme fear is a CONTRARIAN falling-knife entry — the direct opposite
of the A+ doctrine ([D-011](D-011-aplus-doctrine.md)), which REFUSES knives
(MRNA graded C for "falling into its mean, momentum down"). An extreme-fear
signal cannot be bolted on; it must be a **deliberate, ruled exception** to the
doctrine with its own guardrails, or it silently reintroduces the knife-buying
the doctrine exists to prevent.

## Why this is now testable (it wasn't before)

This is [D-005](D-005-sentiment-not-a-voter.md)'s F&G-overlay question —
previously blocked on "credible historical F&G data" (CNN publishes no
archive; third-party scrapes fail the no-proxies fence). The user's version is
different and testable: not reconstructing CNN's number, but testing a
**behavioral rule** (buy when OUR OWN computed fear composite hits an extreme)
against a historical reconstruction of our F&G components (VIX, momentum,
breadth, safe-haven — all computable from data the backtest already has).
[D-012](D-012-fear-greed-rebuild.md) began logging live F&G history
2026-07-13 (the forward series accrues).

## Plan (gated; reuses the Gauge B campaign machinery — the throttle sweep is
the MIRROR image: fear extremes ADDING exposure via contrarian release rather
than cutting it)

1. Historical F&G reconstruction (our components, 2015→, point-in-time).
2. Backtest the specific rule: single-digit composite → permitted entry even
   in Caution/Risk-off — with WHAT stop, WHAT size, WHAT hold period. Full
   11yr sample INCLUDING real bears.
3. Survives the full sample → rule as a backtested EXCEPTION to the A+
   doctrine: "extreme-fear override, sized small, stopped tight, permitted
   only below composite X." A conviction with evidence + guardrails.
4. Does NOT survive (works only in mean-reverting 2021–26) → question closed —
   the favorable-sample amplifier was the whole effect.

## Revisit trigger

Gauge B campaign + build complete (machinery free) AND appetite to build the
F&G historical reconstruction. Related: [D-005](D-005-sentiment-not-a-voter.md)
(its overlay question, now testable), [D-011](D-011-aplus-doctrine.md) (the
doctrine this would be a ruled exception to),
[D-012](D-012-fear-greed-rebuild.md) (live F&G history logging).

## Source

PER-508 comment 11723 (2026-07-13). Not a build; a backtest-gated hypothesis
test of a real user observation, protected from the favorable-sample trap by
routing through the full-sample evidence discipline. Record backfilled
2026-07-23 (Phase 3 docs rider) — the D-016 slot was reserved when D-017
shipped first.
