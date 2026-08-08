# Build 6A — implementation declarations

Committed BEFORE any results exist, alongside the pre-registration
([backtest-target-prereg.md](backtest-target-prereg.md), `d184ced`).
The prereg fixes the design and declares the three big mechanics; the
resolutions below are everything the implementation must additionally
pin down. They are declared here, in advance, so no ambiguity is
resolved after the answer is visible.

## Baseline and machinery

1. **T0 is the 5B S1 arm, score ordering** — the production entry gate
   under the selection ordering the system itself endorses. Integrity
   gate, two-sided: (a) re-running the 5B engine must reproduce the
   committed `backtest-systems-results.json` S1/score figures; (b) the
   NEW engine with the target rule switched off must reproduce the 5B
   engine's S1 curve and trade log **to the cent**. Every arm statistic
   rests on that parity.
2. **Entry population fixed by construction.** T0's realized entries
   (ticker, fill day, signal-day stop) are replayed as FORCED entries
   in every arm; admission gates (ceiling, group caps, cash) are not
   re-evaluated per arm — re-evaluating them against a diverged equity
   path would change the population, which the prereg forbids. Two
   loud in-run assertions guard the construction: an arm's cash must
   never go negative, and a scheduled entry must never collide with a
   still-open position (floors only ever RAISE stops, so arms exit at
   or before T0 — the assertion proves it).
3. **Sizing: 6.5% of the arm's own equity at fill.** Compounding is
   the system's own — that is what CAGR measures. Per-trade pairing is
   therefore computed in **R-multiples**, which are share-scale
   invariant (shares scale pnl and r_usd identically); USD figures are
   descriptive only.
4. **R per share = entry fill − signal-day SMA20** (the initial stop
   production and 5B define). Target levels = fill + k·R, set once at
   entry, never re-based. Trades T0 dropped for R ≤ 0 (gap below stop)
   do not exist in any arm — the population is T0's.

## Target-order mechanics

5. **Resting limits are live from the entry fill**, so a target may
   fill on the entry day itself. A day's fill requires a printed bar:
   halted/missing days fill nothing.
6. **Fill price**: open if the day opens at/through the level
   (open ≥ level — a gap fills at the better price), else the level
   itself when the day's high touches it (high ≥ level). Sell-side
   cost of 5 bps applies to every leg, target or stop.
7. **Multiple levels in one day both fill** (T3: a high through 4R
   fills the 2R and 4R legs the same day, each at its own price per
   rule 6). Fractions are of ORIGINAL size: T1/T2 half; T3 exact
   thirds (the trailed remainder is the residual third).
8. **Same-day precedence**: a stop that fired on yesterday's close
   sells the WHOLE remaining position at today's open and cancels all
   resting limits. Intrabar targets precede the same evening's
   close-basis stop check.
9. **Breakeven floors**: T4 arms intrabar the first day high ≥ entry
   + 2R (no sale); T5 arms on the day the 3R leg fills. An armed floor
   applies from THAT day's close check onward. Floored stop level =
   max(SMA20, entry fill); the exit test stays production's
   close-not-above, equality exits (D-018).
10. **A trade is all its legs.** It closes when the last share exits;
    pnl = all proceeds − full cost basis; r_mult = pnl / r_usd on the
    original size; hold_days runs to the final exit. Mark-out of
    window-end survivors is unchanged from 5B (synthetic close, no
    slippage, conservation identity holds to the cent). Partial-sale
    proceeds sit in cash and earn the IRX daily yield — the 5B cash
    convention; that yield is part of the arm's curve, as the prereg's
    "returns to CASH" requires.

## Statistics

11. **Per-trade paired bootstrap** (the prereg's literal instruction):
    paired differences in R-multiples vs T0, resampled by ENTRY-DATE
    cluster (same-day entries are correlated), 2,000 draws, seeded.
12. **CAGR and max-drawdown CIs**: the decision table's clauses name
    portfolio-level quantities, so they are read from a **paired
    moving-block bootstrap of daily returns** — identical 21-day block
    indices applied to both arms' return series, paths rebuilt, ΔCAGR
    and ΔmaxDD computed per replicate, 2,000 draws, seeded. The
    per-trade CI from (11) is reported alongside every verdict.
    Interpretive step, stated now: branch reading uses (12); (11) is
    supporting evidence.
13. **Decision read on the FULL window**; train/validate reported for
    every arm per D-006. Both spans of every metric appear in the
    results JSON.
14. **Concentration check**: per-trade contribution = paired R-multiple
    difference × the trade's T0 r_usd (T0-scale dollars, immune to
    compounding divergence); the top-3 share is |top-3 sum| / |total
    sum|. Targets-reached counts are reported per arm and per level.
15. **Exchange rate** (branch 2): CAGR points surrendered per point of
    max drawdown saved, full window, from the point estimates.
