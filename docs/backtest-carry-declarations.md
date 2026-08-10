# Build 7 — implementation declarations

Committed BEFORE any results exist, alongside the pre-registration
([backtest-carry-prereg.md](backtest-carry-prereg.md), `d8f5f42`,
sha256 `4b11ffd563258b70`). The prereg fixes the design, the arms, the
risk-parity formula and the decision table; the resolutions below are
everything the implementation must additionally pin down. They are
declared in advance so no ambiguity is resolved after the answer is
visible. (6A precedent.)

## What "guard OFF" is

1. **Re-graded, not read off the frame.** The committed Layer A frame's
   `grade` column bakes the guard in (row 2 of the REAL `grade_setup`
   at `extension_guard_max = 1.8`). Guard-off is the same
   `process_ticker` loop with that knob at `+inf` — the knob is read at
   exactly two call sites, both `grade_setup` invocations, so every
   feature, every other row and the runway gap-demotion are identical
   by construction. Side-car only: the committed frame is never
   rewritten.
2. **The re-grade integrity gate**: the guard-ON pass must reproduce
   the committed frame's `grade` column on **all 848,328 rows**, not a
   sample. Plus a monotonicity assertion — removing the guard may only
   ever LIFT a grade, never lower one. Both are in-run assertions and
   pinned.
3. The arms consume the grade ladder as 5B's do: S1/S8 read `grade`,
   S6/S7 read `grade_noguard`. The gate LOGIC is byte-identical
   (B-or-better in Trending, A+-only in Choppy) — only the column
   differs, so nothing but row 2 varies between guard-on and guard-off.

## Pairing — how S8 and S7 are built

4. **S8/S7 are FORCED-ENTRY replays of S1/S6's realized populations**
   (6A's construction, which the prereg invokes by name: "as 6A did").
   S8 replays every entry S1 actually took; S7 replays every entry S6
   actually took; admission gates (ceiling, R28 caps, cash) are NOT
   re-evaluated per arm, because re-evaluating them against a diverged
   equity path would change the population and destroy the exact
   pairing the prereg requires ("identical trades, different share
   counts"). Consequence, declared: **R28/ceiling bind rates are
   measurable only on S1 and S6** — the sims that actually evaluate
   admission. Reported as such, never imputed to S8/S7.
5. Two loud in-run guards on that construction, as in 6A: an arm's
   cash may never go negative, and a scheduled entry may never collide
   with a still-open position.

## Risk-parity mechanics

6. `risk_budget` is the prereg's "S1's REALIZED mean risk-per-trade as
   a % of capital", measured from S1's own output as the mean over
   S1's trades of `r_usd / equity_at_entry`. (Identically:
   `0.065 × mean(1 − stop/fill)` — fixed sizing spends 6.5% of equity
   and risks the fraction of it between fill and stop.) Capital basis
   = the arm's **equity at the entry's fill day**, the same basis the
   fixed 6.5% uses, so "matched aggregate risk" is matched in the same
   units on both sides.
7. `shares = min(risk_budget × equity_prev / (fill − stop),
   0.12 × equity_prev / fill)`, per the prereg's formula with the
   declared capital basis. Trades where `fill ≤ stop` cannot exist —
   S1/S6 already refuse them (`gap_below_stop`), so the population
   inherits that refusal.
8. **CAP-BIND** is recorded per trade as the boolean "the cap term was
   the binding side of the min", and the rate is reported per arm
   BEFORE the conclusions, per the prereg's validity check. Above 50%
   the prereg voids decision clauses 5 and 6 and the report says so.

## Statistics

9. **Guard effect (unpaired)**: S1 vs S6 and S8 vs S7, read against 5B's
   random-K selection-noise bands — 50 seeds, the same construction and
   seed count 5B used, run on the arm's own admitting gate so the band
   measures selection noise within that arm's candidate pool. A
   difference inside the band is not distinguishable, per prereg
   branch 4.
10. **Sizing effect (paired)**: per-trade differences bootstrapped
    directly, resampled by ENTRY-DATE cluster (same-day entries are
    correlated), 2,000 draws, seeded — 6A's construction. Reported in
    R-multiples (scale-free) and in the portfolio metrics.
11. **Portfolio-metric CIs** (ΔCAGR, ΔmaxDD) for the paired comparisons
    use the paired 21-day moving-block bootstrap of daily returns, as
    6A declared: identical block indices applied to both arms.
12. **The interaction** is reported as (S8−S1) versus (S7−S6) on each
    primary metric, with the prereg's instruction honoured: if the
    sizing effect differs materially with the guard on versus off, the
    two are NOT pooled and both are reported separately.
13. Decision read on the FULL window; train/validate reported for every
    arm per D-006. Expectancy per R is 5B's dollar-weighted
    `sum(PnL)/sum(R_usd)` (the tiny-R degeneracy ruling), with the
    per-trade mean and median beside it.

## Amendment 1 — the cash constraint (declared 2026-08-10, BEFORE any
performance result was computed)

Discovered at implementation: under the prereg's formula a
risk-parity arm can want more capital than it holds. Tight-stop names
(entry close to the SMA20) pull `risk_budget/(fill−stop)` far above
the cap, so the cap binds at 12% per position, and concurrent
positions at 12% exhaust cash long before the 90% regime ceiling
would. The smoke replay hit exactly this and stopped on the declared
cash assertion. **No arm's CAGR, drawdown or expectancy had been
computed when this was written** — only the integrity gates, the trade
counts and the derived risk budget.

Resolution, chosen for fidelity to the prereg's STATISTICAL
requirement:

16. `shares = min(risk_budget×equity/(fill−stop), cap×equity/fill,
    affordable)` where `affordable = cash/(fill×(1+slip))`. The trade
    stays in the population, so the pairing the prereg demands
    ("identical trades, different share counts") survives intact; the
    alternative — denying the entry, as 5B's own admission does — would
    diverge the population and destroy the paired comparison the prereg
    calls the higher-powered half of the study.
17. **CASH-BIND is reported as a first-class validity number** beside
    cap-bind, per arm. A high cash-bind rate means the arm is not
    really being sized by risk parity either — it is being sized by its
    own funding limit — and the report says so in those words. If
    cash-bind is material the same reasoning as the prereg's cap-bind
    clause applies and is stated explicitly for the ruling.
18. Declared NOT tested here: a denial variant (skip the entry when
    cash is short, as the live system would). That is a different
    population and therefore a different study.

## Scope

14. Scorer **v1** throughout (prereg). The frozen v1 stack is what the
    committed frame was built on; the transfer of any conclusion to v2
    grades is an assumption, stated in the report, not a finding.
15. This study ships NOTHING to production. Its output is a verdict
    against the prereg's six branches. No threshold study is
    authorised by it (prereg branch 1 says so explicitly).
