# The Systems Replay — Build 5B: does the doctrine earn its keep with stops, sizing and the ladder in play?

**Build 5B · 2026-07-26.** Script: `scripts/backtest_systems.py` ·
Results: [backtest-systems-results.json](backtest-systems-results.json)
(`results_hash e1117af63402f874`, recomputable from the committed file) ·
Pins: `test_backtest_systems.py` (13) · Pre-registration:
[backtest-systems-prereg.md](backtest-systems-prereg.md), committed
(`68d620c`) **before any results existed**.

*Methods note up front: adversarial review of the first full run found a
genuine D-006 lookahead — ceiling admission at the open of day D read
day-D's close-derived regime state. Fixed (admission and the stored
regime label now use the signal-day state, pinned by injection both
directions), the study fully rerun, and this report written once,
against the corrected JSON. The lookahead's removal mattered: it had
been suppressing S4 (+40% relative expectancy when corrected) and it
manufactured the only pair difference that cleared the noise band.*

## Verdict, read against the pre-registration

Applying the two pre-registered tests — a difference counts only if it
**(1) clears the random-K selection-noise band** and **(2) holds sign
across {0, 5, 15} bps costs**:

> **All ten arm-vs-arm differences fail test 1.** Every pairwise
> expectancy difference sits inside its paired 50-seed random-selection
> band, and the date-cluster bootstrap agrees — **no pair's 95% CI
> excludes zero** (closest: S1−S3, p = 0.093). This is
> **pre-registered branch 5: THE DOCTRINE IS UNFALSIFIED BUT
> UNSUPPORTED.** Per the branch's own text, the hard gate softens to
> advisory by default — an unproven restriction should not bind live
> capital — with the ruling itself left to the D-011 revisit
> deliberation.

- **Branch 6 does not fire**: the "nothing counts" verdict is robust
  across all three orderings — native-selection expectancies span
  0.343–0.581 (S3 low-ext 0.343, S1 0.467, S2 0.494, S4 momentum
  0.581), every native pair difference still sits inside the
  corresponding random-K band, and no ordering makes any pair clear.
- **Branches 1–3 do not fire**: nominally the *stricter* doctrine arm S2
  now leads (0.425R, and first at **all three cost levels**), S1 second
  — but nominal-inside-the-band is exactly what the prereg was written
  to refuse to crown. Reported, not counted.
- **Branch 4 does not fire**: the arms beat SPY buy-and-hold on
  risk-adjusted terms (S1/S2 Sharpe 0.91/0.93, MDD −17.5/−13.7 vs SPY
  0.677/−33.7; validate 1.20/1.25 vs 1.07) — though in the **train**
  span alone the arms' Sharpe edge over SPY is thin, and the honest
  attribution stands: **chassis-on-SPY with zero stock selection already
  delivers Sharpe 0.751 / MDD −18.1**. The machinery (ceiling + stops)
  supplies the bulk of the risk-adjusted profile; the stock layer adds
  ~+0.15–0.18 Sharpe and validate CAGR; the *gate choice within it* adds
  nothing measurable.

## Design (locked at the Step-0 hold; amendments applied)

| | |
|---|---|
| Arms | S1 production gate (B@Trending / A+@Choppy) · S2 A+ everywhere · S3 B-or-better · S4 top 6m-momentum decile · S5 score≥50; S4/S5 carry only the c1 stop-coherence requirement |
| Benchmarks | SPY B&H · equal-weight pool B&H (window-start names) · chassis-on-SPY |
| Mechanics (fixed) | Signals at close T → fill next open · stop = production `sma20_close` (**exit when close is NOT above SMA20** — exact equality exits, matching `assess_position`; confirmed closes, D-018; sell next open) · sizing 6.5% of equity · group ≤20% & ≤3 · ceiling 90/50/25/5 gates **new entries only, under the signal-day state** · committed regime series · grade-lite |
| Selection | Headline: score-order · Null: random-K, 50 seeds · Native: momentum (S4), lowest-extension (S1–S3), score (S5) |
| Costs | $0 commission · 5 bps/side (sweep {0, 15}), slippage-aware cash check · cash earns IRX |
| R | Entry fill − SMA20 at signal. **Expectancy = dollar-risk-weighted `sum(PnL)/sum(R_usd)`** (declared at smoke stage, before results: the per-trade mean degenerates on near-zero-R entries; mean and median reported beside it) |
| No profit target | The R24 gap, by design; the R right tail is unbounded. 5B measures entry gate + stop only |
| Re-entry | Permitted whenever the gate re-passes; a queued entry whose ticker prints no bar on the fill day is dropped, counted, never retried (declared) |

## Headline table (score-order, 5 bps, 6.5% sizing, full window)

| Arm | Trades | **Exp/R ($-wtd)** | Median R | Win% | PF ($) | CAGR% | MaxDD% | Sharpe | TIM% |
|---|---|---|---|---|---|---|---|---|---|
| S1 | 1,068 | 0.417 | −0.65 | 34.7 | 1.58 | 14.1 | −17.5 | 0.912 | 54.6 |
| S2 | 1,043 | **0.425** | −0.62 | 34.2 | 1.64 | 13.8 | **−13.7** | **0.929** | 54.2 |
| S3 | 1,101 | 0.307 | −0.67 | 33.5 | 1.42 | 11.3 | −18.1 | 0.721 | 54.7 |
| S4 | 1,018 | 0.345 | −0.41 | 37.8 | 1.68 | **15.9** | −20.1 | 0.808 | 55.3 |
| S5 | 1,143 | 0.413 | −0.66 | 34.1 | 1.57 | 13.6 | −15.1 | 0.849 | 55.6 |
| SPY B&H | — | — | — | — | — | 15.5 | −33.7 | 0.677 | 100 |
| EW pool B&H | — | — | — | — | — | 19.0 | −37.7 | 0.765 | 100 |
| chassis-on-SPY | — | — | — | — | — | 12.0 | −18.1 | 0.751 | 78.7 |

(Trade-mean expectancy — −3.0 to +1.3 across arms — is reported in the
JSON beside the headline; it is the tiny-R degenerate, not the decision
metric. Validate span: S1/S2 expectancy 0.46/0.47, Sharpe 1.20/1.25 vs
SPY 1.07; full tables in the JSON.)

**R-distributions** ([systems_r_dist.png](img/systems_r_dist.png)): all
five arms share the shape — median trade ≈ −0.6R, ~35% win rate, long
right tail. **The small-loss property belongs to the stop, not the
grade** — S5, with no doctrine rows, has it too. And the stop's floor is
soft: **~32–37% of exits fill below the entry stop** (gap-through at the
next open; the −1R floor is not a floor), which is where the < −2R left
tail (9–13% of trades) comes from.

## Pair tests — the full matrix (nothing counts)

| Pair | Headline diff (R) | Random-K band | Cluster-bootstrap 95% CI | p(≤0) |
|---|---|---|---|---|
| S1−S2 | −0.008 | [−0.19, +0.25] | [−0.20, +0.18] | 0.53 |
| S1−S3 | +0.110 | [−0.25, +0.29] | [−0.05, +0.31] | 0.09 |
| S1−S4 | +0.073 | [−0.29, +0.14] | [−0.29, +0.45] | 0.34 |
| S1−S5 | +0.005 | [−0.25, +0.33] | [−0.48, +0.39] | 0.47 |
| S2−S3 | +0.118 | [−0.24, +0.26] | [−0.12, +0.38] | 0.18 |
| S2−S4 | +0.080 | [−0.31, +0.09] | [−0.28, +0.41] | 0.32 |
| S2−S5 | +0.013 | [−0.16, +0.26] | [−0.46, +0.41] | 0.45 |
| S3−S4 | −0.037 | [−0.41, +0.15] | [−0.39, +0.27] | 0.56 |
| S3−S5 | −0.105 | [−0.27, +0.39] | [−0.53, +0.23] | 0.69 |
| S4−S5 | −0.068 | [−0.12, +0.35] | [−0.30, +0.15] | 0.74 |

Two independent nulls, same answer everywhere: *which names you pick*
(the random-K band) and *which trades the tape happened to deal* (the
cluster bootstrap) each swamp every gate difference.

## Selection orderings (the branch-6 check)

| Arm | Score | Native | Random median [2.5–97.5%] |
|---|---|---|---|
| S1 | 0.417 | 0.467 (low-ext) | 0.227 [0.06, 0.42] |
| S2 | 0.425 | 0.494 (low-ext) | 0.234 [0.10, 0.40] |
| S3 | 0.307 | 0.343 (low-ext) | 0.242 [0.08, 0.49] |
| S4 | 0.345 | 0.581 (momentum) | 0.314 [0.18, 0.45] |
| S5 | 0.413 | = score | 0.174 [0.05, 0.38] |

Deliberate picking (any deliberate rule) generally beats random inside a
gate, and each arm does best under an ordering aligned with its own
logic (S4 under momentum, S1/S2 under lowest-extension). The spread
across orderings (±0.1–0.24R) is the same magnitude as the spread across
gates — **selection remains a first-class, undecided design choice in
production**, and any future arm comparison must keep treating it as
the prereg did. But no ordering makes any pair clear its band: branch
5's verdict is ordering-robust, so branch 6 stays down.

## Per-regime (score-order, $-wtd Exp/R, signal-day labels)

| Arm | Trending | Choppy | Caution |
|---|---|---|---|
| S1 | 0.254 (732 tr) | **0.767** (336 tr) | — |
| S2 | 0.293 (701 tr) | 0.695 (342 tr) | — |
| S3 | 0.197 (733 tr) | 0.521 (368 tr) | — |
| S4 | 0.323 (675 tr) | 0.415 (332 tr) | -0.301 (11 tr) |
| S5 | 0.288 (764 tr) | 0.673 (368 tr) | 1.605 (11 tr) |

Choppy remains where expectancy concentrates, for every arm. Choppy-A+
(S1 0.77) vs Choppy-B-inclusive (S3 0.52) is nominally doctrine-friendly
— **descriptive only; no pre-registered test covers this cut**, and the
corrected numbers shrank it from the first run's 1.02-vs-0.81. S1–S3
never enter Caution/Risk-off (c3); S4/S5 took 10–14 Caution entries
under its 25% ceiling; nothing enters Risk-off at 6.5% sizing.

## Sensitivities

- **Costs {0, 5, 15} bps** — the ruled test is ranking stability: **S2
  leads at all three levels**; the tail reorders at 15 bps (S1 slips
  below S5 — churn-driven cost fragility). No counted finding is
  cost-dependent, because nothing counts.
- **Sizing {5, 6.5, 8}%** — the risk-off discontinuity is named, not
  averaged: 5% sizing admits **one** Risk-off slot, 6.5/8% admit zero
  (c3 keeps S1–S3 out regardless; only S4/S5 can use it). Orderings
  reshuffle mildly across sizes — the same selection coupling as above.
- **Cash yield (ruled visible):** IRX contributed ≈$4.6k (train) vs
  ≈$6.4–6.6k (validate) per arm on ~45% cash — validate's ~5% bills
  structurally flatter the ~55%-invested arms against 100%-invested
  benchmarks; the chassis benchmark also earns it when out.

## Concentration, caps, churn

- The **ceiling** is the binding constraint (S1: 58,085 ceiling denials
  vs 16 group-cap denials); books stay diversified (group HHI
  0.015–0.021; top group 4–6% of trades). **No arm is governed by R28's
  group cap.**
- **Churn:** 57–66% of trades are re-entries; average hold 12–13 days.
  The Layer A flicker (median A+ spell: 1 day) materializes as repeated
  stop-out/re-enter cycles and is the mechanism of the 15 bps cost
  fragility.

## Honest caveats (mandatory)

1. **The selection rule is a counterfactual everywhere** — production
   leaves it to a human; ordering effects match gate effects in size.
2. **Survivorship** (inherited from Layer A): all arms and the EW
   benchmark are inflated; **the EW benchmark's window-start membership
   additionally excludes late listings — a tilt in the arms' favor**
   when comparing arms to EW; SPY/chassis benchmarks carry neither bias.
3. **No breaker exits** (row-6 hole); the chassis ceiling replays the
   same macro-stress family; the rest is unmodeled, equally across arms.
4. **Grade-lite** (no c5, no row 6); Layer A's bridge put the full-grade
   entry delta at ~1/130. **S1's "advisory Trending" reading (B-or-better
   admitted, C excluded) is a doctrine-text reading of D-011 Q4, not an
   engine-rendered behavior** — production surfaces grades to a human in
   Trending rather than hard-blocking.
5. **Fill realism**: MOO next-open, 5 bps; no impact modeling; the 15 bps
   arm covers doubt. ~35% of stop exits fill below the stop (gap risk) —
   reported above, identical mechanics across arms.
6. **Full-size-or-skip admission** (no partial sizing) — one structural
   simplification of R15's human-chosen band.
7. **Mark-out**: survivors are *valued* (not sold) at the final close;
   synthetic trades flagged; the accounting identity is exact (pin 1).
8. **The R24 gap**: no profit target exists in production or here; every
   arm leans on the unbounded right tail. A target/trail ruling would
   reshape all arms alike — D-009/R24 territory.
9. **The train span alone is less flattering**: arm Sharpe edges over
   SPY concentrate in validate (where IRX also pays the cash half);
   reported per-span in the JSON, per D-006's both-numbers rule.

## POST-REGISTRATION ADDENDUM — riders (a) and (c)

*Everything in this section was computed AFTER the pre-registered
verdict was recorded, at the reviewing trader's direction, against the
committed artifacts (read-only re-simulation; deterministic seeds). None
of it was a pre-registered hypothesis; it is labeled evidence for the
D-011 revisit and the selection-rule deliberation, not a verdict.*

### Rider (a): arm-vs-benchmark bands — Branch 4's substance

Per-arm Sharpe under 50-seed random selection, against the fixed
benchmark points (chassis-on-SPY 0.751 · SPY B&H 0.677 · EW pool 0.765):

| Arm | Score Sharpe | Random-K Sharpe band [2.5–97.5%] | Chassis inside? |
|---|---|---|---|
| S1 | 0.912 | [0.087, 0.965] · med 0.587 | inside |
| S2 | 0.929 | [0.256, 1.012] · med 0.609 | inside |
| S3 | 0.721 | [−0.030, 0.994] · med 0.575 | inside |
| S4 | 0.808 | [0.638, 1.119] · med 0.923 | inside |
| S5 | 0.849 | [−0.025, 1.092] · med 0.412 | inside |

**All three benchmark Sharpes fall inside every arm's selection-noise
band.** By the same test the prereg applied to gate pairs, the
stock-selection layer as a whole is not demonstrably better than the
naked chassis (or SPY, or the EW pool) on Sharpe — **Branch 4's
substance holds: entry selection is not the edge; the chassis and the
stops are.** What separates the arms from the passive benchmarks under
every seed is DRAWDOWN (arm random-MDD bands span [−25.2, −9.5] vs SPY
−33.7 / EW −37.7) — and that is the chassis's contribution
(chassis-on-SPY: −18.1). The layer's demonstrable addition is CAGR
participation, concentrated in validate. The recorded Branch 5 verdict
stands as pre-registered (its tests were defined over arm pairs);
this construction is the post-hoc analogue for benchmarks.

### Rider (c): the ordering effect, tested with the prereg's discipline

Percentile rank of each realized ordering within that arm's 50-seed
random distribution (midrank; resolution ±2pp at n=50):

| Arm | Score ExpR pct | Score Sharpe pct | Native | Native ExpR pct | Native Sharpe pct |
|---|---|---|---|---|---|
| S1 | 96.0 | 96.0 | low-ext | 98.0 | 78.0 |
| S2 | **98.0** | 96.0 | low-ext | **100** | 96.0 |
| S3 | 80.0 | 80.0 | low-ext | 84.0 | 26.0 |
| S4 | 54.0 | 28.0 | momentum | **100** | **100** |
| S5 | **98.0** | 90.0 | = score | — | — |

- **Clearing the 97.5th percentile**: on ExpR — score-order for S2 and
  S5; native for S2 (100) and S4 (100). On Sharpe — only S4's native
  momentum ordering (100). **The single standout cell of the study is
  S4-native**: momentum-ordered momentum-gate beats all 50 random seeds
  on BOTH metrics (ExpR 0.581, Sharpe 1.354). Post-registration, one
  window, one cell out of eighteen examined — flagged against
  cherry-picking, and exactly the kind of cell the selection-rule
  deliberation exists to pre-register properly.
- **Pooled, score-vs-random**: 5/5 arms above the random median on ExpR
  (exact sign test p = 0.031), 4/5 on Sharpe (p = 0.19); Stouffer
  combined z = 3.04 (p ≈ 0.001) on ExpR, z = 2.25 (p ≈ 0.012) on
  Sharpe. **These p-values overstate the evidence**: the five arms
  share the same days and heavily overlapping candidate sets (S3 ⊃ S2;
  S5 admits nearly everything), so they are five dependent looks at one
  window, not five trials. Directionally consistent, magnitudes large
  (median lift ≈ +0.19R); joint significance not honestly quantifiable
  from this design.
- **Native-vs-score is metric-dependent**: native won 4/4 on ExpR
  (sign p = 0.0625) but only **1/4 on Sharpe** (p = 0.94) — low-ext
  picking earns expectancy while degrading the curve (S3 native Sharpe
  0.431, 26th percentile; S1 0.715, 78th). "Which ordering is best" has
  no single answer across metrics in this sample.
- **What the sample can and cannot resolve**: it CAN support that
  deliberate ordering ≠ worthless within this window — the effect is
  consistent (9 of 10 arm-metric cells at/above the random median) and
  large relative to every gate effect. It CANNOT resolve joint
  significance past the dependence caveat, which deliberate ordering is
  best (metric-dependent), or out-of-window generalization. **What
  would give it power**: pre-registered ordering hypotheses with their
  own decision table; block-split subperiods (quasi-independent
  replicates in time); disjoint pool halves (independence in the cross
  section); more seeds only sharpens the ±2pp resolution and adds no
  independence.
- **For the record (terminology)**: "native orderings span 0.343–0.581R"
  in the main report is the spread ACROSS arms, each under its own
  native ordering. The WITHIN-arm spread across the three orderings is
  0.10–0.27R — larger than the biggest between-arm headline gap
  (0.12R): the ordering choice moves outcomes more than the gate choice
  does, on either statement.

## Reproduction

```
python3 scripts/backtest_systems.py        # full run (~40 min, ~290 sims)
python3 test_backtest_systems.py           # all 13 pins
python3 scripts/charts_systems.py          # equity + R-distribution charts
```

Deterministic: `results_hash e1117af63402f874`, recomputable from the
committed JSON (canonical round-trip hash); `frame_hash 5630fea03050fb8e`
(the Layer A master frame); random-K seed-deterministic (pin 6).
