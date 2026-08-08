# Profit-Target Sweep — Build 6A: does taking profit at an R-target pay?

**Build 6A · 2026-08-08.** Script: `scripts/backtest_target.py` ·
Results: [backtest-target-results.json](backtest-target-results.json)
(`results_hash 847d6306f4355fb9`, recomputable from the committed file)
· Pins: `test_backtest_target.py` (13) · Pre-registration:
[backtest-target-prereg.md](backtest-target-prereg.md), committed
(`d184ced`) **before any results existed** · Implementation
declarations: [backtest-target-declarations.md](backtest-target-declarations.md)
(`6bbca06`), also before results.

**Integrity gate first, two-sided:** (a) the 5B engine reproduces the
committed `backtest-systems-results.json` S1/score block exactly;
(b) this build's target engine with the rule switched off reproduces
that 5B run **to the cent** — max curve divergence $0.000000, max
per-trade P&L divergence $0.000000, all 1,068 trades. Every arm below
differs from T0 only by its exit-target rule, on an entry population
fixed by construction (T0's realized fills, replayed as forced
entries; declaration 2).

## Verdict, read against the pre-registration

> **No arm is adopted. Branch 1 — the free lunch — is empty, as the
> prereg predicted. One arm (T3) lands branch 2 and prices the choice:
> 0.66 CAGR points surrendered per point of max drawdown saved —
> presented as a deliberate operator's choice, not a recommendation —
> and it carries the mandated FRAGILE label (top-3 trades = 57% of the
> difference). Every other arm either sits inside its CIs (T4) or
> falls in combinations the decision table does not name (T1, T2, T5),
> reported below as UNCOVERED rather than coerced into a branch.**
>
> The through-line is the campaign's oldest finding: the system's
> profit lives in its right tail. Selling half a winner at 2R or 3R
> taxes exactly the trades that pay for everything else — every
> partial-sale arm surrenders 3.2–3.9 CAGR points, and the three
> largest contributors to that surrender are the same few moonshots
> (MU 2026-04-13, CIEN 2025-09-02, PRAX 2023-12-21 / ORKA 2024-03-14)
> in every arm. The drawdown relief is real but concentrated in the
> validate span. And the breakeven floor (T4) is economically inert:
> armed on 435 trades, it changed only **19 exits**, for a net
> −$1,506 — by the time a trade is +2R, the SMA20 trail has almost
> always risen past the entry, so max(SMA20, entry) is just the SMA20.

## The table (full window; paired CIs vs T0, 2.5–97.5%)

| Arm | Rule | CAGR | ΔCAGR (CI) | maxDD | ΔmaxDD (CI) | Sharpe | Hits | Top-3 share | Branch |
|---|---|---|---|---|---|---|---|---|---|
| **T0** | SMA20 trail only | **14.06%** | — | **−17.50%** | — | 0.912 | — | — | baseline |
| T1 | 50% at 2R | 10.83% | −3.23 [−7.91, +0.37] | −11.58% | **+5.92 [+1.23, +9.20]** | 0.911 | 435 (40.7%) | 52.9% FRAGILE | UNCOVERED¹ |
| T2 | 50% at 3R | 10.80% | **−3.26 [−7.02, −0.57]** | −14.27% | +3.23 [−0.07, +5.26] | 0.822 | 300 (28.1%) | 52.7% FRAGILE | UNCOVERED² |
| T3 | 33% at 2R + 33% at 4R | 10.13% | **−3.93 [−8.96, −0.12]** | −11.53% | **+5.97 [+0.94, +8.63]** | 0.850 | 435 / 231 | 57.0% FRAGILE | **2 — EXCHANGE RATE** |
| T4 | floor at BE after 2R | 13.97% | −0.09 [−0.70, +0.45] | −17.50% | +0.00 [−0.40, +0.89] | 0.910 | 435 armed³ | (see ³) | 4 — INSIDE CI |
| T5 | 50% at 3R + BE floor | 10.73% | **−3.33 [−7.20, −0.69]** | −14.27% | +3.23 [−0.10, +5.27] | 0.818 | 300 | 51.5% FRAGILE | UNCOVERED² |

¹ T1: the drawdown improvement clears its CI but the CAGR cost does
not — the table's branch 2 requires BOTH outside noise. Nearest
reading: a cheaper version of T3's exchange, unpriceable at this
power. Reported as the table demands: uncovered, no branch.
² T2/T5: the CAGR loss clears its CI but the drawdown improvement
does not — reject-flavored, but branch 3 requires losing on BOTH
outside noise. Uncovered, no branch.
³ T4's top-3 share prints 141.5% — of a total difference of −$1,506
(0.6% of end equity; −0.09pp of CAGR) whose contributions cancel. The
FRAGILE label applies by the letter of the rule; the quantity it
labels is economically negligible.

**Exchange rate (T3, branch 2):** −3.93 CAGR pp for +5.97 maxDD pp
= **0.658 CAGR points per drawdown point saved** (T0 14.06%/−17.50%
→ T3 10.13%/−11.53%). Per the prereg: presented for the operator;
**no arm is adopted by the study** — and the FRAGILE label stands.

## Span decomposition (D-006: both spans, always)

| Arm | Train CAGR/maxDD | Validate CAGR/maxDD |
|---|---|---|
| T0 | 8.96% / −10.30% | 23.24% / −17.50% |
| T1 | 7.89% / −8.09% | 15.91% / −11.58% |
| T2 | 7.72% / −9.40% | 16.10% / −14.27% |
| T3 | 7.57% / −8.50% | 14.58% / −11.53% |
| T4 | 9.02% / −10.13% | 22.87% / −17.50% |
| T5 | 7.71% / −9.40% | 15.92% / −14.27% |

The CAGR surrender concentrates in validate — T3's validate-span gap
is 8.7pp against 1.4pp in train (span gaps are not additive; the
full-window difference is −3.93pp) — because that is where the
moonshots lived —
the same span asymmetry every layer of this campaign has found. The
sign is consistent in BOTH spans for every partial-sale arm: the
target costs return in train too, just less.

## Mechanics: where the money moves

- **Dollar-weighted expectancy falls in every partial-sale arm**:
  T0 0.417R → T1 0.291 / T2 0.295 / T3 0.267 / T5 0.292 (T4 0.413).
  At T0-scale dollars the paired per-trade sum is −$38k (T1), −$37k
  (T2), −$45k (T3) on a book whose total trade P&L is ≈ +$124k.
- **The paired per-trade MEAN in R-multiples is positive (+1.30 to
  +1.76) for the same arms — and that number is a known artifact.**
  It inherits 5B's tiny-R distortion: one trade (TXT 2026-05-29,
  R = **$0.08 of total dollar risk** — a 169-share position whose
  entry filled 0.05 cents per share above its signal-day stop) has
  T0 r_mult −3,638 and a paired diff of +1,727R,
  which alone contributes ≈ +1.6R to the mean. The median paired diff
  among touched trades is +0.32R. The bootstrap CIs on the mean
  include zero in every arm; the decision clauses were declared in
  advance to read the paired path bootstrap (declaration 12), which
  this pathology cannot touch.
- **58.9% of touched trades gain, 41.1% lose** (T1: 256 vs 179 of
  435) — the winners gain a median +0.84R (banked at 2R before a
  round-trip); the median loser gives up −1.30R, and the tail runs to
  −139R (CIEN) with 28 losers beyond −5R — the moonshots halved.
  Count favors the target; dollars favor the trail. The system's edge
  is, once again, its right tail.
- **Freed capital earned its keep and it wasn't enough**: cash yield
  rises from $11.3k (T0) to $14.9k (T1/T3) — the IRX return on the
  banked halves, mostly in the validate span's ~5% rates — and the
  arms still lose 3+ CAGR points net.
- **The intrabar/close-basis asymmetry favors the targets** (declared
  in the prereg as real): target legs fill intrabar at the touch,
  stops wait for a confirmed close. The arms enjoy that advantage and
  still surrender CAGR — the no-adoption verdict is a-fortiori.

## Concentration check (mandated)

Every partial-sale arm is **FRAGILE**: top-3 contributors carry
51.5–57.0% of the total difference vs T0 (threshold: half). The same
trades dominate every arm: MU 2026-04-13 (−$9.4k to −$12.5k of the
difference alone), CIEN 2025-09-02, PRAX 2023-12-21, ORKA 2024-03-14.
Targets-reached counts: 2R = 435 of 1,068 trades (40.7%), 3R = 300
(28.1%), 4R = 231 (21.6%). T4: 435 floors armed, 19 exits changed,
−$1,506 (mean −2.0R across the 19 — the floor mostly clipped
recoveries that T0 rode out).

## Caveats (from the prereg, plus what the run added)

- **Power**: 1,068 paired trades, but the differences are tail-driven
  — the CIs are wide exactly where the prereg said they would be, and
  three of five arms land outside the table's named branches because
  one of their two CIs straddles zero.
- One window; survivorship inherited from the 5B panel (and it cuts
  FOR the trail: dead moonshots would have paid the target and hurt
  T0 — unresolvable here, direction stated).
- Five arms against one baseline, no multiplicity correction — the
  prereg demanded none; the caveat belongs on the record.
- The FRAGILE labels are not decoration: remove three trades from a
  1,068-trade book and half of every arm's difference is gone. A
  target rule priced off this window is priced off MU and CIEN.
- Nothing here touches production. Per the prereg: no threshold
  beyond {2R, 3R, 4R} was tested, no sizing/trail/entry variant was
  run (Build 6B's scope), and any future choice of T3's exchange is
  the operator's, made with the FRAGILE label in view.

## Reproduction

```
python3 scripts/backtest_target.py      # ~4 min against the 5B caches
python3 test_backtest_target.py         # all thirteen pins
```
