# Build 9 — Sizing and Ceiling Sweep

Pre-registered: [backtest-sizing-prereg.md](backtest-sizing-prereg.md)
(`2aa55a4`, sha256 `af870141594b5a03…`, asserted at runtime by the
study itself). Results:
[backtest-sizing-results.json](backtest-sizing-results.json), hash
`7e5448310f176966`, schema `backtest-sizing-1`. Script:
`scripts/backtest_sizing.py`. Pins: `test_backtest_sizing.py`
(durable form). Audit: `scripts/audit_sizing_report.py`, exit 0.

**Process:** the adversarial fan-out (5 reviewers, all completed)
confirmed one material defect — the clause-6 fragility construct
compared gross top-3 trades to a net advantage and saturated FRAGILE
for every positive arm — plus five fixable minors. All six were fixed
BEFORE any full result was read; the first full run's output was
deleted unread; the final run executed in a pristine copy at
`2aa55a4` with the unmodified harness (1 dirty path at run time: the
untracked study script itself). Unlike Build 8 there was NO partial
unblinding: reviewers read only the smoke artifact.

**Gates:** P1 reproduces the committed 5B S1/score block exactly
(cent-exact, verified again documentarily from the results). The
FEASIBILITY GATE — Build 8's lesson as a hard clause — held: min cash
≥ 0 for all nine arms, asserted finite and non-negative before any
statistic; borrowing is structurally impossible in this harness (an
unaffordable entry is denied and counted).

## The verdict: branch 4

**Every arm sits inside the CIs against P1 on BOTH metrics in BOTH
spans** (`inside_ci_both_metrics: true` for all eight). No dominance
(bar 1 — as predicted), nothing dominated, and no CI-clean exchange
rate to buy (bar 2 — the predicted branch, WRONG: the frontier is not
well-ordered, it is statistically flat). No interior optimum clears
bar 3 (P3 beats both size-neighbours on CAGR but not P4 on drawdown).
**The sizing question closes on this window; 6.5% / 90-50-25-5 stands
by incumbency.**

## The frontier, descriptively — flat in CI, structured in point

| arm | size/ladder | CAGR% | MDD% | mean exp | min cash | above own random band? | fragile |
|---|---|---|---|---|---|---|---|
| P1 | 6.5 / 90-50-25-5 | 14.06 | −17.50 | 0.546 | 12,720 | **yes** | — |
| P2 | 5.0 / same | 12.94 | −15.91 | 0.544 | 11,868 | no | no |
| P3 | 8.0 / same | 16.43 | −16.71 | 0.535 | 10,313 | **yes** | **yes** (65.3%) |
| P4 | 10.0 / same | 13.98 | −15.96 | 0.518 | 12,842 | no | no |
| P5 | 6.5 / 100-60-30-10 | 11.83 | −20.21 | 0.613 | **10** | no | no |
| P6 | 6.5 / 70-40-20-5 | 15.02 | −14.01 | 0.424 | 32,524 | **yes** | **yes** (307%) |
| P7 | 6.5 / flat 90 | 14.03 | −20.49 | 0.700 | 10,784 | no | **yes** (1051%) |
| P8 | 10.0 / 100-… | 15.21 | −20.27 | 0.597 | 111 | no | **yes** (179%) |
| P9 | 5.0 / 100-… | 12.97 | −15.83 | 0.621 | 38 | no | no |

Three descriptive regularities, all CI-inside and reported as
structure, not findings:

1. **More money made things worse.** P5 — the same book with a higher
   ceiling — lost 2.23 CAGR points AND deepened drawdown by 2.71,
   while P6 — a lower ceiling — gained 0.96 CAGR AND shallowed
   drawdown by 3.49. The point-frontier is non-monotone in exposure,
   in the direction OPPOSITE to "more risk, more return."
2. **The selection edge dies with depth.** P1, P3 and P6 — the three
   tightest-capacity arms — are the ONLY arms above their own
   random-K bands: score-ordering beats random selection only when
   the book skims the top of the candidate list. Every 100-ladder arm
   and the flat-ladder P7 fall INSIDE their bands — pushed deeper,
   the ordered book is indistinguishable from a random one. This is
   the study's cleanest mechanism: the candidate stream's marginal
   quality, not capital, is the binding economics.
3. **At a 100 ceiling, the cash floor IS the ceiling.** P5/P8/P9 ran
   min-cash to $10–111 and posted the study's only cash denials
   (60 and 50); their extra ceiling headroom was consumed to
   full-investment and bought quality dilution. The prereg's R28
   prediction verified: at 10% size the group cap binds at two names
   (P8's group-pct denials 73 vs P1's 15), and P8 reached 21.45% of
   capital in a single name — position appreciation through a 10%
   entry, the study's max.

The two point-attractive arms (P3, P6) both carry the FRAGILE label
under the fixed advantage-decomposition construct — P3's +$32,783
advantage is 65.3% three trades; P6's +$9,761 is 307% three trades
(the rest of its book underperforms P1). Point structure on fragile
advantages inside straddling CIs adopts nothing.

## Bar 5 — the ladder, ruled separately: UNRESOLVED, leaning "earns"

Neither registered branch fires. P7's drawdown is NOT worse with CI
separation (d_mdd CIs straddle in all spans: full [−14.08, +2.82]),
so the ladder's protective effect is not demonstrated. But P7 does
NOT "match P1 on both metrics" either — the decorative branch's
condition: CAGR matches to 3bp (14.03 vs 14.06), but the drawdown
point estimate is 2.99pp worse, per-span Sharpe drops (0.655 vs 0.714
train; 0.931 vs 1.197 validate), and the descriptive exchange is
damning in direction: the flat ladder deployed 28% more capital
(mean exposure 0.700 vs 0.546) for identical return. P7 also falls
INSIDE its random band while P1 sits above — removing the ladder
erased the selection edge. And P7's over-ceiling days collapse to 2
(vs 353 for P1): the laddered arms' 311–411 over-ceiling days are the
downgrade-hold behaviour, present and counted, exactly as production
defines it.

On the record: one window could not separate the ladder's drawdown
effect from noise — the seed-independent drawdown property that
motivated this arm remains unexplained at CI level. The point
evidence (same return, more capital, worse Sharpe, skill erased) all
runs AGAINST removing the ladder. Nothing changes.

## Prediction scorecard

Branch 2 (well-ordered frontier) — **wrong**: branch 4, and the point
shape is non-monotone, not well-ordered. P7 materially worse drawdown
with CI separation — **not confirmed** (direction right, −2.99pp;
separation absent). 5% arms better risk-adjusted than 10% —
**not confirmed** (P2 vs P4 is a wash on both metrics).

## Mandatory validity — full table in the results JSON

Per arm: min cash (all ≥ $10, none negative), trades (664–1,604),
distinct tickers, mean concurrent positions (4.92–11.85), mean
exposure/equity, max single-name % (12.35–21.45), denials split into
ceiling / group-pct / group-count / cash / no-bar / gap-below-stop /
already-held, days at ceiling and days over ceiling (close-of-day
approximations, basis-noted in the artifact; the harness's own denial
counters are the exact binding record), and trade overlap vs P1
(54.9–94.6% of P1's book, decreasing with size — capacity mechanics,
as the prereg expected).

## Caveats

One window; v1 grades; costs 5bps/side modelled, slippage not (Build
8 measured ~80% of exits filling below trigger at median 0.8%, borne
roughly equally here). The bands are the null for selection skill,
not for the frontier itself. `pct_days_any_position` is deliberately
NOT the committed 5B `time_in_market_pct` (that one is
exposure-weighted). paired_path_boot resamples common market-time
blocks over diverged books — the construction Build 7 used, extended
here to non-identical trade populations; its CIs are wide partly for
that reason, and the flat-in-CI verdict should be read with that
width in mind rather than as proof of equality.
