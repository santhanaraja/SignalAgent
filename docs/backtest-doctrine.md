# The Doctrine Backtest — Layer A: does the A+ grade discriminate?

**Build 5 · 2026-07-25 · D-011's retest recipe, executed for the first
time.** Script: `scripts/backtest_doctrine.py` (supersedes the record's
`replay_ticker.py --grade-entries` per the Build 5 ruling). Results:
[backtest-doctrine-results.json](backtest-doctrine-results.json). Pins:
`test_backtest_doctrine.py`.

## Verdict, stated plainly

**No. On forward returns, the A+ grade does not discriminate positively —
the ordering is inverted.** Graded daily over 529 pool names and 848,328
point-in-time ticker-days (2020-01 → 2026-07), A+ days underperform C
days at every horizon, in the train span AND the validate span, and the
day-cluster bootstrap puts the entire 95% CI below zero (fwd-20d spread
−0.79%, CI [−1.12, −0.46]; validate −0.92%, CI [−1.27, −0.60]). A+ vs B
— the comparison between the two *permitted* entry classes — is flat in
train (+0.06%) and B-better in validate (−0.79%).

Both D-011 revisit triggers are relevant: trigger 1 ("A+ entries
underperforming B in the Build 5 replay") **fires on the validate span**.
This report is the evidence packet for that revisit — the ruling itself
is a deliberation, not this document.

Three structural findings frame the headline:

1. **The inversion is mostly momentum, seen from the other side.** The
   momentum module (below) shows the top momentum decile earning ~3.2%
   per 20d vs ~1.0% for the middle — and every row that makes a name A+
   (fresh reclaim, RSI ≤ 70, extension ≤ 1.8×ATR) is a *low-momentum
   selector*. The doctrine buys the turn; 2020-2026 paid the trend.
2. **Even the blocked-extended names beat A+.** C-days that failed *only*
   the extension guard (passed conditions, too far above the mean) ran
   1.32%/20d vs A+'s 0.83% — the D-004 guard's counterfactual, measured.
3. **The C mass is dominated by below-SMA20 days** (375,802 of 741,983)
   at 1.79%/20d — dip-buying beta in a bull-heavy window, *inflated by
   survivorship* (see caveats): these are drawdown days of names that by
   construction survived into today's pool. The A+ vs B and A+ vs
   C-extension comparisons carry less of that bias and still show no
   positive discrimination.

## Design (locked at the Step-0 hold)

| | |
|---|---|
| Population | Current 533-pool; 529 graded. Not graded: FDXF/HONA/QNT (no usable price history) and HQ (price file too thin — 41 bars; its earnings table is also empty) |
| Window | 2020-01-02 → 2026-07-10 (regime-series end); train ≤ 2023-12-31, validate after (D-006) |
| Grade | **Grade-lite**: rows 1–5 + 7, minus c5 (no universe history before 2026-07), minus row 6 (membership unrecoverable; recorded "clear" unproven pre-D-019) |
| Row 7 | Real report dates (yfinance earnings table; lxml activation this build). **None-split**: coverage gap is never A+ (1.18% of ticker-days); no-entity n/a in an all-stock pool |
| Indicators | Trailing **126-bar frame** per day — production's 6mo-fetch shape (fixed vs calendar-6mo: declared approximation, quantified by the bridge) |
| Fills | T+1 open → T+1+h open (Build 4 `info_test` convention), h ∈ {5, 10, 20, 40} |
| Regime | Committed chassis series `data/regime_daily.json` (production `replay_chassis`, provenance + replay-equality pin) |
| Grader | The **real `grade_setup`** and the **real `runway_sessions_before`** per ticker-day; scores via parity-pinned vectorized replicas of `score_*_points` |

## Headline: per-grade forward returns

Fwd-20d, mean % (T+1 open basis); full tables incl. 5/10/40d in the
results JSON.

| Span | A+ | B | C | A+−C | A+−B |
|---|---|---|---|---|---|
| Full (848k days) | 0.83 (n=29,777) | 1.15 (n=76,568) | 1.63 (n=741,983) | **−0.79** | −0.32 |
| Train 2020–23 | 0.73 (n=16,824) | 0.67 | 1.47 | **−0.73** | +0.06 |
| Validate 2024–26 | 0.97 (n=12,953) | 1.76 | 1.89 | **−0.92** | **−0.79** |

*(ns are ticker-days at grade time; each horizon's stats carry their own
slightly smaller n where the forward window runs past the data — e.g.
A+ fwd-20d n = 29,446.)*

Hit rates barely separate (A+ 55.2% / C 56.1% at 20d) — the spread is in
mean size, not direction frequency. Significance (day-cluster bootstrap,
2,000 resamples; dates are the unit — the cross-section within a day is
one cluster): full-span spread −0.79, CI [−1.12, −0.46], P(spread ≤ 0)
= 1.00; validate −0.92, CI [−1.27, −0.60]. Serial overlap of the 20-day
horizon remains inside clusters; treat the CI as approximate but the sign
as robust.

## The C-bucket, decomposed

"C" mixes opposite populations. First failing row of every C day:

| C sub-population | n | fwd-20d |
|---|---|---|
| Row 1 — conditions (mostly below SMA20, or regime-blocked) | 540,084 | 1.76% |
| — of which below SMA20 ("broken") | 375,802 | 1.79% |
| Row 2 — extension > 1.8×ATR (the D-004 guard) | 122,998 | **1.32%** |
| Row 3 — approach fail (the "knife") | 78,901 | 1.18% |

The fair like-for-like reads: A+ (0.83%) < knife-C (1.18%) <
extension-C (1.32%) — among names *above* their SMA20, the grade's
ordering of forward returns is the reverse of its preference ordering.
The knife check does earn its keep *relatively* (knives are the worst of
the above-SMA20 buckets), but everything the doctrine prefers most did
worst.

## Regime-conditioned

Fwd-20d mean %, by chassis state at signal date:

| State | A+ | B | C |
|---|---|---|---|
| In-Trend-Full (Trending) | 0.87 (n=12.6k) | 0.84 (n=33.7k) | 0.85 |
| In-Trend-Throttled (Choppy) | 0.81 (n=17.1k) | **1.39** (n=42.8k) | 1.56 |
| Out-Defensive | — (blocked) | — | −0.63 |
| Out-Risk-off | — (blocked) | — | 3.03 |

Two policy-relevant reads: in **Trending**, A+ ≈ B ≈ C — the grade adds
nothing where entries are least restricted. In **Choppy — exactly where
the doctrine enforces "A+ only" — B names outperformed A+ by ~0.6%/20d**;
the restriction excluded the better-performing permitted class. (Out
states grade everything C by construction — c3 blocks conditions — and
Risk-off C days ran 3.0%/20d: rebound beta the system deliberately never
touches; that is a risk-policy choice, not a signal failure.)

## Selectivity and persistence

- **18.2 A+ per day** on average across 529 names (p95 = 49, max = 85);
  22% of days have zero. By year: 2020 ≈ 17/day, 2021 ≈ 29, **2022 ≈
  2.9** (the gate all but closed in the bear year — directionally
  correct behavior), 2023–26 ≈ 18–22.
- **Persistence: median A+ spell is 1 day** (mean 1.86, p90 = 4, max
  13; 55% of spells are a single day). The MTB flicker is the norm, not
  an anomaly: A+ is a knife-edge state, which matters operationally —
  a chip that appears at today's close is usually gone by the next.

## Sensitivities

- **`approach_swing_lookback` {10, 20, 40}: perfectly inert.** The
  up-close flags differ on 5.5% / 3.8% of ticker-days, but **zero
  ticker-day grades change** across the three settings. The flag only
  binds when close > SMA5 already holds — and above the SMA5 there has
  essentially always been ≥1 up-close in any window. The Phase-1 flagged
  knob is answered: at these values it is not a knob. (The Q1 variants —
  slope pair, longer stabilization — are Build 5.1, per ruling 4.)
- **Without row 7** (runway): A+ grows 29,777 → 40,035 ticker-days;
  fwd-20d 0.83% → 0.91%. Row 7 excludes ~27% of would-be A+ days and
  slightly *lowers* the measured A+ return — the headline inversion is
  not a row-7 artifact.

## The momentum module (input to a future D-020 — not a code change)

Cross-sectional deciles per day, fwd-20d mean, three definitions × three
cuts (unconditional / above-50DMA / score ≥ 50 on the frame score — an
approximation of the universe gate, whose own score anchors YTD to a
1-year frame; declared, not identical):

**d10 − d1 spread (fwd-20d, %):**

| Definition | Unconditional | Above-50DMA | Score ≥ 50 (frame) |
|---|---|---|---|
| YTD (as-lived) | +0.80 | +1.83 | +1.55 |
| 12-1 month | +0.87 | +1.26 | +0.83 |
| 6-month | +1.02 | **+1.93** | +1.56 |

The full 6-month/above-50DMA decile curve is a **smile with a violent
right edge**: d1 1.24 → d2–d8 ≈ 0.8–1.0 → d9 1.25 → **d10 3.17**. High
momentum won in every definition and every tradability cut; the top
decile is where nearly all of the excess sits.

**The YTD overextension penalty is empirically backwards.** Days where
the penalty fired (YTD > 100%): 7,930 (0.93% of all ticker-days,
seasonal — 233 in 2022, ~1.9k in 2020/2025). Their forward returns:

| Set | n | fwd-20d |
|---|---|---|
| Penalized ticker-days (YTD > 100%) | 7,930 | **+6.95%** |
| All un-penalized days | 840k | +1.51% |
| Pushed below the 50 universe gate by the penalty | 2,204 (105 names) | **+7.35%** |
| Pushed below the 75 A+ bar by the penalty | 2,737 | +6.86% |

The penalty's production effect (mostly universe exclusion, per Step-0)
removed precisely the names that went on to perform best. Both ruled
questions answered: the **sign** is backwards over this window, and the
**window** matters less than the sign (all three definitions agree).
Framed for a D-020 deliberation; `score_stock` is untouched this build.

## Honesty bridge & parity (the production overlap)

- **Exact-function parity:** 516 recorded production candidate grades
  across 12 committed artifacts reproduce **exactly** through the real
  calling convention (`test_backtest_doctrine.py` pin 3).
- **Pipeline bridge (committed as pin 11):** on the 3 committed overlap
  days (2026-07-22/23/24, production-recorded regime), the grade-lite
  pipeline agrees with production's full grade on **129/130 candidate
  ticker-days (99.2%)**. The single disagreement is the ruled deletion
  at work: NVDA B→lite-A+ (production denied A+ **on row 6, breaker
  warning** — the row grade-lite drops; its measured cost: 1/130). An
  earlier draft had a second miss (MSI); adversarial review traced it to
  a confirmation-counter off-by-one in the backtest — fixed, pinned
  prefix-by-prefix against the production counter (pin 10), and the
  bridge improved to 129/130.
- Feature parity: RSI exact to 1e-9, MACD state, ATR, and the approach
  turn check all match the production functions on random real
  ticker-days; the score ladders match at every boundary.

## Honest caveats (mandatory)

1. **Survivorship, with direction.** The population is today's pool with
   history — every bucket is inflated, but *not equally*: C-below-SMA20
   days are drawdown days of names that by construction recovered, so
   the A+−C inversion is partly survivorship. The A+ vs B and A+ vs
   extension-C comparisons (same side of the SMA20) carry much less of
   it and still show no positive discrimination. Absolute levels in all
   tables are biased up; only relative structure is interpreted.
2. **Regime of the window.** 2020–2026 contains three powerful uptrends
   and rewarded momentum persistently. A doctrine that buys fresh turns
   may discriminate in other regimes; this window cannot show it. The
   2022 read is thin by construction (the gate closed: ~2.4 A+/day).
3. **Scheduling clairvoyance (row 7).** Actual report dates are used as
   if known; at real time the date may not yet have been announced.
   Standard construction, biased *toward* passing row 7 slightly early.
4. **Fixed 126-bar frame** vs production's calendar-6mo fetch: quantified
   by the bridge (98.5% agreement incl. this and the ruled deletions).
5. **Ticker-days are not independent** — cross-sectional and serial
   correlation; the day-cluster bootstrap addresses the first, only
   partially the second.
6. **Raw returns, no market adjustment.** Within-day deciles difference
   out the market; the per-grade tables do not (all grades face the same
   days only approximately — A+ concentrates in calm tapes).
7. **Earnings-gap granularity.** The 240-day enclosing-interval rule
   catches listing migrations and long table holes but cannot detect a
   *single* missed quarterly print (~182-day interval). Empirically
   negligible (3/527 tickers with any >240d interval in-window), and the
   no-row-7 sensitivity bounds the whole row's influence.
8. **Population trim vs production.** Requiring a 126-bar frame excludes
   2,885 ticker-days early in young listings' lives that production
   (which grades on whatever its 6mo fetch returns, with the MA-NaN −14
   bias) would have graded. Declared; the bias runs toward excluding the
   noisiest days.
9. **Signal quality only.** No stops, sizing, R28, or exit logic — a
   grade that fails on unconditional forward returns could still earn
   its keep through risk shaping (smaller drawdowns after entry). That
   is Build 5B (strategy replay), explicitly out of scope here.

## Post-review statistical checks

Four checks run after adversarial review (the statistics reviewer died
mid-flight; these cover its questions):

- **Medians preserve the inversion** — A+−C on medians: −0.32 (full),
  −0.63 (validate). Not an outlier artifact.
- **Independent bootstrap** (fresh implementation, different seed):
  CI [−1.12, −0.50], P(≤0) = 1.00 — matches the committed numbers.
- **Survivorship controls**: restricting to 2024+ (where the
  current-pool bias is thinnest) the A+−C spread is −0.92; restricting
  to the 502 names with full-window history: A+ 0.71 < B 1.03 < C 1.57.
  The inversion is not a survivorship artifact, though absolute levels
  remain inflated.
- **The penalty and the top decile are one phenomenon seen twice**: 83%
  of penalized ticker-days sit in the top 6-month-momentum decile.
  Read them as one finding — extreme performers kept performing — not
  two independent confirmations.

## Reproduction

```
python3 scripts/fetch_doctrine_cache.py        # one-time caches (gitignored)
python3 scripts/build_regime_series.py         # committed regime series
python3 scripts/backtest_doctrine.py           # full run (~25 min)
python3 test_backtest_doctrine.py              # all nine pins
```

Determinism: results carry `input_hash` (`d7df85e8ad6ba244`); two runs
on the same caches hash identically (pin 9). Smoke/pin runs write their
frame to `master_frame_smoke.*`, never over the full-run frame.
