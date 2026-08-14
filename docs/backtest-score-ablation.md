# Build 5.2 — Score Component Ablation

Pre-registered: [backtest-score-ablation-prereg.md](backtest-score-ablation-prereg.md),
committed `cdcfa19` before the study script existed. Results:
[backtest-score-ablation-results.json](backtest-score-ablation-results.json),
hash `36c6975e6ca7336e`, generated from commit `cdcfa19`. Script:
`scripts/backtest_score_ablation.py`.

Panel: 848,328 ticker-days; 78,871 pass-others; A+ parity pin
reproduced the committed frame's 29,777 A+ days EXACTLY. Ladder parity
pinned elementwise against the production functions (4,000 sampled
points). The score under test is the AS-LIVED score (v1 YTD + penalty
as one joint component), per prereg.

**Ships nothing. Signal-level only. The one production-adjacent output
is evidence for the D-020a plateau revisit, which was already on
record as REVISITABLE.**

A+ reference (fwd-20d): full n=29,446 mean **0.834** / train 0.733 /
validate 0.968; ex-top-5% ≈ −0.35 in all three spans.

Two review findings were fixed BEFORE any result was read (the
adversarial pass ran while the first run executed, and that run's
output was deleted unread): per-span CIs were missing from the sets
bars 1–3 rule on, and probe (c)1's extreme-bucket contrast had been
pooled to >100 instead of the registered (20,50] vs >150.

## Verdicts, bar by bar

### Bars 1–3 — the five zero-ablations

| component | LOST (A+ only because of C) | GAINED (denied by C alone) | verdict |
|---|---|---|---|
| rsi | n=2,152; CIs straddle 0 both spans | n=1,160; straddle | no bar — inconclusive |
| macd | n=27,080; mean 0.740 vs 0.834; CI excludes 0 in full [−0.14,−0.049], train [−0.169,−0.04], validate [−0.158,−0.02] | n=3,829; straddle | **bar 2 CLEARS — formally** |
| ma | n=27,284; mean 0.738; CI excludes 0: train [−0.151,−0.027], validate [−0.184,−0.039] | n=174; validate-only positive, train straddles | **bar 2 CLEARS — formally** |
| ytd | n=19,501; straddles both spans | n=2,283; validate [0.774,2.47] positive, train straddles | no bar — inconclusive |
| vol | n=74; straddles | n=646; mean −0.502; train [−3.213,−0.67] negative, validate straddles | directionally VINDICATED, formally short of bar 3 |

**The bar-2 clears carry a confound the table did not anticipate, and
it is recorded here rather than discovered later.** Zeroing a ±13/±14
component from a 75-threshold grade drops nearly the entire A+ bucket
(the LOST sets are 92–93% of it), so LOST-vs-A+ degenerates into
low-score-margin days vs the whole bucket. What the CIs then detect is
that HIGH-margin A+ days outperform low-margin ones — score
monotonicity, a property any large component would show — at an effect
size of **−0.09 to −0.10pp** on fwd-20d. Statistically real (n≈27k),
economically almost nothing, and not specifically about MACD or MA.
The GAINED direction, which is free of this confound, clears nothing
for either component.

Per bar 8, macd-neutralized and ma-neutralized go on the shortlist
jointly AND separately — that is what the table authorizes — but the
shortlist entry carries this confound note, and the replay
deliberation should weigh whether a −0.1pp margin-monotonicity echo is
worth a replay at all.

vol's GAINED days underperform (train CI excludes zero), i.e. the
±3-point volume rung is denying days that deserved denying — but its
LOST set is 74 days with 81.8% of the sum in one ticker, and validate
straddles: directionally vindicated, below the bar.

### Bar 4 — the MACD cliff: **INCONCLUSIVE, by the registered
contradiction rule**

Census (pass-others): 0.1% of close holds 23.97%, 0.25% holds 54.25%,
0.5% holds 82.97%, 1.0% holds 97.41% — the "cliff neighborhood" at 1%
of close is essentially the whole population, which is itself a
finding about the band definition.

Side test at 1.0%: train bull−bear **+0.408** [0.038, 0.779], validate
**−0.588** [−0.92, −0.231]. Both CIs exclude zero — in OPPOSITE
directions. At 0.25%: train straddles [−0.174, 0.462], validate
[−0.833, −0.263]. The 0.25% band contradicts the 1.0% reading, and the
prereg rules that contradiction INCONCLUSIVE regardless.

The substantive observation for the record: the side of the crossover
carries span-UNSTABLE information (bullish outperformed in 2020–2023,
underperformed in 2024+, both significant). A 16-point cliff on a
signal whose sign value flips by era is neither vindicated nor
condemned by this study — it is unresolved.

m_smooth (report-only per prereg): LOST n=22,385 underperforms with
CIs excluding zero — the same margin-selection artifact as the bar-2
clears; GAINED n=2,494 straddles everywhere. Nothing here argues for
the smoothed ladder.

### Bar 5 — the MA ±6 paths: **FIRES for P1 vs P2**

| pair | cut | train diff | validate diff |
|---|---|---|---|
| P1 pullback − P2 recovery | full panel | **−1.514** [−2.075,−0.942] | **+0.548** [0.115,0.949] |
| P1 − P2 | c3∩c4 | **−0.718** [−1.391,−0.044] | **+0.904** [0.464,1.318] |
| N1 − N2 | full panel | −1.693 [−2.457,−0.884] | +0.357 straddles |
| N1 − N2 | c3∩c4 | +0.608 straddles | +0.923 [0.332,1.486] |

P1 vs P2 differs with CIs excluding zero in both spans, in both cuts —
and the SIGN FLIPS: recovery-from-crash beat pullback-in-uptrend by
~1.5pp in train; pullback beat recovery by ~0.5–0.9pp in validate. The
component pays both states +6 as if they were the same thing; they are
not, and which one is better is era-dependent. STRUCTURAL FINDING,
recorded for its own deliberation. Nothing carried — per prereg, no
redesigned MA component is authorized by this study. N1 vs N2 does not
fire (neither cut clears both spans).

(P1 and N1 are structurally empty within the A+ set — c1 IS the
component's first input — so this ran on the panel cuts, as
registered.)

### Bar 6 — YTD saturation: **the cap hides structure, in both spans**

Pass-others fwd-20d by as-lived YTD bucket:

| bucket | n | full mean | train | validate |
|---|---|---|---|---|
| (20,50] | 18,030 | 1.156 | 1.059 | 1.315 |
| (50,100] | 3,252 | 2.439 | 2.079 | 2.959 |
| (100,150] | 618 | 5.328 | 6.288 | 4.380 |
| >150 | 390 | 11.764 | 13.454 | 10.600 |

Monotone INCREASING, and the registered extreme contrast ((20,50]
minus >150) is **−10.608** full [−14.304,−7.15], train
[−17.959,−7.268], validate [−14.567,−4.232] — separation in both
spans. The >150 bucket is not tail-only: hit rate 64.4%, ex-top-5%
mean **+5.657**.

Within A+ the same probe is UNEVALUABLE past 100: n=35 in (100,150]
and n=1 above 150 — the as-lived penalties had already emptied those
buckets out of A+, which is itself the cleanest picture of what the
penalty did.

**v2 backcast** (what the CURRENT production ladder would have done on
this frame): LOST empty — v2 dominates v1+penalty pointwise, stated in
the prereg, confirmed by assertion. GAINED n=465: mean **7.417**,
ex-top-5% +1.975, CI vs A+ excludes zero in full [3.548,9.512], train
[4.351,11.535], validate [0.85,10.326]. By bar-1 semantics this
clears: the days the old penalties denied and the current plateau
admits outperformed the A+ bucket massively. Caveat for the revisit:
99 tickers / 292 dates, but ERAS alone is 30.9% of the sum, and the
top-5% share is 74.7% — strongly tail-driven, though ex-top-5% stays
positive in train (3.555) and barely in validate (0.168).

**Uncapped vs v2** (12@>20, 16@>50, 20@>100, 24@>150): LOST empty by
construction. GAINED n=621: mean **5.020**, ex-top-5% +2.512, CI
excludes zero in train [0.866,4.173] AND validate [2.902,9.097],
ex-top-5% positive in both (1.174 / 4.237). Broader than the backcast
set: 181 tickers, top ticker 11.4%, top-5% share 52.5%. By bar-1
semantics: **the uncapped monotone ladder discriminates where the flat
cap does not** — the additional days it admits outperform the A+
bucket in both spans, and not only through their tail.

Per the prereg, ALL of this feeds the D-020a plateau revisit and
nothing else. Read together: the backcast supports the plateau-12
ruling against the old penalties, and the uncapped probe says the
plateau itself is still under-crediting the >50 region. The
plateau-12-vs-plateau-8 question the revisit left open is now the
wrong axis — the recorded evidence points ABOVE 12, not below.

### Bars 7–8

Bar 7 (all inert) does not apply. Bar 8 applies to the two bar-2
clears: carried jointly and separately, with the confound note.

## Outputs

1. SHORTLIST for a 5B-style replay: macd-neutralized, ma-neutralized
   (jointly and separately) — formal bar-2 clears, carrying the
   margin-monotonicity confound note and a −0.1pp effect size.
2. STRUCTURAL FINDING for its own deliberation: the MA component pays
   two era-divergent states the same +6 (P1/P2, sign-flipping
   difference, significant in both spans in both cuts).
3. EVIDENCE for the D-020a plateau revisit: bucket monotonicity with
   both-span extreme separation; the v2 backcast's gained days at
   +7.4pp vs A+ (ERAS-concentrated, tail-heavy, ex-top-5% positive);
   the uncapped probe's gained days at +5.0pp with both-span CIs and
   positive ex-top-5% on twice the ticker breadth. Direction: above
   12, not below.
4. UNRESOLVED, on the record: the MACD crossover side is
   span-unstable; the cliff question needs a construction that is not
   diluted at 1% of close (97.41% of pass-others is inside the band).

## Caveats

Same window examined repeatedly across Builds 5/5.1/5B/6A/7; one
market regime per span. Signal-level only — no stops, sizing, exits.
The as-lived score is v1-era by construction; v2 enters only through
probe (c). Flip counts are not comparable across components (the
prereg's own caveat); the LOST-set confound above is the sharpest form
of that warning. High as-lived-YTD buckets are momentum-cohort
flavored; the validate span (2024+) reproduces the monotonicity
independently of the 2020–2021 cohort. ERAS concentration in the
backcast set is disclosed above; the uncapped set does not share it to
the same degree.
