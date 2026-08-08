# Row Ablation — Build 5.1: which grade rows are expensive?

**Build 5.1 · 2026-08-08.** Script: `scripts/backtest_ablation.py` ·
Results: [backtest-ablation-results.json](backtest-ablation-results.json)
(`results_hash d6dc21745af93826`, recomputable from the committed file) ·
Pins: `test_backtest_ablation.py` (10) · Pre-registration:
[backtest-ablation-prereg.md](backtest-ablation-prereg.md), committed
(`2f98eb0`) **before any results existed**.

**Integrity gate first:** the recomputed row flags, ANDed, reproduce the
committed Layer A frame's A+ set — all 29,777 days, **exactly** —
asserted inside the run and independently pinned on a random sample.
Every excluded-set statistic below rests on validated flags.

## Verdict, read against the pre-registration

> **Two suspects clear bar 1, so bar 5 applies: carry BOTH into a
> 5B-style system replay, jointly and separately. Production
> unchanged; nothing ships on this evidence.** (Interpretive step,
> stated per review: bar 5 says "two or more rows" — the penalty
> construct is carried as a bar-1 clearer under the prereg's fixed
> ROWS TESTED list, which includes it; the carry action is identical
> under either reading.)
>
> 1. **R2 — the extension guard (≤1.8×ATR) — PRIMARY SUSPECT.**
>    Its excluded set is enormous: 23,893 machine-perfect-but-extended
>    ticker-days (80% the size of the A+ bucket itself). Mean fwd-20d
>    **1.43% vs A+'s 0.83%**; bootstrap CI on the difference
>    **[+0.37, +0.84]**, p ≈ 0; sign holds in both spans (train 1.31 >
>    0.73; validate 1.61 > 0.97); and it is the **only row whose
>    excluded set has a positive ex-top-5% mean (+0.16)** — its
>    outperformance is broad-based, not moonshot-carried. This is
>    Layer A's D-004 counterfactual (extension-blocked C days at
>    1.32%), now passing the full pre-registered test.
> 2. **The YTD>100% penalty construct — PRIMARY SUSPECT, with two
>    disclosures the bar reading depends on.** The 384 days whose grade
>    flips to A+ when the penalty is removed are the richest set in the
>    campaign: mean **7.99%/20d**, median 2.46, full-window ex-top-5%
>    mean **+2.13**, CI **[+3.63, +10.56]**, mean above A+ in both
>    spans emphatically (train 9.35, validate 6.53). Disclosures:
>    **(a)** its validate-span ex-top-5% mean is **negative (−0.28)**
>    — the prereg's ex-top clause is unqualified, so this reading
>    applies it full-window (the both-spans clause attaches to the
>    mean test as written); under a stricter all-spans reading this
>    suspect would drop to bar 2, and the ruling should know that.
>    **(b)** The set is concentrated: 73 tickers over 251 dates, and
>    one name (ERAS) carries **31.5%** of the set's summed return.
>    Contrast R2, whose ex-top-5% is positive in BOTH spans
>    (train +0.19, validate +0.13) and whose top ticker carries 3.0%
>    of a 527-ticker set — the broad suspect and the concentrated one
>    are not equally robust, and the carry treats them accordingly.

## The full table (fwd-20d, full window; A+ reference: mean 0.834, ex-top-5% −0.355)

| Row | n | % of A+ | Mean | Median | Ex-top-5% | Train | Validate | CI vs A+ | Bar |
|---|---|---|---|---|---|---|---|---|---|
| c1 close>SMA20 | **0** | — | — | — | — | — | — | — | structurally empty¹ |
| c2 confirmation | 192 | 0.6 | 0.29 | −0.18 | −0.84 | 0.02 | 0.54 | [−1.85, +0.82] | **3 — real work** |
| c3 regime gate | 4,030 | 13.5 | 1.06 | 0.50 | −0.50 | 0.82 | 1.98 | [−0.63, +1.14] | 2 — inconclusive |
| c4 slope | 7,667 | 25.7 | 1.22 | 1.13 | −0.12 | 0.96 | 1.59 | [+0.02, +0.77] | 2 — inconclusive² |
| **R2 extension** | 23,893 | 81.1 | **1.43** | 1.16 | **+0.16** | 1.31 | 1.61 | **[+0.37, +0.84]** | **1 — SUSPECT** |
| R3 approach | 19,662 | 66.0 | 1.03 | 0.99 | −0.23 | 0.77 | 1.35 | [−0.19, +0.59] | 2 — inconclusive |
| R4 RSI 45–70 | 233 | 0.8 | 0.24 | 0.94 | −0.99 | −1.75 | 2.62 | [−1.98, +0.71] | **3 — real work³** |
| R5 score≥75 | 48,698 | 165.4⁴ | 1.16 | 0.95 | −0.09 | 0.69 | 1.73 | [+0.06, +0.59] | 2 — inconclusive⁵ |
| R7 runway | 10,174 | 34.2 | 1.15 | 1.11 | −0.39 | 0.71 | 1.74 | [−0.07, +0.70] | 2 — inconclusive⁶ |
| **YTD-penalty flip** | 384 | 1.3 | **7.99** | 2.46 | **+2.13** | 9.35 | 6.53 | **[+3.63, +10.56]** | **1 — SUSPECT** |

¹ c1 cannot fail alone: below the SMA20 the confirmation count is zero
and the ATR break requires being above, so c2 fails with it — pinned,
with all 17,219 sampled c1-failures also failing c2.
² c4 clears mean, CI and both spans but its ex-top-5% mean is negative
— bar 2 by the table's own fourth clause. The nearest miss.
³ R4's train mean is −1.75 on a very small set (233 days total) —
bar 3 as read, with the small-n caveat stated.
⁴ R5's excluded set is *larger* than the A+ bucket — the score is the
binding row of the grade.
⁵ R5 fails bar 1 on TWO clauses: the train-span sign (0.693 < 0.733)
AND a negative ex-top-5% mean (−0.09). Recorded; not carried.
⁶ R7 fails two clauses: its CI includes zero (p = 0.053) and the
train span (0.709 < 0.733). Recorded.

## What the two suspects mean mechanically

The Layer A decomposition asked where A+'s right tail went. The
ablation's answer: **the extension guard is where most of it goes by
volume** (a excluded set nearly as large as the bucket itself, richer
at every span, and — uniquely — richer even after removing its own top
5%), and **the YTD penalty is where the densest vein goes by
concentration** (384 days at 8%/20d — momentum monsters denied the
grade for having performed). Both suspects had independent prior
evidence (Layer A's guard counterfactual; the +6.95% penalized-days
finding); the ablation puts both through the pre-registered bars and
both pass. The two sets are disjoint by construction (the flip days
pass R2), so joint-and-separate carry is well-defined.

## Caveats (from the prereg, plus what the run added)

- Signal-level only; no stops, sizing or exits. 5B already showed
  gates are hard to distinguish once the machinery applies — that is
  exactly why the carry target is a 5B-style replay, not production.
- One window, examined repeatedly — the reason the decision table
  preceded the run.
- Tail-dependence is normal here (C's own top-5% share is 92%); the
  DIFFERENCE against the A+ reference is the finding, per the prereg.
- Small sets (c2 192, R4 233, flip 384) carry wide CIs; only the flip
  set clears its CI regardless — and the flip set is name-concentrated
  (ERAS 31.5% of its sum), disclosed above.
- **Survivorship bears hardest on exactly these suspects**: extended
  names and momentum monsters that later died are less likely to be in
  today's pool, so both excluded sets are inflated by the same
  mechanism that inflated Layer A's C bucket. The A+ reference is
  inflated too, but less (its members are mid-trend, not
  post-vertical). Direction: works IN FAVOR of the suspects; the
  5B-style replay carries the same bias and cannot resolve it —
  only a point-in-time membership source could.
- **Nine simultaneous comparisons, no multiplicity correction** — the
  prereg demanded none, and none was applied. Two clearing of ten
  tests at these CI widths is unlikely to be pure multiplicity (R2's
  p ≈ 0), but the caveat belongs on the record.
- R6 untestable (ruled Layer A deletion). c1 untestable (structural).
- No threshold sweeps were run; none were authorised.

## Reproduction

```
python3 scripts/backtest_ablation.py     # ~2 min against the Layer A caches
python3 test_backtest_ablation.py        # all ten pins
```
