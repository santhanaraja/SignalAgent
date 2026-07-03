# Composite Score (0–100)

The SCORE column on the dashboard comes from `score_stock()` in
`signal_engine.py`. It is **purely additive from a base of 50** across five
technical components, then hard-clamped to [0, 100]. There are no weights,
multipliers, or interaction terms. The same function scores dashboard rows
(engine, 6-month frames) and universe-rotation qualifiers/near-misses
(builder, 1-year frames); indicator tails are nearly identical between the
two, so scores rarely differ by more than a point.

Per-component points are emitted in `score_components`
(`{rsi, macd, ma, ytd, vol}`) on every stock row and near-miss entry in
signals.json — that is what the score-cell hover tooltip renders.

## Components

| Component | Condition | Points |
|---|---|---|
| **Base** | always | **50** |
| **RSI** (Wilder-14) | <30 / 30–40 / 40–60 / 60–70 / 70–80 / ≥80 | **+15 / +8 / +3 / −3 / −8 / −15** |
| **MACD** (12/26/9) | line > signal (+8), and histogram rising (+5 more) | **+8 or +13** |
| | line ≤ signal (−8), and histogram falling (−5 more) | **−8 or −13** |
| **MA structure** | price>MA20 ±4 · price>MA50 ±6 · MA20>MA50 ±4 | **−14 … +14** |
| **YTD bucket** | >50 / >20 / >5 / >0 / >−10 / ≤−10 | **+8 / +12 / +6 / +2 / −4 / −10** |
| **YTD overextension** | >150%: −15 · >100%: −10 (checked in that order) | **−15 or −10** |
| **Volume** (5d avg ÷ 20d avg) | >1.5 / <0.7 / else | **+3 / −3 / 0** |

Signal bands: ≥75 strong-buy · ≥60 buy · ≥45 hold · ≥30 sell · <30
strong-sell. The weekly universe rotation additionally requires score ≥ 50
(`rotation.min_composite_score`) for a ticker to qualify.

**Not used by the score** (displayed only): 52-week-high distance, 1M/3M
returns, RS vs MA50, trend strength, and all fundamentals.

## Design notes & known quirks

- **The YTD buckets taper deliberately**: +20–50% earns +12 while >+50%
  earns only +8, and >100% / >150% eat an extra −10 / −15. Extended
  momentum is progressively de-rated.
- **Fixed 2026-07-03**: the >150% branch was dead code (unreachable below
  the >100% check). Reordered so it fires; mega-winners (+150%+) now take
  −15 instead of −10, lowering their score by 5 points vs. prior behavior.
- **Short-history bias (kept as-is, documented)**: when MA20/MA50 are NaN
  (fewer than 20/50 bars), they default to the current price, so all three
  MA comparisons fail and a young listing takes the full −14. The
  universe's 90-day history gate keeps such names out in practice.
- **YTD baseline** uses the year's first *valid* close (NaN phantom bars
  from Yahoo are skipped — see `compute_ytd_return`).

## Worked example — MU (2026-07-03, post-semi-crash)

| Step | Points | Running | Why |
|---|---|---|---|
| Base | +50 | 50 | |
| RSI | +3 | 53 | RSI 48.5, neutral zone |
| MACD | −13 | 40 | line 64.5 below signal 85.3, histogram falling — the crash component |
| MA structure | +6 | 46 | below MA20 (−4), above MA50 (+6), MA20>MA50 (+4) |
| YTD | −7 | 39 | +209% → top bucket +8, overextension −15 (>150% rule) |
| Volume | 0 | 39 | ratio 1.02 |
| **Final** | | **39** | sell band, 11 under the ≥50 qualifier gate |

(Before the >150% fix this scored 44 — the overextension penalty was
capped at −10 by the dead branch.)
