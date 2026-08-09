# D-020a — One honest YTD: real calendar anchor + capped component + versioned scorer

| | |
|---|---|
| **ID** | D-020a |
| **Date** | 2026-08-09 (ruled by the user's build order; implemented same day) |
| **Status** | **Ruled** |

## Context — the anchoring defect

`compute_ytd_return` (now frozen as `compute_ytd_return_v1`) anchors on
the **first valid close of the current calendar year present in
whatever frame it is handed** — not the prior year's last close. Two
production paths handed it different frames:

- **Universe path** (`universe_builder._ticker_metrics`, 1y fetch):
  frame reaches January → anchors ~Jan 2. Nearly calendar YTD, minus
  the year's first-day move.
- **Grading path** (`score_stock` on the 6mo fetch — the signals main
  loop, the framework quality-score fetch, the on-demand ticker route):
  after early July the frame no longer reaches January, so the anchor
  silently became **the oldest bar in the window** — a rolling
  six-month return wearing a YTD label.

**Seasonal structure**: the divergence is exactly zero from January to
early July (the 6mo window still contains January), then grows every
trading day to year-end. Measured 2026-08-08 (read-only investigation):
**529 of 533** evaluated names disagreed; ERAS read 412.81% on one path
and 53.29% on the other; 62 names differed by >25 points; the two paths
disagreed across the >100% penalty boundary for 12 names. HPE's
recorded 54-vs-64 score split was this defect straddling the penalty.

## Change 1 — one anchor, both paths

`compute_ytd_return_v2` anchors on the **last close of the prior
calendar year**. Both paths now report the same YTD for the same
ticker-day. The grading path keeps its 6mo indicator frames untouched
and feeds the YTD from a **separate 1y frame** — no indicator window
widened (stop condition verified empirically: across all 533 names,
v1-vs-v2 non-YTD components are bit-identical on both paths).

Fallback, recorded not hidden: a frame with no prior-year bar (listing
younger than the year, or a failed 1y fetch) anchors on the first
current-year close — v1's anchor — with `ytd_basis:
"first_close_of_year"` recorded in the artifact. An outage degrades
visibly; it never impersonates a real YTD (D-019 spirit).

**Deliberately scoped OUT** (own future deliberation; explicit v1
calls with comments): the UUP/XLE macro breaker YTD inputs (their >5%
/ >15% thresholds were calibrated on the old construct — re-anchoring
them silently changes breaker firing) and the ^GSPC drawdown check's
current-year-high window. `sp500_ytd` and the index-strip YTDs DID
move to v2 — `beating_sp500` compares stock YTD against them, and
mixed anchors in one artifact would be a new incoherence.

## Change 2 — the capped component, and the curve as found

The v1 ladder, reported before the change as required: **it was
already declining before the 100% boundary.** Base curve: ≤−10 → −10 ·
(−10,0] → −4 · (0,5] → +2 · (5,20] → +6 · **(20,50] → +12 (the
PEAK)** · (50,100] → +8 (the taper — already on the way down) · then
penalties: >100 → −10 (net −2) · >150 → −15 (net −7). Value at
100.00: **+8**; at 100.01: **−2**.

Per the pre-ruled plateau instruction — "the plateau must be set at
the curve's PEAK value, not at a point already on the way down" — the
v2 ladder plateaus at **12 from the peak onward**: rises exactly as v1
through +2/+6/+12, then holds flat for everything above 20%. No taper
at >50, no penalty at >100/>150, never reverses, never resumes
rising, and no value anywhere exceeds v1's maximum. A name at +400%
scores the YTD points of a name at +100% (both 12). Consequence of
the peak rule, stated plainly: names in (50,100] gain +4 (8→12) —
freezing the plateau at 8 would have anchored it to a value already
on the penalty's downslope, which the instruction forbids.

### The plateau ruling, explicit (amended 2026-08-09, pre-push, by the operator)

- The v1 ladder was **already declining before 100%** — peak +12 on
  (20,50], tapering to +8 on (50,100], then −2 above. The taper
  started at 50%, not at the threshold. **This was not previously
  known** before the curve report above surfaced it.
- Two plateau options existed: **12** (the curve's peak) or **8**
  (the value at the 100% boundary). Plateau-12 additionally lifts
  every name in (50,100] by +4; plateau-8 would have touched only
  names above 100%.
- **Plateau-12 was chosen deliberately by the operator, after the
  trade-off was put to him.** Rationale: a monotone non-decreasing
  curve, and the same mean-reversion logic removed consistently
  rather than halfway.
- **Recorded against it**: the (50,100] band has NO evidence against
  it. Build 5.1 measured only the >100% flip set, and the
  design-coherence argument that justifies removing the >100%
  penalty — its threshold was calibrated on an input being replaced —
  does NOT extend to the 50–100 band, which the anchor change leaves
  untouched. The chat recommendation was plateau-8; it was overruled
  with the reasoning above.
- The (50,100] taper is logged as a **newly visible, never measured
  construct** — available as a pre-specifiable hypothesis for the
  carry replay or a 5.1 follow-up. **This ruling is marked
  REVISITABLE.**

**Why removing the penalty is the conservative move, not the
aggressive one**: the >100% threshold was calibrated against the
frame-anchored input, which no longer exists. Under real YTD, 23 of
533 names sit above 100% today versus 16 under the old number — the
old threshold would now fire on a different, larger population than
the one it was tuned on. Removing a knob whose calibration input
vanished is design coherence, not performance chasing.

## Change 3 — the versioned scorer

- `score_stock_v1` / `score_ytd_points_v1` / `compute_ytd_return_v1`:
  **FROZEN**, verbatim copies (deliberate duplication — a shared body
  would let future edits mutate the frozen scorer silently). Never
  called by production; exist so the committed studies keep
  reproducing. `scripts/backtest_doctrine.py` pins its ladder import
  to v1 explicitly; Layer A's ladder-parity pin drives v1.
- `score_stock_v2` (+ v2 ladder and anchor): what production calls —
  signals main loop, universe builder, framework quality score,
  ticker route, Score Lab's `simulate_score`.
- **Anchors pinned** (`test_score_v2.py`, pin 5): after the cutover,
  v1 still reproduces Layer A's `input_hash d7df85e8ad6ba244` (frame
  rebuilt through the real loop, 848,328 rows), Build 6A's results
  hash `847d6306f4355fb9`, and the recorded candidate-grade parity
  replay (516 grades across 12 committed artifacts).
- **Era stamps**: signals.json, framework.json and the universe
  artifacts now carry `scorer_version: "score_stock_v2"`. Artifacts
  without the key were baked by v1 and are never retro-relabelled
  (`scorer_era`, pinned).

## This ships a Build 5.1 suspect ahead of the carry replay — explicitly

Build 5.1's row ablation named the YTD>100% penalty flip a PRIMARY
SUSPECT, bar 5: *carry to a 5B-style replay; production unchanged.*
This build removes that penalty from production **before** the carry
replay has run. The ruling's reasoning, recorded: the removal is
justified on **design coherence** (the threshold's calibration input
no longer exists once the anchor is honest), not on the 5.1
performance evidence — which remains FRAGILE (ERAS 31.5% of the flip
set's sum) and still owes the system-level replay.

**And the 5.1 finding itself must now be re-read**: the flip set was
measured on the OLD frame-anchored construct. It is evidence about a
**rolling-six-month >100% penalty**, not a calendar-YTD >100% penalty.
The carry replay, when it runs, should carry the construct as
measured — the rolling form — or re-derive the flip set under real
YTD, and say which.

## Impact (measured 2026-08-09, before staging; probe committed as
`scripts/d020a_impact.py`, full per-name table as
[D-020a-impact-table.csv](D-020a-impact-table.csv) — a live-data
snapshot, not a deterministic artifact)

- Grading path: **311/533** names change score (−22 to +19, median
  +6 among changed; 231 up / 80 down). Universe path: 106/533 (87
  up / 19 down).
- Universe ≥50 gate: **5 cross up** (APGE, ERAS, MRNA, MU, STX),
  none down. Row-5 ≥75: **35 up / 20 down** (the anchor swing
  dominates: NOW 86→64 — it is *down* 18% on the real year; IR
  65→81; CLX 59→75).
- **Today's board regraded under v2, date-consistently** (quality
  scores recomputed on frames as of the 2026-07-24 bake, isolating
  the scorer change from subsequent tape; the frozen v1 replay
  reproduces the recorded quality scores exactly): **all 43 recorded
  candidate grades unchanged — no name becomes or stops being A+.**
  (A first, date-mixed probe showed NTAP B→A+; that flip was 16 days
  of market drift contaminating the counterfactual — its
  date-consistent v2 quality is 68, below row 5's 75 — and it is
  recorded here as the methodology lesson, per review.)
- The **twelve boundary names** — the names whose two OLD paths
  disagreed across the >100% penalty boundary (the set the Context
  defines): 000660.KS, AMAT, APGE, CRWD, DDOG, ERAS, FTNT, HUM,
  LITE, STX, VSAT, WDC. Under v2 every one lands on the plateau's
  12; the four that carried the −2 penalty on the grading path
  (APGE 39→53, CRWD 72→86, DDOG 35→49, HUM 49→63) gain +14. The
  largest score *swings* are a different, overlapping-free set led
  by NOW (−22), IR (+16), CLX (+16) — anchor honesty, not the cap.
- `group_ytd` breaker input: two groups' average YTD flips negative →
  positive (Air Freight & Logistics, Construction & Engineering) —
  the "leadership lost" condition disarms for them under the honest
  number.
- SNDK: real YTD **410.66%** (6mo number said 107.78); score 24→38 —
  the cap lifts it nowhere near any gate. ERAS: real **394.89%**
  (53.29 on the 6mo path); universe score 41→60, crossing the ≥50
  gate.

## Consequences

Scores move mid-season by construction (the July–December divergence
was the defect; January would have hidden it). The twelve boundary
names swing hardest in both directions. The committed studies are
insulated by the frozen v1 stack and its three pinned anchors. The
UUP/XLE breaker YTDs still carry the seasonal defect by explicit
scope-out and await their own ruling.
