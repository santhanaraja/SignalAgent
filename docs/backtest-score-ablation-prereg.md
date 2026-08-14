BUILD 5.2 PRE-REGISTRATION — SCORE COMPONENT ABLATION
Written before any results exist.

PURPOSE. Build 5.1 ablated the grade's ROWS; the score's five ladders
never were. R5 (score ≥75) is one row, but the score behind it is a
sum of five independent ladders (RSI, MACD, MA, YTD, volume), and a
row-level ablation cannot see which ladder moves days across the
boundary. This study asks, per component: which ticker-days does this
ladder alone move across the grade line, and how did the moved days
perform?

THIS STUDY SHIPS NOTHING TO PRODUCTION. Its outputs are (i) a
shortlist of components to carry into a 5B-style system replay and
(ii) evidence for the D-020a plateau ruling, which is on record as
REVISITABLE. No ladder is changed in production on this study's
evidence alone.

THE SCORE UNDER TEST is the AS-LIVED score — the one that graded the
committed frame's A+ set: clip(50 + rsi + macd + ma + vol + ytd_v1 +
penalty, 0, 100), with the YTD component defined as ytd_v1 + penalty
JOINTLY (both derive from the same input; removing one but not the
other is not removing the component). The current production ladder
(v2, D-020a) enters only as the pre-named backcast in probe (c).

METHOD. Same panel as 5.1 — same caches, same helpers imported not
reimplemented, same knobs, and the same integrity pin: the joint AND
of recomputed row flags must reproduce the committed frame's A+ set
EXACTLY before any statistic is trusted. Additionally pinned here: the
per-component decomposition must recompose elementwise to
score_components_vec's totals (both with and without penalty).

For each component C in {rsi, macd, ma, ytd, vol}, recompute the score
with C ZEROED (a true recompute through the clip, not a subtraction
from the clipped total). Among ticker-days passing ALL OTHER rows
(c1-c4, r2, r3, r4, r7):
  LOST_C   = baseline r5 passes, ablated fails — days that are A+
             only because of C's points
  GAINED_C = baseline r5 fails, ablated passes — days C's points
             alone deny
For each set report: n and n as a share of the A+ bucket; mean and
median fwd-20d (open-to-open, T+1 → T+21); tails P(>+10) P(>+25)
P(>+50) P(<−10) P(<−25) P(<−50); top-5% concentration and ex-top-5%
mean; all of it for full / train / validate; date-cluster bootstrap CI
on (set mean − A+ mean); ticker/date concentration.

THREE PRE-NAMED STRUCTURES — fixed now, no additions after results:

(a) MACD's 16-point cliff at macd == signal. The band "within 1% of
    the crossover" is DEFINED as |macd − signal| < 0.01 × close (MACD
    is in price units; the normalization must be stated because
    nothing in the ladder provides one). Census: n and share of the
    pass-others set at bands 0.1% / 0.25% / 0.5% / 1.0% of close.
    Side test at each band: within band ∩ pass-others, compare fwd-20d
    of bullish (macd>signal) vs bearish days, UNCONDITIONAL on r5 —
    the question is whether the SIDE carries information at the
    boundary, not whether the grade does. Smoothed variant, exactly:
      m_smooth = clip(13 × (macd − signal) / (0.5 × ATR14), −13, +13)
    ±13 is the ladder's existing extreme; 0.5×ATR is the codebase's
    own ATR quantum (the c2 break knob), reused rather than invented.
    The smoothed form replaces BOTH rungs (direction and confirmation)
    with one ramp — flips include confirm-structure changes far from
    the crossover; this is inherent and accepted. Report grade flips
    (LOST/GAINED vs baseline) under m_smooth and their fwd-20d.

(b) The MA component's ±6 ambiguity. The four paths, named:
      +6 P1 = (below20, above50, 20>50)  "pullback in an uptrend"
      +6 P2 = (above20, above50, 50>20)  "recovery from a crash"
      −6 N1 = (below20, below50, 20>50)  "deep pullback, uptrend"
      −6 N2 = (above20, below50, 50>20)  "bounce in a downtrend"
    STRUCTURAL FACT, stated in advance: c1 (close>SMA20) is the MA
    component's first input, so P1 and N1 are EMPTY within the A+ set
    by construction. The path comparison therefore runs on (i) the
    FULL panel and (ii) the c3∩c4 subset (regime + slope — the trend
    context that does not reference the component's own inputs).
    Primary questions: P1 vs P2, and N1 vs N2 — same points, same
    performance? Date-cluster bootstrap CI on each pairwise mean
    difference, both spans.

(c) YTD saturation above +20%. Three probes:
    1. BUCKETS: within pass-others and within A+, fwd-20d
       distributions by as-lived YTD bucket (20,50] (50,100]
       (100,150] >150. Monotone means with CI separation of the
       extreme buckets = the flat region hides structure.
    2. V2 BACKCAST: replace ytd_v1+penalty with the v2 ladder (flat
       12 above 20, no taper, no penalties). Report grade flips and
       their fwd-20d. This measures what the CURRENT production
       ladder would have done on this frame — direct evidence for the
       plateau ruling.
    3. UNCAPPED MONOTONE, exactly: as v2 below 20; above, 12 + 4 per
       bracket crossed at 50/100/150 (16 at >50, 20 at >100, 24 at
       >150). Never decreases. Versus the v2 backcast this can only
       GAIN days (LOST empty by construction, stated not discovered).
       Report the gained set's fwd-20d.

REFERENCE the A+ bucket itself (recomputed per span by the script
under the parity pin; Layer A's full-span figures for orientation:
mean 0.83%, ex-top-5% −0.35%).

PRIMARY QUESTION, per component: did the days this ladder moved
across the grade line perform differently from the days it left?

DECISION TABLE
1. GAINED_C mean EXCEEDS the A+ mean, CI excludes zero, holds in
   BOTH spans, ex-top-5% mean positive → C EXCLUDES outperformers.
   PRIMARY SUSPECT: carry into a 5B-style replay with C neutralized.
   Production unchanged.
2. LOST_C mean is BELOW the A+ mean, CI excludes zero, both spans →
   C ADMITS underperformers at the margin. Same carry as bar 1 (both
   directions argue for neutralizing C); both at once is the
   strongest form.
3. GAINED_C underperforms or LOST_C outperforms (CI excludes zero,
   both spans) → the ladder is doing real work at the boundary:
   VINDICATED for that component.
4. Structure (a): at the 1.0% band, sides indistinguishable (CI on
   the difference includes zero) in both spans AND the band holds
   ≥2% of pass-others days → the cliff assigns ±13 to noise; carry
   m_smooth into the replay. Sides differ with CI excluding zero →
   cliff VINDICATED. Otherwise INCONCLUSIVE. The 0.25% band is
   reported alongside; if it contradicts the 1.0% verdict, record
   INCONCLUSIVE regardless.
5. Structure (b): any pre-named pair differs with CI excluding zero
   in both spans → the component CONFLATES distinguishable states.
   STRUCTURAL FINDING, recorded for its own deliberation; nothing is
   carried — no redesigned MA component is authorized by this study.
6. Structure (c): probes judged by bars 1-3 semantics; bucket
   monotonicity with extreme-bucket CI separation → the cap hides
   structure. ALL (c) outcomes feed the D-020a plateau revisit and
   nothing else.
7. NOTHING clears any bar → all five ladders are inert at the grade
   boundary; the score's granularity is not where the A+ deficit
   lives. Close the component line; Build 6 (exit × sizing) remains
   the primary programme.
8. Two or more components clear bars 1-2 → carry jointly AND
   separately; interaction is not assumed.

NOT IN SCOPE: threshold sweeps (which rungs, what values); any
redesigned component beyond the two pre-named probes (m_smooth, the
uncapped YTD ladder); anything that ships. A component clearing a bar
earns its own study; none is authorized here.

CAVEATS. Signal-level only — no stops, sizing or exits. One window,
examined repeatedly across Builds 5/5.1/5B/6A/7 — which is precisely
why this table precedes the run. The as-lived score is v1-era by
construction; v2 enters only through probe (c). Zero-ablation moves
the score by the component's magnitude, so components with larger
ranges (RSI ±15, MACD ±13) mechanically move more days than volume
(±3); the flip COUNTS are not comparable across components — only
each set's PERFORMANCE is.
