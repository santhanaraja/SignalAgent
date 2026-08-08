BUILD 5.1 PRE-REGISTRATION — ROW ABLATION
Written before any results exist.

PURPOSE. Layer A's distribution decomposition found A+'s entire
positive expectancy lives in its top 5% of days (140.4% of the bucket
sum; the bottom 95% sums NEGATIVE at −0.35%/20d), while its
construction trims the right tail 2–7× harder than the left. This
study asks which specific rows are responsible.

THIS STUDY SHIPS NOTHING TO PRODUCTION. Its only output is a
shortlist of rows to carry into a 5B-style system replay. Layer A is
signal-level; 5B already showed no gate is distinguishable once stops
apply. No row is added, removed or relaxed in production on this
study's evidence alone.

METHOD. For each grade row R, isolate the EXCLUDED SET — ticker-days
that pass every other row but fail R. For each set report:
  n, and n as a share of the A+ bucket
  mean and median fwd-20d (open-to-open, T+1 → T+21)
  tails: P(>+10%) P(>+25%) P(>+50%) P(<−10%) P(<−25%) P(<−50%)
  top-5% concentration and ex-top-5% mean
  all of the above for full / train / validate

ROWS TESTED — fixed list, no additions after results are seen:
  R2 extension ≤1.8×ATR
  R3 approach filter
  R4 RSI 45–70
  R5 score ≥75
  R7 runway ≥15 sessions
  the c-conditions, separately where separable
  plus one construct: the score's YTD>100% penalty — rescore without
  it, report which days change grade and how they performed
  (R6 breaker is untestable — a ruled Layer A deletion)

REFERENCE the A+ bucket itself: mean 0.83%, ex-top-5% −0.35%,
right tail 10.24 / 1.26 / 0.177, left tail 8.07 / 1.18 / 0.068.

PRIMARY QUESTION per row: did the days this row excluded outperform
the days it admitted?

DECISION TABLE
1. Excluded-set mean EXCEEDS the A+ mean, the bootstrap CI on the
   difference excludes zero, it holds in BOTH spans, and its
   ex-top-5% mean is positive
   → PRIMARY SUSPECT. Carry into a 5B-style system replay with that
     row relaxed or removed. Production unchanged.
2. Exceeds A+ in only one span, or its ex-top-5% mean is also
   negative → INCONCLUSIVE. Record; do not carry.
3. Excluded-set mean is BELOW the A+ mean → the row is doing real
   work. Keep it; the right-tail-trimming hypothesis is falsified
   for that row.
4. NO row clears bar 1 → the mechanism hypothesis is wrong: the
   right-tail deficit is not attributable to any single row. Abandon
   this line and make Build 6 (exit × sizing) the primary programme.
5. Two or more rows clear bar 1 → carry them jointly AND separately;
   interaction is not assumed.

NOT IN SCOPE: threshold sweeps. This asks which rows are expensive,
not what the thresholds should be. A row clearing bar 1 earns its own
threshold study; none is authorised here.

CAVEATS. Signal-level only — no stops, sizing or exits. One window.
C's tail concentration is 92.2%, so tail-dependence is normal and the
DIFFERENCE is the finding. This window has been examined repeatedly,
which is precisely why this table precedes the run.
