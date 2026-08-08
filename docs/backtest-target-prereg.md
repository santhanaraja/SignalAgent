BUILD 6A PRE-REGISTRATION — PROFIT-TARGET SWEEP
Written before any results exist.

QUESTION. Does taking partial profit at an R-multiple target improve
the system, and at what cost?

FIXED, NOT SWEPT: the entry gate (production A+, S1 rules), position
sizing (6.5%), the trailing stop (SMA20, close-basis, D-018), fills
(next open after the signal close), costs (5bps/side), the regime
ceiling ladder. Only the exit-target rule varies.

ARMS
 T0  no target — pure SMA20 trail                    [BASELINE = today]
 T1  sell 50% at 2R, trail the rest
 T2  sell 50% at 3R, trail the rest
 T3  sell 33% at 2R, 33% at 4R, trail the rest
 T4  no partial sale, but the stop floors at breakeven once 2R is hit
 T5  sell 50% at 3R AND floor the remainder at breakeven

R is defined once, at entry: entry price minus the INITIAL stop. It
does not re-base as the stop trails.

MECHANICS TO DECLARE, not decide later
 · Target orders are resting limits and fill INTRABAR when the high
   touches the level. Stops remain close-basis. This asymmetry is
   real, not a modelling shortcut.
 · Capital freed by a partial sale returns to CASH and is NOT
   redeployed. Redeployment would change the trade population and
   destroy the paired comparison.
 · Breakeven floor = stop becomes max(SMA20, entry). It only binds
   while the SMA20 sits below the entry; it never lowers a stop.

STATISTICS. Because every arm trades the same names on the same days,
compare PAIRED per-trade differences vs T0 and bootstrap those
directly. Do not use random-K bands; selection is held fixed.

PRIMARY: CAGR.  SECONDARY: max drawdown.  Both reported with paired
bootstrap CIs against T0.

DECISION TABLE
 1. An arm beats T0 on BOTH CAGR and drawdown, CI excluding zero
    → ADOPT it. (A free lunch; unlikely.)
 2. An arm loses CAGR but improves drawdown, both outside noise
    → report the EXCHANGE RATE — CAGR points surrendered per point
      of drawdown saved — and present it as a deliberate choice for
      the operator. No arm is adopted by the study.
 3. An arm loses on both → REJECT.
 4. All differences sit inside the paired-bootstrap CI
    → NO target rule is adopted. The trailing stop stands alone, and
      any subsequent choice rests on the psychological argument
      ALONE, which must be labelled as such in the record.

MANDATORY CONCENTRATION CHECK. The distribution work found the
system's profit lives in its top 5% of outcomes, so an arm difference
could rest on a handful of trades. For every arm report: how many
trades reached each target, and what share of the total CAGR
difference vs T0 comes from the three largest contributors. If the
top 3 trades drive more than half the difference, the result is
labelled FRAGILE regardless of which branch it lands in.

NOT IN SCOPE: sizing variants, trail-length variants, entry changes.
Those are Build 6B. No threshold beyond {2R, 3R, 4R} is authorised.

CAVEATS. One window. The trade population is whatever the production
gate produced — if that is only a few hundred trades, power is
limited and the CIs will say so.
