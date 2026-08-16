BUILD 9 PRE-REGISTRATION — SIZING AND CEILING
Written before any results exist. This completes what Build 6B
scoped as sizing × exits; Build 8 (the exit sweep — found the 4×ATR
trail earns +1.06R per trade but takes 4.4× the holding time) did
the exit half.

QUESTION. Two numbers govern how much risk the book carries, and
NEITHER WAS EVER MEASURED. They were chosen. Position size has been
6.5% since inception; the ceiling ladder 90/50/25/5 was set by
judgement in D-008 (the regime chassis). Build 5B froze both ON
PURPOSE so entries could be tested cleanly, and nothing has gone
back for them since.

WHY THIS IS NOW THE PRIORITY. The return decomposes to
(risk deployed per year) × (expectancy per R). Eight studies attacked
expectancy and moved it once — Build 8's E7, which pays for its gain
out of turnover. Turnover itself is BLOCKED: the book sits at the
ceiling on 94.0% of days with 58,085 candidate-denials. So the only
untested term with headroom is HOW MUCH RISK PER TRADE, and how much
the ceiling permits in total.

WHAT THE TWO FACTORS ACTUALLY DO — the framing this study rests on:
  the CEILING controls how much money is at work
  the SIZE controls how many names that money is split across
They are not the same lever and must not be collapsed.

ARMS — 9. Notation: size% / ceiling ladder (Trending/Choppy/Caution/
Risk-off).

  P1   6.5% / 90-50-25-5      INCUMBENT — the integrity gate
  P2   5.0% / 90-50-25-5      more names, same exposure
  P3   8.0% / 90-50-25-5      fewer names
  P4  10.0% / 90-50-25-5      fewest names
  P5   6.5% / 100-60-30-10    more exposure
  P6   6.5% / 70-40-20-5      less exposure
  P7   6.5% / 90-90-90-90     FLAT — no regime scaling at all
  P8  10.0% / 100-60-30-10    maximum-risk corner
  P9   5.0% / 100-60-30-10    maximum-names corner

  NO CEILING MAY EXCEED 100%. This study tests allocation, never
  leverage.

  P7 IS THE MOST IMPORTANT ARM AND IS NOT ABOUT RETURN. The one
  seed-independent positive in the entire evidence base is that arm
  drawdown bands never overlapped SPY's −33.7%. The regime ladder is
  the presumed cause and has never been tested. P7 removes the
  scaling while holding everything else: if it matches P1 on
  drawdown, the ladder is decorative; if its drawdown is much worse,
  the ladder is what produces the property the system is actually
  for. Rule on this separately from the frontier.

HELD FIXED: the entry gate (production A+, 5B's S1), the SMA20 exit
(Build 8 adopted nothing), next-open fills, 5bps/side, score-order
selection, R28's per-group caps, and the regime chassis itself —
only the CEILING LEVELS vary, never the regime classification.

REDEPLOYMENT IS REQUIRED AND ENTRIES WILL DIVERGE. That is the
mechanism under test, not a flaw. Freed capital returns to the pool
and competes for the next candidate in score order.

════════ THE FEASIBILITY CLAUSE — Build 8's lesson ════════
Build 8's prereg held three constraints that were JOINTLY
UNSATISFIABLE for slow arms, and the engine resolved it by borrowing
to 3.57× — silently, in a docstring. CAGR then correlated 0.982 with
leverage and the primary metric measured borrowing, not skill.

THIS STUDY MUST NOT REPEAT IT:
  · CASH FLOOR IS ZERO, HARD. No arm may borrow under any
    circumstance, for any reason, at any point.
  · An arm that cannot afford the next entry SKIPS IT. That skip is
    the measurement, not an error.
  · RUN THE FEASIBILITY GATE BEFORE THE STUDY, NOT AFTER: for every
    one of the nine arms, assert min(cash) >= 0 across the entire
    window. If any arm breaches, ABORT AND REPORT — do not proceed
    to results.
  · Report min cash, days-at-ceiling, and denial counts PER ARM in
    the results table, so a future reader can see the constraint
    binding rather than infer it.

OVER-CEILING AFTER A REGIME DOWNGRADE — state it explicitly, because
the ladder moves under held positions. Production behaviour: HOLD
existing positions, BLOCK new entries until back under. That is
action_needed, not forced liquidation. Implement exactly that, and
report how many days each arm spends over its own ceiling.

R28 INTERACTION, worth naming in advance: at 10% position size the
20%-per-group cap binds at TWO names instead of three. So P4 and P8
have a materially tighter group constraint than the others. Report
group-cap denials per arm separately from ceiling denials — Build 8
measured only 16 group denials in six years at 6.5%, and that number
should be expected to move.

SCORER v1 (the frozen pre-D-020a scorer). Required for the integrity
gate and for comparability with five committed studies.

STATISTICS
  Entries diverge, so there is NO PAIRING. Use random-K bands as the
  null, 50 seeds, exactly as 5B did.
  ALSO report a block bootstrap over the equity paths, and the TRADE
  OVERLAP between each arm and P1 — the arms share a selection rule
  and differ only in capacity, so overlap should be high, and if it
  is not, that itself is the finding.

  PRIMARY IS NOT CAGR. CAGR scales with risk by construction; an arm
  taking more risk beating one taking less is arithmetic, not
  evidence. The primary output is THE (CAGR, MaxDD) FRONTIER across
  all nine arms with bands.

DECISION TABLE
 1. DOMINANCE — an arm beats P1 on BOTH CAGR and drawdown, CI-clean,
    in BOTH D-006 spans → ADOPT. This is the only free-lunch branch
    and I do not expect it.
 2. WELL-ORDERED FRONTIER — no dominance, but higher-risk arms buy
    return at a stable price → report THE EXCHANGE RATE in CAGR
    points per drawdown point, per arm. The study adopts nothing;
    the operator picks a drawdown budget. Frame it as 6A did.
 3. NON-MONOTONE FRONTIER — an interior arm beats the ones on either
    side of it on both metrics (e.g. 8% beats both 6.5% and 10%) →
    that is a finding about an OPTIMUM and is reported as such, with
    the caveat that one window cannot locate an optimum precisely.
 4. ALL INSIDE THE BANDS → no change. The sizing question closes on
    this window and the parameters stand by incumbency.
 5. THE LADDER, RULED SEPARATELY (P7 vs P1): if P7's drawdown is
    materially worse with CI separation in both spans, THE REGIME
    LADDER IS EARNING ITS KEEP and that is recorded as the first
    direct evidence for it. If P7 matches P1 on both metrics, the
    ladder is decorative and that is a much bigger finding than
    anything on the frontier.
 6. Any arm whose advantage rests on the top 3 trades by more than
    half is labelled FRAGILE regardless of branch.

MANDATORY VALIDITY CHECKS — before any conclusion
  · P1 vs 5B's S1: cent-exact, or STOP.
  · THE FEASIBILITY GATE above — min cash >= 0, all nine arms.
  · Trade count, distinct positions held, mean concurrent positions,
    and mean exposure/equity per arm.
  · Max single-name % of capital actually reached per arm.
  · Ceiling denials AND group-cap denials, separately.
  · Days spent over the ceiling after a downgrade, per arm.

NOT IN SCOPE: exit variants (Build 8), entry variants (five studies),
leverage of any kind, and any size or ceiling not listed. No arm is
added after results are seen.

PREDICTION, RECORDED SO IT CAN BE WRONG: branch 2 — a well-ordered
frontier with no dominance, because the strategy's Sharpe was already
indistinguishable from benchmarks' and scaling a strategy does not
create edge. I expect P7 to show MATERIALLY WORSE DRAWDOWN than P1,
confirming the ladder does the work. And I expect the 5% arms to show
slightly better risk-adjusted numbers than the 10% arms through
diversification, with lower absolute return.

CAVEATS. One window. v1 grades. Costs modelled, slippage NOT — and
Build 8 measured ~80% of exits filling below trigger at a median
0.8%, a cost carried by none of these figures and borne roughly
equally by all arms.
