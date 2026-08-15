BUILD 8 PRE-REGISTRATION — EXIT SWEEP
Written before any results exist.

QUESTION. The exit rule has never been compared to anything. Every
committed study froze it at "close below the SMA20." Five separate
results now point at the stop as where the edge lives — Layer A (the
848,328-ticker-day signal-level study: the A+ grade does not rank
forward returns), 5B (the strategy replay with real stops: no arm
distinguishable, and the stop owns the small-loss shape), the 5B
addendum (every benchmark inside every arm's band, so the machinery
is the edge), 6A (the profit-target sweep: no arm adopted), and
Build 7 (the extension-guard carry replay: the guard makes no
measurable difference and the book is capacity-bound). Nobody has
tested the stop itself.

SCOPE. Exits only. This is the exit half of what Build 6B scoped as
sizing × exits; the sizing half is deferred because Build 7 measured
risk-parity as worse than fixed (−4.41 CAGR points, CI excluding
zero), and the remaining sizing question — the fixed percentage, and
the ceiling — is a separate study.

ARMS — 8. All close-basis: a CLOSE below the level exits at the NEXT
OPEN, per D-018 (the close-basis law: state transitions happen only
on confirmed closes). No intraday stops; that is settled law.

  E1  SMA20                    INCUMBENT — the integrity gate
  E2  SMA5
  E3  SMA10
  E4  SMA50
  E5  SMA20 + RATCHET
  E6  ATR trail: highest close since entry − 2.5×ATR14
  E7  ATR trail: highest close since entry − 4.0×ATR14
  E8  ATR trail 2.5× + RATCHET

  RATCHET means stop_t = max(stop_{t−1}, computed_stop_t), seeded at
  the entry stop. It never lowers.

  NOTE the ATR trails do NOT ratchet by construction — ATR14 is
  rolling, so an expanding ATR lowers the trail even when the highest
  close is unchanged. That is precisely why E8 exists.

HELD FIXED ACROSS ALL ARMS: the entry gate (production A+, 5B's S1
rules), position sizing (fixed 6.5%), next-open fills, 5bps/side,
R28 caps (≤20% and ≤3 names per group under a regime-scaled ceiling),
the ceiling ladder, and score-order selection among admitted
candidates.

NO REDEPLOYMENT. Capital freed by an earlier exit returns to cash and
is NOT redeployed. This is deliberate: it holds the ENTRY LIST
identical across arms so the comparison is genuinely paired. It means
this study measures THE EXIT RULE and cannot see the capacity effect
Build 7 identified. A redeploying replay of the surviving arms is a
separate study and may point the other way — a tighter stop could
lose per trade and win per year.

SCORER v1 (the frozen pre-D-020a scorer: YTD anchored on a 6-month
frame, with the >100% penalty intact). Required for comparability
with five committed studies and for the integrity gate. State in the
report that the conclusion therefore applies to v1 grades and its
transfer to v2 is an ASSUMPTION, not a finding.

STATISTICS
  Entries are identical across arms; only exits differ.
  So compare PAIRED per-trade differences vs E1 and bootstrap those.

  R-MULTIPLES ARE VALID HERE, unlike in Build 7. There the paired
  per-trade R bootstrap was structurally blind because sizing cancels
  in R (1R = entry minus the initial stop, so shares divide out).
  Here sizing is FIXED and the EXIT varies, so R genuinely differs
  per arm. Report it.

  Portfolio metrics (CAGR, max drawdown) are NOT straightforwardly
  paired because the equity paths diverge once exits differ. Use a
  block bootstrap over the paired equity paths, as Build 7 did after
  that error was caught.

  PRIMARY: CAGR.  SECONDARY: max drawdown, expectancy per R.

DECISION TABLE
 1. An arm beats E1 on BOTH CAGR and drawdown, both CIs excluding
    zero, IN BOTH D-006 SPANS → ADOPT.
 2. Beats on CAGR, loses drawdown, both outside noise → report the
    EXCHANGE RATE in CAGR points per drawdown point. Operator's
    choice; the study adopts nothing.
 3. Loses on both, outside noise → REJECT that arm.
 4. All differences inside the CI → no change. E1 stays BY
    INCUMBENCY and the exit-length question closes on this window.
 5. THE RATCHET, ruled separately: if E5 beats E1, or E8 beats E6,
    with CI excluding zero in both spans → ADOPT THE RATCHET
    independently of which base wins. It is a modifier, not an arm.
 6. Any arm whose advantage rests on the top 3 trades by more than
    half is labelled FRAGILE regardless of branch.

MANDATORY VALIDITY CHECKS — report before any conclusion
  · E1 vs 5B's S1: cent-exact, or STOP. Nothing downstream is
    trustworthy if the incumbent does not reproduce.
  · EXITS PER ARM, and the BREAK-EVEN SLIPPAGE that would erase each
    arm's advantage. 5B measured 32–37% of stop exits filling BELOW
    the entry stop. E2 will exit several times more often than E4 —
    if a fast arm wins by 1% while taking triple the exits,
    unmodelled slippage plausibly eats the whole result. THIS IS THE
    CHECK MOST LIKELY TO OVERTURN A WINNER.
  · RATCHET BIND RATE. On what share of trades does the ratchet
    actually change an exit? 6A's breakeven floor armed on 435 trades
    and changed 19 — economically inert. If the ratchet binds on
    under 5% of trades, say so plainly and treat clause 5 as void.
  · TIME IN MARKET per arm, and average bars held.
  · Trade COUNT per arm — identical by construction. If not, the
    no-redeployment constraint has leaked and the pairing is broken.

NOT IN SCOPE: sizing variants, profit targets (6A answered those),
intraday stops (D-018), redeployment, and any exit basis not listed.
No additional arm is authorised after results are seen.

PREDICTION, RECORDED SO IT CAN BE WRONG: branch 2 or 4 for the moving
-average lengths — faster exits buy drawdown and cost CAGR, as trend
structure has in every prior study. THE RATCHET is where I expect a
finding, and the ATR arms are where I expect a surprise, because they
are the only ones that normalise the exit the way the extension guard
already normalises the entry.

CAVEATS. One window. v1 grades. Costs modelled, slippage not — and
that omission flatters the fast arms specifically. No redeployment,
so the capacity effect is invisible by design.
