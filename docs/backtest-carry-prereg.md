BUILD 7 PRE-REGISTRATION — EXTENSION GUARD CARRY REPLAY
Written before any results exist. Commit verbatim, own commit,
content-immutability pinned by hash.

QUESTION. Build 5.1 found the extension guard (≤1.8×ATR above SMA20)
excluded 23,893 machine-perfect days earning 1.43% against A+'s 0.83%,
both spans, positive ex-top-5%, 527 tickers. That is a SIGNAL-level
finding with no stops. Does it survive at SYSTEM level — and is the
guard a selection rule or a sizing rule in disguise?

ARMS — 2x2 factorial
                  guard ON (1.8)      guard OFF
  fixed 6.5%          S1                  S6
  risk-parity         S8                  S7

S1 is the incumbent and must reproduce 5B's S1 EXACTLY. That is the
integrity gate; if it does not, nothing else in the run is trustworthy.

RISK-PARITY DEFINITION — derived, not chosen
  shares = min( risk_budget / (entry - stop), cap / entry )
  risk_budget = S1's REALIZED mean risk-per-trade as a % of capital,
    measured from S1's own output. Aggregate risk is therefore MATCHED
    across arms and only its DISTRIBUTION differs. No free parameter.
  cap = 12% of capital. This is the one chosen number: roughly 2x the
    fixed size and inside R28's 20% group cap. Declared, not swept.

HELD FIXED ACROSS ALL ARMS — every other entry row, the SMA20
close-basis stop, next-open fills, 5bps/side, R28 caps, the regime
ceiling ladder, and SCORE-ORDER selection among admitted candidates
(matching 5B's S1, so the undefined selection rule cannot confound).

SCORER VERSION: v1. Required for comparability with 5B and for the
integrity gate. State plainly in the report that the conclusion
therefore applies to v1 grades and its transfer to v2 is an
ASSUMPTION, not a finding.

STATISTICS — and the two comparisons are NOT alike
  GUARD effect (S1 vs S6, S8 vs S7): UNPAIRED. Different trade
    populations. Needs 5B's random-K bands. 5B could not distinguish
    any gate this way, so expect limited power and say so.
  SIZING effect (S1 vs S8, S6 vs S7): PAIRED. Identical trades,
    different share counts. Bootstrap the per-trade differences
    directly, as 6A did. Materially more power.
  Report the INTERACTION: does the sizing effect differ with the guard
    on versus off? Do not pool if it does.

PRIMARY: CAGR.  SECONDARY: max drawdown, expectancy per R.

DECISION TABLE
 1. Guard removal beats S1 on BOTH CAGR and drawdown, outside the band
    -> the guard is costing; carry to a threshold study (which this
       does NOT authorise).
 2. Wins CAGR, loses drawdown, both outside the band
    -> report the exchange rate in CAGR points per drawdown point.
       Operator's choice. No adoption by the study.
 3. Loses on both, outside the band
    -> the guard is doing real work; 5.1's finding was signal-level
       only; the question CLOSES and the guard stays.
 4. All guard comparisons inside the band
    -> not distinguishable; the guard stays BY INCUMBENCY; the question
       closes as unresolvable on this window.
 5. Risk-parity beats fixed on the PAIRED bootstrap, CI excluding zero
    -> ADOPT risk-parity sizing, independently of the guard verdict.
 6. Paired comparison inside the CI -> fixed sizing stays.

VALIDITY CHECKS — report before conclusions
 · S1 vs 5B's S1: cent-exact, or stop.
 · CAP-BIND RATE. If the 12% cap binds on more than half of S7/S8's
   trades, risk-parity is not really being tested — those arms have
   degenerated toward fixed sizing. Say so explicitly; the decision
   clauses for 5 and 6 are then void.
 · Trade COUNT per arm. Removing the guard admits ~1.8x the population
   (5.1: the excluded set was 81% the size of the A+ bucket). S6/S7 are
   a materially different strategy, not S1 with a tweak.
 · How often R28 caps or the ceiling bind per arm — with more admitted
   candidates they bind more, and that is part of the effect.

MY PREDICTION, ON THE RECORD SO IT CAN BE WRONG: branch 4 or 2 for the
guard, and branch 5 is where the real chance of a finding lies. The
guard question has 5B's shape and 5B resolved nothing; the sizing
question is paired and mechanical.

CAVEATS. One window. v1 grades. Costs modelled but slippage is not —
5B measured 32-37% of stop exits filling below the entry stop, and
that penalty falls harder on the wider-stop trades S6/S7 admit, so the
no-guard arms are flattered by the model.
