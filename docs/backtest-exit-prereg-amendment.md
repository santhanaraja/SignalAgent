BUILD 8 PRE-REGISTRATION — AMENDMENT 1
Ruled 2026-08-15, after the adversarial review and BEFORE the results
artifact was read. The original pre-registration
(docs/backtest-exit-prereg.md, commit 2e96862, sha256
9b392b5f7e34c5d2…) STANDS UNCHANGED, flaw included. This amendment
cites it and supersedes the clauses named below; it does not edit it.

WHY. The review (11 reviewers, all completed, three independent
confirmations of the blocking finding) established that the original
design's constraints — identical entries, no redeployment, ceilings
held fixed — are jointly unsatisfiable for arms that exit SLOWER than
the incumbent. The implementation resolved the contradiction with
negative cash financed at the IRX rate, which inflates the registered
primary (CAGR) for slow arms specifically: average gross/equity up to
1.77 (max 3.57), cash negative on up to 83% of days, and a 0.982
correlation between average leverage and full-span CAGR across arms.
Both integrity gates are structurally blind to it (they exercise only
E1, which never goes cash-negative).

1 · PRIMARY BECOMES LEVERAGE-CLEAN. Branch mapping runs on the paired
    per-trade dollar expectancy — sum(Δpnl)/sum(r_usd), paired
    bootstrap — valid for all eight arms because shares are identical,
    so total risk deployed is identical. CAGR and drawdown demote to
    DESCRIPTIVE, and every table carrying them must carry the leverage
    columns beside them: min cash, days cash-negative, mean and max
    gross/equity.

2 · PORTFOLIO VERDICTS RESTRICTED. CAGR and drawdown may be ruled on
    ONLY for arms with ZERO cash-negative days — E1, E2, E3, E5.
    E4, E6, E7 and E8 are declared UNMEASURABLE on portfolio metrics
    under this design and deferred to a capital-feasible redeploying
    replay. E8 included: 392 negative days is contaminated, just less.

3 · CLAUSE 5 SURVIVES AS AN ASYMMETRIC TEST ONLY. E8-vs-E6 on the
    per-trade metric is admissible BECAUSE the leverage bias runs
    AGAINST the ratchet — E8 holds less, so a win is conservative.
    The statistic is PINNED (no three readings): the paired per-trade
    dollar expectancy difference, CI excluding zero in BOTH D-006
    spans, for E5-vs-E1 and for E8-vs-E6. The <5% bind-rate voider is
    ENFORCED on the adoption boolean, not just reported: a ratchet
    that changes fewer than 5% of exits cannot be adopted regardless
    of its CI.

4 · THE FOUR REMAINING REVIEW FINDINGS, FIXED. The branch table is
    rewritten to cover the REVERSE of clause 2 (loses CAGR, wins
    drawdown — the original prereg's own predicted outcome, which the
    first mapper filed under "noise") and to COMPUTE the exchange rate
    in CAGR points per drawdown point for both directions; both-span
    decisiveness is stated PER BRANCH: every decisive branch (beats,
    loses, and both exchange-rate branches) requires CI exclusion in
    both spans, and anything else is the inside-noise branch. The
    weighted trade statistic is reported BESIDE the unweighted
    per-trade R mean, with the tiny-R decomposition disclosed (the
    unweighted mean rides on near-zero-R trades — one $0.08-risk trade
    contributes thousands of R — exactly the pathology the parent
    harness documents for its own headline). Break-even slippage is
    guarded against nan-poisoning wherever it is still computed.

5 · THE SLIPPAGE CHECK IS REDESIGNED, NOT REPAIRED. The original
    premise was structurally false — under a fixed ledger every trade
    exits once, so exit counts barely differ across arms. AND THE
    PREMISE LIKELY INVERTS: a tight stop exits on small pullbacks, a
    wide stop exits on genuine breakdowns, so gap-through is plausibly
    WORSE for slow arms. The count-based break-even check is REPLACED
    by the DISTRIBUTION of gap-through on exits per arm — how far
    below the trigger level the next open actually printed: the share
    of exits filling below trigger, and the mean / median / p90 /
    worst gap. Reported; not modelled as a cost.

6 · UNBLINDING, DISCLOSED IN THE REPORT'S OPENING, NOT ITS CAVEATS.
    Through the review channel, the operator and the analyst have both
    seen: the full-span CAGR ordering (E7 > E4 ≈ E6 > E8 > E1), the
    leverage table, and E7 1.475 vs E1 0.417 on the replacement
    primary. The new primary is therefore adopted with its DIRECTION
    already known, and the justification is STRUCTURAL — leverage
    invariance (identical shares, identical total risk) — not
    empirical, and must be stated as such. What neither has seen, and
    what the decision therefore still rests on blind: the CIs, the
    train/validate split, the fragility check, and the bind rate. The
    amended table requires BOTH-SPAN CI EXCLUSION and the TOP-3-TRADE
    FRAGILITY TEST — those are the unseen gates.

Everything else in the original pre-registration — the eight arms,
D-018 close-basis exits, the fixed entry ledger, scorer v1, the
validity checks not named above, clause 6 (fragility), and the
NOT-IN-SCOPE list — stands as written.
