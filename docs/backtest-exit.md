# Build 8 — Exit Sweep

## Unblinding disclosure — read this first, it is not a caveat

This study was **partially unblinded before its primary statistic was
adopted.** The adversarial review that blocked the original design
necessarily quoted headline numbers to prove the leverage mechanism, so
the operator and the analyst had both seen — before amendment 1 was
ruled — the full-span CAGR ordering (E7 > E4 ≈ E6 > E8 > E1), the
leverage table, and E7 1.4753 vs E1 0.4172 on what became the
replacement primary. **The new primary was therefore adopted with its
direction already known.** Its justification is structural — leverage
invariance: shares are identical across arms, so total dollar risk is
identical, and a uniform 7× share scaling leaves the statistic
bit-identical (verified) — not empirical. What nobody had seen when
amendment 1 was ruled, and what every verdict below therefore rests on
blind: **the CIs, the train/validate split, the fragility check, and
the bind rates.** Those unseen gates decided everything that follows.

## The chain

Pre-registration [backtest-exit-prereg.md](backtest-exit-prereg.md)
(`2e96862`, sha256 `9b392b5f7e34c5d2…`) — **stands unchanged, flaw
included.** The adversarial fan-out (11 reviewers, all completed,
three independent confirmations of the blocking finding) established
its design constraints are jointly unsatisfiable for
slower-than-incumbent exits; the implementation's negative-cash
resolution levered slow arms up to 3.57× gross/equity at T-bill
financing, with a 0.982 correlation between average leverage and
CAGR. Amendment 1
([backtest-exit-prereg-amendment.md](backtest-exit-prereg-amendment.md),
`df1f2a2`, sha256 `d7a52c84cd5fef6b…`) was ruled after that review and
**before the results artifact was read**: primary = paired per-trade
dollar expectancy; portfolio verdicts restricted to zero-cash-negative
arms; clause 5 asymmetric with the voider enforced; the slippage check
replaced by the gap-through distribution. Results:
[backtest-exit-results.json](backtest-exit-results.json), hash
`0a3ecd30a9d9e62d`, schema `backtest-exit-2`. Script:
`scripts/backtest_exit.py`. Pins: `test_backtest_exit.py` (durable
form — content hashes + path-resolved ordering, no fixed SHA).

**Integrity:** gate 1 — the harness S1/score reproduces the committed
5B block exactly (end equity 235,200.09, 1,068 trades). Gate 2 — this
study's engine over the extracted ledger reproduces the harness E1
curve to **$0.00** max daily deviation with the identical trade set.
Pairing — 8 arms × 1,068 trades, identical by construction. The
amendment-1 statistical code was verified by three adversarial agents
before results were read: no defects (14/14 and 22/22 synthetic
attack checks; leverage invariance proven by construction).

## One admissibility surprise, ruled by the bright line

Amendment clause 2 names the expected clean set as E1/E2/E3/E5 — but
its RULE is "zero cash-negative days," and **E3 fails it by exactly
one day** (min cash −$12,967, one day negative). The rule governs over
the enumeration: the admissible set is **E1, E2, E5**, and E3's
portfolio metrics are descriptive-only alongside E4/E6/E7/E8. That is
what bright-line rules are for; a judgment call to wave one day
through would be the first step back to the flaw the amendment fixed.

## The primary — paired per-trade dollar expectancy vs E1

Sum(Δpnl)/sum(r_usd), entry-date-cluster bootstrap, decisive only with
CI excluding zero in BOTH D-006 spans:

| arm | full diff (R) | train CI | validate CI | verdict | fragile |
|---|---|---|---|---|---|
| E2 sma5 | −0.3020 | [−0.4213, +0.0523] | [−0.8270, −0.0513] | INSIDE | no |
| E3 sma10 | −0.1982 | [−0.2932, +0.0731] | [−0.5758, −0.0242] | INSIDE | no |
| E4 sma50 | +0.3665 | [−0.0230, +0.5002] | [−0.0145, +1.1492] | INSIDE | **yes** |
| E5 sma20+R | −0.0061 | [−0.0005, +0.0195] | [−0.0586, +0.0141] | INSIDE | no |
| E6 atr2.5 | **+0.3585** | [+0.0157, +0.6160] | [+0.0051, +0.8761] | **BEATS** | no |
| E7 atr4.0 | **+1.0581** | [+0.3136, +1.5684] | [+0.2478, +2.2594] | **BEATS** | no |
| E8 atr2.5+R | +0.1709 | [−0.0065, +0.3587] | [−0.0744, +0.4518] | INSIDE | **yes** |

**The finding: the two pure ATR trails beat the SMA20 stop per unit of
identical risk, with both-span CI exclusion, and neither is fragile.**
E7 (4.0×ATR) adds just over one full R per trade on the identical
1,068 entries — expectancy 1.4753 vs E1's 0.4172, and the effect holds
in both eras (train +1.28 vs 0.36; validate +1.64 vs 0.46). E6 clears
both spans by a thinner margin. E4 (sma50) points the same direction
but straddles in both spans and its advantage rests on its top 3
trades (FRAGILE). The fast MA arms lose decisively in validate and
straddle in train — INSIDE by the both-span rule.

The tiny-R decomposition earns its place in one row: E6's unweighted
per-trade R mean reads **−5.6051** [−18.97, +1.21] — poisoned by
near-zero-R trades — while the same trades excluding r_usd < $100
(n=929) read **+0.4092** [+0.1436, +0.6945], consistent with the
weighted primary. The unweighted mean is reported and is not usable.

## Clause 5 — the ratchet is dead, twice

| test | bind rate | CI both spans? | voided | adopt |
|---|---|---|---|---|
| E5 vs E1 | **1.3%** (14 of 1,068 exits changed; armed 37.5%) | no | **YES — under 5%** | **no** |
| E8 vs E6 | 58.0% | no (train −0.13 [−0.36, +0.06]; validate −0.24 [−0.54, +0.04]) | no | **no** |

The SMA20 ratchet is 6A's breakeven floor all over again: armed on a
third of trades, it changes 14 exits in 1,068 — economically inert,
and the enforced voider kills clause 5 regardless of its CI. The ATR
ratchet binds constantly (58%) and **hurts its base in both spans** —
holding the trail up against a rolling ATR takes the trades out of
exactly the volatility expansions the trail exists to survive. The
prereg's recorded prediction — "THE RATCHET is where I expect a
finding" — is wrong on both counts. Its other half — "the ATR arms are
where I expect a surprise" — is the study's result.

## Portfolio verdicts — measurable arms only

Both admissible non-incumbent arms land in **branch 4** (all
differences inside the CI): E2's descriptive full-span ΔCAGR is −7.3
points (a real cost of exiting fast) but not both-span decisive; E5 is
E1 to within noise (−0.10 CAGR points, expectancy diff −0.0061). **E1
stays by incumbency among the arms this design can measure.**
E4/E6/E7/E8 are UNMEASURABLE on portfolio metrics under the fixed
ledger (amendment clause 2) and defer to a capital-feasible
redeploying replay. Descriptive, with the mandatory leverage columns:

| arm | CAGR% | MDD% | end eq | expR | TIM% | bars | min cash | neg days | mean g/e | max g/e |
|---|---|---|---|---|---|---|---|---|---|---|
| E1 | 14.06 | −17.50 | 235,200 | 0.4172 | 86.1 | 12.2 | +12,720 | 0 | 0.55 | 0.90 |
| E2 | 6.76 | −8.24 | 153,011 | 0.1152 | 72.6 | 3.9 | +15,621 | 0 | 0.19 | 0.90 |
| E3 | 9.58 | −13.33 | 181,206 | 0.2190 | 81.1 | 7.1 | −12,967 | 1 | 0.34 | 1.09 |
| E4 | 20.17 | −24.92 | 330,074 | 0.7837 | 92.2 | 25.7 | −213,725 | 952 | 1.02 | 2.21 |
| E5 | 13.96 | −17.51 | 233,775 | 0.4111 | 86.1 | 12.1 | +12,720 | 0 | 0.54 | 0.90 |
| E6 | 20.01 | −18.96 | 327,357 | 0.7757 | 91.8 | 26.8 | −308,979 | 973 | 1.02 | 2.23 |
| E7 | 28.24 | −38.12 | 503,824 | 1.4753 | 94.4 | 53.3 | −587,316 | 1,363 | 1.77 | 3.57 |
| E8 | 17.36 | −17.77 | 283,102 | 0.5881 | 88.0 | 17.2 | −94,700 | 392 | 0.70 | 1.61 |

Slow-arm CAGR/MDD/end-equity rows are leverage artifacts — E7's 28.24%
CAGR is 1.77× average gross financed at T-bills — which is precisely
why they are not ruled on. Duplicate concurrent lots (a construction a
real book would refuse): E7 120, E6 58, E4 41, E8 17, E3 1.

## Gap-through — the redesigned check, and the premise inverted

Per real exit, how far below the trigger level the next open printed:

| arm | exits | below trigger | mean | median | p90 | worst |
|---|---|---|---|---|---|---|
| E2 | 1,061 | 80.4% | +1.005% | +0.741% | +2.630% | +24.96% |
| E1 | 1,055 | 82.5% | +1.283% | +0.794% | +3.468% | +19.18% |
| E4 | 1,040 | 81.2% | +1.350% | +0.848% | +3.844% | +20.92% |
| E6 | 1,029 | 78.6% | +1.252% | +0.793% | +3.605% | +17.98% |
| E7 | 1,005 | 75.8% | +1.286% | +0.785% | +3.742% | **+40.31%** |
| E8 | 1,045 | 80.8% | +1.428% | +0.801% | +3.926% | +22.68% |

(E3 +1.202/+3.090, E5 +1.291/+3.486 — between their neighbors.)
The amendment's inversion hypothesis is confirmed **at the tails, not
the means**: medians are flat (~0.8%) and means differ little, but p90
grows monotonically from the fastest arm (E2 2.63%) through the slow
ones (E8 3.93%), and E7 owns the worst single fill — 40% below
trigger. Roughly 80% of ALL exits fill below trigger in every arm:
unmodelled slippage is a real cost for every rule, not a
fast-arm-specific one. Reported, not modelled, per the amendment.

## Verdict

**No adoption.** Nothing measurable beats E1 on portfolio metrics; the
exit-length question among fast MAs closes on this window (branch 4,
E1 by incumbency). Both ratchets are rejected — one inert, one
harmful. **The carry: E6 and especially E7 — the pure ATR trails —
enter the capital-feasible redeploying replay as PRIMARY-BEATS
candidates**, non-fragile, both-span, on identical risk. That replay
(the study amendment clause 2 defers to) decides whether the per-trade
edge survives real capital constraints, where Build 7 showed the book
is capacity-bound and a longer hold means fewer entries taken.

Prediction scorecard, per the prereg's recorded prediction: MA lengths
branch 2-or-4 — broadly right (branch 4 / INSIDE). The ratchet finding
— wrong, twice. The ATR surprise — right, and it is the build's
result.

## Caveats

Scorer v1 grades; transfer to v2 is an ASSUMPTION, not a finding (per
prereg). One window, examined across seven builds. The per-trade
primary is blind to capacity AND to compounding — a +1R/trade edge on
53-bar holds is not a portfolio result until the redeploying replay
runs. Costs 5bps/side modelled; slippage reported (gap-through), not
charged. E3 excluded from portfolio verdicts by one negative-cash day
— the bright line, applied as written. Two cosmetic residues from the
verification pass, recorded: the bind rate is rounded to one decimal
before the 5% comparison (a true rate in [4.95, 5.0) would escape the
voider), and the exchange rate uses full-span point estimates while
decisiveness is span-CI-based (no exchange-rate branch fired, so
neither residue touched a verdict).
