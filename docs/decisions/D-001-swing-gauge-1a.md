# D-001 — Build 1A swing gauge: 3 voters + backdrop gate + exact-spec ladder

| | |
|---|---|
| **ID** | D-001 |
| **Date** | 2026-07-05 |
| **Status** | Ruled — **revisit trigger fired 2026-07-11** (Build 4); successor architecture ruled ([D-008](D-008-gauge-b-architecture.md), 2026-07-12). This record governs production until the Gauge B build ships, then flips to Superseded-by-D-008 |

## Context

The pre-1A gauge averaged 5 equal voters (VIX, HY, breadth, SPY/200DMA,
yield curve) on a weeks-horizon question, which let slow inputs dilute
fast ones and produced entry-loose readings. Build 1A recommitted the
gauge to the swing horizon: 3 fast voters, the 200DMA demoted to a binary
backdrop gate, the yield curve demoted to a non-voting macro input (with
the ^IRX-mislabeled-as-2Y bug fixed by honest labeling).

## Options considered

| Option | Summary | Why (not) |
|---|---|---|
| 1 — keep 2/3 → Trending | Looser top rung | **Rejected by user**: "reproduces the old gauge's entry looseness and nullifies the refactor" |
| 2 — exact-spec ladder | Trending only at 3/3; 2→Choppy; 1→Caution; 0→Risk-off; ANY risk_off vote caps at Caution; Risk-off needs 0 risk_on or ≥2 risk_off; gate caps risk-on at Caution, fails closed | **Ruled.** Caution on Jun 24/25/30 was desired: "stops were firing, the sector re-crashed Jul 2 — those days SHOULD read defensive" |
| 3 — Choppy floor at 1 risk_on | Softer middle | Conditional fallback only if Jun 19–22 replay had shown Caution-or-worse; it showed Choppy, so the condition never fired |

Decision rule, as recorded at ruling time in [docs/regime.md](../regime.md)
("Known coupling", committed in 466637e): *"The gauge measures; rules
act. […] do not tune the measurement to quiet the rule."* (The elided
middle clause routes the R4-confirmation question to its own review —
which became [D-002](D-002-r4-sunday-cadence.md).)

## Evidence

- 10-day June/July replay comparison (old vs new gauge): 6/10 days
  diverge; accepted at ruling.
- Voting thresholds as shipped: VIX 5d ≤18/≤22; HY OAS ≤3.0/≤4.0
  (percentile 60/40 on the HYG/IEF fallback basis); breadth RSP/SPY 20d
  ±0.5; gate at raw pct < 0 ([framework/config.yaml](../../framework/config.yaml) `regime.gauges`).

## Ruling + rationale

Option 2, exactly as specified by the trader. Strictness at the top rung
is the point: Trending is a full-deployment claim and must require all
three fast voters. Any credit deterioration (risk_off vote) caps optimism
regardless of tally.

## Consequences

Trending becomes rare by construction. The ladder's "0 risk_on →
Risk-off" floor prints Risk-off in ordinary low-conviction tape when the
HY voter is structurally dark-red (see revisit).

## Revisit triggers

1. A statistical harness contradicting the configuration across regimes.
   → **FIRED 2026-07-11** by Build 4 ([docs/backtest-regime.md](../backtest-regime.md)):
   the OAS absolute thresholds cast zero HY risk-on votes 2015-2023, the
   gauge sat out nine of eleven years, lost decisively to the naked
   200DMA, and the info test inverted. The threshold *family* — absolute
   OAS cutoffs — is the indicted component; the ladder itself replayed
   production exactly (pin 1).
2. Trending median run-length ≤2 days in live operation (flicker, not
   regime) — also observed in the Build 4 reconstruction.

## Retest recipe

```
python3 test_regime.py                 # ladder + gate + streak pins (19)
python3 test_regime_extraction.py      # replay of recorded production
python3 scripts/backtest_regime.py     # the harness that fired the trigger
```

## Links

- Commits: 466637e (1A)
- Docs: [docs/regime.md](../regime.md)
- Jira: PER-508 description item 10; superseding thread → D-008
