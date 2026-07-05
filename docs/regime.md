# Swing Regime Gauge

Horizon: **swing** — weeks-long holds, checked daily. The gauge answers one
question: is aggressive swing deployment permitted right now, and at what
aggression level? Months-horizon macro inputs do not vote here.

Implementation: `framework/regime_calculator.py`. Output states are exact,
load-bearing strings consumed by the framework page, rule engine, and theme
entry gates — never rename them:
`Risk-on / Trending`, `Risk-on / Choppy`, `Caution`, `Risk-off`.

## Architecture: 3 voters + 1 gate + macro inputs

### Voters (equal weight, EOD values)

| Voter | Measure | risk_on | caution | risk_off |
|---|---|---|---|---|
| `vix_5d_avg` | VIX 5-day simple average | ≤ 18 | 18–22 | > 22 |
| `hy_spread` | HY OAS, FRED `BAMLH0A0HYM2` (pp) | ≤ 3.0 | 3.0–4.0 | > 4.0 |
| `hy_spread` (fallback) | HYG/IEF ratio, 60d percentile | ≥ 60th | 40–60th | < 40th |
| `breadth` | % of S&P 500 above 50DMA (S5FI) | ≥ 60 | 50–60 | < 50 |
| `breadth` (fallback) | RSP/SPY 20d ratio change | > +0.5% | ±0.5% | < −0.5% |

Note the voting boundaries are **two-threshold**: the `risk_off_threshold`
values in `config.yaml` (VIX 27, HY 5.0, breadth 30) are *not* used in
voting — risk_off begins where caution ends (VIX 22, HY 4.0, breadth 50).
VIX 27 is used only by the intra-week override banner. Also note the two
fallbacks are range-relative/momentum measures, not the same quantity as
their primaries; the active source is published in each gauge's `source`
field. As of 2026-07, production runs on both fallbacks because
`FRED_API_KEY` is not set in the deploy environment.

### Backdrop gate: SPY vs 200DMA (not a voter)

Any SPY close below its 200DMA **caps risk-on states at Caution** — no
aggressive swing deployment below the 200DMA, ever. Semantics:

- The gate is a **cap, not a vote**: it downgrades `Risk-on / *` to
  `Caution`; it never upgrades and never touches `Risk-off`.
- **Fails closed**: if SPY/200DMA data is unavailable, the cap applies
  (unknown ≠ permitted). The `reason` field distinguishes `below_200dma`
  from `data_unavailable` so a data outage is never mistaken for a real
  breach.
- The gate is a zero-hysteresis knife edge at 0.0% by design. Under the
  old 5-gauge system SPY had a graded band (−2% was still just a caution
  vote); the gate is deliberately stricter.

### Macro inputs (computed, never voting)

`macro_inputs.yield_curve` — 30Y−2Y Treasury spread. Preserved for future
consumers (Gauge B, WFC-specific logic). Classification (>+50bp risk_on,
0–50bp caution, inverted risk_off) is informational only.

Sources, in order:
1. **FRED API** `DGS30`/`DGS2` — requires `FRED_API_KEY` (same fetch
   pattern as the HY spread gauge). This is the only source of the true 2Y.
2. **FRED public CSV** (keyless) — Akamai resets non-browser clients as of
   2026-07; kept in case it recovers.
3. **yfinance `^TYX` − `^IRX`** — labeled **"30Y-3mo"**. `^IRX` is the
   13-week bill; it is never presented as the 2Y (the pre-refactor code
   mislabeled it, overstating the spread ~40–55bp). The `dgs30`/`dgs2`
   output keys are emitted only when the data really is DGS30/DGS2.

## State mapping (`_determine_state`)

Ladder on the risk_on tally, with qualifiers, evaluated in this order:

1. `risk_off >= 2` → **Risk-off** (two hard negatives dominate everything)
2. `unavailable >= 2` → **Caution** (a one-voter tally is not evidence;
   also prevents a data outage from printing Risk-off)
3. `risk_on == 0` and all voters reporting → **Risk-off** (zero
   affirmative evidence — note three *caution* votes print Risk-off)
4. `risk_off >= 1` → **Caution** (risk_off votes are heavier than caution:
   ANY risk_off vote caps the state, regardless of the tally — credit
   deteriorating while vol is calm must never print risk-on)
5. `risk_on == 3` → **Risk-on / Trending**
6. `risk_on == 2` → **Risk-on / Choppy**
7. else (`risk_on == 1`, or partial data) → **Caution**

Then the backdrop gate caps `Risk-on / *` at `Caution` when shut.

Full truth table, (risk_on, caution, risk_off) with all voters reporting:

| Tally | State | Via |
|---|---|---|
| 3,0,0 | Risk-on / Trending | ladder |
| 2,1,0 | Risk-on / Choppy | ladder |
| 2,0,1 | Caution | any-risk_off cap |
| 1,2,0 | Caution | ladder |
| 1,1,1 | Caution | any-risk_off cap |
| 1,0,2 | Risk-off | 2+ risk_off |
| 0,3,0 | Risk-off | zero risk_on |
| 0,2,1 | Risk-off | zero risk_on |
| 0,1,2 | Risk-off | 2+ risk_off |
| 0,0,3 | Risk-off | 2+ risk_off |

With unavailable voters: `(2,0,0)+1dark` → Choppy, `(1,·,0)+1dark` →
Caution, `(0,·,0)+1dark` → Caution (Risk-off requires full data unless
2 risk_off are confirmed), any 2+ dark → Caution, `(0,0,2)+1dark` →
Risk-off.

## Behavioral deltas vs the old 5-gauge system

Replayed over Jun 19 – Jul 2 2026 (recorded voter signals): the new gauge
prints Choppy where the old printed Trending on Jun 19/22 and Jul 1
(2-of-3 voters risk_on no longer qualifies for full deployment), and
Caution where the old printed Choppy on Jun 24/25/30 (a single risk_on
voter no longer holds the entry zone open). Jun 26/29 (HY risk_off)
remain Caution. Structural changes to know:

- **Curve inversion no longer vetoes the swing gauge.** The old system
  could not print a risk-on state while 30Y−2Y was inverted (one risk_off
  vote blocked both risk-on states). Post-refactor it can — by design;
  the curve is a months-horizon input. (Real instance: Jun 16–17 2026
  printed Caution solely on the old TLT/SHY curve-proxy vote.)
- **Below the 200DMA is now an absolute cap** (old: −2% band voted
  caution and Trending was still printable at SPY −1%).
- **All-caution now prints Risk-off** (old: Choppy, held up by the two
  structurally risk-on slow voters).

## History schema

`framework/state/regime_history.json` entries from the 3-voter cutover
carry `gauges` with 3 keys plus `backdrop_gate` and `macro_inputs`
snapshots; counts range 0–3 (0–5 before). The history keeps only the
**last run per date** — intraday runs are overwritten by the EOD run
(e.g. 2026-06-25 flashed Caution intraday on an HY dip; the surviving
EOD entry reads Choppy).

## Known coupling (flagged, intentionally not changed here)

`theme_ranker.py` fires `EXIT SIGNAL — regime degraded` and **removes
active themes from `qualified_themes.json` on a single Caution/Risk-off
print** (any run, including intraday; the 2-consecutive-Sunday
confirmation protocol is computed but not consulted). The gauge measures;
rules act. Whether R4 should require confirmation is a separate semantics
review — do not tune the measurement to quiet the rule.

## API

`/api/framework/gauges.json` returns `gauges` (3 voters), `backdrop_gate`,
`macro_inputs`, counts (0–3), regime string and action. The full payload
(`framework.json`, `/api/framework/latest.json`) carries the same fields
under `regime`. All framework output passes through
`signal_engine.sanitize_for_json` (NaN/Inf → null) before writing.
