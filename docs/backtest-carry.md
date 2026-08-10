# Extension-Guard Carry Replay — Build 7: **STOPPED at the integrity gate**

**Build 7 · 2026-08-10.** Script: `scripts/backtest_carry.py` ·
Side-car builder: `scripts/build_noguard_frame.py` · Pins:
`test_backtest_carry.py` (7) · Pre-registration:
[backtest-carry-prereg.md](backtest-carry-prereg.md) (`a6bfd58`,
sha256 `4b11ffd563258b70`) · Declarations:
[backtest-carry-declarations.md](backtest-carry-declarations.md)
(`6330dcc`, amended `+ cash constraint` before any performance result).

> ## The run stopped where the pre-registration told it to
>
> The prereg's first validity check reads: **"S1 vs 5B's S1:
> cent-exact, or stop."** S1 came back **14.11% CAGR / $235,787.28**
> against the committed 5B block's **14.06% / $235,200.09**. Same 1,068
> trades, same −17.50% drawdown, $587 of end equity apart.
>
> **No arm's performance is reported below.** The 2x2 was not run to a
> verdict, the decision table was not read, and none of the six
> branches is claimed. That is the instruction, honoured.

## Why it diverged — diagnosed, not guessed

**5B's panel reads a MUTABLE artifact.** `load_panel` builds its
group map from `public/universe_ranking.json`, which the weekly
rotation rewrites. The 2026-08-08 rotation (`fd83fa4`) **regrouped 3
tickers, added 13 and dropped 12** — APP moved Application Software →
Advertising, DD Specialty Chemicals → Industrial Conglomerates, EA
Interactive Home Entertainment → Electronic Gaming. R28's group caps
(≤20% of equity, ≤3 names per group) therefore bind on different days,
and the equity path moves — without changing the trade count, which is
why the divergence is small and easy to miss.

Verified in both directions (pin 7):

- With the group map **pinned to the artifact as it stood when 5B and
  6A were run** (`1d67d1c`, the 2026-08-01 rotation), the committed
  S1 block reproduces **cent-exactly**: 14.06%, −17.50%, $235,200.09,
  1,068 trades.
- With the working tree's current map, it does not.
- The cause is **not** in Build 7: the *pristine* 5B code path, with no
  carry module loaded and no gate patch applied, reproduces the
  divergence exactly.

**Blast radius: Build 6A is affected too.** Re-running
`scripts/backtest_target.py` today fails its own gate (a) with the
identical `14.11 vs 14.06`. Layer A and Build 5.1 are **not** affected
— they are signal-level and never read the group map; their committed
hashes still reproduce.

This is the [sliding-window pin](testing.md) lesson in a second guise.
That entry says: *anchor a pin to a fixed commit or a content hash,
never to a position in a growing sequence.* The same law applies to a
study's INPUTS — a backtest anchored to a live artifact stops
reproducing the moment the artifact moves, and the committed result
becomes unverifiable through no fault of the code.

## What WAS established before the stop (all gated, all pinned)

- **The guard-off grade column is trustworthy.** It is a genuine
  re-grade through the real `grade_setup` at
  `extension_guard_max = +inf`, and the guard-ON pass of the same
  builder reproduces the committed Layer A frame's `grade` column on
  **all 848,328 rows** — not a sample. Removing the guard only ever
  LIFTS a grade (asserted), and lifts **110,792 ticker-days**: 86,590
  C→B and 24,202 C→A+. The A+ set goes **29,777 → 53,979 (1.81×)**,
  matching the prereg's ~1.8× expectation.
- **The arms differ by the column and nothing else.** The S6 gate
  logic pointed at the guard-ON column reproduces S1 to the cent
  ($0.000000), so S1 and S6 run through one implementation — 5B's own
  engine — with only the grade column varying.
- **The replay engine adds nothing.** A forced replay under fixed
  sizing reproduces its source sim to the cent.
- **The derived risk budget** is **0.1986% of equity per trade**,
  measured from S1's own output, with its identity
  (`size_pct × mean(1 − stop/fill)`) verified to 5.2e-18 against the
  sim's own `r_usd`.
- **A funding constraint the prereg's formula does not contemplate**
  (declarations amendment 1, written before any performance number):
  tight-stop names push size to the 12% cap, and concurrent capped
  positions exhaust cash. Resolved by clamping to what the arm can
  fund — the trade stays, so the prereg's pairing survives — with
  CASH-BIND reported as a first-class validity number beside cap-bind.

## A design note that changes how the resumed study must be read

**R-multiples are share-scale invariant, so the prereg's paired
per-trade R bootstrap is structurally blind to the sizing effect.**
`r_mult = pnl / r_usd`, and both scale linearly with share count:

```
pnl   = shares × [exit×(1−slip) − entry×(1+slip)]
r_usd = shares × (entry − stop)
```

Shares cancel exactly. Demonstrated (pin 6): across 980 funded trades,
fixed vs risk-parity R-multiples are identical to 4.5e-13 and every
exit date matches — while the dollar outcome differs materially. The
sizing effect lives **entirely in the portfolio metrics** (CAGR,
drawdown), which is what declaration 11's paired block bootstrap
measures. When Build 7 resumes, decision clauses 5 and 6 must be read
off the **portfolio** comparison; a per-trade R difference near zero
is the arithmetic, not a finding. (The only trades where R does move
are those clamped to zero shares by the funding constraint, where no
R exists at all.)

## The decision this needs — operator's, not the study's

The study can resume the moment the group-map question is ruled. Two
coherent options:

1. **Pin the studies' panel inputs to a commit** (recommended). 5B,
   6A and 7 read `public/universe_ranking.json` from a fixed SHA —
   `1d67d1c`, the artifact their committed results were produced
   under — exactly as the pin-doctrine law prescribes. Every committed
   result becomes reproducible forever; nothing is re-baselined.
2. **Re-baseline 5B and 6A** against the current artifact and accept
   that the committed numbers move with each rotation. This is the
   option the testing doctrine argues against.

Not recommended and not done: silently pinning the map inside Build 7
so its own gate passes. That would resolve, after seeing a number,
an ambiguity the prereg deliberately made a stop condition.

Held for the ruling: the machinery, the side-car and the pins are all
committed; only the verdict is absent.
