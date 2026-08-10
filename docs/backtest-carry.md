# Extension-Guard Carry Replay — Build 7: the guard is not the binding constraint

**Build 7 · 2026-08-10.** Script: `scripts/backtest_carry.py` ·
Side-car builder: `scripts/build_noguard_frame.py` · Results:
[backtest-carry-results.json](backtest-carry-results.json)
(`results_hash 3a6a3d60b4ef8084`) · Pins: `test_backtest_carry.py` (7) ·
Pre-registration: [backtest-carry-prereg.md](backtest-carry-prereg.md)
(`d8f5f42`, sha256 `4b11ffd563258b70`; post-registration amendment 1
`0cc36d8`) · Declarations:
[backtest-carry-declarations.md](backtest-carry-declarations.md)
(`faf824b`, amendment 1 `5ce1627`).

**Integrity gates, all three, before any number below:**
(a) S1 reproduces the committed 5B S1/score block **cent-exactly** —
14.06%, −17.50%, $235,200.09, 1,068 trades — under the PINNED
universe artifact (`1d67d1c`; see "the stop that preceded this run");
(b) the S6 gate logic pointed at the guard-ON column reproduces S1 to
**$0.000000**, so the arms differ by the grade column and nothing else;
(c) both forced replays under fixed sizing reproduce their source sims
to the cent.

## Verdict, read against the pre-registration

> **GUARD — BRANCH 4. Not distinguishable; the guard stays BY
> INCUMBENCY and the question closes as unresolvable on this window.**
> Paired by seed (5B's addendum construction — both gates under the
> same shuffle, so selection noise is matched), the guard-off gate
> beats the guard-on gate by a mean of **+0.40 CAGR points inside a
> [−6.57, +6.73] band, with 54% of seeds positive** — a coin flip. The
> observed score-ordered difference, **+0.97pp**, sits INSIDE that
> band; so does the drawdown difference (**+0.37pp** inside
> [−5.38, +8.10]). The second guard comparison agrees (S8→S7: +0.84pp
> CAGR, +6.65pp drawdown, both inside).
>
> **SIZING — risk-parity LOSES, and the table does not name this
> cell.** Under amendment 1's portfolio reading: with the guard on,
> **−4.41 CAGR points, CI [−8.24, −1.14] excluding zero**, and
> **−$53,152** of terminal equity; with the guard off, −4.54pp,
> CI [−10.24, +0.42] (straddling), −$57,249. Clause 5 required
> risk-parity to BEAT fixed with a CI excluding zero; clause 6 covers
> "inside the CI". The measured cell — **loses, CI-clean in the
> guard-on half** — is neither. *Interpretive step, stated: the ACTION
> is clause 6's (fixed sizing stays) but the evidence is stronger than
> clause 6 contemplates, so it is reported as uncovered rather than
> filed under 6.* **No adoption. Fixed sizing stays.**

## The mechanism — why branch 4 is not merely a power failure

Removing the guard lifts the A+ population by **1.81×** (29,777 →
53,979 ticker-days) and yet produces **fewer trades**: 1,012 vs 1,068
(0.95×). The prereg expected ~1.8× and called S6/S7 "a materially
different strategy". They are different — Jaccard **0.523**, with 714
trades shared, 354 only-S1 and 298 only-S6 — but not *larger*.

The reason is in the denial counts: **ceiling denials more than
double, 58,085 → 122,717.** At 6.5% sizing under a 90% regime ceiling
the book is **capacity-bound, not candidate-bound**. Extra admitted
candidates queue for capacity that is already full; they displace each
other rather than adding positions. A selection row cannot move a
system whose binding constraint is capacity — which is a mechanistic
explanation for "not distinguishable", not merely an absence of power.

## The 2×2 (full window)

| Arm | Guard | Sizing | CAGR | maxDD | End equity | Trades | Exp/R |
|---|---|---|---|---|---|---|---|
| **S1** | on | fixed 6.5% | **14.06%** | −17.50% | $235,200 | 1,068 | 0.4172 |
| **S6** | off | fixed 6.5% | 15.03% | −17.13% | $248,504 | 1,012 | 0.3768 |
| **S8** | on | risk-parity | 9.65% | −16.72% | $182,048 | 1,068 (1,051)* | 0.3073 |
| **S7** | off | risk-parity | 10.49% | **−10.07%** | $191,255 | 1,012 (1,002)* | 0.3495 |

\* Two counts, deliberately distinguished: the **population** (which is
the source arm's, by construction — that is what makes the pairing
exact) and, in brackets, the trades with a **defined R-multiple**. The
funding clamp of declarations amendment 1 reduced **17 S8 entries and
10 S7 entries to zero shares** — they remain in the population so the
pairing holds, but they transact nothing and have no R, so every
per-trade statistic in this table is computed on the bracketed count.
S1 and S6 have no such trades.

Spans (CAGR train / validate): S1 8.96/23.24 · S6 10.61/22.89 ·
S8 6.47/15.26 · S7 7.69/15.38. Every arm's edge is validate-weighted,
as every layer of this campaign has found.

## Interaction (declaration 12) — pooled on CAGR, NOT on drawdown

The sizing effect is essentially identical with the guard on or off on
CAGR: −4.41 vs −4.54pp, **difference −0.13pp** — poolable, and pooled
in the verdict above. On drawdown it is **not**: **+0.78 vs +7.06pp**,
6.3 points apart. Risk-parity shrinks drawdown materially only when
the guard is off (S7's −10.07% is the shallowest cell in the study,
7.4 points shallower than the incumbent). The two are therefore
reported separately, per the prereg's instruction not to pool a
differing interaction. **This is an observation, not a branch:** S7 is
a compound change (guard AND sizing), and the prereg's exchange-rate
clause applies to the guard comparison alone, which landed inside the
band.

## Validity checks — reported before the conclusions, as instructed

- **S1 vs 5B's S1: cent-exact.** Achieved only after pinning the
  universe artifact; see below.
- **Cap-bind: 20.2% (S8) / 17.6% (S7)** — under the prereg's 50%
  threshold, so **clauses 5 and 6 stand** (not void). Risk-parity was
  genuinely tested on ~80% of its trades.
- **Cash-bind (declarations amendment 1): 9.4% / 5.3%** — the funding
  clamp binds on roughly one trade in eleven of S8. Material enough to
  name, far from dominant. On those trades the arm is sized by its own
  funding limit, not by risk parity. **Of those, 17 (S8) and 10 (S7)
  were clamped all the way to zero shares** — entries that exist in the
  population for pairing but transact nothing.
- **Trade counts:** S1 1,068 · S6 1,012 · S8 1,068 · S7 1,012 (the
  replays inherit their source populations by construction; 1,051 and
  1,002 of those carry a defined R — see the table's footnote).
- **Admission binds:** S1 ceiling 58,085 / group-pct 15 / group-count 1
  / gap-below-stop 1,217. S6 ceiling 122,717 / group-count 5 /
  group-pct 12 / gap-below-stop 1,313. Per declaration 4 these are
  measurable only on S1/S6 — the forced replays do not re-evaluate
  admission — and are never imputed to S8/S7.

## The stop that preceded this run, and the fix

The first attempt **stopped at its own integrity gate**: S1 came back
14.11%/$235,787.28 against the committed 14.06%/$235,200.09. Cause:
`backtest_systems.load_panel` read `public/universe_ranking.json`
**live**, and the 2026-08-08 rotation (`fd83fa4`) regrouped 3 tickers,
added 13 and dropped 12 — moving how R28's group caps bind. Build 6A
failed identically. **Ruled and fixed:** the studies now read that
artifact from the fixed commit `1d67d1c` (the version their committed
results were produced under) and **raise** if the anchor is
unreachable — never a silent fallback to the live rotation. This is
the [sliding-window law](testing.md) applied to a study's *inputs*.
See [backtest-inputs.md](backtest-inputs.md) for the full sweep of
every study input and its pinned/live status.

## Caveats

- **v1 scorer throughout** (prereg). Every conclusion here is about v1
  grades; transfer to the D-020a v2 scorer is an **assumption, not a
  finding**.
- One window; the guard's band is ~13 CAGR points wide, so this window
  could not have resolved a small guard effect even in principle —
  which is itself the honest reading of branch 4.
- **Slippage is not modelled** (prereg caveat): 5B measured 32–37% of
  stop exits filling below the entry stop, and that penalty falls
  harder on the wider-stop trades the guard-off arms admit. The
  no-guard arms are **flattered** by the model, which makes branch 4's
  "no advantage" reading conservative in the right direction.
- The paired per-trade R bootstrap specified in the original clauses 5
  and 6 is **void** (amendment 1): R-multiples are share-scale
  invariant, verified at 4.5e-13 across 980 funded trades with
  identical exit dates. It is computed and carried in the results JSON
  under `void_per_trade_r_bootstrap` so the arithmetic is on the
  record, and it feeds no verdict.
- 5.1's signal-level finding is **not overturned** — it is unrepeated
  at system level. The guard's excluded days really did earn more; the
  system cannot convert that into portfolio return because it has no
  capacity to deploy it.

## What this authorises

Nothing. Production is unchanged, the guard stays, fixed sizing stays,
and per prereg branch 1 no threshold study is authorised (branch 1 did
not fire in any case). The open question the mechanism raises — that
the binding constraint is capacity, not selection — belongs to a
sizing/ceiling study, which is Build 6B's territory and needs its own
pre-registration.

## Reproduction

```
python3 scripts/build_noguard_frame.py    # ~25 min, two full re-grades
python3 scripts/backtest_carry.py         # ~20 min incl. 100 random-K sims
python3 test_backtest_carry.py            # all seven pins
```
