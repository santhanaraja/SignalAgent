# Dashboard vs Framework — YTD, score, signal, trade signal

Two pages show overlapping numbers for the same tickers, computed by different paths from
different artifacts. This is what each one calculates, in what order, from which constants —
and the nine places where they disagree. Read from source, not inferred.

> **Sources.** score `signal_engine.py:1157–1266` · YTD `:835` · band `:1297` ·
> trade signal `:1614` · swing `:1800` · intraday `:1993` · group `:2643` ·
> grade `position_signals.py:433` · gauges `framework/regime_calculator.py:90–135` ·
> row thresholds `config.yaml:200–217`

---

## 1 · What each surface reads

This is the root of every divergence below.

| surface | reads |
|---|---|
| `index.html` — the dashboard | **`signals.json` only** |
| `framework.html` — the framework tab | **`/api/framework/latest` AND `signals.json`** |
| `search.html` — ticker search | `/api/ticker` (computed live) |

**The dashboard reads one artifact. The framework tab reads two.** So if the two artifacts ever
skew, the dashboard stays self-consistent and the framework tab contradicts *itself*.

The skew window is real but currently shut: `POST /api/framework/run` re-bakes `framework.json`
*without* re-running `signal_engine`. After a deploy carrying a rotated universe, the two sides
could carry different group names — and row 6 is a string join between them (§11).

---

## 2 · YTD — how it works now

`compute_ytd_return` (`signal_engine.py:835`).

**Since D-020a shipped on 2026-08-09, both paths compute real calendar year-to-date, anchored on
the last close of the prior calendar year.** The historical defect below is what that change
fixed — it no longer describes the system.

| path | frame it uses now | anchor |
|---|---|---|
| **UNIVERSE** — score ≥ 50 gate | `period="1y"` | prior-year last close ✓ |
| **GRADING** — row 5, score ≥ 75 | indicators still `period="6mo"`; **YTD on a separate 1y fetch** | prior-year last close ✓ |

**The 6-month frame was deliberately NOT widened.** It still serves RSI, MACD and the moving
averages byte-untouched; only YTD moved to its own fetch. That is why the change was safe — v1
versus v2 differed in the `ytd` component and nothing else, across all 533 names, pinned.

**How to confirm it live:** every grade payload carries `"ytd_basis": "prior_year_close"`. That
field exists because of this fix and reports the anchor actually used.

### What it used to do

The same function was called on two frames. The universe path's 1-year window reached back to the
previous August, so it contained January and anchored correctly. **The grading path's 6-month
window did not — after early July, "the first close of the current year" became simply the oldest
bar in the window, so the number silently became a rolling six-month return wearing a YTD label.**

The defect was seasonal and perfectly predictable: **exactly zero divergence from January to early
July, then growing every day to year-end.** At discovery, **529 of 533 names disagreed** — ERAS
read 412.81% on one path and 53.29% on the other, and twelve names sat on opposite sides of the
score penalty boundary on the same day.

The scorer was **versioned rather than replaced**: `v1` frozen so the committed studies still
reproduce, `v2` in production since 2026-08-09.

### Three caveats on `ytd_basis`

- **Five values exist** — `prior_year_close`, `first_close_of_year`, `no_current_year_bars`,
  `insufficient`, `None` — but **only `prior_year_close` has ever been emitted.**
- **Nothing validates the field.** No enum, no assertion. And **a `None` does not mean a caller
  omitted it — it means the scorer died** (`position_signals.py:808`; `_det` stays None when the
  deferred import or the scoring itself raises).
- **`first_close_of_year` names the method, not the value.** On a 6-month fallback frame in August
  the anchor is ~mid-February, so the number is a six-month rolling return — and it flows
  **unflagged** into the score ladder, `beating_sp500`, group `avg_ytd` and the ranking sort.
  Annotated, not quarantined.

**Coverage is partial:** `signals.json` 67/67 rows carry it; `candidate_grades` **0/56**; universe
artifacts **none at all**; `/api/ticker` computes and drops it. **No page renders it.**

---

## 3 · The composite score

```
score = clamp(0, 100, 50 + rsi + macd + ma + ytd + vol)
```

Raw values span **−5 to +107**, so both clamps bind. Every component returns an integer.

| component | ladder | operator trap |
|---|---|---|
| **RSI** | `<30` +15 · `<40` +8 · `<60` +3 · `<70` −3 · `<80` −8 · else −15 | Strict `<`, so **each boundary falls to the lower-scoring band**: 30 → +8, 60 → −3, 70 → −8 |
| **MACD** | bullish_rising +13 · bullish +8 · bearish −8 · bearish_falling −13 | **0 is unreachable.** "Confirms" magnifies whichever side you are on. `macd == signal` counts as bearish |
| **MA** | (±4 above SMA20) + (±6 above SMA50) + (±4 SMA20 > SMA50) | Eight combinations, **six distinct values** {−14, −6, −2, +2, +6, +14}. 0 and ±10 unreachable; ±6 two-way ambiguous |
| **Volume** | `>1.5` +3 · `<0.7` −3 · else 0 | Asymmetric, so **both boundaries score 0** — the zero band is the closed interval [0.7, 1.5] |
| **YTD** | ≤−10 → −10 · (−10,0] → −4 · (0,5] → +2 · (5,20] → +6 · **(20,∞) → +12** | v1 was (50,100] +8 · (100,150] −2 · >150 −7. **Any example showing `ytd: −7` is a v1 score** |

**Degradation defaults are not neutral.** A missing RSI scores **+3**, a missing MACD **−8**,
missing moving averages **−14**. A data outage does not produce a neutral score — it produces a
bad one, silently.

**Two conventions, one number.** The score *bands* use inclusive `>=` (75 / 60 / 45 / 30) while
every point *ladder* uses strict comparisons that push boundaries downward. **And none of the
ladders live in config** — they are module constants and inline literals.

### Known internal conflicts

- **The MACD cliff.** `macd == signal` is bearish and 0 is unreachable, so a stock a basis point
  below the crossover scores **−8** and one a basis point above scores **+8**. A **16-point
  discontinuity** — 21% of the 75-point bar — at a threshold stocks cross constantly.
- **The MA component's ±6 ambiguity.** `+6` means *either* "above both MAs but SMA20 < SMA50"
  (recovering from a crash) *or* "below SMA20, above SMA50, SMA20 > SMA50" (a pullback inside an
  uptrend — **the setup the doctrine exists to buy**). `−6` is similarly two-way.
- **RSI fights MACD and MA by design.** RSI is mean-reverting; the other two are trend-following.
  Max trend contribution is +27 and RSI can claw back 15 of it — enough to fail a clean uptrend at
  the 75 bar when RSI crosses 80.
- **YTD now saturates.** Capping at +12 above 20% removed the penalty and the *discrimination*
  with it: a stock up 25% and one up 400% score identically on the momentum component.

**None of these has been ablated.** Build 5.1 tested the seven grade *rows*; it never touched the
five ladders inside the score.

---

## 4 · Six different things are called "signal"

| name | what it bands | values | rendered by |
|---|---|---|---|
| `signal` | the composite score, **only** | 5 · strong-buy, buy, hold, sell, strong-sell | every page |
| `trade_signal` | breaker + 8-block tally + the band | 7 · incl. `SIGNAL WITHHELD` | every page |
| `group_signal` | **the arithmetic MEAN** of selected members' scores | 4 · no strong-sell | index, history |
| `swing_signal` | a 5-block swing tally | 6 | **search only** — `/api/ticker`, absent from `signals.json` |
| `intraday_signal` | a 6-block intraday tally | 6 | **search only** |
| gauge `signal` | one macro input, three ways | 4 · incl. `unavailable` | framework, history |

---

## 5 · `signal` — the score band

`score_signal_band`, `signal_engine.py:1297`.

```python
if score >= 75: return "strong-buy"
if score >= 60: return "buy"
if score >= 45: return "hold"
if score >= 30: return "sell"
return "strong-sell"
```

Takes the score and nothing else — no breaker, no timing, no grade — and has **no withheld state**.

**A NaN score returns "strong-sell."** Every comparison against NaN is False, so it falls through
to the terminal return. **That is the only way this function reports a bearish extreme without a
bearish input** — and the score's own degradation defaults already bias it downward before this
point.

---

## 6 · `trade_signal` — the full ladder

`compute_trade_signal`, `signal_engine.py:1614`. A breaker block, then eight scoring blocks, then
`net = bullish − bearish`, then a sixteen-step first-match ladder.

### The eight blocks

| # | block | condition | side | amount |
|---|---|---|---|---|
| 1 | Breaker | `status == "degraded"` | — | 0 (reason only) |
| | | `status in (critical, warning)` | bearish | +3 |
| 2 | Trend structure | `above_ma50 and ma20 > ma50` | bullish | +2 |
| | | `above_ma50` only | bullish | +1 |
| | | else | bearish | +2 |
| | | `above_ma200` · else ma200 present | bull · bear | +1 · +1 |
| 3 | RSI | ≥75 · ≥65 · 40–60 · ≤35 | bear · bear · bull · bull | +2 · +1 · +1 · +2 |
| 4 | MACD | `macd > sig and hist > 0` | bullish | +2 |
| | | `macd > sig` only · no cross and `hist < 0` | bull · bear | +1 · +2 |
| 5 | Volume | `vol_ratio ≥ 1.5` · `≤ 0.7` | bull · bear | +1 · +1 |
| 6 | 52w high | > −3% · < −20% · −15% to −5% | bear · bear · bull | +1 · +1 · +1 |
| 7 | Trend strength | ≥ 16 · ≤ 5 | bull · bear | +1 · +1 |
| 8 | 1M return | > +15% · < −10% | bear · — | +1 · 0 (reason only) |

### The ladder — first match wins

| # | condition | result |
|---|---|---|
| 1 | `breaker == "degraded"` | **SIGNAL WITHHELD** |
| 2 | `breaker == "critical"` | **AVOID** |
| 3 | `breaker == "warning"` and `net ≤ 0` | **AVOID** |
| 4 | signal in (sell, strong-sell) and `net ≤ −2` | **AVOID** |
| 5 | signal in (sell, strong-sell), else | REDUCE/EXIT |
| 6 | hold and `net ≥ 2` | ACCUMULATE ON DIP |
| 7 | hold and `net ≤ −1` | REDUCE/EXIT |
| 8 | hold, else | HOLD POSITION |
| 9 | buy: `rsi ≥ 70` and `pct_from_high > −5` | WAIT FOR PULLBACK |
| 10 | buy: `rsi ≤ 60`, macd cross, above_ma50 | BUY NOW |
| 11 | buy: `rsi ≤ 65`, hist_increasing, above_ma50 | BUY NOW |
| 12 | buy: `vol_ratio ≥ 1.5` and macd cross | BUY NOW |
| 13 | buy: `net ≥ 3` | BUY NOW |
| 14 | buy: `rsi ≥ 65` | WAIT FOR PULLBACK |
| 15 | buy, else | ACCUMULATE ON DIP |
| 16 | terminal else | *unreachable* |

### AVOID has three unrelated causes

Two of the three are **group-level**, not about the ticker:

| cause | what it means |
|---|---|
| breaker **critical** | a macro condition on the ticker's group |
| breaker **warning** with net ≤ 0 | live example **AMAT** — and because breakers are group-level, **the whole group reads AVOID**, including buy-band names (TER 67, ASML 69) |
| band-driven, breaker **clear** | live example **AAPL** — score 40, signal sell, net ≤ −2. This one *is* about the ticker |

### Five confirmed defects

1. **`hist_increasing = macd_hist > 0`** — that is the *sign*, not the slope, yet the reason
   strings say "histogram expanding" and "histogram fading", which the test cannot tell apart.
   The swing function next door *does* compute a true three-bar turn.
2. **A warning breaker is double-counted** — it adds +3 bearish, then the same condition is
   re-tested against the already-penalised net. **A warning name with a raw tally of +3 nets 0 and
   is forced to AVOID.**
3. **The terminal `else` is unreachable** — the branches cover exactly the five band values.
4. **`score`, `ytd` and `rs_vs_ma50` are read and never used.** The composite is fetched and
   discarded; the ladder branches on the band instead.
5. **RSI in (35,40) or (60,65) matches nothing** — no credit to either side.

*Docstring drift:* the function's docstring lists six outputs. There are seven — it omits
`SIGNAL WITHHELD`, added later by D-019.

---

## 7 · `group_signal` — a different statistic over a different population

`signal_engine.py:2643`.

```python
avg_score = round(np.mean([s["score"] for s in stocks_in_group]), 1)
if   avg_score >= 70: "strong-buy"
elif avg_score >= 58: "buy"
elif avg_score >= 45: "hold"
else:                 "sell"
```

- **What it averages:** the arithmetic **mean** of member `score`, over the **selected** members
  only (≤ 7 per group), rounded to 1dp.
- **What it is not:** the universe composite —
  `0.50·median(YTD) + 0.30·median(3M) + 0.20·median(1M)`, computed over **all valid members**.
  Different statistic, different population.

**The same word means two different things.** The chain ends in a bare `else`, so everything below
45 collapses into "sell" — there is no fourth cut at 30. **"sell" at group level means score < 45;
"sell" at ticker level means 30 ≤ score < 45.**

---

## 8 · `swing_signal` — and what produces WAIT FOR DIP

`compute_swing_trade_signal`, `signal_engine.py:1800`. Returns a dict — signal, reasoning, entry,
stop, target, risk-reward. Guard: fewer than 10 bars → NO SETUP.

| block | condition | effect |
|---|---|---|
| MA20 | `price > ma20` · else | bull +1 · bear +1 |
| MA20 proximity | dist < 2% **and** above | bull +1 |
| RSI | ≤35 · ≤45 · ≥75 · ≥65 | bull +3 · +2 · bear +3 · +1 |
| **MACD** | **a true 3-bar turn** — hist today > yesterday **and** yesterday < the bar before | bull +2 / bear +2 |
| | else `macd > signal` · else | bull +1 · bear +1 |
| Volume | `vol_ratio ≥ 1.5` | bull +1 |

| condition | value |
|---|---|
| `net ≥ 4` and `rsi ≤ 45` | BUY SWING |
| `net ≥ 3` and hist turning up | BUY SWING |
| **`net ≥ 2` and `price > ma20`** | **WAIT FOR DIP** |
| `net ≤ −3` and `rsi ≥ 70` | FADE THE RALLY |
| `net ≤ −2` · `net ≥ 1` · `net ≤ −1` · else | EXIT · HOLD · EXIT · NO SETUP |

**WAIT FOR DIP has exactly one producer** — reached only after both BUY SWING tests fail, so in
practice net is 2–3 with either RSI above 45 or no histogram turn. It means *"bullish enough, but
already extended above the MA20."* It takes the same long entry/stop/target block as BUY SWING.

**Note the contrast with §6:** this function computes a genuine three-bar histogram turn. The
trade-signal function's `hist_increasing` only tests the sign. Two functions in the same file, one
right and one not.

---

## 9 · `intraday_signal` — and the forming-bar problem

`compute_intraday_trade_signal`, `signal_engine.py:1993`. Guard: fewer than 15 bars → NO SETUP.
Six blocks: gap, previous-day levels, RSI, MACD, volume, and a volatility check that is
**reason-only and never scored**.

| condition | value |
|---|---|
| **`vol_ratio ≤ 0.5`** — checked FIRST, before anything else | **RANGE BOUND** |
| `net ≥ 3` and `price > prev_high` | LONG ENTRY |
| `net ≥ 2`, macd bullish, `price > prev_close` | LONG ENTRY |
| `net ≥ 2` and `price ≤ prev_close` | WAIT FOR BREAKOUT |
| `net ≤ −3` and `price < prev_low` | SHORT ENTRY |
| `net ≤ −2`, not macd bullish, `price < prev_close` | SHORT ENTRY |
| `net ≤ −2` and `price ≥ prev_close` | BREAKDOWN WATCH |
| **`|net| ≤ 1` and yesterday's range < 0.8×ATR** | **RANGE BOUND** |
| `net ≥ 1` · `net ≤ −1` · else | WAIT · BREAKDOWN WATCH · NO SETUP |

**RANGE BOUND has two unrelated producers and the label does not distinguish them.** One is a
thin-volume veto checked before everything — it overrides even a clean breakout. The other is a
genuine coil.

**The forming bar is used unconditionally and is not gated.** `iloc[-1]` is labelled "today" with
no check that it *is* today. `_is_market_hours()` exists in the codebase but never gates this
function. **So outside market hours every label silently shifts by one session** — "today's" open,
high and low are the last completed session's, "prev" is the session before, and the gap shown is
yesterday's gap. The signal still renders with full confidence. During market hours `iloc[-1]` is
a genuine partial bar, so the high and low drift through the session.

*BREAKDOWN WATCH* was renamed from WAIT FOR PULLBACK (PER-509) because it collided with the
position ladder's long-side signal of the same name.

---

## 10 · The regime gauge's signal — and why `caution` does nothing

`framework/regime_calculator.py:90–135`. Three-way per gauge, plus `unavailable` on non-finite
input.

| gauge | risk_on | caution | risk_off | cuts live in |
|---|---|---|---|---|
| VIX (5d avg) | ≤ 18 | ≤ 22 | > 22 | config |
| HY (FRED OAS %) | ≤ 3.0 | ≤ 4.0 | > 4.0 | config |
| HY (pctile fallback) | ≥ 60 | ≥ 40 | < 40 | function default |
| Breadth (S5FI) | ≥ 60 | ≥ 50 | < 50 | config |
| **Breadth (RSP/SPY) — the live path** | > +0.5 | > −0.5 | ≤ −0.5 | **function default** |

**`caution` is a real threshold consumed by nothing.** `ladder_state` takes only `risk_on`,
`risk_off` and `unavailable` — caution is the residual, acting solely by not being risk_on. It is
computed, written to the artifact, and rendered.

**And under the live engine even that is decorative:** the chassis uses boolean stress tests, not
the three-way vote, and `regime_calculator.py:459` says so outright — *"the voter tally above is
still computed for display/back-compat but does NOT decide the state."*

**The config value for breadth is for a path that cannot run.** Config carries the S5FI thresholds
(60/50/30); the live path is RSP/SPY, whose ±0.5 band is a **Python default**. So the one quantity
here that *looks* configurable is not the one in use.

---

## 11 · Where the two surfaces diverge

| divergence | consequence |
|---|---|
| **The 20-bar boundary** | The engine tests `> 20`; the API tests `< 20`. **A ticker with exactly 20 bars passes one and fails the other.** |
| **Grade vs trade signal** | **Independent.** The only connection runs one way, through `signal`. And the shared score differs — **the grade uses the confirmed-close frame, the signal row uses the live frame** |
| **Multi-group tickers** | A ticker in two groups can render **AVOID and BUY NOW simultaneously** on the dashboard |
| **Row 6's cross-artifact join** | `breaker_by_group` is keyed from `signals.json`'s **15 selected groups**, but the lookup key comes from the universe artifact, which **deliberately maps names outside the 15**. A miss returns None → "group breaker unknown" → capped at B, never A+. Live on 5 of 14 watchers |
| **Display order ≠ selection order** | Display sorts `(score, ticker)`; selection sorts `(score, ytd, ticker)`. **And the dashboard shows LIVE scores against a FROZEN selection** — reading the live top-7 names a different set than was selected |
| **The empty meta segment** | `GROUP_METADATA` has 12 entries for 131 sub-industries, so **11 of 15 group cards** render an empty segment: "Sector · · 3/5…" |
| **Two clocks** | `signals.json`'s embedded timestamp and its HTTP `Last-Modified` disagree by ~95s, because `framework_runner` writes `candidate_grades` *back* into it. **Last-Modified is a write clock; `generated_at` is a bake clock** |
| **No shared run ID** | `candidate_signals_timestamp` exists, guards the write-back, then is **discarded**. A consumer cannot *detect* a two-artifact mismatch — only infer one from clock skew, which the row above shows is misleading. One-line fix |
| **Chip mappers** | Three of them, and they disagreed. Fixed 2026-08-11 |

---

## 12 · Where the thresholds live

| location | count |
|---|---|
| **inline literals** | **52** |
| JS literals | 13 |
| module constants | 7 |
| function defaults | 2 |
| **config keys** | **1** |

…and the single config key gates **neither** of the two values you would most want to change. The
band ladder is hand-copied into JS **twice**. The qualifier gate of 50 exists in **five** separate
places.

**The seven A+ row thresholds ARE in config** (`config.yaml:200–217`) — extension 1.8, RSI 45–70,
score 75, runway 15. It is the *score ladders* that are not.

---

## What this page does not cover, and what to distrust

**Stale prose exists elsewhere and contradicts this.** `architecture.html:414` still describes the
removed YTD over-extension penalties. The A+ doctrine brief carries the pre-amendment grade rule
with no superseded banner. The `grade_setup` docstring omits row 7 from its B list and omits the
row-3 availability carve-out. **Where those disagree with this page, they are the ones that are
wrong** — but the D-011 decision record itself is correct.

**D-020a is live on the score and grade paths but not yet on the universe path.** As of
2026-08-14, `signals.json` is stamped `scorer_version: "score_stock_v2"` while carrying near-miss
rows with YTD points v2 cannot emit, because those rows come from a universe artifact built before
the cutover and the rotation cron is weekly. **The artifact stamp does not distinguish them.**

**Every number here is a mechanism, not an endorsement.** Whether the score or the grade predicts
anything is answered elsewhere, and the answer is that the grade does not rank forward returns.
The evidence for keeping the apparatus rests on the stop and the regime chassis, not on the chip.
