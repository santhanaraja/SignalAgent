# Composite Score (0–100)

The SCORE column on the dashboard comes from `score_stock()` in
`signal_engine.py`. It is **purely additive from a base of 50** across five
technical components, then hard-clamped to [0, 100]. There are no weights,
multipliers, or interaction terms. The same function scores dashboard rows
(engine, 6-month frames) and universe-rotation qualifiers/near-misses
(builder, 1-year frames); indicator tails are nearly identical between the
two, so scores rarely differ by more than a point.

Per-component points are emitted in `score_components`
(`{rsi, macd, ma, ytd, vol}`) on every stock row and near-miss entry in
signals.json — that is what the score-cell hover tooltip renders.

## Components

| Component | Condition | Points |
|---|---|---|
| **Base** | always | **50** |
| **RSI** (Wilder-14) | <30 / 30–40 / 40–60 / 60–70 / 70–80 / ≥80 | **+15 / +8 / +3 / −3 / −8 / −15** |
| **MACD** (12/26/9) | line > signal (+8), and histogram rising (+5 more) | **+8 or +13** |
| | line ≤ signal (−8), and histogram falling (−5 more) | **−8 or −13** |
| **MA structure** | price>MA20 ±4 · price>MA50 ±6 · MA20>MA50 ±4 | **−14 … +14** |
| **YTD bucket** | >50 / >20 / >5 / >0 / >−10 / ≤−10 | **+8 / +12 / +6 / +2 / −4 / −10** |
| **YTD overextension** | >150%: −15 · >100%: −10 (checked in that order) | **−15 or −10** |
| **Volume** (5d avg ÷ 20d avg) | >1.5 / <0.7 / else | **+3 / −3 / 0** |

Signal bands: ≥75 strong-buy · ≥60 buy · ≥45 hold · ≥30 sell · <30
strong-sell. The weekly universe rotation additionally requires score ≥ 50
(`rotation.min_composite_score`) for a ticker to qualify.

**Not used by the score** (displayed only): 52-week-high distance, 1M/3M
returns, RS vs MA50, trend strength, and all fundamentals.

## Design notes & known quirks

- **The YTD buckets taper deliberately**: +20–50% earns +12 while >+50%
  earns only +8, and >100% / >150% eat an extra −10 / −15. Extended
  momentum is progressively de-rated.
- **Fixed 2026-07-03**: the >150% branch was dead code (unreachable below
  the >100% check). Reordered so it fires; mega-winners (+150%+) now take
  −15 instead of −10, lowering their score by 5 points vs. prior behavior.
- **Short-history bias (kept as-is, documented)**: when MA20/MA50 are NaN
  (fewer than 20/50 bars), they default to the current price, so all three
  MA comparisons fail and a young listing takes the full −14. The
  universe's 90-day history gate keeps such names out in practice.
- **YTD baseline** uses the year's first *valid* close (NaN phantom bars
  from Yahoo are skipped — see `compute_ytd_return`).

## Worked example — MU (2026-07-03, post-semi-crash)

| Step | Points | Running | Why |
|---|---|---|---|
| Base | +50 | 50 | |
| RSI | +3 | 53 | RSI 48.5, neutral zone |
| MACD | −13 | 40 | line 64.5 below signal 85.3, histogram falling — the crash component |
| MA structure | +6 | 46 | below MA20 (−4), above MA50 (+6), MA20>MA50 (+4) |
| YTD | −7 | 39 | +209% → top bucket +8, overextension −15 (>150% rule) |
| Volume | 0 | 39 | ratio 1.02 |
| **Final** | | **39** | sell band, 11 under the ≥50 qualifier gate |

(Before the >150% fix this scored 44 — the overextension penalty was
capped at −10 by the dead branch.)

---

# SIGNAL column (STRONG-BUY / BUY / HOLD / SELL / STRONG-SELL)

**Confirmed: purely the score→band mapping, no other inputs.**
`score_stock()` maps the clamped composite score and nothing downstream
mutates it:

| Band | Condition |
|---|---|
| strong-buy | score ≥ 75 |
| buy | 60–74 |
| hold | 45–59 |
| sell | 30–44 |
| strong-sell | < 30 |

Breakers, stage analysis, fundamentals, and group context play no role.
(The *group-level* badge `group_signal` is a different mapping on the
group's average score — ≥70 strong-buy / ≥58 buy / ≥45 hold / else sell —
don't confuse the two.)

---

# TRADE SIGNAL column (`compute_trade_signal`)

Two-phase design in `signal_engine.py`: an **evidence tally** builds
`net = bullish − bearish` from up to 8 factor groups, then a **decision
ladder** keyed on the breaker state and the SIGNAL band picks one of six
outputs: BUY NOW · WAIT FOR PULLBACK · ACCUMULATE ON DIP · HOLD POSITION ·
REDUCE/EXIT · AVOID.

Inputs: the stock's technicals (RSI, MACD trio, price/MA20/MA50/MA200,
volume ratio, 52W-high distance, trend strength, 1M return), the SIGNAL
band, and the **group's** breaker status. **Not inputs:** Weinstein stage,
fundamentals, YTD, RS-vs-MA50, and the raw score (the last three are read
into locals but never used — see quirks).

## Phase 1 — evidence tally

| Factor | Condition | Tally |
|---|---|---|
| Breaker | group status critical or warning | bear +3 |
| Trend | price>MA50 and MA20>MA50 | bull +2 |
| | price>MA50 only | bull +1 |
| | price<MA50 | bear +2 |
| MA200 | above / below (when MA200 exists) | bull +1 / bear +1 |
| RSI | ≥75 / 65–75 / 40–60 / ≤35 | bear +2 / bear +1 / bull +1 / bull +2 |
| MACD | line>signal and hist>0 / line>signal only / line<signal and hist<0 | bull +2 / bull +1 / bear +2 |
| Volume | ratio ≥1.5 / ≤0.7 | bull +1 / bear +1 |
| 52W high | within 3% / below −20% / in −15…−5% | bear +1 / bear +1 / bull +1 |
| Trend strength | ≥16/20 / ≤5/20 | bull +1 / bear +1 |
| 1M return | > +15% | bear +1 |

## Phase 2 — decision ladder (first match wins)

| # | Condition | Output |
|---|---|---|
| 1 | breaker critical | **AVOID** (unconditional — overrides any tally) |
| 2 | breaker warning **and** net ≤ 0 | **AVOID** (warning with net > 0 falls through) |
| 3 | band sell/strong-sell, net ≤ −2 | **AVOID** |
| 4 | band sell/strong-sell, net > −2 | **REDUCE/EXIT** |
| 5 | band hold, net ≥ 2 | **ACCUMULATE ON DIP** |
| 6 | band hold, net ≤ −1 | **REDUCE/EXIT** |
| 7 | band hold, net 0 or 1 | **HOLD POSITION** |
| 8 | band buy/strong-buy: RSI ≥ 70 **and** within 5% of 52W high | **WAIT FOR PULLBACK** |
| 9 | … RSI ≤ 60 and MACD line>signal and above MA50 | **BUY NOW** |
| 10 | … RSI ≤ 65 and hist>0 and above MA50 | **BUY NOW** |
| 11 | … volume ≥ 1.5× and MACD line>signal | **BUY NOW** (breakout) |
| 12 | … net ≥ 3 | **BUY NOW** |
| 13 | … RSI ≥ 65 | **WAIT FOR PULLBACK** |
| 14 | … (none of the above) | **ACCUMULATE ON DIP** |
| 15 | unknown band (unreachable) | HOLD POSITION |

**The "extension threshold" is a conjunction, not a single RSI cutoff**:
rule 8 needs RSI ≥ 70 *and* price within 5% of the 52W high. An extended
name that has already pulled back >5% escapes rule 8 and can still earn
BUY NOW via rules 9–12; RSI ≥ 65 alone (rule 13) only forces WAIT when
every BUY NOW path has failed.

## Worked examples (2026-07-03, replayed through the function — all match)

- **J** (score 76, strong-buy, net +3 bullish): **AVOID**. Rule 1 —
  Construction & Engineering's breaker is *critical* (group avg YTD
  negative), and a critical breaker overrides everything, including a
  strong-buy band with net-bullish evidence. The trade signal is the only
  column where the breaker bites.
- **DAL vs LUV** (both strong-buy, both Passenger Airlines, breaker clear):
  DAL (RSI 72.0, −3.1% from 52W high) trips rule 8's conjunction →
  **WAIT FOR PULLBACK**. LUV (RSI 65.9, −8.0% from high) escapes rule 8
  (66 < 70), fails rules 9–11 (RSI 65.9 > 65, vol 0.92), then wins rule 12
  with net +4 — its −8% sits in the healthy-pullback bull zone where DAL's
  −3.1% is a dead zone → **BUY NOW**. Same band, 6-RSI-point gap, opposite
  timing calls.
- **PEP** (score 66, buy band, Stage S4, below MA50): **ACCUMULATE ON
  DIP**. Stage is not an input; what actually decides it is price<MA50
  failing rules 9–10, vol 1.1 failing rule 11, and net +1 failing rule 12
  → falls to the rule-14 default. The S4 label and the trade signal
  agreeing is coincidence of inputs, not wiring.

## Quirks & dead code (documented, unfixed)

1. **`hist_increasing` is a misnomer** — it tests `histogram > 0`
  (positivity), not increase; the score's version compares against the
  prior bar. Since histogram ≡ line − signal, `hist > 0` ⟺ `line > signal`
  up to 3-decimal rounding, so the tally's "+1 MACD above signal but
  histogram fading" branch is *nearly dead* (needs rounded hist exactly
  0.000 with line microscopically above signal), and rule 10's hist test is
  effectively the same as rule 9's cross test.
2. **Unused inputs**: `ytd`, `rs_vs_ma50`, and `score` are read into
  locals and never referenced again.
3. **Tally dead zones**: RSI 35–40 and 60–65 contribute nothing; 52W-high
  −5…−3 and −20…−15 likewise (DAL's −3.1% landed in one).
4. **1M return asymmetry**: > +15% costs bear +1, but < −10% only appends
  a reasoning line — no tally effect.
5. **Reasoning truncation**: comment says "top 3" but code keeps 4; order
  is fixed code order (breaker first), not relevance-ranked.
6. **Warning ≠ critical**: a *warning* breaker with net-positive evidence
  falls through to the normal ladder — only *critical* is absolute.

---

# Search page (search.html → /api/ticker/&lt;SYM&gt;)

**Verdict on the central question: SHARED code, not duplicated.**
`ticker_api._analyze_ticker()` imports and calls the *same*
signal_engine functions the dashboard pipeline uses — `fetch_data`
(same `period="6mo"` daily frame), `score_stock`, `compute_trade_signal`,
`compute_swing_trade_signal`, `compute_intraday_trade_signal`,
`compute_stage_analysis`, `fetch_fundamentals_yfinance` — with a 5-minute
per-symbol cache and a search-history side effect. The page's JS is pure
display (formatting + the same color helpers as index.html). Formula
drift between the pages is therefore impossible; what CAN differ:

| Divergence | Effect |
|---|---|
| **`breaker_status="clear"` is hardcoded** (an ad-hoc symbol has no group context) | The Position signal ignores thesis breakers on the search page. Live example (2026-07-03): **J is AVOID on the dashboard (critical group breaker) but BUY NOW on the search page** — same function, one input. |
| Freshness | Search computes live (≤5 min cache); the dashboard reads the last CI engine run (~hourly). Same-day values can differ by intraday drift, not by formula. |
| `score_components` | Computed but **not included** in the search payload — the search page has no breakdown-tooltip data (candidate future fix). |

## 1. SCORE + SIGNAL — identical

Same `score_stock()`, same 6-month frame as `run_engine`. AMAT traced on
the search path (2026-07-03): base 50 · RSI +3 (54.0) · MACD +8 (line
51.48 > signal 50.79, but histogram not rising vs prior bar, so no +5) ·
MA +14 (above MA20 585, above MA50 490, aligned) · YTD −2 (+112.6% → +8
bucket, −10 overextension) · Vol 0 (1.37×) → **73 (buy)** — matching the
dashboard's stored 73 exactly.

## 2. Weinstein stage — shared, display-only on both pages

Same `compute_stage_analysis()` feeds the dashboard's S-badge and the
search page's stage panel. Nothing consumes it — not the score, not any
trade signal, not selection. It is a **voting system**, not a rule chain:
six factors add points to stages 1–4, argmax wins.

| Factor | Condition | Votes |
|---|---|---|
| Price vs MA150 | >+5% / 0…+5% / −3…0% / <−3% | S2+3 / S2+1,S1+1 / S3+2,S1+1 / S4+3 |
| MA150 slope (vs 20d ago) | >+0.5% / −0.2…+0.5% / <−0.2% | S2+3 / S1+2,S3+2 / S4+3 |
| MA50 slope (vs 20d ago) | >+1% / >0 / >−1% / ≤−1% | S2+2 / S2+1 / S3+1 / S4+2 |
| 30d range compression | range <8% and \|MA150 slope\| <0.5% | S1+2 |
| Volume | ≥1.5× above MA150 / ≥1.5× below MA150 | S2+1 / S4+1 |
| Trend strength | ≥16/20 / ≤5/20 | S2+1 / S4+1 |

Confidence = winning share of total votes: >50% high, >35% medium, else
low. Narrative strings are fixed per stage (Basing / Advancing / Topping
/ Declining + advice sentence).

**Headline quirk — "MA150" is actually MA100 in production**: both pages
run on 6-month frames (~124 bars < 150), so the code always takes its
fallback branch, `rolling(min(len, 100))`. The "≈30-week MA" label, the
`ma150` field, and its slope are really a **100-day MA** everywhere users
see them. Only a ≥150-bar frame (nothing currently) would use a true MA150.

## 3. Trade signals × 3

**Position** — the dashboard's `compute_trade_signal` exactly (see the
TRADE SIGNAL section above), except the hardcoded clear breaker noted in
the divergence table.

**Swing (2–10 day holds)** — evidence tally then ladder:

| Tally factor | Condition | Points |
|---|---|---|
| MA20 | above / below | bull +1 / bear +1 |
| MA20 proximity | above and within 2% | bull +1 |
| RSI | ≤35 / ≤45 / ≥75 / ≥65 (45–65: 0) | bull+3 / bull+2 / bear+3 / bear+1 |
| MACD histogram **turn** (recomputed, see quirks) | V-turn up / down; else line>signal / line≤signal | bull+2 / bear+2 / bull+1 / bear+1 |
| Volume | ≥1.5× | bull +1 |

| Ladder (first match) | Output |
|---|---|
| net ≥ 4 and RSI ≤ 45 | BUY SWING |
| net ≥ 3 and histogram turning up | BUY SWING |
| net ≥ 2 and above MA20 | WAIT FOR DIP |
| net ≤ −3 and RSI ≥ 70 | FADE THE RALLY |
| net ≤ −2 | EXIT SWING |
| net ≥ 1 | HOLD SWING |
| net = −1 | EXIT SWING |
| net = 0 | NO SETUP |

Long levels (BUY SWING / WAIT FOR DIP / HOLD SWING): entry =
min(MA20, 5-day swing low + 0.3·ATR14), snapped to price if >2% above it;
stop = 10-day swing low − 0.3·ATR; target = entry + 2×risk, capped at the
10-day swing high if closer (so displayed R/R can be <2). Short levels
(FADE/EXIT) mirror it around the swing highs. ATR14 = mean true range,
default 2% of price when <14 bars.

**Intraday (same-day)** — computed from **daily bars** (no intraday
data): "today" is simply the last bar, previous-day H/L/C are bar −2.

| Tally factor | Condition | Points |
|---|---|---|
| Gap (today open vs prev close) | >+0.5% / <−0.5% | bull+1 / bear+1 |
| Price vs prev day | >prev high / >prev close / <prev low / <prev close | bull+2 / bull+1 / bear+2 / bear+1 |
| RSI | ≥70 / ≤30 / 40–60 | bear+1 / bull+1 / bull+1 |
| MACD | bullish+hist>0 / bearish+hist<0 | bull+1 / bear+1 |
| Volume ≥1.5× | direction of price vs prev close | bull+1 or bear+1 |

| Ladder (first match) | Output |
|---|---|
| volume ≤ 0.5× | RANGE BOUND |
| net ≥ 3 and price > prev high | LONG ENTRY |
| net ≥ 2 and MACD bullish and price > prev close | LONG ENTRY |
| net ≥ 2 and price ≤ prev close | WAIT FOR BREAKOUT |
| net ≤ −3 and price < prev low | SHORT ENTRY |
| net ≤ −2 and MACD bearish and price < prev close | SHORT ENTRY |
| net ≤ −2 and price ≥ prev close | WAIT FOR PULLBACK |
| \|net\| ≤ 1 and prev range < 0.8·ATR | RANGE BOUND |
| net ≥ 1 / net ≤ −1 / net = 0 | WAIT FOR BREAKOUT / WAIT FOR PULLBACK / NO SETUP |

Levels: LONG entry = max(prev high, price), stop −1.5·ATR, target
+2.5·ATR (1:1.7); SHORT mirrors; WAIT FOR BREAKOUT arms a trigger at prev
high (stop −1.5·ATR, target +2·ATR, 1:1.3); **WAIT FOR PULLBACK here is a
short-side breakdown trigger at prev low** — a naming collision with the
Position ladder's WAIT FOR PULLBACK, which is long-side patience.

**Worked example — AMAT BUY NOW two days after the −7.35% crash bar**
(bars: Jul 1 −9.97%, Jul 2 −7.35%): band is buy (score 73), so the
Position ladder runs; rule 8 needs RSI ≥70 (RSI is 54 — the crash *reset*
RSI from extended to neutral); rule 9 = RSI ≤60 ✓ + MACD line still above
signal ✓ (51.48 vs 50.79 — two red days dented but didn't cross the slow
crossover) + price still 23% above MA50 ✓ → **BUY NOW**. The crash
changed the inputs in offsetting directions: it *removed* the
overextension objections (RSI, 52W-high proximity) faster than it damaged
the trend evidence. The swing timeframe is more cautious on the same data
(WAIT FOR DIP, entry at MA20 585.42, stop 554.05, 1:2.0) and the intraday
signal arms a breakout trigger at the pre-crash high (693.78) — the three
timeframes are intentionally independent reads.

## 4. Trend strength (N/20)

`compute_momentum_metrics`: the count of the last 20 sessions whose close
was above the same-day MA20 — an integer 0–20 (default 10 when <20
bars). ≥16 reads "strong sustained uptrend", ≤5 "no uptrend present".
Shared by both pages; feeds the Position tally (±1), the swing/stage
votes, and reasoning text.

## 5. Momentum + Fundamentals tabs

Pure display. Momentum fields all come from `compute_momentum_metrics`
(the same details dict); fundamentals are a raw
`fetch_fundamentals_yfinance` pull-through (yfinance `.info`). Nothing on
the search page feeds any of them back into scores or signals — market
cap's only systemic role remains the universe qualifier gate.

## 6. Quirks & inconsistencies (search stack)

1. **Breaker blindness** (by construction): the search page's Position
   signal can directly contradict the dashboard's for the same ticker at
   the same moment (J: AVOID vs BUY NOW). The page gives no hint that
   group breaker context was skipped.
2. **"MA150" is MA100** on every 6-month frame (see stage section) — the
   Weinstein implementation never sees a true 30-week MA in production.
3. **Swing recomputes MACD with different math**: it rebuilds the
   histogram via `ewm(span=…)` *default* (`adjust=True`) while
   `compute_macd` uses `adjust=False` — the swing turn-detection runs on
   a slightly different histogram than the one displayed. The `macd_hist`
   local it also reads goes unused.
4. **Intraday on daily bars**: after the close, "today" is the completed
   last bar, so gap/breakout language describes a finished session; on a
   crash day the armed trigger levels can sit 10–15% away from price
   (AMAT: entry 693.78 vs price 603).
5. **Intraday unused values**: `today_high`/`today_low` are computed and
   never referenced.
6. **Stage tie-break** favors the lower stage number (`max()` on a dict
   returns the first maximum in insertion order 1→4).
7. **Stage reads `rsi` and never uses it**; swing reads `pct_from_high`
   and `trend_strength` and never uses them.
8. **Search payload omits `score_components`** even though score_stock
   now emits it — the search page can't render the score tooltip until
   the endpoint passes it through.
9. Score *colors* on both pages use 70/55 thresholds while the signal
   *bands* break at 75/60/45/30 — a 73 renders green ("buy" band) and a
   57 renders yellow ("hold") — cosmetic-only, but the two scales are
   easy to conflate.
