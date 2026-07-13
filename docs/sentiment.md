# Sentiment & Fear/Greed surfaces (pre-rebuild extraction)

The system's two **display-only** market-sentiment surfaces — the Fear & Greed
Index and the Sentiment page — and the last logic in the repo that had never
been extracted (everything else lives in [scoring.md](scoring.md) and
[regime.md](regime.md)). Neither trades: [D-005](decisions/D-005-sentiment-not-a-voter.md)
rules that sentiment is **not a gauge voter**.

This doc freezes what each surface computed *before* the 2026-07-13 rebuilds
([D-012](decisions/D-012-fear-greed-rebuild.md) Fear & Greed de-duplication +
persistence; [D-013](decisions/D-013-sentiment-rebuild.md) Sentiment rebuild), so
the changes have a baseline. Sources as extracted: `fear_greed_engine.py`
(`/api/fear-greed`, `feargreed.html`) and `sentiment_engine.py`
(`/api/sentiment/*`, `sentiment.html`).

---

# Part 1 — Fear & Greed Index (`fear_greed_engine.py`)

A CNN-style index. Seven components each map a raw market reading linearly onto
**0–100** (0 = Extreme Fear, 100 = Extreme Greed), clamped to `[0, 100]`. The
composite is the **equal-weight mean** of the seven — `composite =
round(sum(scores) / 7)`, no weights/multipliers/interaction terms (confirmed:
`get_fear_greed_index()`, the scores are averaged directly).

Served on demand via `/api/fear-greed`, computed at most once per **15 min**
(`CACHE_TTL = 900`, in-process cache); `feargreed.html` auto-refreshes every 15
min and renders a 0–100 needle gauge plus seven expandable rows, each showing its
score, band label, and `(Raw: value)`. **Not persisted** — each response is the
live compute or the ≤15-min cache; no history is written. (D-012 Q2 adds daily
persistence.)

## Composite bands (`_label()`)

| Composite | Label |
|---|---|
| ≤ 25 | Extreme Fear |
| 26 – 45 | Fear |
| 46 – 55 | Neutral |
| 56 – 75 | Greed |
| > 75 | Extreme Greed |

The same thresholds color the per-component labels and the gauge needle.

## Components

Each row's score is `clamp((raw − lo) / (hi − lo) × 100)` over the endpoints below
(linear; `lo` → 0 = max fear, `hi` → 100 = max greed).

| # | Component (page name) | Raw input | 0-endpoint (fear) | 100-endpoint (greed) |
|---|---|---|---|---|
| 1 | **Market Momentum** | S&P 500 (`^GSPC`) % above its 125-day SMA | −8% | +8% |
| 2 | **Stock Price Strength** | of 11 sector SPDRs, `near_high / (near_high + near_low)` where near-high = 52w-range position ≥ 0.8, near-low ≤ 0.2 | all near lows | all near highs |
| 3 | **Stock Price Breadth** | of 11 sector SPDRs, fraction with bullish 20-day *volume* breadth (up-day volume > down-day volume) | none bullish | all bullish |
| 4 | **Put/Call Proxy** | VIX momentum — a 60/40 blend of VIX's 5-day and 20-day % change, sign-flipped | VIX ≈ +20% (surging) | VIX ≈ −20% (falling) |
| 5 | **Market Volatility** | VIX vs its own 50-day MA (% difference) | +30% above MA | −30% below MA |
| 6 | **Safe Haven Demand** | SPY − TLT 20-day return spread | −6% | +6% |
| 7 | **Junk Bond Demand** | HYG − TLT 20-day return spread | −4% | +4% |

11 sector SPDRs = `XLK XLF XLV XLE XLI XLC XLY XLP XLU XLRE XLB`. Components with
no usable data return a neutral 50. Raw values are surfaced per row on the page.

## Motivating defects (why D-012)

**Seven rows collapse to five independent signals** — a single instrument is read
twice in two places:

- **VIX is double-counted.** Rows 4 (Put/Call Proxy) and 5 (Market Volatility) are
  both pure VIX transforms (recent momentum vs level-relative-to-MA). One VIX move
  swings two of seven votes in the same direction.
- **TLT is double-counted.** Rows 6 (Safe Haven, SPY vs TLT) and 7 (Junk Bond, HYG
  vs TLT) both anchor the long-Treasury leg, so TLT weakness reads as greed in
  both. Separately, CNN's junk-demand measure is junk **vs investment-grade** (HYG
  vs LQD), not junk vs Treasuries — the TLT leg here is the wrong comparator.

(Rows 2 and 3 also draw on the same 11-ETF sector basket, but read genuinely
orthogonal statistics from it — 52-week range *position* vs 20-day *volume-breadth
direction* — so they count as two independent signals and D-012 leaves them alone.
The VIX and TLT pairs are the fixable redundancy the swaps target.)

**D-012 fix** (targeted swaps, still seven rows): row 5 → **"Market Internals"** =
% of the ~530-name universe above its 50DMA (real breadth, from the committed
universe artifact — **no new fetch**; the artifact is weekly-rotated, so this row
steps on Saturday's rotation, not intraday — a deliberate data-integrity trade
against the 531/no-fetch/daily three-way); row 7 → **HYG vs LQD** 20-day spread (junk vs IG, CNN's actual
definition). Result: VIX and TLT each appear exactly once → seven rows, seven
independent inputs. Row 4 keeps its VIX-trend proxy under its honest label; all
raws stay displayed. D-012 also adds daily persistence (`{date, composite, 7
components + raws}` → `data/fear_greed_history.json`, once per day) — beginning the
credible historical series that D-005's F&G-extremes overlay variant is blocked on.

---

# Part 2 — Sentiment page (`sentiment_engine.py`) — obituary

**Status: RETIRED 2026-07-13 ([D-013](decisions/D-013-sentiment-rebuild.md)).** The
VADER scorer and the StockTwits client are **deleted** in the rebuild; this is the
record of what the page computed and why it died. The page is *rebuilt* (not
renamed) into a per-ticker Technical Sentiment surface — see D-013.

## What it computed

- **Sources.** Primary: the StockTwits public API (`api.stocktwits.com/api/2`) —
  `/trending/symbols.json` for the trending list and `/streams/symbol/{sym}.json`
  for up to 30 recent messages per ticker. Supplement: Yahoo Finance news headlines
  (`yfinance .news`, up to 20). Fallback when StockTwits returned no trending list:
  the dashboard's top-20 movers by `|YTD return|` (`signals.json`), scored on Yahoo
  news alone.
- **Scoring (VADER).** Each message body → `SentimentIntensityAnalyzer
  .polarity_scores()["compound"]` (−1…+1). A message is bullish if compound ≥
  **+0.05**, bearish if ≤ **−0.05**, else neutral. Per ticker: `avg_score` = mean
  compound, plus `bullish_pct / bearish_pct / neutral_pct`.
- **Buckets.** **Bullish** if `bullish_pct > 55` AND `avg_score > +0.1`; **Bearish**
  if `bearish_pct > 55` AND `avg_score < −0.1`; else **Trending** (the catch-all).
  Cached 15 min per symbol.

## Why it died

1. **Source shutdown.** StockTwits retired its public API; `/trending` and
   `/streams` now return non-200, so both StockTwits fetchers yield `[]`. The page
   silently collapsed to Yahoo-news-only.
2. **Method mismatch (VADER on headlines).** VADER's lexicon is tuned for social
   micro-text (slang, emoji, intensifiers, "!!!"). News headlines are flat and
   declarative and rarely carry those cue words, so `compound` clusters near 0 →
   almost everything scored neutral → almost every ticker printed **Trending**.
   With the social feed gone and the surviving source ill-suited to the method, the
   bucket lost all discriminating power.

Net: a dead source plus a method that never fit financial text = a page that
reported "Trending" for nearly everything.

## Cross-reference

[D-005](decisions/D-005-sentiment-not-a-voter.md) — sentiment does not vote
(state-dependent polarity at extremes + heavy component overlap with the gauge);
any life is display-only. That ruling is why the rebuild (D-013) stays display-only
too. The **Bullish / Bearish / Trending** labels survive the rebuild (user ruling,
D-013 Q2) but are recomputed from a per-ticker Technical Sentiment score rather than
scraped chatter.

---

## Links

- Jira: PER-508 comment 11718 (D-012 + D-013 rulings); comment 11708 (D-005 origin).
- Decisions: [D-005](decisions/D-005-sentiment-not-a-voter.md); D-012 and D-013
  registry records to follow (docs registry pass).
- Code as extracted (pre-rebuild): `fear_greed_engine.py`, `sentiment_engine.py`,
  `public/feargreed.html`, `public/sentiment.html`.
- Sibling extractions: [scoring.md](scoring.md), [regime.md](regime.md).
