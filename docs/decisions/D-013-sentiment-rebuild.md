# D-013 — Sentiment page rebuilt as per-ticker behavioral analysis

| | |
|---|---|
| **ID** | D-013 |
| **Date** | 2026-07-13 |
| **Status** | Ruled (revised — supersedes an earlier "downgrade to News" recommendation) |

## Context

The Sentiment page measured social chatter (StockTwits messages → VADER). The
data source shut down and the method never fit financial text, so the page
reported "Trending" for nearly everything (obituary in
[docs/sentiment.md](../sentiment.md)). The first recommendation was to downgrade
it to a News page; the user rejected that — sentiment doesn't require social
feeds. The **behavioral** sentiment that matters for swing trading is encoded in
price, volume, and options, all already fetched. Deliberation brief:
[sentiment-rebuild-brief.md](../briefs/sentiment-rebuild-brief.md). Display-only:
[D-005](D-005-sentiment-not-a-voter.md).

## Options considered

| Option | Summary | Why (not) |
|---|---|---|
| **A — Technical Sentiment** | Per-ticker 0-100 from OHLCV behaviour | **This build.** F&G methodology at ticker granularity; always-available |
| **C — Relative Strength** | Ticker 20d vs group and vs market | **This build.** Cheap; feeds the post-theme-retirement group-health question |
| **D — News strip** | yfinance headlines + earnings chips | **This build**, as *supporting* context — News demoted to where it belongs |
| B — Options-Implied | Put/Call, IV skew, IV rank | **Follow-up commit** — flakier fetch path deserves its own graceful-degrade testing |
| (rejected) Downgrade to News | Retire sentiment, keep headlines | Superseded — the page is rebuilt, not downgraded |

## Evidence

The dead engine's two failure modes (recorded in the obituary): **source
shutdown** (StockTwits public API retired → both fetchers empty) and **method
mismatch** (VADER's social-micro-text lexicon clusters near 0 on flat news
headlines → everything neutral → "Trending"). The ARWR-Jul-6 vs MRNA-Jul-12 knife
anatomy (same green pips, opposite momentum) motivated behaviour-over-chatter.

## Ruling + rationale

Rebuild the page as a **per-ticker behavioral analyzer** (not renamed — nav stays
"Sentiment"). Ship **A + C + D-strip this build; B as a follow-up commit**.
**Technical Sentiment** = 0-100 from five behavioral factors (52w range position,
volume accumulation/distribution, RSI+MACD posture, SMA structure,
20d-return-percentile-vs-self), equal-weight mean. **Buckets keep
Bullish / Bearish / Trending** (Q2 — retain the familiar labels), now *computed*
from the score, not scraped. VADER + the StockTwits client are **deleted** (not
commented; obituary carries the record). Lab pattern applies
([D-010](D-010-lab-pattern-laws.md)): `technical_sentiment()` is a pure function
behind `POST /api/sentiment/simulate` — no drift, and it seeds a future Sentiment
Lab.

## Consequences

- The page becomes a per-ticker read: Technical Sentiment (bucketed) + relative
  strength vs group/market + a news/earnings strip. Every number computed, nothing
  scraped.
- Display-only stays (D-005). The relative-strength **group median comes from the
  weekly universe artifact** — honestly labeled "group as of last rotation" (same
  weekly-freshness caveat as D-012's Market Internals).
- `/api/sentiment/trending`, VADER, the StockTwits client, and the `vaderSentiment`
  dependency are removed.
- Options-Implied (B) is deferred to its own commit with graceful-degrade testing.

## Revisit triggers

- Option B's put/call data, once flowing, becomes a candidate input to revisit
  [D-005](D-005-sentiment-not-a-voter.md) (sentiment-as-signal) — but stays
  display-only until a **ruled consumer** exists, same gate as F&G.
- The Technical Sentiment factor weighting (currently equal) proving mis-calibrated
  against trader judgment on a named row → a weighting record.

## Retest recipe

```
python3 test_sentiment.py     # factor pins, aggregator/buckets, simulate parity,
                              # relative-strength math, news/degradation matrix
```

## Links

- Jira: PER-508 comment 11718 (ruling); comment 11708 (D-005 origin)
- Brief: [docs/briefs/sentiment-rebuild-brief.md](../briefs/sentiment-rebuild-brief.md)
- Commits: `2bb3ccb` (rebuild), `39934a6` (Step 0 extraction / obituary)
- Docs: [docs/sentiment.md](../sentiment.md) (obituary + D-005 cross-ref)
