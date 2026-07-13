#!/usr/bin/env python3
"""
Fear & Greed Index Engine — Computes a CNN-style Fear & Greed Index
using 7 equally-weighted market indicators sourced from yfinance.

Score: 0 (Extreme Fear) to 100 (Extreme Greed)
"""

import time
import os
import json
import datetime
import numpy as np
from signal_engine import fetch_data

# ------------------------------------------------------------------
# Config
# ------------------------------------------------------------------
CACHE_TTL = 900  # 15 minutes

SECTOR_ETFS = [
    "XLK", "XLF", "XLV", "XLE", "XLI",
    "XLC", "XLY", "XLP", "XLU", "XLRE", "XLB",
]

_BASE_DIR = os.path.dirname(os.path.abspath(__file__))
# Committed universe artifact — source of the "Market Internals" breadth row
# (D-012); read-only, no new fetch. NOTE: rebuilt only on the weekly Saturday
# rotation, so this breadth steps weekly, not intraday (per-name above_50dma is
# the prior Friday close). A daily 530-name recompute would need a new fetch,
# which D-012 ruled out; the daily 54-name option was rejected (wrong population).
UNIVERSE_ACTIVE_PATH = os.path.join(_BASE_DIR, "data", "universe_active.json")
# Daily Fear & Greed series (D-012 Q2) — accrues the history D-005's
# gated overlay backtest needs.
FG_HISTORY_PATH = os.path.join(_BASE_DIR, "data", "fear_greed_history.json")

_fg_cache = {"data": None, "ts": 0}


def _clamp(val, lo=0, hi=100):
    return max(lo, min(hi, val))


def _label(score):
    if score <= 25:
        return "Extreme Fear"
    elif score <= 45:
        return "Fear"
    elif score <= 55:
        return "Neutral"
    elif score <= 75:
        return "Greed"
    else:
        return "Extreme Greed"


# ------------------------------------------------------------------
# Pure raw -> 0-100 score maps (one per component). Extracted so the
# curves are unit-pinnable without a network fetch — the raw endpoints
# are documented in docs/sentiment.md. Each compute_* fetches its raw
# reading, then calls the matching map here.
# ------------------------------------------------------------------
def _score_momentum(pct_above_sma):
    """S&P vs 125d SMA: -8% -> 0, +8% -> 100."""
    return _clamp((pct_above_sma + 8) / 16 * 100)


def _score_put_call(combined):
    """VIX momentum. `combined` is the sign-flipped 60/40 blend of VIX's 5d/20d
    % change, so combined -20 (VIX surging = fear) -> 0, +20 (VIX falling) -> 100."""
    return _clamp((combined + 20) / 40 * 100)


def _score_internals(pct_above_50dma):
    """% of universe above its 50DMA is already a 0-100 breadth reading."""
    return _clamp(pct_above_50dma)


def _score_safe_haven(spread):
    """SPY-TLT 20d return spread: -6% -> 0, +6% -> 100."""
    return _clamp((spread + 6) / 12 * 100)


def _score_junk(spread):
    """HYG-LQD 20d return spread: -4% -> 0, +4% -> 100."""
    return _clamp((spread + 4) / 8 * 100)


# ------------------------------------------------------------------
# 1. Market Momentum — S&P 500 vs 125-day SMA
# ------------------------------------------------------------------
def compute_market_momentum():
    """S&P 500 price relative to its 125-day moving average."""
    df = fetch_data("^GSPC", period="1y")
    if df is None or len(df) < 126:
        return {"score": 50, "value": 0, "detail": "Insufficient data", "label": "Neutral"}

    price = float(df["Close"].iloc[-1])
    sma125 = float(df["Close"].rolling(125).mean().iloc[-1])
    pct_above = ((price - sma125) / sma125) * 100

    # Map: -8% below → 0, +8% above → 100
    score = _score_momentum(pct_above)
    return {
        "score": round(score),
        "value": round(pct_above, 2),
        "detail": f"S&P 500 is {pct_above:+.2f}% {'above' if pct_above >= 0 else 'below'} its 125-day SMA (${sma125:,.0f})",
        "label": _label(score),
    }


# ------------------------------------------------------------------
# 2. Stock Price Strength — Sector ETFs near 52W highs vs lows
# ------------------------------------------------------------------
def compute_stock_strength():
    """Ratio of sector ETFs near 52-week highs vs those near 52-week lows."""
    near_high = 0
    near_low = 0
    total = 0

    for etf in SECTOR_ETFS:
        df = fetch_data(etf, period="1y")
        if df is None or len(df) < 50:
            continue
        total += 1
        price = float(df["Close"].iloc[-1])
        high_52 = float(df["High"].max())
        low_52 = float(df["Low"].min())
        range_52 = high_52 - low_52
        if range_52 == 0:
            continue

        position = (price - low_52) / range_52  # 0 = at low, 1 = at high
        if position >= 0.8:
            near_high += 1
        elif position <= 0.2:
            near_low += 1

    if total == 0:
        return {"score": 50, "value": 0, "detail": "No data", "label": "Neutral"}

    # Net ratio: all near highs = 100, all near lows = 0
    if near_high + near_low == 0:
        score = 50
    else:
        score = _clamp((near_high / (near_high + near_low)) * 100)

    # Also factor in overall position bias
    return {
        "score": round(score),
        "value": f"{near_high}H / {near_low}L",
        "detail": f"{near_high} of {total} sector ETFs near 52W highs, {near_low} near lows",
        "label": _label(score),
    }


# ------------------------------------------------------------------
# 3. Stock Price Breadth — Advancing vs declining sectors
# ------------------------------------------------------------------
def compute_stock_breadth():
    """20-day volume-weighted breadth across sector ETFs."""
    advancing = 0
    declining = 0

    for etf in SECTOR_ETFS:
        df = fetch_data(etf, period="6mo")
        if df is None or len(df) < 25:
            continue
        recent = df.tail(20)
        daily_returns = recent["Close"].pct_change().dropna()
        up_days = (daily_returns > 0).sum()
        down_days = (daily_returns < 0).sum()

        # Volume-weight: heavier volume on up days = more bullish
        vol = recent["Volume"].values[-20:]
        closes = recent["Close"].values[-20:]
        if len(closes) > 1:
            changes = np.diff(closes)
            up_vol = sum(float(vol[i + 1]) for i in range(len(changes)) if changes[i] > 0)
            dn_vol = sum(float(vol[i + 1]) for i in range(len(changes)) if changes[i] < 0)
            if up_vol > dn_vol:
                advancing += 1
            else:
                declining += 1

    total = advancing + declining
    if total == 0:
        return {"score": 50, "value": "0/0", "detail": "No data", "label": "Neutral"}

    ratio = advancing / total
    score = _clamp(ratio * 100)

    return {
        "score": round(score),
        "value": f"{advancing}A / {declining}D",
        "detail": f"{advancing} sectors with bullish volume breadth, {declining} bearish (20-day)",
        "label": _label(score),
    }


# ------------------------------------------------------------------
# 4. Put/Call Proxy — VIX momentum (slope of VIX vs its MA)
# ------------------------------------------------------------------
def compute_put_call_proxy():
    """
    Proxy for put/call ratio using VIX term structure behavior.
    Rising VIX relative to its recent trend = fear (more puts).
    Falling VIX trend = greed (more calls).
    """
    df = fetch_data("^VIX", period="6mo")
    if df is None or len(df) < 55:
        return {"score": 50, "value": 0, "detail": "Insufficient VIX data", "label": "Neutral"}

    vix = float(df["Close"].iloc[-1])
    vix_5d_ago = float(df["Close"].iloc[-6])
    vix_20d_ago = float(df["Close"].iloc[-21])

    # 5-day change and 20-day change
    change_5d = ((vix - vix_5d_ago) / vix_5d_ago) * 100
    change_20d = ((vix - vix_20d_ago) / vix_20d_ago) * 100

    # Combine: falling VIX = greed, rising = fear
    combined = -(change_5d * 0.6 + change_20d * 0.4)  # Negative because rising VIX = fear

    # Map: -20 (VIX surging) → 0, +20 (VIX crashing) → 100
    score = _score_put_call(combined)

    direction = "falling" if combined > 0 else "rising"
    return {
        "score": round(score),
        "value": round(vix, 2),
        "detail": f"VIX at {vix:.2f}, {direction} trend (5d: {change_5d:+.1f}%, 20d: {change_20d:+.1f}%)",
        "label": _label(score),
    }


# ------------------------------------------------------------------
# 5. Market Internals — % of the active universe above its 50-day MA
#    (D-012: replaces the 2nd VIX transform; genuine breadth, no new fetch)
# ------------------------------------------------------------------
# above_50dma derivation from the score `ma` component — TEMPORARY FALLBACK.
# The universe ranking stores each ticker's `ma` score-component points
# (score_stock: +4 price>MA20, +6 price>MA50, +4 MA20>MA50). The +6 term
# (price>MA50) is uniquely recoverable from the component TOTAL by set
# membership — all 8 sign combinations enumerated, and the sum determines the
# +6 sign unambiguously. Used only for tickers lacking the explicit
# above_50dma field (rows built before the field shipped).
# REMOVE this fallback once the 2026-07-18 rotation confirms above_50dma
# populates across the universe artifact.
_MA_ABOVE_50 = frozenset({14, 6, -2})    # ma totals whose price>MA50 (+6) term is set
_MA_BELOW_50 = frozenset({2, -6, -14})   # ma totals whose price<MA50 (-6) term is set


def _ticker_above_50dma(t):
    """One universe-ranking ticker -> True / False / None. Explicit field first
    (honest source), else derive from the `ma` component (temporary fallback)."""
    v = t.get("above_50dma")
    if isinstance(v, bool):
        return v
    ma = (t.get("components") or {}).get("ma")
    if ma in _MA_ABOVE_50:
        return True
    if ma in _MA_BELOW_50:
        return False
    return None


def compute_market_internals(path=None):
    """% of the active universe trading above its 50-day MA — genuine market
    breadth from the committed universe artifact (no new fetch; weekly-rotated,
    so it steps on Saturday's rotation). The reading is already 0-100 (50%
    above = neutral 50)."""
    path = path or UNIVERSE_ACTIVE_PATH
    try:
        with open(path) as f:
            uni = json.load(f)
    except (OSError, ValueError):
        return {"score": 50, "value": "n/a", "detail": "Universe breadth unavailable", "label": "Neutral"}

    states = [
        _ticker_above_50dma(t)
        for g in (uni.get("ranking") or [])
        for t in (g.get("tickers") or [])
    ]
    states = [s for s in states if s is not None]
    total = len(states)
    if total == 0:
        return {"score": 50, "value": "n/a", "detail": "No universe breadth data", "label": "Neutral"}

    above = sum(1 for s in states if s)
    pct = above / total * 100
    score = _score_internals(pct)
    return {
        "score": round(score),
        "value": f"{above}/{total} ({pct:.0f}%)",
        "detail": f"{above} of {total} universe names above their 50-day MA ({pct:.0f}%)",
        "label": _label(score),
    }


# ------------------------------------------------------------------
# 6. Safe Haven Demand — SPY vs TLT 20-day returns
# ------------------------------------------------------------------
def compute_safe_haven_demand():
    """
    Stocks vs bonds: when SPY outperforms TLT, investors are risk-on (greed).
    When TLT outperforms SPY, investors are fleeing to safety (fear).
    """
    spy_df = fetch_data("SPY", period="6mo")
    tlt_df = fetch_data("TLT", period="6mo")

    if spy_df is None or tlt_df is None or len(spy_df) < 25 or len(tlt_df) < 25:
        return {"score": 50, "value": 0, "detail": "Insufficient data", "label": "Neutral"}

    spy_ret = (float(spy_df["Close"].iloc[-1]) - float(spy_df["Close"].iloc[-21])) / float(spy_df["Close"].iloc[-21]) * 100
    tlt_ret = (float(tlt_df["Close"].iloc[-1]) - float(tlt_df["Close"].iloc[-21])) / float(tlt_df["Close"].iloc[-21]) * 100

    spread = spy_ret - tlt_ret  # Positive = stocks winning (greed)

    # Map: -6% (bonds crushing stocks) → 0, +6% (stocks crushing bonds) → 100
    score = _score_safe_haven(spread)

    return {
        "score": round(score),
        "value": round(spread, 2),
        "detail": f"SPY 20d return {spy_ret:+.1f}% vs TLT {tlt_ret:+.1f}% (spread: {spread:+.1f}%)",
        "label": _label(score),
    }


# ------------------------------------------------------------------
# 7. Junk Bond Demand — HYG (junk) vs LQD (investment-grade) 20-day
#    (D-012: the credit-QUALITY spread — CNN's actual definition. The prior
#    HYG-vs-TLT leg double-counted TLT with row 6's Safe Haven.)
# ------------------------------------------------------------------
def compute_junk_bond_demand():
    """
    When junk bonds (HYG) outperform investment-grade (LQD), investors are
    reaching for credit risk = greed. When quality outperforms = fear.
    """
    hyg_df = fetch_data("HYG", period="6mo")
    lqd_df = fetch_data("LQD", period="6mo")

    if hyg_df is None or lqd_df is None or len(hyg_df) < 25 or len(lqd_df) < 25:
        return {"score": 50, "value": 0, "detail": "Insufficient data", "label": "Neutral"}

    hyg_ret = (float(hyg_df["Close"].iloc[-1]) - float(hyg_df["Close"].iloc[-21])) / float(hyg_df["Close"].iloc[-21]) * 100
    lqd_ret = (float(lqd_df["Close"].iloc[-1]) - float(lqd_df["Close"].iloc[-21])) / float(lqd_df["Close"].iloc[-21]) * 100

    spread = hyg_ret - lqd_ret  # Positive = junk outperforming quality (greed)

    # Map: -4% → 0, +4% → 100. Band inherited from the prior HYG-TLT leg;
    # uncalibrated for the tighter junk-vs-IG spread (revisit candidate).
    score = _score_junk(spread)

    return {
        "score": round(score),
        "value": round(spread, 2),
        "detail": f"HYG 20d return {hyg_ret:+.1f}% vs LQD {lqd_ret:+.1f}% (spread: {spread:+.1f}%)",
        "label": _label(score),
    }


# ------------------------------------------------------------------
# Composite Index
# ------------------------------------------------------------------
def get_fear_greed_index():
    """
    Compute the composite Fear & Greed Index from 7 equally-weighted indicators.
    Cached for 15 minutes.
    """
    global _fg_cache
    now = time.time()

    if _fg_cache["data"] is not None and (now - _fg_cache["ts"]) < CACHE_TTL:
        return _fg_cache["data"]

    print("\nComputing Fear & Greed Index...")
    indicators = []

    computations = [
        ("Market Momentum", compute_market_momentum),
        ("Stock Price Strength", compute_stock_strength),
        ("Stock Price Breadth", compute_stock_breadth),
        ("Put/Call (VIX proxy)", compute_put_call_proxy),
        ("Market Internals", compute_market_internals),
        ("Safe Haven Demand", compute_safe_haven_demand),
        ("Junk Bond Demand", compute_junk_bond_demand),
    ]

    for name, fn in computations:
        print(f"  {name}...", end=" ")
        try:
            result = fn()
            result["name"] = name
            indicators.append(result)
            print(f"Score: {result['score']} ({result['label']})")
        except Exception as e:
            print(f"Error: {e}")
            indicators.append({
                "name": name, "score": 50, "value": "N/A",
                "detail": f"Error: {str(e)}", "label": "Neutral",
            })

    # Composite: equal weight average
    scores = [ind["score"] for ind in indicators]
    composite = round(sum(scores) / len(scores)) if scores else 50

    result = {
        "composite_score": composite,
        "composite_label": _label(composite),
        "indicators": indicators,
    }

    print(f"\n  Composite: {composite} — {_label(composite)}\n")

    _fg_cache = {"data": result, "ts": now}
    return result


# ------------------------------------------------------------------
# Daily persistence (D-012 Q2) — accrue the historical series that
# D-005's gated F&G-overlay backtest is blocked on.
# ------------------------------------------------------------------
def append_daily_history(path=None, reading=None, today=None):
    """Upsert today's reading into data/fear_greed_history.json.

    Once-per-day: each pipeline run refreshes today's entry (last write wins),
    so the final post-close run leaves the closing value; a new date appends.
    Rides the daily data commit like every other artifact. `reading` and
    `today` are injectable for tests.
    """
    path = path or FG_HISTORY_PATH
    reading = reading if reading is not None else get_fear_greed_index()
    today = today or datetime.date.today().isoformat()

    entry = {
        "date": today,
        "composite": reading["composite_score"],
        "label": reading["composite_label"],
        "components": [
            {"name": ind["name"], "score": ind["score"], "raw": ind.get("value")}
            for ind in reading["indicators"]
        ],
    }

    history = []
    if os.path.exists(path):
        try:
            with open(path) as f:
                loaded = json.load(f)
            if isinstance(loaded, list):
                history = loaded
        except (OSError, ValueError):
            history = []

    # Upsert by date, scanning the whole series (not just the tail), so a
    # non-monotonic history — a backfill, a replayed old commit, clock skew —
    # can't append a duplicate date. Last write for a given day wins (the
    # post-close run leaves the closing value).
    for i, h in enumerate(history):
        if isinstance(h, dict) and h.get("date") == today:
            history[i] = entry
            break
    else:
        history.append(entry)

    with open(path, "w") as f:
        json.dump(history, f, indent=2)
    return entry


if __name__ == "__main__":
    # Pipeline step (D-012): never exit non-zero — an F&G/network hiccup must
    # not block the data commit (paired with continue-on-error in the workflow).
    try:
        e = append_daily_history()
        print(f"Fear & Greed history: {e['date']} composite {e['composite']} ({e['label']})")
    except Exception as exc:
        print(f"[fear_greed] history persist skipped: {exc}")
