#!/usr/bin/env python3
"""
Fear & Greed Index Engine — Computes a CNN-style Fear & Greed Index
using 7 equally-weighted market indicators sourced from yfinance.

Score: 0 (Extreme Fear) to 100 (Extreme Greed)
"""

import time
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
    score = _clamp((pct_above + 8) / 16 * 100)
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
    score = _clamp((combined + 20) / 40 * 100)

    direction = "falling" if combined > 0 else "rising"
    return {
        "score": round(score),
        "value": round(vix, 2),
        "detail": f"VIX at {vix:.2f}, {direction} trend (5d: {change_5d:+.1f}%, 20d: {change_20d:+.1f}%)",
        "label": _label(score),
    }


# ------------------------------------------------------------------
# 5. Market Volatility — VIX vs its 50-day moving average
# ------------------------------------------------------------------
def compute_market_volatility():
    """VIX relative to its 50-day moving average. Low VIX = greed, high = fear."""
    df = fetch_data("^VIX", period="6mo")
    if df is None or len(df) < 55:
        return {"score": 50, "value": 0, "detail": "Insufficient data", "label": "Neutral"}

    vix = float(df["Close"].iloc[-1])
    ma50 = float(df["Close"].rolling(50).mean().iloc[-1])
    pct_diff = ((vix - ma50) / ma50) * 100

    # Map: VIX 30% above MA → 0 (extreme fear), VIX 30% below MA → 100 (extreme greed)
    score = _clamp((30 - pct_diff) / 60 * 100)

    return {
        "score": round(score),
        "value": round(vix, 2),
        "detail": f"VIX {vix:.1f} vs 50-day MA {ma50:.1f} ({pct_diff:+.1f}% {'above' if pct_diff > 0 else 'below'})",
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
    score = _clamp((spread + 6) / 12 * 100)

    return {
        "score": round(score),
        "value": round(spread, 2),
        "detail": f"SPY 20d return {spy_ret:+.1f}% vs TLT {tlt_ret:+.1f}% (spread: {spread:+.1f}%)",
        "label": _label(score),
    }


# ------------------------------------------------------------------
# 7. Junk Bond Demand — HYG vs TLT relative performance
# ------------------------------------------------------------------
def compute_junk_bond_demand():
    """
    When junk bonds (HYG) outperform treasuries (TLT), investors are
    reaching for yield = greed. When treasuries outperform = fear.
    """
    hyg_df = fetch_data("HYG", period="6mo")
    tlt_df = fetch_data("TLT", period="6mo")

    if hyg_df is None or tlt_df is None or len(hyg_df) < 25 or len(tlt_df) < 25:
        return {"score": 50, "value": 0, "detail": "Insufficient data", "label": "Neutral"}

    hyg_ret = (float(hyg_df["Close"].iloc[-1]) - float(hyg_df["Close"].iloc[-21])) / float(hyg_df["Close"].iloc[-21]) * 100
    tlt_ret = (float(tlt_df["Close"].iloc[-1]) - float(tlt_df["Close"].iloc[-21])) / float(tlt_df["Close"].iloc[-21]) * 100

    spread = hyg_ret - tlt_ret  # Positive = junk outperforming (greed)

    # Map: -4% → 0, +4% → 100
    score = _clamp((spread + 4) / 8 * 100)

    return {
        "score": round(score),
        "value": round(spread, 2),
        "detail": f"HYG 20d return {hyg_ret:+.1f}% vs TLT {tlt_ret:+.1f}% (spread: {spread:+.1f}%)",
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
        ("Put/Call Proxy", compute_put_call_proxy),
        ("Market Volatility", compute_market_volatility),
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
