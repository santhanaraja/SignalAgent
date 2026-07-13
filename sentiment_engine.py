#!/usr/bin/env python3
"""
Behavioral Sentiment Engine (D-013 rebuild).

Per-ticker sentiment computed from what traders DO — price and volume
behaviour — not what they say. Replaces the retired StockTwits->VADER
social-chatter engine (obituary in docs/sentiment.md: source shutdown +
VADER-on-headlines mismatch). Display-only: D-005 rules sentiment is not a
gauge voter; this is analysis, nothing trades on it.

Three layers (D-013 scope A + C + D; Options-Implied B is a follow-up commit):
  A. Technical Sentiment — a 0-100 behavioural score from five factors, bucketed
     Bullish / Bearish / Trending (labels kept per D-013 Q2, now computed).
     technical_sentiment() is a PURE function (Lab law 1, D-010) shared by the
     live path and POST /api/sentiment/simulate — a future Sentiment Lab twists
     the factors through the same code.
  C. Relative Strength — ticker 20d vs SPY and vs its GICS group; group vs market.
  D. News strip — yfinance headlines + earnings chip as supporting context.
"""
import os
import json
import time

import numpy as np

from signal_engine import (
    fetch_data,
    compute_rsi,
    compute_macd,
    compute_moving_averages,
    compute_momentum_metrics,
)

CACHE_TTL = 900  # 15 minutes

_BASE_DIR = os.path.dirname(os.path.abspath(__file__))
UNIVERSE_ACTIVE_PATH = os.path.join(_BASE_DIR, "data", "universe_active.json")

_symbol_cache = {}          # {symbol: {"data": ..., "ts": ...}}
_spy_cache = {"ret": None, "ts": 0}

# --- Buckets (D-013 Q2: KEEP Bullish / Bearish / Trending, now computed from
# the Technical Sentiment score rather than scraped chatter). Trending is the
# neutral middle band. ---
BULLISH_MIN = 60
BEARISH_MAX = 40

# The five behavioural factors, in composite order.
FACTORS = ["range_position", "volume_trend", "momentum_posture",
           "sma_structure", "return_percentile"]

FACTOR_LABELS = {
    "range_position": "52w range position",
    "volume_trend": "Volume trend (accum/dist)",
    "momentum_posture": "RSI + MACD posture",
    "sma_structure": "SMA structure",
    "return_percentile": "20d return vs self",
}


def _clamp(v, lo=0.0, hi=100.0):
    return max(lo, min(hi, v))


def _bucket(score):
    if score >= BULLISH_MIN:
        return "Bullish"
    if score <= BEARISH_MAX:
        return "Bearish"
    return "Trending"


# ------------------------------------------------------------------
# The five factors — pure 0-100 maps (unit-pinnable, no network).
# ------------------------------------------------------------------
def _range_position_score(price, low52, high52):
    """Where price sits in its 52-week range: at low -> 0, at high -> 100."""
    if high52 <= low52:
        return 50.0
    return _clamp((price - low52) / (high52 - low52) * 100)


def _volume_trend_score(up_volume, down_volume):
    """Accumulation vs distribution: share of recent volume on up days.
    All volume on up days -> 100 (accumulation); all on down days -> 0."""
    total = up_volume + down_volume
    if total <= 0:
        return 50.0
    return _clamp(up_volume / total * 100)


def _momentum_posture_score(rsi, macd_above_signal, hist_rising):
    """RSI is the 0-100 base posture; MACD tilts it up to +/-10
    (bullish-and-rising most positive, bearish-and-falling most negative)."""
    base = _clamp(rsi)
    if macd_above_signal:
        adj = 10 if hist_rising else 5
    else:
        adj = -5 if hist_rising else -10
    return _clamp(base + adj)


def _sma_structure_score(price, sma20, sma50):
    """Count of the three bullish MA conditions (price>SMA20, price>SMA50,
    SMA20>SMA50) -> 0 / 33 / 67 / 100."""
    trues = sum((price > sma20, price > sma50, sma20 > sma50))
    return trues / 3 * 100


def _return_percentile_score(ret_20d, history_20d):
    """Percentile of the current 20-day return within the ticker's own recent
    history of 20-day returns — is it hot or cold *for itself*."""
    if not history_20d:
        return 50.0
    below = sum(1 for r in history_20d if r <= ret_20d)
    return _clamp(below / len(history_20d) * 100)


# ------------------------------------------------------------------
# Pure aggregator — Lab law 1: the live path AND /api/sentiment/simulate
# both call THIS. The page owns zero math.
# ------------------------------------------------------------------
def technical_sentiment(components):
    """components: {factor: 0-100} for each of FACTORS. Returns the composite
    (equal-weight mean, like F&G at ticker granularity), its bucket, and the
    normalised per-factor scores. Raises KeyError/ValueError on a bad factor."""
    vals = []
    for f in FACTORS:
        v = components[f]
        if isinstance(v, bool) or not isinstance(v, (int, float)):
            raise ValueError(f"{f} must be a number in [0, 100]")
        if not (0 <= v <= 100):
            raise ValueError(f"{f} out of range [0, 100]")
        vals.append(float(v))
    score = round(sum(vals) / len(vals))
    return {
        "score": score,
        "bucket": _bucket(score),
        "components": {f: round(float(components[f]), 1) for f in FACTORS},
    }


def relative_strength(ticker_20d, spy_20d, group_20d=None, group_name=None):
    """Pure relative-strength math from three 20-day returns. group_20d None
    (ticker not in the active universe) degrades the group rows gracefully."""
    rows = {
        "ticker_20d": round(ticker_20d, 2),
        "spy_20d": round(spy_20d, 2),
        "vs_spy": round(ticker_20d - spy_20d, 2),          # alpha
        "group_name": group_name,
        "group_20d": round(group_20d, 2) if group_20d is not None else None,
        "vs_group": round(ticker_20d - group_20d, 2) if group_20d is not None else None,
        "group_vs_market": round(group_20d - spy_20d, 2) if group_20d is not None else None,
    }
    return rows


# ------------------------------------------------------------------
# Real computation from a ticker's OHLCV.
# ------------------------------------------------------------------
def compute_factors(df):
    """Compute the five 0-100 behavioural factors from a ticker's OHLCV frame."""
    close = df["Close"]
    price = float(close.iloc[-1])

    mom = compute_momentum_metrics(df)
    # NaN MAs (young listing, <20/<50 rows) fall back to price — matching
    # score_stock's convention. Note: this biases sma_structure downward for
    # 30-49-row names (price>SMA50 reads False), same short-history artifact the
    # rest of the system carries; the universe's 90-day gate keeps most out.
    ma20s, ma50s, _ = compute_moving_averages(close)
    ma20 = float(ma20s.iloc[-1]) if not np.isnan(ma20s.iloc[-1]) else price
    ma50 = float(ma50s.iloc[-1]) if not np.isnan(ma50s.iloc[-1]) else price

    rsi_series = compute_rsi(close)
    rsi = float(rsi_series.iloc[-1]) if not np.isnan(rsi_series.iloc[-1]) else 50.0
    macd_line, macd_sig, macd_hist = compute_macd(close)
    macd_above = float(macd_line.iloc[-1]) > float(macd_sig.iloc[-1])
    hist_rising = len(macd_hist) > 1 and float(macd_hist.iloc[-1]) > float(macd_hist.iloc[-2])

    # up/down volume over the last 20 sessions (accumulation vs distribution)
    window = df.iloc[-21:]
    chg = window["Close"].diff().iloc[1:]
    vol = window["Volume"].iloc[1:]
    up_vol = float(vol[chg > 0].sum())
    down_vol = float(vol[chg < 0].sum())

    # 20-day return + its distribution over the trailing year (percentile-vs-self)
    r20 = close.pct_change(20).dropna() * 100
    ret_20d = float(r20.iloc[-1]) if len(r20) else 0.0
    history_20d = [float(x) for x in r20.iloc[-252:]] if len(r20) else []

    return {
        "range_position": _range_position_score(price, mom["low_52w"], mom["high_52w"]),
        "volume_trend": _volume_trend_score(up_vol, down_vol),
        "momentum_posture": _momentum_posture_score(rsi, macd_above, hist_rising),
        "sma_structure": _sma_structure_score(price, ma20, ma50),
        "return_percentile": _return_percentile_score(ret_20d, history_20d),
    }


def _group_for_symbol(symbol, path=None):
    """(group_name, group_median_1m) for a symbol from the committed universe
    ranking, or (None, None) if it isn't an active-universe name. The group
    median is the weekly-rotation value (universe_active.json is rebuilt
    Saturdays) — honest caveat surfaced on the page."""
    path = path or UNIVERSE_ACTIVE_PATH
    try:
        with open(path) as f:
            uni = json.load(f)
    except (OSError, ValueError):
        return None, None
    for g in (uni.get("ranking") or []):
        for t in (g.get("tickers") or []):
            if t.get("ticker") == symbol:
                return g.get("name"), g.get("median_1m")
    return None, None


def _spy_20d():
    """SPY 20-day (1-month) return, cached 15 min. None if unavailable."""
    now = time.time()
    if _spy_cache["ret"] is not None and (now - _spy_cache["ts"]) < CACHE_TTL:
        return _spy_cache["ret"]
    df = fetch_data("SPY", period="6mo")
    if df is None or len(df) < 22:
        return None
    ret = compute_momentum_metrics(df)["return_1m"]
    _spy_cache.update(ret=ret, ts=now)
    return ret


def fetch_news(symbol, limit=8):
    """yfinance news headlines for the strip — supporting context, never the
    feature. Graceful: any failure (fetch OR a malformed item) yields [] or
    skips the item, so a feed hiccup omits the strip, never crashes the page."""
    try:
        import yfinance as yf
        news = yf.Ticker(symbol).news
    except Exception:
        return []
    if not isinstance(news, list):
        return []
    items = []
    for it in news[:limit]:
        # yfinance news items are dicts, but a schema change / bad item must
        # not sink the whole page — skip anything unparseable.
        try:
            if not isinstance(it, dict):
                continue
            content = it.get("content") if isinstance(it.get("content"), dict) else it
            title = (content.get("title") or it.get("title") or "").strip()
            if not title:
                continue
            provider = content.get("provider")
            publisher = provider.get("displayName") if isinstance(provider, dict) else it.get("publisher", "")
            canon = content.get("canonicalUrl")
            link = canon.get("url") if isinstance(canon, dict) else it.get("link", "")
            items.append({"title": title, "publisher": publisher or "", "link": link or ""})
        except Exception:
            continue
    return items


def _earnings_chip(symbol):
    """Next-earnings chip for the news strip (best-effort, cached via
    earnings_calendar). None on any failure — graceful."""
    try:
        from earnings_calendar import get_earnings_map, days_to_earnings
        date_iso = get_earnings_map([symbol]).get(symbol)
        if not date_iso:
            return None
        return {"date": date_iso, "days": days_to_earnings(date_iso)}
    except Exception:
        return None


def get_symbol_sentiment(symbol):
    """Full per-ticker behavioural sentiment payload for the page: Technical
    Sentiment (A), Relative Strength (C), and a news strip (D). Cached 15 min."""
    symbol = symbol.upper().strip()
    now = time.time()
    cached = _symbol_cache.get(symbol)
    if cached and (now - cached["ts"]) < CACHE_TTL:
        return cached["data"]

    df = fetch_data(symbol, period="1y")
    if df is None or len(df) < 30:
        return {"symbol": symbol, "available": False,
                "error": "Insufficient price history for a technical read."}

    factors = compute_factors(df)
    tech = technical_sentiment(factors)

    ticker_20d = compute_momentum_metrics(df)["return_1m"]
    spy_20d = _spy_20d()
    group_name, group_20d = _group_for_symbol(symbol)
    rel = relative_strength(ticker_20d, spy_20d if spy_20d is not None else 0.0,
                            group_20d, group_name) if spy_20d is not None else None

    result = {
        "symbol": symbol,
        "available": True,
        "score": tech["score"],
        "bucket": tech["bucket"],
        "factors": tech["components"],
        "factor_labels": FACTOR_LABELS,
        "relative_strength": rel,
        "news": fetch_news(symbol),
        "earnings": _earnings_chip(symbol),
        "price": round(float(df["Close"].iloc[-1]), 2),
    }
    _symbol_cache[symbol] = {"data": result, "ts": now}
    return result
