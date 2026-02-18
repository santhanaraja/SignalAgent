#!/usr/bin/env python3
"""
Sentiment Analysis Engine — Fetches social media data from StockTwits
public API and runs VADER sentiment scoring to classify tickers as
Bullish, Bearish, or Trending.
"""

import time
import requests
from vaderSentiment.vaderSentiment import SentimentIntensityAnalyzer

# ------------------------------------------------------------------
# Config
# ------------------------------------------------------------------
STOCKTWITS_BASE = "https://api.stocktwits.com/api/2"
CACHE_TTL = 900  # 15 minutes
REQUEST_TIMEOUT = 10

_analyzer = SentimentIntensityAnalyzer()

# In-memory caches
_trending_cache = {"data": None, "ts": 0}
_symbol_cache = {}  # { "AAPL": { "data": {...}, "ts": timestamp } }


# ------------------------------------------------------------------
# StockTwits Data Fetching
# ------------------------------------------------------------------
def fetch_trending_tickers():
    """
    Fetch currently trending ticker symbols from StockTwits.
    Uses public endpoint — no authentication required.
    Returns list of dicts: [{symbol, title, watchlist_count}, ...]
    """
    try:
        url = f"{STOCKTWITS_BASE}/trending/symbols.json"
        resp = requests.get(url, timeout=REQUEST_TIMEOUT, headers={
            "User-Agent": "SignalAgent/1.0"
        })
        if resp.status_code != 200:
            print(f"  StockTwits trending API returned {resp.status_code}")
            return []

        data = resp.json()
        symbols = data.get("symbols", [])
        result = []
        for s in symbols:
            result.append({
                "symbol": s.get("symbol", ""),
                "title": s.get("title", ""),
                "watchlist_count": s.get("watchlist_count", 0),
            })
        return result
    except Exception as e:
        print(f"  Error fetching trending tickers: {e}")
        return []


def fetch_stocktwits_messages(symbol, limit=30):
    """
    Fetch recent messages for a symbol from StockTwits.
    Public endpoint, no auth. Rate limited (200 req/hour for unauthenticated).
    Returns list of dicts: [{body, created_at, sentiment}, ...]
    """
    try:
        url = f"{STOCKTWITS_BASE}/streams/symbol/{symbol}.json"
        resp = requests.get(url, timeout=REQUEST_TIMEOUT, headers={
            "User-Agent": "SignalAgent/1.0"
        })
        if resp.status_code == 429:
            print(f"  Rate limited for {symbol}, skipping")
            return []
        if resp.status_code != 200:
            print(f"  StockTwits messages API returned {resp.status_code} for {symbol}")
            return []

        data = resp.json()
        messages = data.get("messages", [])
        result = []
        for msg in messages[:limit]:
            # StockTwits sometimes has user-tagged sentiment
            user_sentiment = None
            if msg.get("entities", {}).get("sentiment"):
                user_sentiment = msg["entities"]["sentiment"].get("basic")

            result.append({
                "body": msg.get("body", ""),
                "created_at": msg.get("created_at", ""),
                "user": msg.get("user", {}).get("username", ""),
                "user_sentiment": user_sentiment,  # "Bullish" / "Bearish" / None
                "likes": msg.get("likes", {}).get("total", 0),
            })
        return result
    except Exception as e:
        print(f"  Error fetching messages for {symbol}: {e}")
        return []


# ------------------------------------------------------------------
# VADER Sentiment Analysis
# ------------------------------------------------------------------
def analyze_sentiment(messages):
    """
    Run VADER sentiment analysis on a list of messages.
    Returns aggregated sentiment metrics for the ticker.
    """
    if not messages:
        return {
            "avg_score": 0,
            "bullish_pct": 0,
            "bearish_pct": 0,
            "neutral_pct": 100,
            "classification": "Neutral",
            "message_count": 0,
            "sample_bullish": [],
            "sample_bearish": [],
        }

    scores = []
    bullish_msgs = []
    bearish_msgs = []

    for msg in messages:
        body = msg.get("body", "")
        if not body:
            continue

        vs = _analyzer.polarity_scores(body)
        compound = vs["compound"]
        scores.append(compound)

        msg_with_score = {
            "body": body[:200],  # Truncate long messages
            "user": msg.get("user", ""),
            "score": round(compound, 3),
            "created_at": msg.get("created_at", ""),
            "likes": msg.get("likes", 0),
        }

        if compound >= 0.05:
            bullish_msgs.append(msg_with_score)
        elif compound <= -0.05:
            bearish_msgs.append(msg_with_score)

    if not scores:
        return {
            "avg_score": 0, "bullish_pct": 0, "bearish_pct": 0,
            "neutral_pct": 100, "classification": "Neutral",
            "message_count": 0, "sample_bullish": [], "sample_bearish": [],
        }

    total = len(scores)
    bullish_count = sum(1 for s in scores if s >= 0.05)
    bearish_count = sum(1 for s in scores if s <= -0.05)
    neutral_count = total - bullish_count - bearish_count

    avg_score = round(sum(scores) / total, 3)
    bullish_pct = round(bullish_count / total * 100, 1)
    bearish_pct = round(bearish_count / total * 100, 1)
    neutral_pct = round(neutral_count / total * 100, 1)

    # Classification
    if bullish_pct > 55 and avg_score > 0.1:
        classification = "Bullish"
    elif bearish_pct > 55 and avg_score < -0.1:
        classification = "Bearish"
    else:
        classification = "Trending"

    # Sort and pick top samples
    bullish_msgs.sort(key=lambda x: x["score"], reverse=True)
    bearish_msgs.sort(key=lambda x: x["score"])

    return {
        "avg_score": avg_score,
        "bullish_pct": bullish_pct,
        "bearish_pct": bearish_pct,
        "neutral_pct": neutral_pct,
        "classification": classification,
        "message_count": total,
        "sample_bullish": bullish_msgs[:3],
        "sample_bearish": bearish_msgs[:3],
    }


# ------------------------------------------------------------------
# Orchestration Functions
# ------------------------------------------------------------------
def get_trending_with_sentiment():
    """
    Fetch trending tickers and analyze sentiment for each.
    Results are cached for 15 minutes.
    Returns list of ticker sentiment dicts sorted by message activity.
    """
    global _trending_cache
    now = time.time()

    if _trending_cache["data"] is not None and (now - _trending_cache["ts"]) < CACHE_TTL:
        return _trending_cache["data"]

    print("Fetching trending tickers from StockTwits...")
    trending = fetch_trending_tickers()

    if not trending:
        # Return cached data if available, even if stale
        if _trending_cache["data"]:
            return _trending_cache["data"]
        return []

    results = []
    for i, ticker_info in enumerate(trending[:20]):  # Top 20 trending
        symbol = ticker_info["symbol"]
        print(f"  Analyzing sentiment for {symbol} ({i+1}/{min(len(trending),20)})...")

        messages = fetch_stocktwits_messages(symbol, limit=30)
        sentiment = analyze_sentiment(messages)

        results.append({
            "symbol": symbol,
            "title": ticker_info.get("title", ""),
            "watchlist_count": ticker_info.get("watchlist_count", 0),
            **sentiment,
        })

        # Small delay to avoid rate limiting
        if i < len(trending) - 1:
            time.sleep(0.3)

    # Sort by message count (most discussed first)
    results.sort(key=lambda x: x["message_count"], reverse=True)

    _trending_cache = {"data": results, "ts": now}
    return results


def get_symbol_sentiment(symbol):
    """
    Get sentiment analysis for a specific ticker symbol.
    Results cached for 15 minutes per symbol.
    """
    symbol = symbol.upper().strip()
    now = time.time()

    if symbol in _symbol_cache and (now - _symbol_cache[symbol]["ts"]) < CACHE_TTL:
        return _symbol_cache[symbol]["data"]

    print(f"Analyzing sentiment for {symbol}...")
    messages = fetch_stocktwits_messages(symbol, limit=30)
    sentiment = analyze_sentiment(messages)

    result = {
        "symbol": symbol,
        **sentiment,
    }

    _symbol_cache[symbol] = {"data": result, "ts": now}
    return result
