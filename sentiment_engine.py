#!/usr/bin/env python3
"""
Sentiment Analysis Engine — Multi-source sentiment analysis using
StockTwits, Yahoo Finance news headlines, and VADER scoring.
Classifies tickers as Bullish, Bearish, or Trending.
"""

import time
import json
import os
import requests

try:
    from vaderSentiment.vaderSentiment import SentimentIntensityAnalyzer
    _analyzer = SentimentIntensityAnalyzer()
except ImportError:
    _analyzer = None
    print("[WARN] vaderSentiment not installed — sentiment scoring disabled")

try:
    import yfinance as yf
    USE_YFINANCE = True
except ImportError:
    USE_YFINANCE = False

# ------------------------------------------------------------------
# Config
# ------------------------------------------------------------------
STOCKTWITS_BASE = "https://api.stocktwits.com/api/2"
CACHE_TTL = 900  # 15 minutes
REQUEST_TIMEOUT = 10

# In-memory caches
_trending_cache = {"data": None, "ts": 0}
_symbol_cache = {}  # { "AAPL": { "data": {...}, "ts": timestamp } }

# Dashboard tickers for fallback trending list
DATA_DIR = os.path.join(os.path.dirname(os.path.abspath(__file__)), "data")

# Browser-like headers (StockTwits blocks simple User-Agents)
_HEADERS = {
    "User-Agent": "Mozilla/5.0 (Macintosh; Intel Mac OS X 10_15_7) AppleWebKit/537.36 (KHTML, like Gecko) Chrome/122.0.0.0 Safari/537.36",
    "Accept": "application/json",
    "Accept-Language": "en-US,en;q=0.9",
}


# ------------------------------------------------------------------
# StockTwits Data Fetching
# ------------------------------------------------------------------
def fetch_trending_tickers():
    """
    Fetch currently trending ticker symbols from StockTwits.
    Returns list of dicts: [{symbol, title, watchlist_count}, ...]
    """
    try:
        url = f"{STOCKTWITS_BASE}/trending/symbols.json"
        resp = requests.get(url, timeout=REQUEST_TIMEOUT, headers=_HEADERS)
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
        print(f"  StockTwits trending unavailable: {e}")
        return []


def fetch_stocktwits_messages(symbol, limit=30):
    """
    Fetch recent messages for a symbol from StockTwits.
    Returns list of dicts: [{body, created_at, sentiment}, ...]
    """
    try:
        url = f"{STOCKTWITS_BASE}/streams/symbol/{symbol}.json"
        resp = requests.get(url, timeout=REQUEST_TIMEOUT, headers=_HEADERS)
        if resp.status_code == 429:
            print(f"  StockTwits rate limited for {symbol}")
            return []
        if resp.status_code != 200:
            print(f"  StockTwits messages returned {resp.status_code} for {symbol}")
            return []

        data = resp.json()
        messages = data.get("messages", [])
        result = []
        for msg in messages[:limit]:
            user_sentiment = None
            if msg.get("entities", {}).get("sentiment"):
                user_sentiment = msg["entities"]["sentiment"].get("basic")

            result.append({
                "body": msg.get("body", ""),
                "created_at": msg.get("created_at", ""),
                "user": msg.get("user", {}).get("username", ""),
                "user_sentiment": user_sentiment,
                "likes": msg.get("likes", {}).get("total", 0),
                "source": "stocktwits",
            })
        return result
    except Exception as e:
        print(f"  StockTwits error for {symbol}: {e}")
        return []


# ------------------------------------------------------------------
# Yahoo Finance News Headlines (Fallback data source)
# ------------------------------------------------------------------
def fetch_yahoo_news_headlines(symbol):
    """
    Fetch recent news headlines for a ticker from Yahoo Finance via yfinance.
    Returns list of message-like dicts compatible with analyze_sentiment().
    """
    if not USE_YFINANCE:
        return []

    try:
        ticker = yf.Ticker(symbol)
        news = ticker.news
        if not news:
            return []

        messages = []
        for item in news[:20]:
            title = item.get("title", "")
            # Some yfinance versions use different field names
            publisher = item.get("publisher", item.get("source", ""))
            pub_date = item.get("providerPublishTime", item.get("publishedDate", ""))
            link = item.get("link", "")

            if title:
                messages.append({
                    "body": title,
                    "created_at": str(pub_date) if pub_date else "",
                    "user": publisher or "Yahoo Finance",
                    "user_sentiment": None,
                    "likes": 0,
                    "source": "yahoo_news",
                    "link": link,
                })
        return messages
    except Exception as e:
        print(f"  Yahoo news error for {symbol}: {e}")
        return []


def get_dashboard_tickers():
    """
    Get the list of tickers from the dashboard's signals.json
    to use as a fallback trending list when StockTwits is unavailable.
    Returns top movers (highest absolute YTD return).
    """
    try:
        signals_path = os.path.join(DATA_DIR, "signals.json")
        if not os.path.exists(signals_path):
            return []

        with open(signals_path, "r") as f:
            data = json.load(f)

        tickers = []
        for group in data.get("groups", []):
            for stock in group.get("stocks", []):
                tickers.append({
                    "symbol": stock.get("ticker", ""),
                    "title": group.get("name", ""),
                    "watchlist_count": 0,
                    "ytd_return": stock.get("ytd_return", 0),
                    "score": stock.get("score", 50),
                })

        # Sort by absolute YTD return (biggest movers = most interesting)
        tickers.sort(key=lambda t: abs(t.get("ytd_return", 0)), reverse=True)
        return tickers[:20]  # Top 20 movers
    except Exception as e:
        print(f"  Error loading dashboard tickers: {e}")
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
        if not body or not _analyzer:
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
    Uses multi-source approach:
      1. Try StockTwits trending + messages
      2. If StockTwits fails, use dashboard tickers + Yahoo Finance news
    Results are cached for 15 minutes.
    Returns list of ticker sentiment dicts sorted by message activity.
    """
    global _trending_cache
    now = time.time()

    if _trending_cache["data"] is not None and (now - _trending_cache["ts"]) < CACHE_TTL:
        return _trending_cache["data"]

    # Attempt 1: StockTwits trending
    print("Fetching trending tickers from StockTwits...")
    trending = fetch_trending_tickers()
    source_label = "StockTwits"

    # Attempt 2: Fallback to dashboard tickers if StockTwits unavailable
    if not trending:
        print("  StockTwits unavailable — falling back to dashboard tickers + Yahoo Finance news...")
        trending = get_dashboard_tickers()
        source_label = "Yahoo Finance News"

    if not trending:
        if _trending_cache["data"]:
            return _trending_cache["data"]
        return []

    results = []
    for i, ticker_info in enumerate(trending[:20]):
        symbol = ticker_info["symbol"]
        print(f"  Analyzing sentiment for {symbol} ({i+1}/{min(len(trending),20)})...")

        # Gather messages from all available sources
        messages = []

        # Try StockTwits messages
        st_msgs = fetch_stocktwits_messages(symbol, limit=30)
        messages.extend(st_msgs)

        # Always try Yahoo Finance news headlines (supplements StockTwits)
        yf_msgs = fetch_yahoo_news_headlines(symbol)
        messages.extend(yf_msgs)

        sentiment = analyze_sentiment(messages)

        results.append({
            "symbol": symbol,
            "title": ticker_info.get("title", ""),
            "watchlist_count": ticker_info.get("watchlist_count", 0),
            "data_source": source_label,
            **sentiment,
        })

        # Small delay to avoid rate limiting
        if i < len(trending) - 1:
            time.sleep(0.2)

    # Sort by message count (most discussed first)
    results.sort(key=lambda x: x["message_count"], reverse=True)

    _trending_cache = {"data": results, "ts": now}
    return results


def get_symbol_sentiment(symbol):
    """
    Get sentiment analysis for a specific ticker symbol.
    Uses multi-source: StockTwits + Yahoo Finance news.
    Results cached for 15 minutes per symbol.
    """
    symbol = symbol.upper().strip()
    now = time.time()

    if symbol in _symbol_cache and (now - _symbol_cache[symbol]["ts"]) < CACHE_TTL:
        return _symbol_cache[symbol]["data"]

    print(f"Analyzing sentiment for {symbol}...")
    messages = []

    # Try StockTwits
    st_msgs = fetch_stocktwits_messages(symbol, limit=30)
    messages.extend(st_msgs)

    # Always try Yahoo Finance news
    yf_msgs = fetch_yahoo_news_headlines(symbol)
    messages.extend(yf_msgs)

    sentiment = analyze_sentiment(messages)

    sources = []
    if st_msgs:
        sources.append(f"StockTwits ({len(st_msgs)})")
    if yf_msgs:
        sources.append(f"Yahoo News ({len(yf_msgs)})")

    result = {
        "symbol": symbol,
        "data_sources": ", ".join(sources) if sources else "No data available",
        **sentiment,
    }

    _symbol_cache[symbol] = {"data": result, "ts": now}
    return result
