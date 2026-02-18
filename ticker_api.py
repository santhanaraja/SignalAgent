#!/usr/bin/env python3
"""
Ticker Search API — Flask backend for on-demand ticker analysis.
Reuses signal_engine.py functions to compute technicals, momentum,
fundamentals, and trade signals for any ticker.
"""

import json
import os
import time
import datetime
from flask import Flask, jsonify, request, send_from_directory
from flask_cors import CORS

from signal_engine import (
    fetch_data,
    fetch_fundamentals_yfinance,
    score_stock,
    compute_trade_signal,
    compute_swing_trade_signal,
    compute_intraday_trade_signal,
    compute_stage_analysis,
    NumpyEncoder,
)
from sentiment_engine import get_trending_with_sentiment, get_symbol_sentiment
from fear_greed_engine import get_fear_greed_index

app = Flask(__name__, static_folder="public", static_url_path="")
CORS(app)  # Enable CORS for flexibility

DATA_DIR = os.path.join(os.path.dirname(__file__), "data")
HISTORY_FILE = os.path.join(DATA_DIR, "search_history.json")

# In-memory cache: { "AAPL": { "data": {...}, "ts": timestamp } }
_cache = {}
CACHE_TTL = 300  # 5 minutes


# ------------------------------------------------------------------
# Helpers
# ------------------------------------------------------------------
def _load_history():
    if os.path.exists(HISTORY_FILE):
        with open(HISTORY_FILE, "r") as f:
            return json.load(f)
    return {"searches": []}


def _save_history(history):
    os.makedirs(DATA_DIR, exist_ok=True)
    with open(HISTORY_FILE, "w") as f:
        json.dump(history, f, indent=2, cls=NumpyEncoder)


def _analyze_ticker(symbol):
    """Run full analysis pipeline for a single ticker."""
    # Check cache
    now = time.time()
    if symbol in _cache and (now - _cache[symbol]["ts"]) < CACHE_TTL:
        return _cache[symbol]["data"]

    # Fetch price data
    df = fetch_data(symbol, period="6mo")
    if df is None or len(df) < 20:
        return None

    # Score + signals
    score, signal, details = score_stock(df)

    # Fundamentals
    fundamentals = fetch_fundamentals_yfinance(symbol)
    details["fundamentals"] = fundamentals

    # Trade signals (no breaker context for ad-hoc)
    trade_sig, trade_reason = compute_trade_signal(details, breaker_status="clear")
    swing_signal = compute_swing_trade_signal(details, df)
    intraday_signal = compute_intraday_trade_signal(details, df)
    stage_analysis = compute_stage_analysis(details, df)

    result = {
        "ticker": symbol,
        "timestamp": datetime.datetime.now().isoformat(),
        "price": details.get("price", 0),
        "score": score,
        "signal": signal,
        "trade_signal": trade_sig,
        "trade_reasoning": trade_reason,
        # Technicals
        "rsi": details.get("rsi", 0),
        "macd": details.get("macd", 0),
        "macd_signal": details.get("macd_signal", 0),
        "macd_histogram": details.get("macd_histogram", 0),
        "ma20": details.get("ma20", 0),
        "ma50": details.get("ma50", 0),
        "ma200": details.get("ma200"),
        "volume_ratio": details.get("volume_ratio", 1.0),
        # Momentum
        "ytd_return": details.get("ytd_return", 0),
        "return_1m": details.get("return_1m", 0),
        "return_3m": details.get("return_3m", 0),
        "high_52w": details.get("high_52w", 0),
        "low_52w": details.get("low_52w", 0),
        "pct_from_52w_high": details.get("pct_from_52w_high", 0),
        "rs_vs_ma50": details.get("rs_vs_ma50", 0),
        "trend_strength": details.get("trend_strength", 0),
        # Fundamentals
        "fundamentals": fundamentals,
        # Swing & Intraday signals
        "swing_signal": swing_signal,
        "intraday_signal": intraday_signal,
        # Stage Analysis
        "stage_analysis": stage_analysis,
    }

    # Cache it
    _cache[symbol] = {"data": result, "ts": now}
    return result


# ------------------------------------------------------------------
# API Routes
# ------------------------------------------------------------------
@app.route("/api/ticker/<symbol>")
def search_ticker(symbol):
    """Analyze a single ticker on-demand."""
    symbol = symbol.upper().strip()
    if not symbol or len(symbol) > 10:
        return jsonify({"status": "error", "error": "Invalid ticker symbol"}), 400

    try:
        result = _analyze_ticker(symbol)
        if result is None:
            return jsonify({
                "status": "error",
                "error": f"No data found for '{symbol}'. Check the ticker symbol."
            }), 404

        # Add to search history
        history = _load_history()
        history["searches"].insert(0, {
            "ticker": symbol,
            "timestamp": datetime.datetime.now().isoformat(),
            "score": result["score"],
            "signal": result["signal"],
            "trade_signal": result["trade_signal"],
            "price": result["price"],
        })
        history["searches"] = history["searches"][:100]  # Keep last 100
        _save_history(history)

        return app.response_class(
            response=json.dumps({"status": "success", **result}, cls=NumpyEncoder),
            mimetype="application/json",
        )
    except Exception as e:
        return jsonify({"status": "error", "error": str(e)}), 500


@app.route("/api/history")
def get_history():
    """Return search history."""
    history = _load_history()
    return app.response_class(
        response=json.dumps(history, cls=NumpyEncoder),
        mimetype="application/json",
    )


@app.route("/api/history", methods=["DELETE"])
def clear_history():
    """Clear search history."""
    _save_history({"searches": []})
    return jsonify({"status": "success", "message": "History cleared"})


# ------------------------------------------------------------------
# Sentiment API Routes
# ------------------------------------------------------------------
@app.route("/api/sentiment/trending")
def sentiment_trending():
    """Return trending tickers with sentiment analysis."""
    try:
        tickers = get_trending_with_sentiment()
        return app.response_class(
            response=json.dumps({
                "status": "success",
                "timestamp": datetime.datetime.now().isoformat(),
                "tickers": tickers,
            }, cls=NumpyEncoder),
            mimetype="application/json",
        )
    except Exception as e:
        return jsonify({"status": "error", "error": str(e)}), 500


@app.route("/api/sentiment/<symbol>")
def sentiment_symbol(symbol):
    """Return sentiment analysis for a specific ticker."""
    symbol = symbol.upper().strip()
    if not symbol or len(symbol) > 10:
        return jsonify({"status": "error", "error": "Invalid ticker symbol"}), 400

    try:
        result = get_symbol_sentiment(symbol)
        return app.response_class(
            response=json.dumps({
                "status": "success",
                "timestamp": datetime.datetime.now().isoformat(),
                **result,
            }, cls=NumpyEncoder),
            mimetype="application/json",
        )
    except Exception as e:
        return jsonify({"status": "error", "error": str(e)}), 500


# ------------------------------------------------------------------
# Fear & Greed API Route
# ------------------------------------------------------------------
@app.route("/api/fear-greed")
def fear_greed():
    """Return the composite Fear & Greed Index with all 7 indicators."""
    try:
        result = get_fear_greed_index()
        return app.response_class(
            response=json.dumps({
                "status": "success",
                "timestamp": datetime.datetime.now().isoformat(),
                **result,
            }, cls=NumpyEncoder),
            mimetype="application/json",
        )
    except Exception as e:
        return jsonify({"status": "error", "error": str(e)}), 500


# ------------------------------------------------------------------
# Static file serving
# ------------------------------------------------------------------
@app.route("/")
def index():
    return send_from_directory("public", "index.html")


@app.route("/<path:path>")
def static_files(path):
    return send_from_directory("public", path)


# ------------------------------------------------------------------
# Main
# ------------------------------------------------------------------
if __name__ == "__main__":
    port = int(os.environ.get("PORT", 5002))
    print("\n" + "=" * 60)
    print("  SignalAgent Ticker Search API")
    print("=" * 60)
    print(f"  Dashboard:     http://localhost:{port}/")
    print(f"  Ticker Search: http://localhost:{port}/search.html")
    print(f"  API Endpoint:  http://localhost:{port}/api/ticker/<SYMBOL>")
    print(f"  History:       http://localhost:{port}/api/history")
    print("=" * 60 + "\n")
    app.run(host="0.0.0.0", port=port, debug=False)
