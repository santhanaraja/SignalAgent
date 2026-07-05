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
import threading
import traceback
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
    compute_rsi,
    compute_macd,
    run_engine,
    NumpyEncoder,
)
import numpy as np
import pandas as pd
from sentiment_engine import get_trending_with_sentiment, get_symbol_sentiment
from fear_greed_engine import get_fear_greed_index

app = Flask(__name__, static_folder="public", static_url_path="")
CORS(app)  # Enable CORS for flexibility

DATA_DIR = os.path.join(os.path.dirname(__file__), "data")
PUBLIC_DIR = os.path.join(os.path.dirname(__file__), "public")
HISTORY_FILE = os.path.join(DATA_DIR, "search_history.json")

# In-memory cache: { "AAPL": { "data": {...}, "ts": timestamp } }
_cache = {}
CACHE_TTL = 300  # 5 minutes

# Refresh state
REFRESH_INTERVAL_MINUTES = 15
_refresh_lock = threading.Lock()
_refresh_status = {
    "running": False,
    "last_run": None,
    "last_duration_sec": None,
    "last_error": None,
    "next_scheduled": None,
}


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
        "timestamp": datetime.datetime.now(datetime.timezone.utc).isoformat(),
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
            "timestamp": datetime.datetime.now(datetime.timezone.utc).isoformat(),
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
                "timestamp": datetime.datetime.now(datetime.timezone.utc).isoformat(),
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
                "timestamp": datetime.datetime.now(datetime.timezone.utc).isoformat(),
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
                "timestamp": datetime.datetime.now(datetime.timezone.utc).isoformat(),
                **result,
            }, cls=NumpyEncoder),
            mimetype="application/json",
        )
    except Exception as e:
        return jsonify({"status": "error", "error": str(e)}), 500


# ------------------------------------------------------------------
# AI Chat API (Claude-powered, grounded in dashboard data)
# ------------------------------------------------------------------
ANTHROPIC_API_KEY = os.environ.get("ANTHROPIC_API_KEY", "")
_chat_sessions = {}  # session_id -> [messages]
CHAT_MAX_HISTORY = 20  # keep last 20 messages per session


def _build_market_context():
    """
    Build a concise market data summary from signals.json for the LLM.
    This is injected as system context so Claude can answer questions
    purely based on the live dashboard data.
    """
    signals_path = os.path.join(PUBLIC_DIR, "signals.json")
    try:
        with open(signals_path, "r") as f:
            data = json.load(f)
    except Exception:
        return "No market data available. The signal engine has not run yet."

    ts = data.get("timestamp", "unknown")
    sp500 = data.get("sp500_ytd", "N/A")
    indexes = data.get("indexes", {})
    groups = data.get("groups", [])

    lines = []
    lines.append(f"DATA TIMESTAMP: {ts}")
    lines.append(f"S&P 500 YTD: {sp500}%")

    # Index summary
    idx_parts = []
    for sym, info in indexes.items():
        if isinstance(info, dict):
            idx_parts.append(f"{info.get('name', sym)}: ${info.get('price', 'N/A')} ({info.get('ytd_return', 'N/A')}% YTD)")
    if idx_parts:
        lines.append("INDEXES: " + " | ".join(idx_parts))

    lines.append(f"\nTOTAL GROUPS: {len(groups)}")
    lines.append("")

    for g in groups:
        name = g.get("name", "")
        rank = g.get("rank", "")
        avg_ytd = g.get("avg_ytd", 0)
        avg_score = g.get("avg_score", 0)
        signal = g.get("group_signal", "")
        breaker = g.get("breaker_status", "clear")
        thesis = g.get("thesis", "")
        thesis_breaker = g.get("thesis_breaker", "")
        cycle = g.get("cycle_stage", "")
        sector = g.get("sector", "")
        beating = g.get("beating_sp500_count", 0)
        total = g.get("stock_count", 0)

        lines.append(f"--- {name} (Rank #{rank}) ---")
        lines.append(f"  Sector: {sector} | Cycle: {cycle} | Signal: {signal} | Breaker: {breaker}")
        lines.append(f"  Avg YTD: {avg_ytd}% | Avg Score: {avg_score}/100 | Beating S&P: {beating}/{total}")
        lines.append(f"  Thesis: {thesis}")
        lines.append(f"  Thesis Breaker: {thesis_breaker}")

        # Individual stocks
        stocks = g.get("stocks", [])
        for s in stocks:
            ticker = s.get("ticker", "")
            price = s.get("price", 0)
            ytd = s.get("ytd_return", 0)
            score = s.get("score", 0)
            rsi = s.get("rsi", 0)
            ts_signal = s.get("trade_signal", "")
            trade_reason = s.get("trade_reasoning", "")
            stage = s.get("stage_analysis", {})
            stage_name = stage.get("stage_name", "N/A") if stage else "N/A"
            stage_num = stage.get("stage", 0) if stage else 0
            fund = s.get("fundamentals", {})
            mcap = fund.get("market_cap")
            fpe = fund.get("forward_pe")
            pct_52h = s.get("pct_from_52w_high", 0)

            mcap_str = ""
            if mcap:
                if mcap >= 1e12:
                    mcap_str = f"${mcap/1e12:.1f}T"
                elif mcap >= 1e9:
                    mcap_str = f"${mcap/1e9:.1f}B"
                else:
                    mcap_str = f"${mcap/1e6:.0f}M"

            lines.append(
                f"    {ticker}: ${price:.2f} | YTD:{ytd:+.1f}% | Score:{score} | RSI:{rsi:.0f} | "
                f"Stage:S{stage_num} {stage_name} | Trade:{ts_signal} | "
                f"MCap:{mcap_str} | FwdPE:{fpe or 'N/A'} | %from52wH:{pct_52h:.1f}%"
            )
            if trade_reason:
                lines.append(f"      Reasoning: {trade_reason}")
        lines.append("")

    return "\n".join(lines)


SYSTEM_PROMPT = """You are the Market Pulse AI Assistant — an expert market analyst embedded in the Market Pulse Dashboard.

You answer questions ONLY based on the live market data provided below. You have access to:
- All 12 GICS industry groups with their tickers, scores, signals, and thesis
- Individual stock data: price, YTD return, RSI, MACD, trade signals, stage analysis, fundamentals
- Thesis breaker status and alerts for each group
- S&P 500 and major index performance

RULES:
1. Answer ONLY from the data provided. If the data doesn't contain what's asked, say so.
2. Be concise and specific — cite actual numbers, prices, and percentages from the data.
3. When asked about a ticker, provide its score, signal, stage, YTD, and trade reasoning.
4. When asked for recommendations, base them strictly on the signals, scores, and trade signals in the data.
5. Format responses clearly. Use bold for tickers and key metrics.
6. You are NOT a financial advisor. Always note that these are algorithmic signals, not financial advice.
7. If asked about topics outside the dashboard data (politics, weather, etc.), politely redirect to market-related questions.

CURRENT MARKET DATA:
{market_data}
"""


@app.route("/api/chat", methods=["POST"])
def chat():
    """AI chat endpoint powered by Claude, grounded in live dashboard data."""
    if not ANTHROPIC_API_KEY:
        return jsonify({
            "status": "error",
            "error": "Chat is not configured. Set ANTHROPIC_API_KEY environment variable."
        }), 503

    try:
        body = request.get_json()
        if not body or not body.get("message"):
            return jsonify({"status": "error", "error": "Message is required"}), 400

        user_message = body["message"].strip()
        session_id = body.get("session_id", "default")

        if not user_message or len(user_message) > 2000:
            return jsonify({"status": "error", "error": "Message must be 1-2000 characters"}), 400

        # Get or create session history
        if session_id not in _chat_sessions:
            _chat_sessions[session_id] = []
        history = _chat_sessions[session_id]

        # Build context from live market data
        market_data = _build_market_context()
        system = SYSTEM_PROMPT.format(market_data=market_data)

        # Add user message to history
        history.append({"role": "user", "content": user_message})

        # Trim history to last N messages
        if len(history) > CHAT_MAX_HISTORY:
            history = history[-CHAT_MAX_HISTORY:]
            _chat_sessions[session_id] = history

        # Call Claude API
        import anthropic
        client = anthropic.Anthropic(api_key=ANTHROPIC_API_KEY)

        response = client.messages.create(
            model="claude-sonnet-4-20250514",
            max_tokens=1024,
            system=system,
            messages=history,
        )

        assistant_message = response.content[0].text

        # Add assistant response to history
        history.append({"role": "assistant", "content": assistant_message})

        return app.response_class(
            response=json.dumps({
                "status": "success",
                "message": assistant_message,
                "session_id": session_id,
            }, cls=NumpyEncoder),
            mimetype="application/json",
        )

    except Exception as e:
        traceback.print_exc()
        return jsonify({"status": "error", "error": str(e)}), 500


@app.route("/api/chat/clear", methods=["POST"])
def clear_chat():
    """Clear chat session history."""
    body = request.get_json() or {}
    session_id = body.get("session_id", "default")
    _chat_sessions.pop(session_id, None)
    return jsonify({"status": "success", "message": "Chat history cleared"})


# ------------------------------------------------------------------
# Dashboard Refresh API
# ------------------------------------------------------------------
def _run_signal_refresh():
    """Run the signal engine to regenerate signals.json with fresh data."""
    global _refresh_status
    if _refresh_status["running"]:
        return False, "Refresh already in progress"

    with _refresh_lock:
        _refresh_status["running"] = True
        _refresh_status["last_error"] = None
        start = time.time()
        try:
            print(f"\n{'='*60}")
            print(f"  SIGNAL REFRESH STARTED at {datetime.datetime.now(datetime.timezone.utc).isoformat()}")
            print(f"{'='*60}")
            run_engine()
            duration = time.time() - start
            _refresh_status["last_run"] = datetime.datetime.now(datetime.timezone.utc).isoformat()
            _refresh_status["last_duration_sec"] = round(duration, 1)
            _refresh_status["running"] = False
            # Clear ticker cache so individual lookups also get fresh data
            _cache.clear()
            print(f"  REFRESH COMPLETED in {duration:.1f}s")
            return True, f"Refresh completed in {duration:.1f}s"
        except Exception as e:
            duration = time.time() - start
            _refresh_status["last_error"] = str(e)
            _refresh_status["running"] = False
            print(f"  REFRESH FAILED after {duration:.1f}s: {e}")
            traceback.print_exc()
            return False, str(e)


def _background_refresh_loop():
    """Background thread that runs signal engine every REFRESH_INTERVAL_MINUTES."""
    while True:
        next_run = datetime.datetime.now() + datetime.timedelta(minutes=REFRESH_INTERVAL_MINUTES)
        _refresh_status["next_scheduled"] = next_run.isoformat()
        print(f"  Next scheduled refresh: {next_run.strftime('%H:%M:%S')}")
        time.sleep(REFRESH_INTERVAL_MINUTES * 60)
        print(f"\n  [SCHEDULER] Auto-refresh triggered at {datetime.datetime.now(datetime.timezone.utc).isoformat()}")
        _run_signal_refresh()


@app.route("/api/refresh", methods=["POST"])
def trigger_refresh():
    """Manually trigger a signal engine refresh. Runs in background."""
    if _refresh_status["running"]:
        return app.response_class(
            response=json.dumps({
                "status": "in_progress",
                "message": "Refresh already running",
                "started": _refresh_status.get("last_run"),
            }, cls=NumpyEncoder),
            mimetype="application/json",
        ), 202

    # Run in a thread so the API returns immediately
    thread = threading.Thread(target=_run_signal_refresh, daemon=True)
    thread.start()

    return app.response_class(
        response=json.dumps({
            "status": "started",
            "message": "Signal refresh started — dashboard will update when complete",
        }, cls=NumpyEncoder),
        mimetype="application/json",
    )


@app.route("/api/refresh/status")
def refresh_status():
    """Check the current refresh status."""
    # Also read the timestamp from signals.json to show data freshness
    signals_path = os.path.join(PUBLIC_DIR, "signals.json")
    data_timestamp = None
    try:
        with open(signals_path, "r") as f:
            signals = json.load(f)
            data_timestamp = signals.get("timestamp")
    except Exception:
        pass

    return app.response_class(
        response=json.dumps({
            "status": "success",
            **_refresh_status,
            "data_timestamp": data_timestamp,
            "refresh_interval_minutes": REFRESH_INTERVAL_MINUTES,
        }, cls=NumpyEncoder),
        mimetype="application/json",
    )


# ------------------------------------------------------------------
# Framework API Routes
# ------------------------------------------------------------------
_framework_lock = threading.Lock()
_framework_status = {
    "running": False,
    "last_run": None,
    "last_error": None,
}


def _run_framework_refresh():
    """Run the framework engine to generate framework.json."""
    global _framework_status
    if _framework_status["running"]:
        return False, "Framework run already in progress"

    with _framework_lock:
        _framework_status["running"] = True
        _framework_status["last_error"] = None
        start = time.time()
        try:
            from framework.framework_runner import run_framework
            result = run_framework(force_fetch=True)
            duration = time.time() - start
            _framework_status["last_run"] = datetime.datetime.now(datetime.timezone.utc).isoformat()
            _framework_status["running"] = False
            print(f"  FRAMEWORK RUN COMPLETED in {duration:.1f}s")
            return True, result
        except Exception as e:
            _framework_status["running"] = False
            _framework_status["last_error"] = str(e)
            traceback.print_exc()
            return False, str(e)


@app.route("/api/framework/latest")
def framework_latest():
    """Return the latest framework output (regime + themes + rules)."""
    framework_path = os.path.join(PUBLIC_DIR, "framework.json")
    if not os.path.exists(framework_path):
        return jsonify({
            "status": "error",
            "error": "Framework has not been run yet. Trigger a run via POST /api/framework/run.",
        }), 404

    try:
        with open(framework_path, "r") as f:
            data = json.load(f)
        return app.response_class(
            response=json.dumps({"status": "success", **data}, cls=NumpyEncoder),
            mimetype="application/json",
        )
    except Exception as e:
        return jsonify({"status": "error", "error": str(e)}), 500


@app.route("/api/framework/run", methods=["POST"])
def framework_run():
    """Trigger a fresh framework run. Fetches live data and re-computes everything."""
    if _framework_status["running"]:
        return app.response_class(
            response=json.dumps({
                "status": "in_progress",
                "message": "Framework run already in progress",
            }, cls=NumpyEncoder),
            mimetype="application/json",
        ), 202

    # Run in background thread
    thread = threading.Thread(target=_run_framework_refresh, daemon=True)
    thread.start()

    return app.response_class(
        response=json.dumps({
            "status": "started",
            "message": "Framework run started — will fetch fresh data and compute regime, themes, and rules.",
        }, cls=NumpyEncoder),
        mimetype="application/json",
    )


@app.route("/api/framework/status")
def framework_status():
    """Check framework run status."""
    framework_path = os.path.join(PUBLIC_DIR, "framework.json")
    data_timestamp = None
    try:
        with open(framework_path, "r") as f:
            data = json.load(f)
            data_timestamp = data.get("generated_at")
    except Exception:
        pass

    return app.response_class(
        response=json.dumps({
            "status": "success",
            **_framework_status,
            "data_timestamp": data_timestamp,
        }, cls=NumpyEncoder),
        mimetype="application/json",
    )


@app.route("/api/framework/history")
def framework_history():
    """Return regime and theme history for charting."""
    import os as _os
    state_dir = _os.path.join(_os.path.dirname(__file__), "framework", "state")
    regime_path = _os.path.join(state_dir, "regime_history.json")
    theme_path = _os.path.join(state_dir, "theme_history.json")

    regime_hist = []
    theme_hist = []
    try:
        if _os.path.exists(regime_path):
            with open(regime_path, "r") as f:
                regime_hist = json.load(f)
        if _os.path.exists(theme_path):
            with open(theme_path, "r") as f:
                theme_hist = json.load(f)
    except Exception:
        pass

    return app.response_class(
        response=json.dumps({
            "status": "success",
            "regime_history": regime_hist,
            "theme_history": theme_hist,
        }, cls=NumpyEncoder),
        mimetype="application/json",
    )


# ------------------------------------------------------------------
# Framework Public JSON API (clean endpoints for external consumers)
# ------------------------------------------------------------------
FRAMEWORK_DIR = os.path.join(os.path.dirname(__file__), "framework")
FRAMEWORK_STATE_DIR = os.path.join(FRAMEWORK_DIR, "state")


@app.route("/api/framework/latest.json")
def framework_latest_json():
    """
    Public JSON API — returns the full framework output as clean JSON.
    Same data the dashboard reads from framework.json.
    No wrapper, no status field — just the raw framework object.
    """
    framework_path = os.path.join(PUBLIC_DIR, "framework.json")
    if not os.path.exists(framework_path):
        return app.response_class(
            response=json.dumps({
                "error": "Framework has not been run yet.",
                "hint": "POST /api/framework/run to trigger a run.",
            }),
            status=404,
            mimetype="application/json",
        )
    try:
        with open(framework_path, "r") as f:
            data = json.load(f)
        return app.response_class(
            response=json.dumps(data, cls=NumpyEncoder),
            mimetype="application/json",
        )
    except Exception as e:
        return app.response_class(
            response=json.dumps({"error": str(e)}),
            status=500,
            mimetype="application/json",
        )


@app.route("/api/framework/history.json")
def framework_history_json():
    """
    Public JSON API — returns regime + theme run history as clean JSON.
    """
    regime_path = os.path.join(FRAMEWORK_STATE_DIR, "regime_history.json")
    theme_path = os.path.join(FRAMEWORK_STATE_DIR, "theme_history.json")

    regime_hist = []
    theme_hist = []
    try:
        if os.path.exists(regime_path):
            with open(regime_path, "r") as f:
                regime_hist = json.load(f)
        if os.path.exists(theme_path):
            with open(theme_path, "r") as f:
                theme_hist = json.load(f)
    except Exception:
        pass

    return app.response_class(
        response=json.dumps({
            "regime_history": regime_hist,
            "theme_history": theme_hist,
        }, cls=NumpyEncoder),
        mimetype="application/json",
    )


@app.route("/api/framework/gauges.json")
def framework_gauges_json():
    """
    Public JSON API — returns the swing regime gauges as a compact object.
    Useful for quick regime checks without downloading the full framework output.

    gauges holds the 3 swing VOTERS (counts range 0-3). spy_vs_200dma is
    the backdrop_gate (binary cap, not a voter); yield_curve lives under
    macro_inputs (computed, non-voting).

    Response shape:
    {
      "generated_at": "...",
      "regime": "Risk-on / Trending",
      "risk_on_count": 3,
      "caution_count": 0,
      "risk_off_count": 0,
      "consecutive_weeks": 2,
      "gauges": {
        "vix_5d_avg": { "value": 15.8, "signal": "risk_on", "detail": "..." },
        "hy_spread":  { ... },
        "breadth":    { ... }
      },
      "backdrop_gate": { "gauge": "spy_vs_200dma", "open": true, "capped": false, ... },
      "macro_inputs":  { "yield_curve": { "value": 0.84, "signal": "risk_on", ... } }
    }
    """
    framework_path = os.path.join(PUBLIC_DIR, "framework.json")
    if not os.path.exists(framework_path):
        return app.response_class(
            response=json.dumps({
                "error": "Framework has not been run yet.",
                "hint": "POST /api/framework/run to trigger a run.",
            }),
            status=404,
            mimetype="application/json",
        )
    try:
        with open(framework_path, "r") as f:
            data = json.load(f)

        regime = data.get("regime", {})
        compact = {
            "generated_at": data.get("generated_at"),
            "regime": regime.get("regime"),
            "regime_action": regime.get("action"),
            "risk_on_count": regime.get("risk_on_count"),
            "caution_count": regime.get("caution_count"),
            "risk_off_count": regime.get("risk_off_count"),
            "consecutive_weeks": regime.get("consecutive_weeks_at_state"),
            "regime_change_pending": regime.get("regime_change_pending", False),
            "gauges": regime.get("gauges", {}),
            "backdrop_gate": regime.get("backdrop_gate"),
            "macro_inputs": regime.get("macro_inputs"),
        }

        return app.response_class(
            response=json.dumps(compact, cls=NumpyEncoder),
            mimetype="application/json",
        )
    except Exception as e:
        return app.response_class(
            response=json.dumps({"error": str(e)}),
            status=500,
            mimetype="application/json",
        )


@app.route("/api/framework/leaders.json")
def framework_leaders_json():
    """
    Public JSON API — returns the per-theme constituent leaders for ALL themes.

    Each theme carries a "qualified" flag (deployment-ready top-2 + active) and
    its top-3 "leaders". Warnings (earnings_within_7d / at_52w_high /
    rsi_overbought) are populated only for qualified themes; non-qualified themes
    are informational and always return warnings: [].

    Response shape:
    {
      "generated_at": "...",
      "theme_leaders": {
        "Semis": {
          "qualified": true,
          "leaders": [
            { "ticker": "MU", "current_price": 1133.99, "return_4w": 51.0,
              "return_12w": 219.15, "composite_rank": 2, "rsi_14": 66,
              "warnings": ["earnings_within_7d", "at_52w_high"] },
            ...
          ]
        },
        "Energy": { "qualified": false, "leaders": [ { ..., "warnings": [] }, ... ] },
        ...
      }
    }
    """
    framework_path = os.path.join(PUBLIC_DIR, "framework.json")
    if not os.path.exists(framework_path):
        return app.response_class(
            response=json.dumps({
                "error": "Framework has not been run yet.",
                "hint": "POST /api/framework/run to trigger a run.",
            }),
            status=404,
            mimetype="application/json",
        )
    try:
        with open(framework_path, "r") as f:
            data = json.load(f)
        return app.response_class(
            response=json.dumps({
                "generated_at": data.get("generated_at"),
                "theme_leaders": data.get("theme_leaders", {}),
            }, cls=NumpyEncoder),
            mimetype="application/json",
        )
    except Exception as e:
        return app.response_class(
            response=json.dumps({"error": str(e)}),
            status=500,
            mimetype="application/json",
        )


@app.route("/api/universe/candidates.json")
def universe_candidates_json():
    """
    Public JSON API — inspect the candidate source universe (read-only).

    Serves the pre-built public/universe_candidates.json, refreshed by the
    weekly rotation (`python universe_builder.py --force`) or a standalone
    `python universe_source.py` run. This is the candidate SOURCE pool the
    weekly top-N active universe is selected from (see /api/universe/ranking.json).

    Response shape:
    {
      "generated_at": "...",
      "total_count": 540,
      "by_source": { "etf_holdings": .., "sp500": .., "manual_additions": .., "overlap_count": .. },
      "by_gics_sub_industry": { "Semiconductors": 18, ... },
      "unclassified": 12,
      "unclassified_tickers": [ ... ]
    }
    """
    path = os.path.join(PUBLIC_DIR, "universe_candidates.json")
    if not os.path.exists(path):
        return app.response_class(
            response=json.dumps({
                "error": "Candidate universe has not been built yet.",
                "hint": "Run: python universe_source.py",
            }),
            status=404,
            mimetype="application/json",
        )
    try:
        with open(path, "r") as f:
            data = json.load(f)
        return app.response_class(
            response=json.dumps(data, cls=NumpyEncoder),
            mimetype="application/json",
        )
    except Exception as e:
        return app.response_class(
            response=json.dumps({"error": str(e)}),
            status=500,
            mimetype="application/json",
        )


@app.route("/api/universe/ranking.json")
def universe_ranking_json():
    """
    Public JSON API — full group-ranking audit table (read-only).

    Serves the pre-built public/universe_ranking.json produced by the weekly
    rotation (`python universe_builder.py --force`). Contains EVERY GICS
    sub-industry group with composite/median returns, candidate + qualifier
    counts, and a per-ticker status (selected / failed_score_gate /
    group_below_min_candidates / group_outranked / ...) — see status_legend
    in the payload. Observability only: the active universe consumed by
    signal_engine is the separate top-N artifact.
    """
    path = os.path.join(PUBLIC_DIR, "universe_ranking.json")
    if not os.path.exists(path):
        return app.response_class(
            response=json.dumps({
                "error": "Universe ranking has not been built yet.",
                "hint": "Run: python universe_builder.py --force",
            }),
            status=404,
            mimetype="application/json",
        )
    try:
        with open(path, "r") as f:
            data = json.load(f)
        return app.response_class(
            response=json.dumps(data, cls=NumpyEncoder),
            mimetype="application/json",
        )
    except Exception as e:
        return app.response_class(
            response=json.dumps({"error": str(e)}),
            status=500,
            mimetype="application/json",
        )


# ------------------------------------------------------------------
# Technicals API — Moving averages, indicators, ranges for any ticker
# ------------------------------------------------------------------
TECHNICALS_DEFAULT_WATCHLIST = [
    "AIQ", "SMH", "QTUM", "XLE", "GLD", "ITA", "IBIT", "XBI",
    "SPY", "QQQ", "VIX", "MU", "SNDK", "MRVL", "NVDA", "AVGO",
    "GOOGL", "GGLL", "IONQ", "MSTY", "SGOV",
]
TECHNICALS_BATCH_LIMIT = 20


def _framework_technicals_symbols():
    """
    Read theme proxies + all constituents from framework/config.yaml so the
    technicals cache is preloaded for everything the framework references.
    Returns [] on any failure (config missing, yaml unavailable, etc.).
    """
    symbols = []
    try:
        import yaml
        cfg_path = os.path.join(FRAMEWORK_DIR, "config.yaml")
        with open(cfg_path, "r") as f:
            cfg = yaml.safe_load(f) or {}
        for theme in cfg.get("themes", {}).get("watchlist", []):
            if theme.get("proxy"):
                symbols.append(theme["proxy"])
            for c in theme.get("constituents", []) or []:
                symbols.append(c)
    except Exception as e:
        print(f"[technicals] Could not load framework constituents: {e}")
    return symbols


def _effective_technicals_watchlist():
    """Default watchlist unioned with framework proxies + constituents (deduped, order-preserving)."""
    seen = set()
    out = []
    for sym in list(TECHNICALS_DEFAULT_WATCHLIST) + _framework_technicals_symbols():
        u = (sym or "").upper()
        if u and u not in seen:
            seen.add(u)
            out.append(u)
    return out

# Separate cache for technicals: { "SMH": { "data": {...}, "ts": time.time() } }
_technicals_cache = {}
_technicals_cache_lock = threading.Lock()


def _is_market_hours():
    """Check if current time is within US market hours (9:30 AM - 4:00 PM ET, Mon-Fri)."""
    try:
        from zoneinfo import ZoneInfo
    except ImportError:
        from backports.zoneinfo import ZoneInfo
    et = datetime.datetime.now(ZoneInfo("America/New_York"))
    if et.weekday() >= 5:  # Saturday/Sunday
        return False
    market_open = et.replace(hour=9, minute=30, second=0, microsecond=0)
    market_close = et.replace(hour=16, minute=0, second=0, microsecond=0)
    return market_open <= et <= market_close


def _technicals_cache_ttl():
    """15 min during market hours, 60 min outside."""
    return 900 if _is_market_hours() else 3600


def _compute_technicals(symbol):
    """
    Compute full technicals for a single ticker.
    Reuses signal_engine.fetch_data for yfinance data, plus compute_rsi/compute_macd.
    Returns (result_dict, warnings_list) or (None, error_string).
    """
    # Check cache
    now = time.time()
    ttl = _technicals_cache_ttl()
    with _technicals_cache_lock:
        cached = _technicals_cache.get(symbol)
        if cached and (now - cached["ts"]) < ttl:
            return cached["data"], cached.get("warnings", [])

    # Fetch 1 year of daily data (need 200+ bars for SMA200)
    df = fetch_data(symbol, period="1y")
    if df is None or len(df) < 5:
        return None, f"No data found for '{symbol}'"

    close = df["Close"]
    high = df["High"]
    low = df["Low"]
    price = round(float(close.iloc[-1]), 2)
    warnings = []

    # --- Moving Averages ---
    def _sma(n):
        if len(close) >= n:
            return round(float(close.rolling(n).mean().iloc[-1]), 2)
        warnings.append(f"sma_{n} requires {n} bars, only {len(close)} available")
        return None

    def _ema(n):
        if len(close) >= n:
            return round(float(close.ewm(span=n, adjust=False).mean().iloc[-1]), 2)
        warnings.append(f"ema_{n} requires {n} bars, only {len(close)} available")
        return None

    moving_averages = {
        "sma_5": _sma(5),
        "sma_10": _sma(10),
        "sma_20": _sma(20),
        "sma_50": _sma(50),
        "sma_200": _sma(200),
        "ema_9": _ema(9),
        "ema_21": _ema(21),
    }

    # --- Ranges ---
    def _atr(period=14):
        if len(df) < period + 1:
            warnings.append(f"atr_{period} requires {period+1} bars")
            return None
        h = high.iloc[-(period + 1):]
        l = low.iloc[-(period + 1):]
        c = close.iloc[-(period + 1):]
        prev_c = c.shift(1)
        tr = pd.concat([
            h - l,
            (h - prev_c).abs(),
            (l - prev_c).abs(),
        ], axis=1).max(axis=1)
        return round(float(tr.iloc[1:].mean()), 2)

    high_52w = round(float(high.iloc[-min(252, len(high)):].max()), 2)
    low_52w = round(float(low.iloc[-min(252, len(low)):].min()), 2)
    high_20d = round(float(high.iloc[-min(20, len(high)):].max()), 2) if len(high) >= 2 else None
    low_20d = round(float(low.iloc[-min(20, len(low)):].min()), 2) if len(low) >= 2 else None

    ranges = {
        "atr_14": _atr(14),
        "high_52w": high_52w,
        "low_52w": low_52w,
        "high_20d": high_20d,
        "low_20d": low_20d,
    }

    # --- Indicators (reuse signal_engine functions) ---
    rsi_series = compute_rsi(close, period=14)
    rsi_val = round(float(rsi_series.iloc[-1]), 2) if not pd.isna(rsi_series.iloc[-1]) else None
    if rsi_val is None:
        warnings.append("rsi_14 requires 14+ bars of data")

    macd_line, signal_line, histogram = compute_macd(close)
    macd_val = round(float(macd_line.iloc[-1]), 2) if not pd.isna(macd_line.iloc[-1]) else None
    macd_sig_val = round(float(signal_line.iloc[-1]), 2) if not pd.isna(signal_line.iloc[-1]) else None
    macd_hist_val = round(float(histogram.iloc[-1]), 2) if not pd.isna(histogram.iloc[-1]) else None

    indicators = {
        "rsi_14": rsi_val,
        "macd": macd_val,
        "macd_signal": macd_sig_val,
        "macd_histogram": macd_hist_val,
    }

    # --- Trend ---
    sma200 = moving_averages["sma_200"]
    sma50 = moving_averages["sma_50"]
    sma20 = moving_averages["sma_20"]

    # 200DMA slope: compare current 200DMA to 20 bars ago
    slope_200 = None
    if len(close) >= 220:
        ma200_series = close.rolling(200).mean()
        cur = float(ma200_series.iloc[-1])
        prev = float(ma200_series.iloc[-20])
        if not (pd.isna(cur) or pd.isna(prev)):
            slope_200 = "rising" if cur > prev else ("falling" if cur < prev else "flat")

    # Golden cross: SMA50 > SMA200
    golden_cross = None
    if sma50 is not None and sma200 is not None:
        golden_cross = sma50 > sma200

    trend = {
        "above_200dma": price > sma200 if sma200 is not None else None,
        "above_50dma": price > sma50 if sma50 is not None else None,
        "above_20dma": price > sma20 if sma20 is not None else None,
        "200dma_slope": slope_200,
        "golden_cross": golden_cross,
    }

    result = {
        "ticker": symbol,
        "generated_at": datetime.datetime.now(datetime.timezone.utc).isoformat(),
        "price": price,
        "bars_available": len(df),
        "moving_averages": moving_averages,
        "ranges": ranges,
        "indicators": indicators,
        "trend": trend,
    }
    if warnings:
        result["warnings"] = warnings

    # Cache it
    with _technicals_cache_lock:
        _technicals_cache[symbol] = {"data": result, "warnings": warnings, "ts": now}

    return result, warnings


@app.route("/api/technicals/batch.json")
def technicals_batch():
    """
    Public JSON API — technicals for multiple tickers.
    Query param: ?tickers=SMH,QTUM,AIQ  (comma-separated, max 20).
    Returns array of technicals objects. Failed tickers included with error field.
    """
    tickers_param = request.args.get("tickers", "")
    if not tickers_param:
        return app.response_class(
            response=json.dumps({"error": "Missing ?tickers= query parameter", "hint": "e.g. /api/technicals/batch.json?tickers=SMH,QTUM,AIQ"}),
            status=400,
            mimetype="application/json",
        )

    symbols = [t.strip().upper() for t in tickers_param.split(",") if t.strip()]
    symbols = list(dict.fromkeys(symbols))  # dedupe preserving order

    if len(symbols) > TECHNICALS_BATCH_LIMIT:
        return app.response_class(
            response=json.dumps({"error": f"Too many tickers. Maximum {TECHNICALS_BATCH_LIMIT} per request.", "requested": len(symbols)}),
            status=400,
            mimetype="application/json",
        )

    results = []
    for sym in symbols:
        if not sym or len(sym) > 10:
            results.append({"ticker": sym, "error": "Invalid ticker symbol"})
            continue
        result, _ = _compute_technicals(sym)
        if result is None:
            results.append({"ticker": sym, "error": "Ticker not found"})
        else:
            results.append(result)

    resp = app.response_class(
        response=json.dumps(results, cls=NumpyEncoder),
        mimetype="application/json",
    )
    ttl = _technicals_cache_ttl()
    resp.headers["Cache-Control"] = f"public, max-age={ttl}"
    return resp


@app.route("/api/technicals/<symbol>.json")
def technicals_single(symbol):
    """
    Public JSON API — full technicals for a single ticker.
    Returns moving averages, ranges, indicators, and trend analysis.
    Cached 15 min (market hours) / 60 min (off hours).
    """
    symbol = symbol.upper().strip()
    if not symbol or len(symbol) > 10:
        return app.response_class(
            response=json.dumps({"error": "Invalid ticker symbol", "ticker": symbol}),
            status=400,
            mimetype="application/json",
        )

    result, warnings_or_err = _compute_technicals(symbol)
    if result is None:
        return app.response_class(
            response=json.dumps({"error": "Ticker not found", "ticker": symbol}),
            status=404,
            mimetype="application/json",
        )

    resp = app.response_class(
        response=json.dumps(result, cls=NumpyEncoder),
        mimetype="application/json",
    )
    ttl = _technicals_cache_ttl()
    resp.headers["Cache-Control"] = f"public, max-age={ttl}"
    return resp


def _preload_technicals_watchlist():
    """Preload default watchlist + framework constituents into technicals cache."""
    watchlist = _effective_technicals_watchlist()
    print(f"[technicals] Preloading {len(watchlist)} tickers (incl. framework constituents)...")
    loaded = 0
    for sym in watchlist:
        try:
            result, _ = _compute_technicals(sym)
            if result is not None:
                loaded += 1
        except Exception as e:
            print(f"[technicals] Preload failed for {sym}: {e}")
    print(f"[technicals] Preloaded {loaded}/{len(watchlist)} tickers")


def _technicals_refresh_loop():
    """Background thread that refreshes the watchlist every hour."""
    while True:
        time.sleep(3600)  # 1 hour
        print(f"[technicals] Hourly watchlist refresh starting...")
        _preload_technicals_watchlist()


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

    # Run initial signal refresh on startup
    print("\n" + "=" * 60)
    print("  SignalAgent — Running initial data refresh...")
    print("=" * 60)
    init_thread = threading.Thread(target=_run_signal_refresh, daemon=True)
    init_thread.start()

    # Start background scheduler for auto-refresh every 15 minutes
    scheduler_thread = threading.Thread(target=_background_refresh_loop, daemon=True)
    scheduler_thread.start()

    # Also run initial framework computation
    print("  Running initial framework computation...")
    fw_thread = threading.Thread(target=_run_framework_refresh, daemon=True)
    fw_thread.start()

    # Preload technicals watchlist (after a short delay to not compete with signal refresh)
    def _delayed_technicals_preload():
        time.sleep(30)  # Wait 30s for signal engine to finish first fetches
        _preload_technicals_watchlist()

    tech_preload_thread = threading.Thread(target=_delayed_technicals_preload, daemon=True)
    tech_preload_thread.start()

    # Start hourly technicals refresh
    tech_refresh_thread = threading.Thread(target=_technicals_refresh_loop, daemon=True)
    tech_refresh_thread.start()

    print(f"\n{'='*60}")
    print("  SignalAgent Market Pulse Dashboard")
    print("=" * 60)
    print(f"  Dashboard:     http://localhost:{port}/")
    print(f"  Ticker Search: http://localhost:{port}/search.html")
    print(f"  Framework:     http://localhost:{port}/framework.html")
    print(f"  API Endpoint:  http://localhost:{port}/api/ticker/<SYMBOL>")
    print(f"  Refresh:       POST http://localhost:{port}/api/refresh")
    print(f"  Framework:     POST http://localhost:{port}/api/framework/run")
    print(f"  Auto-refresh:  Every {REFRESH_INTERVAL_MINUTES} minutes")
    print(f"")
    print(f"  Public JSON API (CORS-enabled):")
    print(f"    GET http://localhost:{port}/api/framework/latest.json")
    print(f"    GET http://localhost:{port}/api/framework/history.json")
    print(f"    GET http://localhost:{port}/api/framework/gauges.json")
    print(f"    GET http://localhost:{port}/api/framework/leaders.json")
    print(f"    GET http://localhost:{port}/api/technicals/<TICKER>.json")
    print(f"    GET http://localhost:{port}/api/technicals/batch.json?tickers=SMH,QTUM")
    print("=" * 60 + "\n")
    app.run(host="0.0.0.0", port=port, debug=False)
