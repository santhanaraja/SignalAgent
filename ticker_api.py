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
    run_engine,
    NumpyEncoder,
)
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
    print("=" * 60 + "\n")
    app.run(host="0.0.0.0", port=port, debug=False)
