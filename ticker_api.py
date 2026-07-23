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
    simulate_score,
    MACD_STATES,
    QUALIFIER_GATE,
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
from sentiment_engine import get_symbol_sentiment, technical_sentiment, FACTORS
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


def _group_breaker_context(symbol):
    """
    Group + breaker context for a symbol (PER-509).

    Membership comes from the active universe artifact (authoritative for
    the week); the group's live breaker status comes from the latest
    signals.json group entry — breakers are group-level and only computed
    by the engine run. Returns None when the symbol has no group in the
    active universe (caller then keeps breaker_status="clear" but must
    say so explicitly).
    """
    try:
        ua_path = os.path.join(DATA_DIR, "universe_active.json")
        if not os.path.exists(ua_path):
            return None
        with open(ua_path, "r") as f:
            groups = json.load(f).get("groups", {})
        group_name = None
        for gname, g in groups.items():
            if symbol in (g.get("tickers") or []):
                group_name = gname
                break
        if group_name is None:
            return None
        ctx = {"group": group_name, "breaker_status": "clear",
               "breaker_reasons": []}
        signals_path = os.path.join(DATA_DIR, "signals.json")
        if os.path.exists(signals_path):
            with open(signals_path, "r") as f:
                for g in json.load(f).get("groups", []):
                    if g.get("name") == group_name:
                        ctx["breaker_status"] = g.get("breaker_status", "clear")
                        ctx["breaker_reasons"] = [
                            a.get("message", a.get("description", ""))
                            for a in (g.get("breaker_alerts") or [])
                            if a.get("triggered")
                        ][:3]
                        break
        return ctx
    except Exception:
        return None


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

    # Trade signal with REAL group breaker context when the symbol belongs
    # to a selected group (PER-509 fix 1: hardcoded "clear" made J show
    # BUY NOW here while the dashboard showed AVOID from the same function)
    group_ctx = _group_breaker_context(symbol)
    breaker_status = group_ctx["breaker_status"] if group_ctx else "clear"
    trade_sig, trade_reason = compute_trade_signal(details, breaker_status=breaker_status)
    swing_signal = compute_swing_trade_signal(details, df)
    intraday_signal = compute_intraday_trade_signal(details, df)
    stage_analysis = compute_stage_analysis(details, df)

    result = {
        "ticker": symbol,
        "timestamp": datetime.datetime.now(datetime.timezone.utc).isoformat(),
        "price": details.get("price", 0),
        "score": score,
        "signal": signal,
        "score_components": details.get("score_components"),
        "score_inputs": details.get("score_inputs"),
        "trade_signal": trade_sig,
        "trade_reasoning": trade_reason,
        "group_context": group_ctx,
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


@app.route("/api/score/simulate", methods=["POST"])
def score_simulate():
    """Score Lab (PER-508 item 19): feed slider values through the REAL
    score_stock component functions. The page owns zero math — a drifted
    hand-mirrored formula (81 vs production 76 at YTD 209%) is why this
    endpoint exists."""
    try:
        body = request.get_json(silent=True)
    except Exception:
        # silent=True only swallows JSONDecodeError — pathological bodies
        # (e.g. thousands-deep nesting -> RecursionError) still raise
        body = None
    if not isinstance(body, dict):
        return jsonify({"status": "error", "error": "JSON body required"}), 400

    def _num_in(key, lo, hi):
        v = body.get(key)
        if isinstance(v, bool) or not isinstance(v, (int, float)):
            raise ValueError(f"{key} must be a number")
        if not (lo <= v <= hi):
            raise ValueError(f"{key} out of range [{lo}, {hi}]")
        return float(v)

    try:
        rsi = _num_in("rsi", 0, 100)
        ytd_pct = _num_in("ytd_pct", -100, 10000)
        vol_ratio = _num_in("vol_ratio", 0, 100)
        macd_state = body.get("macd_state")
        # isinstance first: `in` on a dict hashes the candidate, and a JSON
        # list/dict here raised TypeError -> 500 (review finding)
        if not isinstance(macd_state, str) or macd_state not in MACD_STATES:
            raise ValueError(
                f"macd_state must be one of {sorted(MACD_STATES)}")
        toggles = {}
        for key in ("above_ma20", "above_ma50", "ma20_gt_ma50"):
            if not isinstance(body.get(key), bool):
                raise ValueError(f"{key} must be a boolean")
            toggles[key] = body[key]
    except ValueError as e:
        return jsonify({"status": "error", "error": str(e)}), 400

    score, band, components = simulate_score(
        rsi, macd_state, toggles["above_ma20"], toggles["above_ma50"],
        toggles["ma20_gt_ma50"], ytd_pct, vol_ratio)
    return jsonify({
        "status": "success",
        "score": score,
        "band": band,
        "score_components": components,
        "gate_distance": score - QUALIFIER_GATE,
    })


_REGIME_CFG = None


def _regime_cfg():
    """Action lines + voter thresholds from framework config — the same
    values production passes to the shared functions, so a config change
    can never silently diverge the lab from the live gauge."""
    global _REGIME_CFG
    if _REGIME_CFG is None:
        import yaml
        cfg_path = os.path.join(os.path.dirname(os.path.abspath(__file__)),
                                "framework", "config.yaml")
        with open(cfg_path) as f:
            cfg = yaml.safe_load(f)
        gauges = cfg["regime"]["gauges"]
        pos = cfg.get("positions", {}) or {}
        chassis = cfg["regime"].get("chassis") or {}
        _REGIME_CFG = {
            "actions": {s["name"]: s.get("action", "")
                        for s in cfg["regime"]["states"]},
            "vix": (gauges["vix_5d_avg"]["risk_on_threshold"],
                    gauges["vix_5d_avg"]["caution_threshold"]),
            "hy": (gauges["hy_spread"]["risk_on_threshold"],
                   gauges["hy_spread"]["caution_threshold"]),
            # Gauge B chassis (D-008): engine flag + locked calibration —
            # the same values production passes to the shared chassis step
            "engine": cfg["regime"].get("engine", "chassis"),
            "regime_raw": cfg["regime"],
            "hy_cfg": gauges["hy_spread"],
            "chassis": {
                "n": int(chassis.get("hysteresis_n", 2)),
                "mode": chassis.get("hysteresis_mode", "asymmetric"),
                "vix_thr": float(chassis.get("vix_throttle", 22.0)),
                "hy_cut": float(chassis.get("hy_pctile_cut", 90.0)),
                "hy_window": int(chassis.get("hy_pctile_window", 60)),
                "breadth_thr": float(chassis.get("breadth_throttle", -0.5)),
                "require_k": int(chassis.get("require_k", 1)),
            },
            # Position Lab (24b): the same config values the 1B engine
            # passes to assess_position
            "positions": {
                "confirmation_closes": pos.get("confirmation_closes", 2),
                "atr_mult": pos.get("atr_mult", 0.5),
                "extension_guard_max": pos.get("extension_guard_max", 1.8),
                "slope_lookback_days": pos.get("slope_lookback_days", 5),
            },
            # D-011 A+ thresholds — same config values the engine grades
            # with, so the lab can never drift from production (law 1)
            "aplus": {
                "rsi_min": (pos.get("aplus", {}) or {}).get("rsi_min", 45.0),
                "rsi_max": (pos.get("aplus", {}) or {}).get("rsi_max", 70.0),
                "score_min": (pos.get("aplus", {}) or {}).get("score_min", 75.0),
                "runway_min_sessions": (pos.get("aplus", {}) or {}).get(
                    "runway_min_sessions", 15),
            },
        }
    return _REGIME_CFG


def _regime_actions():
    return _regime_cfg()["actions"]


# Gauge Lab probing domains — SEARCH bounds only (where to look for a vote
# flip), not thresholds. Every vote decision inside comes from the real
# compute_regime; no threshold lives here (Score Lab law 1).
_GAUGE_LAB_DOMAINS = {
    "vix_5d": (0.0, 100.0),
    "hy_oas": (0.0, 30.0),
    "breadth_20d": (-20.0, 20.0),
    "spy_vs_200dma_pct": (-50.0, 50.0),
}


def _regime_probe(inputs):
    """compute_regime on an inputs dict, with the CONFIG thresholds —
    exactly what production passes (the breadth band is production's
    function default; its config entries belong to the unused S5FI basis)."""
    from framework.regime_calculator import compute_regime
    cfg = _regime_cfg()
    return compute_regime(inputs["vix_5d"], inputs["hy_oas"],
                          inputs["breadth_20d"],
                          inputs["spy_vs_200dma_pct"],
                          vix_thresholds=cfg["vix"],
                          oas_thresholds=cfg["hy"])


def _gauge_signature(result, key):
    """What 'this input's vote' means per input: the gauge vote for the
    three voters, the gate open/closed for the SPY input."""
    if key == "spy_vs_200dma_pct":
        return "gate_open" if result["gate"]["open"] else "gate_closed"
    gauge = {"vix_5d": "vix_5d_avg", "hy_oas": "hy_spread",
             "breadth_20d": "breadth"}[key]
    return result["votes"][gauge]


# --- Gauge B chassis lab plumbing (D-008) ----------------------------------

def _chassis_oas_tail():
    """The lab's OAS->percentile basis, served from the PERSISTED chassis
    record (written by the last engine run) — the request path performs ZERO
    external fetches (perf fix: FRED inside the probe loop was the 34s cold
    path). Returns (tail_values, hy_window) or (None, None) when the record
    predates the field or the last run used the HYG/IEF fallback basis (a
    what-if OAS print can't be ranked against a ratio window — the lab reads
    hy unavailable, honestly, rather than faking a percentile)."""
    import json as _json
    path = os.path.join(os.path.dirname(os.path.abspath(__file__)),
                        "framework", "state", "regime_chassis_state.json")
    try:
        with open(path) as f:
            rec = _json.load(f)
        tail = rec.get("oas_window_tail")
        window = int(rec.get("hy_window") or 0)
        if isinstance(tail, list) and window > 1 and len(tail) >= window - 1:
            return [float(v) for v in tail], window
    except (OSError, ValueError, TypeError):
        pass
    return None, None


def _chassis_ctx():
    """Request-scoped chassis context, built ONCE per simulate request and
    passed to every flip-distance probe: the persisted carry, the persisted
    OAS window, and the config. The probe loop runs ~500 evaluations per
    request — per-probe file reads put a 500x multiplier on disk latency
    (the ~2s warm-request cost on Render; perf fix)."""
    tail, window = _chassis_oas_tail()
    return {"carry": _chassis_carry(), "oas_tail": tail,
            "oas_window": window, "ccfg": _regime_cfg()["chassis"]}


def _chassis_carry():
    """The persisted hysteresis carry (read-only) the lab steps FROM. Uses
    the record's carry_pre_final — the carry production held BEFORE today's
    replay step — so an untouched seeded lab reproduces today's confirmed
    state instead of double-stepping the current bar (review finding).
    Falls back to the post-step counters for records predating the field."""
    import json as _json
    path = os.path.join(os.path.dirname(os.path.abspath(__file__)),
                        "framework", "state", "regime_chassis_state.json")
    try:
        with open(path) as f:
            rec = _json.load(f)
        from framework.regime_calculator import CHASSIS_RANK
        pre = rec.get("carry_pre_final")
        if isinstance(pre, dict) and pre.get("confirmed") in CHASSIS_RANK:
            return {"confirmed": pre["confirmed"],
                    "up": int(pre.get("up", 0)),
                    "down": int(pre.get("down", 0))}
        if rec.get("confirmed") in CHASSIS_RANK:
            return {"confirmed": rec["confirmed"],
                    "up": int(rec.get("up", 0)),
                    "down": int(rec.get("down", 0))}
    except (OSError, ValueError, TypeError):
        pass
    from framework.regime_calculator import new_chassis_hysteresis
    return new_chassis_hysteresis()


def _chassis_probe(inputs, ctx=None):
    """One chassis step on the lab inputs through the REAL production
    functions: hy_oas -> percentile against the PERSISTED trailing OAS window
    (production's own pctile_of_last), raw state via chassis_raw_state,
    confirmed via chassis_step from the persisted carry. The lab owns zero
    math (law 1) and performs zero fetches: it simulates "what would the
    chassis say with these inputs given the CURRENT persisted window" — the
    window and carry come from the last engine run's state record, via the
    request-scoped ctx (built once; ~500 probes reuse it)."""
    from framework.regime_calculator import (
        chassis_step, pctile_of_last, CHASSIS_TO_REGIME)
    if ctx is None:
        ctx = _chassis_ctx()
    ccfg = ctx["ccfg"]

    hy_pctile = None
    hy_stress = False
    hy_basis = "unavailable"
    if inputs["hy_oas"] is not None and ctx["oas_tail"] is not None:
        # "if today's OAS printed X, given the persisted trailing window" —
        # a WHAT-IF framing, deliberately one observation ahead of the live
        # path (production's shift(1) window is [T-60..T-1] and never
        # contains today's print). At a pctile sitting exactly on the cut
        # this can differ from the live block by ~1/60 pctile pt. The window
        # is as of the last engine run (record as_of), not refetched.
        window = ctx["oas_window"]
        window_vals = ctx["oas_tail"][-(window - 1):] + [inputs["hy_oas"]]
        hy_pctile = pctile_of_last(window_vals, window)
        hy_stress = hy_pctile is not None and hy_pctile >= ccfg["hy_cut"]
        hy_basis = "fred_oas_pctile"

    trend = inputs["spy_vs_200dma_pct"]
    vix = inputs["vix_5d"]
    breadth = inputs["breadth_20d"]
    # unavailable inputs fail toward the conservative side of each role:
    # trend unknown -> out-of-trend (fail closed, parliament convention);
    # vix/breadth unknown -> throttle can't prove stress -> not firing
    trend_v = trend if trend is not None else -1.0
    vix_v = vix if vix is not None else 0.0
    breadth_v = breadth if breadth is not None else 0.0

    r = chassis_step(trend_v, vix_v, hy_stress, breadth_v, ctx["carry"],
                     n=ccfg["n"], mode=ccfg["mode"], vix_thr=ccfg["vix_thr"],
                     breadth_thr=ccfg["breadth_thr"],
                     require_k=ccfg["require_k"])
    return {
        "raw_state": r["raw_state"],
        "confirmed_state": r["state"],
        "state": CHASSIS_TO_REGIME[r["state"]],
        "exposure_ceiling_pct": r["exposure"] * 100.0,
        "trend_in": bool(trend_v >= 0),
        "throttles": {
            "vix": {"firing": bool(vix_v >= ccfg["vix_thr"]),
                    "cut": ccfg["vix_thr"],
                    "available": vix is not None},
            "hy": {"firing": bool(hy_stress), "pctile": hy_pctile,
                   "cut": ccfg["hy_cut"], "basis": hy_basis,
                   "available": hy_basis != "unavailable"},
            "breadth": {"firing": bool(breadth_v < ccfg["breadth_thr"]),
                        "cut": ccfg["breadth_thr"],
                        "available": breadth is not None},
        },
        "hysteresis": {**r["hysteresis_state"], "n": ccfg["n"],
                       "mode": ccfg["mode"]},
    }


def _chassis_signature(result, key):
    """Per-input chassis signature: the trend direction for SPY, the
    throttle firing/clear for the three modifiers."""
    if key == "spy_vs_200dma_pct":
        return "in_trend" if result["trend_in"] else "out_of_trend"
    t = {"vix_5d": "vix", "hy_oas": "hy", "breadth_20d": "breadth"}[key]
    return "throttle_firing" if result["throttles"][t]["firing"] else "clear"


def _nearest_flip(inputs, key, coarse=0.25, tol=1e-3,
                  probe_fn=None, sig_fn=None):
    """Distance from inputs[key] to the nearest value that changes that
    input's vote/gate (parliament) or throttle/trend signature (chassis) —
    found by PROBING the real regime function (coarse scan out from the
    current value, then bisection), never by reading thresholds.
    Returns None if no flip inside the search domain, else a dict with the
    signed distance, boundary, resulting signature, and whether the STATE
    flips. probe_fn/sig_fn default to the parliament pair.
    """
    probe_fn = probe_fn or _regime_probe
    sig_fn = sig_fn or _gauge_signature
    base = probe_fn(inputs)
    base_sig = sig_fn(base, key)
    lo, hi = _GAUGE_LAB_DOMAINS[key]
    v0 = inputs[key]
    candidates = []
    for direction in (1.0, -1.0):
        limit = hi if direction > 0 else lo
        prev = v0
        cur = v0
        found = None
        while (cur < limit) if direction > 0 else (cur > limit):
            cur = min(cur + coarse, hi) if direction > 0 \
                else max(cur - coarse, lo)
            probe = dict(inputs, **{key: cur})
            if sig_fn(probe_fn(probe), key) != base_sig:
                found = (prev, cur)
                break
            prev = cur
        if not found:
            continue
        a, b = found                       # same-vote side, flipped side
        while abs(b - a) > tol:
            mid = (a + b) / 2.0
            probe = dict(inputs, **{key: mid})
            if sig_fn(probe_fn(probe), key) != base_sig:
                b = mid
            else:
                a = mid
        flipped = probe_fn(dict(inputs, **{key: b}))
        candidates.append({
            "distance": round(b - v0, 3),
            "boundary_value": round(b, 3),
            "from": base_sig,
            "to": sig_fn(flipped, key),
            "state_if_flipped": flipped["state"],
            "state_flips": flipped["state"] != base["state"],
        })
    if not candidates:
        return None
    return min(candidates, key=lambda c: abs(c["distance"]))


@app.route("/api/regime/simulate", methods=["POST"])
def regime_simulate():
    """Gauge Lab (PER-508 item 24a): slider values through the REAL
    compute_regime (Build 4 extraction — production delegates to the same
    function). The lab owns zero math; even the flip distances are found
    by probing this function, not by hardcoding thresholds."""
    try:
        body = request.get_json(silent=True)
    except Exception:
        body = None
    if not isinstance(body, dict):
        return jsonify({"status": "error", "error": "JSON body required"}), 400

    inputs = {}
    try:
        for key, (lo, hi) in _GAUGE_LAB_DOMAINS.items():
            v = body.get(key)
            if v is None:
                inputs[key] = None          # simulates an unavailable gauge
                continue
            if isinstance(v, bool) or not isinstance(v, (int, float)):
                raise ValueError(f"{key} must be a number or null")
            if not (lo <= v <= hi):
                raise ValueError(f"{key} out of range [{lo}, {hi}]")
            inputs[key] = float(v)
    except ValueError as e:
        return jsonify({"status": "error", "error": str(e)}), 400

    if _regime_cfg()["engine"] == "chassis":
        # Gauge B (D-008): the lab reflects the LIVE chassis — raw state
        # from the inputs, confirmed via the persisted hysteresis carry.
        # ONE context for the whole request (carry + persisted OAS window +
        # config): the ~500 flip probes below reuse it — zero fetches, zero
        # per-probe file reads (perf fix: was ~2s warm / 34s cold on Render).
        ctx = _chassis_ctx()
        probe = lambda p: _chassis_probe(p, ctx)  # noqa: E731
        result = probe(inputs)
        flips = {key: (None if inputs[key] is None
                       else _nearest_flip(inputs, key,
                                          probe_fn=probe,
                                          sig_fn=_chassis_signature))
                 for key in _GAUGE_LAB_DOMAINS}
        return jsonify({
            "status": "success",
            "engine": "chassis",
            "state": result["state"],
            "action_line": _regime_actions().get(result["state"], ""),
            "chassis": {
                "raw_state": result["raw_state"],
                "confirmed_state": result["confirmed_state"],
                "exposure_ceiling_pct": result["exposure_ceiling_pct"],
                "trend_in": result["trend_in"],
                "throttles": result["throttles"],
                "hysteresis": result["hysteresis"],
            },
            "flip_distances": flips,
        })

    result = _regime_probe(inputs)
    votes = result["votes"]
    counts = {s: sum(1 for v in votes.values() if v == s)
              for s in ("risk_on", "caution", "risk_off", "unavailable")}
    flips = {key: (None if inputs[key] is None
                   else _nearest_flip(inputs, key))
             for key in _GAUGE_LAB_DOMAINS}
    return jsonify({
        "status": "success",
        "engine": "parliament",
        "state": result["state"],
        "action_line": _regime_actions().get(result["state"], ""),
        "votes": votes,
        "counts": counts,
        "gate": result["gate"],
        "flip_distances": flips,
    })


@app.route("/api/position/simulate", methods=["POST"])
def position_simulate():
    """Position Lab (PER-508 item 24b): field values through the REAL
    assess_position — the extracted 1B condition evaluators + extension
    guard the production engine delegates to. Zero math here."""
    try:
        body = request.get_json(silent=True)
    except Exception:
        body = None
    if not isinstance(body, dict):
        return jsonify({"status": "error", "error": "JSON body required"}), 400

    def _num(key, lo, hi, allow_null=False):
        v = body.get(key)
        if v is None and allow_null:
            return None
        if isinstance(v, bool) or not isinstance(v, (int, float)):
            raise ValueError(f"{key} must be a number")
        if not (lo <= v <= hi):
            raise ValueError(f"{key} out of range [{lo}, {hi}]")
        return float(v)

    try:
        close = _num("close", 0.01, 1e6)
        sma20 = _num("sma20", 0.01, 1e6)
        sma20_5d_ago = _num("sma20_5d_ago", 0.01, 1e6)
        atr14 = _num("atr14", 0.0, 1e5, allow_null=True)
        # subnormal ATR (e.g. 1e-320) divides to Infinity, which json.dumps
        # emits as a bare token browsers cannot parse (review finding);
        # anything below a tenth of a cent is "no ATR"
        if atr14 is not None and atr14 < 1e-4:
            atr14 = None
        consec = body.get("consecutive_closes_above")
        if isinstance(consec, bool) or not isinstance(consec, int) \
                or not (0 <= consec <= 1000):
            raise ValueError("consecutive_closes_above must be an integer 0-1000")
        regime_state = body.get("regime_state")
        if not isinstance(regime_state, str) or len(regime_state) > 50:
            raise ValueError("regime_state must be a string")
        # D-007 Phase 1: canonical key is group_in_universe; the legacy
        # theme_qualified key is accepted as an alias (same bool semantics)
        group_in_universe = body.get("group_in_universe",
                                     body.get("theme_qualified"))
        if not isinstance(group_in_universe, bool):
            raise ValueError("group_in_universe must be a boolean")
        kind = body.get("kind")
        if kind not in ("holding", "watching"):
            raise ValueError("kind must be 'holding' or 'watching'")

        # Optional D-011 grade inputs — the grade section computes only
        # when any of these keys is present (the lab's grade panel)
        GRADE_KEYS = ("sma5", "up_close_since_swing_low", "rsi14",
                      "quality_score", "score_waived", "breaker_status",
                      "runway_sessions")
        want_grade = any(k in body for k in GRADE_KEYS)
        if want_grade:
            sma5 = _num("sma5", 0.01, 1e6, allow_null=True)
            rsi14 = _num("rsi14", 0.0, 100.0, allow_null=True)
            quality_score = _num("quality_score", 0.0, 100.0, allow_null=True)
            up_close = body.get("up_close_since_swing_low", False)
            if not isinstance(up_close, bool):
                raise ValueError("up_close_since_swing_low must be a boolean")
            score_waived = body.get("score_waived", False)
            if not isinstance(score_waived, bool):
                raise ValueError("score_waived must be a boolean")
            breaker_status = body.get("breaker_status")
            if breaker_status is not None and breaker_status not in (
                    "clear", "watch", "warning", "critical"):
                raise ValueError("breaker_status must be clear/watch/warning/"
                                 "critical or null")
            runway = body.get("runway_sessions")
            if runway is not None and (isinstance(runway, bool)
                                       or not isinstance(runway, int)
                                       or not (0 <= runway <= 1000)):
                raise ValueError("runway_sessions must be an integer 0-1000 "
                                 "or null")
    except ValueError as e:
        return jsonify({"status": "error", "error": str(e)}), 400

    from framework.position_signals import assess_position, grade_setup
    result = assess_position(close, sma20, sma20_5d_ago, atr14, consec,
                             regime_state, group_in_universe, kind,
                             **_regime_cfg()["positions"])
    if want_grade:
        ap = _regime_cfg()["aplus"]
        # row 2 uses the UNROUNDED extension (the guard's own comparison) —
        # the rounded display field disagrees exactly at the 1.8 boundary
        ext_raw = None if not atr14 else (close - sma20) / atr14
        grade = grade_setup(
            all_conditions_met=result["all_conditions_met"],
            extension_atr=ext_raw,
            close=close, sma5=sma5,
            up_close_since_swing_low=up_close,
            rsi14=rsi14, quality_score=quality_score,
            score_waived=score_waived, breaker_status=breaker_status,
            runway_sessions=runway,
            extension_guard_max=_regime_cfg()["positions"].get(
                "extension_guard_max", 1.8),
            rsi_min=ap["rsi_min"], rsi_max=ap["rsi_max"],
            score_min=ap["score_min"],
            runway_min_sessions=ap["runway_min_sessions"])
        result["grade"] = grade
        # Q4 enforcement preview, same rule the engine applies
        if result.get("a_plus_only") and grade["grade"] != "A+":
            result["grade_gate"] = (
                f"READY blocked — grade {grade['grade']} under Choppy "
                f"(A+ required): {grade['reasons'] or 'see rows'}")
    return app.response_class(
        response=json.dumps({"status": "success", **result},
                            cls=NumpyEncoder),
        mimetype="application/json")


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
@app.route("/api/sentiment/simulate", methods=["POST"])
def sentiment_simulate():
    """Sentiment Lab (Lab law 1, D-010): feed the five behavioural factors
    through the REAL technical_sentiment() aggregator. The page/future lab owns
    zero math — same no-drift guarantee as the Score and Gauge labs."""
    try:
        body = request.get_json(silent=True)
    except Exception:
        body = None
    if not isinstance(body, dict):
        return jsonify({"status": "error", "error": "JSON body required"}), 400

    components = {}
    for f in FACTORS:
        v = body.get(f)
        if isinstance(v, bool) or not isinstance(v, (int, float)):
            return jsonify({"status": "error", "error": f"{f} must be a number in [0, 100]"}), 400
        if not (0 <= v <= 100):
            return jsonify({"status": "error", "error": f"{f} out of range [0, 100]"}), 400
        components[f] = float(v)

    result = technical_sentiment(components)
    return jsonify({"status": "success", **result})


@app.route("/api/sentiment/<symbol>")
def sentiment_symbol(symbol):
    """Per-ticker behavioural sentiment: Technical Sentiment + Relative
    Strength + news strip (D-013)."""
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
            # D-017: the engine rewrite just produced a signals.json whose
            # rows carry fresh grade_inputs but NO candidate annotation —
            # chase it with a framework pass (the same engine->framework
            # ordering CI uses) so the candidate chips re-grade against
            # the fresh rows instead of honestly vanishing for the whole
            # refresh interval. Fire-and-forget; its own running guard
            # dedupes overlapping kicks.
            _kick_framework_refresh()
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


# ------------------------------------------------------------------
# Serve-layer shape sentinel (stale-serve fix): a regime endpoint must
# never serve a shape the current code cannot produce. framework.json is
# baked into every deploy from the last CI commit and can predate the
# current schema (the pre-1A 5-voter artifact served 2026-07-06 ~9:15 ET
# between Render cold-boot and the first boot refresh). Validity is
# STRUCTURAL — the "schema" tag stamped by framework_runner is
# informational/versioning; the invariants below decide.
# ------------------------------------------------------------------
def FRAMEWORK_SCHEMA():
    """The schema tag the configured engine produces (informational; the
    structural invariants below decide validity)."""
    from framework.regime_calculator import artifact_schema
    return artifact_schema(_regime_cfg()["regime_raw"])


_REGIME_VOTERS = {"vix_5d_avg", "hy_spread", "breadth"}
FRAMEWORK_STALE_AFTER_HOURS = 6   # age flag threshold — informational, never a 503


def _framework_shape_valid(data):
    """Structural invariants of the current regime output schema. Includes
    the ENGINE invariant (D-008 cutover): an artifact produced by the other
    regime engine — e.g. a deploy-baked parliament artifact under a chassis
    server — is schema-stale and must regenerate, never serve as current
    (the R28 artifact-baking lesson)."""
    regime = (data or {}).get("regime") or {}
    base_ok = (
        isinstance(regime.get("backdrop_gate"), dict)
        and isinstance(regime.get("macro_inputs"), dict)
        and set((regime.get("gauges") or {}).keys()) == _REGIME_VOTERS
    )
    if not base_ok:
        return False
    if _regime_cfg()["engine"] == "chassis":
        return regime.get("engine") == "chassis" \
            and isinstance(regime.get("chassis"), dict)
    # parliament: legacy artifacts carry no engine key — accept absent
    return regime.get("engine") != "chassis"


def _framework_stale_hours(data):
    """Age in hours when past the freshness threshold, else None.
    Weekend/overnight age is legitimate (Friday close IS the latest
    close) — this flags it, it never blocks serving."""
    try:
        gen = datetime.datetime.fromisoformat(str(data.get("generated_at")))
        if gen.tzinfo is None:
            gen = gen.replace(tzinfo=datetime.timezone.utc)
        age = (datetime.datetime.now(datetime.timezone.utc) - gen).total_seconds() / 3600
        return round(age, 1) if age > FRAMEWORK_STALE_AFTER_HOURS else None
    except (TypeError, ValueError):
        return None


def _kick_framework_refresh():
    """Fire-and-forget refresh; _run_framework_refresh's running guard
    makes duplicate kicks harmless."""
    if not _framework_status["running"]:
        threading.Thread(target=_run_framework_refresh, daemon=True).start()


def _load_framework_payload():
    """
    (payload, error_response) for the current-state framework endpoints.
    Missing file -> 404. Unparseable or shape-invalid -> kick a refresh
    and 503 with Retry-After — the baked artifact predates the current
    schema and must not be served. A valid payload past the freshness
    threshold gets an informational top-level stale_hours field.
    """
    framework_path = os.path.join(PUBLIC_DIR, "framework.json")
    if not os.path.exists(framework_path):
        return None, app.response_class(
            response=json.dumps({
                "status": "error",
                "error": "Framework has not been run yet.",
                "hint": "POST /api/framework/run to trigger a run.",
            }),
            status=404,
            mimetype="application/json",
        )
    try:
        with open(framework_path, "r") as f:
            data = json.load(f)
    except Exception:
        data = None
    if not data or not _framework_shape_valid(data):
        _kick_framework_refresh()
        resp = app.response_class(
            response=json.dumps({
                "status": "warming_up",
                "error": "regime state regenerating — the stored artifact "
                         "predates the current schema (fresh computation "
                         "kicked off; typically ready within ~3 minutes)",
                "retry_after": 180,
                "framework_status": {
                    "running": _framework_status.get("running"),
                    "last_run": _framework_status.get("last_run"),
                    "last_error": _framework_status.get("last_error"),
                },
            }, cls=NumpyEncoder),
            status=503,
            mimetype="application/json",
        )
        resp.headers["Retry-After"] = "180"
        return None, resp
    stale = _framework_stale_hours(data)
    if stale is not None:
        data = dict(data)
        data["stale_hours"] = stale
    return data, None


@app.route("/api/framework/latest")
def framework_latest():
    """Return the latest framework output (regime + themes + rules).
    503 while a schema-stale artifact is being regenerated; stale_hours
    flags valid-but-old data (e.g. Friday close served on a Sunday)."""
    data, err = _load_framework_payload()
    if err is not None:
        return err
    try:
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
    No wrapper, no status field — just the raw framework object
    (plus stale_hours when past the freshness threshold). 503 while a
    schema-stale artifact is being regenerated.
    """
    data, err = _load_framework_payload()
    if err is not None:
        return err
    return app.response_class(
        response=json.dumps(data, cls=NumpyEncoder),
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
    503 while a schema-stale artifact is being regenerated (shape
    sentinel); stale_hours flags valid-but-old data.
    """
    data, err = _load_framework_payload()
    if err is not None:
        return err
    try:
        regime = data.get("regime", {})
        compact = {
            "generated_at": data.get("generated_at"),
            "regime": regime.get("regime"),
            "regime_action": regime.get("action"),
            # One-grammar retirement (Phase 3): the chassis sets the regime
            # (D-008); the parliament vote counts are informational leftovers.
            # The flat *_count fields remain ONE deprecation cycle for API
            # consumers, then go — read legacy_voters instead.
            "risk_on_count": regime.get("risk_on_count"),
            "caution_count": regime.get("caution_count"),
            "risk_off_count": regime.get("risk_off_count"),
            "legacy_voters": {
                "risk_on": regime.get("risk_on_count"),
                "caution": regime.get("caution_count"),
                "risk_off": regime.get("risk_off_count"),
                "note": "parliament-era vote counts — informational only; "
                        "the chassis sets the regime (D-008). The flat "
                        "*_count fields are deprecated and will be removed "
                        "after a cycle.",
            },
            "consecutive_weeks": regime.get("consecutive_weeks_at_state"),
            "regime_change_pending": regime.get("regime_change_pending", False),
            "gauges": regime.get("gauges", {}),
            "backdrop_gate": regime.get("backdrop_gate"),
            "macro_inputs": regime.get("macro_inputs"),
        }
        if "stale_hours" in data:
            compact["stale_hours"] = data["stale_hours"]

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


@app.route("/api/framework/signals.json")
def framework_signals_json():
    """
    Public JSON API — position signal engine output (Build 1B).
    Distinct from the dashboard's /signals.json (stock scoring): this is
    the per-ticker exit/re-entry state machine over positions.json.

    Response shape:
    {
      "generated_at": "...",
      "regime_state": "Risk-on / Trending",
      "tickers": {
        "IWM": { "state": "RE_ENTRY_READY", "close": ..., "sma20": ...,
                 "conditions": { "1_trigger": {...}, ... "5_thesis": {...} },
                 "conditions_met": 5, "stop": {...} },
        ...
      },
      "transitions": [ ...today's state-change events... ]
    }
    503 while a schema-stale artifact is being regenerated (shape
    sentinel); stale_hours flags valid-but-old data.
    """
    data, err = _load_framework_payload()
    if err is not None:
        return err
    try:
        pos = data.get("position_signals")
        if not pos:
            return app.response_class(
                response=json.dumps({
                    "error": "No position signals in the latest framework run.",
                    "hint": "Re-run the framework (POST /api/framework/run).",
                }),
                status=404,
                mimetype="application/json",
            )
        if "stale_hours" in data:
            pos = dict(pos)
            pos["stale_hours"] = data["stale_hours"]
        return app.response_class(
            response=json.dumps(pos, cls=NumpyEncoder),
            mimetype="application/json",
        )
    except Exception as e:
        return app.response_class(
            response=json.dumps({"error": str(e)}),
            status=500,
            mimetype="application/json",
        )


# ------------------------------------------------------------------
# Consolidated daily assessment (PER-508 item 21): one endpoint, the
# whole daily read. Read-only AGGREGATION of already-computed artifacts
# (framework.json, signals.json, regime_history.json,
# position_events.json) — never refetches from Yahoo. Sections degrade
# independently to {"error": ...}; the regime section rides
# _load_framework_payload, so the shape sentinel's 503-until-valid
# applies to the whole endpoint.
# ------------------------------------------------------------------

_ASSESSMENT_VOTERS = ("vix_5d_avg", "hy_spread", "breadth")


def _assessment_r28(data):
    """R28 real-dollar enforcement block (PER-508 Phase 0) — passthrough
    of the framework artifact's computed block; no recomputation here."""
    r28 = data.get("r28")
    if not isinstance(r28, dict):
        return {"error": "r28 block absent from framework artifact "
                         "(pre-R28 artifact — regenerates on next run)"}
    return r28


def _assessment_regime(data):
    regime = data.get("regime") or {}
    out = {
        "regime": regime.get("regime"),
        "action": regime.get("action"),
        # Gauge B (D-008): engine + the chassis readout (raw vs confirmed,
        # throttles, hysteresis) ride the assessment like every other layer
        "engine": regime.get("engine"),
        "chassis": regime.get("chassis"),
        # deprecated flat counts (one cycle) + the annotated block —
        # see /api/regime's legacy_voters note (one-grammar retirement)
        "risk_on_count": regime.get("risk_on_count"),
        "caution_count": regime.get("caution_count"),
        "risk_off_count": regime.get("risk_off_count"),
        "legacy_voters": {
            "risk_on": regime.get("risk_on_count"),
            "caution": regime.get("caution_count"),
            "risk_off": regime.get("risk_off_count"),
            "note": "parliament-era vote counts — informational only; "
                    "the chassis sets the regime (D-008). Deprecated flat "
                    "*_count fields will be removed after a cycle.",
        },
        "gauges": regime.get("gauges", {}),
        "backdrop_gate": regime.get("backdrop_gate"),
        "macro_inputs": regime.get("macro_inputs"),
        "consecutive_degraded_weeks": regime.get("consecutive_degraded_weeks"),
        "intra_week_override": regime.get("intra_week_override"),
    }
    if "stale_hours" in data:
        out["stale_hours"] = data["stale_hours"]
    return out


def _assessment_positions(data):
    ps = data.get("position_signals") or {}
    if ps.get("error"):
        return {"error": ps["error"]}
    out = {}
    for t, x in (ps.get("tickers") or {}).items():
        row = {
            "state": x.get("state"),
            "kind": x.get("kind"),
            "theme": x.get("theme"),
            "close": x.get("close"),
            "sma20": x.get("sma20"),
            "stop": x.get("stop"),
            "extension_pct": x.get("extension_pct"),
            "extension_atr": x.get("extension_atr"),
            "next_earnings_date": x.get("next_earnings_date"),
            "days_to_earnings": x.get("days_to_earnings"),
        }
        for opt in ("earnings_note", "distance_to_sma20_pct", "a_plus_only",
                    "extension_guard", "conditions_met", "insufficient_data",
                    "grade", "grade_gate", "grade_inputs", "group",
                    "weeks_in_universe"):
            if x.get(opt) is not None:
                row[opt] = x[opt]
        # conditions itemized only for non-HELD names (the re-entry ladder);
        # a HELD name's read is the stop, not the entry conditions
        if x.get("state") != "HELD":
            row["conditions"] = x.get("conditions")
        out[t] = row
    return out


def _assessment_candidates(data):
    """D-017 candidates tier passthrough: {ticker: {grade, reasons,
    failing, group}} for every un-tracked signals.json name. None on a
    pre-emission artifact (era-aware — the section renders null, never a
    fabricated empty tier)."""
    return data.get("candidate_grades")


def _assessment_technicals(data):
    """Full technicals per HOLDING, merged from the hourly engine's
    signals.json stock block + the position engine's own values
    (close/sma5/sma10/sma20/atr14/stop/extensions). Artifacts predating
    the sma5/sma10 producer amendment degrade to null + note."""
    ps = data.get("position_signals") or {}
    holdings = {t: x for t, x in (ps.get("tickers") or {}).items()
                if x.get("kind") == "holding"}
    if not holdings:
        return {"note": "no holdings"}

    stock_by_ticker = {}
    try:
        with open(os.path.join(PUBLIC_DIR, "signals.json"), "r") as f:
            for g in json.load(f).get("groups", []):
                for s in g.get("stocks", []):
                    stock_by_ticker[s.get("ticker")] = s
    except Exception:
        pass  # degrade to position-engine values only

    out = {}
    for t, pos in holdings.items():
        stock = stock_by_ticker.get(t)
        atr = ((pos.get("conditions") or {}).get("2_confirmation") or {}).get("atr14")
        stop = pos.get("stop") or {}
        close = pos.get("close")
        cushion = None
        if close is not None and stop.get("level") is not None:
            cushion = round(close - stop["level"], 2)
        tech = {
            "close": close,
            "price_hourly_engine": stock.get("price") if stock else None,
            "sma5": pos.get("sma5"),
            "sma10": pos.get("sma10"),
            "sma20": pos.get("sma20"),
            "sma50": stock.get("ma50") if stock else None,
            "sma200": stock.get("ma200") if stock else None,
            "atr14": atr,
            "rsi": stock.get("rsi") if stock else None,
            "macd": {
                "macd": stock.get("macd"),
                "signal": stock.get("macd_signal"),
                "histogram": stock.get("macd_histogram"),
            } if stock else None,
            "volume_ratio": stock.get("volume_ratio") if stock else None,
            "trend_strength": stock.get("trend_strength") if stock else None,
            "stop": stop or None,
            "cushion_to_stop": {
                "dollars": cushion,
                "atr_multiple": round(cushion / atr, 2)
                                if cushion is not None and atr else None,
            },
            "extension_pct": pos.get("extension_pct"),
            "extension_atr": pos.get("extension_atr"),
            "notes": [],
        }
        if pos.get("sma5") is None:
            tech["notes"].append(
                "sma5/sma10 absent — artifact predates the producer "
                "amendment (populates on the next framework run)")
        if not stock:
            tech["notes"].append(
                "not in active universe — position-engine values only "
                "(no rsi/macd/sma50/sma200 from the hourly engine)")
        out[t] = tech
    return out


def _assessment_vol(data):
    """VIX complex. VIX spot + 5d avg come from the regime voter (which
    also carries the signal); ^VIX9D/^VIX3M come from the hourly
    engine's indexes block (producer amendment). A missing/flaky symbol
    degrades to null + note. term_structure = VIX3M vs VIX spot."""
    gauges = (data.get("regime") or {}).get("gauges") or {}
    vg = gauges.get("vix_5d_avg") or {}
    idx = {}
    try:
        with open(os.path.join(PUBLIC_DIR, "signals.json"), "r") as f:
            idx = json.load(f).get("indexes") or {}
    except Exception:
        pass

    missing = ("unavailable — not in the latest hourly-engine artifact "
               "(fetch failed or artifact predates the producer amendment)")

    def vol_row(tkr):
        row = idx.get(tkr) or {}
        if row.get("level") is None:
            return {"spot": None, "avg_5d": None, "error": missing}
        return {"spot": row["level"], "avg_5d": row.get("avg_5d")}

    vix = {"spot": vg.get("spot"), "avg_5d": vg.get("value"),
           "signal": vg.get("signal")}
    if vix["spot"] is None:            # gauge unavailable → indexes fallback
        fb = vol_row("^VIX")
        vix = {"spot": fb.get("spot"), "avg_5d": fb.get("avg_5d"),
               "signal": None}
        if "error" in fb:
            vix["error"] = fb["error"]
    vix9d = vol_row("^VIX9D")
    vix3m = vol_row("^VIX3M")
    if vix3m.get("spot") is not None and vix.get("spot") is not None:
        term = "contango" if vix3m["spot"] > vix["spot"] else "inverted"
    else:
        term = "unavailable — requires VIX3M spot"
    return {"vix": vix, "vix9d": vix9d, "vix3m": vix3m,
            "term_structure": term}


def _assessment_themes(data):
    themes = data.get("themes") or {}
    active = set(themes.get("active_themes") or [])
    rows = []
    for t in themes.get("ranked_themes") or []:
        if t.get("rank") is None:
            continue
        status = ("active" if t.get("name") in active
                  else "qualified" if t["rank"] <= 2 else "ranked")
        rows.append({"rank": t["rank"], "theme": t.get("name"),
                     "composite": t.get("composite"), "status": status})
    rows.sort(key=lambda r: r["rank"])
    return rows


def _assessment_changes(data):
    """What changed vs the prior dated framework run: regime shift, voter
    flips, gate open/shut, and today's position-state transitions."""
    hist_path = os.path.join(FRAMEWORK_STATE_DIR, "regime_history.json")
    with open(hist_path, "r") as f:
        entries = [e for e in json.load(f) if e.get("gauges")]
    if len(entries) < 2:
        return {"error": "fewer than two dated runs in regime history"}
    prior, curr = entries[-2], entries[-1]

    regime_shift = None
    if prior.get("regime") != curr.get("regime"):
        regime_shift = {"from": prior.get("regime"), "to": curr.get("regime")}

    voters_flipped = []
    for v in _ASSESSMENT_VOTERS:
        p = (prior["gauges"].get(v) or {}).get("signal")
        c = (curr["gauges"].get(v) or {}).get("signal")
        if p != c:
            voters_flipped.append({"gauge": v, "from": p, "to": c})

    gate_changed = None
    pg, cg = prior.get("backdrop_gate"), curr.get("backdrop_gate")
    if pg and cg and pg.get("open") != cg.get("open"):
        gate_changed = {"from_open": pg.get("open"), "to_open": cg.get("open")}

    transitions = []
    try:
        with open(os.path.join(DATA_DIR, "position_events.json"), "r") as f:
            for ev in json.load(f).get("changes", []):
                if str(ev.get("timestamp", "")).startswith(curr.get("date", "")):
                    d = ev.get("detail") or {}
                    transitions.append({
                        "ticker": ev.get("ticker"),
                        "from_state": d.get("from_state"),
                        "to_state": d.get("to_state"),
                        "timestamp": ev.get("timestamp"),
                    })
    except Exception:
        transitions = [{"error": "position_events.json unavailable"}]

    return {
        "prior_run_date": prior.get("date"),
        "current_run_date": curr.get("date"),
        "regime_shift": regime_shift,
        "voters_flipped": voters_flipped,
        "gate_changed": gate_changed,
        "position_transitions": transitions,
    }


@app.route("/api/assessment.json")
def assessment_json():
    """
    Public JSON API — the single consolidated daily read (PER-508 #21).
    Aggregates the regime gauges, position-engine states, holding
    technicals, VIX complex, theme leaderboard, and a what-changed diff
    vs the prior run. Read-only over cached artifacts; inherits the
    shape sentinel (503 while a schema-stale artifact regenerates).
    Sections degrade independently to {"error": ...}.
    """
    data, err = _load_framework_payload()
    if err is not None:
        return err

    now_utc = datetime.datetime.now(datetime.timezone.utc)
    try:
        from zoneinfo import ZoneInfo
        generated_at_et = now_utc.astimezone(
            ZoneInfo("America/New_York")).strftime("%Y-%m-%d %I:%M %p %Z")
    except Exception:
        generated_at_et = None

    out = {"generated_at": now_utc.isoformat(),
           "generated_at_et": generated_at_et,
           "framework_generated_at": data.get("generated_at")}
    for key, fn in (("regime", _assessment_regime),
                    ("positions", _assessment_positions),
                    ("candidates", _assessment_candidates),
                    ("r28", _assessment_r28),
                    ("technicals", _assessment_technicals),
                    ("vol_complex", _assessment_vol),
                    ("themes", _assessment_themes),
                    ("changes_since_prior", _assessment_changes)):
        try:
            out[key] = fn(data)
        except Exception as e:
            out[key] = {"error": str(e)}
    if "stale_hours" in data:
        out["stale_hours"] = data["stale_hours"]

    return app.response_class(
        response=json.dumps(out, cls=NumpyEncoder),
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
