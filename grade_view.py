"""Grade view — the D-011 grade for display surfaces (search page, Score
Lab). DISPLAY AND EVALUATION ONLY: grade_setup is untouched and the
framework's own artifact is untouched; this composes the exact same
machinery (compute_grade_inputs → PositionSignalEngine's candidate
convention → grade_setup) into a read-only payload.

THE BAR LAW (load-bearing): the PRIMARY grade is computed on the last
CONFIRMED close — the bar that matches candidate_grades and would
actually drive an entry. A PROVISIONAL grade exists ONLY while the
market is open (defined by the same D-018 splitter the whole system
uses: a forming bar dated today before the 16:10 ET settle), is always
labelled, and is never rendered as "the grade". Every grade carries its
basis bar and timestamp — no grade may appear without its basis.

WITHHOLDING (inherited, not weakened): a group whose breaker coverage
is degraded/unavailable gets NO grade — the same law that withholds the
trade signal. A grade computed on unchecked breaker data would be the
fabricated-clear class arriving on a new surface.

OFF-UNIVERSE: the underlying logic currently treats the same missing
fact two ways (condition 5 passes "not applicable" on no group; row 6
fails "breaker unknown") — a known defect reported separately, NOT
rendered here. Off-universe tickers get one honest statement and no
grade.
"""

import datetime
import json
import os

import numpy as np

BASE_DIR = os.path.dirname(os.path.abspath(__file__))
SIGNALS_PATH = os.path.join(BASE_DIR, "data", "signals.json")
FRAMEWORK_PATH = os.path.join(BASE_DIR, "public", "framework.json")


def _now_et_clock():
    from framework.regime_calculator import _now_et
    return _now_et()


def _load_artifacts(signals=None, framework=None):
    if signals is None:
        with open(SIGNALS_PATH) as f:
            signals = json.load(f)
    if framework is None:
        with open(FRAMEWORK_PATH) as f:
            framework = json.load(f)
    return signals, framework


def _context_for(symbol, signals, framework):
    """Universe membership + group context from the served artifacts."""
    groups = signals.get("groups") or []
    selected = {g.get("name") for g in groups}
    breakers = {g.get("name"): g.get("breaker_status") for g in groups}
    for g in groups:
        for s in g.get("stocks") or []:
            if s.get("ticker") == symbol:
                return {"group": g.get("name"), "selected": selected,
                        "breakers": breakers,
                        "breaker_status": g.get("breaker_status"),
                        "next_earnings_date": s.get("next_earnings_date"),
                        "regime": (framework.get("regime") or {}).get(
                            "regime")}
    return None


def _row_lines(inputs):
    """Threshold-to-PRICE translations — the difference between a
    verdict and a decision. Only rows whose threshold IS a price."""
    sma20, atr = inputs.get("sma20"), inputs.get("atr14")
    lines = {}
    if sma20 is not None:
        lines["1_conditions"] = round(sma20, 2)        # c1: close > SMA20
    if sma20 is not None and atr:
        lines["2_extension"] = round(
            sma20 + inputs["extension_guard_max"] * atr, 2)
    if inputs.get("sma5") is not None:
        lines["3_approach"] = round(inputs["sma5"], 2)
    return lines


def _grade_block(engine, ctx, symbol, gi, today):
    """One grading pass through the candidate convention, returning the
    display payload (grade + seven rows with actual/threshold/line)."""
    row = {"ticker": symbol, "grade_inputs": gi,
           "next_earnings_date": ctx.get("next_earnings_date")}
    full = engine._grade_candidate_full(
        ctx["group"], row, ctx["selected"], ctx["breakers"],
        ctx["regime"], today)
    if full.get("grade") is None:
        return {"grade": None, "reasons": full.get("reasons")}
    inputs = full["inputs"]
    lines = _row_lines(inputs)

    def rnd(v, n=2):
        return round(v, n) if isinstance(v, (int, float)) else v

    actuals = {
        "1_conditions": {"actual": sum(
            1 for c in (full.get("conditions") or {}).values()
            if c.get("met")), "threshold": "5 of 5"},
        "2_extension": {"actual": rnd(inputs.get("extension_atr")),
                        "threshold": inputs["extension_guard_max"],
                        "unit": "xATR"},
        "3_approach": {"actual": rnd(inputs.get("close")),
                       "threshold": rnd(inputs.get("sma5")),
                       "unit": "close vs SMA5"},
        "4_rsi": {"actual": rnd(inputs.get("rsi14"), 1),
                  "threshold": [inputs["rsi_min"], inputs["rsi_max"]]},
        "5_score": {"actual": inputs.get("quality_score"),
                    "threshold": inputs["score_min"]},
        "6_breaker": {"actual": inputs.get("breaker_status"),
                      "threshold": "clear"},
        "7_runway": {"actual": inputs.get("runway_sessions"),
                     "threshold": inputs["runway_min_sessions"],
                     "unit": "sessions"},
    }
    rows = []
    for key, r in (full.get("rows") or {}).items():
        rows.append({"row": key, "met": r.get("met"),
                     "detail": r.get("detail"),
                     "line_price": lines.get(key),
                     **actuals.get(key, {})})
    return {"grade": full["grade"], "failing": full["failing"],
            "reasons": full["reasons"], "rows": rows,
            "conditions": full.get("conditions"),
            "inputs": {k: (round(v, 4) if isinstance(v, float) else v)
                       for k, v in inputs.items()}}


def compute_grade_view(symbol, df=None, ytd_df=None, signals=None,
                       framework=None, now_et=None, fetcher=None):
    """The full payload. df/ytd_df/signals/framework/now_et injectable
    for pins; production callers pass nothing and get live fetches."""
    from framework.position_signals import (PositionSignalEngine,
                                            strip_synthetic_last_bar)
    from framework.regime_calculator import confirmed_close_frame
    import signal_engine as se

    signals, framework = _load_artifacts(signals, framework)
    out = {"symbol": symbol,
           "generated_at": datetime.datetime.now(
               datetime.timezone.utc).isoformat()}

    ctx = _context_for(symbol, signals, framework)
    if ctx is None:
        out["status"] = "off_universe"
        out["message"] = (
            "outside the active universe: the group-dependent rows "
            "(c5 thesis, row 6 breaker) cannot be evaluated — no grade")
        return out

    out["group"] = ctx["group"]
    out["breaker_status"] = ctx["breaker_status"]
    if ctx["breaker_status"] in (None, "degraded"):
        # the fabricated-clear gate, inherited: unchecked breaker data
        # grades NOTHING (same law as the withheld trade signal)
        out["status"] = "breaker_unavailable"
        out["message"] = (
            f"group breaker coverage "
            f"{'degraded' if ctx['breaker_status'] else 'unavailable'} — "
            "grade withheld (a grade on unchecked breaker data would be "
            "a fabricated clear)")
        return out

    fetch = fetcher or se.fetch_data
    if df is None:
        df = fetch(symbol, period="6mo")
    if df is None or len(df) < 25:
        out["status"] = "no_data"
        out["message"] = "insufficient price history"
        return out
    if ytd_df is None:
        ytd_df = fetch(symbol, period="1y")

    # the SAME engine construction as the framework runner — thresholds
    # sourced from framework/config.yaml, never copied (review finding:
    # a default-built engine silently diverges on any approved-knob
    # change); the SAME clock as the runner's grading (ET, not host-local
    # — a UTC host between 20:00 and 24:00 ET is already tomorrow)
    from framework.framework_runner import load_config
    from earnings_calendar import _today_et
    engine = PositionSignalEngine(load_config(), fetcher=None)
    today = _today_et()

    sdf = strip_synthetic_last_bar(df)
    conf, forming = confirmed_close_frame(sdf, now_et=now_et)
    if conf is None or len(conf) < 25:
        out["status"] = "no_data"
        out["message"] = "insufficient confirmed history"
        return out

    gi_conf = se.compute_grade_inputs(conf, ytd_df=ytd_df,
                                      now_et=now_et)
    if gi_conf is None:
        out["status"] = "no_data"
        out["message"] = "grade inputs could not be computed"
        return out
    confirmed = _grade_block(engine, ctx, symbol, gi_conf, today)
    confirmed["basis"] = "confirmed_close"
    confirmed["basis_bar"] = str(conf.index[-1])[:10]
    out["confirmed"] = confirmed

    # PROVISIONAL — only while the market is open, always labelled,
    # never "the grade". market_open is the CLOCK's verdict; whether a
    # forming bar is AVAILABLE is a separate fact — conflating them let
    # the basis line assert "market closed" during a feed lag (review
    # finding: a fabricated state, the D-019 class)
    now = now_et or _now_et_clock()
    out["market_open"] = bool(
        now.weekday() < 5
        and datetime.time(9, 30) <= now.time() < datetime.time(16, 10))
    out["forming_bar_available"] = forming is not None
    out["provisional"] = None
    if forming is not None:
        from framework.position_signals import grade_inputs_from_df
        from signal_engine import compute_rsi, compute_ytd_return_v2, \
            score_stock_v2
        gi_prov = grade_inputs_from_df(sdf)
        if gi_prov is not None:
            r = compute_rsi(sdf["Close"]).iloc[-1]
            gi_prov["rsi14"] = float(r) if np.isfinite(r) else None
            ytd_v = ytd_b = None
            if ytd_df is not None and len(ytd_df):
                ytd_v, ytd_b = compute_ytd_return_v2(
                    strip_synthetic_last_bar(ytd_df), with_basis=True)
            s_v, _sig, _det = score_stock_v2(sdf, ytd_return=ytd_v,
                                             ytd_basis=ytd_b)
            gi_prov["quality_score"] = None if s_v is None else float(s_v)
            gi_prov["ytd_basis"] = _det.get("ytd_basis")
            prov = _grade_block(engine, ctx, symbol, gi_prov, today)
            prov["basis"] = "forming_bar_provisional"
            prov["basis_bar"] = str(sdf.index[-1])[:19]
            out["provisional"] = prov

    out["differs"] = bool(
        out["provisional"]
        and out["provisional"].get("grade") is not None
        and out["provisional"]["grade"] != confirmed.get("grade"))
    out["status"] = "graded"
    return out
