#!/usr/bin/env python3
"""
Tests for the SWING-horizon regime calculator (3 voters + 200DMA gate).

Run: python3 test_regime.py
Style matches test_pipeline.py: module-level test_*() with plain asserts,
deterministic injected fetchers, no network.
"""

import copy
import datetime
import os
import sys

import numpy as np
import pandas as pd

sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))

from framework.regime_calculator import RegimeCalculator

TRENDING = "Risk-on / Trending"
CHOPPY = "Risk-on / Choppy"
CAUTION = "Caution"
RISK_OFF = "Risk-off"

BASE_CONFIG = {
    "regime": {
        # These tests pin the Build-1A PARLIAMENT (kept intact behind the
        # engine flag for reversibility — D-008 cutover). The chassis path
        # is pinned by test_gauge_chassis.py.
        "engine": "parliament",
        "gauges": {
            "spy_vs_200dma": {"proxy": "SPY", "lookback_days": 200,
                              "risk_on_threshold": 0, "caution_threshold": -2,
                              "risk_off_threshold": -5},
            "vix_5d_avg": {"proxy": "^VIX", "lookback_days": 5,
                           "risk_on_threshold": 18, "caution_threshold": 22,
                           "risk_off_threshold": 27},
            "hy_spread": {"fred_series": "BAMLH0A0HYM2",
                          "fallback_long": "HYG", "fallback_short": "IEF",
                          "risk_on_threshold": 3.0, "caution_threshold": 4.0,
                          "risk_off_threshold": 5.0},
            "breadth": {"proxy": "S5FI", "fallback_proxy": "RSP",
                        "fallback_benchmark": "SPY",
                        "risk_on_threshold": 60, "caution_threshold": 50,
                        "risk_off_threshold": 30},
            "yield_curve": {"fred_long": "DGS30", "fred_short": "DGS2",
                            "risk_on_threshold": 0.50, "caution_threshold": 0.00,
                            "risk_off_threshold": 0.00},
        },
        "states": [
            {"name": TRENDING, "action": "full deploy", "color": "#3fb950"},
            {"name": CHOPPY, "action": "a+ only", "color": "#d29922"},
            {"name": CAUTION, "action": "no new aggressive", "color": "#f0883e"},
            {"name": RISK_OFF, "action": "defensive", "color": "#f85149"},
        ],
        "change_protocol": {
            "consecutive_confirmations_required": 2,
            "intra_week_override_triggers": [
                {"gauge": "vix_5d_avg", "condition": ">", "value": 27,
                 "message": "VIX > 27 — intra-week risk-off override"},
                {"gauge": "spy_vs_200dma", "condition": "<", "value": 0,
                 "message": "SPY closed below 200DMA — intra-week caution override"},
            ],
        },
    }
}


def _df(closes):
    """Build an OHLCV DataFrame from a list of closes."""
    closes = list(closes)
    idx = pd.date_range(end="2026-07-02", periods=len(closes), freq="B")
    return pd.DataFrame({
        "Open": closes, "High": closes, "Low": closes,
        "Close": closes, "Volume": [1000] * len(closes),
    }, index=idx)


def make_fetcher(spy_price=700.0, spy_ma=650.0, vix=15.0,
                 hyg=80.0, ief=95.0, hyg_trend="up",
                 rsp_trend="up", spy_missing=False):
    """Deterministic fetcher.

    SPY: 230 days at spy_ma then 21 days at spy_price — the 200DMA lands
    near spy_ma (so spy_price vs spy_ma sets the gate) while the last 20
    closes are flat (so SPY does not distort the breadth fallback's 20d
    window). VIX: constant. HYG vs flat IEF: 60d ramp direction drives the
    HY percentile vote. RSP vs flat SPY: 130d ramp direction drives the
    breadth fallback vote (+/-0.75% over the 20d window, clear of the
    +/-0.5 thresholds)."""
    def fetcher(ticker, period="1y"):
        if ticker == "SPY":
            if spy_missing:
                return None
            return _df([spy_ma] * 230 + [spy_price] * 21)
        if ticker == "^VIX":
            return _df([vix] * 25)
        if ticker == "HYG":
            if hyg_trend == "up":
                base = np.linspace(hyg * 0.97, hyg, 130)
            elif hyg_trend == "down":
                base = np.linspace(hyg * 1.03, hyg, 130)
            else:  # "mid": current ratio sits mid-range of the 60d window
                base = np.concatenate([np.full(70, hyg),
                                       np.linspace(hyg * 1.03, hyg * 0.97, 59),
                                       [hyg]])
            return _df(base)
        if ticker == "IEF":
            return _df([ief] * 130)
        if ticker == "S5FI":
            return None  # force RSP/SPY breadth fallback
        if ticker == "RSP":
            if rsp_trend == "up":
                return _df(np.linspace(162.0, 180.0, 130))
            if rsp_trend == "down":
                return _df(np.linspace(198.0, 180.0, 130))
            return _df([180.0] * 130)
        return None
    return fetcher


def _calc(config=None, **fetcher_kw):
    cfg = copy.deepcopy(config or BASE_CONFIG)
    calc = RegimeCalculator(cfg, make_fetcher(**fetcher_kw))
    # Keep unit tests fully offline: stub every network path that does not
    # go through the injected fetcher (all three FRED fetchers — the HY one
    # would silently fire whenever FRED_API_KEY is set in the environment).
    calc._try_fred_api_yield_curve = lambda c: None
    calc._try_fred_csv_yield_curve = lambda c: None
    calc._try_fred_hy_spread = lambda c: None
    return calc


# ------------------------------------------------------------------
# _determine_state: the approved spec, pinned exhaustively
# ------------------------------------------------------------------

def test_determine_state_full_data():
    calc = _calc()
    f = lambda ro, rf: calc._determine_state(ro, rf, 0)
    # Ladder
    assert f(3, 0) == TRENDING
    assert f(2, 0) == CHOPPY
    assert f(1, 0) == CAUTION           # (1,2,0)
    assert f(0, 0) == RISK_OFF          # (0,3,0): zero risk_on -> Risk-off
    # ANY-risk_off cap: caps at Caution regardless of tally
    assert f(2, 1) == CAUTION           # (2,0,1)
    assert f(1, 1) == CAUTION           # (1,1,1)
    # 2+ risk_off -> Risk-off
    assert f(1, 2) == RISK_OFF          # (1,0,2)
    assert f(0, 2) == RISK_OFF          # (0,1,2)
    assert f(0, 3) == RISK_OFF          # (0,0,3)
    assert f(0, 1) == RISK_OFF          # (0,2,1): zero risk_on
    print("  determine_state full-data ladder + qualifiers: OK")


def test_determine_state_unavailable_guards():
    calc = _calc()
    f = calc._determine_state
    # 2 confirmed risk_off dominate missing data (order pin: rule 1 > rule 2)
    assert f(0, 2, 1) == RISK_OFF
    # 2+ dark voters -> Caution, never Risk-off, never Choppy
    assert f(1, 0, 2) == CAUTION
    assert f(0, 0, 3) == CAUTION
    assert f(0, 0, 2) == CAUTION
    # 1 dark voter: zero-risk_on floor requires full data
    assert f(0, 0, 1) == CAUTION        # (0,2,0)+1dark: NOT Risk-off
    assert f(0, 1, 1) == CAUTION        # (0,1,1)+1dark: any-risk_off cap
    # 1 dark voter with confirmed risk_on
    assert f(2, 0, 1) == CHOPPY         # (2,0,0)+1dark
    assert f(1, 0, 1) == CAUTION        # ladder: 1 risk_on -> Caution
    # Trending is only reachable with all 3 voters risk_on (3,0,0)
    assert f(3, 0, 0) == TRENDING
    print("  determine_state unavailable guards + ordering pins: OK")


# ------------------------------------------------------------------
# compute(): gate semantics, output shape, override rewire
# ------------------------------------------------------------------

def _risk_on_everything():
    """Fetcher kwargs that make all 3 voters risk_on."""
    return dict(vix=15.0, hyg_trend="up", rsp_trend="up")


def test_gate_open_passes_state_through():
    calc = _calc(spy_price=700.0, spy_ma=650.0, **_risk_on_everything())
    r = calc.compute()
    assert r["regime"] == TRENDING, r["regime"]
    assert r["backdrop_gate"]["open"] is True
    assert r["backdrop_gate"]["capped"] is False
    assert r["backdrop_gate"]["reason"] is None
    print("  gate open, 3/3 risk_on -> Trending: OK")


def test_gate_shut_caps_risk_on_at_caution():
    # SPY 1% below its 200DMA — old system printed Trending here
    calc = _calc(spy_price=643.5, spy_ma=650.0, **_risk_on_everything())
    r = calc.compute()
    assert r["regime"] == CAUTION, r["regime"]
    assert r["backdrop_gate"]["open"] is False
    assert r["backdrop_gate"]["capped"] is True
    assert r["backdrop_gate"]["reason"] == "below_200dma"
    # the voters were still risk_on — cap did not rewrite the tally
    assert r["risk_on_count"] == 3
    print("  gate shut caps Trending -> Caution (voters untouched): OK")


def test_gate_never_upgrades_risk_off():
    # Vol spiking + credit widening -> 2 risk_off votes; SPY below 200DMA
    calc = _calc(spy_price=600.0, spy_ma=650.0, vix=30.0, hyg_trend="down",
                 rsp_trend="up")
    r = calc.compute()
    assert r["regime"] == RISK_OFF, r["regime"]
    assert r["backdrop_gate"]["open"] is False
    assert r["backdrop_gate"]["capped"] is False   # cap only touches risk-on
    print("  gate never upgrades Risk-off: OK")


def test_gate_fails_closed_on_missing_spy():
    # A SPY outage also darkens the breadth fallback (it needs SPY for the
    # RSP/SPY ratio) — realistic coupling. Actual scenario: VIX+HY risk_on,
    # breadth unavailable -> base Choppy (2 risk_on, 1 dark), then the gate
    # fails closed and caps it at Caution.
    calc = _calc(spy_missing=True, **_risk_on_everything())
    r = calc.compute()
    assert r["risk_on_count"] == 2 and r["unavailable_count"] == 1
    assert r["regime"] == CAUTION, r["regime"]
    assert r["backdrop_gate"]["open"] is False
    assert r["backdrop_gate"]["reason"] == "data_unavailable"
    assert r["backdrop_gate"]["capped"] is True   # base was Choppy -> capped
    print("  gate fails closed on missing SPY data (base Choppy -> capped): OK")


def test_gate_shut_on_caution_base_leaves_capped_false():
    # Base state already Caution (1 risk_on: breadth; VIX + HY caution).
    # Gate shut must NOT mark capped — the cap only records an actual
    # downgrade of a risk-on state.
    calc = _calc(spy_price=643.5, spy_ma=650.0, vix=20.0, hyg_trend="mid",
                 rsp_trend="up")
    r = calc.compute()
    assert (r["risk_on_count"], r["caution_count"], r["risk_off_count"]) == (1, 2, 0)
    assert r["regime"] == CAUTION
    assert r["backdrop_gate"]["open"] is False
    assert r["backdrop_gate"]["reason"] == "below_200dma"
    assert r["backdrop_gate"]["capped"] is False
    print("  gate shut on Caution base leaves capped False: OK")


def test_gate_uses_raw_pct_not_rounded():
    # SPY 0.003% below the 200DMA: display value rounds to -0.0 but the
    # gate must still shut (raw comparison, no open sliver below the MA).
    calc = _calc(spy_price=650.0 * (1 - 0.00003 / (1 - 0.105 * 0.00003)),
                 spy_ma=650.0, **_risk_on_everything())
    r = calc.compute()
    g = r["backdrop_gate"]
    assert g["value"] in (0.0, -0.0), g["value"]      # display rounds to zero
    assert g["open"] is False and g["reason"] == "below_200dma"
    assert r["regime"] == CAUTION
    # and pct_raw stays internal — not published in backdrop_gate
    assert "pct_raw" not in g
    print("  gate compares raw pct (no rounded open sliver): OK")


def test_gate_fails_closed_on_nan_200dma():
    # A NaN close inside the 200d window poisons the rolling mean; the
    # gauge must read unavailable and the gate must fail CLOSED, not open.
    def nan_fetcher(ticker, period="1y"):
        base = make_fetcher(**_risk_on_everything())
        df = base(ticker, period)
        if ticker == "SPY" and df is not None:
            df = df.copy()
            df.iloc[100, df.columns.get_loc("Close")] = float("nan")
        return df
    cfg = copy.deepcopy(BASE_CONFIG)
    calc = RegimeCalculator(cfg, nan_fetcher)
    calc._try_fred_api_yield_curve = lambda c: None
    calc._try_fred_csv_yield_curve = lambda c: None
    calc._try_fred_hy_spread = lambda c: None
    r = calc.compute()
    assert r["backdrop_gate"]["open"] is False
    assert r["backdrop_gate"]["reason"] == "data_unavailable"
    assert r["backdrop_gate"]["value"] is None
    assert r["regime"] == CAUTION
    print("  gate fails closed on NaN 200DMA: OK")


def test_output_shape():
    calc = _calc(**_risk_on_everything())
    r = calc.compute()
    assert sorted(r["gauges"].keys()) == ["breadth", "hy_spread", "vix_5d_avg"]
    assert r["backdrop_gate"]["gauge"] == "spy_vs_200dma"
    assert r["backdrop_gate"]["role"] == "backdrop_gate"
    assert "yield_curve" in r["macro_inputs"]
    assert r["macro_inputs"]["yield_curve"]["role"] == "macro_input"
    # In this offline setup the macro input is degraded (FRED stubbed,
    # ^TYX not in the fake fetcher): must read unavailable with a null
    # value — and, per the total==3 assert below, must NOT enter the tally.
    assert r["macro_inputs"]["yield_curve"]["signal"] == "unavailable"
    assert r["macro_inputs"]["yield_curve"]["value"] is None
    total = (r["risk_on_count"] + r["caution_count"] + r["risk_off_count"]
             + r["unavailable_count"])
    assert total == 3, f"counts must cover exactly the 3 voters, got {total}"
    assert r["regime"] in (TRENDING, CHOPPY, CAUTION, RISK_OFF)
    # R4 confirmation-protocol export — theme_ranker consumes exactly these
    # key names; renaming either silently disables confirmed regime exits
    assert "consecutive_degraded_weeks" in r
    assert "consecutive_degraded_weeks_completed" in r
    assert "confirmations_needed" in r
    print("  output shape (3 voters + backdrop_gate + macro_inputs): OK")


def test_yield_curve_source_order():
    """FRED API (true DGS2) must be tried before CSV, then yfinance."""
    cfg = copy.deepcopy(BASE_CONFIG)
    ycfg = cfg["regime"]["gauges"]["yield_curve"]
    calc = RegimeCalculator(cfg, lambda t, period="1y": None)
    A = {"value": 0.84, "signal": "risk_on", "detail": "api"}
    B = {"value": 0.80, "signal": "risk_on", "detail": "csv"}
    calc._try_fred_api_yield_curve = lambda c: A
    calc._try_fred_csv_yield_curve = lambda c: B
    assert calc._compute_yield_curve(ycfg) is A
    calc._try_fred_api_yield_curve = lambda c: None
    assert calc._compute_yield_curve(ycfg) is B
    calc._try_fred_csv_yield_curve = lambda c: None
    out = calc._compute_yield_curve(ycfg)   # falls to yfinance; fetcher None
    assert out["signal"] == "unavailable"
    print("  yield-curve source order (API -> CSV -> yfinance): OK")


def test_yield_curve_tyx_only_never_fakes_a_spread():
    """30Y present but no ^IRX: value must be None (a 30Y LEVEL must never
    masquerade as a spread) and the detail must not mention 2Y."""
    cfg = copy.deepcopy(BASE_CONFIG)

    def fetcher(ticker, period="1y"):
        return _df([4.98] * 10) if ticker == "^TYX" else None

    calc = RegimeCalculator(cfg, fetcher)
    calc._try_fred_api_yield_curve = lambda c: None
    calc._try_fred_csv_yield_curve = lambda c: None
    yc = calc._compute_yield_curve(cfg["regime"]["gauges"]["yield_curve"])
    assert yc["value"] is None
    assert yc["signal"] == "unavailable"
    assert "2Y" not in yc["detail"]
    assert yc["long_yield"] == 4.98
    print("  yield-curve ^TYX-only branch stays honest (no fake spread): OK")


def test_run_framework_routes_output_through_sanitize():
    # Cheap wiring pin: run_framework must sanitize the assembled output
    # before writing (full-pipeline integration is out of unit-test scope).
    import inspect
    from framework import framework_runner
    src = inspect.getsource(framework_runner.run_framework)
    assert "sanitize_for_json(output)" in src
    src_hist = inspect.getsource(framework_runner.save_regime_history)
    assert "sanitize_for_json(entry)" in src_hist
    print("  sanitize wiring present in run_framework + save_regime_history: OK")


def test_intra_week_override_reads_gate_input():
    # SPY below 200DMA must still fire the config spy_vs_200dma trigger
    # even though spy is no longer in the voters dict.
    calc = _calc(spy_price=643.5, spy_ma=650.0, **_risk_on_everything())
    r = calc.compute()
    assert r["intra_week_override"] is not None
    assert "200DMA" in r["intra_week_override"]
    # and the VIX trigger still works off the voters dict
    calc2 = _calc(vix=30.0, hyg_trend="up", rsp_trend="up")
    r2 = calc2.compute()
    assert r2["intra_week_override"] is not None
    assert "VIX" in r2["intra_week_override"]
    print("  intra-week overrides (gate input + voter): OK")


# ------------------------------------------------------------------
# Yield curve: DGS2 fix + honest fallback relabel
# ------------------------------------------------------------------

def test_yield_curve_fallback_relabels_30y_3mo():
    """No FRED key, CSV dead -> ^TYX-^IRX fallback must say 30Y-3mo,
    never 30Y-2Y, and must not emit a dgs2 key."""
    cfg = copy.deepcopy(BASE_CONFIG)

    def fetcher(ticker, period="1y"):
        if ticker == "^TYX":
            return _df([4.98] * 10)
        if ticker == "^IRX":
            return _df([3.82] * 10)
        return None

    calc = RegimeCalculator(cfg, fetcher)
    calc._try_fred_api_yield_curve = lambda c: None   # no key
    calc._try_fred_csv_yield_curve = lambda c: None   # Akamai-blocked
    yc = calc._compute_yield_curve(cfg["regime"]["gauges"]["yield_curve"])
    assert "30Y-3mo" in yc["detail"], yc["detail"]
    assert "30Y-2Y" not in yc["detail"], yc["detail"]
    assert yc["spread_label"] == "30Y-3mo"
    assert "dgs2" not in yc, "dgs2 key must never carry ^IRX"
    assert abs(yc["value"] - 1.16) < 0.01
    assert yc["signal"] == "risk_on"      # informational only (macro input)
    assert yc["source"] == "yfinance (^TYX-^IRX)"
    print("  yield-curve fallback relabeled 30Y-3mo, no dgs2 key: OK")


def test_yield_curve_true_dgs2_keeps_label_and_keys():
    calc = _calc()
    yc = calc._classify_yield_spread(0.84, 4.98, 4.14, "FRED API",
                                     date_str="2026-07-02", short_label="2Y")
    assert "30Y-2Y" in yc["detail"]
    assert yc["dgs2"] == 4.14 and yc["dgs30"] == 4.98
    assert yc["spread_bp"] == 84
    assert yc["signal"] == "risk_on"
    # inverted curve classifies risk_off (informational)
    inv = calc._classify_yield_spread(-0.30, 4.0, 4.30, "FRED API")
    assert inv["signal"] == "risk_off"
    print("  yield-curve true-DGS2 labeling + classification: OK")


def test_yield_curve_fred_api_uses_dgs2(monkeypatched=None):
    """FRED API path fetches DGS30/DGS2 with the HY-gauge fetch pattern."""
    import requests as _requests
    cfg = copy.deepcopy(BASE_CONFIG)
    calls = []

    class FakeResp:
        status_code = 200
        def json(self):
            series = calls[-1]
            val = "4.98" if series == "DGS30" else "4.14"
            return {"observations": [{"date": "2026-07-02", "value": val}]}

    def fake_get(url, params=None, timeout=None, **kw):
        calls.append(params["series_id"])
        return FakeResp()

    old_get, old_key = _requests.get, os.environ.get("FRED_API_KEY")
    _requests.get = fake_get
    os.environ["FRED_API_KEY"] = "test-key"
    try:
        calc = RegimeCalculator(cfg, lambda t, period="1y": None)
        yc = calc._try_fred_api_yield_curve(cfg["regime"]["gauges"]["yield_curve"])
    finally:
        _requests.get = old_get
        if old_key is None:
            os.environ.pop("FRED_API_KEY", None)
        else:
            os.environ["FRED_API_KEY"] = old_key
    assert calls == ["DGS30", "DGS2"], calls
    assert yc is not None and yc["dgs2"] == 4.14
    assert abs(yc["value"] - 0.84) < 1e-9
    assert "30Y-2Y" in yc["detail"]
    print("  FRED API path pulls true DGS2: OK")


# ------------------------------------------------------------------
# Runner integration: sanitize + history snapshot
# ------------------------------------------------------------------

def test_sanitize_covers_new_fields():
    from signal_engine import sanitize_for_json
    dirty = {
        "regime": {
            "backdrop_gate": {"value": float("nan"), "open": True},
            "macro_inputs": {"yield_curve": {"value": float("inf")}},
            "gauges": {"vix_5d_avg": {"value": 17.0}},
        }
    }
    clean = sanitize_for_json(dirty)
    assert clean["regime"]["backdrop_gate"]["value"] is None
    assert clean["regime"]["macro_inputs"]["yield_curve"]["value"] is None
    assert clean["regime"]["gauges"]["vix_5d_avg"]["value"] == 17.0
    print("  sanitize_for_json covers backdrop_gate + macro_inputs: OK")


def test_history_snapshot_keeps_gate_and_macro():
    # Redirect STATE_DIR to a temp dir: this test must never touch the real
    # framework/state/regime_history.json (a crash mid-test would corrupt
    # production state).
    import json as _json
    import tempfile
    from framework import framework_runner
    old_state_dir = framework_runner.STATE_DIR
    framework_runner.STATE_DIR = tempfile.mkdtemp(prefix="regime_test_")
    try:
        entry = {
            "date": "2099-01-01",
            "regime": CHOPPY,
            "risk_on_count": 2, "caution_count": 1, "risk_off_count": 0,
            "gauges": {"vix_5d_avg": {"value": 17.0, "signal": "risk_on",
                                      "detail": "d", "extra": "dropme"}},
            "backdrop_gate": {"open": True, "capped": False, "reason": None,
                              "value": 8.1, "detail": "SPY above", "price": 745.0},
            "macro_inputs": {"yield_curve": {"value": float("nan"),
                                             "signal": "risk_on",
                                             "detail": "84bp", "source": "FRED API"}},
        }
        framework_runner.save_regime_history([], entry)
        path = os.path.join(framework_runner.STATE_DIR, "regime_history.json")
        with open(path, "r") as fh:
            raw = fh.read()
        assert "NaN" not in raw, "history write must be sanitized (no bare NaN tokens)"
        hist = _json.loads(raw)
        saved = [h for h in hist if h.get("date") == "2099-01-01"][0]
        assert saved["backdrop_gate"]["open"] is True
        assert saved["backdrop_gate"]["value"] == 8.1
        assert saved["macro_inputs"]["yield_curve"]["value"] is None  # NaN -> null
        assert saved["gauges"]["vix_5d_avg"]["signal"] == "risk_on"
        assert "extra" not in saved["gauges"]["vix_5d_avg"]
    finally:
        framework_runner.STATE_DIR = old_state_dir
    print("  save_regime_history snapshots gate + macro (sanitized, isolated): OK")


# ------------------------------------------------------------------

if __name__ == "__main__":
    print("\n=== Regime calculator tests (3-voter swing gauge) ===")
    test_determine_state_full_data()
    test_determine_state_unavailable_guards()
    test_gate_open_passes_state_through()
    test_gate_shut_caps_risk_on_at_caution()
    test_gate_never_upgrades_risk_off()
    test_gate_fails_closed_on_missing_spy()
    test_gate_shut_on_caution_base_leaves_capped_false()
    test_gate_uses_raw_pct_not_rounded()
    test_gate_fails_closed_on_nan_200dma()
    test_output_shape()
    test_yield_curve_source_order()
    test_yield_curve_tyx_only_never_fakes_a_spread()
    test_run_framework_routes_output_through_sanitize()
    test_intra_week_override_reads_gate_input()
    test_yield_curve_fallback_relabels_30y_3mo()
    test_yield_curve_true_dgs2_keeps_label_and_keys()
    test_yield_curve_fred_api_uses_dgs2()
    test_sanitize_covers_new_fields()
    test_history_snapshot_keeps_gate_and_macro()
    print("\nAll regime tests passed.\n")
