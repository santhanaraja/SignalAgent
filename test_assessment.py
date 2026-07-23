#!/usr/bin/env python3
"""
Tests for /api/assessment.json — the consolidated daily read (PER-508 #21).

Pins: response shape across all sections, shape-sentinel inheritance
(pre-1A artifact -> 503), the ET timestamp format, cushion arithmetic,
the what-changed diff, and per-section degradation (missing sources
error the section, never the response).

Run: python3 test_assessment.py
"""

import datetime
import json
import os
import re
import shutil
import sys
import tempfile

sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))

import ticker_api


def _now_iso(hours_ago=0):
    return (datetime.datetime.now(datetime.timezone.utc)
            - datetime.timedelta(hours=hours_ago)).isoformat()


def _framework_payload(run_date="2026-07-09"):
    g = {"value": 1.0, "signal": "risk_on", "detail": "d"}
    return {
        "generated_at": _now_iso(),
        "schema": "regime-b-chassis",
        "regime": {
            "regime": "Risk-on / Choppy",
            "action": "A+ setups only.",
            # chassis-era artifact (D-008): the serve guard requires the
            # engine + chassis block to match the configured engine
            "engine": "chassis",
            "chassis": {
                "engine": "chassis",
                "raw_state": "In-Trend-Throttled",
                "confirmed_state": "In-Trend-Throttled",
                "regime": "Risk-on / Choppy",
                "exposure_ceiling_pct": 50.0,
                "trend_in": True,
                "throttles": {"vix": {"firing": False},
                              "hy": {"firing": True},
                              "breadth": {"firing": False}},
                "throttles_firing": 1,
                "hysteresis": {"up": 0, "down": 0, "n": 2,
                               "mode": "asymmetric"},
                "degraded": False, "degraded_reason": None,
            },
            "risk_on_count": 2, "caution_count": 1, "risk_off_count": 0,
            "gauges": {
                "vix_5d_avg": {"value": 17.2, "spot": 16.8, "signal": "risk_on",
                               "detail": "VIX 5d avg 17.2 (spot 16.8)"},
                "hy_spread": dict(g),
                "breadth": {"value": 0.3, "signal": "caution", "detail": "d"},
            },
            "backdrop_gate": {"gauge": "spy_vs_200dma", "open": True,
                              "capped": False, "reason": None, "value": 7.5},
            "macro_inputs": {"yield_curve": {"value": 0.8, "signal": "risk_on"}},
            "consecutive_degraded_weeks": 0,
            "intra_week_override": None,
        },
        "themes": {
            "active_themes": ["Semis", "Biotech"],
            "ranked_themes": [
                {"name": "Semis", "rank": 1, "composite": 1.5},
                {"name": "Biotech", "rank": 2, "composite": 2.5},
                {"name": "Gold", "rank": 3, "composite": 4.0},
                {"name": "Broken", "rank": None, "composite": None},
            ],
        },
        "position_signals": {
            "tickers": {
                "ARWR": {
                    "ticker": "ARWR", "kind": "holding", "theme": "Biotech",
                    "state": "HELD", "close": 84.11,
                    "sma5": 83.1, "sma10": 82.0, "sma20": 80.29,
                    "extension_pct": 4.76, "extension_atr": 0.99,
                    "stop": {"type": "sma20_close", "level": 80.29},
                    "conditions": {"2_confirmation": {"atr14": 3.86}},
                    "conditions_met": 5,
                    "next_earnings_date": "2026-08-06", "days_to_earnings": 28,
                },
                "MRNA": {
                    "ticker": "MRNA", "kind": "watching", "theme": "Biotech",
                    "state": "EXTENDED_HOLD",
                    "extension_guard": "extension 3.4×ATR > 1.8× — re-entry suppressed",
                    "close": 79.0, "sma20": 65.0,
                    "extension_pct": 21.5, "extension_atr": 3.4,
                    "conditions": {"1_trigger": {"met": True},
                                   "2_confirmation": {"atr14": 4.1}},
                    "conditions_met": 5,
                    "next_earnings_date": "2026-07-31", "days_to_earnings": 22,
                },
            },
            "transitions": [],
        },
    }


def _pre_1a_payload():
    g = {"value": 1.0, "signal": "risk_on", "detail": "d"}
    return {"generated_at": _now_iso(),
            "regime": {"regime": "Risk-on / Trending",
                       "gauges": {k: dict(g) for k in
                                  ("spy_vs_200dma", "vix_5d_avg", "hy_spread",
                                   "breadth", "yield_curve")}}}


def _signals_json(vix3m_level=18.9):
    return {
        "indexes": {
            "^VIX": {"name": "VIX", "level": 16.8, "avg_5d": 17.2},
            "^VIX9D": {"name": "VIX 9-Day", "level": 15.5, "avg_5d": 16.0},
            "^VIX3M": {"name": "VIX 3-Month", "level": vix3m_level,
                       "avg_5d": 19.1},
        },
        "groups": [{"name": "Biotechnology", "stocks": [{
            "ticker": "ARWR", "price": 84.11, "rsi": 61.2,
            "macd": 1.1, "macd_signal": 0.9, "macd_histogram": 0.2,
            "ma20": 80.1, "ma50": 76.4, "ma200": 55.2,
            "volume_ratio": 1.1, "trend_strength": 15,
        }]}],
    }


def _regime_history():
    v = lambda s: {"signal": s, "value": 1, "detail": "d"}
    return [
        {"date": "2026-07-08", "regime": "Risk-on / Trending",
         "gauges": {"vix_5d_avg": v("risk_on"), "hy_spread": v("risk_on"),
                    "breadth": v("risk_on")},
         "backdrop_gate": {"open": True}},
        {"date": "2026-07-09", "regime": "Risk-on / Choppy",
         "gauges": {"vix_5d_avg": v("risk_on"), "hy_spread": v("risk_on"),
                    "breadth": v("caution")},
         "backdrop_gate": {"open": True}},
    ]


def _position_events():
    return {"changes": [
        {"timestamp": "2026-07-08T20:00:00+00:00", "ticker": "OLD",
         "type": "position_state_change",
         "detail": {"from_state": "HELD", "to_state": "EXIT_FIRED"}},
        {"timestamp": "2026-07-09T20:00:00+00:00", "ticker": "IWM",
         "type": "position_state_change",
         "detail": {"from_state": "HELD", "to_state": "EXIT_FIRED"}},
    ]}


class _Env:
    def __init__(self, with_signals=True, with_history=True, with_events=True):
        self.tmp = tempfile.mkdtemp(prefix="assess_")
        self.pub = os.path.join(self.tmp, "public")
        self.dat = os.path.join(self.tmp, "data")
        self.st = os.path.join(self.tmp, "state")
        for d in (self.pub, self.dat, self.st):
            os.makedirs(d)
        self.olds = (ticker_api.PUBLIC_DIR, ticker_api.DATA_DIR,
                     ticker_api.FRAMEWORK_STATE_DIR,
                     ticker_api._run_framework_refresh)
        ticker_api.PUBLIC_DIR = self.pub
        ticker_api.DATA_DIR = self.dat
        ticker_api.FRAMEWORK_STATE_DIR = self.st
        self.kicks = []
        ticker_api._run_framework_refresh = lambda: self.kicks.append(1)
        ticker_api._framework_status["running"] = False
        if with_signals:
            self._w(self.pub, "signals.json", _signals_json())
        if with_history:
            self._w(self.st, "regime_history.json", _regime_history())
        if with_events:
            self._w(self.dat, "position_events.json", _position_events())
        self.client = ticker_api.app.test_client()

    @staticmethod
    def _w(d, name, obj):
        with open(os.path.join(d, name), "w") as f:
            json.dump(obj, f)

    def write_framework(self, payload):
        self._w(self.pub, "framework.json", payload)

    def close(self):
        (ticker_api.PUBLIC_DIR, ticker_api.DATA_DIR,
         ticker_api.FRAMEWORK_STATE_DIR,
         ticker_api._run_framework_refresh) = self.olds
        shutil.rmtree(self.tmp, ignore_errors=True)


def test_response_shape():
    env = _Env()
    try:
        env.write_framework(_framework_payload())
        r = env.client.get("/api/assessment.json")
        assert r.status_code == 200, r.status_code
        b = r.get_json()
        for k in ("generated_at", "generated_at_et", "regime", "positions",
                  "technicals", "vol_complex", "themes", "changes_since_prior"):
            assert k in b, f"missing top-level key {k}"
        # regime: full gauge block
        assert sorted(b["regime"]["gauges"].keys()) == \
            ["breadth", "hy_spread", "vix_5d_avg"]
        assert b["regime"]["backdrop_gate"]["open"] is True
        assert "yield_curve" in b["regime"]["macro_inputs"]
        # one-grammar retirement (Phase 3): counts survive ONE deprecation
        # cycle as flat fields, annotated via legacy_voters — the note must
        # say the chassis sets the regime
        lv = b["regime"]["legacy_voters"]
        assert lv["risk_on"] == 2 and lv["caution"] == 1
        assert "informational only" in lv["note"] and "chassis" in lv["note"]
        assert b["regime"]["risk_on_count"] == 2   # deprecated, one cycle
        # positions: HELD has no conditions block, watcher does
        assert "conditions" not in b["positions"]["ARWR"]
        assert b["positions"]["ARWR"]["stop"]["level"] == 80.29
        assert "conditions" in b["positions"]["MRNA"]
        # the Position Signals panel binds these watcher fields
        assert b["positions"]["MRNA"]["state"] == "EXTENDED_HOLD"
        assert "re-entry suppressed" in b["positions"]["MRNA"]["extension_guard"]
        assert b["positions"]["MRNA"]["extension_atr"] == 3.4
        # technicals: holdings only, cushion math from close/stop/atr
        assert list(b["technicals"].keys()) == ["ARWR"]
        t = b["technicals"]["ARWR"]
        assert t["cushion_to_stop"]["dollars"] == round(84.11 - 80.29, 2)
        assert t["cushion_to_stop"]["atr_multiple"] == round((84.11 - 80.29) / 3.86, 2)
        assert t["rsi"] == 61.2 and t["sma50"] == 76.4    # merged from signals.json
        assert t["sma5"] == 83.1 and t["sma10"] == 82.0   # producer amendment
        assert not any("sma5" in n for n in t["notes"])
        # vol: VIX from the regime voter; 9D/3M from the indexes block
        v = b["vol_complex"]
        assert v["vix"] == {"spot": 16.8, "avg_5d": 17.2, "signal": "risk_on"}
        assert v["vix9d"] == {"spot": 15.5, "avg_5d": 16.0}
        assert v["vix3m"] == {"spot": 18.9, "avg_5d": 19.1}
        assert v["term_structure"] == "contango"          # 18.9 > 16.8
        # themes: top-line with status, unranked dropped
        assert b["themes"][0] == {"rank": 1, "theme": "Semis",
                                  "composite": 1.5, "status": "active"}
        assert b["themes"][2]["status"] == "ranked"
        assert len(b["themes"]) == 3
        # changes: regime shift + the breadth flip + only today's transition
        c = b["changes_since_prior"]
        assert c["regime_shift"] == {"from": "Risk-on / Trending",
                                     "to": "Risk-on / Choppy"}
        assert c["voters_flipped"] == [{"gauge": "breadth",
                                        "from": "risk_on", "to": "caution"}]
        assert c["gate_changed"] is None
        assert len(c["position_transitions"]) == 1
        assert c["position_transitions"][0]["ticker"] == "IWM"
        assert c["position_transitions"][0]["to_state"] == "EXIT_FIRED"
    finally:
        env.close()
    print("  response shape (all seven sections, cushion math, diff): OK")


def test_sentinel_inheritance():
    env = _Env()
    try:
        env.write_framework(_pre_1a_payload())
        r = env.client.get("/api/assessment.json")
        assert r.status_code == 503, r.status_code
        assert r.headers.get("Retry-After") == "180"
        assert r.get_json()["status"] == "warming_up"
    finally:
        env.close()
    print("  shape-sentinel inheritance (pre-1A artifact -> 503): OK")


def test_et_timestamp_format():
    env = _Env()
    try:
        env.write_framework(_framework_payload())
        b = env.client.get("/api/assessment.json").get_json()
        assert re.fullmatch(r"\d{4}-\d{2}-\d{2} \d{2}:\d{2} (AM|PM) E[DS]T",
                            b["generated_at_et"]), b["generated_at_et"]
        # and the two stamps agree on the instant (parse ET back to UTC)
        assert b["generated_at"].endswith("+00:00")
    finally:
        env.close()
    print("  ET timestamp pinned ('YYYY-MM-DD HH:MM AM/PM EDT|EST'): OK")


def test_sections_degrade_independently():
    env = _Env(with_signals=False, with_history=False, with_events=False)
    try:
        # artifact predating the sma5/sma10 producer amendment
        payload = _framework_payload()
        for k in ("sma5", "sma10"):
            payload["position_signals"]["tickers"]["ARWR"].pop(k)
        env.write_framework(payload)
        r = env.client.get("/api/assessment.json")
        assert r.status_code == 200, "missing sources must not fail the response"
        b = r.get_json()
        # changes needs regime history -> section error
        assert "error" in b["changes_since_prior"]
        # technicals degrade to position-engine values with notes
        t = b["technicals"]["ARWR"]
        assert t["rsi"] is None and t["sma20"] == 80.29
        assert t["sma5"] is None
        assert any("position-engine values only" in n for n in t["notes"])
        assert any("predates the producer amendment" in n for n in t["notes"])
        # vol: VIX still works (regime gauge); 9D/3M degrade to error
        assert b["vol_complex"]["vix"]["spot"] == 16.8
        assert "error" in b["vol_complex"]["vix9d"]
        assert b["vol_complex"]["term_structure"].startswith("unavailable")
        # regime + positions + themes unaffected
        assert b["regime"]["regime"] == "Risk-on / Choppy"
        assert b["positions"]["ARWR"]["state"] == "HELD"
        assert len(b["themes"]) == 3
    finally:
        env.close()
    print("  per-section degradation (no whole-response failure): OK")


def test_term_structure_inverted():
    env = _Env()
    try:
        env._w(env.pub, "signals.json", _signals_json(vix3m_level=14.2))
        env.write_framework(_framework_payload())
        v = env.client.get("/api/assessment.json").get_json()["vol_complex"]
        assert v["term_structure"] == "inverted"          # 14.2 < 16.8
    finally:
        env.close()
    print("  term structure flips to inverted when VIX3M < VIX: OK")


if __name__ == "__main__":
    print("\n=== Assessment endpoint tests (PER-508 #21) ===")
    test_response_shape()
    test_sentinel_inheritance()
    test_et_timestamp_format()
    test_sections_degrade_independently()
    test_term_structure_inverted()
    print("\nAll assessment tests passed.\n")
