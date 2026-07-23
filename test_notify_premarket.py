#!/usr/bin/env python3
"""
Tests for the pre-market briefing push (PER-508 item 23).

Pins: formatting (gap-breach day with the no-pre-emption discipline
line, warn band, calm day, empty holdings, fetch-failure degradation),
the AM marker matrix, absent secret, and the never-fail/never-leak
contract shared with item 22.

Run: python3 test_notify_premarket.py
"""

import contextlib
import datetime
import io
import json
import os
import shutil
import sys
import tempfile

sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))

import notify_premarket as pm


def _et(iso_dt):
    from zoneinfo import ZoneInfo
    return datetime.datetime.fromisoformat(iso_dt).replace(
        tzinfo=ZoneInfo("America/New_York"))


NOW = "2026-07-10T07:05:00"


def _artifact(state="HELD"):
    return {
        "generated_at": "2026-07-09T20:23:00+00:00",
        "regime": {"regime": "Risk-on / Choppy", "risk_on_count": 2,
                   "caution_count": 1, "risk_off_count": 0},
        "position_signals": {"tickers": {
            "ARWR": {"ticker": "ARWR", "kind": "holding", "state": state,
                     "close": 76.68, "sma20": 80.59,
                     "stop": {"type": "sma20_close", "level": 80.59},
                     "conditions": {"2_confirmation": {"atr14": 4.0}}},
            "MRNA": {"ticker": "MRNA", "kind": "watching",
                     "state": "RE_ENTRY_READY", "close": 66.9,
                     "sma20": 64.5, "conditions": {}},
        }},
    }


POSITIONS = {"holdings": [{"ticker": "ARWR", "shares": 71,
                           "stop_on_entry": "sma20_close"}],
             "watching": [{"ticker": "MRNA"}]}

QUOTES_OK = {
    "ES=F": (6412.25, 0.4), "NQ=F": (23890.0, 0.7), "RTY=F": (2280.5, -0.2),
    "CL=F": (71.10, 1.2), "GC=F": (3325.0, -0.3),
    "^VIX": (15.3, None), "^VIX9D": (12.4, None), "^VIX3M": (18.9, None),
    "ARWR": (76.10, -0.8),
}


def test_gap_breach_day():
    msg = pm.build_message(_artifact(state="WATCHING"), POSITIONS,
                           QUOTES_OK, False, _et(NOW))
    assert "*SignalAgent PRE-MARKET* — 2026-07-10 07:05 AM ET" in msg
    assert "Futures: ES 6,412.25 +0.4% · NQ 23,890.00 +0.7%" in msg
    assert "CL 71.10 +1.2%" in msg and "GC 3,325.00 -0.3%" in msg
    assert "Vol: VIX 15.30 · 9D 12.40 · 3M 18.90 · contango" in msg
    line = next(l for l in msg.splitlines() if "ARWR" in l and "stop" in l)
    assert line.startswith("🔴 "), line
    assert "(WATCHING)" in line            # artifact state surfaced
    assert "$76.10 vs stop $80.59" in line
    assert "below stop pre-market — close decides, no pre-emption" in line
    assert "Watchlist: MRNA RE_ENTRY_READY" in msg
    # one-grammar retirement (Phase 3): no parliament counts in the line
    assert "Regime: Risk-on / Choppy — as of last close (2026-07-09)" in msg
    assert "(2/1/0)" not in msg
    print("  gap-breach day: 🔴 + no-pre-emption discipline line: OK")


def test_warn_band_and_calm_day():
    quotes = dict(QUOTES_OK)
    quotes["ARWR"] = (82.0, 1.5)            # stop 80.59 + 0.5*4.0 = 82.59
    msg = pm.build_message(_artifact(), POSITIONS, quotes, False, _et(NOW))
    line = next(l for l in msg.splitlines() if "ARWR" in l and "stop" in l)
    assert line.startswith("⚠️ ") and "within 0.5×ATR of stop" in line
    quotes["ARWR"] = (84.5, 2.0)            # comfortably above the band
    msg = pm.build_message(_artifact(), POSITIONS, quotes, False, _et(NOW))
    line = next(l for l in msg.splitlines() if "ARWR" in l and "stop" in l)
    assert not line.startswith(("🔴", "⚠️"))
    assert "(HELD)" not in line             # HELD is the default, not echoed
    print("  warn band (⚠️ within 0.5×ATR) and calm day: OK")


def test_exited_holding_falls_back_to_sma20_reference():
    """A holding whose exit already fired has no active stop in the
    artifact — the line references the SMA20 reclaim level, unflagged
    (the alarm was yesterday's close report)."""
    art = _artifact(state="WATCHING")
    del art["position_signals"]["tickers"]["ARWR"]["stop"]
    msg = pm.build_message(art, POSITIONS, QUOTES_OK, False, _et(NOW))
    line = next(l for l in msg.splitlines() if "ARWR" in l)
    assert not line.startswith(("🔴", "⚠️")), line
    assert "(WATCHING)" in line
    assert "vs SMA20 $80.59" in line
    assert "exit signaled; reclaim = close above SMA20" in line
    print("  exited holding (no active stop): SMA20 reclaim reference: OK")


def test_empty_holdings_still_sends():
    msg = pm.build_message(_artifact(), {"holdings": [], "watching": []},
                           QUOTES_OK, False, _et(NOW))
    assert "• none — cash" in msg
    assert "Futures: ES 6,412.25" in msg    # the value survives
    print("  empty-holdings day still carries futures + vol: OK")


def test_fetch_failure_degradation():
    # every quote failed: message still goes out on last-close data
    quotes = {s: None for s in QUOTES_OK}
    msg = pm.build_message(_artifact(state="WATCHING"), POSITIONS,
                           quotes, True, _et(NOW))
    assert "_live quotes unavailable — showing last-close data_" in msg
    assert "Futures: ES n/a · NQ n/a" in msg
    assert "· n/a" in msg                   # term structure degraded
    line = next(l for l in msg.splitlines() if "ARWR" in l and "stop" in l)
    assert "(last close)" in line           # artifact close used
    assert line.startswith("🔴 ")           # 76.68 < 80.59 still flags
    print("  all-fetch-failure: sends anyway on last-close data: OK")


def test_am_marker_matrix():
    tmp = tempfile.mkdtemp(prefix="pm_marker_")
    marker = os.path.join(tmp, "last_notified_am")
    try:
        ok, why = pm.should_notify(_et("2026-07-10T06:15:00"), marker)
        assert not ok and "outside pre-market" in why
        ok, _ = pm.should_notify(_et("2026-07-10T06:30:00"), marker)
        assert ok
        ok, _ = pm.should_notify(_et(NOW), marker)
        assert ok
        pm.write_marker(_et(NOW), marker)
        ok, why = pm.should_notify(_et("2026-07-10T08:00:00"), marker)
        assert not ok and "already briefed" in why
        ok, _ = pm.should_notify(_et("2026-07-13T07:00:00"), marker)  # Monday
        assert ok
        ok, why = pm.should_notify(_et("2026-07-10T09:30:00"), marker)
        assert not ok and "outside pre-market" in why    # market open
        ok, why = pm.should_notify(_et("2026-07-11T07:00:00"), marker)
        assert not ok and "weekend" in why               # Saturday
    finally:
        shutil.rmtree(tmp, ignore_errors=True)
    print("  AM marker matrix: window edges, once-per-day, weekend: OK")


def test_absent_secret_and_failure_contract():
    old_env = dict(os.environ)
    old_post, old_now, old_fetch = pm.post_to_slack, pm._now_et, pm.fetch_quotes
    old_marker, old_fw, old_pos = pm.MARKER_PATH, pm.FRAMEWORK_JSON, pm.POSITIONS_JSON
    tmp = tempfile.mkdtemp(prefix="pm_main_")
    try:
        pm.MARKER_PATH = os.path.join(tmp, "last_notified_am")
        pm.FRAMEWORK_JSON = os.path.join(tmp, "framework.json")
        pm.POSITIONS_JSON = os.path.join(tmp, "positions.json")
        json.dump(_artifact(), open(pm.FRAMEWORK_JSON, "w"))
        json.dump(POSITIONS, open(pm.POSITIONS_JSON, "w"))
        pm._now_et = lambda: _et(NOW)
        pm.fetch_quotes = lambda syms: ({s: QUOTES_OK.get(s) for s in syms}, False)

        # absent secret: clean exit, nothing attempted
        os.environ.pop("ASSESSMENT_WEBHOOK_URL", None)
        posted = []
        pm.post_to_slack = lambda *a: posted.append(a)
        out = io.StringIO()
        with contextlib.redirect_stdout(out):
            rc = pm.main()
        assert rc == 0 and posted == []
        assert "no webhook configured" in out.getvalue()

        # failure: rc=0, URL scrubbed, marker not written
        secret = "https://hooks.slack.com/services/T0/B0/PREMARKETSECRET"
        os.environ["ASSESSMENT_WEBHOOK_URL"] = secret

        def boom(url, text):
            raise RuntimeError(f"refused for url: {url}")
        pm.post_to_slack = boom
        out = io.StringIO()
        with contextlib.redirect_stdout(out):
            rc = pm.main()
        assert rc == 0
        assert secret not in out.getvalue() and "<webhook>" in out.getvalue()
        assert not os.path.exists(pm.MARKER_PATH)

        # success: marker written
        sent = []
        pm.post_to_slack = lambda url, text: sent.append(text)
        with contextlib.redirect_stdout(io.StringIO()):
            rc = pm.main()
        assert rc == 0 and len(sent) == 1
        assert "PRE-MARKET" in sent[0]
        assert open(pm.MARKER_PATH).read() == "2026-07-10"
    finally:
        os.environ.clear()
        os.environ.update(old_env)
        pm.post_to_slack, pm._now_et, pm.fetch_quotes = old_post, old_now, old_fetch
        pm.MARKER_PATH, pm.FRAMEWORK_JSON, pm.POSITIONS_JSON = old_marker, old_fw, old_pos
        shutil.rmtree(tmp, ignore_errors=True)
    print("  absent secret / never-fail / never-leak / marker-on-success: OK")


if __name__ == "__main__":
    print("\n=== Pre-market briefing tests (PER-508 #23) ===")
    test_gap_breach_day()
    test_warn_band_and_calm_day()
    test_exited_holding_falls_back_to_sma20_reference()
    test_empty_holdings_still_sends()
    test_fetch_failure_degradation()
    test_am_marker_matrix()
    test_absent_secret_and_failure_contract()
    print("\nAll pre-market briefing tests passed.\n")
