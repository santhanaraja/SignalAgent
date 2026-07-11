#!/usr/bin/env python3
"""
Tests for PER-508 item 24b — Position Lab.

Pins per the ticket, all through the endpoint (the lab owns zero math —
/api/position/simulate calls the extracted assess_position, the same
function evaluate() now delegates to): extension guard EXTENDED_HOLD,
healthy reclaim RE_ENTRY_READY, holding exit path, falling-slope failure,
Choppy a_plus_only. Plus the bit-identical replay pin: the committed
framework.json rows (produced by the PRE-refactor engine) reproduce
through the extracted path.

Run: python3 test_position_lab.py   (pipeline venv)
"""

import json
import os
import sys

sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))

import ticker_api
from framework.position_signals import assess_position

REPO = os.path.dirname(os.path.abspath(__file__))
C = ticker_api.app.test_client()

BASE = {"close": 66.0, "sma20": 65.0, "sma20_5d_ago": 64.0, "atr14": 4.0,
        "consecutive_closes_above": 3, "regime_state": "Risk-on / Trending",
        "theme_qualified": True, "kind": "watching"}


def _sim(**over):
    body = dict(BASE, **over)
    r = C.post("/api/position/simulate", json=body)
    assert r.status_code == 200, r.get_json()
    return r.get_json()


def test_ticket_pins():
    # healthy reclaim: <0.5xATR above, all five met -> RE_ENTRY_READY
    d = _sim()
    assert d["state"] == "RE_ENTRY_READY"
    assert d["all_conditions_met"] and d["conditions_met"] == 5
    assert d["extension_atr"] == 0.25 and not d.get("extension_guard")

    # extension >1.8xATR on a watching name -> EXTENDED_HOLD, guard verdict
    d = _sim(close=73.2)                       # (73.2-65)/4 = 2.05xATR
    assert d["state"] == "EXTENDED_HOLD"
    assert "re-entry suppressed" in d["extension_guard"]
    assert d["all_conditions_met"]             # conditions still evaluate

    # a holding with close < sma20 -> the exit path
    d = _sim(kind="holding", close=63.0)
    assert d["state"] == "EXIT_FIRED"
    assert not d["conditions"]["1_trigger"]["met"]
    assert d["distance_to_sma20_pct"] < 0

    # falling sma20 (5d-ago higher) -> condition 4 fails
    d = _sim(sma20_5d_ago=66.5)
    assert not d["conditions"]["4_slope"]["met"]
    assert d["state"] == "RE_ENTRY_ARMING"     # 4/5 met, above SMA20

    # regime Choppy -> conditional gate, a_plus_only on READY
    d = _sim(regime_state="Risk-on / Choppy")
    assert d["state"] == "RE_ENTRY_READY" and d["a_plus_only"] is True
    assert d["conditions"]["3_regime_gate"]["mode"] == "conditional"

    # regime Caution -> gate blocked, never READY
    d = _sim(regime_state="Caution")
    assert not d["conditions"]["3_regime_gate"]["met"]
    assert d["state"] == "RE_ENTRY_ARMING"

    # holding whose conditions complete -> HELD (not READY)
    d = _sim(kind="holding")
    assert d["state"] == "HELD"
    print("  ticket pins: guard/reclaim/exit/slope/a_plus_only/HELD: OK")


def test_replay_committed_artifact():
    """Bit-identical pin: rows the PRE-refactor engine produced (committed
    framework.json) reproduce through the extracted assess_position.
    Inputs reconstructed from the recorded row; sma20_5d_ago synthesized
    on the recorded slope verdict (its exact value is not persisted
    pre-refactor — the met-flags and state are the pin)."""
    fw = json.load(open(os.path.join(REPO, "public", "framework.json")))
    rows = (fw.get("position_signals") or {}).get("tickers") or {}
    regime = fw["regime"]["regime"]
    checked = 0
    for t, row in rows.items():
        conds = row.get("conditions") or {}
        if not conds or row.get("insufficient_data"):
            continue
        c2, c4, c5 = conds["2_confirmation"], conds["4_slope"], conds["5_thesis"]
        inputs = row.get("assess_inputs") or {
            "close": row["close"], "sma20": row["sma20"],
            "sma20_5d_ago": row["sma20"] + (-1.0 if c4["met"] else 1.0),
            "atr14": c2.get("atr14"),
            "consecutive_closes_above": c2.get("consecutive_closes_above", 0),
            "regime_state": regime, "theme_qualified": c5["met"],
            "kind": row.get("kind", "watching"),
        }
        got = assess_position(**inputs)
        assert got["state"] == row["state"], \
            f"{t}: {got['state']} != recorded {row['state']}"
        for k in ("1_trigger", "2_confirmation", "3_regime_gate", "5_thesis"):
            assert got["conditions"][k]["met"] == conds[k]["met"], f"{t}:{k}"
        assert got["conditions"]["4_slope"]["met"] == c4["met"], t
        checked += 1
    assert checked >= 2, f"only {checked} tracked tickers replayed"
    print(f"  replay pin: {checked} committed tracked tickers reproduce: OK")


def test_validation():
    r = C.post("/api/position/simulate", data="junk",
               content_type="text/plain")
    assert r.status_code == 400
    r = C.post("/api/position/simulate", json=dict(BASE, kind="short"))
    assert r.status_code == 400
    r = C.post("/api/position/simulate", json=dict(BASE, close="high"))
    assert r.status_code == 400
    r = C.post("/api/position/simulate", json=dict(BASE, close=True))
    assert r.status_code == 400
    r = C.post("/api/position/simulate",
               json=dict(BASE, regime_state=["Caution"]))
    assert r.status_code == 400
    r = C.post("/api/position/simulate", json=dict(BASE, atr14=None))
    d = r.get_json()                            # ATR unknown: legal, degrades
    assert r.status_code == 200 and d["extension_atr"] is None
    # subnormal ATR must not divide to Infinity and emit a bare JSON token
    # the browser cannot parse (review finding)
    r = C.post("/api/position/simulate", json=dict(BASE, atr14=1e-320))
    assert r.status_code == 200
    d = json.loads(r.get_data(as_text=True))    # strict parse: no Infinity
    assert d["extension_atr"] is None
    r = C.post("/api/position/simulate",
               json=dict(BASE, consecutive_closes_above=2.5))
    assert r.status_code == 400                 # must be an integer
    print("  validation: junk/kind/string/bool/list 400; null ATR degrades: OK")


if __name__ == "__main__":
    print("\n=== Position Lab tests (PER-508 #24b) ===")
    test_ticket_pins()
    test_replay_committed_artifact()
    test_validation()
    print("\nAll Position Lab tests passed.\n")
