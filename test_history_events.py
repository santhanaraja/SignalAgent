#!/usr/bin/env python3
"""
Tests for PER-508 item 18 — the four new history event types.

Pins: breaker_change (any status change, reason extraction, severity
ladder), trade_signal_change (pole-subset semantics — HOLD↔ACCUMULATE
churn must stay silent), gate_crossing (>=50 boundary in both directions,
NaN tolerance for pre-sanitizer snapshots), regime_change (voter counts,
bootstrap silence), detect_changes integration, and backfill idempotence.

Run: python3 test_history_events.py
"""

import json
import os
import shutil
import sys
import tempfile

sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))

from history_manager import (detect_changes, detect_coverage_events,
                             detect_regime_change)
import backfill_history_events as bf


def _stock(ticker, score=60, trade_signal="HOLD POSITION", signal="hold"):
    return {"ticker": ticker, "score": score, "trade_signal": trade_signal,
            "signal": signal, "ytd_return": 5.0, "rsi": 55}


def _group(name, breaker_status="clear", alerts=None, stocks=None, rank=1):
    return {"name": name, "breaker_status": breaker_status,
            "breaker_alerts": alerts or [], "group_signal": "hold",
            "avg_ytd": 10.0, "avg_score": 60, "rank": rank, "stock_count": 1,
            "stocks": stocks if stocks is not None else [_stock("AAA")]}


def _snap(ts, groups):
    return {"timestamp": ts, "groups": groups, "total_groups": len(groups),
            "total_tickers": sum(len(g["stocks"]) for g in groups),
            "sp500_ytd": 3.0}


T1, T2 = "2026-07-01T14:00:00+00:00", "2026-07-01T16:00:00+00:00"


def _events(prev_groups, curr_groups, etype=None):
    ev = detect_coverage_events(_snap(T1, prev_groups), _snap(T2, curr_groups))
    return [e for e in ev if etype is None or e["type"] == etype]


def test_breaker_change():
    alerts = [{"check": "sp500_drawdown_10pct", "triggered": True,
               "message": "S&P down 11% from high", "description": "x"},
              {"check": "rsi", "triggered": False, "message": "Not triggered"}]
    ev = _events([_group("Semis", "clear")],
                 [_group("Semis", "critical", alerts=alerts)], "breaker_change")
    assert len(ev) == 1
    e = ev[0]
    assert e["severity"] == "critical" and e["ticker"] is None
    assert e["description"] == "Semis: Breaker CLEAR → CRITICAL"
    assert e["detail"]["reason"] == "S&P down 11% from high"
    assert e["timestamp"] == T2

    # severity ladder + clear-side reason fallback
    ev = _events([_group("Semis", "critical")], [_group("Semis", "warning")],
                 "breaker_change")
    assert ev[0]["severity"] == "high"
    ev = _events([_group("Semis", "warning")], [_group("Semis", "clear")],
                 "breaker_change")
    assert ev[0]["severity"] == "medium"
    assert ev[0]["detail"]["reason"] == "all breaker checks clear"
    ev = _events([_group("Semis", "clear")], [_group("Semis", "watch")],
                 "breaker_change")
    assert ev[0]["severity"] == "medium"          # watch is a real state

    # unchanged / missing (old schema) -> silent
    assert not _events([_group("Semis", "clear")], [_group("Semis", "clear")],
                       "breaker_change")
    old = _group("Semis"); del old["breaker_status"]; del old["breaker_alerts"]
    assert not _events([old], [_group("Semis", "warning")], "breaker_change")
    print("  breaker_change: transitions, severities, reasons, old schema: OK")


def test_trade_signal_change_pole_subset():
    def one(pts, cts):
        return _events(
            [_group("Semis", stocks=[_stock("NVDA", trade_signal=pts)])],
            [_group("Semis", stocks=[_stock("NVDA", trade_signal=cts)])],
            "trade_signal_change")

    e = one("WAIT FOR PULLBACK", "BUY NOW")
    assert len(e) == 1 and e[0]["severity"] == "high"
    assert e[0]["description"] == "NVDA: Trade signal WAIT FOR PULLBACK → BUY NOW"
    e = one("BUY NOW", "ACCUMULATE ON DIP")
    assert len(e) == 1 and e[0]["severity"] == "medium"   # leaving a pole
    e = one("BUY NOW", "AVOID")
    assert len(e) == 1 and e[0]["severity"] == "critical"  # pole-to-pole
    e = one("ACCUMULATE ON DIP", "AVOID")
    assert len(e) == 1 and e[0]["severity"] == "high"

    # the noise the spec excludes
    assert not one("HOLD POSITION", "ACCUMULATE ON DIP")
    assert not one("ACCUMULATE ON DIP", "WAIT FOR PULLBACK")
    assert not one("BUY NOW", "BUY NOW")
    # old schema without trade_signal -> silent
    prev = _group("Semis", stocks=[{"ticker": "NVDA", "score": 60}])
    assert not _events([prev],
                       [_group("Semis", stocks=[_stock("NVDA", trade_signal="AVOID")])],
                       "trade_signal_change")
    print("  trade_signal_change: pole subset, severities, churn excluded: OK")


def test_gate_crossing():
    def one(ps, cs):
        return _events([_group("Semis", stocks=[_stock("AMD", score=ps)])],
                       [_group("Semis", stocks=[_stock("AMD", score=cs)])],
                       "gate_crossing")

    up = one(49, 57)
    assert len(up) == 1 and up[0]["detail"]["direction"] == "up"
    assert up[0]["description"] == "AMD: Score crossed above 50 (49 → 57)"
    down = one(50, 49)                       # 50 itself is qualified (>=)
    assert len(down) == 1 and down[0]["detail"]["direction"] == "down"
    assert len(one(49, 50)) == 1             # landing exactly on the line
    assert not one(55, 99) and not one(10, 49) and not one(50, 50)
    assert not one(float("nan"), 57)          # pre-sanitizer snapshots
    assert not one(49, float("nan"))
    print("  gate_crossing: >=50 boundary both ways, NaN tolerated: OK")


def test_group_rename_does_not_swallow_ticker_transitions():
    """Stock identity is the ticker, not group|ticker — the Feb 16-17
    re-clustering renamed groups and must not eat crossings/pole flips
    (review finding; 18 real gate events were dropped by the old keying)."""
    prev = [_group("Gold", stocks=[_stock("KGC", score=63,
                                          trade_signal="BUY NOW")])]
    curr = [_group("Gold Mining / Precious Metals",
                   stocks=[_stock("KGC", score=35, trade_signal="AVOID")])]
    gate = _events(prev, curr, "gate_crossing")
    assert len(gate) == 1 and gate[0]["detail"]["direction"] == "down"
    assert gate[0]["group"] == "Gold Mining / Precious Metals"
    trade = _events(prev, curr, "trade_signal_change")
    assert len(trade) == 1 and trade[0]["severity"] == "critical"
    # breaker identity legitimately IS the group name — rename stays silent
    assert not _events([_group("Gold", "warning")],
                       [_group("Gold Mining / Precious Metals", "clear")],
                       "breaker_change")
    print("  group rename: ticker transitions survive, breaker silent: OK")


def test_regime_change():
    prev = {"regime": "Risk-on / Trending", "risk_on_count": 3,
            "caution_count": 0, "risk_off_count": 0}
    curr = {"regime": "Risk-on / Choppy", "risk_on_count": 2,
            "caution_count": 1, "risk_off_count": 0}
    ev = detect_regime_change(prev, curr, T2)
    assert len(ev) == 1
    e = ev[0]
    assert e["severity"] == "high" and e["group"] is None
    # one-grammar retirement (Phase 3): the description carries NO
    # parliament counts — the chassis sets the regime; counts stay in
    # detail as data (demoted at render)
    assert e["description"] == ("Swing regime: Risk-on / Trending → "
                                "Risk-on / Choppy")
    assert "(2/1/0)" not in e["description"]
    assert e["detail"]["risk_on_count"] == 2

    off = dict(curr, regime="Risk-off")
    assert detect_regime_change(curr, off, T2)[0]["severity"] == "critical"
    assert not detect_regime_change(None, curr, T2)        # bootstrap: silent
    assert not detect_regime_change(prev, dict(prev), T2)  # unchanged
    assert not detect_regime_change(prev, {"gauges": {}}, T2)  # malformed
    print("  regime_change: counts, Risk-off critical, bootstrap silent: OK")


def test_detect_changes_integration():
    prev = _snap(T1, [_group("Semis", "clear",
                             stocks=[_stock("NVDA", score=49,
                                            trade_signal="WAIT FOR PULLBACK")])])
    curr = _snap(T2, [_group("Semis", "warning",
                             alerts=[{"check": "c", "triggered": True,
                                      "message": "RSI broke down"}],
                             stocks=[_stock("NVDA", score=57,
                                            trade_signal="BUY NOW")])])
    types = {c["type"] for c in detect_changes(prev, curr)}
    assert {"breaker_change", "trade_signal_change", "gate_crossing"} <= types
    print("  detect_changes integration: coverage events ride along: OK")


def test_backfill_normalize_and_idempotence():
    assert bf._normalize_ts("2026-02-17T15:06:02.260489") == \
        "2026-02-17T15:06:02.260489+00:00"
    assert bf._normalize_ts(T2) == T2                       # already aware
    assert bf._normalize_ts("2026-02-17T15:06:02Z") == "2026-02-17T15:06:02Z"

    tmp = tempfile.mkdtemp(prefix="bf_test_")
    old = (bf.SNAPSHOTS_DIR, bf.PUBLIC_SNAPSHOTS_DIR, bf.REGIME_HISTORY,
           bf.DATA_DIR, bf.PUBLIC_DIR)
    try:
        bf.SNAPSHOTS_DIR = os.path.join(tmp, "data", "snapshots")
        bf.PUBLIC_SNAPSHOTS_DIR = os.path.join(tmp, "public", "snapshots")
        bf.REGIME_HISTORY = os.path.join(tmp, "regime_history.json")
        bf.DATA_DIR = os.path.join(tmp, "data")
        bf.PUBLIC_DIR = os.path.join(tmp, "public")
        os.makedirs(bf.SNAPSHOTS_DIR)
        os.makedirs(bf.PUBLIC_SNAPSHOTS_DIR)

        s1 = _snap("2026-07-01T14:00:00+00:00",
                   [_group("Semis", "clear",
                           stocks=[_stock("NVDA", score=49)])])
        s2 = _snap("2026-07-01T16:00:00+00:00",
                   [_group("Semis", "warning",
                           alerts=[{"check": "c", "triggered": True,
                                    "message": "m"}],
                           stocks=[_stock("NVDA", score=57)])])
        for name, s in (("snapshot_20260701_140000.json", s1),
                        ("snapshot_20260701_160000.json", s2)):
            json.dump(s, open(os.path.join(bf.SNAPSHOTS_DIR, name), "w"))
        json.dump([{"date": "2026-06-30", "regime": "Caution",
                    "risk_on_count": 1, "caution_count": 2, "risk_off_count": 0},
                   {"date": "2026-07-01", "regime": "Risk-on / Choppy",
                    "risk_on_count": 2, "caution_count": 1, "risk_off_count": 0}],
                  open(bf.REGIME_HISTORY, "w"))

        sys.argv = ["backfill_history_events.py", "--apply"]
        assert bf.main() == 0
        h = json.load(open(os.path.join(bf.DATA_DIR, "history.json")))
        types = sorted(c["type"] for c in h["changes"])
        assert types == ["breaker_change", "gate_crossing", "regime_change"]
        assert all(c["detail"]["backfilled"] for c in h["changes"])
        assert h["last_states"]["regime"]["regime"] == "Risk-on / Choppy"
        # regime EOD fallback timestamp is UTC-stamped
        rev = next(c for c in h["changes"] if c["type"] == "regime_change")
        assert rev["timestamp"] == "2026-07-01T20:00:00+00:00"
        # public mirror written
        assert os.path.exists(os.path.join(bf.PUBLIC_DIR, "history.json"))

        # second run: everything dedupes, nothing grows
        assert bf.main() == 0
        h2 = json.load(open(os.path.join(bf.DATA_DIR, "history.json")))
        assert len(h2["changes"]) == len(h["changes"])

        # a LIVE-recorded regime event carries framework.json's generated_at
        # — a different clock than regime_history's. Same transition, same
        # date, different timestamp must still dedupe (date-level identity).
        n = len(h2["changes"])
        h2["changes"] = [c for c in h2["changes"]
                         if c["type"] != "regime_change"]
        h2["changes"].append({
            "timestamp": "2026-07-01T20:00:13.123456+00:00",  # +13s clock skew
            "type": "regime_change", "severity": "high",
            "group": None, "ticker": None,
            "description": "Swing regime: Caution → Risk-on / Choppy (2/1/0)",
            "detail": {"from_regime": "Caution",
                       "to_regime": "Risk-on / Choppy",
                       "risk_on_count": 2, "caution_count": 1,
                       "risk_off_count": 0}})
        json.dump(h2, open(os.path.join(bf.DATA_DIR, "history.json"), "w"))
        assert bf.main() == 0
        h2b = json.load(open(os.path.join(bf.DATA_DIR, "history.json")))
        assert len(h2b["changes"]) == n            # no duplicate re-added
        assert sum(1 for c in h2b["changes"]
                   if c["type"] == "regime_change") == 1

        # public/ wins a snapshot filename collision (write-once mirror;
        # the data/ copy of snapshot_20260216_204905 was test-contaminated)
        s2_bad = json.loads(json.dumps(s2))
        s2_bad["groups"][0]["stocks"][0]["score"] = 49   # would kill the event
        json.dump(s2_bad, open(os.path.join(
            bf.SNAPSHOTS_DIR, "snapshot_20260701_160000.json"), "w"))
        shutil.copy(os.path.join(bf.SNAPSHOTS_DIR, "snapshot_20260701_140000.json"),
                    os.path.join(bf.PUBLIC_SNAPSHOTS_DIR, "snapshot_20260701_140000.json"))
        json.dump(s2, open(os.path.join(
            bf.PUBLIC_SNAPSHOTS_DIR, "snapshot_20260701_160000.json"), "w"))
        ev, n_snaps, _ = bf.backfill_snapshot_events()
        assert n_snaps == 2
        assert any(e["type"] == "gate_crossing" for e in ev)  # public copy used
        # live-recorded regime state must not be clobbered by a re-run
        h2["last_states"]["regime"] = {"regime": "Risk-off", "as_of": "x"}
        json.dump(h2, open(os.path.join(bf.DATA_DIR, "history.json"), "w"))
        assert bf.main() == 0
        h3 = json.load(open(os.path.join(bf.DATA_DIR, "history.json")))
        assert h3["last_states"]["regime"]["regime"] == "Risk-off"
    finally:
        (bf.SNAPSHOTS_DIR, bf.PUBLIC_SNAPSHOTS_DIR, bf.REGIME_HISTORY,
         bf.DATA_DIR, bf.PUBLIC_DIR) = old
        shutil.rmtree(tmp, ignore_errors=True)
    print("  backfill: normalize, apply, idempotence, seed protection: OK")


if __name__ == "__main__":
    print("\n=== History event coverage tests (PER-508 #18) ===")
    test_breaker_change()
    test_trade_signal_change_pole_subset()
    test_gate_crossing()
    test_group_rename_does_not_swallow_ticker_transitions()
    test_regime_change()
    test_detect_changes_integration()
    test_backfill_normalize_and_idempotence()
    print("\nAll history event tests passed.\n")
