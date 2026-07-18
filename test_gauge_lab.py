#!/usr/bin/env python3
"""
Tests for PER-508 item 24a — Gauge Lab, chassis era (D-008).

The lab owns zero math: every pin goes THROUGH /api/regime/simulate, which
under engine=chassis calls the real chassis_step / pctile_of_last (the same
functions production replays). Deterministic via a stubbed OAS tail and a
stubbed persisted carry. Truth-table pins over the four chassis states,
hysteresis visibility (raw vs confirmed), throttle flip distances (probed,
never hardcoded server-side), and the hostile-input 400s.

Run: python3 test_gauge_lab.py
"""

import json
import os
import sys

sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))

import ticker_api

REPO = os.path.dirname(os.path.abspath(__file__))
C = ticker_api.app.test_client()

# deterministic fixtures: flat OAS history at 3.0 (probe >= 3.0 -> pctile 100
# -> credit throttle fires; probe < 3.0 -> pctile ~1.7 -> clear), and a
# controllable "persisted" hysteresis carry
ticker_api._chassis_oas_tail = lambda: ([3.0] * 59, 60)

_CARRY = {"confirmed": "In-Trend-Full", "up": 0, "down": 0}
ticker_api._chassis_carry = lambda: dict(_CARRY)


def _carry(confirmed="In-Trend-Full", up=0, down=0):
    _CARRY.update(confirmed=confirmed, up=up, down=down)


def _sim(vix=16.0, oas=2.7, br=0.2, spy=8.0):
    r = C.post("/api/regime/simulate", json={
        "vix_5d": vix, "hy_oas": oas, "breadth_20d": br,
        "spy_vs_200dma_pct": spy})
    assert r.status_code == 200, r.get_json()
    return r.get_json()


def test_truth_table_chassis():
    _carry("In-Trend-Full")
    d = _sim()                                        # in trend, all clear
    assert d["engine"] == "chassis"
    ch = d["chassis"]
    assert ch["raw_state"] == "In-Trend-Full" and \
        ch["confirmed_state"] == "In-Trend-Full"
    assert d["state"] == "Risk-on / Trending"
    assert ch["exposure_ceiling_pct"] == 90.0
    assert d["action_line"], "action line must come from config states"

    d = _sim(vix=25.0)                                # vix throttle — instant
    assert d["chassis"]["throttles"]["vix"]["firing"]
    assert d["chassis"]["confirmed_state"] == "In-Trend-Throttled"
    assert d["state"] == "Risk-on / Choppy"
    assert d["chassis"]["exposure_ceiling_pct"] == 50.0

    d = _sim(br=-1.0)                                 # breadth throttle
    assert d["chassis"]["throttles"]["breadth"]["firing"]
    assert d["chassis"]["confirmed_state"] == "In-Trend-Throttled"

    d = _sim(oas=5.0)                                 # credit spike -> pctile 100
    assert d["chassis"]["throttles"]["hy"]["firing"]
    assert d["chassis"]["throttles"]["hy"]["pctile"] == 100.0
    assert d["chassis"]["confirmed_state"] == "In-Trend-Throttled"

    d = _sim(spy=-3.0)                                # out of trend, calm
    assert d["chassis"]["trend_in"] is False
    assert d["chassis"]["confirmed_state"] == "Out-Defensive"
    assert d["state"] == "Caution"
    assert d["chassis"]["exposure_ceiling_pct"] == 25.0

    d = _sim(spy=-3.0, vix=25.0)                      # out + vol stress
    assert d["chassis"]["confirmed_state"] == "Out-Risk-off"
    assert d["state"] == "Risk-off"
    assert d["chassis"]["exposure_ceiling_pct"] == 5.0
    print("  truth table (chassis): Full/90, vix/breadth/hy->Throttled/50, "
          "out->Defensive/25, out+stress->Risk-off/5: OK")


def test_hysteresis_visible_in_lab():
    # confirmed lags raw by design: from Out-Defensive, a qualifying close is
    # 1 of the 2 the upgrade needs — the lab must SHOW that, not hide it
    _carry("Out-Defensive", up=0)
    d = _sim()                                        # raw In-Trend-Full
    assert d["chassis"]["raw_state"] == "In-Trend-Full"
    assert d["chassis"]["confirmed_state"] == "Out-Defensive"
    assert d["state"] == "Caution"
    assert d["chassis"]["hysteresis"]["up"] == 1
    # the second consecutive qualifying close completes the upgrade
    _carry("Out-Defensive", up=1)
    d = _sim()
    assert d["chassis"]["confirmed_state"] == "In-Trend-Full"
    assert d["state"] == "Risk-on / Trending"
    # downgrades never lag (asymmetric crash brake)
    _carry("In-Trend-Full")
    d = _sim(spy=-3.0)
    assert d["chassis"]["confirmed_state"] == "Out-Defensive"
    _carry("In-Trend-Full")
    print("  hysteresis in the lab: raw!=confirmed shown, N=2 completes, "
          "downgrade instant: OK")


def test_flip_distances_probed():
    _carry("In-Trend-Full")
    d = _sim()                                        # vix 16, oas 2.7, br .2, spy 8
    f = d["flip_distances"]
    # vix 16 -> throttle fires at 22 (probed): distance +6
    assert abs(f["vix_5d"]["distance"] - 6.0) < 0.01, f["vix_5d"]
    assert f["vix_5d"]["from"] == "clear" and f["vix_5d"]["to"] == "throttle_firing"
    assert f["vix_5d"]["state_if_flipped"] == "Risk-on / Choppy"
    # spy 8 -> out of trend at 0: distance -8
    assert abs(f["spy_vs_200dma_pct"]["distance"] + 8.0) < 0.01
    assert f["spy_vs_200dma_pct"]["to"] == "out_of_trend"
    assert f["spy_vs_200dma_pct"]["state_if_flipped"] == "Caution"
    # breadth 0.2 -> throttle below -0.5: distance ~ -0.7
    assert abs(f["breadth_20d"]["distance"] + 0.7) < 0.01
    assert f["breadth_20d"]["to"] == "throttle_firing"
    # hy 2.7 vs the flat-3.0 window -> fires from 3.0 up: distance ~ +0.3
    assert f["hy_oas"] and f["hy_oas"]["to"] == "throttle_firing"
    assert abs(f["hy_oas"]["distance"] - 0.3) < 0.02, f["hy_oas"]
    # dark input -> no flip probe
    r = C.post("/api/regime/simulate", json={
        "vix_5d": None, "hy_oas": 2.7, "breadth_20d": 0.2,
        "spy_vs_200dma_pct": 8.0})
    d = r.get_json()
    assert d["flip_distances"]["vix_5d"] is None
    assert d["chassis"]["throttles"]["vix"]["available"] is False
    print("  flip distances (chassis): vix +6->22, spy -8->trend, breadth "
          "-0.7->-0.5, hy +0.3->pctile cut, dark input None: OK")


def test_validation():
    r = C.post("/api/regime/simulate", data="not json",
               content_type="text/plain")
    assert r.status_code == 400
    r = C.post("/api/regime/simulate", json={"vix_5d": "high", "hy_oas": 3,
                                             "breadth_20d": 0,
                                             "spy_vs_200dma_pct": 0})
    assert r.status_code == 400
    r = C.post("/api/regime/simulate", json={"vix_5d": True, "hy_oas": 3,
                                             "breadth_20d": 0,
                                             "spy_vs_200dma_pct": 0})
    assert r.status_code == 400                        # bool is not a number
    r = C.post("/api/regime/simulate", json={"vix_5d": 500, "hy_oas": 3,
                                             "breadth_20d": 0,
                                             "spy_vs_200dma_pct": 0})
    assert r.status_code == 400                        # out of domain
    r = C.post("/api/regime/simulate", json={"vix_5d": ["x"], "hy_oas": 3,
                                             "breadth_20d": 0,
                                             "spy_vs_200dma_pct": 0})
    assert r.status_code == 400                        # unhashable-safe
    print("  validation: non-JSON/string/bool/out-of-domain/list all 400: OK")


def test_chassis_demonstration():
    """Live gauge values + a 2015-era credit spike (OAS 5.0): the credit
    throttle fires and an in-trend regime reads Choppy, not Trending — the
    chassis-era version of the Build-4 teaching exhibit."""
    _carry("In-Trend-Full")
    fw = json.load(open(os.path.join(REPO, "public", "framework.json")))
    g = fw["regime"]["gauges"]
    gate = fw["regime"]["backdrop_gate"]
    if g["vix_5d_avg"]["value"] is None or gate["value"] is None:
        print("  chassis demo SKIPPED — live artifact has dark gauges")
        return
    d = _sim(vix=g["vix_5d_avg"]["value"], oas=5.0,
             br=g["breadth"]["value"] if g["breadth"]["value"] is not None else 0.0,
             spy=gate["value"])
    assert d["chassis"]["throttles"]["hy"]["firing"]
    if d["chassis"]["trend_in"]:
        assert d["state"] in ("Risk-on / Choppy", "Caution", "Risk-off")
    print(f"  chassis demo: live gauges + OAS 5.0 -> {d['state']} "
          f"(raw {d['chassis']['raw_state']}): OK")


def test_parliament_lab_behind_flag():
    """The reversion lever's lab contract must stay alive (review finding):
    with the engine flipped to parliament, the endpoint serves the original
    votes/counts/gate shape through the untouched _regime_probe path."""
    cfg = ticker_api._regime_cfg()
    old_engine = cfg["engine"]
    try:
        cfg["engine"] = "parliament"
        d = _sim(vix=15.0, oas=2.5, br=1.0, spy=5.0)
        assert d["engine"] == "parliament"
        assert d["state"] == "Risk-on / Trending"
        assert d["counts"]["risk_on"] == 3 and d["gate"]["open"]
        assert d["votes"]["vix_5d_avg"] == "risk_on"
        f = d["flip_distances"]["vix_5d"]
        assert abs(f["distance"] - 3.0) < 0.01, f          # 15 -> 18 boundary
        assert f["from"] == "risk_on" and f["to"] == "caution"
    finally:
        cfg["engine"] = old_engine
    print("  parliament lab behind the flag: votes/counts/gate contract + "
          "probed 18-boundary intact: OK")


if __name__ == "__main__":
    print("\n=== Gauge Lab tests (chassis era, PER-508 #24a / D-008) ===")
    test_truth_table_chassis()
    test_hysteresis_visible_in_lab()
    test_flip_distances_probed()
    test_validation()
    test_chassis_demonstration()
    test_parliament_lab_behind_flag()
    print("\nAll Gauge Lab tests passed.\n")
