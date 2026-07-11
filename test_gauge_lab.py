#!/usr/bin/env python3
"""
Tests for PER-508 item 24a — Gauge Lab.

The lab owns zero math: every pin here goes THROUGH the endpoint, which
calls the real compute_regime (Build 4 extraction). Truth-table pins per
the ticket, flip-distance probing (asserted against values discovered by
the probe — thresholds never hardcoded in the lab; here they appear only
as expected test values, which is what a pin is), hostile-input 400s, and
the Build-4 demonstration case (OAS 5.0 on today's live gauges).

Run: python3 test_gauge_lab.py   (pipeline venv)
"""

import json
import os
import sys

sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))

import ticker_api

REPO = os.path.dirname(os.path.abspath(__file__))
C = ticker_api.app.test_client()


def _sim(vix=15.0, oas=2.5, br=1.0, spy=5.0):
    r = C.post("/api/regime/simulate", json={
        "vix_5d": vix, "hy_oas": oas, "breadth_20d": br,
        "spy_vs_200dma_pct": spy})
    assert r.status_code == 200, r.get_json()
    return r.get_json()


def test_truth_table():
    d = _sim(15.0, 2.5, 1.0, 5.0)                     # 3 risk_on, gate open
    assert d["state"] == "Risk-on / Trending"
    assert d["counts"]["risk_on"] == 3 and d["gate"]["open"]
    assert d["action_line"], "action line must come from config states"

    d = _sim(15.0, 3.5, 0.0, 5.0)                     # exactly 1 risk_on
    assert d["counts"]["risk_on"] == 1 and d["state"] == "Caution"

    d = _sim(15.0, 5.0, 1.0, 5.0)                     # any risk_off vote
    assert d["votes"]["hy_spread"] == "risk_off"
    assert d["state"] in ("Caution", "Risk-off")
    assert d["state"] == "Caution"                    # 2 risk_on present

    d = _sim(15.0, 2.5, 1.0, -0.5)                    # gate caps a 3/3
    assert d["counts"]["risk_on"] == 3
    assert d["state"] == "Caution" and d["gate"]["capped"]
    assert d["gate"]["reason"] == "below_200dma"

    r = C.post("/api/regime/simulate", json={          # 2 unavailable
        "vix_5d": None, "hy_oas": None, "breadth_20d": 1.0,
        "spy_vs_200dma_pct": 5.0})
    d = r.get_json()
    assert d["counts"]["unavailable"] == 2 and d["state"] == "Caution"
    assert d["flip_distances"]["vix_5d"] is None       # no flip for a dark input

    # Risk-off itself, both ladder rows (review finding: never pinned):
    d = _sim(25.0, 5.0, -1.0, 5.0)                     # >=2 risk_off votes
    assert d["counts"]["risk_off"] >= 2 and d["state"] == "Risk-off"
    d = _sim(20.0, 3.5, 0.0, 5.0)                      # 0 risk_on, all report
    assert d["counts"]["risk_on"] == 0 and d["counts"]["unavailable"] == 0
    assert d["state"] == "Risk-off"
    print("  truth table: Trending/Caution/caps/2-dark/both Risk-off rows: OK")


def test_flip_distances_probed():
    d = _sim(17.0, 2.5, 1.0, 5.0)
    f = d["flip_distances"]
    # vix 17 -> nearest flip just above 18 (risk_on -> caution), state
    # Trending -> Choppy. Probed, tol 1e-3.
    assert abs(f["vix_5d"]["distance"] - 1.0) < 0.01, f["vix_5d"]
    assert f["vix_5d"]["from"] == "risk_on" and f["vix_5d"]["to"] == "caution"
    assert f["vix_5d"]["state_flips"] and \
        f["vix_5d"]["state_if_flipped"] == "Risk-on / Choppy"
    assert abs(f["hy_oas"]["distance"] - 0.5) < 0.01
    assert abs(f["breadth_20d"]["distance"] + 0.5) < 0.01   # down to caution
    assert abs(f["spy_vs_200dma_pct"]["distance"] + 5.0) < 0.01
    assert f["spy_vs_200dma_pct"]["to"] == "gate_closed"
    assert f["spy_vs_200dma_pct"]["state_if_flipped"] == "Caution"
    # a mid-band value flips DOWN to the nearer boundary: vix 21 is closer
    # to 22 (caution->risk_off) than to 18
    d = _sim(21.0, 2.5, 1.0, 5.0)
    assert abs(d["flip_distances"]["vix_5d"]["distance"] - 1.0) < 0.01
    assert d["flip_distances"]["vix_5d"]["to"] == "risk_off"
    print("  flip distances: probed boundaries at 18/3.0/±0.5/gate-0: OK")


def test_validation():
    for bad, why in (
        ({}, "empty body is 4 nulls -> allowed, so use non-dict"),
    ):
        pass
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


def test_build4_demonstration():
    """OAS at 5.0 with the other three at today's live values: the HY vote
    goes risk_off and the ladder caps the state — tomorrow's teaching
    exhibit (what July would have looked like with 2015-era spreads)."""
    fw = json.load(open(os.path.join(REPO, "public", "framework.json")))
    g = fw["regime"]["gauges"]
    gate = fw["regime"]["backdrop_gate"]
    d = _sim(g["vix_5d_avg"]["value"], 5.0, g["breadth"]["value"],
             gate["value"])
    assert d["votes"]["hy_spread"] == "risk_off"
    assert d["state"] in ("Caution", "Risk-off")       # any risk_off caps
    print(f"  Build-4 demo: live gauges + OAS 5.0 -> {d['state']} "
          f"(votes {d['votes']}): OK")


if __name__ == "__main__":
    print("\n=== Gauge Lab tests (PER-508 #24a) ===")
    test_truth_table()
    test_flip_distances_probed()
    test_validation()
    test_build4_demonstration()
    print("\nAll Gauge Lab tests passed.\n")
