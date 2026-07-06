#!/usr/bin/env python3
"""
Tests for the serve-layer shape sentinel (stale-serve fix): a regime
endpoint must never serve a shape the current code cannot produce.

Pins, through the real Flask serve path (test_client):
  - a pre-1A-shaped artifact (5 voters, no backdrop_gate/macro_inputs)
    -> 503 with Retry-After AND a refresh kick
  - a current-shape artifact -> 200, no refresh kick
  - a valid-but-old artifact -> 200 + stale_hours flag, no refresh kick

Run: python3 test_serve_guard.py
"""

import datetime
import json
import os
import shutil
import sys
import tempfile
import time

sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))

import ticker_api

GUARDED = ["/api/framework/gauges.json", "/api/framework/latest",
           "/api/framework/latest.json", "/api/framework/signals.json"]


def _now_iso(hours_ago=0):
    return (datetime.datetime.now(datetime.timezone.utc)
            - datetime.timedelta(hours=hours_ago)).isoformat()


def _pre_1a_payload():
    """The exact stale class served 2026-07-06 ~9:15 ET: 5 voters incl.
    spy_vs_200dma + yield_curve, no backdrop_gate, no macro_inputs."""
    g = {"value": 1.0, "signal": "risk_on", "detail": "d"}
    return {
        "generated_at": _now_iso(),
        "framework_version": "1.0",
        "regime": {
            "regime": "Risk-on / Trending",
            "risk_on_count": 5, "caution_count": 0, "risk_off_count": 0,
            "gauges": {k: dict(g) for k in
                       ("spy_vs_200dma", "vix_5d_avg", "hy_spread",
                        "breadth", "yield_curve")},
            "action": "Full deployment.",
        },
        "position_signals": {"tickers": {}, "transitions": []},
    }


def _current_payload(hours_ago=0):
    g = {"value": 1.0, "signal": "risk_on", "detail": "d"}
    return {
        "generated_at": _now_iso(hours_ago),
        "schema": "regime-1a-3voter",
        "framework_version": "1.0",
        "regime": {
            "regime": "Risk-on / Trending",
            "risk_on_count": 3, "caution_count": 0, "risk_off_count": 0,
            "gauges": {k: dict(g) for k in
                       ("vix_5d_avg", "hy_spread", "breadth")},
            "backdrop_gate": {"gauge": "spy_vs_200dma", "open": True,
                              "capped": False, "reason": None, "value": 8.0},
            "macro_inputs": {"yield_curve": {"value": 0.8,
                                             "signal": "risk_on"}},
            "action": "Full deployment.",
        },
        "position_signals": {"tickers": {"IWM": {"state": "HELD"}},
                             "transitions": []},
    }


class _Env:
    def __init__(self):
        self.tmp = tempfile.mkdtemp(prefix="serve_guard_")
        self.old_public = ticker_api.PUBLIC_DIR
        self.old_refresh = ticker_api._run_framework_refresh
        ticker_api.PUBLIC_DIR = self.tmp
        self.kicks = []
        ticker_api._run_framework_refresh = lambda: self.kicks.append(1)
        ticker_api._framework_status["running"] = False
        self.client = ticker_api.app.test_client()

    def write(self, payload):
        with open(os.path.join(self.tmp, "framework.json"), "w") as f:
            json.dump(payload, f)

    def wait_for_kick(self, timeout=2.0):
        deadline = time.time() + timeout
        while time.time() < deadline:
            if self.kicks:
                return True
            time.sleep(0.05)
        return False

    def close(self):
        ticker_api.PUBLIC_DIR = self.old_public
        ticker_api._run_framework_refresh = self.old_refresh
        shutil.rmtree(self.tmp, ignore_errors=True)


def test_pre_1a_shape_503_and_refresh_kick():
    env = _Env()
    try:
        env.write(_pre_1a_payload())
        for ep in GUARDED:
            env.kicks.clear()
            r = env.client.get(ep)
            assert r.status_code == 503, f"{ep}: expected 503, got {r.status_code}"
            assert r.headers.get("Retry-After") == "180", ep
            body = r.get_json()
            assert body["status"] == "warming_up", ep
            assert body["retry_after"] == 180, ep
            assert "framework_status" in body, ep
            assert env.wait_for_kick(), f"{ep}: refresh was not kicked"
        # unparseable file: same treatment, never a 500
        with open(os.path.join(env.tmp, "framework.json"), "w") as f:
            f.write("{corrupt")
        r = env.client.get(GUARDED[0])
        assert r.status_code == 503
    finally:
        env.close()
    print("  pre-1A shape -> 503 + Retry-After + refresh kick (all 4 endpoints): OK")


def test_current_shape_200_no_kick():
    env = _Env()
    try:
        env.write(_current_payload())
        for ep in GUARDED:
            r = env.client.get(ep)
            assert r.status_code == 200, f"{ep}: {r.status_code}"
            body = r.get_json()
            assert "stale_hours" not in json.dumps(body), ep
        g = env.client.get("/api/framework/gauges.json").get_json()
        assert sorted(g["gauges"].keys()) == ["breadth", "hy_spread", "vix_5d_avg"]
        assert g["backdrop_gate"]["open"] is True
        s = env.client.get("/api/framework/signals.json").get_json()
        assert s["tickers"]["IWM"]["state"] == "HELD"
        assert env.kicks == [], "fresh valid payload must not kick a refresh"
    finally:
        env.close()
    print("  current shape -> 200, correct projections, no refresh kick: OK")


def test_valid_but_old_200_with_stale_flag():
    env = _Env()
    try:
        env.write(_current_payload(hours_ago=60))   # Friday close on a Sunday
        for ep in GUARDED:
            r = env.client.get(ep)
            assert r.status_code == 200, f"{ep}: valid-but-old must SERVE, got {r.status_code}"
        g = env.client.get("/api/framework/gauges.json").get_json()
        assert 59 <= g["stale_hours"] <= 61, g.get("stale_hours")
        lj = env.client.get("/api/framework/latest.json").get_json()
        assert 59 <= lj["stale_hours"] <= 61
        s = env.client.get("/api/framework/signals.json").get_json()
        assert 59 <= s["stale_hours"] <= 61
        assert env.kicks == [], "age alone must never kick a refresh"
    finally:
        env.close()
    print("  valid-but-old -> 200 + stale_hours flag, no refresh kick: OK")


def test_missing_file_still_404():
    env = _Env()
    try:
        r = env.client.get("/api/framework/gauges.json")
        assert r.status_code == 404
        assert env.kicks == []
    finally:
        env.close()
    print("  missing framework.json -> 404 (unchanged), no kick: OK")


if __name__ == "__main__":
    print("\n=== Serve-layer shape sentinel tests ===")
    test_pre_1a_shape_503_and_refresh_kick()
    test_current_shape_200_no_kick()
    test_valid_but_old_200_with_stale_flag()
    test_missing_file_still_404()
    print("\nAll serve-guard tests passed.\n")
