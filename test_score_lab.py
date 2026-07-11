#!/usr/bin/env python3
"""
Tests for PER-508 item 19 — Score Lab (interactive scoring calculator).

The four ticket acceptance criteria, verbatim:
1. The drifted-prototype inputs (YTD 209, RSI 52.8, bullish-cross-rising,
   full MA structure, vol 2.33) score 76, NOT 81 — the -15 branch fires.
2. A pre-seeded live ticker reproduces its exact dashboard score — pinned
   against every stock in the real data/signals.json via the same seeding
   rules the page uses.
3. Band boundaries: 74 -> 75 flips buy -> strong-buy; 49 -> 50 crosses the
   qualifier gate.
4. Drift-proof pin: 200 random synthetic price series through score_stock
   vs simulate_score with extracted inputs — identical score + components.

Plus the endpoint contract (validation 400s, response shape).

Run: python3 test_score_lab.py   (needs the pipeline venv: numpy/pandas/flask)
"""

import json
import os
import sys

import numpy as np
import pandas as pd

sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))

from signal_engine import (MACD_STATES, compute_macd, compute_moving_averages,
                           compute_rsi, compute_volume_trend,
                           compute_ytd_return, score_stock, simulate_score)

REPO = os.path.dirname(os.path.abspath(__file__))


def test_acceptance_1_prototype_drift_case():
    score, band, comps = simulate_score(
        rsi=52.8, macd_state="bullish_rising", above_ma20=True,
        above_ma50=True, ma20_gt_ma50=True, ytd_pct=209, vol_ratio=2.33)
    assert score == 76, f"got {score} — the >150%% overextension branch must fire"
    assert comps == {"rsi": 3, "macd": 13, "ma": 14, "ytd": -7, "vol": 3}, comps
    assert band == "strong-buy"
    print("  acceptance 1: drifted-prototype inputs -> 76 (not 81), ytd -7: OK")


def _seed_inputs(stock):
    """The page's seeding rules, mirrored: prefer the engine-emitted
    score_inputs (exact by construction); fall back to reconstruction from
    the rounded display fields for payloads that predate score_inputs."""
    if stock.get("score_inputs"):
        return stock["score_inputs"]
    macd_from_pts = {13: "bullish_rising", 8: "bullish",
                     -8: "bearish", -13: "bearish_falling"}
    c = stock["score_components"]
    return dict(
        rsi=stock["rsi"], macd_state=macd_from_pts[c["macd"]],
        above_ma20=stock["price"] > stock["ma20"],
        above_ma50=stock["price"] > stock["ma50"],
        ma20_gt_ma50=stock["ma20"] > stock["ma50"],
        ytd_pct=stock["ytd_return"], vol_ratio=stock["volume_ratio"])


def test_acceptance_2_live_payload_parity():
    """Every stock in the production signals.json, seeded exactly as the
    page seeds, must reproduce its dashboard score and components."""
    signals = json.load(open(os.path.join(REPO, "data", "signals.json")))
    checked = 0
    for group in signals["groups"]:
        for stock in group["stocks"]:
            if not stock.get("score_components"):
                continue
            score, _, comps = simulate_score(**_seed_inputs(stock))
            assert score == stock["score"], \
                f"{stock['ticker']}: simulate {score} != dashboard {stock['score']}"
            assert comps == stock["score_components"], \
                f"{stock['ticker']}: components diverge: {comps}"
            checked += 1
    assert checked >= 40, f"only {checked} stocks checked — payload missing components?"
    print(f"  acceptance 2: {checked} live dashboard scores reproduced exactly: OK")


def test_acceptance_3_band_and_gate_boundaries():
    # 74 -> 75: buy flips to strong-buy
    s74, b74, _ = simulate_score(55, "bullish", True, True, True, 3, 0.5)
    s75, b75, _ = simulate_score(35, "bullish_rising", True, False, True, 3, 1.0)
    assert (s74, b74) == (74, "buy"), (s74, b74)
    assert (s75, b75) == (75, "strong-buy"), (s75, b75)
    # 49 -> 50: crossing the universe qualifier gate
    s49, b49, _ = simulate_score(65, "bullish", False, True, False, -5, 1.0)
    s50, b50, _ = simulate_score(65, "bullish", True, False, True, -5, 0.5)
    assert s49 == 49 and s50 == 50, (s49, s50)
    assert b49 == "hold" and b50 == "hold"
    assert s49 - 50 == -1 and s50 - 50 == 0     # gate_distance semantics
    print("  acceptance 3: 74->75 buy->strong-buy; 49->50 crosses gate: OK")


def _synthetic_df(seed):
    rng = np.random.RandomState(seed)
    days = rng.randint(25, 130)
    dates = pd.bdate_range(end="2026-07-10", periods=days)
    base = 5 + rng.rand() * 300
    drift = rng.choice([-0.004, -0.001, 0.0005, 0.002, 0.006])
    vol = rng.choice([0.005, 0.02, 0.05])
    closes = base * np.cumprod(1 + drift + vol * rng.randn(days))
    closes = np.maximum(closes, 0.5)
    return pd.DataFrame({
        "Open": closes * (1 + 0.002 * rng.randn(days)),
        "High": closes * 1.01, "Low": closes * 0.99, "Close": closes,
        "Volume": rng.randint(1e5, 5e7, days).astype(float),
    }, index=dates)


def _extract_inputs(df):
    """Derive simulate inputs from a df exactly as score_stock computes
    them (same functions, same NaN defaults) — the fuzz harness."""
    close = df["Close"]
    rsi_s = compute_rsi(close)
    rsi = rsi_s.iloc[-1] if not pd.isna(rsi_s.iloc[-1]) else 50
    macd_line, signal_line, hist = compute_macd(close)
    m = macd_line.iloc[-1] if not pd.isna(macd_line.iloc[-1]) else 0
    s = signal_line.iloc[-1] if not pd.isna(signal_line.iloc[-1]) else 0
    h = hist.iloc[-1] if not pd.isna(hist.iloc[-1]) else 0
    hp = hist.iloc[-2] if len(hist) > 1 and not pd.isna(hist.iloc[-2]) else 0
    bullish = m > s
    confirms = (h > hp) if bullish else (h < hp)
    state = next(k for k, v in MACD_STATES.items() if v == (bullish, confirms))
    ma20, ma50, _ = compute_moving_averages(close)
    price = close.iloc[-1]
    ma20_v = ma20.iloc[-1] if not pd.isna(ma20.iloc[-1]) else price
    ma50_v = ma50.iloc[-1] if not pd.isna(ma50.iloc[-1]) else price
    return dict(rsi=rsi, macd_state=state,
                above_ma20=price > ma20_v, above_ma50=price > ma50_v,
                ma20_gt_ma50=ma20_v > ma50_v,
                ytd_pct=compute_ytd_return(df),
                vol_ratio=compute_volume_trend(df))


def test_acceptance_4_fuzz_200_combos():
    for seed in range(200):
        df = _synthetic_df(seed)
        score, band, details = score_stock(df)
        sim_score, sim_band, sim_comps = simulate_score(**_extract_inputs(df))
        assert sim_score == score, \
            f"seed {seed}: simulate {sim_score} != score_stock {score}"
        assert sim_comps == details["score_components"], \
            f"seed {seed}: components diverge {sim_comps} vs {details['score_components']}"
        assert sim_band == band, f"seed {seed}: band diverges"
        # the engine-emitted seed block must reproduce the score verbatim —
        # this is what the page actually seeds from
        si_score, _, si_comps = simulate_score(**details["score_inputs"])
        assert si_score == score and si_comps == details["score_components"], \
            f"seed {seed}: score_inputs block diverges"
    print("  acceptance 4: 200-df fuzz — simulate == score_stock exactly: OK")


def test_endpoint_contract():
    import ticker_api
    c = ticker_api.app.test_client()
    ok = c.post("/api/score/simulate", json={
        "rsi": 52.8, "macd_state": "bullish_rising", "above_ma20": True,
        "above_ma50": True, "ma20_gt_ma50": True, "ytd_pct": 209,
        "vol_ratio": 2.33})
    d = ok.get_json()
    assert ok.status_code == 200 and d["status"] == "success"
    assert d["score"] == 76 and d["band"] == "strong-buy"
    assert d["gate_distance"] == 26
    assert d["score_components"] == {"rsi": 3, "macd": 13, "ma": 14,
                                     "ytd": -7, "vol": 3}
    for bad, why in (
        ({}, "missing everything"),
        ({"rsi": 300, "macd_state": "bullish", "above_ma20": True,
          "above_ma50": True, "ma20_gt_ma50": True, "ytd_pct": 0,
          "vol_ratio": 1}, "rsi out of range"),
        ({"rsi": 50, "macd_state": "sideways", "above_ma20": True,
          "above_ma50": True, "ma20_gt_ma50": True, "ytd_pct": 0,
          "vol_ratio": 1}, "bad macd_state"),
        ({"rsi": 50, "macd_state": "bullish", "above_ma20": "yes",
          "above_ma50": True, "ma20_gt_ma50": True, "ytd_pct": 0,
          "vol_ratio": 1}, "non-bool toggle"),
        ({"rsi": True, "macd_state": "bullish", "above_ma20": True,
          "above_ma50": True, "ma20_gt_ma50": True, "ytd_pct": 0,
          "vol_ratio": 1}, "bool masquerading as number"),
        ({"rsi": 50, "macd_state": ["bullish"], "above_ma20": True,
          "above_ma50": True, "ma20_gt_ma50": True, "ytd_pct": 0,
          "vol_ratio": 1}, "unhashable macd_state (list) must 400 not 500"),
        ({"rsi": 50, "macd_state": {}, "above_ma20": True,
          "above_ma50": True, "ma20_gt_ma50": True, "ytd_pct": 0,
          "vol_ratio": 1}, "unhashable macd_state (dict) must 400 not 500"),
    ):
        r = c.post("/api/score/simulate", json=bad)
        assert r.status_code == 400, f"{why}: expected 400, got {r.status_code}"
    r = c.post("/api/score/simulate", data="not json",
               content_type="text/plain")
    assert r.status_code == 400
    # pathological nesting: RecursionError inside get_json must 400 not 500
    r = c.post("/api/score/simulate", data="[" * 3000 + "]" * 3000,
               content_type="application/json")
    assert r.status_code == 400, f"deep nesting: expected 400, got {r.status_code}"
    print("  endpoint contract: response shape + nine 400 validations: OK")


if __name__ == "__main__":
    print("\n=== Score Lab tests (PER-508 #19) ===")
    test_acceptance_1_prototype_drift_case()
    test_acceptance_2_live_payload_parity()
    test_acceptance_3_band_and_gate_boundaries()
    test_acceptance_4_fuzz_200_combos()
    test_endpoint_contract()
    print("\nAll Score Lab tests passed.\n")
