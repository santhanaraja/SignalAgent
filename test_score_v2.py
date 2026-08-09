#!/usr/bin/env python3
"""
D-020a pins — YTD anchor fix + penalty cap + scorer versioning.

Per the pin doctrine (docs/testing.md). The load-bearing pins are the
FROZEN-V1 anchors: after production moved to score_stock_v2, the v1
scorer must still reproduce the three committed evidence anchors —
Layer A's input_hash, Build 6A's results hash, and the recorded
candidate-grade parity — because every committed study reads against
them.

Run: python3 test_score_v2.py            (~3-5 min: rebuilds the Layer A
                                          frame through the real loop)
     python3 test_score_v2.py --fast     (skips the frame rebuild pin)
"""

import datetime
import hashlib
import json
import os
import subprocess
import sys

sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
sys.path.insert(0, os.path.join(os.path.dirname(os.path.abspath(__file__)),
                                "scripts"))
REPO = os.path.dirname(os.path.abspath(__file__))

import numpy as np
import pandas as pd

from signal_engine import (compute_ytd_return_v1, compute_ytd_return_v2,
                           score_ytd_points_v1, score_ytd_points_v2,
                           score_stock_v1, score_stock_v2, scorer_era)

YEAR = datetime.datetime.now().year


def _df(rows):
    """rows = [(iso_date, close)] -> OHLCV frame (Close is what YTD reads)."""
    idx = pd.to_datetime([r[0] for r in rows])
    c = [r[1] for r in rows]
    return pd.DataFrame({"Open": c, "High": c, "Low": c, "Close": c,
                         "Volume": [1e6] * len(rows)}, index=idx)


def test_real_ytd_anchor():
    """Change 1: v2 anchors on the LAST CLOSE OF THE PRIOR CALENDAR
    YEAR — including the January edge where YTD is two days old — and
    v1's first-current-year-close anchor is demonstrated beside it."""
    df = _df([(f"{YEAR - 1}-12-29", 90.0), (f"{YEAR - 1}-12-31", 100.0),
              (f"{YEAR}-01-02", 110.0), (f"{YEAR}-01-03", 121.0)])
    v2, basis = compute_ytd_return_v2(df, with_basis=True)
    assert v2 == 21.0 and basis == "prior_year_close", (v2, basis)
    # v1 (frozen) anchors on Jan 2 — the day-1 move is invisible to it
    assert compute_ytd_return_v1(df) == 10.0
    # January two-days-old edge: one current-year bar suffices for v2
    dfj = _df([(f"{YEAR - 1}-12-31", 200.0), (f"{YEAR}-01-02", 210.0)])
    v2j, bj = compute_ytd_return_v2(dfj, with_basis=True)
    assert v2j == 5.0 and bj == "prior_year_close"
    assert compute_ytd_return_v1(dfj) == 0.0    # v1 needs 2 current bars
    # fallback is VISIBLE: a frame with no prior-year bar says so
    dff = _df([(f"{YEAR}-02-10", 100.0), (f"{YEAR}-03-10", 150.0)])
    vf, bf = compute_ytd_return_v2(dff, with_basis=True)
    assert vf == 50.0 and bf == "first_close_of_year"
    print("  (1) v2 anchors on the prior-year last close (Jan edge: two "
          "days old works); fallback recorded, never silent: OK")


def test_cap_flat_never_reverses():
    """Change 2: the plateau sits at the curve's PEAK (12 — the v1
    ladder was already declining before 100%), holds flat, and NEVER
    reverses; 100.01 scores exactly what 100.00 scores; no value
    anywhere exceeds v1's maximum."""
    assert score_ytd_points_v2(100.00) == score_ytd_points_v2(100.01) == 12
    grid = np.arange(-60.0, 500.0, 0.25)
    pts = [score_ytd_points_v2(y) for y in grid]
    assert all(b >= a for a, b in zip(pts, pts[1:])), \
        "v2 ladder decreased somewhere — the cap reversed"
    assert max(pts) == 12 == max(
        score_ytd_points_v1(y) for y in grid), "new reward introduced"
    # below the peak the ladders are IDENTICAL; the divergence is only
    # where v1 tapered (>50) or penalized (>100/>150)
    for y in (-60, -10.01, -10, -9.9, 0, 0.01, 5, 5.01, 20):
        assert score_ytd_points_v2(y) == score_ytd_points_v1(y), y
    # the plateau's ENTRY is pinned exactly (review finding: a >30-entry
    # mutant passed): 20 is still the +6 bucket, 20.01 is the peak
    assert score_ytd_points_v2(20) == 6 and score_ytd_points_v2(20.01) == 12
    assert score_ytd_points_v2(30) == 12 and score_ytd_points_v2(50.01) == 12
    assert score_ytd_points_v1(75) == 8 and score_ytd_points_v2(75) == 12
    assert score_ytd_points_v1(120) == -2 and score_ytd_points_v2(120) == 12
    assert score_ytd_points_v1(200) == -7 and score_ytd_points_v2(200) == 12
    print("  (2) cap: flat 12 from the peak, monotone non-decreasing, "
          "100.01 == 100.00, v1 untouched below the peak: OK")


def test_both_paths_same_ytd():
    """Change 1: the universe path (1y frame, internal v2) and the
    grading path (6mo indicator frame + ytd override from the wider
    frame) must report the SAME YTD for the same ticker-day."""
    days = pd.bdate_range(f"{YEAR - 1}-08-01", f"{YEAR}-08-07")
    closes = np.linspace(80, 160, len(days))
    full = pd.DataFrame({"Open": closes, "High": closes * 1.01,
                         "Low": closes * 0.99, "Close": closes,
                         "Volume": np.full(len(days), 2e6)}, index=days)
    six = full[full.index >= full.index[-1] - pd.DateOffset(months=6)]
    # universe path: the REAL entry point (review finding: it was a
    # stand-in) — universe_builder._ticker_metrics on the 1y frame
    import universe_builder as ub
    m = ub._ticker_metrics("TSTV2", full)
    uni = compute_ytd_return_v2(full)
    assert m is not None and m["ytd"] == uni, (m and m["ytd"], uni)
    # grading path: 6mo indicators + override from the wider frame
    ytd_v, ytd_b = compute_ytd_return_v2(full, with_basis=True)
    _s, _sig, det = score_stock_v2(six, ytd_return=ytd_v, ytd_basis=ytd_b)
    assert det["ytd_return"] == uni and det["ytd_basis"] == \
        "prior_year_close", (det["ytd_return"], uni)
    # and the frozen v1 on the 6mo frame demonstrates the OLD defect:
    # a frame-anchored number that is not the calendar YTD
    _s1, _g1, det1 = score_stock_v1(six)
    assert det1["ytd_return"] != uni, "fixture failed to span the defect"
    assert det1["scorer_version"] == "score_stock_v1"
    assert det["scorer_version"] == "score_stock_v2"
    print(f"  (3) both paths agree ({uni}%) while frozen v1 on the 6mo "
          f"frame shows the old rolling number ({det1['ytd_return']}%): OK")


def test_v2_touches_only_ytd():
    """The stop condition, pinned: on the SAME frame, v1 and v2 differ
    ONLY in the ytd component — RSI, MACD, MA, volume are bit-identical
    (the indicator windows are untouched by D-020a)."""
    rng = np.random.default_rng(7)
    days = pd.bdate_range(f"{YEAR - 1}-06-01", f"{YEAR}-08-07")
    closes = 50 * np.cumprod(1 + 0.002 + 0.02 * rng.standard_normal(len(days)))
    df = pd.DataFrame({"Open": closes, "High": closes * 1.01,
                       "Low": closes * 0.99, "Close": closes,
                       "Volume": rng.integers(1e5, 5e7, len(days)).astype(float)},
                      index=days)
    _s1, _g1, d1 = score_stock_v1(df)
    _s2, _g2, d2 = score_stock_v2(df)
    c1, c2 = d1["score_components"], d2["score_components"]
    for k in ("rsi", "macd", "ma", "vol"):
        assert c1[k] == c2[k], (k, c1[k], c2[k])
    assert d1["rsi"] == d2["rsi"] and d1["macd"] == d2["macd"] \
        and d1["ma20"] == d2["ma20"] and d1["volume_ratio"] == d2["volume_ratio"]
    print("  (4) same frame, v1 vs v2: rsi/macd/ma/vol components "
          "bit-identical — only the ytd component moves: OK")


def test_v1_reproduces_all_three_anchors(fast=False):
    """Change 3's non-negotiable: after the production cutover, the
    FROZEN v1 stack still reproduces (a) Layer A's input_hash by
    rebuilding the frame through the real build loop, (b) Build 6A's
    committed results hash, (c) the recorded candidate-grade parity
    replay (the 516-grade honesty bridge, driven via its own pin)."""
    # (b) Build 6A results hash — recomputable from the committed file
    p = os.path.join(REPO, "docs", "backtest-target-results.json")
    r = json.load(open(p))
    stored = r.pop("results_hash")
    got = hashlib.sha256(json.dumps(json.loads(json.dumps(r)),
                                    sort_keys=True).encode()).hexdigest()[:16]
    assert got == stored == "847d6306f4355fb9", (got, stored)

    # (c) recorded candidate-grade parity — drive the committed-artifact
    # replay pin itself (it re-grades from RECORDED inputs, exactly the
    # era-honesty this build must not break)
    import test_backtest_doctrine as tbd
    tbd.test_production_grade_parity_overlap()

    if fast:
        print("  (5) v1 anchors: 6A hash + grade parity OK; frame "
              "rebuild SKIPPED (--fast) — run the full pin before "
              "staging")
        return
    # (a) Layer A input_hash — rebuild the frame through the REAL loop
    import backtest_doctrine as bd
    with open(bd.REGIME_PATH) as f:
        regime = json.load(f)
    regime_by_day = regime["states"]
    end = bd.END or regime["provenance"]["series_range"][1]
    with open(bd.EARNINGS_PATH) as f:
        earnings = json.load(f)
    files = sorted(os.listdir(bd.PRICES))
    rows = []
    for fn in files:
        t = fn[:-4]
        r = bd.process_ticker(t, os.path.join(bd.PRICES, fn),
                              earnings.get(t) or {"status": "missing"},
                              regime_by_day, bd.START, end)
        if r:
            rows.extend(r)
    df = pd.DataFrame(rows, columns=bd.COLS)
    got = bd._df_hash(df)
    assert got == "d7df85e8ad6ba244", (
        f"Layer A frame rebuild hashes {got}, committed input_hash is "
        "d7df85e8ad6ba244 — the frozen v1 evidence base broke; STOP")
    print(f"  (5) v1 anchors: Layer A input_hash d7df85e8ad6ba244 "
          f"(frame rebuilt, {len(df):,} rows), 6A results hash "
          f"847d6306f4355fb9, candidate-grade parity replay: OK")


def test_era_awareness():
    """v1-era artifacts carry no scorer_version and are never
    retro-relabelled; the era helper says so, and the last COMMITTED
    signals.json (baked pre-D-020a) parses and reads as v1-era."""
    # testing.md Law 3: anchor the "before" to a COMMIT, never HEAD —
    # 32373ec is the last v1-era bake of data/signals.json and stays
    # v1-era forever no matter what lands after the cutover
    old = subprocess.run(["git", "show", "32373ec:data/signals.json"],
                         capture_output=True, text=True, cwd=REPO)
    assert old.returncode == 0
    art = json.loads(old.stdout.replace(": NaN", ": null"))
    assert "scorer_version" not in art, \
        "a committed pre-D-020a artifact already carries the stamp?"
    assert scorer_era(art) == "score_stock_v1-era"
    assert scorer_era({"scorer_version": "score_stock_v2"}) \
        == "score_stock_v2"
    # renderers' contract: the stamp is additive — the committed
    # artifact's own fields still read exactly as before
    assert art.get("sp500_ytd") is not None and art.get("groups")
    print("  (6) era-awareness: committed artifact = v1-era by omission, "
          "new stamp additive, no retro-relabelling: OK")




def test_run_engine_ytd_wiring():
    """THE PRODUCTION WIRING PIN (review finding: pin 3 hand-passes the
    override, so a regression dropping it — reverting every grading-path
    YTD to the rolling 6mo number, the original defect — was invisible).
    Drives the REAL run_engine with the network stubbed at the fetch
    boundary (the test_breaker_coverage writer-pin pattern): the 6mo
    frame starts in February (rolling anchor would say +80%), the 1y
    frame reaches the prior-year close (real YTD +36.88%). The artifact
    row must carry the REAL number and its basis — and with the 1y
    fetch DEAD, the visible fallback."""
    import shutil
    import tempfile
    import signal_engine as eng

    def _frame(start, closes_start, closes_end):
        days = pd.bdate_range(start, f"{YEAR}-08-07")
        c = np.linspace(closes_start, closes_end, len(days))
        return pd.DataFrame(
            {"Open": c, "High": c * 1.01, "Low": c * 0.99, "Close": c,
             "Volume": np.full(len(days), 3e6)}, index=days)

    # 6mo indicator frame: Feb start at 50 -> 90 (frame-anchored +80%)
    # 1y frame: prior-year Aug start, prior-year LAST close such that the
    # real YTD is +36.88% of it; both frames share the last close 90.0
    six = _frame(f"{YEAR}-02-09", 50.0, 90.0)
    prior_days = pd.bdate_range(f"{YEAR - 1}-08-08", f"{YEAR - 1}-12-31")
    pc = np.linspace(60.0, 65.766, len(prior_days))   # last prior close
    cur = six.copy()
    full = pd.concat([pd.DataFrame(
        {"Open": pc, "High": pc * 1.01, "Low": pc * 0.99, "Close": pc,
         "Volume": np.full(len(prior_days), 3e6)}, index=prior_days), cur])
    expect_real = round((90.0 / 65.766 - 1) * 100, 2)     # 36.85-ish

    real_fetch = eng.fetch_data
    real_idx = eng.get_index_data
    real_groups = eng.get_industry_groups
    real_fund = eng.fetch_fundamentals_yfinance
    saved_dirs = {}
    outs = {}

    def make_fetch(dead_1y):
        def fake_fetch(ticker, period="6mo", **kw):
            if ticker == "AAA":
                if period == "1y":
                    return None if dead_1y else full.copy()
                return six.copy()
            return full.copy()          # macro/index tickers: healthy
        return fake_fetch

    def fake_groups():
        return {"TestGrp": {"tickers": ["AAA"], "sector": "IT",
                            "cycle_stage": "mid",
                            "macro_sensitivities": ["group_momentum"],
                            "commodity_proxy": None,
                            "sector_type": "semiconductor"}}

    for label, dead in (("healthy", False), ("dead_1y", True)):
        tmp = tempfile.mkdtemp(prefix=f"wiring_{label}_")
        stubs = []
        try:
            eng.fetch_data = make_fetch(dead)
            eng.get_index_data = lambda: {"^GSPC": {"ytd": 8.0}}
            eng.get_industry_groups = fake_groups
            eng.fetch_fundamentals_yfinance = lambda t: {}
            for _mod in ("earnings_calendar",):
                _m = sys.modules.get(_mod) or __import__(_mod)
                for fn in ("refresh_for_tickers", "next_earnings_map",
                           "get_earnings_map"):
                    if hasattr(_m, fn):
                        stubs.append((_m, fn, getattr(_m, fn)))
                        setattr(_m, fn, lambda *a, **k: {})
            for attr in ("DATA_DIR", "PUBLIC_DIR"):
                if hasattr(eng, attr):
                    saved_dirs[attr] = getattr(eng, attr)
                    setattr(eng, attr, tmp)
            outs[label] = eng.run_engine()
        finally:
            eng.fetch_data = real_fetch
            eng.get_index_data = real_idx
            eng.get_industry_groups = real_groups
            eng.fetch_fundamentals_yfinance = real_fund
            for _obj, _name, _orig in stubs:
                setattr(_obj, _name, _orig)
            for attr, val in saved_dirs.items():
                setattr(eng, attr, val)
            shutil.rmtree(tmp, ignore_errors=True)

    row = outs["healthy"]["groups"][0]["stocks"][0]
    assert abs(row["ytd_return"] - expect_real) < 0.51, (
        row["ytd_return"], expect_real,
        "run_engine did not wire the 1y-frame REAL YTD into the row — "
        "the rolling number would read ~80")
    assert row["ytd_basis"] == "prior_year_close", row.get("ytd_basis")
    assert outs["healthy"].get("scorer_version") == "score_stock_v2"
    row_d = outs["dead_1y"]["groups"][0]["stocks"][0]
    assert row_d["ytd_basis"] == "first_close_of_year", (
        row_d.get("ytd_basis"), "a dead 1y fetch must degrade VISIBLY")
    assert abs(row_d["ytd_return"] - 80.0) < 0.51, row_d["ytd_return"]
    print(f"  (7) run_engine wiring: artifact row carries the real YTD "
          f"({row['ytd_return']}%, basis prior_year_close); dead 1y "
          f"fetch degrades visibly (basis first_close_of_year, "
          f"{row_d['ytd_return']}%): OK")


def test_cutover_event_suppression():
    """D-020a blindspot fix, pinned: a scorer-era boundary comparison
    emits ONE scorer_cutover event and suppresses per-name score-derived
    events (signal/score/gate/trade-pole) plus group-signal/rank churn —
    while breaker events still flow. The next v2-vs-v2 comparison
    narrates normally."""
    import history_manager as hm

    def art(stamp, score, signal, trade, breaker):
        a = {"timestamp": f"{YEAR}-08-09T12:00:00",
             "groups": [{"name": "G", "group_signal": "hot",
                         "avg_ytd": 10.0, "avg_score": 60.0, "rank": 1,
                         "stock_count": 1, "breaker_status": breaker,
                         "breaker_checks_expected": ["a"],
                         "breaker_checks_run": ["a"],
                         "breaker_degraded_reasons": [],
                         "stocks": [{"ticker": "AAA", "signal": signal,
                                     "score": score, "ytd_return": 12.0,
                                     "rsi": 55, "trade_signal": trade}]}]}
        if stamp:
            a["scorer_version"] = stamp
        return a

    prev = art(None, 80, "strong-buy", "BUY NOW", "clear")
    curr = art("score_stock_v2", 40, "sell", "AVOID", "watch")
    ev = hm.detect_changes(prev, curr)
    kinds = [e["type"] for e in ev]
    assert kinds.count("scorer_cutover") == 1, kinds
    for banned in ("signal_change", "score_change", "gate_crossing",
                   "trade_signal_change", "group_signal_change",
                   "group_rank_change"):
        assert banned not in kinds, (banned, kinds)
    assert "breaker_change" in kinds, kinds
    # next day, v2 vs v2: the same delta narrates normally again
    curr2 = art("score_stock_v2", 80, "strong-buy", "BUY NOW", "watch")
    ev2 = hm.detect_changes(curr, curr2)
    kinds2 = [e["type"] for e in ev2]
    assert "signal_change" in kinds2 and "scorer_cutover" not in kinds2
    print("  (8) cutover suppression: one scorer_cutover event, "
          "score-derived events muted across the era boundary, breakers "
          "still flow, v2-vs-v2 narrates normally: OK")


if __name__ == "__main__":
    fast = "--fast" in sys.argv
    test_real_ytd_anchor()
    test_cap_flat_never_reverses()
    test_both_paths_same_ytd()
    test_v2_touches_only_ytd()
    test_v1_reproduces_all_three_anchors(fast=fast)
    test_era_awareness()
    test_run_engine_ytd_wiring()
    test_cutover_event_suppression()
    print("All D-020a scorer pins passed.")
