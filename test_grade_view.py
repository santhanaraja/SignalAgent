#!/usr/bin/env python3
"""
Grade-view pins — the D-011 grade on the search page + Score Lab.

Per the pin doctrine (docs/testing.md). Display and evaluation only:
these pins hold the new surface to the bar law (no grade without its
basis), the withholding law (unchecked breaker data grades nothing),
the off-universe honesty rule, and exact parity with candidate_grades.

Run: python3 test_grade_view.py
"""

import datetime
import json
import os
import sys

sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
REPO = os.path.dirname(os.path.abspath(__file__))

import numpy as np
import pandas as pd

from grade_view import compute_grade_view

# the framework artifact carries the regime LABEL (e.g. "Risk-on /
# Choppy") and the grading path consumes it raw — fixtures must speak
# the same vocabulary (a chassis-state string reads as blocked)
REGIME = "Risk-on / Choppy"


def _frames(last_day=None, n=280):
    """A year-spanning OHLC frame ending on a business day."""
    end = last_day or (datetime.date.today()
                       - datetime.timedelta(days=3))
    idx = pd.bdate_range(end=end, periods=n)
    c = np.linspace(80, 160, n)
    df = pd.DataFrame({"Open": c - 0.5, "High": c + 1.5, "Low": c - 1.5,
                       "Close": c, "Volume": np.full(n, 2e6)}, index=idx)
    return df.iloc[-130:], df          # (6mo-ish, 1y)


def _artifacts(breaker="clear", with_ticker=True):
    signals = {"groups": [{
        "name": "TestGrp", "breaker_status": breaker,
        "stocks": ([{"ticker": "TST",
                     "next_earnings_date": "2027-01-15"}]
                   if with_ticker else [])}]}
    framework = {"regime": {"regime": REGIME}}
    return signals, framework


def test_basis_always_present():
    """The bar law: every rendered grade carries its basis and the
    basis bar — confirmed grades say confirmed_close + the bar date."""
    df, ydf = _frames()
    sig, fw = _artifacts()
    v = compute_grade_view("TST", df=df, ytd_df=ydf, signals=sig,
                           framework=fw)
    assert v["status"] == "graded", v
    c = v["confirmed"]
    assert c["basis"] == "confirmed_close" and c["basis_bar"], c
    assert c["basis_bar"] == str(df.index[-1])[:10]
    assert c["grade"] in ("A+", "B", "C")
    assert len(c["rows"]) == 7
    for r in c["rows"]:
        assert "actual" in r and "threshold" in r, r
    ext = [r for r in c["rows"] if r["row"] == "2_extension"][0]
    # VALUE-pinned translation (review finding: presence-only let a
    # sign-flipped line survive): recompute the line prices on paper
    # from the payload's own inputs
    gi = c["inputs"]
    assert ext["line_price"] == round(
        gi["sma20"] + gi["extension_guard_max"] * gi["atr14"], 2), ext
    appr = [r for r in c["rows"] if r["row"] == "3_approach"][0]
    assert appr["line_price"] == round(gi["sma5"], 2)
    cond = [r for r in c["rows"] if r["row"] == "1_conditions"][0]
    assert cond["line_price"] == round(gi["sma20"], 2)
    assert ext["actual"] == round(gi["extension_atr"], 2) \
        and ext["threshold"] == gi["extension_guard_max"]
    r4 = [r for r in c["rows"] if r["row"] == "4_rsi"][0]
    assert r4["threshold"] == [gi["rsi_min"], gi["rsi_max"]]
    r5 = [r for r in c["rows"] if r["row"] == "5_score"][0]
    assert r5["actual"] == gi["quality_score"] \
        and r5["threshold"] == gi["score_min"]
    # met flags cross-checked against the actual-vs-threshold arithmetic
    assert r4["met"] == (gi["rsi_min"] <= gi["rsi14"] <= gi["rsi_max"])
    assert r5["met"] == (gi["quality_score"] >= gi["score_min"])
    assert ext["met"] == (gi["extension_atr"]
                          <= gi["extension_guard_max"])
    print(f"  (1) basis law: grade {c['grade']} carries basis "
          f"confirmed_close @ {c['basis_bar']}; 7 rows with "
          f"actual/threshold; extension line ${ext['line_price']}: OK")


def test_provisional_only_when_market_open():
    """PROVISIONAL exists ONLY while the market is open (a forming bar
    per the D-018 splitter) — never when closed, always labelled."""
    today = datetime.date.today()
    # frame ending TODAY (forming candidate)
    idx = pd.DatetimeIndex(
        list(pd.bdate_range(end=pd.Timestamp(today)
                            - pd.Timedelta(days=1), periods=279))
        + [pd.Timestamp(today)])
    c = np.linspace(80, 160, 280)
    full = pd.DataFrame({"Open": c - 0.5, "High": c + 1.5,
                         "Low": c - 1.5, "Close": c,
                         "Volume": np.full(280, 2e6)}, index=idx)
    df = full.iloc[-130:]
    sig, fw = _artifacts()
    intraday = datetime.datetime.combine(today, datetime.time(13, 0))
    settled = datetime.datetime.combine(today, datetime.time(17, 0))
    v_open = compute_grade_view("TST", df=df, ytd_df=full, signals=sig,
                                framework=fw, now_et=intraday)
    assert v_open["market_open"] and v_open["provisional"] is not None
    assert v_open["provisional"]["basis"] == "forming_bar_provisional"
    assert v_open["provisional"]["basis_bar"]
    # confirmed basis is the PRIOR bar, not the forming one
    assert v_open["confirmed"]["basis_bar"] == str(idx[-2])[:10]
    v_closed = compute_grade_view("TST", df=df, ytd_df=full, signals=sig,
                                  framework=fw, now_et=settled)
    assert not v_closed["market_open"] and v_closed["provisional"] is None
    print("  (2) provisional: present + labelled while open (confirmed "
          "basis = prior bar); absent after the settle: OK")


def test_breaker_withholding_both_surfaces():
    """Unchecked breaker data grades NOTHING — degraded or missing
    breaker status returns a payload with NO grade keys at all, so
    neither the search page nor the Lab CAN render one (both surfaces
    read this payload)."""
    df, ydf = _frames()
    for st in ("degraded", None):
        sig, fw = _artifacts(breaker=st)
        v = compute_grade_view("TST", df=df, ytd_df=ydf, signals=sig,
                               framework=fw)
        assert v["status"] == "breaker_unavailable", (st, v["status"])
        assert "confirmed" not in v and "provisional" not in v
        assert not any("rows" in str(k) for k in v), "row data leaked"
        assert "fabricated clear" in v["message"]
    # and a CHECKED verdict (warning) still grades — row 6 fails, the
    # withhold is for UNCHECKED data only (never weakened, never widened)
    sig, fw = _artifacts(breaker="warning")
    v = compute_grade_view("TST", df=df, ytd_df=ydf, signals=sig,
                           framework=fw)
    assert v["status"] == "graded"
    r6 = [r for r in v["confirmed"]["rows"] if r["row"] == "6_breaker"][0]
    assert r6["met"] is False and r6["actual"] == "warning"
    print("  (3) withholding: degraded/missing breaker -> no grade keys "
          "in the payload (render-impossible on both surfaces); a "
          "checked 'warning' still grades with row 6 failing: OK")


def test_off_universe_single_state():
    """Off-universe: one honest statement, no rows, no partial grade —
    the c5/row-6 contradiction is never rendered."""
    df, ydf = _frames()
    sig, fw = _artifacts(with_ticker=False)
    v = compute_grade_view("NOTIN", df=df, ytd_df=ydf, signals=sig,
                           framework=fw)
    assert v["status"] == "off_universe"
    assert "confirmed" not in v and "provisional" not in v
    assert not any("rows" in str(k) for k in v), "row data leaked"
    assert "group-dependent rows" in v["message"] \
        and "cannot be evaluated" in v["message"]
    print("  (4) off-universe: single cannot-be-evaluated statement, "
          "never a partial row set: OK")


def test_parity_with_candidate_grades():
    """The spec's parity law: same inputs, same answer. Every recorded
    candidate grade in the committed artifacts reproduces through the
    refactored grading path (the wrapper) AND the full path agrees with
    the wrapper on every grade — the display path can never diverge
    from the artifact."""
    from framework.position_signals import PositionSignalEngine
    import subprocess
    # committed artifacts via git show HEAD: (testing.md convention —
    # a sweep regenerates working-tree artifacts; review finding)
    sig = json.loads(subprocess.run(
        ["git", "show", "HEAD:data/signals.json"], capture_output=True,
        text=True, cwd=REPO).stdout.replace(": NaN", ": null"))
    fw = json.loads(subprocess.run(
        ["git", "show", "HEAD:public/framework.json"],
        capture_output=True, text=True, cwd=REPO).stdout)
    cand = fw.get("candidate_grades") or {}
    assert cand, "framework artifact carries no candidate_grades"
    eng = PositionSignalEngine({"positions": {}}, fetcher=None)
    regime = (fw.get("regime") or {}).get("regime")
    universe = {g.get("name") for g in sig.get("groups") or []}
    breakers = {g.get("name"): g.get("breaker_status")
                for g in sig.get("groups") or []}
    gen = (fw.get("generated_at") or "")[:10]
    today = datetime.date.fromisoformat(gen)
    checked = mismatched = 0
    for g in sig.get("groups") or []:
        for row in g.get("stocks") or []:
            t = row.get("ticker")
            rec = cand.get(t)
            if not rec or rec.get("grade") is None \
                    or not isinstance(row.get("grade_inputs"), dict):
                continue
            got = eng._grade_one_candidate(g["name"], row, universe,
                                           breakers, regime, today)
            full = eng._grade_candidate_full(g["name"], row, universe,
                                             breakers, regime, today)
            checked += 1
            if got.get("grade") != rec.get("grade") \
                    or full.get("grade") != rec.get("grade"):
                mismatched += 1
            assert set(got.keys()) == {"grade", "reasons", "failing",
                                       "group"}, \
                "the artifact wrapper grew keys — framework shape changed"
            assert "rows" in full and len(full["rows"]) == 7
    assert checked >= 20, f"only {checked} candidates replayed"
    assert mismatched == 0, f"{mismatched}/{checked} diverged"
    print(f"  (5) parity: {checked} recorded candidate grades reproduce "
          f"through both the artifact wrapper and the display path; "
          f"wrapper shape unchanged: OK")


def test_differs_both_polarities():
    """Side-by-side ONLY when they differ: the payload's differs flag
    is pinned in BOTH directions with engineered forming bars."""
    today = datetime.date.today()
    n = 280
    idx = pd.DatetimeIndex(
        list(pd.bdate_range(end=pd.Timestamp(today)
                            - pd.Timedelta(days=1), periods=n - 1))
        + [pd.Timestamp(today)])
    c = np.linspace(80, 160, n)
    intraday = datetime.datetime.combine(today, datetime.time(13, 0))
    sig, fw = _artifacts()

    def frame(last_close):
        cc = c.copy()
        cc[-1] = last_close
        return pd.DataFrame({"Open": cc - 0.5, "High": cc + 1.5,
                             "Low": cc - 1.5, "Close": cc,
                             "Volume": np.full(n, 2e6)}, index=idx)

    agree = frame(c[-1])                       # forming bar in-trend
    v1 = compute_grade_view("TST", df=agree.iloc[-130:], ytd_df=agree,
                            signals=sig, framework=fw, now_et=intraday)
    assert v1["provisional"] is not None and v1["differs"] is False, v1
    crash = frame(60.0)                        # forming close below SMA20
    v2 = compute_grade_view("TST", df=crash.iloc[-130:], ytd_df=crash,
                            signals=sig, framework=fw, now_et=intraday)
    assert v2["provisional"] is not None
    assert v2["provisional"]["grade"] != v2["confirmed"]["grade"]
    assert v2["differs"] is True, v2["differs"]
    print("  (6) differs: False on an in-trend forming bar, True when "
          "the forming close breaks the grade — both polarities: OK")


def test_route_serves_withheld_payload():
    """Law 1 for the serving layer: the REAL route is driven through the
    test client and a withheld payload passes through with no grade keys
    — the render-impossibility both surfaces rely on survives the
    sanitize/serialize hop."""
    import ticker_api
    import grade_view as gv
    real = gv.compute_grade_view
    df, ydf = _frames()
    sig, fw = _artifacts(breaker="degraded")

    def fake(symbol, **kw):
        return real(symbol, df=df, ytd_df=ydf, signals=sig, framework=fw)
    gv.compute_grade_view = fake
    try:
        cl = ticker_api.app.test_client()
        r = cl.get("/api/grade/TST.json")
        d = r.get_json()
    finally:
        gv.compute_grade_view = real
    assert r.status_code == 200
    assert d["status"] == "breaker_unavailable"
    assert "confirmed" not in d and "provisional" not in d
    print("  (7) route: withheld payload survives the real serving hop "
          "with no grade keys: OK")


if __name__ == "__main__":
    test_basis_always_present()
    test_provisional_only_when_market_open()
    test_breaker_withholding_both_surfaces()
    test_off_universe_single_state()
    test_parity_with_candidate_grades()
    test_differs_both_polarities()
    test_route_serves_withheld_payload()
    print("All grade-view pins passed.")
