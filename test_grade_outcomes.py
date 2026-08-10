#!/usr/bin/env python3
"""
Grade-outcome tracker pins.

Per the pin doctrine (docs/testing.md). The simulator is driven through
its real entry points (detect_spells / simulate_ticker / build) over
hand frames where every fill is computable on paper. The tracker
OBSERVES ONLY — these pins also hold it to the honesty rules: no spell
silently dropped, nothing censored counted as closed, medians
closed-only.

Run: python3 test_grade_outcomes.py
"""

import json
import os
import sys
import tempfile

sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))

import numpy as np
import pandas as pd

import grade_outcomes as go


def _frame(closes, opens=None, highs=None, start="2026-07-01"):
    idx = pd.bdate_range(start, periods=len(closes))
    c = np.array(closes, dtype=float)
    o = np.array(opens, dtype=float) if opens is not None else c.copy()
    h = np.array(highs, dtype=float) if highs is not None else \
        np.maximum(c, o) * 1.001
    return pd.DataFrame({"Open": o, "High": h,
                         "Low": np.minimum(c, o) * 0.999, "Close": c,
                         "Volume": np.full(len(c), 1e6)}, index=idx)


def _dates(df):
    return [str(d)[:10] for d in df.index]


def test_spell_detection():
    """First day in a grade starts a spell; continuation does not;
    absence then reappearance (even same grade) starts one."""
    days = [
        {"date": "2026-08-01", "scorer_version": "v1",
         "grades": {"AAA": "B", "BBB": "C"}},
        {"date": "2026-08-02", "scorer_version": "v1",
         "grades": {"AAA": "B", "BBB": "A+"}},        # BBB enters A+
        {"date": "2026-08-03", "scorer_version": "v1",
         "grades": {"AAA": "B"}},                     # BBB absent
        {"date": "2026-08-04", "scorer_version": "v2",
         "grades": {"AAA": "B", "BBB": "A+"}},        # BBB re-enters A+
    ]
    sp = go.detect_spells(days)
    assert ("2026-08-01", "AAA", "B", "v1") in sp
    assert ("2026-08-01", "BBB", "C", "v1") in sp
    assert ("2026-08-02", "BBB", "A+", "v1") in sp
    assert ("2026-08-04", "BBB", "A+", "v2") in sp    # era travels
    assert len([s for s in sp if s[1] == "AAA"]) == 1  # continuation: one
    print("  (1) spell detection: entry/continuation/absence-return, era "
          "carried: OK")


def test_trade_mechanics_paper():
    """Entry at next open, stop = SMA20 of the grade day, equality exit
    at the FIRST close not above SMA20, fill next open — every number
    recomputed on paper in this test."""
    n = 30
    closes = [100.0 + i for i in range(n)]            # rising: stays above
    closes[27] = 80.0                                 # crash: close <= sma
    opens = [c - 0.5 for c in closes]
    highs = [c + 2.0 for c in closes]
    df = _frame(closes, opens, highs)
    d = _dates(df)
    grade_date = d[24]
    trades, skips = go.simulate_ticker(
        [(grade_date, "TST", "A+", "score_stock_v2")], df)
    assert not skips and len(trades) == 1
    t = trades[0]
    sma_paper = float(np.mean(closes[5:25]))          # 20 closes ending @24
    assert abs(t["initial_stop"] - sma_paper) < 1e-9
    assert t["entry_date"] == d[25] and t["entry"] == opens[25]
    assert abs(t["initial_r"] - (opens[25] - sma_paper)) < 1e-9
    # closes 25,26 stay above their SMA; 27 (=80) is not above -> fires,
    # exit at open of 28
    assert t["status"] == "STOPPED"
    assert t["exit_date"] == d[28] and t["exit"] == opens[28]
    assert t["days_held"] == 3                        # closes 25,26,27
    exp_pnl = round((opens[28] / opens[25] - 1) * 100, 2)
    assert t["pnl_pct"] == exp_pnl
    exp_mfe = round((max(highs[25:28]) / opens[25] - 1) * 100, 2)
    assert t["mfe_pct"] == exp_mfe and t["ever_profitable"] is (exp_mfe > 0)
    assert t["scorer_era"] == "score_stock_v2"
    # THE D-018 BOUNDARY (review finding: a strict-< mutant survived):
    # a close EXACTLY EQUAL to its SMA20 must fire. 20 flat bars at 100
    # pin sma20 == close == 100 on every bar past warmup.
    flat = _frame([100.0] * 25)
    dflat = _dates(flat)
    tr_eq, _ = go.simulate_ticker([(dflat[20], "EQ", "B", "v1")], flat)
    assert tr_eq[0]["status"] == "STOPPED" and tr_eq[0]["days_held"] == 1, \
        (tr_eq[0]["status"], "close == SMA20 must fire (equality exits)")
    assert tr_eq[0]["exit_date"] == dflat[22]
    # and MFE can never sit below realized P&L (gap-up exit fill)
    gap_closes = [100.0 + i for i in range(30)]
    gap_closes[27] = 80.0
    gap_opens = [c - 0.5 for c in gap_closes]
    gap_opens[28] = 200.0                       # exit fill gaps ABOVE
    gap = _frame(gap_closes, gap_opens, [c + 0.1 for c in gap_closes])
    tg, _ = go.simulate_ticker([(_dates(gap)[24], "GP", "B", "v1")], gap)
    assert tg[0]["status"] == "STOPPED" and \
        tg[0]["mfe_pct"] >= tg[0]["pnl_pct"], (tg[0]["mfe_pct"],
                                               tg[0]["pnl_pct"])
    print("  (2) mechanics on paper: stop=SMA20(grade day), entry next "
          "open, EQUALITY exit fires, exit fill inside MFE, "
          "days/P&L exact: OK")


def test_no_overlap_and_reentry():
    """A spell during a running trade is a RECORDED skip, never silence;
    a spell on/after the exit-fill date re-enters."""
    n = 40
    closes = [100.0 + i for i in range(n)]
    closes[27] = 80.0                                 # first trade dies @27
    df = _frame(closes)
    d = _dates(df)
    spells = [(d[24], "TST", "A+", "v1"),
              (d[26], "TST", "B", "v1"),              # while running
              (d[30], "TST", "C", "v1")]              # after exit (d[28])
    trades, skips = go.simulate_ticker(spells, df)
    assert len(trades) == 2 and len(skips) == 1
    assert skips[0]["reason"] == "trade_running" \
        and skips[0]["grade_date"] == d[26]
    assert trades[1]["grade"] == "C" and trades[1]["entry_date"] == d[31]
    # skip labels (review finding: neither path was pinned): a spell
    # before the frame -> price_window_rolled; inside the frame but
    # under SMA warmup -> insufficient_history — both RECORDED
    df2 = _frame([100.0 + i for i in range(30)], start="2026-06-01")
    d2 = _dates(df2)
    _, sk2 = go.simulate_ticker([("2026-05-01", "TST", "B", "v1"),
                                 (d2[5], "TST", "C", "v1")], df2)
    assert [k["reason"] for k in sk2] == \
        ["price_window_rolled", "insufficient_history"], sk2
    print("  (3) overlap: mid-trade spell is a recorded skip; re-entry "
          "after the exit fill; both skip labels recorded: OK")


def test_censoring_statuses_and_aggregates():
    """PENDING (no next open yet) and EXIT_FIRED (fired on the last
    close) are never counted closed; aggregates carry the censoring
    counts and compute medians on closed only."""
    n = 25
    closes = [100.0 + i for i in range(n)]
    df = _frame(closes)
    d = _dates(df)
    # PENDING: spell on the last bar
    tr_p, _ = go.simulate_ticker([(d[-1], "PND", "A+", "v2")], df)
    assert tr_p[0]["status"] == "PENDING" and tr_p[0]["entry"] is None
    # EXIT_FIRED: fires on the final close, no next open
    closes2 = [100.0 + i for i in range(n)]
    closes2[-1] = 50.0
    df2 = _frame(closes2)
    tr_f, _ = go.simulate_ticker([(d[20], "FRD", "B", "v1")], df2)
    assert tr_f[0]["status"] == "EXIT_FIRED" and tr_f[0]["exit"] is None
    # OPEN
    tr_o, _ = go.simulate_ticker([(d[20], "OPN", "A+", "v1")], df)
    assert tr_o[0]["status"] == "OPEN"
    # closed trade for the aggregate
    closes3 = [100.0 + i for i in range(n)]
    closes3[22] = 60.0
    tr_c, _ = go.simulate_ticker([(d[20], "CLS", "A+", "v1")],
                                 _frame(closes3))
    assert tr_c[0]["status"] == "STOPPED"
    agg = go.aggregate(tr_p + tr_f + tr_o + tr_c)
    a = agg["A+"]
    assert a["n_closed"] == 1 and a["n_open"] == 2
    assert a["open_breakdown"] == {"OPEN": 1, "PENDING": 1}
    assert agg["B"]["n_closed"] == 0 and \
        agg["B"]["open_breakdown"] == {"EXIT_FIRED": 1}
    assert "days_held_median_closed" not in agg["B"], \
        "a median rendered with zero closed trades"
    assert a["days_held_median_closed"] == tr_c[0]["days_held"]
    print("  (4) censoring: PENDING/EXIT_FIRED/OPEN never counted "
          "closed; medians closed-only; breakdown beside counts: OK")


def test_build_and_staleness():
    """build() writes the artifact with a state key; refresh_if_stale is
    a no-op for the same series+day and rebuilds when either moves."""
    with tempfile.TemporaryDirectory() as tmp:
        hist = os.path.join(tmp, "gh.json")
        out = os.path.join(tmp, "go.json")
        days = [{"date": "2026-08-01", "generated_at": "t",
                 "source_commit": "x", "scorer_version": "v1",
                 "graded": 1, "grades": {"TST": "A+"}}]
        json.dump({"schema": "grade-history-1", "days": days}, open(hist, "w"))
        frames = {"TST": _frame([100.0 + i for i in range(30)],
                                start="2026-06-20")}
        r = go.build(history_path=hist, frames=frames, out_path=out,
                     today="2026-08-09")
        assert os.path.exists(out) and len(r["trades"]) == 1
        assert r["disclaimer"].startswith("SIMULATED")
        old_hist, old_out = go.GRADE_HISTORY_PATH, go.OUTCOMES_PATH
        try:
            go.GRADE_HISTORY_PATH, go.OUTCOMES_PATH = hist, out
            # same series, same day -> fresh; but a naive rebuild would
            # refetch prices — prove the skip actually skips
            def _boom(*a, **k):
                raise AssertionError("refresh_if_stale refetched on a "
                                     "fresh artifact")
            old_fetch = go.fetch_frames
            go.fetch_frames = _boom
            try:
                assert go.refresh_if_stale(today="2026-08-09") == "fresh"
            finally:
                go.fetch_frames = old_fetch
            # day rolls -> rebuild (with frames injectable only via
            # fetch; give it a working fetch stub)
            go.fetch_frames = lambda tickers, period="6mo": frames
            try:
                assert go.refresh_if_stale(today="2026-08-10") == "rebuilt"
            finally:
                go.fetch_frames = old_fetch
            # the D-012 upsert churns generated_at every bake — the key
            # must ignore it (review finding: byte-hash rebuilt+refetched
            # every 15 minutes)
            days2 = json.load(open(hist))
            days2["days"][0]["generated_at"] = "CHURNED"
            json.dump(days2, open(hist, "w"))
            go.fetch_frames = _boom
            try:
                # the artifact was last rebuilt by the day-roll test
                # above, so its stored day is 2026-08-10
                assert go.refresh_if_stale(today="2026-08-10") == "fresh", \
                    "generated_at churn must not trigger a rebuild"
            finally:
                go.fetch_frames = old_fetch
        finally:
            go.GRADE_HISTORY_PATH, go.OUTCOMES_PATH = old_hist, old_out
    print("  (5) build + staleness: artifact written with state key; "
          "fresh day is a true no-op (no refetch); generated_at churn "
          "ignored; day-roll rebuilds: OK")


def test_forming_bar_never_fires():
    """D-018 confirmed closes only (review finding, HIGH): a frame
    whose LAST bar is today's forming bar must not fire the stop nor
    seed anchors — build() strips it via confirmed_close_frame; after
    the 16:10 ET settle the same bar counts."""
    import datetime as _dt
    today = _dt.date.today()
    n = 26
    # the last bar must be dated TODAY even on a weekend (the splitter
    # compares dates, and bdate_range would land the end on Friday)
    idx = pd.DatetimeIndex(
        list(pd.bdate_range(end=pd.Timestamp(today)
                            - pd.Timedelta(days=1), periods=n - 1))
        + [pd.Timestamp(today)])
    closes = [100.0 + i for i in range(n)]
    closes[-1] = 50.0                    # the dip lives on the LAST bar
    c = np.array(closes)
    df = pd.DataFrame({"Open": c - 0.5, "High": c + 1, "Low": c - 1,
                       "Close": c, "Volume": np.full(n, 1e6)}, index=idx)
    d = [str(x)[:10] for x in idx]
    with tempfile.TemporaryDirectory() as tmp:
        hist = os.path.join(tmp, "gh.json")
        out = os.path.join(tmp, "go.json")
        json.dump({"schema": "grade-history-1", "days": [
            {"date": d[22], "generated_at": "t", "source_commit": None,
             "scorer_version": "v1", "graded": 1,
             "grades": {"TST": "B"}}]}, open(hist, "w"))
        intraday = _dt.datetime.combine(today, _dt.time(14, 0))
        r1 = go.build(history_path=hist, frames={"TST": df}, out_path=out,
                      today=str(today), now_et=intraday)
        assert r1["trades"][0]["status"] == "OPEN", (
            r1["trades"][0]["status"],
            "forming-bar dip fired a stop before the close settled")
        settled = _dt.datetime.combine(today, _dt.time(17, 0))
        r2 = go.build(history_path=hist, frames={"TST": df}, out_path=out,
                      today=str(today), now_et=settled)
        assert r2["trades"][0]["status"] == "EXIT_FIRED"
    print("  (7) forming bar: intraday dip cannot fire (stripped); the "
          "same bar fires after the 16:10 settle: OK")


def test_carry_forward_closed_trades():
    """Closed trades are facts (review finding): a STOPPED trade in
    the prior artifact whose ticker the rebuild cannot price is carried
    forward, marked, and never silently deleted."""
    with tempfile.TemporaryDirectory() as tmp:
        hist = os.path.join(tmp, "gh.json")
        out = os.path.join(tmp, "go.json")
        json.dump({"schema": "grade-history-1", "days": [
            {"date": "2026-08-01", "generated_at": "t",
             "source_commit": None, "scorer_version": "v1", "graded": 1,
             "grades": {"GONE": "A+"}}]}, open(hist, "w"))
        prior = {"schema": go.SCHEMA, "trades": [
            {"ticker": "GONE", "grade": "A+", "grade_date": "2026-08-01",
             "scorer_era": "v1", "initial_stop": 100.0,
             "status": "STOPPED", "entry_date": "2026-08-02",
             "entry": 101.0, "initial_r": 1.0, "exit_date": "2026-08-05",
             "exit": 99.0, "days_held": 3, "pnl_pct": -1.98,
             "pnl_r": -2.0, "mfe_pct": 0.5, "ever_profitable": True}]}
        json.dump(prior, open(out, "w"))
        r = go.build(history_path=hist, frames={}, out_path=out,
                     today="2026-08-09")
        keys = {(t["ticker"], t["grade_date"], t.get("carried"))
                for t in r["trades"]}
        assert ("GONE", "2026-08-01", True) in keys, r["trades"]
        assert r["provenance"]["carried_closed_trades"] == 1
        assert "GONE" in r["provenance"]["tickers_no_price_data"]
        assert r["aggregates"]["A+"]["n_closed"] == 1
    print("  (8) carry-forward: unresolvable STOPPED trade carried, "
          "marked, still aggregated: OK")


def test_committed_artifact_sane():
    """The committed artifact obeys its own honesty rules."""
    p = os.path.join(os.path.dirname(os.path.abspath(__file__)),
                     "data", "grade_outcomes.json")
    doc = json.load(open(p))
    assert doc["schema"] == go.SCHEMA and doc["disclaimer"]
    n = {"STOPPED": 0, "OPEN": 0, "EXIT_FIRED": 0, "PENDING": 0}
    for t in doc["trades"]:
        n[t["status"]] += 1
        if t["status"] == "STOPPED":
            assert t["exit"] is not None and t["pnl_pct"] is not None
        else:
            assert t["pnl_pct"] is None, t
    for g, blk in doc["aggregates"].items():
        cl = [t for t in doc["trades"]
              if t["grade"] == g and t["status"] == "STOPPED"]
        assert blk["n_closed"] == len(cl)
    assert doc["provenance"]["grade_history_sha16"]
    # FULL RECONCILIATION (review finding: a build-level silent drop
    # survived): every spell detected from the committed series must be
    # a trade, a recorded skip, or belong to a recorded no-price ticker
    hist = json.load(open(os.path.join(
        os.path.dirname(os.path.abspath(__file__)), "data",
        "grade_history.json")))
    spells = {(gd, t) for gd, t, _, _ in go.detect_spells(hist["days"])}
    covered = {(t["grade_date"], t["ticker"]) for t in doc["trades"]} | \
        {(k["grade_date"], k["ticker"]) for k in doc["skipped_spells"]}
    nopx = set(doc["provenance"]["tickers_no_price_data"])
    orphans = [sp for sp in spells
               if sp not in covered and sp[1] not in nopx]
    assert not orphans, f"spells unaccounted for: {orphans[:5]}"
    for t in doc["trades"]:
        if t["status"] == "STOPPED" and t["mfe_pct"] is not None:
            assert t["pnl_pct"] <= t["mfe_pct"] + 1e-9, t
    print(f"  (6) committed artifact: {sum(n.values())} trades "
          f"({n}), censored never carry P&L, aggregate counts match, "
          f"ALL {len(spells)} spells accounted for, pnl<=mfe: OK")


if __name__ == "__main__":
    test_spell_detection()
    test_trade_mechanics_paper()
    test_no_overlap_and_reentry()
    test_censoring_statuses_and_aggregates()
    test_build_and_staleness()
    test_committed_artifact_sane()
    test_forming_bar_never_fires()
    test_carry_forward_closed_trades()
    print("All grade-outcome pins passed.")
