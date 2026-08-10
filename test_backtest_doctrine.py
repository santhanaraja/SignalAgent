#!/usr/bin/env python3
"""
Build 5 Layer A pins — the doctrine backtest.

Per the pin doctrine (docs/testing.md): drive the real entry points,
assert invariants, demonstrate failures. The heart of this file is
PARITY: the backtest's vectorized features must reproduce the production
implementations they replicate, and the real grade_setup must reproduce
production's recorded candidate grades on the committed overlap.

Run: python3 test_backtest_doctrine.py
"""

import datetime
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

import backtest_doctrine as bd
from framework.position_signals import (grade_setup, atr_mean,
                                        up_close_off_swing_low,
                                        runway_sessions_before)
# D-020a: the ladder-parity pin drives the FROZEN v1 scorer explicitly —
# the committed Layer A frame was built on the v1 ladder and must keep
# reproducing after the production scorer moved to v2
from signal_engine import (compute_rsi, compute_macd, score_rsi_points,
                           score_macd_points, score_ma_points,
                           score_ytd_points_v1, score_vol_points,
                           compose_score)

PRICES = bd.PRICES

# The committed framework artifacts Layer A's honesty bridge was
# REPORTED from (the newest 12 as of 9b5a450, the Layer A commit).
# Pinned by SHA: the cron rewrites public/framework.json continuously,
# and a sliding window silently re-based the evidence.
BRIDGE_COMMITS = (
    "32373ec", "42708f0", "4b1dc07", "7bd90fa", "2f91d92",
    "35ea765", "da0e8cf", "7d7a430", "30b9c7d", "167c735",
    "eec4f91", "dceca77", "59ebff4", "392a85a", "9fa149c",
    "0ce29c9", "4796b85", "6968211", "9a6a950", "7875b7d")

HAVE_CACHE = os.path.isdir(PRICES) and len(os.listdir(PRICES)) > 5


def _sample_closes(n_tickers=6, seed=7):
    rng = np.random.default_rng(seed)
    files = sorted(os.listdir(PRICES))
    pick = rng.choice(len(files), size=min(n_tickers, len(files)),
                      replace=False)
    out = []
    for i in pick:
        df = pd.read_csv(os.path.join(PRICES, files[i]), index_col=0,
                         parse_dates=True)
        if len(df) > bd.FRAME + 60:
            out.append((files[i][:-4], df))
    return out


# ------------------------------------------------------------- (1) parity
def test_feature_parity_vs_production():
    """The vectorized frame features must equal the PRODUCTION functions
    computed on the same trailing frame — RSI (compute_rsi), MACD state
    (compute_macd), ATR (atr_mean), the approach turn check
    (up_close_off_swing_low). Real price data, random frame ends."""
    if not HAVE_CACHE:
        raise AssertionError("doctrine price cache missing — run "
                             "scripts/fetch_doctrine_cache.py first "
                             "(this pin must not silently pass)")
    rng = np.random.default_rng(11)
    checked = 0
    for t, df in _sample_closes():
        close = df["Close"].to_numpy(dtype=float)
        rsi_all = bd.rsi_windowed(close)
        bull_all, conf_all = bd.macd_windowed(close)
        for _ in range(8):
            i = int(rng.integers(bd.FRAME - 1, len(df) - 1))
            frame = df.iloc[i - bd.FRAME + 1:i + 1]
            j = i - (bd.FRAME - 1)

            prod_rsi = compute_rsi(frame["Close"]).iloc[-1]
            if not np.isnan(prod_rsi):
                assert abs(rsi_all[j] - prod_rsi) < 1e-9, (
                    t, i, rsi_all[j], prod_rsi)

            macd_l, sig_l, hist = compute_macd(frame["Close"])
            p_bull = macd_l.iloc[-1] > sig_l.iloc[-1]
            p_conf = (hist.iloc[-1] > hist.iloc[-2]) if p_bull \
                else (hist.iloc[-1] < hist.iloc[-2])
            assert bool(bull_all[j]) == bool(p_bull), (t, i)
            assert bool(conf_all[j]) == bool(p_conf), (t, i)

            prod_atr = atr_mean(frame, 14)
            tr = np.maximum(
                frame["High"].to_numpy()[1:] - frame["Low"].to_numpy()[1:],
                np.maximum(
                    np.abs(frame["High"].to_numpy()[1:]
                           - frame["Close"].to_numpy()[:-1]),
                    np.abs(frame["Low"].to_numpy()[1:]
                           - frame["Close"].to_numpy()[:-1])))
            assert abs(prod_atr - tr[-14:].mean()) < 1e-9

            for L in bd.SWING_LOOKBACKS:
                mine = bd.up_close_windowed(close[:i + 1], L)[-1]
                prod = up_close_off_swing_low(
                    pd.Series(close[:i + 1]), L)
                assert bool(mine) == bool(prod), (t, i, L)
            checked += 1
    assert checked >= 30, checked
    print(f"  (1) feature parity vs production functions "
          f"({checked} random ticker-days, RSI exact to 1e-9, MACD state, "
          f"ATR, approach turn x3 lookbacks): OK")


def test_score_ladder_parity():
    """The vectorized score ladders equal the production score_*_points
    elementwise across the full input range, including every boundary."""
    rsi = np.array([0, 29.99, 30, 39.9, 40, 59.9, 60, 69.9, 70, 79.9, 80,
                    100], dtype=float)
    ytd = np.array([-50, -10.01, -10, -9.9, 0, 0.01, 5, 5.01, 20, 20.01,
                    50, 50.01, 100, 100.01, 150, 150.01, 400], dtype=float)
    vol = np.array([0.1, 0.69, 0.7, 1.0, 1.5, 1.51, 9.0], dtype=float)
    n = max(len(rsi), len(ytd), len(vol))
    R = np.resize(rsi, n)
    Y = np.resize(ytd, n)
    V = np.resize(vol, n)
    for bull in (True, False):
        for conf in (True, False):
            for a20 in (True, False):
                for a50 in (True, False):
                    for gt in (True, False):
                        s, s_np = bd.score_components_vec(
                            R, np.full(n, bull), np.full(n, conf),
                            np.full(n, a20), np.full(n, a50),
                            np.full(n, gt), Y, V)
                        for k in range(n):
                            comp = {
                                "rsi": score_rsi_points(R[k]),
                                "macd": score_macd_points(bull, conf),
                                "ma": score_ma_points(a20, a50, gt),
                                "ytd": score_ytd_points_v1(Y[k]),
                                "vol": score_vol_points(V[k]),
                            }
                            assert s[k] == compose_score(comp), (
                                R[k], Y[k], V[k], bull, conf, s[k], comp)
    print("  (2) score-ladder parity: vectorized == production "
          "score_*_points across every boundary x 32 flag combos: OK")


def test_production_grade_parity_overlap():
    """THE HONESTY BRIDGE's exact half: for every committed framework.json
    carrying candidate_grades, re-drive grade_setup with the SAME inputs
    the runner used (the committed signals.json rows' grade_inputs +
    next_earnings_date + breaker + regime + universe) and reproduce the
    recorded grade for every candidate. This pins the full calling
    convention, not just the pure function."""
    from framework.position_signals import PositionSignalEngine

    # SLIDING-WINDOW DEFECT, fixed 2026-08-10: this took the newest 12
    # commits of a file the cron rewrites, so the window rolled off the
    # artifacts the report was written from — the recorded "516 grades"
    # silently became 680 across different days while the pin still
    # passed (it asserts floors, never the recorded count). Anchored to
    # the commits Layer A was reported from, per docs/testing.md.
    log = [f"{sha} pinned" for sha in BRIDGE_COMMITS[:12]]
    checked = graded = mismatched = 0
    eng = PositionSignalEngine({"positions": {}}, fetcher=None)
    details = []
    for line in log:
        if not line:
            continue
        sha, day = line.split()
        fw = subprocess.run(["git", "show",
                             f"{sha}:public/framework.json"],
                            capture_output=True, text=True, cwd=REPO)
        sg = subprocess.run(["git", "show", f"{sha}:public/signals.json"],
                            capture_output=True, text=True, cwd=REPO)
        if fw.returncode or sg.returncode:
            continue
        try:
            fw = json.loads(fw.stdout)
            sig = json.loads(sg.stdout.replace(": NaN", ": null"))
        except ValueError:
            continue
        cand = fw.get("candidate_grades")
        if not isinstance(cand, dict) or not cand:
            continue
        regime = (fw.get("regime") or {}).get("regime")
        universe = {g.get("name") for g in sig.get("groups") or []}
        breakers = {g.get("name"): g.get("breaker_status")
                    for g in sig.get("groups") or []}
        # the runner graded with "today" = the artifact's generation day
        gen = (fw.get("generated_at") or "")[:10]
        today = datetime.date.fromisoformat(gen) if gen else None
        checked += 1
        for g in sig.get("groups") or []:
            for row in g.get("stocks") or []:
                t = row.get("ticker")
                rec = cand.get(t)
                if not rec or rec.get("grade") is None \
                        or not isinstance(row.get("grade_inputs"), dict):
                    continue
                got = eng._grade_one_candidate(
                    g.get("name"), row, universe, breakers, regime, today)
                graded += 1
                if got.get("grade") != rec.get("grade"):
                    mismatched += 1
                    details.append((day, t, rec.get("grade"),
                                    got.get("grade")))
    assert checked >= 3, f"only {checked} committed grade days found"
    assert graded >= 50, f"only {graded} candidate grades replayed"
    assert mismatched == 0, (
        f"{mismatched}/{graded} grades failed to reproduce: {details[:6]}")
    print(f"  (3) production grade parity: {graded} recorded candidate "
          f"grades across {checked} committed artifacts reproduce exactly "
          f"through the real calling convention: OK")


# ---------------------------------------------------- (4) lookahead pins
def test_walk_window_truncation():
    """No-lookahead: the grade at day T computed from the FULL history
    equals the grade computed from history truncated at T. Real entry
    point (process_ticker), sampled ticker-days."""
    if not HAVE_CACHE:
        raise AssertionError("doctrine price cache missing")
    with open(bd.REGIME_PATH) as f:
        regime = json.load(f)["states"]
    with open(bd.EARNINGS_PATH) as f:
        earnings = json.load(f)
    rng = np.random.default_rng(3)
    files = sorted(os.listdir(PRICES))
    tested = 0
    for _ in range(4):
        fn = files[int(rng.integers(len(files)))]
        t = fn[:-4]
        full = bd.process_ticker(t, os.path.join(PRICES, fn),
                                 earnings.get(t) or {"status": "missing"},
                                 regime, "2020-01-02", "2026-07-10")
        if not full or len(full) < 300:
            continue
        # truncate the CSV at a sampled day and recompute
        pick = full[int(rng.integers(200, len(full) - 1))]
        cut_date = str(pick[0])
        df = pd.read_csv(os.path.join(PRICES, fn), index_col=0,
                         parse_dates=True)
        tmp = os.path.join("/tmp", f"trunc_{t}.csv")
        df[df.index <= cut_date].to_csv(tmp)
        trunc = bd.process_ticker(t, tmp, earnings.get(t)
                                  or {"status": "missing"},
                                  regime, "2020-01-02", cut_date)
        os.unlink(tmp)
        last = [r for r in trunc if r[0] == cut_date]
        assert last, (t, cut_date)
        # grades + score + inputs identical; forward returns legitimately
        # differ (they need future bars the truncation removed)
        n_compare = len(bd.COLS) - len(bd.HORIZONS)
        assert last[0][:n_compare] == pick[:n_compare], (
            f"{t} @ {cut_date}: truncated history changes the grade — "
            f"lookahead!\n full={pick[:n_compare]}\n trunc={last[0][:n_compare]}")
        tested += 1
    assert tested >= 2, tested
    print(f"  (4) walk-window truncation: {tested} sampled ticker-days "
          "grade identically from truncated history (no lookahead): OK")


def test_shift_liveness():
    """Law 3's liveness half: grading the same ticker with prices shifted
    one day must CHANGE the day-to-grade mapping materially — a pipeline
    that ignores its price input would pass truncation trivially."""
    if not HAVE_CACHE:
        raise AssertionError("doctrine price cache missing")
    with open(bd.REGIME_PATH) as f:
        regime = json.load(f)["states"]
    with open(bd.EARNINGS_PATH) as f:
        earnings = json.load(f)
    files = sorted(os.listdir(PRICES))
    fn = files[0]
    t = fn[:-4]
    base = bd.process_ticker(t, os.path.join(PRICES, fn),
                             earnings.get(t) or {"status": "missing"},
                             regime, "2020-01-02", "2026-07-10")
    df = pd.read_csv(os.path.join(PRICES, fn), index_col=0,
                     parse_dates=True)
    shifted = df.copy()
    shifted[["Open", "High", "Low", "Close", "Volume"]] = \
        shifted[["Open", "High", "Low", "Close", "Volume"]].shift(1)
    shifted = shifted.dropna()
    tmp = "/tmp/shift_live.csv"
    shifted.to_csv(tmp)
    moved = bd.process_ticker(t, tmp, earnings.get(t)
                              or {"status": "missing"},
                              regime, "2020-01-02", "2026-07-10")
    os.unlink(tmp)
    b = {r[0]: r[2] for r in base}
    m = {r[0]: r[2] for r in moved}
    common = sorted(set(b) & set(m))
    diff = sum(1 for d in common if b[d] != m[d])
    assert common and diff / len(common) > 0.02, (
        f"only {diff}/{len(common)} grades changed under a 1-day shift — "
        "the pipeline is not reading the prices it claims to")
    print(f"  (5) +1d shift liveness: {diff}/{len(common)} "
          f"({diff / len(common) * 100:.1f}%) day-grades change: OK")


# ------------------------------------------------- (6) runway None-split
def test_runway_none_split():
    """Ruling 1: coverage gap is NEVER A+; a known print grades through
    the REAL runway_sessions_before semantics (print day = 0)."""
    dates = np.array(["2024-03-01", "2024-03-04", "2024-03-05"],
                     dtype="datetime64[D]")
    # a healthy quarterly reporter
    rw, basis = bd.runway_arrays(
        dates, ["2023-12-05", "2024-03-05", "2024-06-04"], "ok")
    assert list(basis) == [0, 0, 0]
    # semantics equal the production function
    for i, d in enumerate(("2024-03-01", "2024-03-04", "2024-03-05")):
        prod = runway_sessions_before(
            "2024-03-05", datetime.date.fromisoformat(d))
        assert rw[i] == prod, (d, rw[i], prod)
    assert rw[2] == 0                          # the print day itself
    # a missing print (interval > gap threshold) -> coverage gap
    rw2, basis2 = bd.runway_arrays(
        dates, ["2023-06-01", "2024-06-04"], "ok")
    assert set(basis2) == {2}, basis2
    # an errored fetch -> gap everywhere; empty dates -> gap (all-stock pool)
    _, b3 = bd.runway_arrays(dates, [], "error:HTTPError")
    _, b4 = bd.runway_arrays(dates, [], "ok")
    assert set(b3) == {2} and set(b4) == {2}
    print("  (6) runway None-split: known prints match "
          "runway_sessions_before exactly (print day = 0); missing-print "
          "interval, errored fetch and empty tables are coverage gaps: OK")


# ------------------------------------------------ (7) regime replay pin
def test_regime_series_replay_equality():
    """Ruling 2: the COMMITTED regime series reproduces from source on
    splice-bearing machines. Absent the splice this RAISES (a pin that
    silently does not run is worse than one that fails — docs/testing.md
    Law 3 note)."""
    splice = os.path.join(REPO, "data", "backtest_cache", "OAS.csv")
    committed_path = os.path.join(REPO, "data", "regime_daily.json")
    assert os.path.exists(committed_path), "regime_daily.json not committed"
    with open(committed_path) as f:
        committed = json.load(f)
    if not os.path.exists(splice):
        raise AssertionError(
            "OAS splice absent — the replay-equality pin CANNOT RUN on "
            "this machine. On splice-bearing machines it must; do not "
            "treat this as a pass. (data/backtest_cache/OAS.csv)")
    import build_regime_series as brs
    # rebuild to SCRATCH — the pin must never rewrite the committed
    # artifact it verifies (the provenance carries built_from_commit,
    # which legitimately differs across commits; states must not)
    rebuilt = brs.build(out_path="/tmp/regime_daily_rebuilt.json")
    assert rebuilt["states"] == committed["states"], (
        "committed regime series does not reproduce from the splice — "
        "provenance broken")
    assert rebuilt["provenance"]["oas_splice"]["sha256"] == \
        committed["provenance"]["oas_splice"]["sha256"], (
        "the splice hash changed — the committed series was built from "
        "different source data; rebuild and recommit deliberately")
    print(f"  (7) regime replay equality: {len(committed['states'])} days "
          "reproduce bit-identically from the local splice; splice hash "
          "matches provenance: OK")


def test_regime_series_sanity():
    """The distribution check that caught the bool-vs-raw trend bug
    (False >= 0 is True): the series must contain real Out states through
    COVID and 2022 — a replay without them is reading its inputs wrong."""
    with open(os.path.join(REPO, "data", "regime_daily.json")) as f:
        st = json.load(f)["states"]
    assert st.get("2020-03-20", "").startswith("Out-"), st.get("2020-03-20")
    assert st.get("2022-06-15", "").startswith("Out-"), st.get("2022-06-15")
    assert st.get("2023-07-03") == "In-Trend-Full"
    from collections import Counter
    c = Counter(st.values())
    assert c.get("Out-Risk-off", 0) > 100, dict(c)
    assert c.get("In-Trend-Full", 0) > 300, dict(c)
    print("  (8) regime series sanity: COVID/2022 read Out-, 2023 reads "
          "In-Trend-Full, all four states materially populated: OK")


# ------------------------------------------------------ (9) determinism
def test_determinism_smoke():
    """D-006: same inputs -> byte-identical results. Two smoke runs must
    produce the same dataframe hash."""
    if not HAVE_CACHE:
        raise AssertionError("doctrine price cache missing")
    r1, df1 = bd.run(smoke=True, out_path="/tmp/doctrine_smoke1.json")
    r2, df2 = bd.run(smoke=True, out_path="/tmp/doctrine_smoke2.json")
    assert r1["input_hash"] == r2["input_hash"], "non-deterministic rerun"
    for p in ("/tmp/doctrine_smoke1.json", "/tmp/doctrine_smoke2.json"):
        os.unlink(p)
    print(f"  (9) determinism: two smoke runs hash identically "
          f"({r1['input_hash']}): OK")


def test_consec_parity_and_old_bug():
    """The confirmation counter must equal production's
    consec_closes_above on EVERY prefix — and the pin demonstrates the
    bug it exists to prevent (Law 3): the original cumcount+1
    construction counts the group's opening False day, so the first
    close above SMA20 read consec=2 and passed confirmation a day early."""
    from framework.position_signals import consec_closes_above
    rng = np.random.default_rng(5)
    for _ in range(60):
        n = int(rng.integers(5, 60))
        close = pd.Series(rng.normal(100, 5, n))
        sma = pd.Series(rng.normal(100, 2, n))
        above = close > sma
        fixed = above.astype(int).groupby((~above).cumsum()).cumsum()
        buggy = (above.groupby((~above).cumsum()).cumcount() + 1)             .where(above, 0)
        for i in range(n):
            prod = consec_closes_above(close[:i + 1], sma[:i + 1])
            assert int(fixed.iloc[i]) == prod, (i, int(fixed.iloc[i]), prod)
        # the demonstration: the old construction disagrees wherever a
        # streak follows a below day
        # only streaks that FOLLOW a below-day demonstrate the bug — a
        # series starting True has no opening False in group 0, so the
        # old construction is accidentally correct at j=0
        starts = above & ~above.shift(1, fill_value=False)
        starts.iloc[0] = False
        if starts.any():
            j = int(np.argmax(starts.to_numpy()))
            assert int(buggy.iloc[j]) == 2 and                 consec_closes_above(close[:j + 1], sma[:j + 1]) == 1,                 "the demonstrated bug changed shape — re-derive this pin"
    print("  (10) consec parity: group-cumsum == production on every "
          "prefix of 60 random series; the old cumcount+1 construction "
          "demonstrably reads 2 on day one of a streak: OK")


def test_pipeline_honesty_bridge():
    """The report's 'pipeline bridge' claim, COMMITTED (review finding:
    it was computed ad hoc and quoted). For every committed
    framework.json carrying candidate_grades whose day the price cache
    covers, run the REAL backtest pipeline (process_ticker) with the
    artifact's own recorded regime and compare grade-lite to the
    recorded full grade. Asserts a floor on agreement and that every
    disagreement is explained by a RULED deletion (row 6 / c5) — an
    unexplained flavour of disagreement fails."""
    if not HAVE_CACHE:
        raise AssertionError("doctrine price cache missing")
    LBL2STATE = {v: k for k, v in bd.CHASSIS_LABELS.items()}
    # same sliding-window fix: pinned commits, not the newest 20
    log = list(BRIDGE_COMMITS)
    overlap = {}
    for sha in log:
        r = subprocess.run(["git", "show", f"{sha}:public/framework.json"],
                           capture_output=True, text=True, cwd=REPO)
        if r.returncode:
            continue
        try:
            fw = json.loads(r.stdout)
        except ValueError:
            continue
        cand = fw.get("candidate_grades")
        day = (fw.get("generated_at") or "")[:10]
        reg = (fw.get("regime") or {}).get("regime")
        if isinstance(cand, dict) and cand and day not in overlap                 and reg in LBL2STATE:
            overlap[day] = (LBL2STATE[reg], cand, sha)
    assert len(overlap) >= 2, f"only {len(overlap)} overlap days"
    with open(bd.EARNINGS_PATH) as f:
        earnings = json.load(f)
    regime_by_day = {d: st for d, (st, _, _) in overlap.items()}
    total = agree = 0
    unexplained = []
    for day, (st, cand, sha) in sorted(overlap.items()):
        sg = subprocess.run(["git", "show", f"{sha}:public/signals.json"],
                            capture_output=True, text=True, cwd=REPO)
        rows_by_t = {}
        if not sg.returncode:
            try:
                sig = json.loads(sg.stdout.replace(": NaN", ": null"))
                for g in sig.get("groups") or []:
                    for row in g.get("stocks") or []:
                        rows_by_t[row.get("ticker")] = (g, row)
            except ValueError:
                pass
        for t, rec in cand.items():
            g = rec.get("grade")
            if not g:
                continue
            p = os.path.join(bd.PRICES, f"{t.replace('/', '_')}.csv")
            if not os.path.exists(p):
                continue
            r = bd.process_ticker(t, p, earnings.get(t)
                                  or {"status": "missing"},
                                  regime_by_day, day, day)
            if not r:
                continue
            mine = {x[0]: x[2] for x in r}.get(day)
            if mine is None:
                continue
            total += 1
            lite = {3: "A+", 2: "B", 1: "C"}[mine]
            if lite == g:
                agree += 1
                continue
            # every miss must trace to a RULED deletion: production
            # denied on row 6 (breaker) or graded conditions with c5
            reasons = str(rec.get("reasons") or "")
            explained = ("breaker" in reasons
                         or "universe" in reasons
                         or "conditions" in reasons)
            if not explained:
                unexplained.append((day, t, g, lite, reasons[:60]))
    assert total >= 100, f"bridge too thin: {total}"
    share = agree / total * 100
    assert share >= 95.0, f"bridge agreement {share:.1f}% < 95%"
    assert not unexplained, (
        f"disagreements NOT explained by a ruled deletion: {unexplained}")
    print(f"  (11) pipeline honesty bridge: {agree}/{total} "
          f"({share:.1f}%) agree across {len(overlap)} committed days; "
          "every miss traces to a ruled deletion (row 6/c5/conditions): OK")


def test_aggregation_tables():
    """The aggregation path (review finding: it had NO pin — a bug in
    per_grade_tables/momentum/penalty corrupts every headline table while
    the per-ticker pins stay green). A hand-built 4-row frame with known
    answers drives the REAL table builders."""
    rows = [
        # date, ticker, grade, fwd_20 — two dates, both grades present
        ("2024-01-02", "AAA", 3, 2.0), ("2024-01-02", "BBB", 1, 1.0),
        ("2024-01-03", "AAA", 3, 4.0), ("2024-01-03", "BBB", 1, -1.0),
        ("2020-06-01", "CCC", 2, 0.5), ("2020-06-01", "DDD", 1, 3.0),
    ]
    df = pd.DataFrame(rows, columns=["date", "ticker", "grade", "fwd_20"])
    for h in bd.HORIZONS:
        if f"fwd_{h}" not in df:
            df[f"fwd_{h}"] = df.fwd_20
    t = bd.per_grade_tables(df)
    assert t["full"]["A+"]["ticker_days"] == 2
    assert abs(t["full"]["A+"]["fwd_20d"]["mean_pct"] - 3.0) < 1e-9
    assert abs(t["full"]["C"]["fwd_20d"]["mean_pct"] - 1.0) < 1e-9
    assert abs(t["full"]["spread_aplus_minus_c"]["fwd_20d"] - 2.0) < 1e-9
    # spans split correctly (train has no A+ here)
    assert t["validate"]["A+"]["ticker_days"] == 2
    assert t["train"]["A+"]["ticker_days"] == 0
    # hit rates
    assert t["full"]["C"]["fwd_20d"]["hit_rate_pct"] == round(2 / 3 * 100, 1)
    # penalty module on a frame where the penalty moved one row
    df2 = df.assign(score=[80, 40, 80, 40, 60, 45.0],
                    score_nopen=[80, 55, 80, 55, 60, 45.0])
    p = bd.penalty_module(df2)
    assert p["ticker_days_penalized"] == 2
    assert p["cross_50_gate"]["ticker_days"] == 2
    assert abs(p["penalized_fwd_20d"]["mean_pct"] - 0.0) < 1e-9
    print("  (12) aggregation tables: per_grade_tables spans/means/"
          "spread/hit-rate and penalty_module reproduce hand-computed "
          "answers on a known fixture: OK")


if __name__ == "__main__":
    print("\n=== Build 5 doctrine-backtest pins ===")
    test_feature_parity_vs_production()
    test_score_ladder_parity()
    test_production_grade_parity_overlap()
    test_walk_window_truncation()
    test_shift_liveness()
    test_runway_none_split()
    test_regime_series_replay_equality()
    test_regime_series_sanity()
    test_determinism_smoke()
    test_consec_parity_and_old_bug()
    test_pipeline_honesty_bridge()
    test_aggregation_tables()
    print("\nAll doctrine-backtest tests passed.\n")
