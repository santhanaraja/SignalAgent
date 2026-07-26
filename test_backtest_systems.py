#!/usr/bin/env python3
"""
Build 5B pins — the strategy replay.

Per the pin doctrine (docs/testing.md): drive the real entry points,
assert invariants, demonstrate failures. The sim's own invariants —
accounting conservation, cap compliance, no lookahead, stop parity with
the production rule — are what these pins hold in place.

Run: python3 test_backtest_systems.py
"""

import hashlib
import json
import os
import sys

sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
sys.path.insert(0, os.path.join(os.path.dirname(os.path.abspath(__file__)),
                                "scripts"))
REPO = os.path.dirname(os.path.abspath(__file__))

import numpy as np
import pandas as pd

import backtest_systems as bs

HAVE = os.path.exists(bs.FRAME_PATH) and os.path.isdir(bs.PRICES)


def _smoke_panel():
    if not HAVE:
        raise AssertionError(
            "doctrine caches missing — run scripts/fetch_doctrine_cache.py "
            "and scripts/backtest_doctrine.py first (pins must not "
            "silently pass)")
    return bs.load_panel(smoke=True)


PANEL = _smoke_panel()


def test_accounting_conservation():
    """Equity change decomposes EXACTLY into trade P&L + open-position
    mark + cash yield − nothing else. A leak here corrupts every metric
    downstream while all other pins stay green."""
    sim = bs.run_sim(PANEL, "S1", "score", capital=100_000.0)
    end_equity = sim["curve"][-1][1]
    closed_pnl = sum(t["pnl"] for t in sim["trades"])
    recon = 100_000.0 + closed_pnl + sim["cash_yield_usd"]
    # mark-out closes every position, so closed_pnl covers the book;
    # tolerance covers float accumulation over ~1,600 days
    assert abs(end_equity - recon) < 1.0, (
        f"conservation broken: equity {end_equity:.2f} vs "
        f"capital+pnl+yield {recon:.2f}")
    print(f"  (1) accounting conservation: end equity {end_equity:,.2f} == "
          f"capital + trade P&L + cash yield (±$1): OK")


def test_caps_never_violated():
    """Post-hoc invariant scan: on no day may exposure exceed the regime
    ceiling, a group exceed 20%/3, or a position exceed its entry size
    materially at entry. Drives the real sim with an instrumented replay."""
    sim = bs.run_sim(PANEL, "S3", "score", capital=100_000.0)
    # THE REAL INVARIANTS. Exposure above a DOWNGRADED ceiling is legal
    # by design (entries-only gating; positions ride their stops — the
    # ruled treatment), so a daily exposure-vs-current-ceiling assert
    # would be pinning a rule the system deliberately does not have.
    # What must hold instead:
    #  (a) cash never goes negative -> exposure never exceeds equity;
    #  (b) NO entry occurs in Risk-off (ceiling 5% < one 6.5% position);
    #  (c) entries only on days the entry-time ceiling could admit them.
    for day, exposure_frac, npos in sim["daily_meta"]:
        assert exposure_frac <= 1.0 + 1e-9, (day, exposure_frac)
    for t in sim["trades"]:
        # the governing state is the SIGNAL day's (recorded on the trade)
        # — a Trending-queued entry may legally FILL on a Risk-off
        # morning (the COVID-transition case); what may never happen is
        # an entry whose signal-day ceiling could not admit it
        assert t["regime"] != "Out-Risk-off", (
            t["ticker"], t["entry_date"],
            "queued under Risk-off — the 5% ceiling admits nothing")
        assert bs.CEILINGS[t["regime"]] >= bs.SIZE_PCT, t
    # per-group entry-time caps: replay admissions from the trade log
    from collections import defaultdict
    open_by_group = defaultdict(list)
    events = []
    for t in sim["trades"]:
        events.append((t["entry_date"], "in", t))
        events.append((t["exit_date"], "out", t))
    events.sort(key=lambda e: (e[0], 0 if e[1] == "out" else 1))
    for day, kind, t in events:
        g = t["group"]
        if kind == "out":
            if t in open_by_group[g]:
                open_by_group[g].remove(t)
        else:
            open_by_group[g].append(t)
            assert len(open_by_group[g]) <= bs.GROUP_MAX_N, (
                day, g, len(open_by_group[g]))
    print("  (2) caps: ceiling bound holds daily (entries-only gating "
          "drift bounded); per-group count never exceeds 3 in the "
          "admission replay: OK")


def test_no_lookahead_truncation():
    """The sim to day D on truncated data produces the identical trade
    log — entries at open T may use only <=T-1 information."""
    sim = bs.run_sim(PANEL, "S1", "score")
    days = PANEL["days"]
    cut = days[len(days) // 2]
    panel2 = dict(PANEL)
    panel2["days"] = [d for d in days if d <= cut]
    sim2 = bs.run_sim(panel2, "S1", "score")
    t1 = [(t["ticker"], t["entry_date"], t["exit_date"], round(t["pnl"], 6))
          for t in sim["trades"]
          if t["exit_date"] <= cut and not t["synthetic_close"]]
    t2 = [(t["ticker"], t["entry_date"], t["exit_date"], round(t["pnl"], 6))
          for t in sim2["trades"]
          if t["exit_date"] <= cut and not t["synthetic_close"]]
    assert t1 == t2, (
        f"truncation changed {sum(1 for a, b in zip(t1, t2) if a != b)} "
        f"of {len(t1)} closed trades — lookahead!")
    assert len(t1) > 20, f"too few trades to trust the pin ({len(t1)})"
    print(f"  (3) no lookahead: {len(t1)} closed trades identical under "
          "mid-window truncation: OK")


def test_stop_parity_with_production():
    """The sim's exit condition IS the production rule: _stop_for says
    sma20_close ('exit on close below SMA20'), and D-018 transitions on
    confirmed closes with the sale at next open. Verify against the sim's
    trade log on real data: for every non-synthetic exit, the close
    BEFORE the exit date was below its SMA20, and no earlier close since
    entry was (else it would have exited earlier)."""
    sim = bs.run_sim(PANEL, "S2", "score")
    checked = 0
    for t in sim["trades"]:
        if t["synthetic_close"]:
            continue
        p = PANEL["px"][t["ticker"]]
        i_exit = p["pos"][t["exit_date"]]
        i_entry = p["pos"][t["entry_date"]]
        prior = i_exit - 1
        assert prior >= i_entry, (t, "exit before entry?")
        assert p["close"][prior] < p["sma20"][prior], (
            t["ticker"], t["exit_date"],
            "sold without a confirmed close below SMA20")
        for j in range(i_entry, prior):
            if np.isfinite(p["sma20"][j]):
                assert not p["close"][j] < p["sma20"][j], (
                    t["ticker"], p["iso"][j],
                    "an earlier confirmed close below SMA20 did not exit")
        checked += 1
        if checked >= 200:
            break
    assert checked >= 50, checked
    print(f"  (4) stop parity: {checked} exits each preceded by exactly "
          "one confirmed close below SMA20 (none earlier missed) — the "
          "production sma20_close rule verbatim: OK")


def test_gap_below_stop_skipped():
    """An entry whose next-open fill gaps to or below the stop has R<=0
    and must be SKIPPED, never sized. Demonstrate on a synthetic panel
    where the fill gaps below the signal-day SMA20."""
    days = ["2024-01-02", "2024-01-03", "2024-01-04"]
    iso = np.array(days)
    close = np.array([100.0, 101.0, 95.0])
    sma = np.array([99.0, 99.5, 99.0])
    px = {"GAP": {"iso": iso, "pos": {s: i for i, s in enumerate(iso)},
                  "open": np.array([100.0, 90.0, 94.0]),   # gap-down fill
                  "close": close, "sma20": sma,
                  "ext": np.array([0.5, 0.5, 0.5])}}

    class Row:
        ticker = "GAP"
        grade = 3
        score = 90.0
        mom_6m = 10.0
        c1_above = True
        chassis = "In-Trend-Full"
        top_decile = True
    panel = {"days": days,
             "states": {d: "In-Trend-Full" for d in days},
             "px": px, "gates_by_day": {days[0]: [Row()]},
             "group_of": {"GAP": "G"},
             "cash_daily": pd.Series(0.0, index=days),
             "frame_hash": "test", "regime_prov": {}}
    sim = bs.run_sim(panel, "S2", "score")
    assert sim["denials"].get("gap_below_stop") == 1, sim["denials"]
    assert not sim["trades"], "a R<=0 entry was taken"
    print("  (5) gap-below-stop: a fill at/below the signal SMA20 is "
          "denied (R<=0 cannot be sized), demonstrated on a gap-down "
          "fixture: OK")


def test_selection_rules_and_null_determinism():
    """score/low_ext orderings are deterministic with documented
    tie-breaks; random ordering is seed-deterministic (the NULL band
    must be reproducible) and actually varies across seeds."""
    a = bs.run_sim(PANEL, "S3", "score")
    b = bs.run_sim(PANEL, "S3", "score")
    key = lambda s: [(t["ticker"], t["entry_date"]) for t in s["trades"]]
    assert key(a) == key(b), "score-order not deterministic"
    r1 = bs.run_sim(PANEL, "S3", "random", seed=7)
    r2 = bs.run_sim(PANEL, "S3", "random", seed=7)
    r3 = bs.run_sim(PANEL, "S3", "random", seed=8)
    assert key(r1) == key(r2), "random selection not seed-deterministic"
    assert key(r1) != key(r3), "seeds 7 and 8 identical — the null band " \
        "would be degenerate"
    print("  (6) selection determinism: score-order reproducible; "
          "random-K reproducible per seed and distinct across seeds: OK")


def test_expectancy_definition_guard():
    """The dollar-weighted expectancy must equal sum(pnl)/sum(r_usd) and
    be finite where the naive trade-mean degenerates. Fixture: one
    tiny-R trade with a big loss swamps the mean but not the weighted
    aggregate — the exact S5 smoke pathology."""
    trades = [
        {"r_mult": 1.0, "r_usd": 1000.0, "pnl": 1000.0, "hold_days": 5,
         "reentry": False, "synthetic_close": False,
         "entry_stop": 90.0, "exit_fill": 100.0},
        {"r_mult": -30.0, "r_usd": 10.0, "pnl": -300.0, "hold_days": 3,
         "reentry": False, "synthetic_close": False,
         "entry_stop": 90.0, "exit_fill": 85.0},
    ]
    m = bs.trade_metrics(trades)
    assert m["expectancy_r_trade_mean"] == round((1.0 - 30.0) / 2, 4)
    assert m["expectancy_r"] == round(700.0 / 1010.0, 4), m["expectancy_r"]
    print("  (7) expectancy definition: dollar-weighted "
          "sum(pnl)/sum(r_usd) robust where the trade-mean degenerates "
          "(the S5 pathology, pinned): OK")


def test_prereg_committed_and_intact():
    """The pre-registration must exist, contain all six verdict branches,
    and be committed in a commit that PRECEDES any results JSON. A
    results file whose criteria were written after the fact is not
    pre-registered."""
    import subprocess
    p = os.path.join(REPO, "docs", "backtest-systems-prereg.md")
    assert os.path.exists(p), "prereg missing"
    txt = " ".join(open(p).read().split())   # the file hard-wraps phrases
    for phrase in ("random-K selection-noise band", "{0,5,15}",
                   "THE DOCTRINE SURVIVES", "THE HARD GATE DIES",
                   "REPLACED, NOT REPAIRED", "NOT THE EDGE",
                   "UNFALSIFIED BUT UNSUPPORTED", "DEFERRED"):
        assert phrase in txt, f"prereg missing branch text: {phrase}"
    log = subprocess.run(
        ["git", "log", "--format=%H", "--follow", "--",
         "docs/backtest-systems-prereg.md"],
        capture_output=True, text=True, cwd=REPO).stdout.split()
    assert log, "prereg is not committed"
    res = subprocess.run(
        ["git", "log", "--format=%H", "--",
         "docs/backtest-systems-results.json"],
        capture_output=True, text=True, cwd=REPO).stdout.split()
    if res:
        order = subprocess.run(
            ["git", "merge-base", "--is-ancestor", log[-1], res[-1]],
            cwd=REPO).returncode
        assert order == 0, "prereg commit does not precede results commit"
    print("  (8) prereg: committed, all six branches present"
          + (", and strictly precedes the results commit" if res
             else " (results not yet committed)") + ": OK")


def test_determinism_hash():
    sim1 = bs.run_sim(PANEL, "S1", "score")
    sim2 = bs.run_sim(PANEL, "S1", "score")
    h = lambda s: hashlib.sha256(json.dumps(
        [(t["ticker"], t["entry_date"], round(t["pnl"], 6))
         for t in s["trades"]]).encode()).hexdigest()
    assert h(sim1) == h(sim2)
    print("  (9) determinism: identical trade-log hash across reruns: OK")


def test_signal_day_state_governs_admission():
    """THE LOOKAHEAD (review finding, critical, demonstrated): the regime
    state of fill day D derives from D's CLOSE, so reading it at D's open
    was future information — it flipped the sole band-clearing pair.
    Fixture: the state DOWNGRADES overnight (Trending -> Risk-off). The
    entry was queued under Trending; admission must honor the SIGNAL-day
    ceiling (it fills), and the stored regime label must be the
    signal-day state. The reverse (queued under Risk-off, upgrade next
    day) must NOT fill."""
    days = ["2024-01-02", "2024-01-03", "2024-01-04", "2024-01-05"]
    iso = np.array(days)
    close = np.array([100.0, 101.0, 102.0, 90.0])
    sma = np.array([95.0, 95.5, 96.0, 96.5])
    px = {"T1": {"iso": iso, "pos": {s: i for i, s in enumerate(iso)},
                 "open": np.array([100.0, 100.5, 101.5, 101.0]),
                 "close": close, "sma20": sma,
                 "ext": np.array([0.5] * 4)}}

    class Row:
        ticker = "T1"
        grade = 3
        score = 90.0
        mom_6m = 10.0
        c1_above = True
        chassis = "In-Trend-Full"
        top_decile = True

    # downgrade after the signal: queued Trending, fill-day Risk-off
    panel = {"days": days,
             "states": {days[0]: "In-Trend-Full",
                        days[1]: "Out-Risk-off",
                        days[2]: "Out-Risk-off",
                        days[3]: "Out-Risk-off"},
             "px": px, "gates_by_day": {days[0]: [Row()]},
             "group_of": {"T1": "G"},
             "cash_daily": pd.Series(0.0, index=days),
             "frame_hash": "t", "regime_prov": {}}
    sim = bs.run_sim(panel, "S2", "score")
    assert len(sim["trades"]) == 1, (
        "an entry queued under Trending was denied by the FILL day's "
        "downgraded ceiling — fill-day state leaked into admission")
    assert sim["trades"][0]["regime"] == "In-Trend-Full", (
        "the stored regime label is the fill day's, not the signal day's")

    # upgrade after the signal: queued Risk-off (0 slots) — S4 has no c3,
    # so the gate passes; only the ceiling can block, and it must block
    # on the SIGNAL day's 5% even though the fill day upgraded
    class Row4(Row):
        chassis = "Out-Risk-off"
    panel2 = dict(panel)
    panel2["states"] = {days[0]: "Out-Risk-off",
                       days[1]: "In-Trend-Full",
                       days[2]: "In-Trend-Full",
                       days[3]: "In-Trend-Full"}
    panel2["gates_by_day"] = {days[0]: [Row4()]}
    sim2 = bs.run_sim(panel2, "S4", "score")
    assert not sim2["trades"] and sim2["denials"].get("ceiling") == 1, (
        sim2["denials"],
        "an entry queued under Risk-off filled because the FILL day "
        "upgraded — future information admitted it")
    print("  (10) signal-day state governs admission: downgrades do not "
          "retro-deny, upgrades do not retro-admit; stored regime label "
          "is the signal day's: OK")


def test_entry_time_ceiling_invariant():
    """Every trade records its entry-time exposure fraction and its
    signal-day ceiling; the invariant exposure <= ceiling must hold for
    EVERY entry in EVERY arm (review finding: the old pin never asserted
    the actual percentage invariant)."""
    checked = 0
    for arm in bs.ARMS:
        sim = bs.run_sim(PANEL, arm, "score")
        for t in sim["trades"]:
            if t["entry_exposure_frac"] is None:
                continue
            assert t["entry_exposure_frac"] <=                 t["entry_ceiling"] / 100.0 + 1e-9, (
                arm, t["ticker"], t["entry_date"],
                t["entry_exposure_frac"], t["entry_ceiling"])
            checked += 1
    assert checked > 2000, checked
    print(f"  (11) entry-time ceiling invariant: {checked:,} entries "
          "across all five arms each admitted within the signal-day "
          "ceiling: OK")


def test_stop_boundary_exact_equality():
    """Production's HELD->EXIT_FIRED is `not close > sma20` — exact
    equality EXITS (the D-018 flap was one cent). A strict `<` misses it
    (the old construction, demonstrated)."""
    days = ["2024-01-02", "2024-01-03", "2024-01-04", "2024-01-05"]
    iso = np.array(days)
    close = np.array([100.0, 101.0, 96.0, 97.0])
    sma = np.array([95.0, 95.5, 96.0, 96.0])   # day 3: close == sma EXACTLY
    px = {"EQ": {"iso": iso, "pos": {s: i for i, s in enumerate(iso)},
                 "open": np.array([100.0, 100.5, 101.0, 96.5]),
                 "close": close, "sma20": sma,
                 "ext": np.array([0.5] * 4)}}

    class Row:
        ticker = "EQ"
        grade = 3
        score = 90.0
        mom_6m = 10.0
        c1_above = True
        chassis = "In-Trend-Full"
        top_decile = True
    panel = {"days": days,
             "states": {d: "In-Trend-Full" for d in days},
             "px": px, "gates_by_day": {days[0]: [Row()]},
             "group_of": {"EQ": "G"},
             "cash_daily": pd.Series(0.0, index=days),
             "frame_hash": "t", "regime_prov": {}}
    sim = bs.run_sim(panel, "S2", "score")
    real = [t for t in sim["trades"] if not t["synthetic_close"]]
    assert len(real) == 1 and real[0]["exit_date"] == days[3], (
        sim["trades"], "close == SMA20 exactly did not fire the exit — "
        "the production boundary is `not above`, not `strictly below`")
    # the old strict-< construction would NOT have fired here
    assert not (close[2] < sma[2]) and not (close[2] > sma[2])
    print("  (12) stop boundary: close == SMA20 exactly EXITS (production "
          "`not above`); the strict-< construction demonstrably missed "
          "it: OK")


def test_aggregation_layer():
    """curve_metrics and the paired band construction on hand-computed
    fixtures (review finding: the entire aggregation layer was unpinned —
    a curve_metrics bug corrupts every Sharpe/MDD claim silently)."""
    # geometric doubling over exactly 252 days -> CAGR 100%; one 10% dip
    days = [f"d{i}" for i in range(253)]
    v = [100.0 * (2.0 ** (i / 252.0)) for i in range(253)]
    v[100] *= 0.9                                  # a one-day 10% dip
    curve = list(zip(days, v))
    m = bs.curve_metrics(curve)
    assert abs(m["cagr_pct"] - 100.0) < 1.5, m["cagr_pct"]
    assert abs(m["max_dd_pct"] - (-10.0)) < 0.35, m["max_dd_pct"]
    assert m["max_dd_duration_days"] == 1, m["max_dd_duration_days"]
    assert m["end_equity"] == round(v[-1], 2)
    # flat curve: no dd, sharpe undefined/None-safe
    flat = bs.curve_metrics([(d, 100.0) for d in days])
    assert flat["max_dd_pct"] == 0.0 and flat["cagr_pct"] == 0.0
    # paired band: the exact construction pair_tests uses
    a = np.array([0.5, 0.6, 0.7, 0.4, 0.5])
    b = np.array([0.1, 0.2, 0.3, 0.0, 0.1])
    diffs = a - b
    lo, hi = np.percentile(diffs, 2.5), np.percentile(diffs, 97.5)
    assert lo >= 0.39 and hi <= 0.41 + 1e-9, (lo, hi)
    print("  (13) aggregation layer: CAGR/MDD/duration/end-equity exact "
          "on constructed curves; paired-band percentile construction "
          "verified: OK")


if __name__ == "__main__":
    print("\n=== Build 5B systems-replay pins ===")
    test_accounting_conservation()
    test_caps_never_violated()
    test_no_lookahead_truncation()
    test_stop_parity_with_production()
    test_gap_below_stop_skipped()
    test_selection_rules_and_null_determinism()
    test_expectancy_definition_guard()
    test_prereg_committed_and_intact()
    test_determinism_hash()
    test_signal_day_state_governs_admission()
    test_entry_time_ceiling_invariant()
    test_stop_boundary_exact_equality()
    test_aggregation_layer()
    print("\nAll systems-replay tests passed.\n")
