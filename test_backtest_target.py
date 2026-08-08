#!/usr/bin/env python3
"""
Build 6A pins — profit-target sweep.

Per the pin doctrine (docs/testing.md). The load-bearing pin is ENGINE
PARITY: the target engine with the rule switched off must reproduce
the 5B engine's curve and trade log to the cent — every arm statistic
rests on that. The mechanics pins drive run_target_sim itself (the
real engine) over hand-built panels where every fill is computable on
paper.

Run: python3 test_backtest_target.py
"""

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

import backtest_target as bt
import backtest_systems as bs

HAVE = os.path.isdir(bs.PRICES) and os.path.exists(bs.FRAME_PATH)
if not HAVE:
    raise AssertionError(
        "doctrine caches missing — run scripts/fetch_doctrine_cache.py "
        "first (pins must not silently pass)")

SLIP = bs.SLIP_BPS / 10_000.0


# ---------------------------------------------------------- fixtures
def fixture_panel(bars, ticker="TST"):
    """A hand panel: bars = list of (iso, open, high, close, sma20).
    Zero cash yield so every cent is trade arithmetic."""
    iso = np.array([b[0] for b in bars])
    px = {ticker: {
        "iso": iso, "pos": {s: i for i, s in enumerate(iso)},
        "open": np.array([b[1] for b in bars], dtype=float),
        "high": np.array([b[2] for b in bars], dtype=float),
        "close": np.array([b[3] for b in bars], dtype=float),
        "sma20": np.array([b[4] for b in bars], dtype=float),
        "ext": np.full(len(bars), np.nan)}}
    days = list(iso)
    return {"days": days, "px": px,
            "cash_daily": pd.Series(0.0, index=days)}


def sched(day, stop, ticker="TST"):
    return {day: [{"ticker": ticker, "stop": stop, "group": "G",
                   "regime": "In-Trend-Full", "reentry": False}]}


D = [f"2021-01-{i:02d}" for i in range(1, 21)]


def test_intrabar_fill_and_gap_fill():
    """Declaration 6: a touched level fills AT the level; a day that
    opens through it fills at the OPEN (better price)."""
    # entry open 100, stop 90 -> R=10, 2R target 120
    bars = [(D[0], 100, 105, 102, 90),
            (D[1], 103, 121, 110, 91),     # touch -> fill at 120
            (D[2], 104, 118, 108, 92)]
    sim = bt.run_target_sim(fixture_panel(bars), sched(D[0], 90.0), "T1")
    hits = sim["trades"][0]["targets_hit"]
    assert len(hits) == 1 and hits[0]["fill"] == 120.0, hits
    bars2 = [(D[0], 100, 105, 102, 90),
             (D[1], 125, 130, 118, 91)]    # gap OPEN above 120
    sim2 = bt.run_target_sim(fixture_panel(bars2), sched(D[0], 90.0), "T1")
    hits2 = sim2["trades"][0]["targets_hit"]
    assert len(hits2) == 1 and hits2[0]["fill"] == 125.0, hits2
    # declaration 5: the limit is live from the entry fill — a touch on
    # the ENTRY DAY itself fills (review finding: was unpinned)
    bars3 = [(D[0], 100, 121, 110, 90),
             (D[1], 110, 111, 112, 91)]
    sim3 = bt.run_target_sim(fixture_panel(bars3), sched(D[0], 90.0), "T1")
    hits3 = sim3["trades"][0]["targets_hit"]
    assert len(hits3) == 1 and hits3[0]["date"] == D[0] \
        and hits3[0]["fill"] == 120.0, hits3
    print("  (1) intrabar fill at the level; gap-through fills at the "
          "open; entry-day touch fills: OK")


def test_r_never_rebases():
    """The prereg's R definition: set at entry, never re-based. The
    SMA20 trails far above the entry; the target must still sit at
    entry+2R, and the REBASED level (close-stop distance) would differ
    — demonstrated, not assumed."""
    bars = [(D[0], 100, 101, 100, 90)] + \
        [(D[i], 100 + i, 101 + i, 100 + i, 96 + i) for i in range(1, 15)]
    # day 10: sma20 = 106 (> entry); a re-based 2R from there would be
    # close+2*(close-sma20) — far below entry+2R=120. High touches 119.
    bars[10] = (D[10], 110, 119.0, 111, 106)
    sim = bt.run_target_sim(fixture_panel(bars), sched(D[0], 90.0), "T1")
    tr = sim["trades"][0]
    assert tr["targets_hit"] == [], "119 < 120: the ORIGINAL level " \
        "must not have filled"
    rebased_level = 111 + 2 * (111 - 106)          # 121 -> would differ
    assert rebased_level != 120.0
    bars[10] = (D[10], 110, 120.0, 111, 106)       # exact touch
    sim2 = bt.run_target_sim(fixture_panel(bars), sched(D[0], 90.0), "T1")
    assert sim2["trades"][0]["targets_hit"][0]["fill"] == 120.0
    print("  (2) R fixed at entry (touch at 120 fills, 119 does not; "
          "re-based level would be 121): OK")


def test_stop_exit_cancels_limits():
    """Declaration 8: a stop fired on yesterday's close sells the WHOLE
    position at today's open; today's high touching the target fills
    nothing."""
    bars = [(D[0], 100, 101, 100, 90),
            (D[1], 100, 101, 95, 96),      # close 95 <= sma20 96 -> fired
            (D[2], 97, 125, 99, 96)]       # high 125 >= 120 — too late
    sim = bt.run_target_sim(fixture_panel(bars), sched(D[0], 90.0), "T1")
    tr = sim["trades"][0]
    assert tr["targets_hit"] == [] and tr["exit_date"] == D[2] \
        and tr["exit_fill"] == 97.0, tr
    print("  (3) fired stop exits whole at the open and cancels the "
          "resting limit: OK")


def test_floor_semantics_t4():
    """Declaration 9 + prereg: the floor is max(SMA20, entry) — it
    never lowers a stop, binds only while SMA20 < entry, and an
    intrabar arm applies to THAT evening's close check. Equality exits
    (D-018)."""
    # arm day: high 121 >= 120 (2R); close 100.0 == entry -> NOT above
    # the floored level -> exits that same evening
    bars = [(D[0], 100, 101, 100, 90),
            (D[1], 102, 121, 100.0, 92),
            (D[2], 101, 102, 103, 92)]
    sim = bt.run_target_sim(fixture_panel(bars), sched(D[0], 90.0), "T4")
    tr = sim["trades"][0]
    assert tr["floored"] and tr["exit_date"] == D[2], tr
    # discriminate the SAME-EVENING clause (review finding): an engine
    # that floors only from the NEXT close survives to the panel end
    # and closes synthetically at 103 — these two assertions kill it
    assert not tr["synthetic_close"] and tr["exit_fill"] == 101.0, tr
    assert tr["targets_hit"] == []            # T4 sells nothing at 2R
    # same tape, no floor arm (high 119): close 100 > sma20 92 -> holds
    bars2 = [(D[0], 100, 101, 100, 90),
             (D[1], 102, 119, 100.0, 92),
             (D[2], 101, 102, 103, 92)]
    sim2 = bt.run_target_sim(fixture_panel(bars2), sched(D[0], 90.0), "T4")
    assert sim2["trades"][0]["exit_date"] == D[2] \
        and sim2["trades"][0]["synthetic_close"], "un-floored must hold"
    # never LOWERS: sma20 above entry governs even when floored
    bars3 = [(D[0], 100, 101, 100, 90),
             (D[1], 102, 121, 110, 92),
             (D[2], 110, 112, 105, 106),   # close 105 <= sma20 106
             (D[3], 104, 105, 106, 100)]
    sim3 = bt.run_target_sim(fixture_panel(bars3), sched(D[0], 90.0), "T4")
    assert sim3["trades"][0]["exit_date"] == D[3] \
        and sim3["trades"][0]["exit_fill"] == 104.0
    print("  (4) T4 floor: arms intrabar, applies same evening, "
          "equality exits, never lowers the SMA20 stop: OK")


def test_t5_floor_arms_on_fill_only():
    """Declaration 9: T5's floor arms the day the 3R leg FILLS — a 2.9R
    excursion arms nothing."""
    bars = [(D[0], 100, 101, 100, 90),
            (D[1], 102, 128, 95, 91),      # 2.8R high; no 3R, no floor;
                                           # close 95 > sma20 91 holds
            (D[2], 96, 131, 101, 91),      # 3R=130 fills; floor arms —
                                           # close 101 > entry survives
                                           # the SAME-evening floor check
            (D[3], 97, 98, 96, 91),        # close 96 <= entry 100 (floor)
            (D[4], 95, 96, 99, 91)]
    sim = bt.run_target_sim(fixture_panel(bars), sched(D[0], 90.0), "T5")
    tr = sim["trades"][0]
    assert tr["targets_hit"][0]["date"] == D[2]
    assert tr["floored"] and tr["exit_date"] == D[4] \
        and tr["exit_fill"] == 95.0, tr
    sim_b = bt.run_target_sim(fixture_panel(bars[:2] + [
        (D[2], 96, 97, 96, 91)]), sched(D[0], 90.0), "T5")
    assert not sim_b["trades"][0]["floored"], "floor armed without a fill"
    print("  (5) T5 floor arms only on the 3R fill; floored remainder "
          "exits at breakeven: OK")


def test_t3_fractions_and_conservation():
    """Declaration 7: exact thirds of ORIGINAL size; both legs can fill
    the same day; the accounting identity holds to the cent."""
    bars = [(D[0], 100, 101, 100, 90),
            (D[1], 102, 145, 120, 91),     # high through 4R=140: BOTH
            (D[2], 120, 121, 118, 119)]    # close 118 <= sma20 -> exit
    sim = bt.run_target_sim(fixture_panel(bars), sched(D[0], 90.0), "T3")
    tr = sim["trades"][0]
    assert [h["r"] for h in tr["targets_hit"]] == [2.0, 4.0]
    assert [h["fill"] for h in tr["targets_hit"]] == [120.0, 140.0]
    # PAPER ARITHMETIC, independent of the engine's own identity
    # (review finding: a slip-sign flip on leg proceeds passed the old
    # pin). Entry: 6.5% of 100k at open 100 -> 65 shares, cost
    # 65*100*(1+slip). Legs: 65/3 shares at 120 and 140, each *(1-slip).
    # The D[2] close 118 <= sma20 119 fires the stop, but the panel
    # ends there -> remainder marks synthetically at 118, NO slip.
    cap = 100_000.0
    shares = cap * 0.065 / 100.0
    paper_pnl = (shares / 3 * 120.0 * (1 - SLIP)
                 + shares / 3 * 140.0 * (1 - SLIP)
                 + shares / 3 * 118.0
                 - shares * 100.0 * (1 + SLIP))
    assert abs(tr["pnl"] - paper_pnl) < 1e-6, (tr["pnl"], paper_pnl)
    assert tr["synthetic_close"]
    # conservation: engine asserts internally; verify externally too
    end = sim["curve"][-1][1]
    total_pnl = sum(t["pnl"] for t in sim["trades"])
    assert abs(end - (cap + total_pnl + sim["cash_yield_usd"])) < 0.01
    # remainder third: r_usd = shares*(100-90); pnl consistent with legs
    assert abs(tr["r_usd"] - shares * 10.0) < 1e-9
    print("  (6) T3 thirds, same-day double fill, and the trade P&L "
          "equals the PAPER number (slip on every leg): OK")


# ------------------------------------------------- smoke-panel pins
PANEL = bt.add_highs(bs.load_panel(smoke=True))
S1 = bs.run_sim(PANEL, "S1", "score")
SCHED = bt.build_schedule(S1["trades"])
T0 = bt.run_target_sim(PANEL, SCHED, "T0")


def test_engine_parity_rule_off():
    """THE load-bearing pin: rule off == the 5B engine, to the cent,
    on the smoke panel (the full-panel twin runs inside run())."""
    dv = max(abs(a[1] - b[1]) for a, b in zip(S1["curve"], T0["curve"]))
    assert dv < 0.01, f"curve diverges by ${dv:.4f}"
    k = lambda t: (t["ticker"], t["entry_date"])
    p0 = {k(t): t for t in S1["trades"]}
    pt = {k(t): t for t in T0["trades"]}
    assert set(p0) == set(pt) and len(S1["trades"]) == len(T0["trades"])
    dpnl = max(abs(p0[x]["pnl"] - pt[x]["pnl"]) for x in p0)
    assert dpnl < 0.01, f"trade P&L diverges by ${dpnl:.4f}"
    assert len(p0) > 400, f"smoke population too thin ({len(p0)})"
    print(f"  (7) engine parity, rule off: {len(p0)} trades, max curve "
          f"diff ${dv:.6f}, max trade diff ${dpnl:.6f}: OK")


def test_population_fixed_and_scale_free():
    """Declaration 2 + 3: every arm trades EXACTLY T0's population, and
    a trade the target never touched has the SAME r_mult as T0's
    (share-scale invariance — sizing off a diverged equity cannot leak
    into the paired R comparison)."""
    k = lambda t: (t["ticker"], t["entry_date"])
    t0map = {k(t): t for t in T0["trades"]}
    for arm in ("T1", "T4"):
        sim = bt.run_target_sim(PANEL, SCHED, arm)
        amap = {k(t): t for t in sim["trades"]}
        assert set(amap) == set(t0map), f"{arm}: population differs"
        untouched = [x for x in amap
                     if not amap[x]["targets_hit"]
                     and not amap[x]["floored"]]
        assert len(untouched) > 100, "no untouched trades to compare"
        worst = max(abs(amap[x]["r_mult"] - t0map[x]["r_mult"])
                    for x in untouched)
        assert worst < 1e-9, f"{arm}: untouched trade r_mult drifts " \
            f"{worst} — scale leaked into the pairing"
    print(f"  (8) population fixed across arms; untouched trades' "
          f"r_mult identical to T0 (scale-free pairing): OK")


def test_determinism():
    sim_a = bt.run_target_sim(PANEL, SCHED, "T3")
    sim_b = bt.run_target_sim(PANEL, SCHED, "T3")
    assert sim_a["curve"] == sim_b["curve"]
    assert sim_a["trades"] == sim_b["trades"]
    print("  (9) determinism: identical T3 replay twice: OK")


# ---------------------------------------------------- stats fixtures
def test_stats_layer_fixtures():
    """cluster bootstrap, paired path bootstrap, concentration and the
    verdict table — each on hand inputs with paper answers."""
    diffs = np.array([1.0, 1.0, 3.0, 3.0])
    clusters = np.array(["a", "a", "b", "b"])
    mean, ci = bt.cluster_boot_mean(diffs, clusters, n_boot=500, seed=7)
    assert mean == 2.0 and ci[0] >= 1.0 and ci[1] <= 3.0, (mean, ci)
    curve = [(d, 100.0 + i) for i, d in enumerate(D)]
    pb = bt.paired_path_boot(curve, curve, n_boot=50, block=5, seed=7)
    assert pb["d_cagr_ci"] == [0.0, 0.0] and pb["d_mdd_ci"] == [0.0, 0.0]
    # NON-degenerate fixtures (review finding: the identical-curves
    # fixture is blind to sign flips, reversed bounds and annualization).
    # (a) constant daily returns 0.10% vs 0.11%: EVERY block resample
    # reproduces the same constant series, so the CI collapses to the
    # PAPER value of the CAGR gap — pins annualization and sign exactly.
    days2 = [f"2022-{m:02d}-{dd:02d}" for m in range(1, 13)
             for dd in range(1, 29)][:200]
    c0 = [(d, 100.0 * 1.001 ** i) for i, d in enumerate(days2)]
    ca = [(d, 100.0 * 1.0011 ** i) for i, d in enumerate(days2)]
    pb2 = bt.paired_path_boot(c0, ca, n_boot=50, block=21, seed=7)
    paper = round((1.0011 ** 252 - 1.001 ** 252) * 100, 3)
    assert pb2["d_cagr_ci"] == [paper, paper], (pb2["d_cagr_ci"], paper)
    assert pb2["d_mdd_ci"] == [0.0, 0.0]
    # (b) one -10% crash day only in the arm: dMDD is negative whenever
    # a replicate samples the crash (blocks resample WITH replacement,
    # so k inclusions compound to 0.9^k-1 — the lower bound may sit
    # below -10), the upper bound cannot exceed zero, and the bounds
    # must be ordered
    cb = []
    v = 100.0
    for i, d in enumerate(days2):
        if i == 100:
            v *= 0.9
        cb.append((d, v))
    flat = [(d, 100.0) for d in days2]
    pb3 = bt.paired_path_boot(flat, cb, n_boot=400, block=21, seed=7)
    lo, hi = pb3["d_mdd_ci"]
    assert lo <= hi <= 0.0 and -100.0 < lo < 0.0, pb3
    keys = [("A", "d1"), ("B", "d2"), ("C", "d3"), ("Dd", "d4")]
    r0 = np.zeros(4)
    ra = np.array([10.0, 1.0, 0.5, 0.5])
    conc = bt.concentration(keys, r0, ra, np.ones(4) * 100.0)
    assert conc["top3_share_pct"] == round(11.5 / 12.0 * 100, 1)
    assert conc["fragile"] and conc["top3"][0]["ticker"] == "A"
    keys10 = [(f"T{i}", f"d{i}") for i in range(10)]
    conc2 = bt.concentration(keys10, np.zeros(10), np.ones(10),
                             np.ones(10) * 100.0)
    assert not conc2["fragile"] and conc2["top3_share_pct"] == 30.0
    assert bt.verdict(1.0, [0.5, 1.5], 1.0, [0.5, 1.5]) == "1-ADOPT"
    assert bt.verdict(-1.0, [-1.5, -0.5], 1.0, [0.5, 1.5]) \
        == "2-EXCHANGE-RATE"
    assert bt.verdict(-1.0, [-1.5, -0.5], -1.0, [-1.5, -0.5]) \
        == "3-REJECT"
    assert bt.verdict(0.1, [-0.5, 0.5], 0.1, [-0.5, 0.5]) == "4-INSIDE-CI"
    assert bt.verdict(1.0, [0.5, 1.5], -1.0, [-1.5, -0.5]) \
        .startswith("UNCOVERED")
    print("  (10) stats fixtures: cluster CI, degenerate path boot, "
          "concentration shares, all five verdict branches: OK")


# ------------------------------------------------- record discipline
def test_prereg_and_declarations_discipline():
    """Both records committed, content-immutable against their SHAs,
    and strictly preceding any results commit."""
    for path, sha, phrases in (
            ("docs/backtest-target-prereg.md", "d184ced",
             ("ADOPT", "EXCHANGE RATE", "REJECT",
              "psychological argument ALONE", "FRAGILE",
              "No threshold beyond", "NOT redeployed")),
            ("docs/backtest-target-declarations.md", "6bbca06",
             ("to the cent", "R-multiples", "21-day block",
              "FORCED entries", "max(SMA20, entry fill)"))):
        committed = subprocess.run(
            ["git", "show", f"{sha}:{path}"], capture_output=True,
            cwd=REPO).stdout
        assert committed, f"{path} not in {sha}"
        with open(os.path.join(REPO, path), "rb") as f:
            tree = f.read()
        assert hashlib.sha256(committed).hexdigest() \
            == hashlib.sha256(tree).hexdigest(), \
            f"{path}: working tree differs from the committed record"
        txt = " ".join(tree.decode().split())
        for ph in phrases:
            assert ph in txt, f"{path} missing: {ph}"
        res = subprocess.run(
            ["git", "log", "--format=%H", "--",
             "docs/backtest-target-results.json"],
            capture_output=True, text=True, cwd=REPO).stdout.split()
        if res:
            full = subprocess.run(["git", "rev-parse", sha],
                                  capture_output=True, text=True,
                                  cwd=REPO).stdout.strip()
            order = subprocess.run(
                ["git", "merge-base", "--is-ancestor", full, res[-1]],
                cwd=REPO).returncode
            assert order == 0, f"{path} does not precede the results"
    print("  (11) prereg + declarations: committed, immutable vs "
          "d184ced/6bbca06, precede any results commit: OK")


def test_touch_recount_independent():
    """Review finding: no pin anchored a TOUCHED arm's aggregate.
    Recount T1's 2R touches straight from the raw bars — a position is
    touch-eligible from its entry day through the day BEFORE its exit
    fill (the exit sells at the open; synthetic closes stay eligible
    through the final day) — and the recounted touched-trade set must
    equal the engine's exactly."""
    t1 = bt.run_target_sim(PANEL, SCHED, "T1")
    k = lambda t: (t["ticker"], t["entry_date"])
    days = PANEL["days"]
    daypos = {d: i for i, d in enumerate(days)}
    recount = set()
    for tr in t1["trades"]:
        p = PANEL["px"][tr["ticker"]]
        i0 = p["pos"][tr["entry_date"]]
        entry_fill = p["open"][i0]
        level = entry_fill + 2.0 * (entry_fill - tr["entry_stop"])
        lastpos = daypos[tr["exit_date"]] - (
            0 if tr["synthetic_close"] else 1)
        for di in range(daypos[tr["entry_date"]], lastpos + 1):
            i = p["pos"].get(days[di])
            if i is not None and np.isfinite(p["high"][i]) \
                    and p["high"][i] >= level:
                recount.add(k(tr))
                break
    engine = {k(t) for t in t1["trades"] if t["targets_hit"]}
    assert recount == engine, (
        f"recount {len(recount)} vs engine {len(engine)}; "
        f"only-recount {list(recount - engine)[:3]}, "
        f"only-engine {list(engine - recount)[:3]}")
    assert len(engine) > 200, f"smoke touched set too thin ({len(engine)})"
    print(f"  (13) independent 2R touch recount from raw bars matches "
          f"the engine's touched set exactly ({len(engine)} trades): OK")


def test_results_hash_recomputable():
    p = os.path.join(REPO, "docs", "backtest-target-results.json")
    if not os.path.exists(p):
        print("  (12) results hash: results not yet written — SKIPPED "
              "(loudly)")
        return
    r = json.load(open(p))
    stored = r.pop("results_hash")
    got = hashlib.sha256(
        json.dumps(json.loads(json.dumps(r)),
                   sort_keys=True).encode()).hexdigest()[:16]
    assert got == stored, (got, stored)
    print(f"  (12) results hash recomputable ({stored}): OK")


if __name__ == "__main__":
    for fn in (test_intrabar_fill_and_gap_fill,
               test_r_never_rebases,
               test_stop_exit_cancels_limits,
               test_floor_semantics_t4,
               test_t5_floor_arms_on_fill_only,
               test_t3_fractions_and_conservation,
               test_engine_parity_rule_off,
               test_population_fixed_and_scale_free,
               test_determinism,
               test_stats_layer_fixtures,
               test_prereg_and_declarations_discipline,
               test_touch_recount_independent,
               test_results_hash_recomputable):
        fn()
    print("All profit-target pins passed.")
