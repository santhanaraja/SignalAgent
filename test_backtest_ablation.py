#!/usr/bin/env python3
"""
Build 5.1 pins — row ablation.

Per the pin doctrine (docs/testing.md). The load-bearing pin is FLAG
PARITY: the ablation's recomputed row flags, ANDed together, must
reproduce the committed Layer A frame's A+ set — on the sampled panel
here and (asserted inside the run itself) on all 848,328 ticker-days.

Run: python3 test_backtest_ablation.py
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

import backtest_ablation as ba

HAVE = os.path.isdir(ba.PRICES) and os.path.exists(ba.FRAME_PATH)


def _sample_panel(n_tickers=25, seed=13):
    if not HAVE:
        raise AssertionError(
            "doctrine caches missing — run scripts/fetch_doctrine_cache.py "
            "and scripts/backtest_doctrine.py first (pins must not "
            "silently pass)")
    with open(ba.REGIME_PATH) as f:
        regime = json.load(f)
    with open(ba.EARNINGS_PATH) as f:
        earnings = json.load(f)
    end = regime["provenance"]["series_range"][1]
    rng = np.random.default_rng(seed)
    files = sorted(os.listdir(ba.PRICES))
    pick = sorted(rng.choice(len(files), size=n_tickers, replace=False))
    rows = []
    for i in pick:
        t = files[i][:-4]
        r = ba.ticker_flags(t, os.path.join(ba.PRICES, files[i]),
                            earnings.get(t) or {"status": "missing"},
                            regime["states"], end)
        if r:
            rows.extend(r)
    return pd.DataFrame(rows, columns=ba.COLS)


PANEL = _sample_panel()


def test_flag_parity_with_committed_frame():
    """The joint AND of the recomputed flags must equal the committed
    frame's A+ set on the sampled tickers — every flag is validated
    jointly. (The full-panel assertion runs inside run() itself; this
    pin proves the mechanism independently on a random sample.)"""
    df = PANEL.copy()
    df["aplus"] = df[list(ba.ROWS)].all(axis=1)
    frame = pd.read_csv(ba.FRAME_PATH,
                        usecols=["date", "ticker", "grade"])
    m = df.merge(frame, on=["date", "ticker"], how="inner")
    assert len(m) == len(df), "sampled days missing from the frame"
    bad = m[(m.aplus) != (m.grade == 3)]
    assert len(bad) == 0, f"{len(bad)} parity mismatches:\n{bad.head(5)}"
    n_ap = int(df.aplus.sum())
    assert n_ap > 100, f"sample too thin to trust ({n_ap} A+ days)"
    print(f"  (1) flag parity: {len(m):,} sampled ticker-days, "
          f"{n_ap:,} A+ — recomputed flags reproduce the committed "
          f"frame exactly: OK")


def test_excluded_sets_fail_exactly_one_row():
    """Each excluded set must fail its own row and pass every other —
    by construction, verified by re-checking the flags."""
    df = PANEL
    checked = 0
    for rname in ba.ROWS:
        others = [x for x in ba.ROWS if x != rname]
        excl = df[df[others].all(axis=1) & ~df[rname]]
        if not len(excl):
            continue
        assert (~excl[rname]).all(), rname
        assert excl[others].all(axis=1).all(), rname
        n_fail = (~excl[list(ba.ROWS)]).sum(axis=1)
        assert (n_fail == 1).all(), (
            rname, "a member fails more than its own row")
        checked += 1
    assert checked >= 5, checked
    print(f"  (2) excluded sets: {checked} non-empty sets each fail "
          "EXACTLY their own row: OK")


def test_c1_structurally_empty():
    """c1 cannot fail alone: with close<=SMA20 the confirmation count
    is 0 and the ATR break requires being above — so c2 fails whenever
    c1 does. Asserted on the panel AND demonstrated from the flags."""
    df = PANEL
    others = [x for x in ba.ROWS if x != "c1"]
    excl = df[df[others].all(axis=1) & ~df.c1]
    assert len(excl) == 0, f"{len(excl)} days fail c1 alone — the " \
        "structural dependency claim is wrong"
    # the demonstration: every c1-failing day also fails c2
    c1_fail = df[~df.c1]
    assert len(c1_fail) > 100, "sample carries no c1 failures"
    assert (~c1_fail.c2).all(), "a day failed c1 but passed c2 — " \
        "the confirmation count survived below the SMA20?"
    print(f"  (3) c1 structural emptiness: 0 lone-c1 failures; all "
          f"{len(c1_fail):,} c1-failing days also fail c2: OK")


def test_penalty_flip_subset():
    """The YTD-penalty construct is a strict subset of R5's excluded
    set: same everything-else, r5 fails, r5_nopen passes."""
    df = PANEL
    others5 = [x for x in ba.ROWS if x != "r5"]
    r5_excl = df[df[others5].all(axis=1) & ~df.r5]
    flip = df[df[others5].all(axis=1) & ~df.r5 & df.r5_nopen]
    assert len(flip) <= len(r5_excl)
    keys = set(map(tuple, flip[["date", "ticker"]].to_numpy()))
    keys5 = set(map(tuple, r5_excl[["date", "ticker"]].to_numpy()))
    assert keys <= keys5, "flip days outside the r5 excluded set"
    print(f"  (4) penalty-flip construct: {len(flip)} days, strict "
          f"subset of r5's excluded set ({len(r5_excl)}): OK")


def test_results_hash_recomputable():
    """The committed results JSON must rehash to its own stored value
    (the 5B canonical-hash convention)."""
    p = os.path.join(REPO, "docs", "backtest-ablation-results.json")
    assert os.path.exists(p), "results JSON missing"
    r = json.load(open(p))
    stored = r.pop("results_hash")
    got = hashlib.sha256(
        json.dumps(json.loads(json.dumps(r)),
                   sort_keys=True).encode()).hexdigest()[:16]
    assert got == stored, (got, stored)
    print(f"  (5) results hash recomputable from the committed file "
          f"({stored}): OK")


def test_prereg_precedes_results():
    """The prereg commit must strictly precede any results commit —
    a decision table written after the answer is not a prereg."""
    pre = subprocess.run(
        ["git", "log", "--format=%H", "--",
         "docs/backtest-ablation-prereg.md"],
        capture_output=True, text=True, cwd=REPO).stdout.split()
    assert pre, "prereg is not committed"
    txt = " ".join(open(os.path.join(
        REPO, "docs", "backtest-ablation-prereg.md")).read().split())
    for phrase in ("PRIMARY SUSPECT", "INCONCLUSIVE",
                   "the row is doing real work",
                   "the mechanism hypothesis is wrong",
                   "jointly AND separately", "no additions after"):
        assert phrase in txt, f"prereg missing: {phrase}"
    res = subprocess.run(
        ["git", "log", "--format=%H", "--",
         "docs/backtest-ablation-results.json"],
        capture_output=True, text=True, cwd=REPO).stdout.split()
    if res:
        order = subprocess.run(
            ["git", "merge-base", "--is-ancestor", pre[-1], res[-1]],
            cwd=REPO).returncode
        assert order == 0, "prereg does not precede the results commit"
    print("  (6) prereg: committed, all five branches present"
          + (", strictly precedes results" if res
             else " (results not yet committed)") + ": OK")


def test_determinism():
    """Same sample -> identical panel rows."""
    p2 = _sample_panel()
    h = lambda d: hashlib.sha256(pd.util.hash_pandas_object(
        d, index=False).values.tobytes()).hexdigest()
    assert h(PANEL) == h(p2)
    print("  (7) determinism: identical sampled panel across reruns: OK")


def test_stats_layer_fixture():
    """The statistics layer had NO pin (review finding): dist_block and
    the cluster-CI construction on hand-computed fixtures — a wrong
    tails computation would otherwise corrupt every table silently."""
    v = np.array([-60.0, -30.0, -12.0, -5.0, 0.5, 1.0, 2.0, 12.0, 30.0,
                  60.0])
    b = ba.dist_block(v)
    assert b["n"] == 10 and abs(b["mean"] - (-0.15)) < 1e-9
    assert b["hit_pct"] == 60.0
    assert b["gt10_pct"] == 30.0 and b["gt25_pct"] == 20.0         and b["gt50_pct"] == 10.0
    assert b["lt10_pct"] == 30.0 and b["lt25_pct"] == 20.0         and b["lt50_pct"] == 10.0
    # top-5% of 10 obs = ceil(0.5)=1 obs = the 60; ex-top mean over 9
    assert b["ex_top5_mean"] == round((v.sum() - 60.0) / 9, 3)
    # NaN handling: NaNs dropped before everything
    v2 = np.array([np.nan, 1.0, np.nan, -1.0])
    b2 = ba.dist_block(v2)
    assert b2["n"] == 2 and b2["mean"] == 0.0
    # cluster CI on a deterministic fixture: excluded always 1.0 higher
    ex = pd.DataFrame({"date": ["d1"] * 3 + ["d2"] * 3,
                       "fwd_20": [2.0, 2.0, 2.0, 3.0, 3.0, 3.0]})
    ap_ = pd.DataFrame({"date": ["d1"] * 3 + ["d2"] * 3,
                        "fwd_20": [1.0, 1.0, 1.0, 2.0, 2.0, 2.0]})
    ci = ba.cluster_ci(ex, ap_, n_boot=200)
    assert ci["ci_2_5"] == 1.0 and ci["ci_97_5"] == 1.0         and ci["p_diff_le_0"] == 0.0
    print("  (8) stats layer: dist_block tails/top5/ex-top and the "
          "cluster-CI construction exact on hand fixtures: OK")


def test_individual_flags_vs_frame_fail_row():
    """Joint-AND parity cannot validate INDIVIDUAL flags (two errors
    could cancel — review finding). The frame's fail_row column (the
    FIRST failing row, from the real grade_setup) decomposes the AND:
    for every sampled non-A+ day, all rows BEFORE fail_row must pass
    and the fail_row's own flag must fail."""
    frame = pd.read_csv(ba.FRAME_PATH,
                        usecols=["date", "ticker", "grade", "fail_row"])
    m = PANEL.merge(frame, on=["date", "ticker"], how="inner")
    non_ap = m[m.grade != 3]
    # fail_row 1 = the conditions block (any of c1..c4); then r2..r7
    order = {2: ["r2"], 3: ["r3"], 4: ["r4"], 5: ["r5"], 7: ["r7"]}
    before = {2: ["r2"], 3: ["r2", "r3"], 4: ["r2", "r3", "r4"],
              5: ["r2", "r3", "r4", "r5"],
              7: ["r2", "r3", "r4", "r5", "r7"]}
    checked = 0
    for code, cols in order.items():
        sel = non_ap[non_ap.fail_row == code]
        if not len(sel):
            continue
        # the conditions block passes for any row-level fail_row
        assert sel[["c1", "c2", "c3", "c4"]].all(axis=1).all(), code
        assert (~sel[cols[0]]).all(), (code, "flag does not fail")
        prior = [c for c in before[code][:-1]]
        if prior:
            assert sel[prior].all(axis=1).all(), (code, "an earlier row "
                                                  "fails first")
        checked += len(sel)
    cond = non_ap[non_ap.fail_row == 1]
    assert (~cond[["c1", "c2", "c3", "c4"]].all(axis=1)).all(),         "fail_row=1 days with all conditions passing"
    assert checked > 500, checked
    print(f"  (9) individual flags vs the frame's fail_row: "
          f"{checked:,} sampled non-A+ days decompose the AND with zero "
          "violations (no canceling-error blind spot): OK")


def test_prereg_content_immutable():
    """Pin 6 checked commit ORDER; this pins CONTENT — a post-hoc
    amendment to the prereg would rehash differently."""
    p = os.path.join(REPO, "docs", "backtest-ablation-prereg.md")
    h = hashlib.sha256(open(p, "rb").read()).hexdigest()[:16]
    committed = subprocess.run(
        ["git", "show", "2f98eb0:docs/backtest-ablation-prereg.md"],
        capture_output=True, cwd=REPO)
    assert committed.returncode == 0, "prereg commit 2f98eb0 unreachable"
    hc = hashlib.sha256(committed.stdout).hexdigest()[:16]
    assert h == hc, (
        "the working-tree prereg differs from the committed 2f98eb0 "
        "version — a decision table amended after results is not a "
        "pre-registration")
    print(f"  (10) prereg content immutable: working tree == commit "
          f"2f98eb0 ({h}): OK")


if __name__ == "__main__":
    print("\n=== Build 5.1 row-ablation pins ===")
    test_flag_parity_with_committed_frame()
    test_excluded_sets_fail_exactly_one_row()
    test_c1_structurally_empty()
    test_penalty_flip_subset()
    test_results_hash_recomputable()
    test_prereg_precedes_results()
    test_determinism()
    test_stats_layer_fixture()
    test_individual_flags_vs_frame_fail_row()
    test_prereg_content_immutable()
    print("\nAll row-ablation tests passed.\n")
