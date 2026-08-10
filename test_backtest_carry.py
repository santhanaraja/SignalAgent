#!/usr/bin/env python3
"""
Build 7 pins — extension-guard carry replay.

Per the pin doctrine (docs/testing.md). The study STOPPED at its own
pre-registered integrity gate (S1 must reproduce 5B's S1 cent-exactly);
these pins hold everything that WAS established: the re-grade integrity
gate on the guard-off column, the machinery's fidelity, the derived
risk budget's identity, the scale-invariance fact that governs how the
paired comparison must be read, and — Law 3 — a DEMONSTRATION of the
reproducibility defect that stopped the run.

Run: python3 test_backtest_carry.py
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

import backtest_carry as bc
import backtest_systems as bs
from backtest_systems import curve_metrics

# the universe artifact as it stood when 5B and 6A were RUN — the group
# map their committed results were produced under
PINNED_MAP_COMMIT = "1d67d1c"


def _group_map(ref):
    r = subprocess.run(["git", "show", f"{ref}:public/universe_ranking.json"],
                       capture_output=True, text=True, cwd=REPO)
    assert r.returncode == 0, f"{ref} unreachable — anchor a COMMIT"
    rk = json.loads(r.stdout)
    m = {}
    for g in rk.get("groups", []):
        for t in g.get("tickers", []) or []:
            sym = t.get("ticker") if isinstance(t, dict) else t
            if sym:
                m.setdefault(sym, g.get("name"))
    return m


def test_regrade_integrity_gate():
    """THE load-bearing pin for the guard-off column: the side-car's
    guard-ON grade must reproduce the committed Layer A frame on ALL
    rows (not a sample), and removing the guard may only ever LIFT a
    grade. If the reconstruction cannot reproduce the incumbent, the
    guard-off column is not trustworthy either."""
    assert os.path.exists(bc.NOGUARD_PATH), (
        "guard-off side-car missing — run "
        "scripts/build_noguard_frame.py (pins must not silently pass)")
    ng = pd.read_csv(bc.NOGUARD_PATH)
    frame = pd.read_csv(os.path.join(bs.CACHE, "master_frame.csv.gz"),
                        usecols=["date", "ticker", "grade"])
    m = frame.merge(ng, on=["date", "ticker"], how="outer",
                    suffixes=("_committed", "_sidecar"), indicator=True)
    assert (m["_merge"] == "both").all(), \
        m["_merge"].value_counts().to_dict()
    bad = m[m.grade_committed != m.grade_sidecar]
    assert len(bad) == 0, f"{len(bad)} of {len(m):,} rows diverge"
    lower = m[m.grade_noguard < m.grade_sidecar]
    assert len(lower) == 0, \
        f"{len(lower)} rows graded LOWER with the guard removed"
    lifted = int((m.grade_noguard > m.grade_sidecar).sum())
    ap_on = int((m.grade_sidecar == 3).sum())
    ap_off = int((m.grade_noguard == 3).sum())
    assert ap_on == 29777, ap_on          # the committed Layer A A+ set
    assert lifted > 0 and ap_off > ap_on
    print(f"  (1) re-grade integrity: {len(m):,} rows reproduce the "
          f"committed grade EXACTLY; guard removal only lifts "
          f"({lifted:,} days); A+ {ap_on:,} -> {ap_off:,}: OK")


PANEL = bc.load_carry_panel(smoke=True)


def test_gate_patch_introduces_nothing():
    """The S6 gate logic, pointed at the guard-ON column, must
    reproduce S1 to the cent — so the guard comparison differs by the
    COLUMN and nothing else (integrity gate b)."""
    with bc.patched_gate():
        s1 = bs.run_sim(PANEL, "S1", "score")
        oncol = bs.run_sim(PANEL, "S1_ONCOL", "score")
    dv = max(abs(a[1] - b[1]) for a, b in zip(s1["curve"], oncol["curve"]))
    assert dv < 0.01 and len(s1["trades"]) == len(oncol["trades"]), dv
    # and the guard-off column really does admit a different population
    with bc.patched_gate():
        s6 = bs.run_sim(PANEL, "S6", "score")
    k = lambda t: (t["ticker"], t["entry_date"])
    assert {k(t) for t in s6["trades"]} != {k(t) for t in s1["trades"]}
    print(f"  (2) gate patch: S6 logic on the guard-ON column == S1 to "
          f"the cent (${dv:.6f}); the guard-OFF column admits a "
          f"different population: OK")


def test_forced_replay_is_faithful():
    """A forced replay under FIXED sizing must reproduce its source sim
    to the cent (integrity gate c) — the replay engine adds nothing."""
    with bc.patched_gate():
        s1 = bs.run_sim(PANEL, "S1", "score")
    sched = bc.build_schedule(s1["trades"], PANEL)
    rep = bc.run_forced(PANEL, sched, "fixed")
    dv = max(abs(a[1] - b[1]) for a, b in zip(s1["curve"], rep["curve"]))
    assert dv < 0.01, dv
    k = lambda t: (t["ticker"], t["entry_date"])
    assert {k(t) for t in rep["trades"]} == {k(t) for t in s1["trades"]}
    print(f"  (3) forced replay under fixed sizing reproduces its "
          f"source sim (${dv:.6f}, {len(rep['trades'])} trades): OK")


def test_risk_parity_formula_on_paper():
    """Each of the three terms binds in turn, and the recorded flags
    say which — hand fixture, every share count computable on paper."""
    days = pd.bdate_range("2021-01-04", periods=40)
    c = np.linspace(100, 140, 40)
    px = {"TST": {"iso": np.array([str(d)[:10] for d in days]),
                  "pos": {str(d)[:10]: i for i, d in enumerate(days)},
                  "open": c.copy(), "close": c.copy(),
                  "sma20": c - 50.0,          # never stops out
                  "ext": np.full(40, np.nan)}}
    panel = {"days": [str(d)[:10] for d in days], "px": px,
             "cash_daily": pd.Series(0.0,
                                     index=[str(d)[:10] for d in days])}
    day0 = str(days[0])[:10]
    fill = float(c[0])                        # 100.0

    def one(stop, budget, cap=bc.CAP_PCT, cash=100_000.0):
        sched = {day0: [{"ticker": "TST", "stop": stop, "group": "G",
                         "regime": "In-Trend-Full", "reentry": False}]}
        sim = bc.run_forced(panel, sched, "risk_parity",
                            risk_budget=budget, cap_pct=cap,
                            capital=cash, slip_bps=0.0)
        return sim["trades"][0]

    # RISK binds: wide stop -> risk term is the smallest
    t = one(stop=50.0, budget=0.02)           # 0.02*100000/50 = 40 shares
    assert abs(t["shares"] - 40.0) < 1e-9 and not t["cap_bound"] \
        and not t["cash_bound"], t
    # CAP binds: tight stop -> risk term explodes, cap holds it
    t = one(stop=99.9, budget=0.02)           # risk 20000 sh; cap 120 sh
    assert abs(t["shares"] - 120.0) < 1e-9 and t["cap_bound"] \
        and not t["cash_bound"], t
    # CASH binds: cap would want more than the arm holds
    # cap 150% of a $5,000 book wants 75 shares; the book funds 50
    t = one(stop=99.9, budget=0.02, cap=150.0, cash=5_000.0)
    assert abs(t["shares"] - 50.0) < 1e-9 and t["cash_bound"] \
        and not t["cap_bound"], t
    print("  (4) risk-parity: each of risk/cap/cash binds in turn with "
          "the flag recorded, share counts exact on paper: OK")


def test_risk_budget_identity():
    """The derived budget's identity — mean(r_usd/equity_at_entry)
    reduces to size_pct*mean(1 - stop/fill), so it needs no equity
    series and is checkable on paper (declaration 6)."""
    with bc.patched_gate():
        s1 = bs.run_sim(PANEL, "S1", "score")
    sched = bc.build_schedule(s1["trades"], PANEL)
    rep = bc.run_forced(PANEL, sched, "fixed")
    rb, fr = bc.realized_risk_budget(rep["trades"])
    paper = np.mean([(bs.SIZE_PCT / 100.0)
                     * (1.0 - t["entry_stop"] / t["entry_fill"])
                     for t in rep["trades"]])
    assert abs(rb - paper) < 1e-12
    # and against the sim's own r_usd, trade by trade
    worst = 0.0
    for t in rep["trades"]:
        implied = (bs.SIZE_PCT / 100.0) * (1 - t["entry_stop"]
                                           / t["entry_fill"])
        actual = t["r_usd"] / (t["shares"] * t["entry_fill"]
                               / (bs.SIZE_PCT / 100.0))
        worst = max(worst, abs(implied - actual))
    assert worst < 1e-12, worst
    assert 0 < rb < 0.065
    print(f"  (5) risk budget {rb * 100:.4f}%/trade: identity holds to "
          f"{worst:.1e} against the sim's own r_usd: OK")


def test_r_multiple_is_scale_invariant():
    """THE READING RULE for the paired comparison: r_mult = pnl/r_usd
    and shares cancel exactly, so a sizing change cannot move any
    trade's R-multiple. The prereg's paired per-trade R bootstrap is
    therefore structurally blind to the sizing effect — which lives in
    the portfolio metrics. Demonstrated, not asserted."""
    with bc.patched_gate():
        s1 = bs.run_sim(PANEL, "S1", "score")
    sched = bc.build_schedule(s1["trades"], PANEL)
    fixed = bc.run_forced(PANEL, sched, "fixed")
    rb, _ = bc.realized_risk_budget(fixed["trades"])
    rp = bc.run_forced(PANEL, sched, "risk_parity", risk_budget=rb)
    k = lambda t: (t["ticker"], t["entry_date"])
    A = {k(t): t for t in fixed["trades"]}
    B = {k(t): t for t in rp["trades"]}
    assert set(A) == set(B)
    worst = 0.0
    funded = 0
    for x in A:
        if B[x]["shares"] < 1e-9:
            continue                     # fully cash-starved: no R at all
        funded += 1
        assert A[x]["exit_date"] == B[x]["exit_date"], x
        worst = max(worst, abs(A[x]["r_mult"] - B[x]["r_mult"]))
    assert funded > 100 and worst < 1e-9, (funded, worst)
    dollars_a = sum(t["pnl"] for t in A.values())
    dollars_b = sum(t["pnl"] for t in B.values())
    assert abs(dollars_a - dollars_b) > 1.0, "sizing moved no dollars?"
    print(f"  (6) R-multiple scale invariance: identical to {worst:.1e} "
          f"across {funded} funded trades (exits identical too) while "
          f"dollars differ — the sizing effect is not visible in R: OK")


def test_committed_5b_reproduces_under_the_pinned_map():
    """LAW 3, the defect that stopped Build 7, demonstrated both ways.

    5B's panel reads public/universe_ranking.json — a MUTABLE artifact
    the weekly rotation rewrites. The committed 5B S1 block reproduces
    cent-exactly under the group map as it stood when 5B was RUN
    (commit 1d67d1c), and does NOT under whatever the working tree
    happens to hold. The ASSERTION is the invariant that must hold
    forever (pinned map -> committed numbers); the working-tree state
    is REPORTED, not asserted, so this pin stays green once the studies
    are anchored."""
    panel = bs.load_panel(smoke=False)
    committed = json.load(open(os.path.join(
        REPO, "docs", "backtest-systems-results.json"))
    )["arms"]["S1"]["score"]["full"]
    # the WORKING-TREE artifact, read from disk — not panel["group_of"],
    # which now returns the pinned map and would compare the pin to
    # itself (a tautology; caught on review of this very pin)
    with open(os.path.join(REPO, "public",
                           "universe_ranking.json")) as f:
        rk_live = json.load(f)
    live_map = {}
    for g in rk_live.get("groups", []):
        for t in g.get("tickers", []) or []:
            sym = t.get("ticker") if isinstance(t, dict) else t
            if sym:
                live_map.setdefault(sym, g.get("name"))
    panel["group_of"] = _group_map(PINNED_MAP_COMMIT)
    s1 = bs.run_sim(panel, "S1", "score")
    m = curve_metrics(s1["curve"], panel["cash_daily"])
    for k in ("cagr_pct", "max_dd_pct", "end_equity"):
        assert m[k] == committed[k], (
            f"the committed 5B S1 no longer reproduces even under the "
            f"pinned map on {k}: {m[k]} vs {committed[k]}")
    assert len(s1["trades"]) == committed["trades"]
    moved = sum(1 for t in set(live_map) & set(panel["group_of"])
                if live_map[t] != panel["group_of"][t])
    added = len(set(live_map) - set(panel["group_of"]))
    dropped = len(set(panel["group_of"]) - set(live_map))
    assert panel["group_of"] != live_map, (
        "the working-tree artifact equals the pinned one — this pin "
        "cannot demonstrate the drift it exists to guard against; "
        "check that it is reading the live file, not the pinned map")
    print(f"  (7) committed 5B S1 reproduces cent-exactly under the "
          f"PINNED map {PINNED_MAP_COMMIT} (${committed['end_equity']:,}); "
          f"the LIVE working-tree artifact differs by {moved} regrouped "
          f"/ {added} added / {dropped} dropped tickers"
          + (" — DRIFTED, Build 7's gate stops on this"
             if (moved or added or dropped) else " — no drift"))


def test_prereg_record_discipline():
    """The prereg's ORIGINAL text is immutable; the post-registration
    amendment is appended and LABELLED, never an edit in place."""
    orig = subprocess.run(["git", "show",
                           "a6bfd58:docs/backtest-carry-prereg.md"],
                          capture_output=True, text=True, cwd=REPO).stdout
    assert orig, "the prereg commit is unreachable"
    assert hashlib.sha256(orig.encode()).hexdigest()[:16] == \
        "4b11ffd563258b70", "the prereg commit's content moved"
    now = open(os.path.join(REPO, "docs",
                            "backtest-carry-prereg.md")).read()
    assert now.startswith(orig.rstrip() + "\n"), (
        "the original pre-registration was EDITED, not appended to")
    tail = now[len(orig.rstrip()):]
    for phrase in ("POST-REGISTRATION AMENDMENT 1",
                   "BEFORE any performance number was read",
                   "IMPLEMENTER, not the author",
                   "share-scale invariant"):
        assert phrase in tail, f"amendment missing: {phrase}"
    # and the void test is carried, labelled, in the results
    res = json.load(open(os.path.join(
        REPO, "docs", "backtest-carry-results.json")))
    for lab in ("sizing_S1_S8", "sizing_S6_S7"):
        v = res["comparisons"][lab]["void_per_trade_r_bootstrap"]
        assert "share-scale invariant" in v["void_reason"]
    print("  (8) prereg: original text immutable vs a6bfd58, amendment "
          "APPENDED and labelled, void test carried labelled: OK")


if __name__ == "__main__":
    test_regrade_integrity_gate()
    test_gate_patch_introduces_nothing()
    test_forced_replay_is_faithful()
    test_risk_parity_formula_on_paper()
    test_risk_budget_identity()
    test_r_multiple_is_scale_invariant()
    test_committed_5b_reproduces_under_the_pinned_map()
    test_prereg_record_discipline()
    print("All carry-replay pins passed.")
