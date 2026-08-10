#!/usr/bin/env python3
"""
Build 7 — extension-guard carry replay (pre-registered:
docs/backtest-carry-prereg.md, a6bfd58, sha256 4b11ffd563258b70;
implementation declarations: docs/backtest-carry-declarations.md,
6330dcc — both committed BEFORE any results existed).

The 2x2:                guard ON (1.8)      guard OFF
        fixed 6.5%           S1                  S6
        risk-parity          S8                  S7

S1/S6 run through 5B's OWN engine (backtest_systems.run_sim) — one
implementation, so the guard comparison cannot be confounded by an
implementation difference: the arms differ only in which grade column
the gate reads. S8/S7 are forced-entry replays of S1/S6's realized
populations (declaration 4), which makes the sizing comparison exactly
paired.

Integrity gates, all in-run and pinned:
  (a) S1 reproduces the committed 5B S1/score block exactly.
  (b) the S6 gate logic, pointed at the guard-ON column, reproduces S1
      to the cent — proving the gate patch introduces nothing.
  (c) each forced replay under FIXED sizing reproduces its source sim
      to the cent — proving the replay engine introduces nothing.

Run: python3 scripts/backtest_carry.py [--smoke] [--out PATH]
"""

import argparse
import hashlib
import json
import os
import sys
from collections import Counter, defaultdict

REPO = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
sys.path.insert(0, REPO)
sys.path.insert(0, os.path.join(REPO, "scripts"))

import numpy as np
import pandas as pd

import backtest_systems as bs
from backtest_systems import (CEILINGS, GROUP_MAX_N, GROUP_MAX_PCT,
                              RANDOM_SEEDS, SIZE_PCT, SLIP_BPS, TRAIN_END,
                              TRADING_DAYS, curve_metrics, trade_metrics,
                              _span)

NOGUARD_PATH = os.path.join(bs.CACHE, "master_frame_noguard.csv.gz")
CAP_PCT = 12.0          # the prereg's one declared number
N_BOOT = 2000
BLOCK = 21
SEED = 20260810

ARMS = ("S1", "S6", "S8", "S7")


# --------------------------------------------------------------- gates
def carry_arm_pass(arm, r):
    """S1 = 5B's incumbent gate on the guard-ON grade; S6 = the SAME
    logic on the guard-OFF grade. Every other arm delegates to 5B
    untouched (declaration 3)."""
    if arm in ("S1", "S6"):
        g = r.grade if arm == "S1" else r.grade_noguard
        return (r.chassis == "In-Trend-Full" and g >= 2) or \
               (r.chassis == "In-Trend-Throttled" and g == 3)
    if arm == "S1_ONCOL":       # gate-patch control (integrity gate b)
        return (r.chassis == "In-Trend-Full" and r.grade >= 2) or \
               (r.chassis == "In-Trend-Throttled" and r.grade == 3)
    return bs.arm_pass(arm, r)


class patched_gate:
    """Scoped swap of 5B's module-level arm_pass. Explicit and
    restored — the alternative is duplicating the engine, which would
    put the guard comparison at the mercy of two implementations."""

    def __enter__(self):
        self._orig = bs.arm_pass
        bs.arm_pass = carry_arm_pass
        return self

    def __exit__(self, *a):
        bs.arm_pass = self._orig
        return False


def load_carry_panel(smoke=False):
    """5B's panel plus the guard-OFF grade column on every frame row."""
    if not smoke:
        from study_inputs import assert_pinned_inputs
        assert_pinned_inputs(["master_frame_noguard"], label="Build 7")
    panel = bs.load_panel(smoke=smoke)
    ng = pd.read_csv(NOGUARD_PATH)
    lookup = {(d, t): int(g) for d, t, g in
              zip(ng.date, ng.ticker, ng.grade_noguard)}
    missing = 0
    for day, rows in panel["gates_by_day"].items():
        new = []
        for r in rows:
            g = lookup.get((r.date, r.ticker))
            if g is None:
                missing += 1
                g = r.grade          # never LIFT on missing data
            new.append(_Row(r, g))
        panel["gates_by_day"][day] = new
    assert missing == 0, f"{missing} frame rows lack a guard-off grade"
    return panel


class _Row:
    """A frame row plus grade_noguard. namedtuple._replace cannot add a
    field, so the sim reads this thin carrier (same attribute names)."""
    __slots__ = ("date", "ticker", "grade", "score", "mom_6m",
                 "c1_above", "chassis", "top_decile", "grade_noguard")

    def __init__(self, r, g):
        for f in ("date", "ticker", "grade", "score", "mom_6m",
                  "c1_above", "chassis", "top_decile"):
            setattr(self, f, getattr(r, f))
        self.grade_noguard = g


# ------------------------------------------------------- forced replay
def build_schedule(trades, panel):
    """(fill day -> entries) from a sim's realized trade log, carrying
    the signal-day stop and the group/regime metadata the caps and the
    reporting need."""
    sched = defaultdict(list)
    for tr in trades:
        sched[tr["entry_date"]].append({
            "ticker": tr["ticker"], "stop": tr["entry_stop"],
            "group": tr["group"], "regime": tr["regime"],
            "reentry": tr["reentry"]})
    return sched


def run_forced(panel, schedule, sizing, risk_budget=None,
               slip_bps=SLIP_BPS, size_pct=SIZE_PCT, cap_pct=CAP_PCT,
               capital=100_000.0):
    """Replay a fixed entry population under a sizing rule. Day ordering
    mirrors 5B exactly: pending exits at the open, forced entries at the
    open, cash yield, confirmed-close mark + stop check."""
    days, px = panel["days"], panel["px"]
    cash_daily = panel["cash_daily"]
    slip = slip_bps / 10_000.0
    cash = capital
    positions, pending_exits, trades, curve = {}, [], [], []
    cash_yield_usd = 0.0
    cap_binds = cash_binds = 0

    for day in days:
        for t in pending_exits:
            pos = positions.pop(t, None)
            if pos is None:
                continue
            p = px[t]
            i = p["pos"].get(day)
            fill = p["open"][i] if i is not None else pos["last_close"]
            proceeds = pos["shares"] * fill * (1.0 - slip)
            cash += proceeds
            pnl = proceeds - pos["cost_basis"]
            trades.append({**pos["rec"], "exit_date": day,
                           "exit_fill": fill, "pnl": pnl,
                           "r_mult": pnl / pos["r_usd"]
                           if pos["r_usd"] > 0 else None,
                           "hold_days": pos["hold"],
                           "synthetic_close": False})
        pending_exits = []

        equity_prev = cash + sum(p_["shares"] * p_["last_close"]
                                 for p_ in positions.values())
        for e in schedule.get(day, ()):
            t = e["ticker"]
            if t in positions:
                raise AssertionError(
                    f"{day} {t}: scheduled entry collides with an open "
                    "position — the fixed-population construction broke")
            p = px[t]
            i = p["pos"][day]
            fill = float(p["open"][i])
            stop = float(e["stop"])
            if sizing == "fixed":
                size_usd = size_pct / 100.0 * equity_prev
                shares = size_usd / fill
                cap_bound = False
            elif sizing == "risk_parity":
                by_risk = (risk_budget * equity_prev) / (fill - stop)
                by_cap = (cap_pct / 100.0 * equity_prev) / fill
                shares = min(by_risk, by_cap)
                cap_bound = by_cap <= by_risk
            else:
                raise ValueError(sizing)
            # amendment 1: funding is a real constraint; the trade
            # STAYS (pairing survives) and the bind is RECORDED
            affordable = cash / (fill * (1.0 + slip))
            cash_bound = affordable < shares
            if cash_bound:
                shares = max(affordable, 0.0)
                cap_bound = False
            cost = shares * fill * (1.0 + slip)
            if cost > cash + 1e-6:
                raise AssertionError(
                    f"{day} {t}: cash went negative after the funding "
                    "clamp — the clamp is wrong")
            cash -= cost
            cap_binds += int(cap_bound)
            cash_binds += int(cash_bound)
            positions[t] = {
                "shares": shares, "cost_basis": cost, "hold": 0,
                "last_close": fill, "r_usd": shares * (fill - stop),
                "rec": {"ticker": t, "entry_date": day,
                        "entry_fill": fill, "entry_stop": stop,
                        "r_usd": shares * (fill - stop),
                        "group": e["group"], "regime": e["regime"],
                        "reentry": e["reentry"],
                        "cap_bound": cap_bound,
                        "cash_bound": cash_bound,
                        "shares": shares}}

        y = cash * cash_daily.loc[day]
        cash += y
        cash_yield_usd += y

        for t, pos in list(positions.items()):
            p = px[t]
            i = p["pos"].get(day)
            if i is not None and np.isfinite(p["close"][i]):
                pos["last_close"] = p["close"][i]
                pos["hold"] += 1
                if np.isfinite(p["sma20"][i]) \
                        and not p["close"][i] > p["sma20"][i]:
                    pending_exits.append(t)

        curve.append((day, cash + sum(p_["shares"] * p_["last_close"]
                                      for p_ in positions.values())))

    last_day = days[-1]
    marked = 0.0
    for t, pos in positions.items():
        proceeds = pos["shares"] * pos["last_close"]
        marked += proceeds
        pnl = proceeds - pos["cost_basis"]
        trades.append({**pos["rec"], "exit_date": last_day,
                       "exit_fill": pos["last_close"], "pnl": pnl,
                       "r_mult": pnl / pos["r_usd"]
                       if pos["r_usd"] > 0 else None,
                       "hold_days": pos["hold"], "synthetic_close": True})
    drift = (cash + marked) - (capital + sum(t["pnl"] for t in trades)
                               + cash_yield_usd)
    if abs(drift) > 0.01:
        raise AssertionError(f"conservation broken by {drift:.4f}")
    return {"curve": curve, "trades": trades,
            "cash_yield_usd": round(cash_yield_usd, 2),
            "cap_bind_rate_pct": round(100.0 * cap_binds / len(trades), 1)
            if trades else None,
            "cash_bind_rate_pct": round(100.0 * cash_binds / len(trades), 1)
            if trades else None}


def realized_risk_budget(trades):
    """The prereg's derived budget: S1's realized mean risk per trade as
    a fraction of equity at entry. r_usd/equity_prev reduces exactly to
    size_pct*(1 - stop/fill) (declaration 6), so no equity series is
    needed and the identity is checkable on paper."""
    fr = [(SIZE_PCT / 100.0) * (1.0 - t["entry_stop"] / t["entry_fill"])
          for t in trades if t.get("entry_fill")]
    return float(np.mean(fr)), fr


# ---------------------------------------------------------- statistics
def paired_trade_boot(a_trades, b_trades, n_boot=N_BOOT, seed=SEED):
    """Entry-date-cluster bootstrap on the paired per-trade R-multiple
    difference (declaration 10)."""
    key = lambda t: (t["ticker"], t["entry_date"])
    A = {key(t): t for t in a_trades}
    B = {key(t): t for t in b_trades}
    keys = sorted(set(A) & set(B))
    if len(keys) != len(A) or len(keys) != len(B):
        raise AssertionError("paired populations differ — pairing broken")
    d = np.array([(B[k]["r_mult"] or 0.0) - (A[k]["r_mult"] or 0.0)
                  for k in keys])
    clusters = np.array([k[1] for k in keys])
    uniq = np.unique(clusters)
    by = {c: d[clusters == c] for c in uniq}
    rng = np.random.default_rng(seed)
    means = np.empty(n_boot)
    for i in range(n_boot):
        pick = rng.choice(uniq, size=len(uniq), replace=True)
        means[i] = np.concatenate([by[c] for c in pick]).mean()
    return {"n_paired": len(keys),
            "mean_diff_r": round(float(d.mean()), 4),
            "ci_r": [round(float(np.percentile(means, q)), 4)
                     for q in (2.5, 97.5)]}


def paired_path_boot(c0, c1, n_boot=N_BOOT, block=BLOCK, seed=SEED):
    """Paired moving-block bootstrap of daily returns (declaration 11)."""
    v0 = np.array([x[1] for x in c0])
    v1 = np.array([x[1] for x in c1])
    r0, r1 = v0[1:] / v0[:-1] - 1.0, v1[1:] / v1[:-1] - 1.0
    n = len(r0)
    nb = int(np.ceil(n / block))
    rng = np.random.default_rng(seed)

    def stats(r):
        v = np.cumprod(1.0 + r)
        cagr = v[-1] ** (TRADING_DAYS / len(r)) - 1.0
        vv = np.concatenate([[1.0], v])
        dd = vv / np.maximum.accumulate(vv) - 1.0
        return cagr * 100.0, float(dd.min()) * 100.0

    dc, dm = np.empty(n_boot), np.empty(n_boot)
    for b in range(n_boot):
        starts = rng.integers(0, n - block + 1, size=nb)
        idx = np.concatenate([np.arange(s, s + block)
                              for s in starts])[:n]
        a_c, a_m = stats(r0[idx])
        b_c, b_m = stats(r1[idx])
        dc[b], dm[b] = b_c - a_c, b_m - a_m
    ci = lambda a: [round(float(np.percentile(a, q)), 3)
                    for q in (2.5, 97.5)]
    return {"d_cagr_ci": ci(dc), "d_mdd_ci": ci(dm)}


def random_k_series(panel, arms, seeds=RANDOM_SEEDS, cash_daily=None):
    """PAIRED-BY-SEED random-K (5B's addendum construction): both arms
    run under the SAME shuffle seed, so the guard effect is measured
    against matched selection noise instead of against two independent
    noise clouds. Comparing a score-ordered arm to a random-ordered
    band would conflate the ORDERING effect with the guard effect —
    5B measured ordering as larger than any gate effect."""
    out = {a: [] for a in arms}
    with patched_gate():
        for sd in range(seeds):
            for a in arms:
                sim = bs.run_sim(panel, a, "random", seed=sd)
                m = curve_metrics(sim["curve"], cash_daily)
                out[a].append({"seed": sd, "cagr_pct": m["cagr_pct"],
                               "max_dd_pct": m["max_dd_pct"],
                               "trades": len(sim["trades"])})
    return out


def band_summary(out):
    """One arm's marginal band."""
    cagrs = sorted(x["cagr_pct"] for x in out)
    mdds = sorted(x["max_dd_pct"] for x in out)
    pct = lambda a, q: round(float(np.percentile(a, q)), 3)
    return {"seeds": len(out),
            "cagr_band": [pct(cagrs, 2.5), pct(cagrs, 97.5)],
            "cagr_median": pct(cagrs, 50),
            "mdd_band": [pct(mdds, 2.5), pct(mdds, 97.5)],
            "mdd_median": pct(mdds, 50),
            "trades_median": int(np.median([x["trades"] for x in out]))}


def span_block(sim, cash_daily):
    out = {"full": {**curve_metrics(sim["curve"], cash_daily),
                    **trade_metrics(sim["trades"])}}
    for span, lo, hi in (("train", None, TRAIN_END),
                         ("validate", TRAIN_END, None)):
        seg = _span(sim["curve"], lo, hi)
        tr = [t for t in sim["trades"]
              if (lo is None or t["entry_date"] > lo)
              and (hi is None or t["entry_date"] <= hi)]
        out[span] = {**curve_metrics(seg, cash_daily),
                     **trade_metrics(tr)}
    return out


# ---------------------------------------------------------------- main
def run(smoke=False, out_path=None):
    panel = load_carry_panel(smoke=smoke)
    cd = panel["cash_daily"]
    print(f"panel: {len(panel['px'])} tickers, {len(panel['days'])} days, "
          f"frame_hash {panel['frame_hash']}")

    # ---- S1 + integrity gate (a) -----------------------------------
    with patched_gate():
        s1 = bs.run_sim(panel, "S1", "score")
        oncol = bs.run_sim(panel, "S1_ONCOL", "score")
    m1 = curve_metrics(s1["curve"], cd)
    gate_a = None
    if not smoke:
        with open(os.path.join(REPO, "docs",
                               "backtest-systems-results.json")) as f:
            committed = json.load(f)["arms"]["S1"]["score"]["full"]
        for k in ("cagr_pct", "max_dd_pct", "end_equity", "trades"):
            got = m1.get(k, len(s1["trades"]) if k == "trades" else None)
            if k == "trades":
                got = len(s1["trades"])
            if got != committed[k]:
                raise AssertionError(
                    f"INTEGRITY GATE (a) FAILED — S1 diverges from the "
                    f"committed 5B block on {k}: {got} vs {committed[k]}")
        gate_a = {"cagr_pct": m1["cagr_pct"], "end_equity":
                  m1["end_equity"], "trades": len(s1["trades"])}
        print(f"gate (a): S1 reproduces the committed 5B S1/score block "
              f"({m1['end_equity']:,} end equity, {len(s1['trades'])} "
              f"trades)")

    # ---- integrity gate (b): the gate patch introduces nothing ------
    dv = max(abs(a[1] - b[1]) for a, b in zip(s1["curve"], oncol["curve"]))
    if dv > 0.01 or len(oncol["trades"]) != len(s1["trades"]):
        raise AssertionError(f"INTEGRITY GATE (b) FAILED — the S6 gate "
                             f"logic on the guard-ON column diverges "
                             f"from S1 by ${dv:.4f}")
    print(f"gate (b): the S6 gate logic on the guard-ON column "
          f"reproduces S1 to the cent (max ${dv:.6f})")

    # ---- S6 --------------------------------------------------------
    with patched_gate():
        s6 = bs.run_sim(panel, "S6", "score")
    print(f"S6 (guard off, fixed): {len(s6['trades'])} trades vs S1's "
          f"{len(s1['trades'])} ({len(s6['trades']) / len(s1['trades']):.2f}x)")

    # ---- risk budget, derived from S1 ------------------------------
    sched1 = build_schedule(s1["trades"], panel)
    sched6 = build_schedule(s6["trades"], panel)
    fixed1 = run_forced(panel, sched1, "fixed")
    fixed6 = run_forced(panel, sched6, "fixed")
    # integrity gate (c): the replay engine reproduces both sims
    for label, src, rep in (("S1", s1, fixed1), ("S6", s6, fixed6)):
        d = max(abs(a[1] - b[1]) for a, b in zip(src["curve"],
                                                 rep["curve"]))
        if d > 0.01:
            raise AssertionError(
                f"INTEGRITY GATE (c) FAILED — forced replay of {label} "
                f"under fixed sizing diverges by ${d:.4f}")
    print("gate (c): forced replays under fixed sizing reproduce S1 and "
          "S6 to the cent")

    risk_budget, fr = realized_risk_budget(fixed1["trades"])
    print(f"risk_budget (derived from S1): {risk_budget * 100:.4f}% of "
          f"equity per trade (mean over {len(fr)} trades)")

    s8 = run_forced(panel, sched1, "risk_parity", risk_budget=risk_budget)
    s7 = run_forced(panel, sched6, "risk_parity", risk_budget=risk_budget)

    sims = {"S1": s1, "S6": s6, "S8": s8, "S7": s7}
    results = {
        "schema": "backtest-carry-1",
        "prereg": "docs/backtest-carry-prereg.md",
        "declarations": "docs/backtest-carry-declarations.md",
        "design": {"size_pct": SIZE_PCT, "cap_pct": CAP_PCT,
                   "slip_bps": SLIP_BPS, "risk_budget_frac": risk_budget,
                   "scorer": "v1", "selection": "score",
                   "n_boot": N_BOOT, "block_days": BLOCK, "seed": SEED},
        "integrity": {"gate_a_s1_vs_5b": gate_a,
                      "gate_b_patch_max_diff_usd": round(dv, 6),
                      "gate_c_replay": "cent-exact"},
        "provenance": {"frame_hash": panel["frame_hash"],
                       "regime": panel["regime_prov"],
                       "pinned_universe_commit":
                           bs.PINNED_UNIVERSE_COMMIT},
        "arms": {}, "validity": {}, "comparisons": {},
    }
    for a in ARMS:
        blk = span_block(sims[a], cd)
        blk["cash_yield_usd"] = sims[a].get("cash_yield_usd")
        for k in ("cap_bind_rate_pct", "cash_bind_rate_pct"):
            if k in sims[a]:
                blk[k] = sims[a][k]
        if "denials" in sims[a]:
            blk["denials"] = sims[a]["denials"]
        results["arms"][a] = blk

    # ---- validity checks, BEFORE conclusions ------------------------
    k = lambda t: (t["ticker"], t["entry_date"])
    p1, p6 = {k(t) for t in s1["trades"]}, {k(t) for t in s6["trades"]}
    results["validity"] = {
        "population_overlap": {
            "shared": len(p1 & p6), "only_S1": len(p1 - p6),
            "only_S6": len(p6 - p1),
            "jaccard": round(len(p1 & p6) / len(p1 | p6), 3)},
        "trade_counts": {a: len(sims[a]["trades"]) for a in ARMS},
        "population_ratio_s6_over_s1": round(
            len(s6["trades"]) / len(s1["trades"]), 3),
        "cap_bind_rate_pct": {"S8": s8["cap_bind_rate_pct"],
                              "S7": s7["cap_bind_rate_pct"]},
        "cash_bind_rate_pct": {"S8": s8["cash_bind_rate_pct"],
                               "S7": s7["cash_bind_rate_pct"]},
        "unconstrained_rate_pct": {
            a: round(100.0 - (sims[a]["cap_bind_rate_pct"] or 0)
                     - (sims[a]["cash_bind_rate_pct"] or 0), 1)
            for a in ("S8", "S7")},
        "cap_bind_voids_clauses_5_6": bool(
            (s8["cap_bind_rate_pct"] or 0) > 50
            or (s7["cap_bind_rate_pct"] or 0) > 50),
        "admission_binds": {
            "S1": s1.get("denials"), "S6": s6.get("denials"),
            "note": ("S8/S7 are forced replays (declaration 4): "
                     "admission is not re-evaluated, so bind rates are "
                     "measurable only on S1/S6 and are never imputed")},
        "risk_budget_frac": risk_budget,
    }
    print(f"validity: cash-bind S8 {s8['cash_bind_rate_pct']}% / "
          f"S7 {s7['cash_bind_rate_pct']}%")
    print(f"validity: cap-bind S8 {s8['cap_bind_rate_pct']}% / "
          f"S7 {s7['cap_bind_rate_pct']}%; clauses 5-6 "
          f"{'VOID' if results['validity']['cap_bind_voids_clauses_5_6'] else 'stand'}")

    # ---- comparisons ----------------------------------------------
    if not smoke:
        rk = random_k_series(panel, ("S1", "S6"), cash_daily=cd)
        results["random_k"] = {a: band_summary(rk[a]) for a in rk}
        # THE guard test: paired by seed. Under one shuffle, does the
        # guard-off gate beat the guard-on gate — and how often?
        d_cagr = np.array([rk["S6"][i]["cagr_pct"] - rk["S1"][i]["cagr_pct"]
                           for i in range(len(rk["S1"]))])
        d_mdd = np.array([rk["S6"][i]["max_dd_pct"] - rk["S1"][i]["max_dd_pct"]
                          for i in range(len(rk["S1"]))])
        pctl = lambda a, q: round(float(np.percentile(a, q)), 3)
        results["random_k_paired_by_seed"] = {
            "seeds": len(d_cagr),
            "d_cagr_mean": round(float(d_cagr.mean()), 3),
            "d_cagr_band": [pctl(d_cagr, 2.5), pctl(d_cagr, 97.5)],
            "d_cagr_share_positive_pct": round(
                100.0 * float((d_cagr > 0).mean()), 1),
            "d_mdd_mean": round(float(d_mdd.mean()), 3),
            "d_mdd_band": [pctl(d_mdd, 2.5), pctl(d_mdd, 97.5)],
            "d_mdd_share_positive_pct": round(
                100.0 * float((d_mdd > 0).mean()), 1),
            "observed_score_order_d_cagr": None,   # filled below
            "observed_score_order_d_mdd": None,
        }
    for label, a, b, kind in (("sizing_S1_S8", "S1", "S8", "paired"),
                              ("sizing_S6_S7", "S6", "S7", "paired"),
                              ("guard_S1_S6", "S1", "S6", "unpaired"),
                              ("guard_S8_S7", "S8", "S7", "unpaired")):
        fa = results["arms"][a]["full"]
        fb = results["arms"][b]["full"]
        cmp_ = {"kind": kind,
                "d_cagr_pp": round(fb["cagr_pct"] - fa["cagr_pct"], 3),
                "d_mdd_pp": round(fb["max_dd_pct"] - fa["max_dd_pct"], 3),
                "d_expectancy_r": round(
                    (fb["expectancy_r"] or 0) - (fa["expectancy_r"] or 0),
                    4)}
        if kind == "paired":
            src = fixed1 if a == "S1" else fixed6
            # prereg post-registration amendment 1: the per-trade R
            # bootstrap is VOID (R is share-scale invariant). Computed
            # and carried VOID-LABELLED so the record shows the
            # arithmetic; clauses 5/6 read the PORTFOLIO comparison.
            cmp_["void_per_trade_r_bootstrap"] = {
                **paired_trade_boot(src["trades"], sims[b]["trades"]),
                "void_reason": ("R-multiples are share-scale invariant; "
                                "this test cannot see a sizing change")}
            cmp_.update(paired_path_boot(sims[a]["curve"],
                                         sims[b]["curve"]))
            cmp_["d_end_equity"] = round(
                results["arms"][b]["full"]["end_equity"]
                - results["arms"][a]["full"]["end_equity"], 2)
        results["comparisons"][label] = cmp_
        print(f"{label} ({kind}): dCAGR {cmp_['d_cagr_pp']:+}pp, "
              f"dMDD {cmp_['d_mdd_pp']:+}pp, dExpR "
              f"{cmp_['d_expectancy_r']:+}"
              + (f", dEnd ${cmp_.get('d_end_equity'):,}, dCAGR CI "
                 f"{cmp_.get('d_cagr_ci')}, dMDD CI "
                 f"{cmp_.get('d_mdd_ci')}" if kind == "paired" else ""))

    if "random_k_paired_by_seed" in results:
        rkp = results["random_k_paired_by_seed"]
        obs_c = results["comparisons"]["guard_S1_S6"]["d_cagr_pp"]
        obs_m = results["comparisons"]["guard_S1_S6"]["d_mdd_pp"]
        rkp["observed_score_order_d_cagr"] = obs_c
        rkp["observed_score_order_d_mdd"] = obs_m
        rkp["observed_inside_band_cagr"] = bool(
            rkp["d_cagr_band"][0] <= obs_c <= rkp["d_cagr_band"][1])
        rkp["observed_inside_band_mdd"] = bool(
            rkp["d_mdd_band"][0] <= obs_m <= rkp["d_mdd_band"][1])
        print(f"guard paired-by-seed: dCAGR mean {rkp['d_cagr_mean']:+} "
              f"band {rkp['d_cagr_band']} ({rkp['d_cagr_share_positive_pct']}%"
              f" of seeds positive); observed {obs_c:+} -> "
              f"{'INSIDE' if rkp['observed_inside_band_cagr'] else 'OUTSIDE'}"
              f" | dMDD mean {rkp['d_mdd_mean']:+} band {rkp['d_mdd_band']}"
              f"; observed {obs_m:+} -> "
              f"{'INSIDE' if rkp['observed_inside_band_mdd'] else 'OUTSIDE'}")

    # interaction (declaration 12)
    s = results["comparisons"]
    results["interaction"] = {
        "sizing_effect_guard_on_d_cagr": s["sizing_S1_S8"]["d_cagr_pp"],
        "sizing_effect_guard_off_d_cagr": s["sizing_S6_S7"]["d_cagr_pp"],
        "difference_pp": round(s["sizing_S6_S7"]["d_cagr_pp"]
                               - s["sizing_S1_S8"]["d_cagr_pp"], 3),
        "sizing_effect_guard_on_d_mdd": s["sizing_S1_S8"]["d_mdd_pp"],
        "sizing_effect_guard_off_d_mdd": s["sizing_S6_S7"]["d_mdd_pp"],
    }

    canon = json.loads(json.dumps(results))
    results["results_hash"] = hashlib.sha256(
        json.dumps(canon, sort_keys=True).encode()).hexdigest()[:16]
    out_path = out_path or os.path.join(REPO, "docs",
                                        "backtest-carry-results.json")
    if smoke:
        out_path = out_path.replace(".json", ".smoke.json")
    with open(out_path, "w") as f:
        json.dump(results, f, indent=1)
    print(f"results -> {out_path}  hash {results['results_hash']}")
    return results


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--smoke", action="store_true")
    ap.add_argument("--out")
    a = ap.parse_args()
    run(smoke=a.smoke, out_path=a.out)


if __name__ == "__main__":
    main()
