#!/usr/bin/env python3
"""
Build 9 — sizing and ceiling sweep (pre-registered:
docs/backtest-sizing-prereg.md, sha256 af870141594b5a03…, committed
2aa55a4 [pre-push breadcrumb] BEFORE this script existed).

Nine (size%, ceiling-ladder) configurations through the UNMODIFIED 5B
harness — bs.run_sim already implements every clause the prereg
holds fixed: score-order selection, R28 group caps, next-open fills,
5bps/side, redeployment, the ceiling gating NEW ENTRIES ONLY (hold
through a regime downgrade, block new entries — production
behaviour), and the HARD ZERO CASH FLOOR (an unaffordable entry is
DENIED and counted; borrowing is structurally impossible). The size
is run_sim's size_pct parameter; the ladder is the module CEILINGS
dict, patched per arm and restored.

GATES, in order, computed before any statistic:
  gate 1       P1 (6.5 / 90-50-25-5) reproduces the committed 5B
               S1/score block — cent-exact, or STOP.
  feasibility  min(cash) >= 0 for EVERY arm across the entire window,
               asserted from the equity curve and exposure series
               BEFORE any statistic is computed. Breach = abort and
               report. (Build 8's lesson, as a hard clause.)

Statistics: no pairing (entries diverge by design). Random-K bands as
the null, 50 seeds per arm, exactly as 5B; shared-block-index path
bootstrap vs P1 (Build 7's paired_path_boot — common market-time
resampling of two diverged curves); trade overlap vs P1. Primary is
the (CAGR, MaxDD) frontier, never CAGR alone.

Run: python3 scripts/backtest_sizing.py [--smoke] [--out PATH]
                                        [--seeds N] [--workers N]
"""

import argparse
import hashlib
import json
import os
import subprocess
import sys
from concurrent.futures import ProcessPoolExecutor
from contextlib import contextmanager

REPO = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
sys.path.insert(0, REPO)
sys.path.insert(0, os.path.join(REPO, "scripts"))

import numpy as np

import backtest_systems as bs
from backtest_systems import (TRAIN_END, curve_metrics, trade_metrics,
                              _span)
from backtest_carry import paired_path_boot, band_summary, SEED

PREREG_PATH = "docs/backtest-sizing-prereg.md"
PREREG_SHA256 = ("af870141594b5a0380ddaf075f69c1fc"
                 "d3c455b5483ffc98d0e38b8c1c1baf8c")
SLIP = bs.SLIP_BPS / 10_000.0
STATE_KEYS = ("In-Trend-Full", "In-Trend-Throttled",
              "Out-Defensive", "Out-Risk-off")
RANDOM_SEEDS = 50

# arm -> (size_pct, ladder Trending/Choppy/Caution/Risk-off)
ARMS = {
    "P1": (6.5,  (90, 50, 25, 5)),     # incumbent — integrity gate
    "P2": (5.0,  (90, 50, 25, 5)),
    "P3": (8.0,  (90, 50, 25, 5)),
    "P4": (10.0, (90, 50, 25, 5)),
    "P5": (6.5,  (100, 60, 30, 10)),
    "P6": (6.5,  (70, 40, 20, 5)),
    "P7": (6.5,  (90, 90, 90, 90)),    # flat — bar 5, ruled separately
    "P8": (10.0, (100, 60, 30, 10)),
    "P9": (5.0,  (100, 60, 30, 10)),
}
ARM_ORDER = tuple(ARMS)
assert all(max(l) <= 100 for _, l in ARMS.values()), \
    "no ceiling may exceed 100% — this study tests allocation, never leverage"


@contextmanager
def ladder(levels):
    """Patch bs.CEILINGS in place (run_sim reads the module dict) and
    restore unconditionally."""
    old = dict(bs.CEILINGS)
    bs.CEILINGS.clear()
    bs.CEILINGS.update(dict(zip(STATE_KEYS, (float(x) for x in levels))))
    try:
        yield
    finally:
        bs.CEILINGS.clear()
        bs.CEILINGS.update(old)


def arm_sim(panel, arm, rule="score", seed=None):
    size, levels = ARMS[arm]
    with ladder(levels):
        return bs.run_sim(panel, "S1", rule, seed=seed, size_pct=size)


# --------------------------------------------------------- derived series
def cash_series(sim):
    """cash(day) = equity × (1 − exposure_frac) — derived, not
    instrumented, so the committed harness stays untouched."""
    eq = {d: v for d, v in sim["curve"]}
    return [(d, eq[d] * (1.0 - f)) for d, f, _n in sim["daily_meta"]]


def occupancy(sim, arm):
    size, levels = ARMS[arm]
    lad = dict(zip(STATE_KEYS, levels))
    n = np.array([x[2] for x in sim["daily_meta"]], dtype=float)
    f = np.array([x[1] for x in sim["daily_meta"]])
    return n, f, lad


def shares_from_trade(t):
    """Build 8's algebraic recovery, verbatim mechanics."""
    stop = t["entry_stop"]
    xslip = 0.0 if t["synthetic_close"] else SLIP
    den = t["exit_fill"] * (1.0 - xslip) - stop * (1.0 + SLIP)
    return (t["pnl"] + t["r_usd"] * (1.0 + SLIP)) / den


def max_single_name_pct(sim, panel):
    """Max over days and positions of shares×close / that day's equity
    — the prereg's 'actually reached', not the nominal size."""
    eq = {d: v for d, v in sim["curve"]}
    best = 0.0
    for t in sim["trades"]:
        p = panel["px"][t["ticker"]]
        sh = shares_from_trade(t)
        i0 = p["pos"].get(t["entry_date"])
        i1 = p["pos"].get(t["exit_date"], len(p["iso"]))
        if i0 is None:
            continue
        hi = i1 + 1 if t["synthetic_close"] else i1
        for j in range(i0, min(hi, len(p["iso"]))):
            d = p["iso"][j]
            e = eq.get(d)
            c = p["close"][j]
            if e and np.isfinite(c) and e > 0:
                v = sh * c / e * 100.0
                if v > best:
                    best = v
    return round(best, 2)


# --------------------------------------------------------------- reporting
def span_block(sim, cd):
    out = {}
    for span, lo, hi in (("full", None, None), ("train", None, TRAIN_END),
                         ("validate", TRAIN_END, None)):
        seg = _span(sim["curve"], lo, hi)
        tr = [t for t in sim["trades"]
              if (lo is None or t["entry_date"] > lo)
              and (hi is None or t["entry_date"] <= hi)]
        out[span] = {**curve_metrics(seg, cd), **trade_metrics(tr)}
    return out


def validity_block(sim, arm, panel):
    n, f, lad = occupancy(sim, arm)
    days = [x[0] for x in sim["daily_meta"]]
    states = panel["states"]
    ceil_frac = np.array([lad[states[d]] / 100.0 for d in days])
    size_frac = ARMS[arm][0] / 100.0
    cs = cash_series(sim)
    min_cash = min(v for _, v in cs)
    den = sim["denials"]
    return {
        "min_cash": round(min_cash, 2),
        "trades": len(sim["trades"]),
        "distinct_tickers": len({t["ticker"] for t in sim["trades"]}),
        "mean_concurrent_positions": round(float(n.mean()), 2),
        "mean_exposure_over_equity": round(float(f.mean()), 3),
        "max_single_name_pct_of_capital": max_single_name_pct(sim, panel),
        "denials_ceiling": den.get("ceiling", 0),
        "denials_group_pct": den.get("group_pct", 0),
        "denials_group_count": den.get("group_count", 0),
        "denials_cash": den.get("cash", 0),
        "denials_no_bar": den.get("no_bar", 0),
        "denials_gap_below_stop": den.get("gap_below_stop", 0),
        "denials_already_held": den.get("already_held", 0),
        "days_at_ceiling": int(((f + size_frac) > ceil_frac).sum()),
        "days_over_ceiling": int((f > ceil_frac + 1e-9).sum()),
        # close-of-day approximations of the harness's entry-time
        # admission test (definitional note, review): the harness
        # denies on pre-mark equity and intraday exposure; these count
        # end-of-day states. The exact binding counts are the denial
        # fields above.
        "days_at_over_ceiling_basis": "close-of-day approximation",
        # NOT the committed 5B field of the same name (that one is
        # exposure-weighted); renamed to say what it is
        "pct_days_any_position": round(100.0 * float((n > 0).mean()), 1),
    }


def overlap_vs_p1(p1_sim, sim):
    key = lambda t: (t["ticker"], t["entry_date"])
    a = {key(t) for t in p1_sim["trades"]}
    b = {key(t) for t in sim["trades"]}
    inter = len(a & b)
    return {"n_p1": len(a), "n_arm": len(b), "shared": inter,
            "share_of_p1_pct": round(100.0 * inter / len(a), 1),
            "jaccard_pct": round(100.0 * inter / len(a | b), 1)}


def fragility(p1_sim, sim):
    """Clause 6 on the ADVANTAGE DECOMPOSITION (review fix, applied
    before results were read — the first cut compared the arm's top-3
    GROSS trades to the NET advantage, which saturates FRAGILE for
    every positive-advantage arm). Per (ticker, entry_date) key the
    advantage decomposes exactly: shared trades contribute
    pnl_arm − pnl_P1, arm-only trades +pnl, P1-only trades −pnl; the
    contributions sum to the advantage to the cent. FRAGILE = the top
    3 contributions exceed half of a positive advantage."""
    key = lambda t: (t["ticker"], t["entry_date"])
    A = {key(t): t["pnl"] for t in p1_sim["trades"]}
    B = {key(t): t["pnl"] for t in sim["trades"]}
    contrib = []
    for k, v in B.items():
        contrib.append(v - A[k] if k in A else v)
    for k, v in A.items():
        if k not in B:
            contrib.append(-v)
    adv = float(sum(contrib))
    if adv <= 0:
        return {"advantage_usd": round(adv, 2), "fragile": False,
                "note": "no positive advantage to concentrate"}
    top3 = float(sum(sorted(contrib, reverse=True)[:3]))
    return {"advantage_usd": round(adv, 2),
            "top3_contrib_usd": round(top3, 2),
            "top3_share_of_advantage_pct":
                round(100.0 * top3 / adv, 1),
            "fragile": bool(top3 > 0.5 * adv)}


def path_ci_vs_p1(p1_sim, sim):
    out = {}
    for span, lo, hi in (("full", None, None), ("train", None, TRAIN_END),
                         ("validate", TRAIN_END, None)):
        pb = paired_path_boot(_span(p1_sim["curve"], lo, hi),
                              _span(sim["curve"], lo, hi))
        out[span] = pb
    return out


# ------------------------------------------------------ random-band pool
_PANEL = None


def _init_worker(smoke):
    global _PANEL
    _PANEL = bs.load_panel(smoke=smoke)


def _band_task(job):
    arm, seed = job
    sim = arm_sim(_PANEL, arm, rule="random", seed=seed)
    m = curve_metrics(sim["curve"], _PANEL["cash_daily"])
    return arm, {"seed": seed, "cagr_pct": m["cagr_pct"],
                 "max_dd_pct": m["max_dd_pct"],
                 "trades": len(sim["trades"])}


# ------------------------------------------------------------------ main
def run(smoke=False, out_path=None, seeds=RANDOM_SEEDS, workers=None):
    # the prereg hash is ASSERTED at runtime, not merely recorded — a
    # post-hoc edit to the decision table aborts the study (review fix)
    with open(os.path.join(REPO, PREREG_PATH), "rb") as f:
        h = hashlib.sha256(f.read()).hexdigest()
    assert h == PREREG_SHA256, \
        f"prereg content changed after ruling: {h[:16]} != " \
        f"{PREREG_SHA256[:16]}"
    panel = bs.load_panel(smoke=smoke)
    cd = panel["cash_daily"]
    print(f"panel: {len(panel['px'])} tickers, {len(panel['days'])} days, "
          f"frame_hash {panel['frame_hash']}")

    # ---- gate 1 -------------------------------------------------------
    sims = {"P1": arm_sim(panel, "P1")}
    m1 = curve_metrics(sims["P1"]["curve"], cd)
    if not smoke:
        with open(os.path.join(REPO, "docs",
                               "backtest-systems-results.json")) as f:
            committed = json.load(f)["arms"]["S1"]["score"]["full"]
        for kk in ("cagr_pct", "max_dd_pct", "end_equity", "trades"):
            got = len(sims["P1"]["trades"]) if kk == "trades" \
                else m1.get(kk)
            if got != committed[kk]:
                raise AssertionError(
                    f"INTEGRITY GATE FAILED — P1 diverges from the "
                    f"committed 5B block on {kk}: {got} vs "
                    f"{committed[kk]}")
        print(f"gate 1: P1 reproduces the committed 5B S1/score block "
              f"({m1['end_equity']:,} end equity, "
              f"{len(sims['P1']['trades'])} trades)")

    # ---- all arms, then the FEASIBILITY GATE before any statistic ----
    for arm in ARM_ORDER:
        if arm not in sims:
            sims[arm] = arm_sim(panel, arm)
    breaches = {}
    for arm in ARM_ORDER:
        cs = [v for _, v in cash_series(sims[arm])]
        assert all(np.isfinite(v) for v in cs), \
            f"{arm}: non-finite cash in the derived series"
        mc = min(cs)
        # NaN-proof form: a NaN fails `mc >= -1e-6` (review fix)
        if not (mc >= -1e-6):
            breaches[arm] = round(mc, 2)
    if breaches:
        raise AssertionError(
            f"FEASIBILITY GATE FAILED — negative cash: {breaches}. "
            "Per prereg: abort and report; no results are produced.")
    print(f"feasibility gate: min cash >= 0 for all {len(ARM_ORDER)} "
          "arms — no borrowing anywhere")

    results = {
        "schema": "backtest-sizing-1",
        "prereg": f"{PREREG_PATH} sha256:{PREREG_SHA256[:16]} "
                  "(committed 2aa55a4, pre-push breadcrumb)",
        "design": {"arms": {a: {"size_pct": s, "ladder": list(l)}
                            for a, (s, l) in ARMS.items()},
                   "harness": "bs.run_sim unmodified; CEILINGS patched "
                              "per arm; hard zero cash floor is the "
                              "harness's own denial logic",
                   "scorer": "v1 (as-lived master frame)"},
        "arms": {}, "validity": {}, "vs_P1": {}, "decision": {},
    }
    for arm in ARM_ORDER:
        results["arms"][arm] = span_block(sims[arm], cd)
        results["validity"][arm] = validity_block(sims[arm], arm, panel)
    for arm in ARM_ORDER:
        if arm == "P1":
            continue
        results["vs_P1"][arm] = {
            "path_ci": path_ci_vs_p1(sims["P1"], sims[arm]),
            "overlap": overlap_vs_p1(sims["P1"], sims[arm]),
            "fragility": fragility(sims["P1"], sims[arm]),
        }

    # ---- decision scaffolding (mechanical; prose rules in the report) -
    for arm in ARM_ORDER:
        if arm == "P1":
            continue
        ci = results["vs_P1"][arm]["path_ci"]
        wins = lambda f: all(ci[s][f][0] > 0
                             for s in ("train", "validate"))
        loses = lambda f: all(ci[s][f][1] < 0
                              for s in ("train", "validate"))
        dc = round(results["arms"][arm]["full"]["cagr_pct"]
                   - results["arms"]["P1"]["full"]["cagr_pct"], 3)
        dm = round(results["arms"][arm]["full"]["max_dd_pct"]
                   - results["arms"]["P1"]["full"]["max_dd_pct"], 3)
        results["decision"][arm] = {
            "d_cagr_full": dc, "d_mdd_full": dm,
            "dominates_P1": bool(wins("d_cagr_ci")
                                 and wins("d_mdd_ci")),
            "dominated_by_P1": bool(loses("d_cagr_ci")
                                    and loses("d_mdd_ci")),
            "exchange_rate_cagr_per_dd_point":
                round(abs(dc) / abs(dm), 3) if dm not in (0, None)
                else None,
            "inside_ci_both_metrics": bool(
                not wins("d_cagr_ci") and not loses("d_cagr_ci")
                and not wins("d_mdd_ci") and not loses("d_mdd_ci")),
            "fragile": results["vs_P1"][arm]["fragility"]["fragile"],
        }
    p7 = results["vs_P1"]["P7"]["path_ci"]
    results["decision"]["ladder_bar5"] = {
        "P7_drawdown_worse_ci_both_spans": bool(
            all(p7[s]["d_mdd_ci"][1] < 0
                for s in ("train", "validate"))),
        "P7_d_mdd_ci": {s: p7[s]["d_mdd_ci"]
                        for s in ("full", "train", "validate")},
        "P7_d_cagr_ci": {s: p7[s]["d_cagr_ci"]
                         for s in ("full", "train", "validate")},
    }

    # ---- random-K bands, 50 seeds per arm, in a process pool ---------
    print(f"random-K bands: {len(ARM_ORDER)} arms x {seeds} seeds…",
          flush=True)
    jobs = [(a, sd) for a in ARM_ORDER for sd in range(seeds)]
    bands = {a: [] for a in ARM_ORDER}
    nw = workers or max(2, (os.cpu_count() or 4) - 2)
    with ProcessPoolExecutor(max_workers=nw, initializer=_init_worker,
                             initargs=(smoke,)) as ex:
        for arm, row in ex.map(_band_task, jobs, chunksize=4):
            bands[arm].append(row)
            done = sum(len(v) for v in bands.values())
            if done % 50 == 0:
                print(f"  [{done}/{len(jobs)}]", flush=True)
    results["random_bands"] = {}
    for arm in ARM_ORDER:
        b = band_summary(bands[arm])
        b["arm_cagr_pct"] = results["arms"][arm]["full"]["cagr_pct"]
        b["arm_above_band"] = bool(
            b["arm_cagr_pct"] > b["cagr_band"][1])
        results["random_bands"][arm] = b

    head = subprocess.run(["git", "rev-parse", "--short", "HEAD"],
                          capture_output=True, text=True,
                          cwd=REPO).stdout.strip()
    porcelain = subprocess.run(["git", "status", "--porcelain"],
                               capture_output=True, text=True,
                               cwd=REPO).stdout
    results["provenance"] = {
        "generated_from_commit": head,
        "worktree_dirty_paths": len(porcelain.splitlines()),
        "frame_hash": panel["frame_hash"],
        "regime_series": panel["regime_prov"]["series_range"],
        "random_seeds": seeds, "seed": SEED, "smoke": bool(smoke),
    }
    canonical = json.loads(json.dumps(results))
    results["results_hash"] = hashlib.sha256(
        json.dumps(canonical, sort_keys=True).encode()).hexdigest()[:16]

    if out_path is None:
        out_path = os.path.join(REPO, "docs",
                                "backtest-sizing-results.json")
        if smoke:
            out_path = out_path.replace(".json", ".smoke.json")
    with open(out_path, "w") as f:
        json.dump(results, f, indent=1)
    print(f"results -> {out_path} (hash {results['results_hash']})")
    return results, sims


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--smoke", action="store_true")
    ap.add_argument("--out")
    ap.add_argument("--seeds", type=int, default=RANDOM_SEEDS)
    ap.add_argument("--workers", type=int)
    a = ap.parse_args()
    run(smoke=a.smoke, out_path=a.out, seeds=a.seeds, workers=a.workers)


if __name__ == "__main__":
    main()
