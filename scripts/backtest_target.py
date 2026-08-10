#!/usr/bin/env python3
"""
Build 6A — profit-target sweep (pre-registered:
docs/backtest-target-prereg.md, committed 6b504ba BEFORE any results;
implementation declarations: docs/backtest-target-declarations.md,
committed e223204, also before results).

Six arms over ONE fixed trade population — T0's (the 5B S1 arm, score
ordering; production gate, production stop, production sizing). Only
the exit-target rule varies. Entries are replayed as FORCED fills from
T0's realized trade log; admission gates are not re-evaluated
(declaration 2), so the paired per-trade comparison is exact by
construction.

Integrity gate, two-sided (declaration 1): the 5B engine must
reproduce the committed backtest-systems-results.json S1/score block,
and THIS engine with the target rule off must reproduce the 5B
engine's curve and trade log to the cent. Every arm statistic rests on
that parity.

Run: python3 scripts/backtest_target.py [--smoke] [--out PATH]
"""

import argparse
import hashlib
import json
import os
import subprocess
import sys
from collections import defaultdict

REPO = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
sys.path.insert(0, REPO)
sys.path.insert(0, os.path.join(REPO, "scripts"))

import numpy as np
import pandas as pd

from backtest_systems import (load_panel, run_sim, curve_metrics,
                              trade_metrics, _span, PRICES, SIZE_PCT,
                              SLIP_BPS, TRAIN_END, TRADING_DAYS)

ARMS = {
    # legs: (R-multiple level, fraction of ORIGINAL size)
    "T0": {"legs": ()},
    "T1": {"legs": ((2.0, 0.5),)},
    "T2": {"legs": ((3.0, 0.5),)},
    "T3": {"legs": ((2.0, 1.0 / 3.0), (4.0, 1.0 / 3.0))},
    "T4": {"legs": (), "floor_arm_r": 2.0},   # touch of 2R arms the floor
    "T5": {"legs": ((3.0, 0.5),), "floor_on_fill": True},
}
ARM_ORDER = ("T0", "T1", "T2", "T3", "T4", "T5")
N_BOOT = 2000
BLOCK = 21
SEED = 20260808


def add_highs(panel):
    """The 5B panel carries open/close/sma20 only; targets need the
    session HIGH (declaration 5-6). Same cache files, same parsing."""
    for t, p in panel["px"].items():
        f = os.path.join(PRICES, f"{t.replace('/', '_')}.csv")
        p["high"] = pd.read_csv(f, usecols=["High"])["High"] \
            .to_numpy(dtype=float)
        if len(p["high"]) != len(p["iso"]):
            raise AssertionError(f"{t}: high column misaligned")
    return panel


def build_schedule(t0_trades):
    """T0's realized entries, replayed as forced fills (declaration 2).
    (fill day, ticker, signal-day stop) is everything an entry needs;
    within-day order follows T0's log order."""
    sched = defaultdict(list)
    for tr in t0_trades:
        sched[tr["entry_date"]].append(
            {"ticker": tr["ticker"], "stop": tr["entry_stop"],
             "group": tr["group"], "regime": tr["regime"],
             "reentry": tr["reentry"]})
    return sched


def run_target_sim(panel, schedule, arm, *, slip_bps=SLIP_BPS,
                   size_pct=SIZE_PCT, capital=100_000.0):
    """One arm's replay over the fixed entry schedule. Day ordering
    mirrors the 5B engine exactly: pending exits at the open, entries
    at the open, intrabar target fills, cash yield, confirmed-close
    mark + stop check."""
    rule = ARMS[arm]
    legs_def = rule["legs"]
    floor_arm_r = rule.get("floor_arm_r")
    floor_on_fill = rule.get("floor_on_fill", False)

    days = panel["days"]
    px = panel["px"]
    cash_daily = panel["cash_daily"]
    slip = slip_bps / 10_000.0

    cash = capital
    positions = {}
    pending_exits = []
    trades = []
    curve = []
    cash_yield_usd = 0.0
    floor_armed_n = 0

    def close_trade(t, pos, day, fill, synthetic):
        nonlocal cash
        proceeds = pos["shares_rem"] * fill * (1.0 if synthetic
                                               else 1.0 - slip)
        cash += 0.0 if synthetic else proceeds
        pnl = pos["leg_proceeds"] + proceeds - pos["cost_basis"]
        trades.append({
            "ticker": t, "entry_date": pos["entry_date"],
            "exit_date": day, "hold_days": pos["hold"],
            "r_usd": pos["r_usd"], "pnl": pnl,
            "r_mult": pnl / pos["r_usd"],
            "entry_stop": pos["entry_stop"], "exit_fill": fill,
            "group": pos["group"], "regime": pos["regime"],
            "reentry": pos["reentry"],
            "targets_hit": pos["targets_hit"],
            "floored": pos["floored"],
            "synthetic_close": synthetic})
        return proceeds

    for day in days:
        # ---- next-open exit fills (whole remaining position;
        # resting limits die with it — declaration 8) ------------------
        for t in pending_exits:
            pos = positions.pop(t, None)
            if pos is None:
                continue
            p = px[t]
            i = p["pos"].get(day)
            fill = p["open"][i] if i is not None else pos["last_close"]
            close_trade(t, pos, day, fill, False)
        pending_exits = []

        # ---- forced entries at the open (declaration 2) --------------
        equity_prev = cash + sum(p_["shares_rem"] * p_["last_close"]
                                 for p_ in positions.values())
        for e in schedule.get(day, ()):
            t = e["ticker"]
            if t in positions:
                raise AssertionError(
                    f"{day} {t}: scheduled entry collides with an open "
                    "position — the fixed-population construction is "
                    "broken (floors may only ever RAISE stops)")
            p = px[t]
            i = p["pos"][day]            # T0 filled here; bar must exist
            fill = p["open"][i]
            stop = e["stop"]
            size_usd = size_pct / 100.0 * equity_prev
            shares = size_usd / fill
            cost = shares * fill * (1.0 + slip)
            if cost > cash + 1e-6:
                raise AssertionError(
                    f"{day} {t}: arm cash went negative — declaration 2's "
                    "no-recheck construction is unsound for this arm")
            cash -= cost
            positions[t] = {
                "shares_orig": shares, "shares_rem": shares,
                "entry_date": day, "entry_fill": fill,
                "entry_stop": stop, "r_ps": fill - stop,
                "r_usd": shares * (fill - stop),
                "cost_basis": cost, "leg_proceeds": 0.0,
                "legs_filled": [False] * len(legs_def),
                "group": e["group"], "regime": e["regime"],
                "reentry": e["reentry"],
                "targets_hit": [], "floored": False,
                "hold": 0, "last_close": fill}

        # ---- intrabar target fills (declarations 5-7, 9) -------------
        for t, pos in positions.items():
            p = px[t]
            i = p["pos"].get(day)
            if i is None:
                continue
            hi = p["high"][i]
            op = p["open"][i]
            if not np.isfinite(hi):
                continue
            for k, (r_mult, frac) in enumerate(legs_def):
                if pos["legs_filled"][k]:
                    continue
                level = pos["entry_fill"] + r_mult * pos["r_ps"]
                if hi >= level:
                    fillp = op if (np.isfinite(op) and op >= level) \
                        else level
                    qty = pos["shares_orig"] * frac
                    proceeds = qty * fillp * (1.0 - slip)
                    cash += proceeds
                    pos["leg_proceeds"] += proceeds
                    pos["shares_rem"] -= qty
                    pos["legs_filled"][k] = True
                    pos["targets_hit"].append(
                        {"r": r_mult, "date": day,
                         "fill": round(fillp, 6)})
                    if floor_on_fill and not pos["floored"]:
                        pos["floored"] = True
                        floor_armed_n += 1
            if pos["shares_rem"] <= 1e-9:
                raise AssertionError(f"{day} {t}: targets consumed the "
                                     "whole position — fractions wrong")
            if floor_arm_r is not None and not pos["floored"]:
                if hi >= pos["entry_fill"] + floor_arm_r * pos["r_ps"]:
                    pos["floored"] = True     # armed intrabar, applies
                    floor_armed_n += 1        # to TONIGHT's close check

        # ---- cash yield (freed capital earns IRX — declaration 10) ---
        y = cash * cash_daily.loc[day]
        cash += y
        cash_yield_usd += y

        # ---- confirmed close: mark + stop check (D-018; floor per
        # declaration 9: max(SMA20, entry), equality exits) ------------
        for t, pos in list(positions.items()):
            p = px[t]
            i = p["pos"].get(day)
            if i is not None and np.isfinite(p["close"][i]):
                pos["last_close"] = p["close"][i]
                pos["hold"] += 1
                if np.isfinite(p["sma20"][i]):
                    level = p["sma20"][i]
                    if pos["floored"]:
                        level = max(level, pos["entry_fill"])
                    if not p["close"][i] > level:
                        pending_exits.append(t)

        curve.append((day, cash + sum(p_["shares_rem"] * p_["last_close"]
                                      for p_ in positions.values())))

    # mark-out survivors (synthetic close — valuation, not a fill)
    last_day = days[-1]
    for t, pos in list(positions.items()):
        pos["marked"] = close_trade(t, pos, last_day,
                                    pos["last_close"], True)
    marked = sum(p_["marked"] for p_ in positions.values())

    final_equity = cash + marked
    total_pnl = sum(tr["pnl"] for tr in trades)
    drift = final_equity - (capital + total_pnl + cash_yield_usd)
    if abs(drift) > 0.01:
        raise AssertionError(f"{arm}: conservation broken by {drift:.4f}"
                             " — equity != capital + P&L + cash yield")

    return {"curve": curve, "trades": trades,
            "cash_yield_usd": round(cash_yield_usd, 2),
            "floor_armed": floor_armed_n}


# ---------------------------------------------------------------- pairing
def pair_trades(t0_trades, arm_trades):
    """1:1 alignment on (ticker, entry_date) — must be exact
    (declaration 2). Returns aligned arrays of r_mult plus T0 r_usd."""
    key = lambda tr: (tr["ticker"], tr["entry_date"])
    a0 = {key(tr): tr for tr in t0_trades}
    aa = {key(tr): tr for tr in arm_trades}
    if set(a0) != set(aa) or len(a0) != len(t0_trades):
        raise AssertionError("trade populations differ — pairing broken")
    keys = sorted(a0)
    r0 = np.array([a0[k]["r_mult"] for k in keys])
    ra = np.array([aa[k]["r_mult"] for k in keys])
    r_usd0 = np.array([a0[k]["r_usd"] for k in keys])
    entry_days = np.array([k[1] for k in keys])
    return keys, r0, ra, r_usd0, entry_days


def cluster_boot_mean(diffs, clusters, n_boot=N_BOOT, seed=SEED):
    """Entry-date-cluster bootstrap CI on the mean paired difference
    (declaration 11)."""
    uniq = np.unique(clusters)
    by = {c: diffs[clusters == c] for c in uniq}
    rng = np.random.default_rng(seed)
    means = np.empty(n_boot)
    for b in range(n_boot):
        pick = rng.choice(uniq, size=len(uniq), replace=True)
        v = np.concatenate([by[c] for c in pick])
        means[b] = v.mean()
    return (round(float(diffs.mean()), 4),
            [round(float(np.percentile(means, q)), 4)
             for q in (2.5, 97.5)])


def paired_path_boot(curve0, curve_a, n_boot=N_BOOT, block=BLOCK,
                     seed=SEED):
    """Paired moving-block bootstrap of daily returns — identical block
    indices on both arms; ΔCAGR and ΔmaxDD per replicate
    (declaration 12)."""
    v0 = np.array([x[1] for x in curve0])
    va = np.array([x[1] for x in curve_a])
    r0 = v0[1:] / v0[:-1] - 1.0
    ra = va[1:] / va[:-1] - 1.0
    n = len(r0)
    nblocks = int(np.ceil(n / block))
    rng = np.random.default_rng(seed)

    def stats(r):
        v = np.cumprod(1.0 + r)
        cagr = v[-1] ** (TRADING_DAYS / len(r)) - 1.0
        peak = np.maximum.accumulate(np.concatenate([[1.0], v]))
        dd = np.concatenate([[1.0], v]) / peak - 1.0
        return cagr * 100.0, float(dd.min()) * 100.0

    d_cagr = np.empty(n_boot)
    d_mdd = np.empty(n_boot)
    for b in range(n_boot):
        starts = rng.integers(0, n - block + 1, size=nblocks)
        idx = np.concatenate([np.arange(s, s + block)
                              for s in starts])[:n]
        c0, m0 = stats(r0[idx])
        ca, ma = stats(ra[idx])
        d_cagr[b] = ca - c0
        d_mdd[b] = ma - m0
    ci = lambda a: [round(float(np.percentile(a, q)), 3)
                    for q in (2.5, 97.5)]
    return {"d_cagr_ci": ci(d_cagr), "d_mdd_ci": ci(d_mdd)}


def concentration(keys, r0, ra, r_usd0):
    """Top-3 contributor share of the total difference, T0-scale
    dollars (declaration 14)."""
    contrib = (ra - r0) * r_usd0
    total = contrib.sum()
    order = np.argsort(-np.abs(contrib))
    top3 = order[:3]
    share = (abs(contrib[top3].sum()) / abs(total)
             if abs(total) > 1e-9 else None)
    return {
        "total_diff_usd_t0_scale": round(float(total), 2),
        "top3_share_pct": round(share * 100, 1)
        if share is not None else None,
        "top3": [{"ticker": keys[i][0], "entry_date": keys[i][1],
                  "contrib_usd": round(float(contrib[i]), 2)}
                 for i in top3],
        "fragile": bool(share is not None and share > 0.5)}


def verdict(d_cagr, d_cagr_ci, d_mdd, d_mdd_ci):
    """Mechanical read of the prereg decision table. ΔmaxDD > 0 =
    shallower drawdown = improvement (dd is negative)."""
    cagr_sig = not (d_cagr_ci[0] <= 0.0 <= d_cagr_ci[1])
    mdd_sig = not (d_mdd_ci[0] <= 0.0 <= d_mdd_ci[1])
    if cagr_sig and mdd_sig and d_cagr > 0 and d_mdd > 0:
        return "1-ADOPT"
    if cagr_sig and mdd_sig and d_cagr < 0 and d_mdd > 0:
        return "2-EXCHANGE-RATE"
    if cagr_sig and mdd_sig and d_cagr < 0 and d_mdd < 0:
        return "3-REJECT"
    if not cagr_sig and not mdd_sig:
        return "4-INSIDE-CI"
    # combinations the table does not name — reported, never coerced
    return ("UNCOVERED: " +
            f"dCAGR {'sig' if cagr_sig else 'ns'}/"
            f"{'+' if d_cagr > 0 else '-'}, "
            f"dMDD {'sig' if mdd_sig else 'ns'}/"
            f"{'+' if d_mdd > 0 else '-'}")


# ------------------------------------------------------------------ main
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


def run(smoke=False, out_path=None):
    # 6A inherits 5B's pinned panel; it also reads the committed 5B
    # results for its own gate (a) and the earnings cache via the frame
    if not smoke:
        from study_inputs import assert_pinned_inputs
        assert_pinned_inputs(["earnings_dates"], label="Build 6A")
    panel = add_highs(load_panel(smoke=smoke))
    print(f"panel: {len(panel['px'])} tickers, {len(panel['days'])} days,"
          f" frame_hash {panel['frame_hash']}")

    # ---- integrity gate, side (a): 5B engine vs committed results ----
    s1 = run_sim(panel, "S1", "score")
    if not smoke:
        with open(os.path.join(REPO, "docs",
                               "backtest-systems-results.json")) as f:
            committed = json.load(f)["arms"]["S1"]["score"]["full"]
        got = curve_metrics(s1["curve"], panel["cash_daily"])
        for k in ("cagr_pct", "max_dd_pct", "end_equity"):
            if got[k] != committed[k]:
                raise AssertionError(
                    f"5B S1 replay diverges from committed results on "
                    f"{k}: {got[k]} vs {committed[k]}")
        print(f"parity (a): 5B S1/score reproduces committed results "
              f"(end equity {got['end_equity']:,})")

    # ---- integrity gate, side (b): this engine, target off -----------
    schedule = build_schedule(s1["trades"])
    t0 = run_target_sim(panel, schedule, "T0")
    dv = max(abs(a[1] - b[1]) for a, b in zip(s1["curve"], t0["curve"]))
    if dv > 0.01:
        raise AssertionError(f"T0 engine parity broken: max curve "
                             f"divergence ${dv:.4f}")
    p0 = {(t["ticker"], t["entry_date"]): t for t in s1["trades"]}
    pt = {(t["ticker"], t["entry_date"]): t for t in t0["trades"]}
    if set(p0) != set(pt):
        raise AssertionError("T0 trade population differs from S1")
    dpnl = max(abs(p0[k]["pnl"] - pt[k]["pnl"]) for k in p0)
    if dpnl > 0.01:
        raise AssertionError(f"T0 trade P&L diverges: max ${dpnl:.4f}")
    print(f"parity (b): target engine with rule off == 5B engine "
          f"(max curve diff ${dv:.6f}, max trade diff ${dpnl:.6f}, "
          f"{len(t0['trades'])} trades)")

    results = {
        "schema": "backtest-target-1",
        "prereg": "docs/backtest-target-prereg.md",
        "declarations": "docs/backtest-target-declarations.md",
        "design": {"size_pct": SIZE_PCT, "slip_bps": SLIP_BPS,
                   "n_boot": N_BOOT, "block_days": BLOCK, "seed": SEED,
                   "arms": {a: {"legs": [list(l) for l in
                                         ARMS[a]["legs"]],
                                "floor_arm_r": ARMS[a].get("floor_arm_r"),
                                "floor_on_fill":
                                    ARMS[a].get("floor_on_fill", False)}
                            for a in ARM_ORDER}},
        "t0_parity": {"max_curve_diff_usd": round(dv, 6),
                      "max_trade_pnl_diff_usd": round(dpnl, 6),
                      "trades": len(t0["trades"])},
        "provenance": {"frame_hash": panel["frame_hash"],
                       "regime": panel["regime_prov"]},
        "arms": {},
    }

    cd = panel["cash_daily"]
    t0_metrics = span_block(t0, cd)
    t0_full = t0_metrics["full"]
    results["arms"]["T0"] = {**t0_metrics,
                             "cash_yield_usd": t0["cash_yield_usd"]}

    for arm in ARM_ORDER[1:]:
        sim = run_target_sim(panel, schedule, arm)
        keys, r0, ra, r_usd0, entry_days = pair_trades(t0["trades"],
                                                       sim["trades"])
        mean_d, ci_d = cluster_boot_mean(ra - r0, entry_days)
        pb = paired_path_boot(t0["curve"], sim["curve"])
        m = span_block(sim, cd)
        d_cagr = round(m["full"]["cagr_pct"] - t0_full["cagr_pct"], 3)
        d_mdd = round(m["full"]["max_dd_pct"] - t0_full["max_dd_pct"], 3)
        hit = defaultdict(int)
        for t in sim["trades"]:
            for h in t["targets_hit"]:
                hit[str(h["r"])] += 1
        conc = concentration(keys, r0, ra, r_usd0)
        v = verdict(d_cagr, pb["d_cagr_ci"], d_mdd, pb["d_mdd_ci"])
        exch = (round(-d_cagr / d_mdd, 3)
                if v == "2-EXCHANGE-RATE" and d_mdd != 0 else None)
        results["arms"][arm] = {
            **m,
            "cash_yield_usd": sim["cash_yield_usd"],
            "targets_reached": dict(hit),
            "floor_armed": sim["floor_armed"],
            "vs_t0": {"d_cagr_pp": d_cagr, "d_mdd_pp": d_mdd,
                      "d_cagr_ci": pb["d_cagr_ci"],
                      "d_mdd_ci": pb["d_mdd_ci"],
                      "paired_trade_mean_diff_r": mean_d,
                      "paired_trade_ci_r": ci_d,
                      "n_paired": len(keys)},
            "concentration": conc,
            "verdict": v,
            "cagr_per_dd_point": exch,
        }
        print(f"{arm}: CAGR {m['full']['cagr_pct']}% "
              f"(d {d_cagr:+}pp CI {pb['d_cagr_ci']}), "
              f"maxDD {m['full']['max_dd_pct']}% (d {d_mdd:+}pp CI "
              f"{pb['d_mdd_ci']}), paired dR {mean_d} CI {ci_d}, "
              f"hits {dict(hit)}, floored {sim['floor_armed']}, "
              f"top3 {conc['top3_share_pct']}% "
              f"{'FRAGILE ' if conc['fragile'] else ''}-> {v}")

    canon = json.loads(json.dumps(results))
    results["results_hash"] = hashlib.sha256(
        json.dumps(canon, sort_keys=True).encode()).hexdigest()[:16]
    out_path = out_path or os.path.join(
        REPO, "docs", "backtest-target-results.json")
    if smoke and out_path == os.path.join(
            REPO, "docs", "backtest-target-results.json"):
        out_path = os.path.join(REPO, "docs",
                                "backtest-target-results.smoke.json")
    with open(out_path, "w") as f:
        json.dump(results, f, indent=1)
    print(f"results -> {out_path}  hash {results['results_hash']}")
    return results


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--smoke", action="store_true")
    ap.add_argument("--out")
    args = ap.parse_args()
    run(smoke=args.smoke, out_path=args.out)


if __name__ == "__main__":
    main()
