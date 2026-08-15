#!/usr/bin/env python3
"""
Build 8 — exit sweep (pre-registered: docs/backtest-exit-prereg.md,
sha256 9b392b5f7e34c5d2…, committed 2e96862 [pre-push breadcrumb]
BEFORE this script existed).

Eight close-basis exit arms replayed over the IDENTICAL entry list —
5B's S1/score entries, extracted from the incumbent replay itself —
with capital freed by earlier exits held in cash, never redeployed.
The design measures THE EXIT RULE and is blind to Build 7's capacity
effect, by construction and on purpose.

INTEGRITY GATES, in order, computed before any statistic:
  gate 1  bs.run_sim(panel, "S1", "score") reproduces the committed 5B
          S1/score block (cagr, mdd, end equity, trade count) — the
          prereg's "cent-exact, or STOP".
  gate 2  THIS study's replay engine, run with the E1 rule over the
          extracted entry ledger, reproduces gate 1's equity curve to
          <$0.01 on every day and the identical trade set. The ledger
          and the engine are both proven against the harness before
          any non-incumbent arm is trusted.

Reuses the 5B harness (backtest_systems) and Build 7's statistics
(backtest_carry.paired_trade_boot / paired_path_boot) — imported,
never reimplemented. Scorer v1 by construction: the entry gate reads
the committed master frame's as-lived grades.

Run: python3 scripts/backtest_exit.py [--smoke] [--out PATH]
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

import backtest_systems as bs
from backtest_systems import (PRICES, TRAIN_END, curve_metrics,
                              trade_metrics, _span)
from backtest_carry import (paired_trade_boot, paired_path_boot,
                            N_BOOT, SEED)

PREREG_PATH = "docs/backtest-exit-prereg.md"
PREREG_SHA256 = ("9b392b5f7e34c5d27502119cf021d380"
                 "1c49882a9a05c7236adb668713f65a68")
AMEND_PATH = "docs/backtest-exit-prereg-amendment.md"
AMEND_SHA256 = ("d7a52c84cd5fef6bbd9adcb7bca51e4c"
                "6682ef0d1eb746bf3dfd76c9d5d58f7d")
SLIP = bs.SLIP_BPS / 10_000.0
CAPITAL = 100_000.0

# arm -> (base, atr_mult_or_None, ratchet)
ARMS = {
    "E1": ("sma20", None, False),
    "E2": ("sma5",  None, False),
    "E3": ("sma10", None, False),
    "E4": ("sma50", None, False),
    "E5": ("sma20", None, True),
    "E6": ("atr",   2.5,  False),
    "E7": ("atr",   4.0,  False),
    "E8": ("atr",   2.5,  True),
}
ARM_ORDER = tuple(ARMS)


# ----------------------------------------------------------- price extras
def extend_px(px, tickers):
    """sma5/10/50 + atr14 for the ledger's tickers, from the SAME price
    CSVs with the SAME constructions load_panel uses (rolling means on
    Close; TR rolling-14 mean). sma20 already lives in px."""
    for t in sorted(tickers):
        p = px.get(t)
        if p is None or "sma5" in p:
            continue
        path = os.path.join(PRICES, f"{t.replace('/', '_')}.csv")
        df = pd.read_csv(path, index_col=0, parse_dates=True)
        close = df["Close"].to_numpy(dtype=float)
        high = df["High"].to_numpy(dtype=float)
        low = df["Low"].to_numpy(dtype=float)
        assert len(close) == len(p["close"]), \
            f"{t}: price file changed length under the study"
        for n in (5, 10, 50):
            p[f"sma{n}"] = pd.Series(close).rolling(n).mean().to_numpy()
        prev = np.concatenate([[np.nan], close[:-1]])
        tr = np.maximum(high - low, np.maximum(np.abs(high - prev),
                                               np.abs(low - prev)))
        p["atr14"] = pd.Series(tr).rolling(14).mean().to_numpy()


# ------------------------------------------------------------- the ledger
def extract_ledger(s1_trades):
    """Recover each S1 entry's shares / fill / cost from the trade
    record algebraically:
        pnl      = proceeds − cost
        proceeds = shares · exit_fill · (1 − slip)   [slip=0 synthetic]
        cost     = shares · entry_fill · (1 + slip)
        r_usd    = shares · (entry_fill − entry_stop)
    =>  shares = (pnl + r_usd·(1+slip)) /
                 (exit_fill·(1∓slip) − entry_stop·(1+slip))
    Gate 2 then proves the reconstruction by replaying E1 to <$0.01."""
    ledger = []
    seen = set()
    for t in s1_trades:
        stop = t["entry_stop"]
        xslip = 0.0 if t["synthetic_close"] else SLIP
        den = t["exit_fill"] * (1.0 - xslip) - stop * (1.0 + SLIP)
        assert abs(den) > 1e-6, (t["ticker"], t["entry_date"], den)
        shares = (t["pnl"] + t["r_usd"] * (1.0 + SLIP)) / den
        assert shares > 0, (t["ticker"], t["entry_date"], shares)
        fill = t["r_usd"] / shares + stop
        key = (t["ticker"], t["entry_date"])
        assert key not in seen, f"duplicate entry {key}"
        seen.add(key)
        ledger.append({
            "ticker": t["ticker"], "entry_date": t["entry_date"],
            "shares": shares, "entry_fill": fill,
            "cost_basis": shares * fill * (1.0 + SLIP),
            "entry_stop": stop, "r_usd": t["r_usd"],
            "group": t["group"], "regime": t["regime"],
            "reentry": t["reentry"]})
    ledger.sort(key=lambda e: (e["entry_date"], e["ticker"]))
    return ledger


# ------------------------------------------------------------- the engine
def replay(ledger, panel, arm, extra_exit_bps=0.0):
    """One arm's replay over the fixed entry ledger. No admission logic
    — every ledger entry is taken at its recorded shares and cost. Cash
    can go NEGATIVE in slow arms (a later exit means the same entries
    overlap more); it then accrues negative yield at the same IRX rate.
    Reported, never hidden.

    Day order mirrors bs.run_sim exactly: fill pending exits at the
    open, enter today's ledger entries, accrue cash yield, then the
    confirmed close — mark, compute the arm's level, queue the exit if
    close <= level (the incumbent's `not close > sma20` boundary)."""
    base, k, ratchet = ARMS[arm]
    days = panel["days"]
    px = panel["px"]
    cash_daily = panel["cash_daily"]
    xslip = SLIP + extra_exit_bps / 10_000.0

    by_day = defaultdict(list)
    for e in ledger:
        by_day[e["entry_date"]].append(e)

    cash = CAPITAL
    positions = {}             # (ticker, entry_date) -> lot: a slow arm
    # can still hold a ticker when the ledger re-enters it, and the
    # prereg's pairing requires EVERY ledger entry to exist in every
    # arm — so lots are the unit, and concurrent duplicate lots of one
    # ticker are counted and reported rather than merged or dropped
    pending_exits = []
    trades = []
    curve = []
    days_in_market = 0
    min_cash = CAPITAL
    neg_cash_days = 0
    ratchet_bound = 0          # positions where the ratchet ever lifted
    dup_lot_opens = 0
    ge_sum, ge_max = 0.0, 0.0  # gross/equity leverage (amendment 1)
    for day in days:
        # ---- exits queued at yesterday's close ------------------------
        for lot in pending_exits:
            pos = positions.pop(lot, None)
            if pos is None:
                continue
            t = lot[0]
            p = px[t]
            i = p["pos"].get(day)
            fill = p["open"][i] if i is not None else pos["last_close"]
            proceeds = pos["shares"] * fill * (1.0 - xslip)
            cash += proceeds
            pnl = proceeds - pos["cost_basis"]
            trades.append({
                "ticker": t, "entry_date": pos["entry_date"],
                "exit_date": day, "hold_days": pos["hold"],
                "r_usd": pos["r_usd"], "pnl": pnl,
                "r_mult": pnl / pos["r_usd"] if pos["r_usd"] > 0 else None,
                "group": pos["group"], "regime": pos["regime"],
                "entry_stop": pos["entry_stop"], "exit_fill": fill,
                "exit_trigger": pos.get("exit_trigger"),
                "reentry": pos["reentry"],
                "ratchet_bound": pos["bound"],
                "synthetic_close": False})
        pending_exits = []

        # ---- today's ledger entries ----------------------------------
        for e in by_day.get(day, ()):  # ledger fills, unconditionally
            cash -= e["cost_basis"]
            if any(k[0] == e["ticker"] for k in positions):
                dup_lot_opens += 1
            positions[(e["ticker"], day)] = {
                "shares": e["shares"], "entry_date": day,
                "cost_basis": e["cost_basis"], "r_usd": e["r_usd"],
                "group": e["group"], "regime": e["regime"],
                "entry_stop": e["entry_stop"],
                "reentry": e["reentry"],
                "hold": 0, "last_close": e["entry_fill"],
                "level": e["entry_stop"],       # ratchet seed
                "hc": -np.inf, "bound": False}

        # ---- cash yield (negative cash accrues negative yield) --------
        cash += cash * cash_daily.loc[day]
        min_cash = min(min_cash, cash)
        if cash < 0:
            neg_cash_days += 1

        # ---- confirmed close ------------------------------------------
        for lot, pos in list(positions.items()):
            t = lot[0]
            p = px[t]
            i = p["pos"].get(day)
            if i is None or not np.isfinite(p["close"][i]):
                continue
            c = p["close"][i]
            pos["last_close"] = c
            pos["hold"] += 1
            pos["hc"] = max(pos["hc"], c)
            if base == "atr":
                a = p["atr14"][i]
                lvl = (pos["hc"] - k * a) if np.isfinite(a) else None
            else:
                m = p[base][i]
                lvl = float(m) if np.isfinite(m) else None
            if ratchet:
                if lvl is not None and lvl > pos["level"]:
                    pos["level"] = lvl
                elif lvl is not None and lvl < pos["level"]:
                    # the ratchet is HOLDING the stop above the raw rule
                    if not pos["bound"]:
                        pos["bound"] = True
                        ratchet_bound += 1
                check = pos["level"]
            else:
                check = lvl
            if check is not None and not c > check:
                pos["exit_trigger"] = check
                pending_exits.append(lot)

        gross = sum(p_["shares"] * p_["last_close"]
                    for p_ in positions.values())
        equity = cash + gross
        curve.append((day, equity))
        ge_sum += gross / equity if equity > 0 else 0.0
        ge_max = max(ge_max, gross / equity if equity > 0 else 0.0)
        if positions:
            days_in_market += 1

    last_day = days[-1]
    for (t, _ed), pos in positions.items():
        proceeds = pos["shares"] * pos["last_close"]   # mark, no slip
        pnl = proceeds - pos["cost_basis"]
        trades.append({
            "ticker": t, "entry_date": pos["entry_date"],
            "exit_date": last_day, "hold_days": pos["hold"],
            "r_usd": pos["r_usd"], "pnl": pnl,
            "r_mult": pnl / pos["r_usd"] if pos["r_usd"] > 0 else None,
            "group": pos["group"], "regime": pos["regime"],
            "entry_stop": pos["entry_stop"],
            "exit_fill": pos["last_close"],
            "reentry": pos["reentry"],
            "ratchet_bound": pos["bound"], "synthetic_close": True})

    return {"curve": curve, "trades": trades,
            "time_in_market_pct": round(100.0 * days_in_market
                                        / len(days), 1),
            "min_cash": round(min_cash, 2),
            "neg_cash_days": neg_cash_days,
            "dup_lot_opens": dup_lot_opens,
            "mean_gross_over_equity": round(ge_sum / len(days), 3),
            "max_gross_over_equity": round(ge_max, 3),
            "ratchet_bound_positions": ratchet_bound}


# --------------------------------------------------------------- helpers
def span_trades(trades, lo=None, hi=None):
    return [t for t in trades
            if (lo is None or t["entry_date"] > lo)
            and (hi is None or t["entry_date"] <= hi)]


def arm_block(sim, cash_daily):
    out = {}
    for span, lo, hi in (("full", None, None), ("train", None, TRAIN_END),
                         ("validate", TRAIN_END, None)):
        seg = _span(sim["curve"], lo, hi)
        tr = span_trades(sim["trades"], lo, hi)
        out[span] = {**curve_metrics(seg, cash_daily),
                     **trade_metrics(tr)}
        out[span]["exits_taken"] = sum(1 for t in tr
                                       if not t["synthetic_close"])
        out[span]["avg_bars_held"] = round(
            float(np.mean([t["hold_days"] for t in tr])), 1) if tr else None
    out["min_equity"] = round(min(v for _, v in sim["curve"]), 2)
    out["mean_gross_over_equity"] = sim["mean_gross_over_equity"]
    out["max_gross_over_equity"] = sim["max_gross_over_equity"]
    out["time_in_market_pct"] = sim["time_in_market_pct"]
    out["min_cash"] = sim["min_cash"]
    out["neg_cash_days"] = sim["neg_cash_days"]
    out["dup_lot_opens"] = sim["dup_lot_opens"]
    return out


def paired_block(e1_sim, sim, cash_daily):
    out = {}
    for span, lo, hi in (("full", None, None), ("train", None, TRAIN_END),
                         ("validate", TRAIN_END, None)):
        a = span_trades(e1_sim["trades"], lo, hi)
        b = span_trades(sim["trades"], lo, hi)
        blk = {"expectancy": paired_dollar_expectancy_boot(a, b),
               "trade_r_unweighted": paired_trade_boot(a, b),
               "trade_r_excl_tiny": {
                   **paired_trade_boot(excl_tiny(a), excl_tiny(b)),
                   "threshold_r_usd": TINY_R_USD}}
        c0 = _span(e1_sim["curve"], lo, hi)
        c1 = _span(sim["curve"], lo, hi)
        pb = paired_path_boot(c0, c1)
        # point estimates beside the CIs
        m0 = curve_metrics(c0, cash_daily)
        m1 = curve_metrics(c1, cash_daily)
        blk["d_cagr_pct"] = round(m1["cagr_pct"] - m0["cagr_pct"], 3)
        blk["d_mdd_pct"] = round(m1["max_dd_pct"] - m0["max_dd_pct"], 3)
        blk["d_cagr_ci"] = pb["d_cagr_ci"]
        blk["d_mdd_ci"] = pb["d_mdd_ci"]
        out[span] = blk
    return out


def paired_dollar_expectancy_boot(a_trades, b_trades, n_boot=N_BOOT,
                                  seed=SEED):
    """AMENDMENT 1 PRIMARY: paired per-trade DOLLAR expectancy
    difference — sum(Δpnl)/sum(r_usd) — leverage-invariant because
    shares (and therefore per-trade dollar risk) are identical across
    arms. Entry-date-cluster bootstrap, same clustering and seed as
    paired_trade_boot."""
    key = lambda t: (t["ticker"], t["entry_date"])
    A = {key(t): t for t in a_trades}
    B = {key(t): t for t in b_trades}
    keys = sorted(set(A) & set(B))
    if len(keys) != len(A) or len(keys) != len(B):
        raise AssertionError("paired populations differ — pairing broken")
    for kk in keys:
        assert abs(A[kk]["r_usd"] - B[kk]["r_usd"]) < 1e-9,             f"r_usd differs across arms for {kk} — pairing broken"
    d = np.array([B[kk]["pnl"] - A[kk]["pnl"] for kk in keys])
    w = np.array([A[kk]["r_usd"] for kk in keys])
    clusters = np.array([kk[1] for kk in keys])
    uniq = np.unique(clusters)
    by = {c: (d[clusters == c], w[clusters == c]) for c in uniq}
    rng = np.random.default_rng(seed)
    stats = np.empty(n_boot)
    for i in range(n_boot):
        pick = rng.choice(uniq, size=len(uniq), replace=True)
        dd = np.concatenate([by[c][0] for c in pick])
        ww = np.concatenate([by[c][1] for c in pick])
        stats[i] = dd.sum() / ww.sum()
    return {"n_paired": len(keys),
            "expectancy_diff_r": round(float(d.sum() / w.sum()), 4),
            "ci": [round(float(np.percentile(stats, q)), 4)
                   for q in (2.5, 97.5)]}


TINY_R_USD = 100.0     # disclosed threshold for the beside-stat only


def excl_tiny(trades):
    return [t for t in trades if t["r_usd"] >= TINY_R_USD]


def gap_through(sim):
    """AMENDMENT 1 clause 5: the DISTRIBUTION of gap-through on real
    exits — how far below the trigger level the next open printed.
    Reported, never modelled as a cost."""
    gaps = []
    for t in sim["trades"]:
        if t["synthetic_close"] or not t.get("exit_trigger"):
            continue
        trig = t["exit_trigger"]
        gaps.append((trig - t["exit_fill"]) / trig * 100.0)
    if not gaps:
        return {"n_exits": 0}
    a = np.array(gaps)
    return {"n_exits": len(a),
            "share_below_trigger_pct": round(
                float((a > 0).mean()) * 100, 1),
            "mean_gap_pct": round(float(a.mean()), 3),
            "median_gap_pct": round(float(np.median(a)), 3),
            "p90_gap_pct": round(float(np.percentile(a, 90)), 3),
            "worst_gap_pct": round(float(a.max()), 3)}


def fragile_check(e1_sim, sim):
    """Clause 6: does the arm's advantage rest on its top 3 trades by
    more than half? Computed on paired per-trade PnL differences."""
    key = lambda t: (t["ticker"], t["entry_date"])
    A = {key(t): t["pnl"] for t in e1_sim["trades"]}
    diffs = sorted((t["pnl"] - A[key(t)] for t in sim["trades"]),
                   reverse=True)
    total = float(sum(diffs))
    if total <= 0:
        return {"advantage_usd": round(total, 2), "fragile": False,
                "note": "no positive advantage to concentrate"}
    top3 = float(sum(diffs[:3]))
    return {"advantage_usd": round(total, 2),
            "top3_usd": round(top3, 2),
            "top3_share_pct": round(100.0 * top3 / total, 1),
            "fragile": bool(top3 > 0.5 * total)}


def primary_verdict(paired):
    """AMENDMENT 1 clause 1: the branch-mapping primary is the paired
    per-trade dollar expectancy. Decisive = CI excluding zero in BOTH
    D-006 spans."""
    lo = {sp: paired[sp]["expectancy"]["ci"][0]
          for sp in ("train", "validate")}
    hi = {sp: paired[sp]["expectancy"]["ci"][1]
          for sp in ("train", "validate")}
    if all(v > 0 for v in lo.values()):
        v = "BEATS"
    elif all(v < 0 for v in hi.values()):
        v = "LOSES"
    else:
        v = "INSIDE"
    return {"verdict": v,
            "expectancy_diff_full":
                paired["full"]["expectancy"]["expectancy_diff_r"],
            "ci_train": paired["train"]["expectancy"]["ci"],
            "ci_validate": paired["validate"]["expectancy"]["ci"]}


def portfolio_branch(paired):
    """AMENDMENT 1 clauses 2+4: the CAGR/drawdown table, admissible
    ONLY for zero-cash-negative arms. Every decisive branch requires
    both-span CI exclusion; the exchange rate is computed for BOTH
    mixed branches (2: wins CAGR loses drawdown; 2R: loses CAGR wins
    drawdown — the original prereg's own predicted outcome)."""
    def wins(field):
        return all(paired[s][field][0] > 0 for s in ("train", "validate"))

    def loses(field):
        return all(paired[s][field][1] < 0 for s in ("train", "validate"))

    dc = paired["full"]["d_cagr_pct"]
    dm = paired["full"]["d_mdd_pct"]
    xr = (round(abs(dc) / abs(dm), 3)
          if dm not in (None, 0) and dc is not None else None)
    if wins("d_cagr_ci") and wins("d_mdd_ci"):
        return {"branch": "1"}
    if wins("d_cagr_ci") and loses("d_mdd_ci"):
        return {"branch": "2",
                "exchange_rate_cagr_per_dd_point": xr}
    if loses("d_cagr_ci") and wins("d_mdd_ci"):
        return {"branch": "2R",
                "exchange_rate_cagr_per_dd_point": xr}
    if loses("d_cagr_ci") and loses("d_mdd_ci"):
        return {"branch": "3"}
    return {"branch": "4"}


# ------------------------------------------------------------------ main
def run(smoke=False, out_path=None):
    panel = bs.load_panel(smoke=smoke)
    cd = panel["cash_daily"]
    print(f"panel: {len(panel['px'])} tickers, {len(panel['days'])} days, "
          f"frame_hash {panel['frame_hash']}")

    # ---- gate 1: the incumbent reproduces the committed 5B block -----
    s1 = bs.run_sim(panel, "S1", "score")
    m1 = curve_metrics(s1["curve"], cd)
    if not smoke:
        with open(os.path.join(REPO, "docs",
                               "backtest-systems-results.json")) as f:
            committed = json.load(f)["arms"]["S1"]["score"]["full"]
        for kk in ("cagr_pct", "max_dd_pct", "end_equity", "trades"):
            got = len(s1["trades"]) if kk == "trades" else m1.get(kk)
            if got != committed[kk]:
                raise AssertionError(
                    f"INTEGRITY GATE 1 FAILED — S1 diverges from the "
                    f"committed 5B block on {kk}: {got} vs {committed[kk]}")
        print(f"gate 1: S1 reproduces the committed 5B S1/score block "
              f"({m1['end_equity']:,} end equity, "
              f"{len(s1['trades'])} trades)")

    # ---- the fixed entry list ----------------------------------------
    ledger = extract_ledger(s1["trades"])
    extend_px(panel["px"], {e["ticker"] for e in ledger})
    print(f"ledger: {len(ledger)} entries, "
          f"{len({e['ticker'] for e in ledger})} tickers")

    # ---- gate 2: this engine reproduces the harness under E1 ---------
    e1 = replay(ledger, panel, "E1")
    hv = {d: v for d, v in s1["curve"]}
    dev = max(abs(v - hv[d]) for d, v in e1["curve"])
    assert dev < 0.01, (
        f"INTEGRITY GATE 2 FAILED — the ledger replay deviates from "
        f"bs.run_sim by up to ${dev:.4f} on a day; the ledger or the "
        "engine is wrong and no arm can be trusted")
    kset = lambda tr: {(t["ticker"], t["entry_date"], t["exit_date"])
                       for t in tr}
    assert kset(e1["trades"]) == kset(s1["trades"]), \
        "INTEGRITY GATE 2 FAILED — trade sets differ"
    print(f"gate 2: ledger replay == harness E1 to ${dev:.6f} max daily "
          f"deviation, identical trade set")

    # ---- the arms ----------------------------------------------------
    sims = {"E1": e1}
    for arm in ARM_ORDER:
        if arm != "E1":
            sims[arm] = replay(ledger, panel, arm)
    if not smoke:
        for a, sm in sims.items():
            me = min(v for _, v in sm["curve"])
            assert me > 0, (
                f"{a}: equity reached ${me:,.2f} — a non-positive equity "
                "path makes CAGR meaningless and the study invalid")
    counts = {a: len(s["trades"]) for a, s in sims.items()}
    assert len(set(counts.values())) == 1, \
        f"trade counts differ — pairing broken: {counts}"
    print(f"arms: {len(ARM_ORDER)} replayed, "
          f"{counts['E1']} trades each (identical by construction)")

    results = {
        "schema": "backtest-exit-2",
        "prereg": f"{PREREG_PATH} sha256:{PREREG_SHA256[:16]} "
                  "(committed 2e96862, pre-push breadcrumb)",
        "amendment": f"{AMEND_PATH} sha256:{AMEND_SHA256[:16]} "
                     "(committed df1f2a2, pre-push breadcrumb; "
                     "primary = paired per-trade dollar expectancy)",
        "design": {"arms": {a: {"base": b, "atr_mult": k,
                                "ratchet": r}
                            for a, (b, k, r) in ARMS.items()},
                   "entry_list": "5B S1/score fills, extracted from the "
                                 "gate-1 replay; no redeployment",
                   "size_pct": bs.SIZE_PCT, "slip_bps": bs.SLIP_BPS,
                   "scorer": "v1 (as-lived master frame)"},
        "arms": {}, "paired_vs_E1": {}, "validity": {}, "decision": {},
    }
    for arm in ARM_ORDER:
        results["arms"][arm] = arm_block(sims[arm], cd)
    for arm in ARM_ORDER:
        if arm == "E1":
            continue
        results["paired_vs_E1"][arm] = paired_block(e1, sims[arm], cd)

    # ---- validity checks ---------------------------------------------
    v = results["validity"]
    v["gate1"] = {"end_equity": m1["end_equity"],
                  "trades": len(s1["trades"])} if not smoke else "smoke"
    v["gate2_max_daily_deviation_usd"] = round(dev, 6)
    v["trade_counts"] = counts
    v["exits_per_arm"] = {a: results["arms"][a]["full"]["exits_taken"]
                          for a in ARM_ORDER}
    v["ratchet"] = {
        "E5_vs_E1": _ratchet_stats(sims["E1"], sims["E5"]),
        "E8_vs_E6": _ratchet_stats(sims["E6"], sims["E8"]),
    }
    # AMENDMENT 1 clause 5: gap-through distribution replaces the
    # count-based break-even check (premise structurally false under a
    # fixed ledger, and likely inverted)
    v["gap_through"] = {a: gap_through(sims[a]) for a in ARM_ORDER}
    v["leverage"] = {a: {
        "min_cash": sims[a]["min_cash"],
        "neg_cash_days": sims[a]["neg_cash_days"],
        "mean_gross_over_equity": sims[a]["mean_gross_over_equity"],
        "max_gross_over_equity": sims[a]["max_gross_over_equity"]}
        for a in ARM_ORDER}
    v["portfolio_admissible_arms"] = sorted(
        a for a in ARM_ORDER if sims[a]["neg_cash_days"] == 0)

    # ---- decision mapping (AMENDMENT 1) ------------------------------
    clean = set(v["portfolio_admissible_arms"])
    for arm in ARM_ORDER:
        if arm == "E1":
            continue
        fr = fragile_check(e1, sims[arm])
        dec = {"primary": primary_verdict(results["paired_vs_E1"][arm]),
               "fragility": fr}
        if arm in clean:
            dec["portfolio"] = portfolio_branch(
                results["paired_vs_E1"][arm])
        else:
            dec["portfolio"] = {
                "branch": None,
                "note": "UNMEASURABLE — cash-negative under the fixed "
                        "ledger (amendment 1 clause 2); deferred to a "
                        "capital-feasible redeploying replay"}
        results["decision"][arm] = dec

    # clause 5 (amendment 1 clause 3): the PINNED statistic is the
    # paired per-trade dollar expectancy, both-span CI exclusion, with
    # the <5% bind-rate voider ENFORCED on the adoption boolean.
    def _clause5(base_sim, mod_sim, bind):
        per = {}
        for span, lo, hi in (("train", None, TRAIN_END),
                             ("validate", TRAIN_END, None)):
            per[span] = paired_dollar_expectancy_boot(
                span_trades(base_sim["trades"], lo, hi),
                span_trades(mod_sim["trades"], lo, hi))
        ci_ok = all(per[sp]["ci"][0] > 0
                    for sp in ("train", "validate"))
        voided = bind["exit_changed_pct"] < bind["clause5_void_below_pct"]
        return {"expectancy_by_span": per,
                "ci_excludes_zero_both_spans": bool(ci_ok),
                "bind_rate_pct": bind["exit_changed_pct"],
                "voided_by_bind_rate": bool(voided),
                "adopt": bool(ci_ok and not voided)}
    results["decision"]["ratchet_clause5"] = {
        "statistic": "paired per-trade dollar expectancy "
                     "(amendment 1, pinned)",
        "E5_vs_E1": _clause5(sims["E1"], sims["E5"],
                             v["ratchet"]["E5_vs_E1"]),
        "E8_vs_E6": _clause5(sims["E6"], sims["E8"],
                             v["ratchet"]["E8_vs_E6"]),
        "asymmetry_note": "E8-vs-E6 admissible because the leverage "
                          "bias runs AGAINST the ratchet (E8 holds "
                          "less); a win is conservative",
    }

    head = subprocess.run(["git", "rev-parse", "--short", "HEAD"],
                          capture_output=True, text=True,
                          cwd=REPO).stdout.strip()
    results["provenance"] = {
        "generated_from_commit": head,
        "frame_hash": panel["frame_hash"],
        "regime_series": panel["regime_prov"]["series_range"],
        "n_boot": N_BOOT, "seed": SEED,
        "smoke": bool(smoke),
    }
    canonical = json.loads(json.dumps(results))
    results["results_hash"] = hashlib.sha256(
        json.dumps(canonical, sort_keys=True).encode()).hexdigest()[:16]

    if out_path is None:
        out_path = os.path.join(REPO, "docs",
                                "backtest-exit-results.json")
        if smoke:
            out_path = out_path.replace(".json", ".smoke.json")
    with open(out_path, "w") as f:
        json.dump(results, f, indent=1)
    print(f"results -> {out_path} (hash {results['results_hash']})")
    return results, sims


def _ratchet_stats(base_sim, ratchet_sim):
    """Bind rate per the prereg: on what share of trades does the
    ratchet actually CHANGE the exit (different exit date vs the
    unratcheted base)? 'armed' = the ratchet level ever held above the
    raw rule's level during the trade."""
    key = lambda t: (t["ticker"], t["entry_date"])
    A = {key(t): t for t in base_sim["trades"]}
    B = {key(t): t for t in ratchet_sim["trades"]}
    assert set(A) == set(B)
    changed = sum(1 for k in A
                  if A[k]["exit_date"] != B[k]["exit_date"])
    armed = sum(1 for k in B if B[k].get("ratchet_bound"))
    n = len(A)
    return {"trades": n,
            "armed_pct": round(100.0 * armed / n, 1),
            "exit_changed": changed,
            "exit_changed_pct": round(100.0 * changed / n, 1),
            "clause5_void_below_pct": 5.0}


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--smoke", action="store_true")
    ap.add_argument("--out")
    a = ap.parse_args()
    run(smoke=a.smoke, out_path=a.out)


if __name__ == "__main__":
    main()
