#!/usr/bin/env python3
"""
Build 5B — strategy replay / counterfactual systems.

Layer A measured signal quality and found the A+ grade inverted on raw
forward returns. This layer answers the question Layer A explicitly
could not: does the doctrine earn its keep at the SYSTEM level —
expectancy per unit of risk — once stops, sizing and the regime ladder
are in play. Verdicts are read against docs/backtest-systems-prereg.md,
committed BEFORE this script ever wrote a results file.

THE ARMS (identical mechanics; only the entry gate differs):
  S1 production gate: B-or-better in Trending (advisory), A+-only in
     Choppy (hard), nothing in Caution/Risk-off (c3 blocks conditions)
  S2 A+-only in every permitted regime — the stricter doctrine
  S3 B-or-better in every permitted regime — what the hard gate excludes
  S4 top 6-month-momentum decile, no doctrine rows (c1 coherence only)
  S5 score>=50, no doctrine rows (c1 coherence only)
  Benchmarks: SPY buy-and-hold · equal-weight pool B&H · chassis-on-SPY

FIXED MECHANICS (production values, never swept):
  entry fill next open after the signal close · trailing stop = the
  production sma20_close rule (confirmed closes only, D-018; sell next
  open) · sizing 6.5% of equity (R15 band midpoint; {5,8} sensitivity —
  NOTE the risk-off discontinuity: 5% sizing yields ONE risk-off slot,
  6.5/8 yield zero) · R28 group caps <=20% and <=3 · regime ceiling
  90/50/25/5 gates NEW entries only · committed regime series ·
  grade-lite (Layer A caveats carry) · NO PROFIT TARGET exists (the R24
  gap): every arm holds to stop, so the R-distribution's right tail is
  unbounded BY DESIGN · re-entry permitted whenever the gate re-passes
  (churn reported).

SELECTION (ruled): the gate over-admits on 78-100%% of active days, so
ordering is first-class. Headline = highest composite score (the only
ordering the system itself endorses). NULL = random-K, 50 seeds — every
arm-vs-arm difference must clear that band to count. Native = momentum
rank for S4, lowest-extension for S1/S2/S3, score for S5 (its gate IS
the score).

Run: python3 scripts/backtest_systems.py [--smoke] [--out PATH]
"""

import argparse
import hashlib
import json
import os
import subprocess
import sys
from collections import Counter, defaultdict

REPO = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
sys.path.insert(0, REPO)
sys.path.insert(0, os.path.join(REPO, "scripts"))

import numpy as np
import pandas as pd

# ---------------------------------------------------------------------
# PINNED STUDY INPUTS (ruled 2026-08-10, Build 7's stop)
# ---------------------------------------------------------------------
# public/universe_ranking.json is REWRITTEN by the weekly rotation. Read
# live, it silently changes how R28's group caps bind and a committed
# backtest stops reproducing through no fault of its code — Build 7's
# integrity gate caught exactly that (the 2026-08-08 rotation moved 3
# tickers between sub-industries and 5B/6A drifted $587 of end equity).
# The studies therefore read the artifact from a FIXED COMMIT: the one
# their committed results were produced under. Same law as the
# sliding-window pin in docs/testing.md — anchor to a commit, never to
# whatever the sequence currently holds.
PINNED_UNIVERSE_COMMIT = "1d67d1c"      # 2026-08-01 rotation


def pinned_universe_ranking(commit=PINNED_UNIVERSE_COMMIT):
    """The group map as of the pinned commit. An unreachable anchor
    RAISES — a study that silently fell back to the live artifact would
    be unreproducible again, and would say nothing about it."""
    r = subprocess.run(
        ["git", "show", f"{commit}:public/universe_ranking.json"],
        capture_output=True, text=True, cwd=REPO)
    if r.returncode != 0 or not r.stdout.strip():
        raise RuntimeError(
            f"pinned universe artifact {commit} unreachable — the "
            "studies must not fall back to the live rotation "
            f"(git said: {r.stderr.strip()[:200]})")
    return json.loads(r.stdout)


CACHE = os.path.join(REPO, "data", "doctrine_cache")
PRICES = os.path.join(CACHE, "prices")
FRAME_PATH = os.path.join(CACHE, "master_frame.csv.gz")
REGIME_PATH = os.path.join(REPO, "data", "regime_daily.json")
B4CACHE = os.path.join(REPO, "data", "backtest_cache")

START, TRAIN_END = "2020-01-02", "2023-12-31"
CEILINGS = {"In-Trend-Full": 90.0, "In-Trend-Throttled": 50.0,
            "Out-Defensive": 25.0, "Out-Risk-off": 5.0}
SIZE_PCT = 6.5                 # R15 band midpoint (5-8); sweep {5,8}
SLIP_BPS = 5.0                 # per side; sweep {0,15}
GROUP_MAX_PCT, GROUP_MAX_N = 20.0, 3
RANDOM_SEEDS = 50
TRADING_DAYS = 252

ARMS = ("S1", "S2", "S3", "S4", "S5")
NATIVE_RULE = {"S1": "low_ext", "S2": "low_ext", "S3": "low_ext",
               "S4": "momentum", "S5": "score"}


# ------------------------------------------------------------------ data
def load_panel(smoke=False):
    """Prices (open/close/sma20/ext), gates from the Layer A frame,
    regime series, group map, IRX cash yield. Everything the sim reads.

    Every input is pinned: the group map by COMMIT, the gitignored
    caches by asserted content hash (2026-08-10 ruling)."""
    if not smoke:
        from study_inputs import assert_pinned_inputs
        assert_pinned_inputs(["prices_manifest", "regime_daily",
                              "master_frame", "IRX", "SPY"],
                             label="5B panel (shared by 6A and 7)")
    with open(REGIME_PATH) as f:
        regime = json.load(f)
    states = regime["states"]
    days = sorted(d for d in states if d >= START)

    frame = pd.read_csv(FRAME_PATH,
                        usecols=["date", "ticker", "grade", "score",
                                 "mom_6m", "c1_above", "chassis"])
    frame_hash = hashlib.sha256(
        pd.util.hash_pandas_object(frame, index=False).values.tobytes()
    ).hexdigest()[:16]
    d6 = frame[frame.mom_6m.notna()]
    dec = d6.groupby("date").mom_6m.transform(
        lambda s: pd.qcut(s.rank(method="first"), 10, labels=False,
                          duplicates="drop"))
    frame["top_decile"] = False
    frame.loc[d6.index, "top_decile"] = (dec == 9)

    tickers = sorted(frame.ticker.unique())
    if smoke:
        tickers = tickers[:60]
    px = {}
    for t in tickers:
        p = os.path.join(PRICES, f"{t.replace('/', '_')}.csv")
        if not os.path.exists(p):
            continue
        df = pd.read_csv(p, index_col=0, parse_dates=True)
        close = df["Close"].to_numpy(dtype=float)
        high = df["High"].to_numpy(dtype=float)
        low = df["Low"].to_numpy(dtype=float)
        sma20 = pd.Series(close).rolling(20).mean().to_numpy()
        prev = np.concatenate([[np.nan], close[:-1]])
        tr = np.maximum(high - low, np.maximum(np.abs(high - prev),
                                               np.abs(low - prev)))
        atr = pd.Series(tr).rolling(14).mean().to_numpy()
        with np.errstate(invalid="ignore", divide="ignore"):
            ext = (close - sma20) / atr
        iso = np.array([str(d)[:10] for d in df.index])
        px[t] = {"iso": iso, "pos": {s: i for i, s in enumerate(iso)},
                 "open": df["Open"].to_numpy(dtype=float),
                 "close": close, "sma20": sma20, "ext": ext}

    gates_by_day = defaultdict(list)
    for r in frame.itertuples(index=False):
        if r.ticker in px:
            gates_by_day[r.date].append(r)

    rk = pinned_universe_ranking()
    group_of = {}
    for g in rk.get("groups", []):
        for t in g.get("tickers", []) or []:
            # rows are dicts ({ticker, score, ...}) in the ranking artifact
            sym = t.get("ticker") if isinstance(t, dict) else t
            if sym:
                group_of.setdefault(sym, g.get("name"))

    irx = pd.read_csv(os.path.join(B4CACHE, "IRX.csv"), index_col=0,
                      parse_dates=True)["Close"]
    irx.index = [str(d)[:10] for d in irx.index]
    cash_daily = ((1.0 + irx / 100.0) ** (1.0 / TRADING_DAYS) - 1.0) \
        .reindex(days).ffill().fillna(0.0)

    return {"days": days, "states": states, "px": px,
            "gates_by_day": gates_by_day, "group_of": group_of,
            "cash_daily": cash_daily, "frame_hash": frame_hash,
            "regime_prov": regime["provenance"]}


def arm_pass(arm, r):
    """Does frame row r pass the arm's entry gate? (c3 lives inside the
    grade for S1-S3; S4/S5 carry only the c1 stop-coherence requirement.)"""
    if arm == "S1":
        return (r.chassis == "In-Trend-Full" and r.grade >= 2) or \
               (r.chassis == "In-Trend-Throttled" and r.grade == 3)
    if arm == "S2":
        return r.chassis in ("In-Trend-Full", "In-Trend-Throttled") \
            and r.grade == 3
    if arm == "S3":
        return r.chassis in ("In-Trend-Full", "In-Trend-Throttled") \
            and r.grade >= 2
    if arm == "S4":
        return bool(r.top_decile) and bool(r.c1_above)
    if arm == "S5":
        return r.score >= 50 and bool(r.c1_above)
    raise ValueError(arm)


def order_candidates(rows, rule, px, day, rng=None):
    """Selection ordering. Deterministic for score/low_ext/momentum
    (documented tie-breaks); a seeded shuffle for random."""
    if rule == "random":
        rows = list(rows)
        rng.shuffle(rows)
        return rows
    if rule == "score":
        def key(r):
            e = _ext_at(px[r.ticker], day)
            return (-r.score, e if e is not None else 99.0, r.ticker)
    elif rule == "low_ext":
        def key(r):
            e = _ext_at(px[r.ticker], day)
            return (e if e is not None else 99.0, -r.score, r.ticker)
    elif rule == "momentum":
        def key(r):
            m = r.mom_6m if r.mom_6m == r.mom_6m else -999.0
            return (-m, r.ticker)
    else:
        raise ValueError(rule)
    return sorted(rows, key=key)


def _ext_at(p, day):
    i = p["pos"].get(day)
    if i is None:
        return None
    e = p["ext"][i]
    return float(e) if np.isfinite(e) else None


# ------------------------------------------------------------------- sim
def run_sim(panel, arm, rule, *, seed=None, slip_bps=SLIP_BPS,
            size_pct=SIZE_PCT, capital=100_000.0):
    """One full replay. Signals at close T admit at open of T's next
    trading day; stops are confirmed-close (D-018) and sell the next
    open; the ceiling gates new entries only."""
    days = panel["days"]
    px = panel["px"]
    states = panel["states"]
    group_of = panel["group_of"]
    cash_daily = panel["cash_daily"]
    slip = slip_bps / 10_000.0
    rng = np.random.default_rng(seed) if rule == "random" else None

    cash = capital
    positions = {}           # t -> dict
    pending_entries = []     # queued at close T, filled next open
    pending_exits = []
    trades = []
    curve = []
    denials = Counter()
    cash_yield_usd = 0.0
    cash_yield_by_day = {}
    daily_meta = []

    for day in days:
        state = states[day]
        ceiling = CEILINGS[state]

        # ---- next-open FILLS queued yesterday -------------------------
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
            trades.append({
                "ticker": t, "entry_date": pos["entry_date"],
                "exit_date": day, "hold_days": pos["hold"],
                "r_usd": pos["r_usd"], "pnl": pnl,
                "r_mult": pnl / pos["r_usd"] if pos["r_usd"] > 0 else None,
                "group": pos["group"], "regime": pos["regime"],
                "entry_stop": pos["entry_stop"],
                "exit_fill": fill,
                "entry_ceiling": pos["entry_ceiling"],
                "entry_exposure_frac": pos["entry_exposure_frac"],
                "reentry": pos["reentry"], "synthetic_close": False})
        pending_exits = []

        equity_prev = cash + sum(p_["shares"] * p_["last_close"]
                                 for p_ in positions.values())
        for cand in pending_entries:
            t = cand["ticker"]
            if t in positions:
                denials["already_held"] += 1
                continue
            p = px[t]
            i = p["pos"].get(day)
            if i is None:
                # a halted/missing bar on the fill day DROPS the entry
                # (no retry) — declared; counted, never silent
                denials["no_bar"] += 1
                continue
            fill = p["open"][i]
            stop = cand["stop"]
            if not np.isfinite(fill) or fill <= stop:
                denials["gap_below_stop"] += 1      # R<=0: cannot size risk
                continue
            entry_ceiling = CEILINGS[cand["state"]]   # signal-day state
            size_usd = size_pct / 100.0 * equity_prev
            exposure = sum(p_["shares"] * p_["last_close"]
                           for p_ in positions.values())
            if exposure + size_usd > entry_ceiling / 100.0 * equity_prev:
                denials["ceiling"] += 1
                continue
            grp = group_of.get(t, "UNMAPPED")
            gpos = [p_ for p_ in positions.values() if p_["group"] == grp]
            if len(gpos) >= GROUP_MAX_N:
                denials["group_count"] += 1
                continue
            gexp = sum(p_["shares"] * p_["last_close"] for p_ in gpos)
            if gexp + size_usd > GROUP_MAX_PCT / 100.0 * equity_prev:
                denials["group_pct"] += 1
                continue
            if size_usd * (1.0 + slip) > cash:
                denials["cash"] += 1
                continue
            shares = size_usd / fill
            cost = shares * fill * (1.0 + slip)
            cash -= cost
            positions[t] = {
                "shares": shares, "entry_date": day, "cost_basis": cost,
                "r_usd": shares * (fill - stop), "group": grp,
                "regime": cand["state"],       # signal-day, not fill-day
                "entry_stop": stop,
                "entry_ceiling": entry_ceiling,
                "entry_exposure_frac": (exposure + size_usd) / equity_prev
                if equity_prev > 0 else None,
                "hold": 0, "last_close": fill,
                "reentry": cand["reentry"]}
        pending_entries = []

        # ---- cash yield ----------------------------------------------
        y = cash * cash_daily.loc[day]
        cash += y
        cash_yield_usd += y
        cash_yield_by_day[day] = y

        # ---- confirmed close: mark, stop check, signals ---------------
        for t, pos in list(positions.items()):
            p = px[t]
            i = p["pos"].get(day)
            if i is not None and np.isfinite(p["close"][i]):
                pos["last_close"] = p["close"][i]
                pos["hold"] += 1
                if np.isfinite(p["sma20"][i]) \
                        and not p["close"][i] > p["sma20"][i]:
                    # production: above_now = close > sma20; HELD exits
                    # when NOT above (close <= sma20 — the boundary case
                    # is real, D-018's flap was one cent). Strict < missed
                    # exact equality (review finding).
                    pending_exits.append(t)      # EXIT_FIRED -> next open

        rows = [r for r in panel["gates_by_day"].get(day, ())
                if arm_pass(arm, r)]
        seen = set()
        stopped_today = set(pending_exits)
        entered = Counter(tr["ticker"] for tr in trades)
        for r in order_candidates(rows, rule, px, day, rng):
            t = r.ticker
            if t in seen or t in positions or t in stopped_today:
                continue
            seen.add(t)
            p = px[t]
            i = p["pos"].get(day)
            if i is None or not np.isfinite(p["sma20"][i]):
                continue
            pending_entries.append({
                "ticker": t, "stop": float(p["sma20"][i]),
                # THE SIGNAL-DAY STATE travels with the queue (review
                # finding, critical): the fill-day state is derived from
                # the fill day's CLOSE — reading it at the open was
                # look-ahead. Admission and the stored regime label both
                # use what was knowable when the decision was made.
                "state": state,
                "reentry": entered[t] > 0})

        equity = cash + sum(p_["shares"] * p_["last_close"]
                            for p_ in positions.values())
        exp = equity - cash
        curve.append((day, equity))
        daily_meta.append((day, exp / equity if equity > 0 else 0.0,
                           len(positions)))

    # mark-out: close survivors at the final close (declared synthetic).
    # A mark is a VALUATION, not a fill — no slippage, so the accounting
    # identity (equity == capital + trade P&L + cash yield) holds exactly
    # and the curve and the trade log agree to the cent.
    last_day = days[-1]
    for t, pos in positions.items():
        proceeds = pos["shares"] * pos["last_close"]
        pnl = proceeds - pos["cost_basis"]
        trades.append({
            "ticker": t, "entry_date": pos["entry_date"],
            "exit_date": last_day, "hold_days": pos["hold"],
            "r_usd": pos["r_usd"], "pnl": pnl,
            "r_mult": pnl / pos["r_usd"] if pos["r_usd"] > 0 else None,
            "group": pos["group"], "regime": pos["regime"],
            "entry_stop": pos["entry_stop"],
            "exit_fill": pos["last_close"],
            "entry_ceiling": pos["entry_ceiling"],
            "entry_exposure_frac": pos["entry_exposure_frac"],
            "reentry": pos["reentry"], "synthetic_close": True})

    return {"curve": curve, "trades": trades, "denials": dict(denials),
            "cash_yield_usd": round(cash_yield_usd, 2),
            "cash_yield_by_day": cash_yield_by_day,
            "daily_meta": daily_meta}


# ------------------------------------------------------------- benchmarks
def bench_curves(panel, capital=100_000.0, slip_bps=SLIP_BPS):
    days = panel["days"]
    cash_daily = panel["cash_daily"]
    slip = slip_bps / 10_000.0
    spy = pd.read_csv(os.path.join(B4CACHE, "SPY.csv"), index_col=0,
                      parse_dates=True)
    spy.index = [str(d)[:10] for d in spy.index]
    spy = spy.reindex(days).ffill()
    out = {}

    sh = capital * (1 - slip) / spy["Open"].iloc[0]
    out["SPY_buy_hold"] = [(d, sh * c) for d, c in
                           zip(days, spy["Close"])]

    # equal-weight pool B&H: names priced at window start, held (late
    # listings excluded — declared)
    have = [t for t, p in panel["px"].items() if days[0] in p["pos"]]
    per = capital * (1 - slip) / len(have)
    shares = {}
    for t in have:
        p = panel["px"][t]
        shares[t] = per / p["open"][p["pos"][days[0]]]
    ew = []
    last = {t: None for t in have}
    for d in days:
        v = 0.0
        for t in have:
            p = panel["px"][t]
            i = p["pos"].get(d)
            if i is not None and np.isfinite(p["close"][i]):
                last[t] = p["close"][i]
            v += shares[t] * (last[t] or 0.0)
        ew.append((d, v))
    out["EW_pool_buy_hold"] = ew

    # chassis-on-SPY: hold SPY while In-Trend-*, else cash at IRX;
    # switches at next open (same fill discipline)
    eq = capital
    holding = False
    sh = 0.0
    switch = None
    chas = []
    for k, d in enumerate(days):
        if switch is not None:
            o = spy["Open"].iloc[k]
            if switch == "buy":
                sh = eq * (1 - slip) / o
                holding = True
            else:
                eq = sh * o * (1 - slip)
                holding = False
                sh = 0.0
            switch = None
        if holding:
            eq = sh * spy["Close"].iloc[k]
        else:
            eq *= (1.0 + cash_daily.loc[d])
        in_trend = panel["states"][d].startswith("In-")
        if in_trend != holding:
            switch = "buy" if in_trend else "sell"
        chas.append((d, eq))
    out["chassis_on_SPY"] = chas
    return out


# ---------------------------------------------------------------- metrics
def _span(curve, lo=None, hi=None):
    seg = [(d, v) for d, v in curve if (lo is None or d > lo)
           and (hi is None or d <= hi)]
    return seg


def curve_metrics(curve, cash_daily=None):
    if len(curve) < 3:
        return {}
    v = np.array([x[1] for x in curve])
    d = [x[0] for x in curve]
    ret = v[1:] / v[:-1] - 1.0
    years = len(v) / TRADING_DAYS
    cagr = (v[-1] / v[0]) ** (1 / years) - 1 if v[0] > 0 else None
    peak = np.maximum.accumulate(v)
    dd = v / peak - 1.0
    mdd = float(dd.min())
    # drawdown duration: longest run below the prior peak
    under = dd < 0
    runs, cur = [], 0
    for u in under:
        cur = cur + 1 if u else 0
        runs.append(cur)
    rf = (cash_daily.reindex(d).fillna(0.0).to_numpy()[1:]
          if cash_daily is not None else 0.0)
    ex = ret - rf
    sharpe = float(ex.mean() / ex.std() * np.sqrt(TRADING_DAYS)) \
        if ex.std() > 0 else None
    downside = ex[ex < 0]
    sortino = float(ex.mean() / downside.std() * np.sqrt(TRADING_DAYS)) \
        if len(downside) > 1 and downside.std() > 0 else None
    return {"cagr_pct": round(cagr * 100, 2) if cagr is not None else None,
            "max_dd_pct": round(mdd * 100, 2),
            "max_dd_duration_days": int(max(runs) if runs else 0),
            "sharpe": round(sharpe, 3) if sharpe is not None else None,
            "sortino": round(sortino, 3) if sortino is not None else None,
            "end_equity": round(float(v[-1]), 2)}


def trade_metrics(trades):
    rs = [t["r_mult"] for t in trades if t["r_mult"] is not None]
    if not rs:
        return {"trades": 0}
    a = np.array(rs)
    wins = a[a > 0]
    losses = a[a <= 0]
    pf = float(wins.sum() / -losses.sum()) if losses.sum() < 0 else None
    # EXPECTANCY DEFINITION (fixed at smoke stage, BEFORE full results —
    # see the report's methods note): with the production sma20_close
    # stop, R = fill − SMA20 can be arbitrarily small (a name entering a
    # hair above its stop), so the MEAN of per-trade R-multiples is
    # dominated by near-zero-R trades (the smoke's S5 read −15R from
    # exactly this). Under R15's NOTIONAL sizing the economically
    # meaningful aggregate is dollar-risk-weighted: sum(PnL)/sum(R_usd).
    # That is the prereg's "expectancy per unit of risk" headline;
    # per-trade mean and median are reported beside it, never hidden.
    r_usd = np.array([t["r_usd"] for t in trades
                      if t["r_mult"] is not None])
    pnl = np.array([t["pnl"] for t in trades if t["r_mult"] is not None])
    # PF on R-MULTIPLES inherits the tiny-R distortion (a small dollar
    # loss at near-zero R is a huge negative multiple), so the dollar
    # PF is reported beside it — same trades, dollar P&L basis.
    wpnl = pnl[pnl > 0]
    lpnl = pnl[pnl <= 0]
    pf_usd = float(wpnl.sum() / -lpnl.sum()) if lpnl.sum() < 0 else None
    return {
        "trades": len(a),
        "expectancy_r": round(float(pnl.sum() / r_usd.sum()), 4)
            if r_usd.sum() > 0 else None,
        "expectancy_r_trade_mean": round(float(a.mean()), 4),
        "median_r": round(float(np.median(a)), 4),
        "win_rate_pct": round(float((a > 0).mean()) * 100, 1),
        "avg_win_r": round(float(wins.mean()), 3) if len(wins) else None,
        "avg_loss_r": round(float(losses.mean()), 3) if len(losses) else None,
        "profit_factor": round(pf, 3) if pf else None,
        "profit_factor_usd": round(pf_usd, 3) if pf_usd else None,
        "avg_pnl_usd": round(float(pnl.mean()), 2),
        "r_percentiles": {str(p): round(float(np.percentile(a, p)), 3)
                          for p in (1, 5, 25, 50, 75, 95, 99)},
        "left_tail_share_lt_minus2r_pct":
            round(float((a < -2).mean()) * 100, 2),
        # gap-through + weekend risk: the exit FILL lands below the stop
        # that defined R (review finding: material share; the -1R floor
        # the stop implies is not a floor at the open)
        "share_exit_fill_below_entry_stop_pct": round(float(np.mean(
            [t["exit_fill"] < t["entry_stop"] for t in trades
             if t["r_mult"] is not None])) * 100, 1),
        "avg_hold_days": round(float(np.mean(
            [t["hold_days"] for t in trades])), 1),
        "reentry_share_pct": round(float(np.mean(
            [t["reentry"] for t in trades])) * 100, 1),
        "synthetic_close_trades": int(sum(
            t["synthetic_close"] for t in trades)),
    }


def result_block(sim, panel):
    curve = sim["curve"]
    cd = panel["cash_daily"]
    out = {"full": {**curve_metrics(curve, cd),
                    **trade_metrics(sim["trades"])}}
    for span, lo, hi in (("train", None, TRAIN_END),
                         ("validate", TRAIN_END, None)):
        seg = _span(curve, lo, hi)
        tr = [t for t in sim["trades"]
              if (lo is None or t["entry_date"] > lo)
              and (hi is None or t["entry_date"] <= hi)]
        out[span] = {**curve_metrics(seg, cd), **trade_metrics(tr)}
    out["denials"] = sim["denials"]
    out["cash_yield_usd"] = sim["cash_yield_usd"]
    # ruled: the cash-yield contribution must be VISIBLE per span — IRX
    # ~0% in train vs ~5% in validate structurally flatters high-cash arms
    cy = sim["cash_yield_by_day"]
    out["cash_yield_by_span_usd"] = {
        "train": round(sum(v for d, v in cy.items() if d <= TRAIN_END), 2),
        "validate": round(sum(v for d, v in cy.items()
                              if d > TRAIN_END), 2)}
    out["time_in_market_pct"] = round(float(np.mean(
        [m[1] for m in sim["daily_meta"]])) * 100, 1)
    out["avg_positions"] = round(float(np.mean(
        [m[2] for m in sim["daily_meta"]])), 1)
    # per-regime trade decomposition
    per_reg = {}
    for st in CEILINGS:
        tr = [t for t in sim["trades"] if t["regime"] == st]
        if tr:
            per_reg[st] = trade_metrics(tr)
    out["per_regime"] = per_reg
    # concentration: HHI of per-group trade counts
    gc = Counter(t["group"] for t in sim["trades"])
    tot = sum(gc.values())
    out["group_hhi"] = round(sum((v / tot) ** 2 for v in gc.values()), 3) \
        if tot else None
    out["top_groups"] = gc.most_common(5)
    return out


# ------------------------------------------------------------------ main
def run(smoke=False, out_path=None):
    panel = load_panel(smoke=smoke)
    print(f"panel: {len(panel['px'])} tickers, {len(panel['days'])} days, "
          f"frame_hash {panel['frame_hash']}")
    seeds = 8 if smoke else RANDOM_SEEDS

    results = {"schema": "backtest-systems-1",
               "prereg": "docs/backtest-systems-prereg.md",
               "design": {
                   "size_pct": SIZE_PCT, "slip_bps": SLIP_BPS,
                   "group_max_pct": GROUP_MAX_PCT,
                   "group_max_n": GROUP_MAX_N,
                   "ceilings": CEILINGS, "train_end": TRAIN_END,
                   "no_profit_target": "R24 gap — every arm holds to "
                   "stop; the right tail is unbounded by design",
                   "reentry": "permitted whenever the gate re-passes",
                   "selection": {"headline": "score",
                                 "null": f"random x{seeds}",
                                 "native": NATIVE_RULE}},
               "arms": {}, "random_bands": {}, "benchmarks": {},
               "sensitivities": {"costs": {}, "sizing": {}}}

    score_sims = {}
    for arm in ARMS:
        print(f"  {arm}: score-order...", flush=True)
        sim = run_sim(panel, arm, "score")
        score_sims[arm] = sim
        results["arms"][arm] = {"score": result_block(sim, panel)}
        native = NATIVE_RULE[arm]
        if native != "score":
            sim_n = run_sim(panel, arm, native)
            results["arms"][arm][f"native_{native}"] = \
                result_block(sim_n, panel)
        else:
            results["arms"][arm]["native_score"] = "same as headline"

    print("  random-K bands...", flush=True)
    rand_runs = {}
    for arm in ARMS:
        per_seed = []
        for s in range(seeds):
            sim = run_sim(panel, arm, "random", seed=s)
            rb = result_block(sim, panel)
            per_seed.append({
                "expectancy_r": rb["full"].get("expectancy_r"),
                "max_dd_pct": rb["full"].get("max_dd_pct"),
                "cagr_pct": rb["full"].get("cagr_pct"),
                "trades": rb["full"].get("trades")})
        rand_runs[arm] = per_seed
        er = [x["expectancy_r"] for x in per_seed
              if x["expectancy_r"] is not None]
        dd = [x["max_dd_pct"] for x in per_seed
              if x["max_dd_pct"] is not None]
        results["random_bands"][arm] = {
            "expectancy_r": {"p2_5": round(float(np.percentile(er, 2.5)), 4),
                             "p50": round(float(np.percentile(er, 50)), 4),
                             "p97_5": round(float(np.percentile(er, 97.5)), 4)},
            "max_dd_pct": {"p2_5": round(float(np.percentile(dd, 2.5)), 2),
                           "p50": round(float(np.percentile(dd, 50)), 2),
                           "p97_5": round(float(np.percentile(dd, 97.5)), 2)},
            "n_seeds": len(per_seed)}

    # prereg test 1: paired-by-seed difference bands for the key pairs
    # ALL 10 pairs (review finding: S4-S5 was untested and would have
    # passed both prereg tests unreported)
    pairs = [(a, b) for i, a in enumerate(ARMS) for b in ARMS[i + 1:]]
    results["pair_tests"] = {}
    for a, b in pairs:
        diffs = [ra["expectancy_r"] - rb_["expectancy_r"]
                 for ra, rb_ in zip(rand_runs[a], rand_runs[b])
                 if ra["expectancy_r"] is not None
                 and rb_["expectancy_r"] is not None]
        head = (results["arms"][a]["score"]["full"].get("expectancy_r"),
                results["arms"][b]["score"]["full"].get("expectancy_r"))
        obs = None if None in head else round(head[0] - head[1], 4)
        band = [round(float(np.percentile(diffs, 2.5)), 4),
                round(float(np.percentile(diffs, 97.5)), 4)] \
            if diffs else None
        results["pair_tests"][f"{a}_minus_{b}"] = {
            "headline_diff_expectancy_r": obs,
            "randomK_band": band,
            "clears_band": (None if obs is None or band is None
                            else bool(obs < band[0] or obs > band[1]))}

    # Layer A's bootstrap treatment on the arm-vs-arm differences (ruled):
    # date-cluster bootstrap of the DOLLAR-WEIGHTED expectancy difference —
    # entry dates are the clusters (same-day entries share the tape), 2000
    # resamples. Complements the random-K band: random-K asks "does the
    # difference exceed selection noise?", this asks "does it exceed
    # sampling noise of the trades themselves?"
    print("  pair bootstrap...", flush=True)
    rng = np.random.default_rng(20260726)
    for a, b in pairs:
        ta = [(t["entry_date"], t["pnl"], t["r_usd"])
              for t in score_sims[a]["trades"] if t["r_mult"] is not None]
        tb = [(t["entry_date"], t["pnl"], t["r_usd"])
              for t in score_sims[b]["trades"] if t["r_mult"] is not None]
        by_a, by_b = defaultdict(list), defaultdict(list)
        for d, p, r_ in ta:
            by_a[d].append((p, r_))
        for d, p, r_ in tb:
            by_b[d].append((p, r_))
        dates = sorted(set(by_a) | set(by_b))
        diffs = []
        for _ in range(2000):
            pick = rng.choice(dates, size=len(dates), replace=True)
            pa = ra = pb = rb = 0.0
            for d in pick:
                for p, r_ in by_a.get(d, ()):
                    pa += p
                    ra += r_
                for p, r_ in by_b.get(d, ()):
                    pb += p
                    rb += r_
            if ra > 0 and rb > 0:
                diffs.append(pa / ra - pb / rb)
        dd = np.array(diffs)
        results["pair_tests"][f"{a}_minus_{b}"]["cluster_bootstrap"] = {
            "ci_2_5": round(float(np.percentile(dd, 2.5)), 4),
            "ci_97_5": round(float(np.percentile(dd, 97.5)), 4),
            "p_diff_le_0": round(float((dd <= 0).mean()), 4),
            "n_boot": len(dd)}

    print("  benchmarks...", flush=True)
    for name, curve in bench_curves(panel).items():
        results["benchmarks"][name] = {
            "full": curve_metrics(curve, panel["cash_daily"]),
            "train": curve_metrics(_span(curve, None, TRAIN_END),
                                   panel["cash_daily"]),
            "validate": curve_metrics(_span(curve, TRAIN_END, None),
                                      panel["cash_daily"])}

    print("  cost sensitivity...", flush=True)
    for bps in (0.0, 15.0):
        blk = {}
        for arm in ARMS:
            sim = run_sim(panel, arm, "score", slip_bps=bps)
            b = result_block(sim, panel)
            blk[arm] = {"expectancy_r": b["full"].get("expectancy_r"),
                        "cagr_pct": b["full"].get("cagr_pct"),
                        "max_dd_pct": b["full"].get("max_dd_pct")}
        results["sensitivities"]["costs"][f"{bps:g}bps"] = blk
    # ranking stability across costs (the ruled decision-relevant test)
    ranks = {}
    for lvl in ("0bps", "5bps_headline", "15bps"):
        if lvl == "5bps_headline":
            vals = {a: results["arms"][a]["score"]["full"].get(
                "expectancy_r") for a in ARMS}
        else:
            vals = {a: results["sensitivities"]["costs"][lvl][a][
                "expectancy_r"] for a in ARMS}
        ranks[lvl] = sorted(ARMS, key=lambda a: -(vals[a] or -99))
    results["sensitivities"]["cost_rank_stable"] = \
        len({tuple(v) for v in ranks.values()}) == 1
    results["sensitivities"]["cost_ranks"] = ranks

    print("  sizing sensitivity...", flush=True)
    for pct in (5.0, 8.0):
        blk = {}
        for arm in ARMS:
            sim = run_sim(panel, arm, "score", size_pct=pct)
            b = result_block(sim, panel)
            blk[arm] = {"expectancy_r": b["full"].get("expectancy_r"),
                        "cagr_pct": b["full"].get("cagr_pct"),
                        "max_dd_pct": b["full"].get("max_dd_pct")}
        results["sensitivities"]["sizing"][f"{pct:g}pct"] = blk
    results["sensitivities"]["sizing_note"] = (
        "NOT a pure rescale at the boundary: 5% sizing admits ONE "
        "risk-off slot (5<=5), 6.5/8% admit zero — a discontinuity in "
        "risk-off behavior, not a scale factor")

    head = subprocess.run(["git", "rev-parse", "--short", "HEAD"],
                          capture_output=True, text=True,
                          cwd=REPO).stdout.strip()
    results["provenance"] = {
        "frame_hash": panel["frame_hash"],
        "regime_series": panel["regime_prov"],
        "generated_from_commit": head,
        "group_map": "public/universe_ranking.json (today's GICS applied "
                     "across the window — declared)"}
    # canonical hash: computed over a JSON ROUND-TRIP of the payload so
    # the committed file rehashes to the same value (review finding:
    # int-vs-str keys made the stored hash unrecomputable from the file)
    canonical = json.loads(json.dumps(results))
    blob = json.dumps(canonical, sort_keys=True).encode()
    results["results_hash"] = hashlib.sha256(blob).hexdigest()[:16]

    out_path = out_path or os.path.join(
        REPO, "docs", "backtest-systems-results.json")
    with open(out_path, "w") as f:
        json.dump(results, f, indent=1)
    print(f"results -> {out_path}  (results_hash {results['results_hash']})")
    return results


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--smoke", action="store_true")
    ap.add_argument("--out")
    a = ap.parse_args()
    run(smoke=a.smoke, out_path=a.out)


if __name__ == "__main__":
    main()
