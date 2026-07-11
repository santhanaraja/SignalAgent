#!/usr/bin/env python3
"""
Build 4 — walk-forward regime backtest (2015-01-01 -> present).

Feeds point-in-time reconstructed gauge inputs through the EXTRACTED
production regime functions (framework.regime_calculator.compute_regime —
the same code production runs; see test_regime_extraction.py for the
replay pin that licenses this reconstruction).

Execution model (Decision 1): signals compute on close T; all simulated
trades execute at T+1 OPEN. Equity accrues open-to-open, so the exposure
held during [open_{t+1}, open_{t+2}) is the weight decided at close t.
The hypothetical T-close fill is recorded on every switch; the signed sum
is the EOD-lag cost.

Point-in-time discipline:
- VIX 5d avg, breadth (RSP/SPY, production formula), SPY-vs-200DMA: all
  trailing windows ending at close T (known at the close).
- HY OAS: FRED publishes BAMLH0A0HYM2 for day X on the following business
  morning, so the value known at close T is dated <= T-1 (ffill to the
  trading calendar, then shift 1). Production reads "latest available" at
  run time and sees the same thing.
- No series access past T anywhere; pinned by the walk-window and shift
  tests in test_backtest_regime.py.

Usage:
    python3 scripts/backtest_regime.py                  # full run + grid
    python3 scripts/backtest_regime.py --skip-grid
Outputs: docs/backtest-regime-results.json, docs/img/backtest_*.png
"""

import argparse
import json
import os
import sys

import numpy as np
import pandas as pd

REPO = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
sys.path.insert(0, REPO)

from framework.regime_calculator import compute_regime  # noqa: E402

CACHE = os.path.join(REPO, "data", "backtest_cache")
TRADING_DAYS = 252

STATES = ["Risk-on / Trending", "Risk-on / Choppy", "Caution", "Risk-off"]
BINARY_WEIGHTS = {"Risk-on / Trending": 1.0, "Risk-on / Choppy": 1.0,
                  "Caution": 0.0, "Risk-off": 0.0}
LADDER_WEIGHTS = {"Risk-on / Trending": 1.0, "Risk-on / Choppy": 0.6,
                  "Caution": 0.3, "Risk-off": 0.0}

# Stress windows for the drawdown table (Decision 3)
STRESS_WINDOWS = [
    ("2015-16 correction", "2015-07-01", "2016-02-29"),
    ("Q4 2018", "2018-09-20", "2018-12-31"),
    ("COVID 2020", "2020-02-19", "2020-03-31"),
    ("2022 bear", "2022-01-03", "2022-10-31"),
    ("2025 tariff shock", "2025-02-15", "2025-05-15"),
    ("2026 semi unwind", "2026-06-15", "2026-07-10"),
]


# ---------------------------------------------------------------- data ----

def load_series(cache_dir=CACHE):
    """Cached pulls -> aligned frames. Raises with instructions if absent."""
    need = ["SPY", "RSP", "VIX", "IRX", "OAS"]
    missing = [n for n in need if not os.path.exists(
        os.path.join(cache_dir, f"{n}.csv"))]
    if missing:
        raise SystemExit(
            f"cache missing {missing} — see docs/backtest-regime.md 'Data' "
            f"for the fetch recipe (yfinance + FRED/Wayback splice)")
    out = {}
    for n in need:
        df = pd.read_csv(os.path.join(cache_dir, f"{n}.csv"),
                         index_col=0, parse_dates=True)
        out[n] = df
    return out


def build_inputs(raw, start, end):
    """Point-in-time gauge inputs, one row per SPY trading day T.
    Every window TRAILS through T; OAS additionally lags one day
    (publication). Returns the frame plus the aligned open/close prices."""
    spy_c, spy_o = raw["SPY"]["Close"], raw["SPY"]["Open"]
    rsp_c = raw["RSP"]["Close"].reindex(spy_c.index).ffill()
    vix_c = raw["VIX"]["Close"].reindex(spy_c.index).ffill()
    irx_c = raw["IRX"]["Close"].reindex(spy_c.index).ffill()
    oas = raw["OAS"].iloc[:, 0]

    f = pd.DataFrame(index=spy_c.index)
    f["vix_5d"] = vix_c.rolling(5).mean()
    # production breadth formula: current RSP/SPY ratio vs ratio of 20d means
    ratio = rsp_c / spy_c
    ratio_20d = rsp_c.rolling(20).mean() / spy_c.rolling(20).mean()
    f["breadth_20d"] = (ratio / ratio_20d - 1.0) * 100.0
    f["spy_vs_200dma"] = (spy_c / spy_c.rolling(200).mean() - 1.0) * 100.0
    # publication lag: known-at-close-T value is dated <= T-1. As-of
    # reindex (method=ffill) so OAS observations dated on non-SPY days
    # (ICE month-end weekend prints) still carry forward — a plain
    # reindex-then-ffill silently drops them.
    f["oas"] = oas.sort_index().reindex(spy_c.index, method="ffill").shift(1)
    f["spy_open"], f["spy_close"] = spy_o, spy_c
    f["cash_daily"] = (1.0 + irx_c / 100.0) ** (1.0 / TRADING_DAYS) - 1.0

    f = f.loc[(f.index >= start) & (f.index <= end)]
    if f[["vix_5d", "breadth_20d", "spy_vs_200dma", "oas"]].isna().any().any():
        bad = f[f[["vix_5d", "breadth_20d", "spy_vs_200dma", "oas"]]
                .isna().any(axis=1)].index
        raise SystemExit(f"NaN gauge inputs on {len(bad)} days "
                         f"(first {bad[0].date()}) — extend the cache warmup")
    return f


def compute_states(inputs, vix_thresholds=(18.0, 22.0),
                   oas_thresholds=(3.0, 4.0), breadth_band=0.5):
    """Regime state per close T through the extracted production function."""
    states = [
        compute_regime(row.vix_5d, row.oas, row.breadth_20d,
                       row.spy_vs_200dma,
                       vix_thresholds=vix_thresholds,
                       oas_thresholds=oas_thresholds,
                       breadth_band=breadth_band)["state"]
        for row in inputs.itertuples()
    ]
    return pd.Series(states, index=inputs.index, name="state")


# ----------------------------------------------------------- simulation ----

def simulate(weights, inputs, fill="next_open"):
    """Open-to-open equity for a weight series decided at close T.

    fill="next_open": exposure over [open_{t+1}, open_{t+2}) = w_t (the
    production execution model). fill="same_close": the hypothetical
    T-close fill — exposure over [close_t, close_{t+1}) = w_t; used only
    to price the EOD lag.
    """
    w = weights.astype(float)
    cash = inputs["cash_daily"]
    if fill == "next_open":
        o = inputs["spy_open"]
        seg_ret = o.shift(-1) / o - 1.0        # open_t -> open_{t+1}
        exposure = w.shift(1)                   # decided at close t-1
        daily = (exposure * seg_ret + (1.0 - exposure) * cash).dropna()
        # final partial leg open_L -> close_L so this sim ends at the same
        # timestamp as the close-fill sim (endpoint parity for the exact
        # EOD-lag cross-check)
        tail_exp = float(w.iloc[-2]) if len(w) > 1 else float(w.iloc[-1])
        tail = tail_exp * (inputs["spy_close"].iloc[-1]
                           / inputs["spy_open"].iloc[-1] - 1.0)
        daily = pd.concat([daily, pd.Series([tail], index=[w.index[-1]])])
    else:
        c = inputs["spy_close"]
        seg_ret = c.shift(-1) / c - 1.0
        exposure = w
        daily = (exposure * seg_ret + (1.0 - exposure) * cash).dropna()
    equity = (1.0 + daily).cumprod()
    return equity, daily


def eod_lag_cost(weights, inputs):
    """Per-switch signed cost of executing at T+1 open instead of close T.
    cost = dw * (open_{T+1}/close_T - 1); positive = the lag hurt
    (bought higher / sold lower than the close you signaled on)."""
    w = weights.astype(float)
    dw = w.diff().fillna(0.0)
    switches = dw[dw != 0.0]
    rows = []
    for t, d in switches.items():
        loc = inputs.index.get_loc(t)
        if loc + 1 >= len(inputs.index):
            continue                            # last-day switch can't fill
        nxt = inputs.index[loc + 1]
        slip = inputs["spy_open"].loc[nxt] / inputs["spy_close"].loc[t] - 1.0
        rows.append({"date": str(t.date()), "dw": round(float(d), 2),
                     "slippage_pct": round(float(slip) * 100, 4),
                     "cost_pct": round(float(d * slip) * 100, 4)})
    total = sum(r["cost_pct"] for r in rows)
    return rows, total


def whipsaws(weights, max_days=10):
    """Round trips reversed within max_days trading days: an exposure cut
    followed by a re-raise (or vice versa) inside the window."""
    w = weights.astype(float)
    dw = w.diff().fillna(0.0)
    switches = [(t, d) for t, d in dw.items() if d != 0.0]
    idx = {t: i for i, t in enumerate(weights.index)}
    out = []
    for (t1, d1), (t2, d2) in zip(switches, switches[1:]):
        if d1 * d2 < 0 and idx[t2] - idx[t1] <= max_days:
            out.append({"out": str(t1.date()), "back": str(t2.date()),
                        "days": idx[t2] - idx[t1]})
    return out


def metrics(equity, daily, weights, cash_daily, years):
    dd = equity / equity.cummax() - 1.0
    ex_ret = daily - cash_daily.reindex(daily.index).fillna(0.0)
    downside = daily[daily < 0]
    return {
        "cagr_pct": round((float(equity.iloc[-1]) ** (1 / years) - 1) * 100, 2),
        "max_drawdown_pct": round(float(dd.min()) * 100, 2),
        "sharpe": round(float(ex_ret.mean() / ex_ret.std()
                              * np.sqrt(TRADING_DAYS)), 2),
        "sortino": round(float(daily.mean() / downside.std()
                               * np.sqrt(TRADING_DAYS)), 2),
        # two distinct facts (review finding): mean weight vs share of days
        # holding ANY exposure — for the graded ladder they differ a lot
        "avg_exposure_pct": round(float(weights.mean()) * 100, 1),
        "days_invested_pct": round(float((weights > 0).mean()) * 100, 1),
        "switches_per_year": round(
            float((weights.diff().fillna(0) != 0).sum()) / years, 1),
    }


# ------------------------------------------------------------ analyses ----

def info_test(states, inputs, horizons=(5, 10, 20)):
    """Per-state forward SPY returns measured from T+1 open (no strategy
    wrapper — is there information in the states at all?)."""
    o = inputs["spy_open"]
    res = {}
    for st in STATES:
        mask = states == st
        res[st] = {"days": int(mask.sum()),
                   "share_pct": round(float(mask.mean()) * 100, 1)}
        for h in horizons:
            fwd = (o.shift(-(1 + h)) / o.shift(-1) - 1.0) * 100
            v = fwd[mask].dropna()
            res[st][f"fwd_{h}d"] = {
                "mean_pct": round(float(v.mean()), 3),
                "median_pct": round(float(v.median()), 3),
                "hit_rate_pct": round(float((v > 0).mean()) * 100, 1),
                "n": int(len(v)),
            }
    return res


def duration_stats(states):
    runs = []
    cur, n = None, 0
    for s in states:
        if s == cur:
            n += 1
        else:
            if cur is not None:
                runs.append((cur, n))
            cur, n = s, 1
    runs.append((cur, n))
    out = {}
    for st in STATES:
        lens = [n for s, n in runs if s == st]
        if lens:
            out[st] = {"runs": len(lens), "median_days": float(np.median(lens)),
                       "mean_days": round(float(np.mean(lens)), 1),
                       "max_days": int(max(lens)),
                       "runs_le_2d": sum(1 for x in lens if x <= 2)}
    return out


def drawdown_table(weights_by_name, states, inputs):
    spy_c = inputs["spy_close"]
    rows = []
    for name, a, b in STRESS_WINDOWS:
        win = (inputs.index >= a) & (inputs.index <= b)
        if not win.any():
            continue
        spy_ret = (spy_c[win].iloc[-1] / spy_c[win].iloc[0] - 1) * 100
        row = {"window": name, "from": a, "to": b,
               "spy_return_pct": round(float(spy_ret), 1),
               "states": dict(states[win].value_counts())}
        for sname, w in weights_by_name.items():
            ww = w[win]
            row[sname] = {"avg_exposure_pct": round(float(ww.mean()) * 100, 1),
                          "min_exposure_pct": round(float(ww.min()) * 100, 1),
                          "min_reached": str(ww.idxmin().date()) if len(ww) else None}
        rows.append(row)
    return rows


# ------------------------------------------------------------------ main ----

def run(start="2015-01-01", end=None, skip_grid=False,
        vix_thresholds=(18.0, 22.0), oas_thresholds=(3.0, 4.0),
        breadth_band=0.5):
    raw = load_series()
    end = end or str(raw["SPY"].index[-1].date())
    inputs = build_inputs(raw, start, end)
    years = len(inputs) / TRADING_DAYS

    states = compute_states(inputs, vix_thresholds, oas_thresholds,
                            breadth_band)

    weights = {
        "binary": states.map(BINARY_WEIGHTS),
        "ladder": states.map(LADDER_WEIGHTS),
        "buy_and_hold": pd.Series(1.0, index=states.index),
        "naked_200dma": (inputs["spy_vs_200dma"] >= 0).astype(float),
    }

    results = {"config": {
        "start": str(inputs.index[0].date()), "end": str(inputs.index[-1].date()),
        "trading_days": len(inputs), "vix_thresholds": list(vix_thresholds),
        "oas_thresholds": list(oas_thresholds), "breadth_band": breadth_band,
        "execution": "signal close T, fill T+1 open (open-to-open equity)",
        "oas_lag": "1 trading day (FRED publication)",
    }, "strategies": {}}

    equities = {}
    for name, w in weights.items():
        eq, daily = simulate(w, inputs, fill="next_open")
        equities[name] = eq
        m = metrics(eq, daily, w, inputs["cash_daily"], years)
        wa = whipsaws(w)
        m["whipsaw_roundtrips"] = len(wa)
        m["whipsaws_per_year"] = round(len(wa) / years, 1)
        if name in ("binary", "ladder"):
            eq_close, _ = simulate(w, inputs, fill="same_close")
            switches, total_cost = eod_lag_cost(w, inputs)
            m["eod_lag_cost_total_pct"] = round(total_cost, 2)
            m["eod_lag_cost_per_switch_bp"] = round(
                total_cost * 100 / max(len(switches), 1), 1)
            m["eod_lag_switches"] = len(switches)
            m["closefill_final_equity_ratio"] = round(
                float(eq_close.iloc[-1] / eq.iloc[-1]), 4)
        results["strategies"][name] = m

    results["state_share"] = {s: int((states == s).sum()) for s in STATES}
    results["info_test"] = info_test(states, inputs)
    results["durations"] = duration_stats(states)
    results["drawdown_table"] = drawdown_table(
        {k: weights[k] for k in ("binary", "ladder", "naked_200dma")},
        states, inputs)

    # case study: what did the config say on the Jul 2/3 2026 closes, and
    # would ANY defensible grid configuration have been non-Trending?
    case = {}
    for d in ("2026-07-01", "2026-07-02", "2026-07-06"):
        if d in inputs.index.strftime("%Y-%m-%d").tolist():
            t = inputs.index[inputs.index.strftime("%Y-%m-%d") == d][0]
            row = inputs.loc[t]
            r = compute_regime(row.vix_5d, row.oas, row.breadth_20d,
                               row.spy_vs_200dma)
            entry = {"state": r["state"], "votes": r["votes"],
                     "vix_5d": round(float(row.vix_5d), 2),
                     "oas": round(float(row.oas), 2),
                     "breadth_20d": round(float(row.breadth_20d), 2),
                     "spy_vs_200dma": round(float(row.spy_vs_200dma), 2),
                     "naked_200dma_invested": bool(row.spy_vs_200dma >= 0)}
            # grid sweep: which threshold combos print non-Trending here,
            # and are those combos usable at all (Trending share over the
            # full backtest as the defensibility proxy)?
            dissenters = []
            for vix_on in (16, 17, 18, 19, 20, 21, 22):
                for oas_on in (2.5, 3.0, 3.5, 4.0, 4.5):
                    for band in (0.25, 0.5, 0.75):
                        rr = compute_regime(
                            row.vix_5d, row.oas, row.breadth_20d,
                            row.spy_vs_200dma,
                            vix_thresholds=(vix_on, vix_on + 4.0),
                            oas_thresholds=(oas_on, oas_on + 1.0),
                            breadth_band=band)
                        if rr["state"] != "Risk-on / Trending":
                            st = compute_states(inputs, (vix_on, vix_on + 4.0),
                                                (oas_on, oas_on + 1.0), band)
                            dissenters.append({
                                "combo": [vix_on, oas_on, band],
                                "state": rr["state"],
                                "binary_invested": BINARY_WEIGHTS[rr["state"]] > 0,
                                "ladder_weight": LADDER_WEIGHTS[rr["state"]],
                                "trending_share_pct": round(float(
                                    (st == "Risk-on / Trending").mean()) * 100, 1),
                                "invested_share_pct": round(float(
                                    st.map(BINARY_WEIGHTS).mean()) * 100, 1),
                            })
            entry["grid_dissenters"] = dissenters
            entry["grid_combos_trending"] = 105 - len(dissenters)
            case[d] = entry
    results["case_study_jul_2026"] = case

    if not skip_grid:
        results["sensitivity_grid"] = sensitivity_grid(inputs)

    return results, states, weights, equities, inputs


def _grid_metrics(inputs, states, span):
    w = states.map(BINARY_WEIGHTS)
    sub = inputs.loc[span[0]:span[1]]
    eq, daily = simulate(w.loc[sub.index], sub, fill="next_open")
    yrs = len(sub) / TRADING_DAYS
    ex = daily - sub["cash_daily"].reindex(daily.index).fillna(0.0)
    # a combo that is never invested has identically-zero excess returns:
    # Sharpe is undefined (0/0 = NaN), and NaN keys silently SCRAMBLE
    # sorted() — mark degenerate instead of letting NaN poison rankings
    std = float(ex.std())
    sharpe = (round(float(ex.mean() / std * np.sqrt(252)), 2)
              if std > 0 and np.isfinite(std) else None)
    return {"cagr_pct": round((float(eq.iloc[-1]) ** (1 / yrs) - 1) * 100, 2),
            "max_dd_pct": round(float((eq / eq.cummax() - 1).min()) * 100, 2),
            "sharpe": sharpe,
            "degenerate_never_invested": sharpe is None}


def sensitivity_grid(inputs):
    """EXPERIMENTAL (Decision 4): train 2015-2021, validate 2022-2026,
    binary strategy, both windows always reported. Production config is
    reported unconditionally elsewhere — nothing here feeds back into it."""
    combos = []
    for vix_on in (16, 17, 18, 19, 20, 21, 22):
        for oas_on in (2.5, 3.0, 3.5, 4.0, 4.5):
            for band in (0.25, 0.5, 0.75):
                st = compute_states(inputs, (vix_on, vix_on + 4.0),
                                    (oas_on, oas_on + 1.0), band)
                combos.append({
                    "vix": vix_on, "oas": oas_on, "band": band,
                    "train": _grid_metrics(inputs, st, ("2015-01-01", "2021-12-31")),
                    "validate": _grid_metrics(inputs, st, ("2022-01-01", "2026-12-31")),
                })
    prod = next(c for c in combos
                if c["vix"] == 18 and c["oas"] == 3.0 and c["band"] == 0.5)
    usable = [c for c in combos if c["train"]["sharpe"] is not None
              and c["validate"]["sharpe"] is not None]
    degenerate = [
        {"combo": [c["vix"], c["oas"], c["band"]],
         "train_never_invested": c["train"]["degenerate_never_invested"],
         "validate_never_invested": c["validate"]["degenerate_never_invested"]}
        for c in combos if c not in usable]
    ranked = sorted(usable, key=lambda c: -c["train"]["sharpe"])
    return {"label": "EXPERIMENTAL — trained 2015-2021, validated 2022-2026; "
                     "production thresholds were NOT tuned on this",
            "n_combos": len(combos),
            "degenerate_combos": degenerate,
            "production": prod,
            "best_by_train_sharpe": ranked[0] if ranked else None,
            "top5_by_train_sharpe": ranked[:5]}


def make_charts(states, weights, equities, inputs, img_dir):
    import matplotlib
    matplotlib.use("Agg")
    import matplotlib.pyplot as plt
    os.makedirs(img_dir, exist_ok=True)
    colors = {"Risk-on / Trending": "#3fb950", "Risk-on / Choppy": "#d29922",
              "Caution": "#f0883e", "Risk-off": "#f85149"}

    # 1. equity curves (log)
    fig, ax = plt.subplots(figsize=(12, 6))
    for name, label in [("buy_and_hold", "Buy & hold SPY"),
                        ("naked_200dma", "Naked 200DMA filter"),
                        ("binary", "3-voter binary"),
                        ("ladder", "3-voter graded ladder")]:
        ax.plot(equities[name].index, equities[name].values, label=label,
                linewidth=1.4)
    ax.set_yscale("log")
    ax.set_title("Regime backtest — growth of $1 (T+1-open execution, "
                 "cash at 3mo T-bill)")
    ax.legend(loc="upper left")
    ax.grid(alpha=0.3)
    fig.tight_layout()
    fig.savefig(os.path.join(img_dir, "backtest_equity.png"), dpi=110)
    plt.close(fig)

    # 2. regime ribbon over SPY
    fig, ax = plt.subplots(figsize=(12, 6))
    spy = inputs["spy_close"]
    ax.plot(spy.index, spy.values, color="#111", linewidth=1.0, zorder=3)
    prev, start_t = None, None
    for t, s in list(states.items()) + [(states.index[-1], None)]:
        if s != prev:
            if prev is not None:
                ax.axvspan(start_t, t, color=colors[prev], alpha=0.35,
                           linewidth=0)
            prev, start_t = s, t
    handles = [plt.Rectangle((0, 0), 1, 1, color=colors[s], alpha=0.5)
               for s in STATES]
    ax.legend(handles, STATES, loc="upper left")
    ax.set_yscale("log")
    ax.set_title("SPY with reconstructed regime states (close-T signal)")
    ax.grid(alpha=0.3)
    fig.tight_layout()
    fig.savefig(os.path.join(img_dir, "backtest_regime_ribbon.png"), dpi=110)
    plt.close(fig)

    # 3. drawdown curves
    fig, ax = plt.subplots(figsize=(12, 4.5))
    for name, label in [("buy_and_hold", "Buy & hold"),
                        ("naked_200dma", "200DMA"), ("binary", "Binary"),
                        ("ladder", "Ladder")]:
        eq = equities[name]
        ax.plot(eq.index, (eq / eq.cummax() - 1) * 100, label=label,
                linewidth=1.1)
    ax.set_title("Drawdown (%)")
    ax.legend(loc="lower left")
    ax.grid(alpha=0.3)
    fig.tight_layout()
    fig.savefig(os.path.join(img_dir, "backtest_drawdowns.png"), dpi=110)
    plt.close(fig)


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--start", default="2015-01-01")
    ap.add_argument("--end", default=None)
    ap.add_argument("--skip-grid", action="store_true")
    ap.add_argument("--out-json",
                    default=os.path.join(REPO, "docs",
                                         "backtest-regime-results.json"))
    ap.add_argument("--img-dir", default=os.path.join(REPO, "docs", "img"))
    ap.add_argument("--no-charts", action="store_true")
    args = ap.parse_args()

    results, states, weights, equities, inputs = run(
        start=args.start, end=args.end, skip_grid=args.skip_grid)

    if not args.no_charts:
        make_charts(states, weights, equities, inputs, args.img_dir)

    os.makedirs(os.path.dirname(args.out_json), exist_ok=True)
    with open(args.out_json, "w") as f:
        json.dump(results, f, indent=1, default=str)

    print(json.dumps({k: results["strategies"][k]
                      for k in results["strategies"]}, indent=1))
    print(f"\nstates: {results['state_share']}")
    print(f"results -> {args.out_json}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
