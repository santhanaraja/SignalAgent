#!/usr/bin/env python3
"""
Gauge B parameter-selection campaign (D-008) — ANALYSIS ONLY.

Extends the Build 4 harness (scripts/backtest_regime.py) with the candidate
TREND-CHASSIS gauge and runs the three D-008 sweeps on the D-006 protocol
(train 2015-2021 / validate 2022-2026, BOTH windows always reported, no
post-hoc tuning presented as validation). Touches NO production code path and
changes NO live gauge — compute_regime (the parliament) is imported unchanged
as a control benchmark only.

STEP 0 — the candidate chassis (pure, backtestable; NOT wired to production):
  compute_regime_chassis(trend_state, vix_5d, hy_measure, breadth_20d,
                         hysteresis_state) -> {state, exposure, hysteresis_state}

D-008 rulings implemented:
  Q1  200DMA trend is the CHASSIS (in/out decides direction); VIX + HY +
      breadth are THROTTLE modifiers that scale exposure WITHIN the trend and
      never override it. Chassis state set (proposed here, mapped to the Q4
      rungs): In-Trend-Full / In-Trend-Throttled / Out-Defensive / Out-Risk-off.
  Q3  asymmetric hysteresis — downgrades instant (crash brake), upgrades require
      N consecutive closes above the confirmed level (slow to re-risk).
      Symmetric (both directions require N) runs as the control.
  Q4  regime state -> R28 ceiling 90/50/25/5 (ruled), swept vs 25/15/5/0 and
      binary(chassis).

Lookahead discipline is inherited from build_inputs (every window trails T; OAS
lags one publication day) and preserved here: the HY shapes use trailing
rolling windows only, and the hysteresis replay is strictly causal. Pinned by
test_backtest_gauge_b.py (shift test, walk-window truncation match incl. the
stateful chassis, no series access past T).

Usage:
    python3 scripts/backtest_gauge_b.py                 # full campaign + charts
    python3 scripts/backtest_gauge_b.py --skip-charts
Outputs: docs/gauge-b-campaign-results.json, docs/img/gaugeb_*.png
"""
import argparse
import json
import os
import sys

import numpy as np
import pandas as pd

REPO = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
sys.path.insert(0, REPO)
sys.path.insert(0, os.path.join(REPO, "scripts"))

from backtest_regime import (  # noqa: E402  — reuse Build 4 machinery unchanged
    load_series, build_inputs, simulate, metrics, whipsaws,
    compute_states, BINARY_WEIGHTS, TRADING_DAYS,
)

TRAIN = ("2015-01-01", "2021-12-31")
VALIDATE = ("2022-01-01", "2026-12-31")
NORMAL_BULL = ("2015-01-01", "2023-12-31")   # the 9 years absolute-OAS sat out

# ---- STEP 0: the trend chassis --------------------------------------------

CHASSIS_STATES = ["Out-Risk-off", "Out-Defensive", "In-Trend-Throttled", "In-Trend-Full"]
STATE_RANK = {s: i for i, s in enumerate(CHASSIS_STATES)}   # higher = more risk-on

# Q4 ladders (chassis state -> R28 exposure ceiling)
LADDER_90_50_25_5 = {"In-Trend-Full": 0.90, "In-Trend-Throttled": 0.50,
                     "Out-Defensive": 0.25, "Out-Risk-off": 0.05}
LADDER_25_15_5_0 = {"In-Trend-Full": 0.25, "In-Trend-Throttled": 0.15,
                    "Out-Defensive": 0.05, "Out-Risk-off": 0.0}
LADDER_BINARY = {"In-Trend-Full": 1.0, "In-Trend-Throttled": 1.0,
                 "Out-Defensive": 0.0, "Out-Risk-off": 0.0}
LADDERS = {"90/50/25/5": LADDER_90_50_25_5, "25/15/5/0": LADDER_25_15_5_0,
           "binary(chassis)": LADDER_BINARY}

# Throttle calibration — FIXED across every sweep (the sweeps vary Q2 shape /
# Q3 N / Q4 ladder, never these). Documented in docs/gauge-b-campaign.md; a
# throttle-threshold sweep is a possible follow-up, out of scope here.
VIX_THROTTLE = 20.0        # 5d-avg VIX at/above this is an elevated-vol throttle
BREADTH_THROTTLE = 0.0     # breadth_20d below this (RSP lagging SPY) is a throttle


def _raw_chassis_state(trend_in, vix_5d, hy_stress, breadth_20d, *,
                       vix_thr=VIX_THROTTLE, breadth_thr=BREADTH_THROTTLE,
                       require_k=1):
    """Direction from the 200DMA chassis; throttles scale WITHIN it, never
    override. In-trend: Throttled once >= require_k throttles fire (require_k=1
    is the campaign's any-one-throttle rule), else Full. Out-of-trend: defensive
    by default, Risk-off on a real stress trigger (elevated vol OR credit
    stress). vix_thr / breadth_thr / require_k are the throttle-calibration
    knobs swept in the throttle follow-up; the DEFAULTS reproduce the campaign."""
    vix_stress = vix_5d >= vix_thr
    breadth_stress = breadth_20d < breadth_thr
    if trend_in:
        stress = int(vix_stress) + int(hy_stress) + int(breadth_stress)
        return "In-Trend-Full" if stress < require_k else "In-Trend-Throttled"
    return "Out-Risk-off" if (vix_stress or hy_stress) else "Out-Defensive"


def new_hysteresis_state(state="Out-Defensive"):
    """Seed carry for the causal replay."""
    return {"confirmed": state, "up": 0, "down": 0}


def compute_regime_chassis(trend_state, vix_5d, hy_measure, breadth_20d,
                           hysteresis_state, *, n=2, mode="asymmetric",
                           ladder=None, hy_stress=None, throttle=None):
    """One close-T step of the candidate gauge (D-008). Returns the confirmed
    state, its exposure under `ladder`, and the updated hysteresis carry.

    trend_state : SPY %-above-200DMA (in-trend iff >= 0) — the chassis.
    hy_measure  : the Q2 credit signal (unused directly when hy_stress is given;
                  kept in the signature for the ruled interface / a future Lab).
    hy_stress   : bool — is credit stressed? (the harness derives it per Q2 shape
                  so the chassis stays shape-agnostic; falls back to hy_measure>0.)
    n, mode     : Q3 hysteresis (asymmetric downgrade-instant / symmetric control).
    ladder      : Q4 exposure map (default 90/50/25/5).
    """
    ladder = ladder or LADDER_90_50_25_5
    if hy_stress is None:
        hy_stress = bool(hy_measure and hy_measure > 0)
    raw = _raw_chassis_state(trend_state >= 0, vix_5d, hy_stress, breadth_20d,
                             **(throttle or {}))
    raw_rank = STATE_RANK[raw]

    h = dict(hysteresis_state)
    cr = STATE_RANK[h["confirmed"]]
    if raw_rank > cr:                       # candidate upgrade — slow to re-risk
        h["up"] += 1
        h["down"] = 0
        if h["up"] >= n:
            h["confirmed"] = raw
            h["up"] = 0
    elif raw_rank < cr:                     # downgrade
        h["down"] += 1
        h["up"] = 0
        if mode == "asymmetric":            # instant crash brake (jump to raw)
            h["confirmed"] = raw
            h["down"] = 0
        elif h["down"] >= n:                # symmetric control: confirm the drop
            h["confirmed"] = raw
            h["down"] = 0
    else:
        h["up"] = h["down"] = 0
    return {"state": h["confirmed"], "exposure": ladder[h["confirmed"]],
            "hysteresis_state": h}


def replay_states(inputs, hy_stress, *, n=2, mode="asymmetric", throttle=None):
    """Causal replay of the chassis over the input frame -> confirmed-state
    Series. hy_stress is a boolean Series aligned to inputs.index. throttle is
    the optional throttle-calibration config (defaults reproduce the campaign)."""
    hstate = new_hysteresis_state()
    out = []
    trend = inputs["spy_vs_200dma"].values
    vix = inputs["vix_5d"].values
    breadth = inputs["breadth_20d"].values
    hs = hy_stress.reindex(inputs.index).fillna(False).values
    for i in range(len(inputs)):
        r = compute_regime_chassis(trend[i], vix[i], None, breadth[i], hstate,
                                   n=n, mode=mode, hy_stress=bool(hs[i]),
                                   throttle=throttle)
        hstate = r["hysteresis_state"]
        out.append(r["state"])
    return pd.Series(out, index=inputs.index, name="state")


def chassis_durations(states):
    """Per-chassis-state run stats (duration_stats in backtest_regime keys off
    the parliament's state names, so it can't see the chassis states)."""
    runs, cur, n = [], None, 0
    for s in states:
        if s == cur:
            n += 1
        else:
            if cur is not None:
                runs.append((cur, n))
            cur, n = s, 1
    runs.append((cur, n))
    out = {}
    for st in CHASSIS_STATES:
        lens = [k for s, k in runs if s == st]
        if lens:
            out[st] = {"runs": len(lens), "median_days": float(np.median(lens)),
                       "mean_days": round(float(np.mean(lens)), 1),
                       "max_days": int(max(lens)),
                       "runs_le_2d": sum(1 for x in lens if x <= 2)}
    return out


# ---- Q2 credit-measure shapes ---------------------------------------------
# Each returns a boolean "credit stressed" Series from the (already 1-day-
# lagged) OAS in build_inputs, using TRAILING windows only. Higher OAS = wider
# spread = more stress. Thresholds are part of each shape's definition.

HY_PCT_THRESHOLD = 80.0    # stress when OAS is in the top fifth of its window
HY_Z_THRESHOLD = 1.0       # stress at >= 1 sigma wide


def _rolling_pctile_of_last(s, window):
    """For each day, the percentile (0-100) of today's value within the
    trailing `window` (inclusive). Trailing-only -> point-in-time."""
    return s.rolling(window, min_periods=window).apply(
        lambda a: (a <= a[-1]).mean() * 100.0, raw=True)


def hy_shapes(oas):
    shapes = {}
    for w in (60, 252, 504):
        shapes[f"pctile_{w}d"] = _rolling_pctile_of_last(oas, w) >= HY_PCT_THRESHOLD
    for w in (252, 504):
        mean = oas.rolling(w, min_periods=w).mean()
        std = oas.rolling(w, min_periods=w).std()
        shapes[f"zscore_{w}d"] = ((oas - mean) / std) >= HY_Z_THRESHOLD
    shapes["direction_20d"] = (oas - oas.shift(20)) > 0        # widening
    return shapes


def warmed_hy_shapes(raw, index):
    """The Q2 shapes computed on the FULL lagged-OAS history (pre-window
    warmup, exactly like build_inputs pre-warms the 200DMA), then aligned to
    `index`. Without this the long 252/504d windows sit un-warmed for their
    first ~2 years and cannot signal stress through 2015-16 — an unfair handicap
    in the Q2 train comparison (adversarial-review finding). Still strictly
    causal: the rolling windows only ever look backward."""
    spy_idx = raw["SPY"]["Close"].index
    oas_full = (raw["OAS"].iloc[:, 0].sort_index()
                .reindex(spy_idx, method="ffill").shift(1))
    return {k: v.reindex(index) for k, v in hy_shapes(oas_full).items()}


# ---- metrics helpers -------------------------------------------------------

def span_metrics(weights, inputs, span):
    """Windowed metrics via the Build 4 simulate/metrics, for a weight Series."""
    sub = inputs.loc[span[0]:span[1]]
    if len(sub) < 30:
        return None
    w = weights.reindex(sub.index)
    eq, daily = simulate(w, sub, fill="next_open")
    yrs = len(sub) / TRADING_DAYS
    m = metrics(eq, daily, w, sub["cash_daily"], yrs)
    m["whipsaws_per_year"] = round(len(whipsaws(w)) / yrs, 1)
    # Standard Sortino (review finding): excess-over-cash numerator matching the
    # Sharpe basis, denominator = downside deviation about the 0 target
    # (RMS of the negative excess returns), guarded for the no-downside case.
    ex = daily - sub["cash_daily"].reindex(daily.index).fillna(0.0)
    downside = float(np.sqrt((np.minimum(ex, 0.0) ** 2).mean()))
    m["sortino"] = (round(float(ex.mean() / downside * np.sqrt(TRADING_DAYS)), 2)
                    if downside > 0 else None)
    return m


def benchmark_weights(inputs, states_parliament):
    """The three Build 4 benchmarks, recomputed here for internal consistency."""
    return {
        "buy_and_hold": pd.Series(1.0, index=inputs.index),
        "naked_200dma": (inputs["spy_vs_200dma"] >= 0).astype(float),
        "parliament_binary": states_parliament.map(BINARY_WEIGHTS),
    }


def bench_row(inputs, states_parliament):
    b = benchmark_weights(inputs, states_parliament)
    out = {}
    for name, w in b.items():
        out[name] = {"train": span_metrics(w, inputs, TRAIN),
                     "validate": span_metrics(w, inputs, VALIDATE),
                     "full": span_metrics(w, inputs, (str(inputs.index[0].date()),
                                                      str(inputs.index[-1].date())))}
    return out


# ---- the three sweeps ------------------------------------------------------

def sweep_q2(inputs, shapes, *, n=2, mode="asymmetric", ladder_name="90/50/25/5"):
    """Credit-measure bakeoff: each shape through the chassis at a fixed
    N/ladder. The headline question — does it stay invested through the
    2015-2023 normal-bull stretch AND defend in real stress — on BOTH windows."""
    ladder = LADDERS[ladder_name]
    rows = {}
    for shape_name, stress in shapes.items():
        states = replay_states(inputs, stress, n=n, mode=mode)
        w = states.map(ladder)
        rows[shape_name] = {
            "stress_days_pct": round(float(stress.reindex(inputs.index)
                                           .fillna(False).mean()) * 100, 1),
            "train": span_metrics(w, inputs, TRAIN),
            "validate": span_metrics(w, inputs, VALIDATE),
            "full": span_metrics(w, inputs, (str(inputs.index[0].date()),
                                             str(inputs.index[-1].date()))),
            # regime-adaptive test: avg exposure through the 2015-2023 normal
            # bull (high = stays invested, unlike absolute-OAS which sat out),
            # and through the two real credit stresses (low = defends)
            "normal_bull_avg_exposure_pct": span_metrics(w, inputs, NORMAL_BULL)["avg_exposure_pct"],
            "covid_avg_exposure_pct": span_metrics(w, inputs, ("2020-02-19", "2020-04-30"))["avg_exposure_pct"],
            "bear2022_avg_exposure_pct": span_metrics(w, inputs, ("2022-01-01", "2022-10-31"))["avg_exposure_pct"],
        }
    return {"config": {"n": n, "mode": mode, "ladder": ladder_name,
                       "pct_threshold": HY_PCT_THRESHOLD, "z_threshold": HY_Z_THRESHOLD},
            "shapes": rows}


def sweep_q3(inputs, best_stress, *, ladder_name="90/50/25/5", ns=(1, 2, 3, 5)):
    """Hysteresis N against the best Q2 shape: asymmetric (ruled) + symmetric
    (control). Reports whipsaws/yr and the In-Trend-Full flicker (runs<=2d)."""
    ladder = LADDERS[ladder_name]
    rows = {}
    for mode in ("asymmetric", "symmetric"):
        for n in ns:
            states = replay_states(inputs, best_stress, n=n, mode=mode)
            w = states.map(ladder)
            dur = chassis_durations(states)
            full = dur.get("In-Trend-Full", {})
            rows[f"{mode}_N{n}"] = {
                "mode": mode, "n": n,
                "train": span_metrics(w, inputs, TRAIN),
                "validate": span_metrics(w, inputs, VALIDATE),
                "full": span_metrics(w, inputs, (str(inputs.index[0].date()),
                                                 str(inputs.index[-1].date()))),
                "in_trend_full_runs": full.get("runs"),
                "in_trend_full_runs_le_2d": full.get("runs_le_2d"),
                "in_trend_full_median_days": full.get("median_days"),
            }
    return {"config": {"ladder": ladder_name}, "variants": rows}


def sweep_q4(inputs, best_stress, *, n=2, mode="asymmetric"):
    """Ladder comparison at the best Q2+Q3 config: 90/50/25/5 vs 25/15/5/0 vs
    binary(chassis) on the SAME chassis states."""
    states = replay_states(inputs, best_stress, n=n, mode=mode)
    rows = {}
    for lname, lad in LADDERS.items():
        w = states.map(lad)
        rows[lname] = {"train": span_metrics(w, inputs, TRAIN),
                       "validate": span_metrics(w, inputs, VALIDATE),
                       "full": span_metrics(w, inputs, (str(inputs.index[0].date()),
                                                        str(inputs.index[-1].date())))}
    return {"config": {"n": n, "mode": mode}, "ladders": rows}


# ---- throttle-calibration sweep (D-008 follow-up) --------------------------
# The campaign verdict: the trend chassis roughly ties the 200DMA at the
# direction level (binary(chassis) 10.13%), but the 90/50/25/5 throttle ladder
# trades ~2.5pp CAGR for drawdown protection. This sweep maps that tradeoff —
# can a LOOSER throttle recover CAGR toward the 200DMA's 9.83% while holding
# sub-15% maxDD? Fixed winning config (pctile_60d shape, asym N=2, 90/50/25/5);
# vary only the In-Trend Full->Throttled cut-points.

THROTTLE_GRID = {
    "vix_thr": [18.0, 20.0, 22.0, 25.0, 30.0],   # higher = looser (30 ~ off)
    "hy_pct_thr": [70.0, 80.0, 90.0, 95.0],       # higher = looser
    "breadth_thr": [0.0, -0.5, -1.0, -2.0],       # lower = looser
    "require_k": [1, 2, 3],                        # throttles needed to downgrade
}


def warmed_pctile_value(raw, index, window=60):
    """The raw pctile_60d percentile series (0-100, not thresholded), pre-warmed
    on the full OAS history — the throttle sweep re-thresholds the HY cut-point
    off this without recomputing the window."""
    spy_idx = raw["SPY"]["Close"].index
    oas_full = (raw["OAS"].iloc[:, 0].sort_index()
                .reindex(spy_idx, method="ffill").shift(1))
    return _rolling_pctile_of_last(oas_full, window).reindex(index)


def sweep_throttle(inputs, pctile_value):
    """Grid over the throttle cut-points at the fixed winning config. Returns one
    row per combo with train/validate/full metrics."""
    import itertools
    rows = []
    for vix_thr, hy_pct, breadth_thr, k in itertools.product(
            THROTTLE_GRID["vix_thr"], THROTTLE_GRID["hy_pct_thr"],
            THROTTLE_GRID["breadth_thr"], THROTTLE_GRID["require_k"]):
        hy_stress = pctile_value >= hy_pct
        throttle = {"vix_thr": vix_thr, "breadth_thr": breadth_thr, "require_k": k}
        w = replay_states(inputs, hy_stress, n=2, mode="asymmetric",
                          throttle=throttle).map(LADDER_90_50_25_5)
        rows.append({
            "vix_thr": vix_thr, "hy_pct_thr": hy_pct, "breadth_thr": breadth_thr,
            "require_k": k,
            "train": span_metrics(w, inputs, TRAIN),
            "validate": span_metrics(w, inputs, VALIDATE),
            "full": span_metrics(w, inputs, (str(inputs.index[0].date()),
                                             str(inputs.index[-1].date()))),
        })
    return rows


def frontier(rows, window="full"):
    """Pareto-efficient points on (higher CAGR, shallower maxDD — maxDD is
    negative, so larger is shallower). A point is on the frontier if no other
    point beats it on both."""
    pts = [(r[window]["cagr_pct"], r[window]["max_drawdown_pct"], r) for r in rows]
    front = []
    for cagr, dd, r in pts:
        dominated = any(
            (oc >= cagr and od >= dd) and (oc > cagr or od > dd)
            for oc, od, o in pts if o is not r)
        if not dominated:
            front.append(r)
    return sorted(front, key=lambda r: r[window]["max_drawdown_pct"])


# ---- orchestration ---------------------------------------------------------

def _pick_best_shape(q2):
    """The winning credit shape, chosen on TRAIN Sharpe (D-006: select in-
    sample, report validate out-of-sample — never mine the validate window).
    The chosen shape's validate rank is reported alongside so the reader can
    see it isn't a train-only overfit. `leads_both` flags whether it also tops
    the validate window (the robust case)."""
    items = list(q2["shapes"].items())
    def tsharpe(kv): return kv[1]["train"]["sharpe"]
    def vsharpe(kv): return kv[1]["validate"]["sharpe"]
    best = max(items, key=tsharpe)
    leads_both = best[0] == max(items, key=vsharpe)[0]
    return best[0], leads_both


def run(start="2015-01-01", end=None, default_n=2, default_mode="asymmetric",
        default_ladder="90/50/25/5"):
    raw = load_series()
    end = end or str(raw["SPY"].index[-1].date())
    inputs = build_inputs(raw, start, end)
    states_parliament = compute_states(inputs)   # control (production, unchanged)

    shapes = warmed_hy_shapes(raw, inputs.index)  # pre-warmed (review F3)

    benchmarks = bench_row(inputs, states_parliament)
    q2 = sweep_q2(inputs, shapes, n=default_n, mode=default_mode,
                  ladder_name=default_ladder)
    best_shape, leads_both = _pick_best_shape(q2)
    best_stress = shapes[best_shape]

    q3 = sweep_q3(inputs, best_stress, ladder_name=default_ladder)
    q4 = sweep_q4(inputs, best_stress, n=default_n, mode=default_mode)

    results = {
        "meta": {
            "window": {"start": str(inputs.index[0].date()),
                       "end": str(inputs.index[-1].date()),
                       "trading_days": len(inputs)},
            "split": {"train": list(TRAIN), "validate": list(VALIDATE),
                      "normal_bull": list(NORMAL_BULL)},
            "chassis_states": CHASSIS_STATES,
            "throttles": {"vix_throttle": VIX_THROTTLE,
                          "breadth_throttle": BREADTH_THROTTLE},
            "defaults": {"n": default_n, "mode": default_mode,
                         "ladder": default_ladder},
            "bar": "beat naked_200dma (Build 4: 9.83% full-window CAGR)",
            "analysis_only": True,
        },
        "benchmarks": benchmarks,
        "sweep_q2_credit_shape": q2,
        "selected_shape": {"shape": best_shape,
                           "leads_both_windows": leads_both},
        "sweep_q3_hysteresis": q3,
        "sweep_q4_ladder": q4,
    }
    return results, inputs, states_parliament, shapes, best_shape


def make_charts(results, inputs, best_stress, states_parliament, img_dir,
                default_n=2, default_mode="asymmetric"):
    import matplotlib
    matplotlib.use("Agg")
    import matplotlib.pyplot as plt
    os.makedirs(img_dir, exist_ok=True)

    # recompute the recommended-config equity vs the three benchmarks
    states = replay_states(inputs, best_stress, n=default_n, mode=default_mode)
    gb = states.map(LADDER_90_50_25_5)
    curves = {
        "Buy & hold SPY": pd.Series(1.0, index=inputs.index),
        "Naked 200DMA": (inputs["spy_vs_200dma"] >= 0).astype(float),
        "Parliament binary (old 3-voter)": states_parliament.map(BINARY_WEIGHTS),
        "Gauge B (chassis, 90/50/25/5)": gb,
    }
    eqs = {}
    for name, w in curves.items():
        eq, _ = simulate(w, inputs, fill="next_open")
        eqs[name] = eq

    fig, ax = plt.subplots(figsize=(12, 6))
    for name, eq in eqs.items():
        ax.plot(eq.index, eq.values, label=name, linewidth=1.4)
    ax.set_yscale("log")
    ax.set_title("Gauge B campaign — growth of $1 vs Build 4 benchmarks "
                 "(T+1-open execution)")
    ax.axvline(pd.Timestamp(VALIDATE[0]), color="#888", linestyle="--",
               linewidth=1, label="train | validate")
    ax.legend(loc="upper left")
    ax.grid(alpha=0.3)
    fig.tight_layout()
    fig.savefig(os.path.join(img_dir, "gaugeb_equity.png"), dpi=110)
    plt.close(fig)

    fig, ax = plt.subplots(figsize=(12, 4.5))
    for name, eq in eqs.items():
        ax.plot(eq.index, (eq / eq.cummax() - 1) * 100, label=name, linewidth=1.1)
    ax.set_title("Gauge B campaign — drawdown (%)")
    ax.legend(loc="lower left")
    ax.grid(alpha=0.3)
    fig.tight_layout()
    fig.savefig(os.path.join(img_dir, "gaugeb_drawdowns.png"), dpi=110)
    plt.close(fig)

    # chassis ribbon over SPY
    colors = {"In-Trend-Full": "#3fb950", "In-Trend-Throttled": "#d29922",
              "Out-Defensive": "#f0883e", "Out-Risk-off": "#f85149"}
    fig, ax = plt.subplots(figsize=(12, 6))
    spy = inputs["spy_close"]
    ax.plot(spy.index, spy.values, color="#111", linewidth=1.0, zorder=3)
    prev, start_t = None, None
    for t, s in list(states.items()) + [(states.index[-1], None)]:
        if s != prev:
            if prev is not None:
                ax.axvspan(start_t, t, color=colors[prev], alpha=0.35, linewidth=0)
            prev, start_t = s, t
    handles = [plt.Rectangle((0, 0), 1, 1, color=colors[s], alpha=0.5)
               for s in CHASSIS_STATES]
    ax.legend(handles, CHASSIS_STATES, loc="upper left")
    ax.set_yscale("log")
    ax.set_title("SPY with Gauge B chassis states (recommended config)")
    ax.grid(alpha=0.3)
    fig.tight_layout()
    fig.savefig(os.path.join(img_dir, "gaugeb_chassis_ribbon.png"), dpi=110)
    plt.close(fig)


def run_throttle_sweep(start="2015-01-01", end=None):
    raw = load_series()
    end = end or str(raw["SPY"].index[-1].date())
    inputs = build_inputs(raw, start, end)
    benchmarks = bench_row(inputs, compute_states(inputs))
    pctile = warmed_pctile_value(raw, inputs.index)
    rows = sweep_throttle(inputs, pctile)
    front = frontier(rows, "full")
    base = next(r for r in rows if r["vix_thr"] == 20.0 and r["hy_pct_thr"] == 80.0
                and r["breadth_thr"] == 0.0 and r["require_k"] == 1)
    target = sorted((r for r in rows if r["full"]["cagr_pct"] >= 9.0
                     and r["full"]["max_drawdown_pct"] > -15.0),
                    key=lambda r: -r["full"]["cagr_pct"])
    results = {
        "meta": {"window": {"start": str(inputs.index[0].date()),
                            "end": str(inputs.index[-1].date())},
                 "fixed_config": "pctile_60d shape, asym N=2, 90/50/25/5 ladder",
                 "swept": "In-Trend Full->Throttled cut-points (vix/hy-pctile/breadth/require_k)",
                 "grid": THROTTLE_GRID, "n_combos": len(rows),
                 "question": "can a looser throttle recover full CAGR toward the "
                             "200DMA's 9.83% while holding maxDD shallower than -15%?",
                 "bar": {"cagr_200dma": benchmarks["naked_200dma"]["full"]["cagr_pct"],
                         "maxdd_ceiling": -15.0},
                 "analysis_only": True},
        "benchmarks": benchmarks,
        "campaign_base_config": base,
        "frontier_full": front,
        "target_zone_cagr_ge_9_dd_gt_-15": target,
        "all_combos": rows,
    }
    return results, inputs, rows, front, base, benchmarks


def make_throttle_chart(rows, front, base, benchmarks, img_dir):
    import matplotlib
    matplotlib.use("Agg")
    import matplotlib.pyplot as plt
    os.makedirs(img_dir, exist_ok=True)
    fig, ax = plt.subplots(figsize=(11, 7))
    ax.scatter([r["full"]["max_drawdown_pct"] for r in rows],
               [r["full"]["cagr_pct"] for r in rows],
               c="#8b949e", s=18, alpha=0.5, label=f"throttle combos ({len(rows)})")
    ax.plot([r["full"]["max_drawdown_pct"] for r in front],
            [r["full"]["cagr_pct"] for r in front], "-o", color="#58a6ff",
            label="Pareto frontier", zorder=4)
    ax.scatter([base["full"]["max_drawdown_pct"]], [base["full"]["cagr_pct"]],
               marker="*", s=280, color="#3fb950", zorder=5,
               label="campaign base (k=1, vix20/hy80/br0)")
    for name, mk, col in [("naked_200dma", "D", "#f0883e"),
                          ("buy_and_hold", "D", "#f85149")]:
        m = benchmarks[name]["full"]
        ax.scatter([m["max_drawdown_pct"]], [m["cagr_pct"]], marker=mk, s=90,
                   color=col, zorder=5, label=name.replace("_", " "))
    ax.axhline(benchmarks["naked_200dma"]["full"]["cagr_pct"], color="#f0883e",
               ls="--", lw=1, alpha=0.6)
    ax.axvline(-15.0, color="#888", ls=":", lw=1, alpha=0.7)
    ax.set_xlabel("max drawdown %  (shallower →)")
    ax.set_ylabel("full-window CAGR %")
    ax.set_title("Gauge B throttle frontier — CAGR vs drawdown\n"
                 "(fixed pctile_60d / asym N=2 / 90-50-25-5; sweeping the throttle cut-points)")
    ax.legend(loc="lower left")
    ax.grid(alpha=0.3)
    fig.tight_layout()
    fig.savefig(os.path.join(img_dir, "gaugeb_throttle_frontier.png"), dpi=110)
    plt.close(fig)


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--start", default="2015-01-01")
    ap.add_argument("--end", default=None)
    ap.add_argument("--out-json",
                    default=os.path.join(REPO, "docs", "gauge-b-campaign-results.json"))
    ap.add_argument("--img-dir", default=os.path.join(REPO, "docs", "img"))
    ap.add_argument("--skip-charts", action="store_true")
    ap.add_argument("--throttle-sweep", action="store_true",
                    help="run the throttle-calibration frontier sweep instead")
    args = ap.parse_args()

    if args.throttle_sweep:
        results, inputs, rows, front, base, benchmarks = run_throttle_sweep(
            start=args.start, end=args.end)
        if not args.skip_charts:
            make_throttle_chart(rows, front, base, benchmarks, args.img_dir)
        out = os.path.join(REPO, "docs", "gauge-b-throttle-results.json")
        with open(out, "w") as f:
            json.dump(results, f, indent=1, default=str)
        b200 = benchmarks["naked_200dma"]["full"]
        print(f"Throttle sweep: {len(rows)} combos. 200DMA bar {b200['cagr_pct']}% "
              f"CAGR / {b200['max_drawdown_pct']}% maxDD.")
        print(f"  base (k=1): {base['full']['cagr_pct']}% / {base['full']['max_drawdown_pct']}%")
        print("  Pareto frontier (full):")
        for r in front:
            f_ = r["full"]
            print(f"    k={r['require_k']} vix{r['vix_thr']:.0f} hy{r['hy_pct_thr']:.0f} "
                  f"br{r['breadth_thr']:+.1f} | CAGR {f_['cagr_pct']:>5} maxDD "
                  f"{f_['max_drawdown_pct']:>6} Sh {f_['sharpe']} val {r['validate']['cagr_pct']}")
        print(f"  target zone (CAGR>=9, maxDD>-15): {len(results['target_zone_cagr_ge_9_dd_gt_-15'])} combos")
        print(f"  -> {out}")
        return 0

    results, inputs, states_parliament, shapes, best_shape = run(
        start=args.start, end=args.end)

    if not args.skip_charts:
        make_charts(results, inputs, shapes[best_shape], states_parliament,
                    args.img_dir)

    os.makedirs(os.path.dirname(args.out_json), exist_ok=True)
    with open(args.out_json, "w") as f:
        json.dump(results, f, indent=1, default=str)

    b = results["benchmarks"]
    print("Gauge B campaign complete.")
    print(f"  window {results['meta']['window']['start']}..{results['meta']['window']['end']}")
    print(f"  benchmarks full CAGR: B&H {b['buy_and_hold']['full']['cagr_pct']}%  "
          f"200DMA {b['naked_200dma']['full']['cagr_pct']}%  "
          f"parliament {b['parliament_binary']['full']['cagr_pct']}%")
    print(f"  selected credit shape: {best_shape} "
          f"(leads both windows: {results['selected_shape']['leads_both_windows']})")
    print(f"  results -> {args.out_json}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
