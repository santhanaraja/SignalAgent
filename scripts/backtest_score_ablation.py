#!/usr/bin/env python3
"""
Build 5.2 — score component ablation (pre-registered:
docs/backtest-score-ablation-prereg.md, committed cdcfa19 BEFORE this
script existed).

5.1 ablated the grade's rows; this ablates the SCORE's five ladders.
For each component C in {rsi, macd, ma, ytd, vol}, the score is
recomputed with C zeroed (through the clip, not subtracted from the
clipped total) and the grade flips are isolated among ticker-days that
pass every other row:
  LOST_C   — A+ only because of C's points
  GAINED_C — denied A+ by C's points alone

Plus the three pre-named structures: (a) the MACD 16-point cliff and
its smoothed replacement, (b) the MA component's ±6 path split, (c)
YTD saturation — buckets, the v2 backcast, the uncapped monotone
ladder.

The score under test is the AS-LIVED score (ytd_v1 + penalty as ONE
component). Ships nothing. Signal-level only. No threshold sweeps.

Integrity pins, all before any statistic is trusted:
  1. joint AND of recomputed rows == committed frame's A+ set EXACTLY
  2. per-component decomposition recomposes elementwise to
     score_components_vec's totals (with AND without penalty)
  3. sign of the raw MACD gap reproduces macd_windowed's bullish flag
  4. sampled elementwise parity of each vectorized ladder against the
     production score_*_points functions

Run: python3 scripts/backtest_score_ablation.py [--out PATH]
"""

import argparse
import hashlib
import json
import os
import subprocess
import sys

REPO = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
sys.path.insert(0, REPO)
sys.path.insert(0, os.path.join(REPO, "scripts"))

import numpy as np
import pandas as pd
from numpy.lib.stride_tricks import sliding_window_view

from backtest_doctrine import (rsi_windowed, macd_windowed,
                               score_components_vec, runway_arrays,
                               _ema_windowed, FRAME, KNOBS, CACHE,
                               PRICES, EARNINGS_PATH, REGIME_PATH,
                               START, TRAIN_END, up_close_windowed,
                               ytd_aslived)
from backtest_ablation import dist_block, cluster_ci

FRAME_PATH = os.path.join(CACHE, "master_frame.csv.gz")
ROWS = ("c1", "c2", "c3", "c4", "r2", "r3", "r4", "r5", "r7")
OTHERS = tuple(x for x in ROWS if x != "r5")
IN_TREND = ("In-Trend-Full", "In-Trend-Throttled")
COMPONENTS = ("rsi", "macd", "ma", "ytd", "vol")
SCORE_BASE = 50
SCORE_MIN = KNOBS["score_min"]
BANDS = (0.001, 0.0025, 0.005, 0.01)      # of close; 1.0% is primary


# ------------------------------------------------------- ladders (vector)
def comp_vectors(rsi, bullish, confirms, above20, above50, ma20_gt_50,
                 ytd, vol_ratio):
    """Each ladder separately — must recompose to score_components_vec
    exactly (pin 2). ytd component = v1 rungs + penalty JOINTLY."""
    r = np.select([rsi < 30, rsi < 40, rsi < 60, rsi < 70, rsi < 80],
                  [15, 8, 3, -3, -8], default=-15)
    m = np.where(bullish, 8 + np.where(confirms, 5, 0),
                 -8 - np.where(confirms, 5, 0))
    ma = (np.where(above20, 4, -4) + np.where(above50, 6, -6)
          + np.where(ma20_gt_50, 4, -4))
    y_v1 = np.select([ytd > 50, ytd > 20, ytd > 5, ytd > 0, ytd > -10],
                     [8, 12, 6, 2, -4], default=-10)
    pen = np.select([ytd > 150, ytd > 100], [-15, -10], default=0)
    v = np.select([vol_ratio > 1.5, vol_ratio < 0.7], [3, -3], default=0)
    return {"rsi": r.astype(float), "macd": m.astype(float),
            "ma": ma.astype(float), "ytd": (y_v1 + pen).astype(float),
            "ytd_v1_only": y_v1.astype(float), "pen": pen.astype(float),
            "vol": v.astype(float)}


def ytd_v2_vec(ytd):
    return np.select([ytd > 20, ytd > 5, ytd > 0, ytd > -10],
                     [12, 6, 2, -4], default=-10).astype(float)


def ytd_uncapped_vec(ytd):
    """Prereg (c)3: v2 below 20; above, 12 + 4 per bracket crossed at
    50/100/150. Never decreases."""
    return np.select([ytd > 150, ytd > 100, ytd > 50, ytd > 20,
                      ytd > 5, ytd > 0, ytd > -10],
                     [24, 20, 16, 12, 6, 2, -4], default=-10) \
        .astype(float)


def clip_score(total):
    return np.clip(total, 0, 100)


def ladder_parity_pin(rng_seed=20260814, n=4000):
    """Pin 4: sampled elementwise parity of each vectorized ladder
    against the production functions."""
    from signal_engine import (score_rsi_points, score_macd_points,
                               score_ma_points, score_ytd_points_v1,
                               score_ytd_points_v2, score_vol_points)
    rng = np.random.default_rng(rng_seed)
    rsi = rng.uniform(0, 100, n)
    ytd = rng.uniform(-80, 400, n)
    vr = rng.uniform(0, 3, n)
    b = rng.random(n) > 0.5
    cf = rng.random(n) > 0.5
    a20 = rng.random(n) > 0.5
    a50 = rng.random(n) > 0.5
    mgt = rng.random(n) > 0.5
    c = comp_vectors(rsi, b, cf, a20, a50, mgt, ytd, vr)
    for i in range(n):
        assert c["rsi"][i] == score_rsi_points(rsi[i]), (i, rsi[i])
        assert c["macd"][i] == score_macd_points(bool(b[i]),
                                                 bool(cf[i])), i
        assert c["ma"][i] == score_ma_points(bool(a20[i]), bool(a50[i]),
                                             bool(mgt[i])), i
        assert c["vol"][i] == score_vol_points(vr[i]), (i, vr[i])
        v1 = score_ytd_points_v1(ytd[i])
        assert c["ytd_v1_only"][i] + c["pen"][i] == v1, (i, ytd[i], v1)
        assert ytd_v2_vec(np.array([ytd[i]]))[0] == \
            score_ytd_points_v2(ytd[i]), (i, ytd[i])
    print(f"ladder parity: {n} sampled points x 5 ladders "
          "elementwise EXACT")


# ------------------------------------------------------------ per ticker
def macd_gap_windowed(close, frame=FRAME):
    """Raw (macd - signal) at the frame end — same construction as
    macd_windowed, exposing the gap for the cliff probe (pin 3 asserts
    sign parity)."""
    F = sliding_window_view(close, frame)
    macd = _ema_windowed(F, 12) - _ema_windowed(F, 26)
    signal = _ema_windowed(macd, 9)
    return (macd[:, -1] - signal[:, -1])


def ticker_panel(t, path, edata, regime_by_day, end):
    """Per grading day: row flags + component vectors + probe inputs +
    fwd20. Mirrors 5.1's ticker_flags; extends it with the component
    decomposition, the raw MACD gap, ATR and close."""
    df = pd.read_csv(path, index_col=0, parse_dates=True)
    if len(df) < FRAME + 1:
        return None
    close = df["Close"].to_numpy(dtype=float)
    high = df["High"].to_numpy(dtype=float)
    low = df["Low"].to_numpy(dtype=float)
    open_ = df["Open"].to_numpy(dtype=float)
    idx = df.index
    n = len(df)

    c = pd.Series(close, index=idx)
    sma5 = c.rolling(5).mean().to_numpy()
    sma20 = c.rolling(20).mean().to_numpy()
    sma20_then = pd.Series(sma20, index=idx).shift(
        KNOBS["slope_lookback"]).to_numpy()
    sma50 = c.rolling(50).mean().to_numpy()
    prev = np.concatenate([[np.nan], close[:-1]])
    tr = np.maximum(high - low, np.maximum(np.abs(high - prev),
                                           np.abs(low - prev)))
    atr14 = pd.Series(tr, index=idx).rolling(KNOBS["atr_period"]).mean() \
        .to_numpy()
    above = close > sma20
    _ab = pd.Series(above, index=idx)
    consec = _ab.astype(int).groupby((~_ab).cumsum()).cumsum().to_numpy()
    vol = df["Volume"].to_numpy(dtype=float)
    vol5 = pd.Series(vol, index=idx).rolling(5).mean().to_numpy()
    vol20 = pd.Series(vol, index=idx).rolling(20).mean().to_numpy()
    with np.errstate(divide="ignore", invalid="ignore"):
        vratio = np.round(np.where(vol20 > 0, vol5 / vol20, 1.0), 2)

    off = FRAME - 1
    rsi = rsi_windowed(close)
    bullish, confirms = macd_windowed(close)
    gap = macd_gap_windowed(close)
    # pin 3: the gap's sign must reproduce bullish exactly
    assert np.array_equal(gap > 0, bullish), \
        f"{t}: macd gap sign disagrees with macd_windowed bullish"
    up20 = np.concatenate([np.zeros(19, dtype=bool),
                           up_close_windowed(close, 20)])
    ytd = ytd_aslived(close, idx)

    fwd = np.full(n, np.nan)
    for i in range(n - 21):
        a, b = open_[i + 1], open_[i + 21]
        if np.isfinite(a) and np.isfinite(b) and a > 0:
            fwd[i] = (b / a - 1.0) * 100.0

    dates = idx.to_numpy()
    day_iso = np.array([str(d)[:10] for d in dates])
    reg = np.array([regime_by_day.get(s, "") for s in day_iso])
    mask = ((np.arange(n) >= off) & (day_iso >= START) & (day_iso <= end)
            & (reg != ""))
    days = np.where(mask)[0]
    if not len(days):
        return None

    runway, rbasis = runway_arrays(dates, edata.get("dates") or [],
                                   edata.get("status", "error"))
    j = days - off
    a20d = close[days] > sma20[days]
    a50d = close[days] > sma50[days]
    mgtd = sma20[days] > sma50[days]
    comps = comp_vectors(rsi[j], bullish[j], confirms[j],
                         a20d, a50d, mgtd, ytd[days], vratio[days])
    scores, scores_np = score_components_vec(
        rsi[j], bullish[j], confirms[j], a20d, a50d, mgtd,
        ytd[days], vratio[days])
    # pin 2: decomposition recomposes to the pinned totals exactly
    base_sum = (SCORE_BASE + comps["rsi"] + comps["macd"] + comps["ma"]
                + comps["vol"])
    assert np.array_equal(clip_score(base_sum + comps["ytd"]), scores), \
        f"{t}: component recomposition != score_components_vec score"
    assert np.array_equal(clip_score(base_sum + comps["ytd_v1_only"]),
                          scores_np), \
        f"{t}: nopen recomposition != score_components_vec score_nopen"

    out = []
    for k, i in enumerate(days):
        jj = i - off
        atr = atr14[i] if np.isfinite(atr14[i]) else None
        c1 = bool(above[i])
        atr_break = c1 and atr is not None \
            and close[i] > sma20[i] + KNOBS["atr_mult"] * atr
        c2 = (consec[i] >= KNOBS["confirmation_closes"]) or atr_break
        c3 = reg[i] in IN_TREND
        c4 = bool(sma20[i] >= sma20_then[i]) \
            if np.isfinite(sma20_then[i]) else False
        ext = None if not atr else (close[i] - sma20[i]) / atr
        r2 = ext is not None and ext <= KNOBS["extension_guard_max"]
        r3 = bool(np.isfinite(sma5[i]) and close[i] > sma5[i]
                  and up20[i])
        r4 = bool(KNOBS["rsi_min"] <= rsi[jj] <= KNOBS["rsi_max"])
        r5 = bool(scores[k] >= SCORE_MIN)
        gp = rbasis[i] == 2
        rw = int(runway[i]) if runway[i] >= 0 else None
        r7 = (not gp) and (rw is None
                           or rw >= KNOBS["runway_min_sessions"])
        out.append((day_iso[i], t, c1, c2, c3, c4, r2, r3, r4, r5, r7,
                    float(scores[k]),
                    float(comps["rsi"][k]), float(comps["macd"][k]),
                    float(comps["ma"][k]), float(comps["ytd"][k]),
                    float(comps["vol"][k]),
                    float(comps["ytd_v1_only"][k]),
                    float(gap[jj]),
                    float(atr) if atr is not None else np.nan,
                    float(close[i]),
                    bool(a20d[k]), bool(a50d[k]), bool(mgtd[k]),
                    float(ytd[i]),
                    float(fwd[i]) if np.isfinite(fwd[i]) else np.nan))
    return out


COLS = ["date", "ticker", "c1", "c2", "c3", "c4", "r2", "r3", "r4",
        "r5", "r7", "score", "p_rsi", "p_macd", "p_ma", "p_ytd",
        "p_vol", "p_ytd_v1_only", "macd_gap", "atr14", "close",
        "above20", "above50", "ma20_gt_50", "ytd_val", "fwd_20"]


# --------------------------------------------------------------- reporting
def safe_cluster_ci(df_ex, df_ap):
    """cluster_ci with the finite guard: a non-empty set whose fwd_20
    is all-NaN (recent-tail days) must report n_boot 0, not crash in
    np.percentile (review finding, fixed before results were read)."""
    if not len(df_ex[["date", "fwd_20"]].dropna()) \
            or not len(df_ap[["date", "fwd_20"]].dropna()):
        return {"n_boot": 0}
    return cluster_ci(df_ex, df_ap)


def set_block(sub, ap, spans):
    """Per-span dist blocks, each carrying its OWN date-cluster CI vs
    the A+ reference — bars 1-3 read 'CI excludes zero, holds in BOTH
    spans', so the train and validate CIs are the ruling statistics
    (review finding: the first cut emitted a full-span CI only, making
    the registered bars unevaluable from the artifact; fixed before
    any result was read)."""
    blk = {}
    for span, sel in spans.items():
        b = dist_block(sel(sub).fwd_20.to_numpy())
        ap_fin = int(sel(ap).fwd_20.notna().sum())
        b["share_of_aplus_pct"] = round(
            b.get("n", 0) / max(1, ap_fin) * 100, 1)
        if b.get("n"):
            b["bootstrap_vs_aplus"] = safe_cluster_ci(sel(sub), sel(ap))
        blk[span] = b
    if len(sub):
        fin = sub[sub.fwd_20.notna()]
        tot = fin.fwd_20.sum()
        by_t = fin.groupby("ticker").fwd_20.sum()
        blk["concentration"] = {
            "distinct_tickers": int(fin.ticker.nunique()),
            "distinct_dates": int(fin.date.nunique()),
            "top_ticker": str(by_t.idxmax()) if len(by_t) else None,
            "top_ticker_share_of_sum_pct": round(
                float(by_t.max() / tot * 100), 1)
            if len(by_t) and tot != 0 else None}
    return blk


def pair_diff_ci(df_a, df_b, n_boot=2000, seed=20260731):
    """Date-cluster bootstrap CI on (mean A - mean B) for two disjoint
    path sets — same clustering as cluster_ci, different pairing."""
    a = df_a[["date", "fwd_20"]].dropna()
    b = df_b[["date", "fwd_20"]].dropna()
    by_a = {k: v.fwd_20.to_numpy() for k, v in a.groupby("date")}
    by_b = {k: v.fwd_20.to_numpy() for k, v in b.groupby("date")}
    dates = sorted(set(by_a) | set(by_b))
    rng = np.random.default_rng(seed)
    diffs = []
    for _ in range(n_boot):
        pick = rng.choice(dates, size=len(dates), replace=True)
        sa = np.concatenate([by_a.get(d, np.empty(0)) for d in pick])
        sb = np.concatenate([by_b.get(d, np.empty(0)) for d in pick])
        if len(sa) and len(sb):
            diffs.append(sa.mean() - sb.mean())
    if not diffs:
        return {"n_boot": 0}
    d = np.array(diffs)
    return {"mean_diff": round(float(a.fwd_20.mean()
                                     - b.fwd_20.mean()), 3),
            "ci_2_5": round(float(np.percentile(d, 2.5)), 3),
            "ci_97_5": round(float(np.percentile(d, 97.5)), 3),
            "p_diff_le_0": round(float((d <= 0).mean()), 4),
            "n_boot": len(d)}


def run(out_path=None):
    from study_inputs import assert_pinned_inputs
    assert_pinned_inputs(["prices_manifest", "earnings_dates",
                          "regime_daily", "master_frame"],
                         label="Build 5.2")
    ladder_parity_pin()
    with open(REGIME_PATH) as f:
        regime = json.load(f)
    regime_by_day = regime["states"]
    end = regime["provenance"]["series_range"][1]
    with open(EARNINGS_PATH) as f:
        earnings = json.load(f)

    rows = []
    files = sorted(os.listdir(PRICES))
    for i, fn in enumerate(files, 1):
        t = fn[:-4]
        r = ticker_panel(t, os.path.join(PRICES, fn),
                         earnings.get(t) or {"status": "missing"},
                         regime_by_day, end)
        if r:
            rows.extend(r)
        if i % 100 == 0:
            print(f"  [{i}/{len(files)}]", flush=True)
    df = pd.DataFrame(rows, columns=COLS)
    print(f"panel: {len(df):,} ticker-days")

    # ---- pin 1: joint AND == committed A+ set ------------------------
    flags = df[list(ROWS)].to_numpy(dtype=bool)
    df["aplus"] = flags.all(axis=1)
    frame = pd.read_csv(FRAME_PATH, usecols=["date", "ticker", "grade"])
    merged = df.merge(frame, on=["date", "ticker"], how="inner")
    assert len(merged) == len(df) == len(frame), \
        (len(df), len(frame), len(merged))
    mism = merged[(merged.aplus) != (merged.grade == 3)]
    assert len(mism) == 0, (
        f"{len(mism)} ticker-days disagree with the committed frame's "
        f"A+ set; sample:\n{mism.head(6)}")
    print(f"A+ parity: recomputed set == committed frame "
          f"({int(df.aplus.sum()):,} days) EXACTLY")

    ap = df[df.aplus]
    others = df[list(OTHERS)].to_numpy(dtype=bool).all(axis=1)
    df["pass_others"] = others
    po = df[df.pass_others]
    spans = {"full": lambda d: d,
             "train": lambda d: d[d.date <= TRAIN_END],
             "validate": lambda d: d[d.date > TRAIN_END]}

    results = {"schema": "backtest-score-ablation-1",
               "prereg": "docs/backtest-score-ablation-prereg.md "
                         "(cdcfa19)",
               "aplus_reference": {}, "components": {},
               "structures": {}}
    for span, sel in spans.items():
        results["aplus_reference"][span] = dist_block(
            sel(ap).fwd_20.to_numpy())

    # ---- zero-ablation per component ---------------------------------
    parts = {"rsi": df.p_rsi, "macd": df.p_macd, "ma": df.p_ma,
             "ytd": df.p_ytd, "vol": df.p_vol}
    total = (SCORE_BASE + df.p_rsi + df.p_macd + df.p_ma + df.p_ytd
             + df.p_vol)
    assert np.array_equal(clip_score(total.to_numpy()),
                          df.score.to_numpy()), "total recomposition"
    for comp in COMPONENTS:
        ablated = clip_score((total - parts[comp]).to_numpy())
        r5_abl = ablated >= SCORE_MIN
        lost = df[df.pass_others & df.r5 & ~r5_abl]
        gained = df[df.pass_others & ~df.r5 & r5_abl]
        results["components"][comp] = {
            "lost": set_block(lost, ap, spans),
            "gained": set_block(gained, ap, spans)}

    # ---- (a) MACD cliff ----------------------------------------------
    a_blk = {"bands": {}}
    gap_abs = (df.macd_gap.abs() / df.close).to_numpy()
    for band in BANDS:
        in_band = df.pass_others & (gap_abs < band)
        sub = df[in_band]
        bull = sub[sub.macd_gap > 0]
        bear = sub[sub.macd_gap <= 0]
        entry = {
            "n_pass_others": int(len(sub)),
            "share_of_pass_others_pct": round(
                len(sub) / max(1, len(po)) * 100, 2),
            "bull_side": {s: dist_block(sel(bull).fwd_20.to_numpy())
                          for s, sel in spans.items()},
            "bear_side": {s: dist_block(sel(bear).fwd_20.to_numpy())
                          for s, sel in spans.items()},
        }
        if len(bull) and len(bear):
            entry["bull_minus_bear"] = {
                s: pair_diff_ci(sel(bull), sel(bear))
                for s, sel in spans.items() if len(sel(bull))
                and len(sel(bear))}
        a_blk["bands"][f"{band * 100:g}pct"] = entry
    # smoothed variant
    with np.errstate(divide="ignore", invalid="ignore"):
        m_smooth = np.clip(
            13.0 * df.macd_gap.to_numpy()
            / (0.5 * df.atr14.to_numpy()), -13, 13)
    m_smooth = np.where(np.isfinite(m_smooth), m_smooth, 0.0)
    sm_total = clip_score((total - df.p_macd + m_smooth).to_numpy())
    r5_sm = sm_total >= SCORE_MIN
    a_blk["smoothed"] = {
        "formula": "clip(13*(macd-signal)/(0.5*ATR14), -13, +13)",
        "lost": set_block(df[df.pass_others & df.r5 & ~r5_sm], ap,
                          spans),
        "gained": set_block(df[df.pass_others & ~df.r5 & r5_sm], ap,
                            spans)}
    results["structures"]["macd_cliff"] = a_blk

    # ---- (b) MA path split -------------------------------------------
    paths = {
        "P1_pullback_uptrend": (~df.above20 & df.above50
                                & df.ma20_gt_50),
        "P2_recovery": (df.above20 & df.above50 & ~df.ma20_gt_50),
        "N1_deep_pullback_uptrend": (~df.above20 & ~df.above50
                                     & df.ma20_gt_50),
        "N2_bounce_downtrend": (df.above20 & ~df.above50
                                & ~df.ma20_gt_50)}
    cuts = {"full_panel": df, "c3_and_c4": df[df.c3 & df.c4]}
    b_blk = {}
    for cut_name, cut in cuts.items():
        cb = {}
        for pname, pmask in paths.items():
            sub = cut[pmask.reindex(cut.index, fill_value=False)]
            cb[pname] = {s: dist_block(sel(sub).fwd_20.to_numpy())
                         for s, sel in spans.items()}
        for x, y in (("P1_pullback_uptrend", "P2_recovery"),
                     ("N1_deep_pullback_uptrend",
                      "N2_bounce_downtrend")):
            sa = cut[paths[x].reindex(cut.index, fill_value=False)]
            sb = cut[paths[y].reindex(cut.index, fill_value=False)]
            if len(sa) and len(sb):
                cb[f"{x}_minus_{y}"] = {
                    s: pair_diff_ci(sel(sa), sel(sb))
                    for s, sel in spans.items()
                    if len(sel(sa)) and len(sel(sb))}
        b_blk[cut_name] = cb
    b_blk["note"] = ("P1 and N1 are structurally EMPTY within A+ "
                     "(c1 == above20); comparison runs on the panel "
                     "cuts, per prereg")
    results["structures"]["ma_paths"] = b_blk

    # ---- (c) YTD saturation ------------------------------------------
    c_blk = {}
    buckets = {"20_50": (df.ytd_val > 20) & (df.ytd_val <= 50),
               "50_100": (df.ytd_val > 50) & (df.ytd_val <= 100),
               "100_150": (df.ytd_val > 100) & (df.ytd_val <= 150),
               "gt150": df.ytd_val > 150}
    for scope_name, scope in (("pass_others", po), ("aplus", ap)):
        sb = {}
        for bname, bmask in buckets.items():
            sub = scope[bmask.reindex(scope.index, fill_value=False)]
            sb[bname] = {s: dist_block(sel(sub).fwd_20.to_numpy())
                         for s, sel in spans.items()}
        # the REGISTERED extreme-bucket contrast: (20,50] vs >150 — the
        # outermost two of the prereg's ordered list (review finding:
        # the first cut pooled (100,150] into the high side, an
        # unregistered contrast that can both dilute a real >150 effect
        # and manufacture separation; fixed before any result was read)
        lo = scope[buckets["20_50"].reindex(scope.index,
                                            fill_value=False)]
        hi = scope[buckets["gt150"].reindex(scope.index,
                                            fill_value=False)]
        if len(lo) and len(hi):
            sb["extreme_diff_20_50_minus_gt150"] = {
                s: pair_diff_ci(sel(lo), sel(hi))
                for s, sel in spans.items()
                if len(sel(lo)) and len(sel(hi))}
        c_blk[f"buckets_{scope_name}"] = sb

    y_v2 = ytd_v2_vec(df.ytd_val.to_numpy())
    v2_total = clip_score((total - df.p_ytd + y_v2).to_numpy())
    r5_v2 = v2_total >= SCORE_MIN
    c_blk["v2_backcast"] = {
        "lost": set_block(df[df.pass_others & df.r5 & ~r5_v2], ap,
                          spans),
        "gained": set_block(df[df.pass_others & ~df.r5 & r5_v2], ap,
                            spans)}

    y_unc = ytd_uncapped_vec(df.ytd_val.to_numpy())
    unc_total = clip_score((total - df.p_ytd + y_unc).to_numpy())
    r5_unc = unc_total >= SCORE_MIN
    lost_unc = df[df.pass_others & r5_v2 & ~r5_unc]
    assert len(lost_unc) == 0, \
        "uncapped ladder lost days vs v2 — impossible by construction"
    c_blk["uncapped_vs_v2"] = {
        "ladder": "12@>20, 16@>50, 20@>100, 24@>150",
        "lost": {"n": 0, "note": "empty by construction (ladder "
                                 "dominates v2 pointwise)"},
        "gained": set_block(df[df.pass_others & ~r5_v2 & r5_unc], ap,
                            spans)}
    results["structures"]["ytd_saturation"] = c_blk

    head = subprocess.run(["git", "rev-parse", "--short", "HEAD"],
                          capture_output=True, text=True,
                          cwd=REPO).stdout.strip()
    results["provenance"] = {
        "generated_from_commit": head,
        "frame_parity": f"{int(df.aplus.sum())} A+ days reproduce the "
                        "committed frame exactly",
        "regime_series": regime["provenance"]["series_range"],
        "pass_others_n": int(len(po)),
    }
    canonical = json.loads(json.dumps(results))
    results["results_hash"] = hashlib.sha256(
        json.dumps(canonical, sort_keys=True).encode()).hexdigest()[:16]

    out_path = out_path or os.path.join(
        REPO, "docs", "backtest-score-ablation-results.json")
    with open(out_path, "w") as f:
        json.dump(results, f, indent=1)
    print(f"results -> {out_path} (hash {results['results_hash']})")
    return results, df


def main():
    ap_ = argparse.ArgumentParser()
    ap_.add_argument("--out")
    a = ap_.parse_args()
    run(out_path=a.out)


if __name__ == "__main__":
    main()
