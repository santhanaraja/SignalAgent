#!/usr/bin/env python3
"""
Build 5.1 — row ablation (pre-registered: docs/backtest-ablation-prereg.md,
committed 7ea58bc BEFORE any results existed).

For each grade row R, the EXCLUDED SET is the ticker-days that pass
every other row but fail R. The primary question, per row: did the days
this row excluded outperform the days it admitted?

Ships nothing to production. Signal-level only. No threshold sweeps.

The row flags are recomputed here from the same caches and feature
functions Layer A used (imported, not reimplemented), and the joint
AND of all flags must reproduce the committed frame's A+ set EXACTLY
(pin 1) before any ablation statistic is trusted.

Rows: c1 close>SMA20 · c2 confirmation · c3 regime gate · c4 slope ·
R2 extension<=1.8xATR · R3 approach · R4 RSI 45-70 · R5 score>=75 ·
R7 runway>=15 (gap never-A+) · construct: R5 without the YTD>100%
penalty (grade-flip days). c1's excluded set is structurally EMPTY
(c2's confirmation count is zero whenever close<=SMA20 and the ATR
break requires being above), reported as such rather than as a zero.

Run: python3 scripts/backtest_ablation.py [--out PATH]
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

from backtest_doctrine import (rsi_windowed, macd_windowed,
                               score_components_vec, runway_arrays,
                               FRAME, KNOBS, CACHE, PRICES, EARNINGS_PATH,
                               REGIME_PATH, START, TRAIN_END,
                               up_close_windowed, ytd_aslived)

FRAME_PATH = os.path.join(CACHE, "master_frame.csv.gz")
ROWS = ("c1", "c2", "c3", "c4", "r2", "r3", "r4", "r5", "r7")
IN_TREND = ("In-Trend-Full", "In-Trend-Throttled")


def ticker_flags(t, path, edata, regime_by_day, end):
    """Per grading day: every row flag + fwd20. Mirrors Layer A's
    process_ticker feature construction (same helpers, same knobs)."""
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
    scores, scores_np = score_components_vec(
        rsi[np.maximum(days - off, 0)],
        bullish[np.maximum(days - off, 0)],
        confirms[np.maximum(days - off, 0)],
        close[days] > sma20[days], close[days] > sma50[days],
        sma20[days] > sma50[days], ytd[days], vratio[days])

    out = []
    for k, i in enumerate(days):
        j = i - off
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
        r4 = bool(KNOBS["rsi_min"] <= rsi[j] <= KNOBS["rsi_max"])
        r5 = bool(scores[k] >= KNOBS["score_min"])
        r5_np = bool(scores_np[k] >= KNOBS["score_min"])
        gap = rbasis[i] == 2
        rw = int(runway[i]) if runway[i] >= 0 else None
        # grade_setup semantics: None runway passes (no known print);
        # the ruled None-split makes a coverage GAP never-A+
        r7 = (not gap) and (rw is None
                            or rw >= KNOBS["runway_min_sessions"])
        out.append((day_iso[i], t, c1, c2, c3, c4, r2, r3, r4, r5, r7,
                    r5_np, float(fwd[i]) if np.isfinite(fwd[i])
                    else np.nan))
    return out


COLS = ["date", "ticker", "c1", "c2", "c3", "c4", "r2", "r3", "r4",
        "r5", "r7", "r5_nopen", "fwd_20"]


def dist_block(v):
    v = v[np.isfinite(v)]
    if len(v) == 0:
        return {"n": 0}
    k = max(1, int(np.ceil(len(v) * 0.05)))
    top = np.sort(v)[-k:]
    total = v.sum()
    return {
        "n": int(len(v)),
        "mean": round(float(v.mean()), 3),
        "median": round(float(np.median(v)), 3),
        "sd": round(float(v.std(ddof=1)), 3) if len(v) > 1 else None,
        "hit_pct": round(float((v > 0).mean()) * 100, 1),
        "gt10_pct": round(float((v > 10).mean()) * 100, 2),
        "gt25_pct": round(float((v > 25).mean()) * 100, 2),
        "gt50_pct": round(float((v > 50).mean()) * 100, 3),
        "lt10_pct": round(float((v < -10).mean()) * 100, 2),
        "lt25_pct": round(float((v < -25).mean()) * 100, 3),
        "lt50_pct": round(float((v < -50).mean()) * 100, 4),
        "top5_share_pct": round(float(top.sum() / total * 100), 1)
        if total != 0 else None,
        "ex_top5_mean": round(float((total - top.sum())
                                    / max(1, len(v) - k)), 3),
    }


def cluster_ci(df_ex, df_ap, n_boot=2000, seed=20260731):
    """Date-cluster bootstrap CI on (excluded mean - A+ mean), fwd20."""
    ex = df_ex[["date", "fwd_20"]].dropna()
    ap = df_ap[["date", "fwd_20"]].dropna()
    by_e = {k: v.fwd_20.to_numpy() for k, v in ex.groupby("date")}
    by_a = {k: v.fwd_20.to_numpy() for k, v in ap.groupby("date")}
    dates = sorted(set(by_e) | set(by_a))
    rng = np.random.default_rng(seed)
    diffs = []
    for _ in range(n_boot):
        pick = rng.choice(dates, size=len(dates), replace=True)
        se = np.concatenate([by_e.get(d, np.empty(0)) for d in pick])
        sa = np.concatenate([by_a.get(d, np.empty(0)) for d in pick])
        if len(se) and len(sa):
            diffs.append(se.mean() - sa.mean())
    d = np.array(diffs)
    return {"ci_2_5": round(float(np.percentile(d, 2.5)), 3),
            "ci_97_5": round(float(np.percentile(d, 97.5)), 3),
            "p_diff_le_0": round(float((d <= 0).mean()), 4),
            "n_boot": len(d)}


def run(out_path=None):
    # pinned inputs (2026-08-10 ruling) — 5.1 reads the frame, the
    # prices, the earnings cache and the regime series
    from study_inputs import assert_pinned_inputs
    assert_pinned_inputs(["prices_manifest", "earnings_dates",
                          "regime_daily", "master_frame"],
                         label="Build 5.1")
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
        r = ticker_flags(t, os.path.join(PRICES, fn),
                         earnings.get(t) or {"status": "missing"},
                         regime_by_day, end)
        if r:
            rows.extend(r)
        if i % 100 == 0:
            print(f"  [{i}/{len(files)}]", flush=True)
    df = pd.DataFrame(rows, columns=COLS)
    print(f"flag panel: {len(df):,} ticker-days")

    # ---- PIN-CRITICAL integrity: joint AND == committed A+ set --------
    flags = df[list(ROWS)].to_numpy(dtype=bool)
    df["aplus"] = flags.all(axis=1)
    frame = pd.read_csv(FRAME_PATH, usecols=["date", "ticker", "grade"])
    merged = df.merge(frame, on=["date", "ticker"], how="inner")
    assert len(merged) == len(df) == len(frame), \
        (len(df), len(frame), len(merged))
    mism = merged[(merged.aplus) != (merged.grade == 3)]
    assert len(mism) == 0, (
        f"{len(mism)} ticker-days disagree with the committed frame's A+ "
        f"set — the recomputed flags have drifted; sample:\n"
        f"{mism.head(6)}")
    print(f"A+ parity: recomputed set == committed frame "
          f"({int(df.aplus.sum()):,} days) EXACTLY")

    ap = df[df.aplus]
    spans = {"full": lambda d: d,
             "train": lambda d: d[d.date <= TRAIN_END],
             "validate": lambda d: d[d.date > TRAIN_END]}

    results = {"schema": "backtest-ablation-1",
               "prereg": "docs/backtest-ablation-prereg.md (7ea58bc)",
               "aplus_reference": {}, "rows": {}, "construct": {},
               "notes": {
                   "c1": "excluded set structurally EMPTY: c2's "
                         "confirmation count is 0 whenever close<=SMA20 "
                         "and the ATR break requires being above — c1 "
                         "cannot fail alone",
                   "r6": "untestable — ruled Layer A deletion"}}
    for span, sel in spans.items():
        results["aplus_reference"][span] = dist_block(
            sel(ap).fwd_20.to_numpy())

    for rname in ROWS:
        others = [x for x in ROWS if x != rname]
        excl = df[df[others].all(axis=1) & ~df[rname]]
        blk = {}
        for span, sel in spans.items():
            b = dist_block(sel(excl).fwd_20.to_numpy())
            # FINITE/FINITE basis (review finding: the first cut divided a
            # finite-fwd numerator by an all-days denominator — a mixed
            # basis off by ~1pp). Both counts now exclude the unresolved
            # trailing ~21 days.
            ap_fin = int(sel(ap).fwd_20.notna().sum())
            b["share_of_aplus_pct"] = round(
                b.get("n", 0) / max(1, ap_fin) * 100, 1)
            blk[span] = b
        if len(excl):
            blk["bootstrap_vs_aplus_full"] = cluster_ci(excl, ap)
            # concentration context (review finding: a carried suspect's
            # mean must be readable against its name/date breadth)
            fin = excl[excl.fwd_20.notna()]
            tot = fin.fwd_20.sum()
            by_t = fin.groupby("ticker").fwd_20.sum()
            blk["concentration"] = {
                "distinct_tickers": int(fin.ticker.nunique()),
                "distinct_dates": int(fin.date.nunique()),
                "top_ticker": str(by_t.idxmax()) if len(by_t) else None,
                "top_ticker_share_of_sum_pct": round(
                    float(by_t.max() / tot * 100), 1)
                if len(by_t) and tot != 0 else None}
        results["rows"][rname] = blk

    # construct: the YTD>100% penalty — days whose GRADE flips when the
    # penalty is removed (pass everything else incl. r7; r5 fails only
    # via the penalty)
    others5 = [x for x in ROWS if x != "r5"]
    flip = df[df[others5].all(axis=1) & ~df.r5 & df.r5_nopen]
    blk = {}
    for span, sel in spans.items():
        b = dist_block(sel(flip).fwd_20.to_numpy())
        ap_fin = int(sel(ap).fwd_20.notna().sum())
        b["share_of_aplus_pct"] = round(
            b.get("n", 0) / max(1, ap_fin) * 100, 1)
        blk[span] = b
    if len(flip):
        blk["bootstrap_vs_aplus_full"] = cluster_ci(flip, ap)
        fin = flip[flip.fwd_20.notna()]
        tot = fin.fwd_20.sum()
        by_t = fin.groupby("ticker").fwd_20.sum()
        blk["concentration"] = {
            "distinct_tickers": int(fin.ticker.nunique()),
            "distinct_dates": int(fin.date.nunique()),
            "top_ticker": str(by_t.idxmax()) if len(by_t) else None,
            "top_ticker_share_of_sum_pct": round(
                float(by_t.max() / tot * 100), 1)
            if len(by_t) and tot != 0 else None}
    results["construct"]["ytd_penalty_flip"] = blk

    head = subprocess.run(["git", "rev-parse", "--short", "HEAD"],
                          capture_output=True, text=True,
                          cwd=REPO).stdout.strip()
    results["provenance"] = {
        "generated_from_commit": head,
        "frame_parity": f"{int(df.aplus.sum())} A+ days reproduce the "
                        "committed frame exactly",
        "regime_series": regime["provenance"]["series_range"],
    }
    canonical = json.loads(json.dumps(results))
    results["results_hash"] = hashlib.sha256(
        json.dumps(canonical, sort_keys=True).encode()).hexdigest()[:16]

    out_path = out_path or os.path.join(
        REPO, "docs", "backtest-ablation-results.json")
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
