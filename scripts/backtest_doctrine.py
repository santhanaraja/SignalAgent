#!/usr/bin/env python3
"""
Build 5 Layer A — the doctrine backtest (D-011's retest recipe, executed).

THE QUESTION: does the A+ grade actually discriminate? Forward returns of
A+ vs B vs C candidates, graded daily and point-in-time over the 533-name
pool, 2020-01 -> 2026-07. The doctrine was ruled by reasoning (D-011);
this is its first measurement. (The D-011 record's recipe named
`replay_ticker.py --grade-entries`; this script supersedes it per the
Build 5 ruling — the record's evidence note says so.)

GRADE-LITE (Step-0 ruling 1): rows 1-5 + 7, MINUS condition 5 (group-in-
universe — no rotation history before 2026-07) and MINUS row 6 (breaker —
membership unrecoverable and the recorded "clear" itself unproven pre-
D-019). Row 7 carries the None-SPLIT: no-earnings-entity passes (production
semantics), a COVERAGE GAP is never A+. A no-row-7 sensitivity bounds how
much row 7 drives the result.

POINT-IN-TIME DISCIPLINE (D-006): every indicator is computed on the
trailing 126-bar frame ending at T — the same shape as production's 6mo
fetch (fixed 126 vs production's calendar-6mo 124..128 is a declared
approximation; the parity pin quantifies it). Fills and forward returns
are measured from T+1 open, mirroring Build 4's info_test. Regime
conditioning reads the committed chassis series (data/regime_daily.json).

LAW 1: the grade verdict comes from the REAL grade_setup and the runway
from the REAL runway_sessions_before. Score components are vectorized
replicas of the score_*_points functions, parity-pinned against the
production implementations.

Run:  python3 scripts/backtest_doctrine.py [--smoke] [--out PATH]
"""

import argparse
import datetime
import hashlib
import json
import os
import sys
from collections import Counter

REPO = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
sys.path.insert(0, REPO)

import numpy as np
import pandas as pd
from numpy.lib.stride_tricks import sliding_window_view

from framework.position_signals import grade_setup, runway_sessions_before
# D-020a: this study is FROZEN to the v1 ladder — the committed frame
# (input_hash d7df85e8ad6ba244) was built on it, and it must keep
# reproducing after production moved to score_stock_v2
from signal_engine import (score_rsi_points, score_macd_points,
                           score_ma_points, score_vol_points, SCORE_BASE,
                           score_ytd_points_v1 as score_ytd_points)

CACHE = os.path.join(REPO, "data", "doctrine_cache")
PRICES = os.path.join(CACHE, "prices")
EARNINGS_PATH = os.path.join(CACHE, "earnings_dates.json")
REGIME_PATH = os.path.join(REPO, "data", "regime_daily.json")

FRAME = 126                    # the production 6mo fetch, fixed (declared)
START, END = "2020-01-02", None   # END = regime-series end
TRAIN_END = "2023-12-31"       # D-006 split: train 2020-2023 / validate 2024-
HORIZONS = (5, 10, 20, 40)
SWING_LOOKBACKS = (10, 20, 40)   # 20 = production; others = ruled sensitivity
EARNINGS_GAP_DAYS = 240        # enclosing print-interval beyond this = gap
GRADE_CODE = {"A+": 3, "B": 2, "C": 1}

# production knobs (PositionSignalEngine defaults — the ruled values)
KNOBS = dict(sma_period=20, slope_lookback=5, atr_period=14,
             confirmation_closes=2, atr_mult=0.5, extension_guard_max=1.8,
             rsi_min=45.0, rsi_max=70.0, score_min=75.0,
             runway_min_sessions=15)


# ---------------------------------------------------------------- features
def rsi_windowed(close, frame=FRAME, period=14):
    """compute_rsi's exact recipe applied per trailing frame: gain[0]=0
    (the NaN first delta), rolling-mean seed at bar `period`, Wilder
    recursion to the frame end. Returns the frame-end RSI per day
    (aligned to close[frame-1:])."""
    F = sliding_window_view(close, frame)
    d = np.diff(F, axis=1)
    g = np.where(d > 0, d, 0.0)
    l = np.where(d < 0, -d, 0.0)
    # seed = mean of gain[0..period-1] where gain[0]=0, gain[k]=d[k-1]
    ag = np.concatenate([np.zeros((g.shape[0], 1)), g[:, :period - 1]],
                        axis=1).mean(axis=1)
    al = np.concatenate([np.zeros((l.shape[0], 1)), l[:, :period - 1]],
                        axis=1).mean(axis=1)
    for i in range(period - 1, g.shape[1]):
        ag = (ag * (period - 1) + g[:, i]) / period
        al = (al * (period - 1) + l[:, i]) / period
    with np.errstate(divide="ignore", invalid="ignore"):
        rsi = 100.0 - 100.0 / (1.0 + ag / al)
    rsi = np.where(al == 0.0, 100.0, rsi)
    rsi = np.where((al == 0.0) & (ag == 0.0), 50.0, rsi)  # NaN -> prod's 50
    return rsi


def _ema_windowed(F, span):
    a = 2.0 / (span + 1.0)
    e = F[:, 0].copy()
    out = np.empty_like(F)
    out[:, 0] = e
    for i in range(1, F.shape[1]):
        e = a * F[:, i] + (1.0 - a) * e
        out[:, i] = e
    return out


def macd_windowed(close, frame=FRAME):
    """MACD 12/26/9 per trailing frame (ewm adjust=False seeded at the
    frame start, exactly like production on its 6mo fetch). Returns
    (bullish, confirms) at the frame end."""
    F = sliding_window_view(close, frame)
    macd = _ema_windowed(F, 12) - _ema_windowed(F, 26)
    signal = _ema_windowed(macd, 9)
    hist = macd - signal
    bullish = macd[:, -1] > signal[:, -1]
    confirms = np.where(bullish, hist[:, -1] > hist[:, -2],
                        hist[:, -1] < hist[:, -2])
    return bullish, confirms


def up_close_windowed(close, lookback):
    """up_close_off_swing_low per trailing window: >=1 positive close-to-
    close diff at or after the window's argmin."""
    Fw = sliding_window_view(close, lookback)
    low_pos = np.argmin(Fw, axis=1)
    pos_diff = np.diff(Fw, axis=1) > 0
    # suffix-any: c[:, j] == any(pos_diff[:, j:])
    c = np.maximum.accumulate(pos_diff[:, ::-1], axis=1)[:, ::-1]
    idx = np.minimum(low_pos, pos_diff.shape[1] - 1)
    out = c[np.arange(len(low_pos)), idx]
    out[low_pos > pos_diff.shape[1] - 1] = False
    return out.astype(bool)


def ytd_aslived(close, index, frame=FRAME):
    """The GRADE PATH's frame-anchored YTD: first valid close of the as-of
    year WITHIN the trailing frame -> last close, rounded 2 (production
    compute_ytd_return on the 6mo fetch, with now().year parameterized to
    the as-of date). In H2 the anchor is the frame start, not Jan — the
    'as-lived' number production actually scored (Step-0 finding: the two
    production YTD variants disagree by up to ~26pts; this is the one the
    grade consumed). <2 rows in-year -> 0.0 (the Jan-boundary quirk,
    replicated not fixed)."""
    years = index.year.to_numpy()
    year_start_idx = {}
    for y in np.unique(years):
        year_start_idx[y] = int(np.argmax(years == y))
    n = len(close)
    out = np.zeros(n)
    anchor = np.empty(n, dtype=int)
    for i in range(n):
        a = max(year_start_idx[years[i]], i - (frame - 1))
        anchor[i] = a
        if a >= i:                       # <2 in-year rows in the frame
            out[i] = 0.0
        else:
            out[i] = round((close[i] / close[a] - 1.0) * 100.0, 2)
    return out


def score_components_vec(rsi, bullish, confirms, above20, above50,
                         ma20_gt_50, ytd, vol_ratio):
    """Vectorized replicas of the production score_*_points ladders —
    parity-pinned elementwise against the real functions."""
    r = np.select([rsi < 30, rsi < 40, rsi < 60, rsi < 70, rsi < 80],
                  [15, 8, 3, -3, -8], default=-15)
    m = np.where(bullish, 8 + np.where(confirms, 5, 0),
                 -8 - np.where(confirms, 5, 0))
    ma = (np.where(above20, 4, -4) + np.where(above50, 6, -6)
          + np.where(ma20_gt_50, 4, -4))
    y = np.select([ytd > 50, ytd > 20, ytd > 5, ytd > 0, ytd > -10],
                  [8, 12, 6, 2, -4], default=-10)
    pen = np.select([ytd > 150, ytd > 100], [-15, -10], default=0)
    v = np.select([vol_ratio > 1.5, vol_ratio < 0.7], [3, -3], default=0)
    base = SCORE_BASE + r + m + ma + v
    score = np.clip(base + y + pen, 0, 100)
    score_nopen = np.clip(base + y, 0, 100)
    return score.astype(float), score_nopen.astype(float)


# ---------------------------------------------------------------- runway
def runway_arrays(dates, edates, status):
    """Per-day (runway_sessions, basis) from the ticker's known report
    dates. basis: 0=known, 1=no_entity, 2=coverage_gap. Mirrors
    runway_sessions_before's semantics on arrays: print day = 0 runway;
    sessions = weekday count in [D, print)."""
    n = len(dates)
    if status != "ok":
        return np.full(n, -1, dtype=int), np.full(n, 2, dtype=np.int8)
    if not edates:
        # an equity with zero known prints is a COVERAGE GAP, not a
        # no-entity pass — the pool is all single stocks (ruling 1)
        return np.full(n, -1, dtype=int), np.full(n, 2, dtype=np.int8)
    E = np.array(edates, dtype="datetime64[D]")
    D = dates.astype("datetime64[D]")
    nxt = np.searchsorted(E, D, side="left")    # first print >= D
    basis = np.zeros(n, dtype=np.int8)
    runway = np.full(n, -1, dtype=int)
    has_next = nxt < len(E)
    basis[~has_next] = 2                        # beyond the known table
    # enclosing-interval gap check: a missing print shows up as an
    # implausibly long gap between the neighbouring known prints
    prev_ok = nxt > 0
    iv = np.full(n, np.timedelta64(0, "D"))
    both = has_next & prev_ok
    iv[both] = E[nxt[both]] - E[nxt[both] - 1]
    gap = both & (iv > np.timedelta64(EARNINGS_GAP_DAYS, "D"))
    lead = has_next & ~prev_ok \
        & ((E[np.minimum(nxt, len(E) - 1)] - D) > np.timedelta64(
            EARNINGS_GAP_DAYS, "D"))
    basis[gap | lead] = 2
    ok = has_next & (basis == 0)
    idx = np.where(ok)[0]
    if len(idx):
        runway[idx] = np.busday_count(D[idx], E[nxt[idx]])
    return runway, basis


# ---------------------------------------------------------------- per ticker
def process_ticker(t, path, edata, regime_by_day, start, end):
    df = pd.read_csv(path, index_col=0, parse_dates=True)
    if len(df) < FRAME + 1:
        return None
    close = df["Close"].to_numpy(dtype=float)
    high = df["High"].to_numpy(dtype=float)
    low = df["Low"].to_numpy(dtype=float)
    open_ = df["Open"].to_numpy(dtype=float)
    vol = df["Volume"].to_numpy(dtype=float)
    idx = df.index
    n = len(df)

    c = pd.Series(close, index=idx)
    sma5 = c.rolling(5).mean().to_numpy()
    sma20 = c.rolling(20).mean().to_numpy()
    sma20_then = pd.Series(sma20, index=idx).shift(
        KNOBS["slope_lookback"]).to_numpy()
    sma50 = c.rolling(50).mean().to_numpy()
    prev_close = np.concatenate([[np.nan], close[:-1]])
    tr = np.maximum(high - low, np.maximum(np.abs(high - prev_close),
                                           np.abs(low - prev_close)))
    atr14 = pd.Series(tr, index=idx).rolling(KNOBS["atr_period"]).mean() \
        .to_numpy()
    above = close > sma20
    # PRODUCTION PARITY (review finding, reproduced): cumcount+1 counts
    # the False day that OPENS each group, so the first close above read
    # consec=2 and passed confirmation a day early. Group-CUMSUM of the
    # booleans is the exact production count (pinned prefix-by-prefix
    # against consec_closes_above in test_backtest_doctrine).
    _ab = pd.Series(above, index=idx)
    consec = _ab.astype(int).groupby((~_ab).cumsum()).cumsum().to_numpy()
    vol5 = pd.Series(vol, index=idx).rolling(5).mean().to_numpy()
    vol20 = pd.Series(vol, index=idx).rolling(20).mean().to_numpy()
    with np.errstate(divide="ignore", invalid="ignore"):
        vratio = np.round(np.where(vol20 > 0, vol5 / vol20, 1.0), 2)

    # frame-end aligned arrays (day j corresponds to absolute index
    # j + FRAME - 1)
    off = FRAME - 1
    rsi = rsi_windowed(close)
    bullish, confirms = macd_windowed(close)
    up_close = {L: np.concatenate([np.zeros(L - 1, dtype=bool),
                                   up_close_windowed(close, L)])
                for L in SWING_LOOKBACKS}
    ytd = ytd_aslived(close, idx)

    # momentum definitions (close-basis)
    mom_12_1 = np.full(n, np.nan)
    if n > 252:
        mom_12_1[252:] = close[252:] * 0.0 + (
            close[252 - 21:n - 21] / close[:n - 252] - 1.0) * 100.0
        # ^ close.shift(21)/close.shift(252) - 1, aligned at day i
    mom_6m = np.full(n, np.nan)
    if n > FRAME:
        mom_6m[FRAME:] = (close[FRAME:] / close[:n - FRAME] - 1.0) * 100.0

    # forward returns, Build 4 info_test convention: T+1 open -> T+1+h open
    fwd = {}
    for h in HORIZONS:
        f = np.full(n, np.nan)
        lim = n - (1 + h)
        if lim > 0:
            f[:lim] = (open_[1 + h:] / open_[1:1 + lim + h][:lim] - 1.0) \
                * 100.0
        fwd[h] = f

    # grading-day mask
    dates = idx.to_numpy()
    day_iso = np.array([str(d)[:10] for d in dates])
    reg = np.array([regime_by_day.get(s, "") for s in day_iso])
    mask = ((np.arange(n) >= off) & (day_iso >= start) & (day_iso <= end)
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

    rows = []
    for k, i in enumerate(days):
        j = i - off
        regime = CHASSIS_LABELS.get(reg[i], reg[i])
        c1 = bool(above[i])
        atr = atr14[i] if np.isfinite(atr14[i]) else None
        atr_break = c1 and atr is not None \
            and close[i] > sma20[i] + KNOBS["atr_mult"] * atr
        c2 = (consec[i] >= KNOBS["confirmation_closes"]) or atr_break
        c3 = regime in ("Risk-on / Trending", "Risk-on / Choppy")
        c4 = sma20[i] >= sma20_then[i]
        all_lite = c1 and c2 and c3 and c4
        ext = None if not atr else (close[i] - sma20[i]) / atr
        rw = int(runway[i]) if runway[i] >= 0 else None
        gap = rbasis[i] == 2

        def _grade(up_flag, use_runway=True):
            g = grade_setup(
                all_conditions_met=all_lite,
                extension_atr=ext, close=float(close[i]),
                sma5=float(sma5[i]) if np.isfinite(sma5[i]) else None,
                up_close_since_swing_low=bool(up_flag),
                rsi14=float(rsi[j]),
                quality_score=float(scores[k]),
                score_waived=False,
                breaker_status="clear",      # row 6 DROPPED (grade-lite):
                # fed the passing value so it can never fail a grade; the
                # honesty bridge quantifies what full-grade row 6 changes
                runway_sessions=(rw if use_runway else None),
                extension_guard_max=KNOBS["extension_guard_max"],
                rsi_min=KNOBS["rsi_min"], rsi_max=KNOBS["rsi_max"],
                score_min=KNOBS["score_min"],
                runway_min_sessions=KNOBS["runway_min_sessions"])
            letter = g["grade"]
            if use_runway and gap and letter == "A+":
                letter = "B"                 # coverage gap is NEVER A+
            return GRADE_CODE[letter]

        def _grade_fail(up_flag):
            g = grade_setup(
                all_conditions_met=all_lite, extension_atr=ext,
                close=float(close[i]),
                sma5=float(sma5[i]) if np.isfinite(sma5[i]) else None,
                up_close_since_swing_low=bool(up_flag),
                rsi14=float(rsi[j]), quality_score=float(scores[k]),
                score_waived=False, breaker_status="clear",
                runway_sessions=rw,
                extension_guard_max=KNOBS["extension_guard_max"],
                rsi_min=KNOBS["rsi_min"], rsi_max=KNOBS["rsi_max"],
                score_min=KNOBS["score_min"],
                runway_min_sessions=KNOBS["runway_min_sessions"])
            fail = (g.get("failing") or [None])[0] or ""
            return fail

        g20 = _grade(up_close[20][i])
        # first failing row for the production-lookback grade — the
        # C-bucket decomposition (extended vs broken vs knife) hangs on it.
        # A gap-demoted A+ (grade_setup passed everything, the coverage
        # gap denied A+) records 7 — _grade already applied the demotion,
        # so g20<3 there and the recompute below would read "" (review
        # finding: the old branch was unreachable).
        fail0 = _grade_fail(up_close[20][i]) if g20 < 3 else ""
        if g20 < 3 and gap and fail0 == "":
            fail0 = "7_runway_gap"
        rows.append((
            day_iso[i], t, g20, FAIL_CODE.get(fail0.split("_")[0]
                                              if fail0 else "", 0)
            if not fail0.startswith("7_runway_gap") else 7,
            bool(up_close[10][i]), bool(up_close[20][i]),
            bool(up_close[40][i]), c1, c3,
            _grade(up_close[10][i]), _grade(up_close[40][i]),
            _grade(up_close[20][i], use_runway=False),
            float(scores[k]), float(scores_np[k]),
            reg[i], float(rsi[j]),
            float(ytd[i]), float(mom_12_1[i]), float(mom_6m[i]),
            bool(close[i] > sma50[i]) if np.isfinite(sma50[i]) else False,
            int(rbasis[i]), rw if rw is not None else -1,
            *(float(fwd[h][i]) if np.isfinite(fwd[h][i]) else np.nan
              for h in HORIZONS),
        ))
    return rows


CHASSIS_LABELS = {"In-Trend-Full": "Risk-on / Trending",
                  "In-Trend-Throttled": "Risk-on / Choppy",
                  "Out-Defensive": "Caution", "Out-Risk-off": "Risk-off"}

FAIL_CODE = {"": 0, "1": 1, "2": 2, "3": 3, "4": 4, "5": 5, "6": 6, "7": 7}

COLS = (["date", "ticker", "grade", "fail_row", "up10", "up20", "up40",
         "c1_above", "c3_regime", "grade_L10", "grade_L40", "grade_no7",
         "score", "score_nopen", "chassis", "rsi", "ytd", "mom_12_1",
         "mom_6m", "above_50dma", "runway_basis", "runway"]
        + [f"fwd_{h}" for h in HORIZONS])


# ---------------------------------------------------------------- analysis
def _stats(v):
    v = v.dropna()
    if not len(v):
        return {"n": 0}
    return {"n": int(len(v)), "mean_pct": round(float(v.mean()), 3),
            "median_pct": round(float(v.median()), 3),
            "hit_rate_pct": round(float((v > 0).mean()) * 100, 1)}


def per_grade_tables(df, col="grade"):
    out = {}
    for span, d in (("full", df),
                    ("train", df[df.date <= TRAIN_END]),
                    ("validate", df[df.date > TRAIN_END])):
        t = {}
        for gname, gcode in GRADE_CODE.items():
            sel = d[d[col] == gcode]
            row = {"ticker_days": int(len(sel)),
                   "tickers": int(sel.ticker.nunique()),
                   "per_day_avg": round(len(sel)
                                        / max(1, d.date.nunique()), 2)}
            for h in HORIZONS:
                row[f"fwd_{h}d"] = _stats(sel[f"fwd_{h}"])
            t[gname] = row
        t["spread_aplus_minus_c"] = {
            f"fwd_{h}d": round(
                (t["A+"][f"fwd_{h}d"].get("mean_pct") or 0)
                - (t["C"][f"fwd_{h}d"].get("mean_pct") or 0), 3)
            for h in HORIZONS
            if t["A+"][f"fwd_{h}d"]["n"] and t["C"][f"fwd_{h}d"]["n"]}
        out[span] = t
    return out


def day_cluster_bootstrap(df, h=20, n_boot=2000, seed=42):
    """A+ - C mean-spread significance with DAY resampling: ticker-days
    within a date are one cluster (cross-sectional correlation), so days —
    not observations — are the unit. Serial overlap of the h-day horizon
    remains; declared in the caveats."""
    col = f"fwd_{h}"
    d = df[["date", "grade", col]].dropna()
    dates = d.date.unique()
    by_date = {k: v for k, v in d.groupby("date")}
    rng = np.random.default_rng(seed)
    obs_a = d[d.grade == 3][col].mean()
    obs_c = d[d.grade == 1][col].mean()
    obs = obs_a - obs_c
    spreads = []
    for _ in range(n_boot):
        pick = rng.choice(dates, size=len(dates), replace=True)
        sub = pd.concat([by_date[p] for p in pick], ignore_index=True)
        a = sub[sub.grade == 3][col]
        cc = sub[sub.grade == 1][col]
        if len(a) and len(cc):
            spreads.append(a.mean() - cc.mean())
    s = np.array(spreads)
    return {"observed_spread_pct": round(float(obs), 3),
            "boot_ci_2_5": round(float(np.percentile(s, 2.5)), 3),
            "boot_ci_97_5": round(float(np.percentile(s, 97.5)), 3),
            "p_spread_le_0": round(float((s <= 0).mean()), 4),
            "n_boot": len(s)}


def persistence(df):
    """A+ spell lengths in trading days (consecutive grading days per
    ticker at A+)."""
    lengths = []
    for _, g in df[["ticker", "date", "grade"]].sort_values(
            ["ticker", "date"]).groupby("ticker"):
        run = 0
        for code in g.grade:
            if code == 3:
                run += 1
            elif run:
                lengths.append(run)
                run = 0
        if run:
            lengths.append(run)
    if not lengths:
        return {"spells": 0}
    a = np.array(lengths)
    return {"spells": int(len(a)), "mean_days": round(float(a.mean()), 2),
            "median_days": float(np.median(a)),
            "p90_days": float(np.percentile(a, 90)),
            "max_days": int(a.max()),
            "share_1_day_pct": round(float((a == 1).mean()) * 100, 1)}


def aplus_counts(df):
    per_day = df[df.grade == 3].groupby("date").size()
    all_days = df.date.nunique()
    by_year = {}
    for y, g in df.groupby(df.date.str[:4]):
        c = g[g.grade == 3].groupby("date").size()
        by_year[y] = {"mean_per_day": round(float(c.reindex(
            g.date.unique()).fillna(0).mean()), 2),
            "max": int(c.max()) if len(c) else 0}
    return {"mean_per_day": round(float(per_day.reindex(
                df.date.unique()).fillna(0).mean()), 2),
            "p95_per_day": float(np.percentile(per_day.reindex(
                df.date.unique()).fillna(0), 95)),
            "max_per_day": int(per_day.max()) if len(per_day) else 0,
            "share_days_zero_pct": round(
                (1 - per_day.reindex(df.date.unique()).fillna(0)
                 .astype(bool).mean()) * 100, 1),
            "by_year": by_year}


def c_decomposition(df):
    """The C bucket by first failing row, plus the above/below-SMA20
    split — committed to the results JSON so the report's decomposition
    is re-runnable (review finding: it lived only in the frame)."""
    c = df[df.grade == 1]
    out = {}
    names = {1: "row1_conditions", 2: "row2_extension", 3: "row3_approach",
             4: "row4_rsi", 5: "row5_score", 7: "row7_runway_gap"}
    for code, name in names.items():
        sel = c[c.fail_row == code]
        if len(sel):
            out[name] = {"ticker_days": int(len(sel)),
                         "fwd_20d": _stats(sel.fwd_20)}
    out["below_sma20"] = {
        "ticker_days": int((~c.c1_above).sum()),
        "fwd_20d": _stats(c[~c.c1_above].fwd_20)}
    out["above_sma20"] = {
        "ticker_days": int(c.c1_above.sum()),
        "fwd_20d": _stats(c[c.c1_above].fwd_20)}
    return out


def momentum_module(df):
    """Deciles per day per definition per cut -> fwd 20d. Three cuts
    (ruling 3): unconditional / above-50DMA (analysis construct) /
    score>=50 (the production gate)."""
    out = {}
    cuts = {"unconditional": pd.Series(True, index=df.index),
            "above_50dma": df.above_50dma,
            # labeled honestly: the universe gate's own score uses a
            # 1y-anchored YTD; this cut uses the grade-path (6mo-frame)
            # score — an approximation of the gate, not the gate itself
            "score_ge_50_framescore": df.score >= 50}
    for defname in ("ytd", "mom_12_1", "mom_6m"):
        out[defname] = {}
        base = df[["date", defname, "fwd_20"]].rename(
            columns={defname: "m"})
        for cutname, cmask in cuts.items():
            d = base[cmask & base.m.notna() & base.fwd_20.notna()]
            if len(d) < 1000:
                out[defname][cutname] = {"n": int(len(d)),
                                         "note": "insufficient"}
                continue
            dec = d.groupby("date").m.transform(
                lambda s: pd.qcut(s.rank(method="first"), 10,
                                  labels=False, duplicates="drop"))
            tbl = {}
            for q in range(10):
                sel = d[dec == q]
                tbl[f"d{q + 1}"] = _stats(sel.fwd_20)
            tbl["d10_minus_d1_mean_pct"] = round(
                (tbl["d10"].get("mean_pct") or 0)
                - (tbl["d1"].get("mean_pct") or 0), 3)
            out[defname][cutname] = tbl
    return out


def penalty_module(df):
    """Score-with vs score-without the YTD overextension penalty:
    membership deltas at the 50 gate and the 75 A+ bar, and forward
    returns of the affected ticker-days."""
    aff = df[df.score != df.score_nopen]
    cross50 = aff[(aff.score < 50) & (aff.score_nopen >= 50)]
    cross75 = aff[(aff.score < 75) & (aff.score_nopen >= 75)]
    return {
        "ticker_days_penalized": int(len(aff)),
        "share_of_all_pct": round(len(aff) / max(1, len(df)) * 100, 2),
        "cross_50_gate": {"ticker_days": int(len(cross50)),
                          "tickers": int(cross50.ticker.nunique()),
                          "fwd_20d": _stats(cross50.fwd_20)},
        "cross_75_bar": {"ticker_days": int(len(cross75)),
                         "tickers": int(cross75.ticker.nunique()),
                         "fwd_20d": _stats(cross75.fwd_20)},
        "penalized_fwd_20d": _stats(aff.fwd_20),
        "unpenalized_same_days_fwd_20d": _stats(
            df[df.score == df.score_nopen].fwd_20),
    }


# ---------------------------------------------------------------- main
def run(smoke=False, out_path=None):
    # pinned inputs, asserted BEFORE any work (2026-08-10 ruling): a
    # moved cache must fail loudly and NAME itself, not silently
    # produce a different study
    from study_inputs import assert_pinned_inputs
    if not smoke:
        assert_pinned_inputs(["prices_manifest", "earnings_dates",
                              "regime_daily"], label="Layer A")
    with open(REGIME_PATH) as f:
        regime = json.load(f)
    regime_by_day = regime["states"]
    end = END or regime["provenance"]["series_range"][1]

    with open(EARNINGS_PATH) as f:
        earnings = json.load(f)

    files = sorted(os.listdir(PRICES))
    if smoke:
        files = files[:20]
    all_rows = []
    skipped = []
    for i, fn in enumerate(files, 1):
        t = fn[:-4]
        r = process_ticker(t, os.path.join(PRICES, fn),
                           earnings.get(t) or {"status": "missing"},
                           regime_by_day, START, end)
        if r:
            all_rows.extend(r)
        else:
            skipped.append(t)
        if i % 50 == 0:
            print(f"  [{i}/{len(files)}] rows={len(all_rows)}", flush=True)

    df = pd.DataFrame(all_rows, columns=COLS)
    print(f"graded {df.ticker.nunique()} tickers, "
          f"{len(df):,} ticker-days, {df.date.nunique()} days; "
          f"skipped {len(skipped)}")

    results = {
        "schema": "backtest-doctrine-1",
        "design": {
            "window": [START, end], "train_end": TRAIN_END,
            "frame_bars": FRAME, "horizons": list(HORIZONS),
            "grade": "grade-lite: rows 1-5+7, minus c5, minus row 6; "
                     "row 7 None-split (coverage gap never A+)",
            "fills": "T+1 open -> T+1+h open (Build 4 info_test)",
            "knobs": KNOBS, "swing_lookbacks": list(SWING_LOOKBACKS),
            "earnings_gap_days": EARNINGS_GAP_DAYS,
        },
        "population": {
            "tickers_graded": int(df.ticker.nunique()),
            "ticker_days": int(len(df)),
            "trading_days": int(df.date.nunique()),
            "skipped": skipped,
            "runway_basis_share": {
                {0: "known", 1: "no_entity", 2: "coverage_gap"}[k]:
                    round(v / len(df) * 100, 2)
                for k, v in df.runway_basis.value_counts().items()},
        },
        "grade_distribution": {
            span: dict(Counter(d.grade.map(
                {v: k for k, v in GRADE_CODE.items()})))
            for span, d in (("full", df),
                            ("train", df[df.date <= TRAIN_END]),
                            ("validate", df[df.date > TRAIN_END]))},
        "per_grade": per_grade_tables(df),
        "significance": {
            "full_fwd20": day_cluster_bootstrap(df, 20),
            "validate_fwd20": day_cluster_bootstrap(
                df[df.date > TRAIN_END], 20),
        },
        "regime_conditioned": {
            st: per_grade_tables(df[df.chassis == st])["full"]
            for st in df.chassis.unique()},
        "aplus_counts": aplus_counts(df),
        "aplus_persistence": persistence(df[["ticker", "date", "grade"]]),
        "sensitivity": {
            "swing_lookback": {
                "L10": per_grade_tables(df, "grade_L10")["full"],
                "L20_production": per_grade_tables(df, "grade")["full"],
                "L40": per_grade_tables(df, "grade_L40")["full"]},
            "no_row7": per_grade_tables(df, "grade_no7")["full"],
        },
        "c_decomposition": c_decomposition(df),
        "aplus_vs_b": {
            span: {
                "aplus_fwd20": _stats(d[d.grade == 3].fwd_20),
                "b_fwd20": _stats(d[d.grade == 2].fwd_20)}
            for span, d in (("train", df[df.date <= TRAIN_END]),
                            ("validate", df[df.date > TRAIN_END]))},
        "lookback_inertness": {
            "flag_disagree_pct": {
                "10v20": round(float((df.up10 != df.up20).mean()) * 100, 2),
                "20v40": round(float((df.up20 != df.up40).mean()) * 100, 2)},
            "ticker_days_grade_changes": int(
                ((df.grade != df.grade_L10)
                 | (df.grade != df.grade_L40)).sum())},
        "momentum": momentum_module(df),
        "ytd_penalty": penalty_module(df),
        "provenance": {
            "regime_series": regime["provenance"],
            "price_cache_files": len(files),
            "generated_from_commit": _git_head(),
        },
    }
    results["input_hash"] = _df_hash(df)
    frame_path = os.path.join(
        CACHE, "master_frame_smoke.parquet" if smoke
        else "master_frame.parquet")
    try:
        df.to_parquet(frame_path)
        print(f"master frame -> {frame_path}")
    except Exception as e:
        df.to_csv(frame_path.replace(".parquet", ".csv.gz"),
                  index=False, compression="gzip")
        print(f"master frame -> csv.gz (parquet unavailable: {e})")

    out_path = out_path or os.path.join(
        REPO, "docs", "backtest-doctrine-results.json")
    with open(out_path, "w") as f:
        json.dump(results, f, indent=1)
    print(f"results -> {out_path}")
    return results, df


def _git_head():
    import subprocess
    return subprocess.run(["git", "rev-parse", "--short", "HEAD"],
                          capture_output=True, text=True,
                          cwd=REPO).stdout.strip()


def _df_hash(df):
    return hashlib.sha256(
        pd.util.hash_pandas_object(df, index=False).values.tobytes()
    ).hexdigest()[:16]


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--smoke", action="store_true")
    ap.add_argument("--out")
    a = ap.parse_args()
    run(smoke=a.smoke, out_path=a.out)


if __name__ == "__main__":
    main()
