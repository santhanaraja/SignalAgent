#!/usr/bin/env python3
"""
Standalone exploratory probe — EMA-ribbon trend + dip-and-turn timing.
NOT part of SignalAgent/MarketPulse. Throwaway research; no commits.

Symbols: SPY, QQQ, WFC. Data: yfinance period="max", auto_adjust=True.

EMAs match TradingView's ta.ema: SMA(n) seed at index n-1, then
alpha = 2/(n+1) recursion — NOT pandas' first-value seed. The seeding
difference persists for years at n=2650, so it is implemented
explicitly and (for SPY) both seedings are printed for chart checking.

Definitions (post-warmup rows only — rows lacking ema2650 or sma200
are dropped):
  stacked  = ema105 > ema465 > ema2650
  above200 = close > sma200
  setup    = stacked AND above200
  dip      = within the last k bars, low_j <= anchor_ema_j * (1 + tol)
  turn     = close > prev close AND close > ema21
  signal   = setup AND dip AND turn
Base case: anchor = ema105, tol = 0.005, k = 5.

Forward returns are open-to-open: T+1 open -> T+1+h open (no fill on
the signal close). Horizons 5, 10, 20, 40.

Run: python3 probe.py
"""

import sys

import numpy as np
import pandas as pd
import yfinance as yf

SYMBOLS = ["SPY", "QQQ", "WFC"]
HORIZONS = (5, 10, 20, 40)
BASE_TOL, BASE_K = 0.005, 5
GRID_TOLS = (0.0, 0.005, 0.01)
GRID_KS = (3, 5, 10)


def ema_tv(close: np.ndarray, n: int) -> np.ndarray:
    """TradingView ta.ema: value at index n-1 = SMA of the first n
    closes, then e[i] = a*close[i] + (1-a)*e[i-1], a = 2/(n+1).
    NaN before index n-1."""
    e = np.full(len(close), np.nan)
    if len(close) < n:
        return e
    e[n - 1] = close[:n].mean()
    a = 2.0 / (n + 1.0)
    for i in range(n, len(close)):
        e[i] = a * close[i] + (1.0 - a) * e[i - 1]
    return e


def ema_first_seed(close: pd.Series, n: int) -> pd.Series:
    """pandas default: ewm(span=n, adjust=False) — seeds at the first
    value. For the SPY seeding comparison only."""
    return close.ewm(span=n, adjust=False).mean()


def fwd_open_to_open(open_: np.ndarray, h: int) -> np.ndarray:
    """fwd_h[t] = open[t+1+h] / open[t+1] - 1, in percent."""
    n = len(open_)
    out = np.full(n, np.nan)
    for t in range(n - 1 - h):
        a, b = open_[t + 1], open_[t + 1 + h]
        if np.isfinite(a) and np.isfinite(b) and a > 0:
            out[t] = (b / a - 1.0) * 100.0
    return out


def dip_flag(low: np.ndarray, anchor: np.ndarray, tol: float,
             k: int) -> np.ndarray:
    """dip[t] = any j in [t-k+1, t]: low[j] <= anchor[j] * (1+tol)."""
    touch = low <= anchor * (1.0 + tol)
    touch = np.where(np.isfinite(anchor), touch, False)
    s = pd.Series(touch.astype(float))
    return s.rolling(k, min_periods=1).max().to_numpy() >= 1.0


def stats_block(mask: np.ndarray, fwd: dict) -> dict:
    out = {"n": int(mask.sum())}
    for h in HORIZONS:
        v = fwd[h][mask]
        v = v[np.isfinite(v)]
        out[f"fwd{h}"] = round(float(v.mean()), 3) if len(v) else None
    v20 = fwd[20][mask]
    v20 = v20[np.isfinite(v20)]
    out["hit20"] = round(float((v20 > 0).mean()) * 100, 1) if len(v20) \
        else None
    return out


def fmt_row(label, s):
    return (f"  {label:22} n={s['n']:6,}  "
            + "  ".join(f"f{h}={s[f'fwd{h}']!s:>7}" for h in HORIZONS)
            + f"  hit20={s['hit20']!s:>5}%")


def analyze(sym: str):
    print("=" * 78)
    print(f"  {sym}")
    print("=" * 78)
    df = yf.Ticker(sym).history(period="max", auto_adjust=True)
    df = df[["Open", "High", "Low", "Close"]].dropna()
    close = df["Close"].to_numpy(dtype=float)
    low = df["Low"].to_numpy(dtype=float)
    open_ = df["Open"].to_numpy(dtype=float)
    idx = df.index

    e21 = ema_tv(close, 21)
    e105 = ema_tv(close, 105)
    e465 = ema_tv(close, 465)
    e2650 = ema_tv(close, 2650)
    sma200 = pd.Series(close).rolling(200).mean().to_numpy()

    if sym == "SPY":
        f105 = ema_first_seed(df["Close"], 105).to_numpy()
        f2650 = ema_first_seed(df["Close"], 2650).to_numpy()
        print("\nSPY seeding comparison — last 5 rows "
              "(check against TradingView):")
        print(f"  {'date':12} {'ema105 SMA-seed':>16} "
              f"{'ema105 1st-seed':>16} {'ema2650 SMA-seed':>17} "
              f"{'ema2650 1st-seed':>17}")
        for i in range(len(df) - 5, len(df)):
            print(f"  {str(idx[i])[:10]:12} {e105[i]:>16.4f} "
                  f"{f105[i]:>16.4f} {e2650[i]:>17.4f} {f2650[i]:>17.4f}")

    # post-warmup mask
    valid = np.isfinite(e2650) & np.isfinite(sma200)
    first = int(np.argmax(valid))
    n_total = len(df)
    n_valid = int(valid.sum())
    print(f"\n1. COVERAGE")
    print(f"  total bars           : {n_total:,}  "
          f"({str(idx[0])[:10]} -> {str(idx[-1])[:10]})")
    print(f"  lost to 2650 warmup  : {n_total - n_valid:,}")
    print(f"  usable               : {n_valid:,}  "
          f"({str(idx[first])[:10]} -> {str(idx[-1])[:10]})")

    stacked = (e105 > e465) & (e465 > e2650) & valid
    above200 = (close > sma200) & valid
    setup = stacked & above200
    above2650 = (close > e2650) & valid
    turn = np.zeros(n_total, dtype=bool)
    turn[1:] = (close[1:] > close[:-1]) & (close[1:] > e21[1:])
    fwd = {h: fwd_open_to_open(open_, h) for h in HORIZONS}

    print(f"\n2. BIND RATES (share of {n_valid:,} usable days)")
    for name, m in (("stacked", stacked), ("above200", above200),
                    ("setup", setup), ("close>ema2650", above2650)):
        print(f"  {name:16}: {m.sum() / n_valid * 100:5.1f}%")

    # 3. does the 2650 earn its keep?
    slope465 = np.zeros(n_total, dtype=bool)
    slope465[21:] = e465[21:] > e465[:-21]
    proxy = (e105 > e465) & slope465 & valid
    agree = float((stacked[valid] == proxy[valid]).mean()) * 100
    pre = (e105 > e465) & above200 & valid
    n_pre = int(pre.sum())
    also = float(((e465 > e2650)[pre]).mean()) * 100 if n_pre else 0.0
    print(f"\n3. DOES THE 2650 EARN ITS KEEP?")
    print(f"  (a) stacked vs (105>465 AND 465 rising over 21 bars): "
          f"{agree:.1f}% agreement")
    print(f"  (b) of days passing (105>465 AND above200) [n={n_pre:,}], "
          f"{also:.1f}% also pass 465>2650")

    def full_case(anchor, label):
        dip = dip_flag(low, anchor, BASE_TOL, BASE_K)
        sig = setup & dip & turn
        print(f"\n  --- anchor = {label}, tol={BASE_TOL}, k={BASE_K} ---")
        print(fmt_row("all usable days", stats_block(valid, fwd)))
        print(fmt_row("setup only", stats_block(setup, fwd)))
        print(fmt_row("signal days", stats_block(sig, fwd)))
        yrs = n_valid / 252.0
        print(f"  signals: {int(sig.sum())} total, "
              f"{sig.sum() / yrs:.1f}/year over {yrs:.1f} usable years")
        return sig

    print(f"\n4/5. THREE-WAY DECOMPOSITION + FREQUENCY (base case)")
    sig_base = full_case(e105, "ema105")

    print(f"\n6. ANCHOR VARIANT")
    full_case(e465, "ema465")

    print(f"\n7. FRAGILITY CHECK — 3x3 grid over tol x k, anchor=ema105.")
    print(f"   (Reading: does the result SURVIVE across the grid — "
          f"not which cell is best. No cell is recommended.)")
    print(f"  {'':10}" + "".join(f"{'k=' + str(k):>22}" for k in GRID_KS))
    for tol in GRID_TOLS:
        cells = []
        for k in GRID_KS:
            dip = dip_flag(low, e105, tol, k)
            sig = setup & dip & turn
            s = stats_block(sig, fwd)
            cells.append(f"n={s['n']:5,} f20={s['fwd20']!s:>6}")
        print(f"  tol={tol:<6}" + "".join(f"{c:>22}" for c in cells))

    out = pd.DataFrame({
        "date": [str(d)[:10] for d in idx],
        "open": open_, "high": df["High"].to_numpy(), "low": low,
        "close": close, "ema21": e21, "ema105": e105, "ema465": e465,
        "ema2650": e2650, "sma200": sma200,
        "stacked": stacked, "above200": above200, "setup": setup,
        "turn": turn, "signal_base": sig_base,
    })
    path = f"ribbon_{sym}.csv"
    out.to_csv(path, index=False)
    print(f"\n8. wrote {path} ({len(out):,} rows)")


if __name__ == "__main__":
    for sym in SYMBOLS:
        analyze(sym)
    print("\n(Exploratory probe. Results reported without recommendation.)")
