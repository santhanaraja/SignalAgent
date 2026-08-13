#!/usr/bin/env python3
"""
Standalone probe, part 2 — cross-sectional ribbon table + data file.
NOT part of SignalAgent/MarketPulse. Throwaway research; no commits.

Part 1: ribbon_data.json — OHLC (2dp) for 22 symbols, period="max",
auto_adjust=True, NaN rows dropped, chronological.

Part 2: ONE fixed configuration, no sweeping:
  MAs   : EMA21 / EMA105 / EMA465 (TradingView SMA-seed) + SMA200
  trend : EMA105 > EMA465          (variant B adds close > SMA200)
  entry : low <= EMA105*1.005 within last 5 bars
          AND close > EMA21 AND close > prev close
  exit  : close < EMA21
  fills : next open after the signal close; non-overlapping positions;
          re-entry once flat; $10,000; cash earns 0; no costs
Table sorted by annualized volatility ascending. The question: does
the CAGR delta or the fwd20 timing gap sort with volatility, or is it
scatter?

Run: python3 part2.py
"""

import json
import os

import numpy as np
import pandas as pd
import yfinance as yf

SYMBOLS = ["SPY", "QQQ", "DIA", "IWM", "MDY",
           "XLK", "XLF", "XLE", "XLV", "XLI", "XLP", "XLY", "XLU", "XLB",
           "EFA", "EEM", "EWJ",
           "TLT", "IEF", "HYG",
           "GLD", "IYR",
           "WFC"]


def ema_tv(close, n):
    e = np.full(len(close), np.nan)
    if len(close) < n:
        return e
    e[n - 1] = close[:n].mean()
    a = 2.0 / (n + 1.0)
    for i in range(n, len(close)):
        e[i] = a * close[i] + (1.0 - a) * e[i - 1]
    return e


def fwd20(open_):
    n = len(open_)
    out = np.full(n, np.nan)
    for t in range(n - 21):
        a, b = open_[t + 1], open_[t + 21]
        if np.isfinite(a) and np.isfinite(b) and a > 0:
            out[t] = (b / a - 1.0) * 100.0
    return out


def fetch_all():
    data, failed = {}, []
    for sym in SYMBOLS:
        try:
            df = yf.Ticker(sym).history(period="max", auto_adjust=True)
            df = df[["Open", "High", "Low", "Close"]].dropna()
            if len(df) < 500:
                failed.append((sym, f"only {len(df)} bars"))
                continue
            data[sym] = df
        except Exception as e:
            failed.append((sym, type(e).__name__))
    return data, failed


def write_json(data):
    out = {"generated": pd.Timestamp.today().strftime("%Y-%m-%d"),
           "symbols": {}}
    for sym, df in data.items():
        out["symbols"][sym] = {
            "dates": [str(d)[:10] for d in df.index],
            "o": [round(float(v), 2) for v in df["Open"]],
            "h": [round(float(v), 2) for v in df["High"]],
            "l": [round(float(v), 2) for v in df["Low"]],
            "c": [round(float(v), 2) for v in df["Close"]],
        }
    with open("ribbon_data.json", "w") as f:
        json.dump(out, f, separators=(",", ":"))
    return os.path.getsize("ribbon_data.json")


def run_symbol(df, use_sma200):
    close = df["Close"].to_numpy(dtype=float)
    low = df["Low"].to_numpy(dtype=float)
    open_ = df["Open"].to_numpy(dtype=float)
    n = len(df)
    e21 = ema_tv(close, 21)
    e105 = ema_tv(close, 105)
    e465 = ema_tv(close, 465)
    sma200 = pd.Series(close).rolling(200).mean().to_numpy()

    valid = np.isfinite(e465) & np.isfinite(sma200)
    filt = (e105 > e465) & valid
    if use_sma200:
        filt = filt & (close > sma200)
    touch = np.where(np.isfinite(e105), low <= e105 * 1.005, False)
    dip = pd.Series(touch.astype(float)).rolling(5, min_periods=1) \
        .max().to_numpy() >= 1.0
    turn = np.zeros(n, dtype=bool)
    turn[1:] = (close[1:] > close[:-1]) & (close[1:] > e21[1:])
    entry_sig = filt & dip & turn
    exit_sig = np.where(np.isfinite(e21), close < e21, False)

    # event loop: signals at close T act at open T+1
    equity = 10_000.0
    shares = 0.0
    in_pos = False
    pend = None                      # "buy" | "sell"
    curve = np.full(n, np.nan)
    trades = 0
    days_in = 0
    first = int(np.argmax(valid))
    for t in range(first, n):
        if pend == "buy" and not in_pos and np.isfinite(open_[t]):
            shares = equity / open_[t]
            in_pos = True
            trades += 1
        elif pend == "sell" and in_pos and np.isfinite(open_[t]):
            equity = shares * open_[t]
            shares = 0.0
            in_pos = False
        pend = None
        if in_pos:
            equity_mark = shares * close[t]
            days_in += 1
        else:
            equity_mark = equity
        curve[t] = equity_mark
        if in_pos:
            equity = equity_mark      # carry the mark
            if exit_sig[t]:
                pend = "sell"
        else:
            if entry_sig[t]:
                pend = "buy"

    v = curve[first:]
    ret = v[1:] / v[:-1] - 1.0
    yrs = len(v) / 252.0
    cagr = ((v[-1] / v[0]) ** (1 / yrs) - 1) * 100
    peak = np.maximum.accumulate(v)
    mdd = float((v / peak - 1.0).min()) * 100

    bh = close[first:]
    bh_ret = bh[1:] / bh[:-1] - 1.0
    bh_cagr = ((bh[-1] / bh[0]) ** (1 / yrs) - 1) * 100
    bh_peak = np.maximum.accumulate(bh)
    bh_mdd = float((bh / bh_peak - 1.0).min()) * 100
    ann_vol = float(bh_ret.std()) * np.sqrt(252) * 100

    f20 = fwd20(open_)
    sig_f = f20[entry_sig]
    sig_f = sig_f[np.isfinite(sig_f)]
    fonly = filt & ~entry_sig
    fo_f = f20[fonly]
    fo_f = fo_f[np.isfinite(fo_f)]
    gap = (float(sig_f.mean()) - float(fo_f.mean())) \
        if len(sig_f) and len(fo_f) else np.nan

    return {"vol": ann_vol, "years": yrs, "trades": trades,
            "tpy": trades / yrs, "tim": days_in / len(v) * 100,
            "cagr": cagr, "bh_cagr": bh_cagr, "delta": cagr - bh_cagr,
            "mdd": mdd, "bh_mdd": bh_mdd, "gap": gap}


def spearman(x, y):
    rx = pd.Series(x).rank()
    ry = pd.Series(y).rank()
    return float(rx.corr(ry))


def table(data, use_sma200, label):
    rows = []
    for sym, df in data.items():
        r = run_symbol(df, use_sma200)
        r["sym"] = sym
        rows.append(r)
    rows.sort(key=lambda r: r["vol"])
    print(f"\n{'=' * 118}")
    print(f"  VARIANT: {label}   (sorted by annualized volatility, asc)")
    print(f"{'=' * 118}")
    hdr = (f"  {'sym':5}{'vol%':>6}{'years':>7}{'trades':>8}{'tr/yr':>7}"
           f"{'TIM%':>7}{'CAGR%':>8}{'B&H%':>8}{'delta':>8}"
           f"{'mDD%':>8}{'bhDD%':>8}{'f20gap':>8}")
    print(hdr)
    for r in rows:
        print(f"  {r['sym']:5}{r['vol']:>6.1f}{r['years']:>7.1f}"
              f"{r['trades']:>8}{r['tpy']:>7.1f}{r['tim']:>7.1f}"
              f"{r['cagr']:>8.2f}{r['bh_cagr']:>8.2f}{r['delta']:>8.2f}"
              f"{r['mdd']:>8.1f}{r['bh_mdd']:>8.1f}{r['gap']:>8.2f}")
    vols = [r["vol"] for r in rows]
    deltas = [r["delta"] for r in rows]
    gaps = [r["gap"] for r in rows]
    print(f"\n  Spearman(vol, CAGR delta) = {spearman(vols, deltas):+.2f}"
          f"    Spearman(vol, f20 gap) = {spearman(vols, gaps):+.2f}")
    return rows


def main():
    data, failed = fetch_all()
    print("PART 1 — DATA FILE")
    for sym in SYMBOLS:
        if sym in data:
            df = data[sym]
            print(f"  {sym:5} {len(df):>7,} bars  "
                  f"{str(df.index[0])[:10]} -> {str(df.index[-1])[:10]}")
    for sym, why in failed:
        print(f"  {sym:5} FAILED: {why}")
    size = write_json(data)
    print(f"  ribbon_data.json written: {size / 1e6:.1f} MB, "
          f"{len(data)} symbols")

    print("\nPART 2 — CROSS-SECTIONAL TABLE (fixed config, no sweeps)")
    table(data, False, "trend = EMA105 > EMA465")
    table(data, True, "trend = EMA105 > EMA465 AND close > SMA200")


if __name__ == "__main__":
    main()
