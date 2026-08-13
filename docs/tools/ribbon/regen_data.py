#!/usr/bin/env python3
"""
Regenerate ribbon_data.json WITH VOLUME. Standalone throwaway research —
not part of SignalAgent/MarketPulse; no commits.

Shape: {"generated": "YYYY-MM-DD", "symbols": {SYM: {dates,o,h,l,c,v}}}
OHLC rounded 2dp, volume integer, six equal-length chronological arrays,
rows dropped where ANY of OHLCV is NaN.

Also checks EMPIRICALLY whether volume is split-adjusted under
auto_adjust=True, using WFC's 2:1 split (2006-08-14): auto_adjust
rescales OHLC only; if volume is raw, average daily volume steps up
~2x across the split.

Run: python3 regen_data.py
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


def main():
    out = {"generated": pd.Timestamp.today().strftime("%Y-%m-%d"),
           "symbols": {}}
    sizes = {}
    failed = []
    for sym in SYMBOLS:
        try:
            df = yf.Ticker(sym).history(period="max", auto_adjust=True)
            df = df[["Open", "High", "Low", "Close", "Volume"]].dropna()
            if len(df) < 500:
                failed.append((sym, f"only {len(df)} bars"))
                continue
        except Exception as e:
            failed.append((sym, type(e).__name__))
            continue
        blob = {
            "dates": [str(d)[:10] for d in df.index],
            "o": [round(float(v), 2) for v in df["Open"]],
            "h": [round(float(v), 2) for v in df["High"]],
            "l": [round(float(v), 2) for v in df["Low"]],
            "c": [round(float(v), 2) for v in df["Close"]],
            "v": [int(v) for v in df["Volume"]],
        }
        assert len({len(blob[k]) for k in blob}) == 1, sym
        out["symbols"][sym] = blob
        sizes[sym] = len(json.dumps(blob, separators=(",", ":")))
        print(f"  {sym:5} {len(df):>7,} bars  "
              f"{blob['dates'][0]} -> {blob['dates'][-1]}  "
              f"(~{sizes[sym] / 1e6:.2f} MB)")
    for sym, why in failed:
        print(f"  {sym:5} FAILED: {why}")

    with open("ribbon_data.json", "w") as f:
        json.dump(out, f, separators=(",", ":"))
    total = os.path.getsize("ribbon_data.json")
    print(f"\n  ribbon_data.json: {total / 1e6:.1f} MB, "
          f"{len(out['symbols'])} symbols")

    # --- empirical split-adjustment check on WFC's 2:1 (2006-08-14) ---
    if "WFC" in out["symbols"]:
        w = out["symbols"]["WFC"]
        dates = w["dates"]
        v = w["v"]
        c = w["c"]

        def win(day, n=30):
            i = min(range(len(dates)),
                    key=lambda k: abs(pd.Timestamp(dates[k])
                                      - pd.Timestamp(day)))
            return (np.mean(v[max(0, i - n):i]),
                    np.mean(v[i + 1:i + 1 + n]),
                    c[max(0, i - 1)], c[min(len(c) - 1, i + 1)])
        pre_v, post_v, pre_c, post_c = win("2006-08-14")
        print(f"\n  VOLUME ADJUSTMENT CHECK — WFC 2:1 split 2006-08-14:")
        print(f"    avg volume 30d before: {pre_v:,.0f}   "
              f"30d after: {post_v:,.0f}   ratio {post_v / pre_v:.2f}x")
        print(f"    close just before/after (auto-adjusted): "
              f"{pre_c} / {post_c}  (no price step = OHLC adjusted)")
        if post_v / pre_v > 1.6:
            print("    => volume steps ~2x across the split: yfinance "
                  "reports RAW (unadjusted) volume under auto_adjust=True."
                  "\n       OHLC is split-adjusted; VOLUME IS NOT — treat "
                  "long-history volume as a stepped series.")
        else:
            print("    => no ~2x step: volume appears split-adjusted in "
                  "this data; treat the doctrine claim as not confirmed "
                  "for this source.")


if __name__ == "__main__":
    main()
