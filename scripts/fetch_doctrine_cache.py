#!/usr/bin/env python3
"""
Build 5 data layer — one-time fetch of the two caches the doctrine
backtest needs. Gitignored like Build 4's cache; the REPORT and the
results JSONs are what get committed.

Two caches:
  1. OHLCV per pool ticker (data/doctrine_cache/prices/<T>.csv).
     Build 4's cache is Open+Close only; ATR14 and the swing low need
     High/Low, so this is a NEW fetch, not a reuse.
  2. Historical earnings REPORT dates per ticker
     (data/doctrine_cache/earnings_dates.json). yfinance's
     get_earnings_dates() is the source and it now works — lxml was
     missing from requirements until this build (see
     test_earnings_fallback.py).

Resumable: an existing per-ticker file is skipped, so a rate-limited or
interrupted run can simply be re-run.

Run: python3 scripts/fetch_doctrine_cache.py [--prices] [--earnings]
"""

import argparse
import datetime
import json
import os
import sys
import time

REPO = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
sys.path.insert(0, REPO)

CACHE = os.path.join(REPO, "data", "doctrine_cache")
PRICES = os.path.join(CACHE, "prices")
EARNINGS_PATH = os.path.join(CACHE, "earnings_dates.json")

START = "2018-06-01"          # >=126-bar warmup before the 2020-01 window
POOL_PATH = os.path.join(REPO, "data", "universe_cache")


def pool_tickers():
    """The 533-name pool: S&P 500 members ∪ the ETF holding lists. The
    candidates artifact stores only COUNTS, so the list is reconstructed
    from the same caches universe_source.build_universe_candidates reads."""
    names = set()
    sp = os.path.join(POOL_PATH, "sp500.json")
    if os.path.exists(sp):
        with open(sp) as f:
            names.update((json.load(f) or {}).get("tickers") or [])
    for fn in sorted(os.listdir(POOL_PATH)):
        if not fn.startswith("etf_") or not fn.endswith(".json"):
            continue
        with open(os.path.join(POOL_PATH, fn)) as f:
            blob = json.load(f) or {}
        # the ETF caches key their members under "tickers" (a flat list of
        # symbols), same as sp500.json — not a "holdings" object list
        for t in (blob.get("tickers") or []):
            if isinstance(t, str) and t.strip():
                names.add(t.strip())
    return sorted(n for n in names if n and n.strip())


def fetch_prices(tickers, sleep=0.25):
    os.makedirs(PRICES, exist_ok=True)
    import yfinance as yf
    import pandas as pd
    done = fail = skip = 0
    for i, t in enumerate(tickers, 1):
        path = os.path.join(PRICES, f"{t.replace('/', '_')}.csv")
        if os.path.exists(path):
            skip += 1
            continue
        try:
            df = yf.Ticker(t).history(start=START, auto_adjust=True)
            if df is None or len(df) < 60:
                fail += 1
                print(f"  [{i}/{len(tickers)}] {t}: THIN "
                      f"({0 if df is None else len(df)} bars)", flush=True)
            else:
                df = df[["Open", "High", "Low", "Close", "Volume"]]
                df.index = pd.to_datetime(df.index).tz_localize(None)
                df.to_csv(path)
                done += 1
                if i % 25 == 0:
                    print(f"  [{i}/{len(tickers)}] ok={done} thin={fail} "
                          f"skip={skip}", flush=True)
        except Exception as e:
            fail += 1
            print(f"  [{i}/{len(tickers)}] {t}: ERR {type(e).__name__}",
                  flush=True)
        time.sleep(sleep)
    print(f"PRICES done: fetched={done} thin/err={fail} skipped={skip}")


def fetch_earnings(tickers, sleep=0.6):
    """Historical REPORT dates. These are the rows the runway row needs:
    for any historical date D the NEXT print after D has already happened,
    so it is present in the table (declared assumption: using the actual
    date is mildly clairvoyant about scheduling — see the report)."""
    os.makedirs(CACHE, exist_ok=True)
    import yfinance as yf
    out = {}
    if os.path.exists(EARNINGS_PATH):
        with open(EARNINGS_PATH) as f:
            out = json.load(f) or {}
    import pandas as pd
    done = fail = skip = 0
    for i, t in enumerate(tickers, 1):
        if t in out:
            skip += 1
            continue
        try:
            df = yf.Ticker(t).get_earnings_dates(limit=100)
            if df is None or not len(df):
                out[t] = {"dates": [], "status": "empty"}
                fail += 1
            else:
                idx = pd.to_datetime(df.index)
                try:
                    idx = idx.tz_localize(None)
                except (TypeError, AttributeError):
                    idx = idx.tz_convert(None)
                out[t] = {"dates": sorted({d.date().isoformat() for d in idx}),
                          "status": "ok"}
                done += 1
        except Exception as e:
            # RECORDED, never swallowed (D-019 habit): a coverage gap must
            # be distinguishable from "this name has no earnings"
            out[t] = {"dates": [], "status": f"error:{type(e).__name__}"}
            fail += 1
        if i % 25 == 0:
            with open(EARNINGS_PATH, "w") as f:
                json.dump(out, f)
            print(f"  [{i}/{len(tickers)}] ok={done} empty/err={fail} "
                  f"skip={skip}", flush=True)
        time.sleep(sleep)
    with open(EARNINGS_PATH, "w") as f:
        json.dump(out, f, indent=0)
    print(f"EARNINGS done: ok={done} empty/err={fail} skipped={skip}")


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--prices", action="store_true")
    ap.add_argument("--earnings", action="store_true")
    ap.add_argument("--limit", type=int, default=0)
    a = ap.parse_args()
    tk = pool_tickers()
    if a.limit:
        tk = tk[:a.limit]
    print(f"pool: {len(tk)} tickers  (start {START})")
    if a.prices or not (a.prices or a.earnings):
        fetch_prices(tk)
    if a.earnings or not (a.prices or a.earnings):
        fetch_earnings(tk)


if __name__ == "__main__":
    main()
