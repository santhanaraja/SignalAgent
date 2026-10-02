#!/usr/bin/env python3
"""READ AN ENTRY STOP OFF THE FRAME — on the basis the ledger actually uses.

    python3 scripts/frame_stop.py --ticker GEN --entry-date 2026-08-12

Prints the SMA20 on the last confirmed close BEFORE the entry, the basis
bar it used, the window, every corporate action inside that window, and
the same figure on a second, GENUINELY DIFFERENT path.

WHY THIS EXISTS AND WHAT IT FIXES. The 2026-09-28 orphan migration first
read its stops off two RAW sources — yfinance auto_adjust=False and the
chart API's `close` series — and reported that "two fetch paths agree to
0.0000" as the verification. Two raw sources agree with each other by
construction; the check could not fail. And the ledger is not on raw.
framework_runner.py fetches with auto_adjust=True, and the stored stops
follow it: of the 35 rows that predate the migration, every one of the
seven with an ex-dividend inside its SMA20 window on or before the basis
bar matches the AS-OF-ADJUSTED value and misses raw — DELL 421.74 (raw
421.95), JNJ 261.89 (263.05), LLY 1196.91 (1198.10), MET 94.62 (95.09),
CFG 72.41 (72.59), FANG 196.74 (197.71), EQT 53.24 (53.28) — while rows
whose ex-date falls AFTER the basis bar match raw, which is the same
rule seen from the other side: EVERY ADJUSTMENT KNOWN AT THE TRADE DATE
IS APPLIED AND NO LATER ONE IS. IWM is what the raw read would have
cost: 293.2820 raw against 293.0438 on the ledger's basis, a 3.8% error
in the R denominator, and invisible to a two-raw-path check.

THE TWO PATHS HERE ARE NOT INDEPENDENT, AND SAYING SO IS THE POINT.
  A  yfinance auto_adjust=True         — literally what the engine calls
  B  chart API adjclose, re-based to the basis bar
yfinance fetches the SAME Yahoo v8 chart endpoint (scrapers/history.py,
on query2) and its auto_adjust Close IS indicators.adjclose[0].adjclose
(utils.py reads it into "Adj Close"; auto_adjust renames it to Close).
So both paths reduce to

    mean(adjclose[window]) * close[basis] / adjclose[basis]

and agree BIT-FOR-BIT by construction. Element-wise the two arrays
differ by exactly 0.0. An earlier version of this docstring called them
"genuinely different in kind", which was wrong, and that phrasing was
copied into seven permanent ledger notes before it was caught.

WHAT THE 0.0000 AGREEMENT ACTUALLY CERTIFIES: the re-basing arithmetic,
row alignment between the two readers, that the basis bar is PRESENT,
and that the window holds 20 bars — the holed-frame failure this project
has hit three times, which is a real failure and worth a gate.
WHAT IT CANNOT CERTIFY: that Yahoo's numbers are right. Feeding both
paths one bad in-window print moves the stop and the gate still prints
0.0000 and exits 0. Closing that would need a non-Yahoo vendor.
RAW is printed as a CONTROL and is the one genuinely independent axis
here — it differs only when a dividend or split falls on or before the
basis bar, so it is INERT for a window with no corporate action, and
the output says so rather than implying a check was performed.

Exit status is 1 if the paths disagree by more than a tenth of a cent,
if the basis bar is missing from either path, or if the window is short
of 20 bars.
"""
import argparse
import datetime as dt
import sys

import pandas as pd
import requests
import yfinance as yf

TOL = 0.001


def _chart(ticker, span="1y"):
    url = (f"https://query1.finance.yahoo.com/v8/finance/chart/{ticker}"
           f"?range={span}&interval=1d&events=div%2Csplit")
    r = requests.get(url, headers={"User-Agent": "Mozilla/5.0"}, timeout=25)
    res = r.json()["chart"]["result"][0]
    q = res["indicators"]["quote"][0]
    adj = res["indicators"].get("adjclose", [{}])[0].get("adjclose")
    idx = pd.to_datetime(res["timestamp"], unit="s", utc=True).tz_convert(
        "America/New_York").tz_localize(None).normalize()
    df = pd.DataFrame({"Open": q["open"], "Close": q["close"], "Adj": adj},
                      index=idx).dropna(subset=["Close"])
    return df, res.get("events", {}) or {}


def frame_stop(ticker, entry_date, span="1y", window=20):
    df, ev = _chart(ticker, span)
    days = [d.strftime("%Y-%m-%d") for d in df.index]
    prior = [d for d in days if d < entry_date]
    if not prior:
        raise SystemExit(f"{ticker}: no session before {entry_date}")
    basis = prior[-1]
    i = days.index(basis)
    if i + 1 < window:
        raise SystemExit(f"{ticker}: only {i + 1} bars before {basis}")
    lo = i - window + 1

    raw = float(df["Close"].iloc[lo:i + 1].mean())
    # as-of: undo every adjustment whose ex-date is AFTER the basis bar
    f = df["Adj"] / df["Close"]
    asof_b = float((df["Close"].iloc[lo:i + 1]
                    * (f.iloc[lo:i + 1] / f.iloc[i])).mean())

    yh = yf.Ticker(ticker).history(period=span, auto_adjust=True)
    yh.index = yh.index.tz_localize(None)
    ydays = [d.strftime("%Y-%m-%d") for d in yh.index]
    if basis not in ydays:
        raise SystemExit(f"{ticker}: basis bar {basis} is MISSING from the "
                         "yfinance path — a holed frame, not a value")
    j = ydays.index(basis)
    # yfinance adjusts to TODAY; divide out the cumulative factor at the
    # basis bar to get back to "as known on that date"
    asof_a = float(yh["Close"].iloc[j - window + 1:j + 1].mean()
                   / (yh["Close"].iloc[j] / df["Close"].iloc[i]))

    def _in(ts):
        return days[lo] <= dt.datetime.utcfromtimestamp(
            int(ts)).strftime("%Y-%m-%d") <= basis

    actions = ([("dividend", dt.datetime.utcfromtimestamp(int(k)).strftime(
                    "%Y-%m-%d"), v.get("amount"))
                for k, v in (ev.get("dividends") or {}).items() if _in(k)]
               + [("split", dt.datetime.utcfromtimestamp(int(k)).strftime(
                    "%Y-%m-%d"), v.get("splitRatio"))
                  for k, v in (ev.get("splits") or {}).items() if _in(k)])
    return {"ticker": ticker, "entry_date": entry_date, "basis_bar": basis,
            "window": (days[lo], days[i]), "bars": i - lo + 1,
            "close_on_basis": round(float(df["Close"].iloc[i]), 4),
            "sma20_asof_yfinance": round(asof_a, 4),
            "sma20_asof_chartapi": round(asof_b, 4),
            "sma20_raw_control": round(raw, 4),
            "corporate_actions_in_window": sorted(actions, key=lambda x: x[1]),
            # THE NEXT OPEN IS QUOTED RAW, DELIBERATELY. It is the
            # doctrine's modelled fill, i.e. a price someone could have
            # traded at on the day; an adjusted open is not one. The stop
            # is a moving average the engine computes on adjusted data,
            # so the two fields are on different bases ON PURPOSE.
            "next_open_raw": (round(float(df["Open"].iloc[i + 1]), 4)
                              if i + 1 < len(df) else None),
            "next_bar": days[i + 1] if i + 1 < len(days) else None}


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--ticker", required=True)
    ap.add_argument("--entry-date", required=True)
    ap.add_argument("--span", default="1y")
    ap.add_argument("--window", type=int, default=20)
    a = ap.parse_args()
    r = frame_stop(a.ticker, a.entry_date, a.span, a.window)
    gap = abs(r["sma20_asof_yfinance"] - r["sma20_asof_chartapi"])
    for k, v in r.items():
        print(f"  {k:30} {v}")
    print(f"  {'paths differ by':30} {gap:.4f}")
    if r["sma20_raw_control"] != r["sma20_asof_chartapi"]:
        print(f"  !! RAW DIFFERS: {r['sma20_raw_control']} — a raw read of "
              "this window would NOT reproduce the ledger's basis")
    else:
        print("  .. raw control INERT: no dividend or split on or before the "
              "basis bar, so raw and the engine basis are bit-identical here "
              "and the control proves nothing about this window")
    print("  .. the two paths share Yahoo's adjclose array; their agreement is "
          "an identity, not independent corroboration (see the docstring)")
    if gap > TOL or r["bars"] != a.window:
        sys.exit(1)


if __name__ == "__main__":
    main()
