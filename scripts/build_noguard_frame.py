#!/usr/bin/env python3
"""Build 7 — the guard-OFF grade side-car.

The committed Layer A frame's `grade` column bakes the extension guard
in (row 2 of the REAL grade_setup, at KNOBS["extension_guard_max"] =
1.8). "Guard OFF" therefore cannot be read off the frame: it must be
RE-GRADED with the guard removed.

This runs the Layer A per-ticker loop TWICE through the same
process_ticker:
  pass A — guard at 1.8. Its grade column must reproduce the committed
    frame EXACTLY, on all 848,328 rows. That is the re-grade integrity
    gate: if the reconstruction cannot reproduce the incumbent, the
    guard-off column is not trustworthy either.
  pass B — guard at +inf (row 2 can never fail; every other row and
    every feature identical — the knob is read at exactly two call
    sites, both grade_setup invocations).

Output: data/doctrine_cache/master_frame_noguard.csv.gz with
(date, ticker, grade, grade_noguard) — a side-car; the committed Layer
A frame is never rewritten.

Run: python3 scripts/build_noguard_frame.py
"""

import hashlib
import json
import os
import sys

REPO = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
sys.path.insert(0, REPO)
sys.path.insert(0, os.path.join(REPO, "scripts"))

import pandas as pd

import backtest_doctrine as bd

OUT = os.path.join(bd.CACHE, "master_frame_noguard.csv.gz")


def _pass(guard, label):
    with open(bd.REGIME_PATH) as f:
        regime = json.load(f)
    regime_by_day = regime["states"]
    end = bd.END or regime["provenance"]["series_range"][1]
    with open(bd.EARNINGS_PATH) as f:
        earnings = json.load(f)
    old = bd.KNOBS["extension_guard_max"]
    bd.KNOBS["extension_guard_max"] = guard
    rows = []
    try:
        files = sorted(os.listdir(bd.PRICES))
        for i, fn in enumerate(files, 1):
            t = fn[:-4]
            r = bd.process_ticker(t, os.path.join(bd.PRICES, fn),
                                  earnings.get(t) or {"status": "missing"},
                                  regime_by_day, bd.START, end)
            if r:
                rows.extend(r)
            if i % 100 == 0:
                print(f"  [{label}] {i}/{len(files)} rows={len(rows):,}",
                      flush=True)
    finally:
        bd.KNOBS["extension_guard_max"] = old
    df = pd.DataFrame(rows, columns=bd.COLS)
    return df[["date", "ticker", "grade"]]


def main():
    print("pass A — guard 1.8 (integrity gate vs the committed frame)")
    a = _pass(1.8, "guard-on")
    committed = pd.read_csv(bd.FRAME_PATH if hasattr(bd, "FRAME_PATH")
                            else os.path.join(bd.CACHE,
                                              "master_frame.csv.gz"),
                            usecols=["date", "ticker", "grade"])
    m = committed.merge(a, on=["date", "ticker"], how="outer",
                        suffixes=("_committed", "_rebuilt"),
                        indicator=True)
    assert (m["_merge"] == "both").all(), (
        "re-grade covers a different ticker-day set than the committed "
        f"frame: {m['_merge'].value_counts().to_dict()}")
    bad = m[m.grade_committed != m.grade_rebuilt]
    assert len(bad) == 0, (
        f"RE-GRADE INTEGRITY GATE FAILED: {len(bad)} of {len(m):,} rows "
        f"differ from the committed frame\n{bad.head(5)}")
    print(f"  integrity gate: {len(m):,} rows reproduce the committed "
          f"grade column EXACTLY")

    print("pass B — guard +inf (row 2 can never fail)")
    b = _pass(float("inf"), "guard-off")
    out = a.merge(b.rename(columns={"grade": "grade_noguard"}),
                  on=["date", "ticker"], how="inner")
    assert len(out) == len(a) == len(committed), (len(out), len(a))
    # the guard can only ever HOLD BACK a grade: removing it must never
    # lower one (a monotonicity check the arms depend on)
    worse = out[out.grade_noguard < out.grade]
    assert len(worse) == 0, (
        f"{len(worse)} rows graded LOWER with the guard removed — the "
        "recompute is not isolating row 2")
    lifted = out[out.grade_noguard > out.grade]
    out.to_csv(OUT, index=False, compression="gzip")
    h = hashlib.sha256(
        pd.util.hash_pandas_object(out, index=False).values.tobytes()
    ).hexdigest()[:16]
    print(f"lifted by guard removal: {len(lifted):,} ticker-days "
          f"({len(lifted) / len(out) * 100:.2f}%)")
    print("  lift breakdown (grade -> grade_noguard):",
          lifted.groupby(["grade", "grade_noguard"]).size().to_dict())
    print(f"A+ set: {int((out.grade == 3).sum()):,} guard-on -> "
          f"{int((out.grade_noguard == 3).sum()):,} guard-off")
    print(f"wrote {OUT}  hash {h}")


if __name__ == "__main__":
    main()
