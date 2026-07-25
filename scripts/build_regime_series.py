#!/usr/bin/env python3
"""
Build 5 ruling 2 — the COMMITTED regime-by-day series.

Replays the production chassis (framework.regime_calculator.replay_chassis
— the same function production runs every cycle) over the Build 4 cache's
full history and writes data/regime_daily.json, WITH provenance. The
motivation is reproducibility: the chassis compute is pure, but its OAS
input only exists locally (FRED truncated BAMLH0A0HYM2 to ~3y in Apr
2026; the pre-truncation history lives in the gitignored Wayback splice).
Without a committed series, no CI machine and no future session can rerun
anything regime-conditioned. The series is DERIVED DATA; the splice stays
local; the replay-equality pin (test_backtest_doctrine.py) proves the
committed artifact reproduces from source on splice-bearing machines.

Inputs mirror production's _chassis_inputs exactly:
  trend_in    : SPY close vs its 200DMA (>= 0)
  vix_5d      : VIX close, 5-day rolling mean
  breadth_20d : (RSP/SPY ratio vs ratio of 20d means - 1) * 100
  hy_stress   : OAS as-of-ffilled onto SPY days, shift(1) publication lag,
                60d trailing percentile >= 90 — with NaN-transparency
                (an un-warmed window yields NaN and the day is excluded,
                never a silent hy_stress=False)

Run: python3 scripts/build_regime_series.py
"""

import datetime
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

from backtest_regime import load_series
from framework.regime_calculator import (replay_chassis, CHASSIS_TO_REGIME)

OUT_PATH = os.path.join(REPO, "data", "regime_daily.json")

# production chassis config (framework/regime_calculator._chassis_cfg
# defaults — the live values per the served artifact's chassis block)
CHASSIS_CFG = {"n": 2, "mode": "asymmetric", "vix_thr": 22.0,
               "hy_cut": 90.0, "hy_window": 60, "breadth_thr": -0.5,
               "require_k": 1}
SEED_STATE = "Out-Defensive"     # replay_chassis default; converges in ~3 steps


def _sha256(path):
    h = hashlib.sha256()
    with open(path, "rb") as f:
        for chunk in iter(lambda: f.read(1 << 16), b""):
            h.update(chunk)
    return h.hexdigest()


def build(start="2019-01-02", out_path=OUT_PATH):
    raw = load_series()
    spy_c = raw["SPY"]["Close"]
    rsp_c = raw["RSP"]["Close"].reindex(spy_c.index).ffill()
    vix_c = raw["VIX"]["Close"].reindex(spy_c.index).ffill()
    oas = raw["OAS"].iloc[:, 0]

    trend = (spy_c / spy_c.rolling(200).mean() - 1.0) * 100.0
    vix_5d = vix_c.rolling(5).mean()
    ratio = rsp_c / spy_c
    ratio_20d = rsp_c.rolling(20).mean() / spy_c.rolling(20).mean()
    breadth = (ratio / ratio_20d - 1.0) * 100.0

    # production hy_stress, verbatim shape (regime_calculator ~L775):
    # as-of ffill onto SPY days, shift(1), 60d trailing pctile >= cut,
    # NaN-transparent
    oas_aligned = oas.sort_index().reindex(spy_c.index, method="ffill").shift(1)
    pct = oas_aligned.rolling(
        CHASSIS_CFG["hy_window"],
        min_periods=CHASSIS_CFG["hy_window"]).apply(
        lambda a: (a <= a[-1]).mean() * 100.0, raw=True)
    hy_stress = (pct >= CHASSIS_CFG["hy_cut"]).where(pct.notna())

    f = pd.DataFrame({"trend": trend, "vix_5d": vix_5d,
                      "breadth": breadth, "hy_stress": hy_stress})
    f = f.loc[f.index >= start].dropna()

    # trend goes in RAW (% vs 200DMA): chassis_step applies its own
    # `>= 0`. Passing a pre-computed bool is a silent inversion —
    # `False >= 0` is True in Python, so every day reads in-trend and the
    # COVID/2022 Out states vanish (caught by the distribution check).
    states, _, _ = replay_chassis(
        list(f["trend"]),
        list(f["vix_5d"]), [bool(s) for s in f["hy_stress"]],
        list(f["breadth"]),
        n=CHASSIS_CFG["n"], mode=CHASSIS_CFG["mode"],
        vix_thr=CHASSIS_CFG["vix_thr"],
        breadth_thr=CHASSIS_CFG["breadth_thr"],
        require_k=CHASSIS_CFG["require_k"], seed_state=SEED_STATE)

    days = {d.date().isoformat(): st for d, st in zip(f.index, states)}

    cache = os.path.join(REPO, "data", "backtest_cache")
    oas_file = os.path.join(cache, "OAS.csv")
    head = subprocess.run(["git", "rev-parse", "--short", "HEAD"],
                          capture_output=True, text=True, cwd=REPO
                          ).stdout.strip()
    artifact = {
        "schema": "regime-daily-1",
        "engine": "chassis",
        "states": days,
        "regime_labels": CHASSIS_TO_REGIME,
        "provenance": {
            # DERIVED DATA: reproducible only on machines carrying the
            # gitignored OAS splice — the pin proves equality there and
            # skips LOUDLY elsewhere. These fields identify exactly which
            # inputs produced this series.
            "chassis_config": CHASSIS_CFG,
            "seed_state": SEED_STATE,
            "built_from_commit": head,
            "oas_splice": {
                "file": "data/backtest_cache/OAS.csv (gitignored, local)",
                "sha256": _sha256(oas_file),
                "rows": int(len(raw["OAS"])),
                "range": [str(raw["OAS"].index.min().date()),
                          str(raw["OAS"].index.max().date())],
            },
            "spy_cache_range": [str(spy_c.index.min().date()),
                                str(spy_c.index.max().date())],
            "series_range": [min(days), max(days)],
            "trading_days": len(days),
        },
    }
    with open(out_path, "w") as fh:
        json.dump(artifact, fh, indent=1)
    from collections import Counter
    print(f"regime series: {len(days)} days "
          f"{artifact['provenance']['series_range']}")
    print("state distribution:", dict(Counter(days.values())))
    return artifact


if __name__ == "__main__":
    build()
