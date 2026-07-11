#!/usr/bin/env python3
"""
Build 4 lookahead pins (tests, not vibes). Pin 1 — the production-replay
pin — lives in test_regime_extraction.py; these are pins 2 and 3 plus
reconstruction sanity anchors.

Pin 2 (shift test): lagging ALL gauge inputs +1 day must materially change
the state series — guards accidental same-day/lookahead alignment (if
shifting changed nothing, the signals wouldn't be using time at all; if
the sim were accidentally reading T+1 data, un-lagged and lagged runs
would coincide under relabeling).

Pin 3 (walk-window): for sampled dates T, recomputing every gauge input
from data TRUNCATED at T must reproduce the full-series row exactly — no
computation anywhere reads past T.

Needs the local backtest cache (gitignored); skips with a message if
absent. Run: python3 test_backtest_regime.py   (pipeline venv)
"""

import os
import sys

import numpy as np

sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
sys.path.insert(0, os.path.join(os.path.dirname(os.path.abspath(__file__)),
                                "scripts"))

from backtest_regime import (CACHE, build_inputs, compute_states,
                             load_series, simulate, BINARY_WEIGHTS)


def _have_cache():
    return all(os.path.exists(os.path.join(CACHE, f"{n}.csv"))
               for n in ("SPY", "RSP", "VIX", "IRX", "OAS"))


def test_pin2_shift_test():
    raw = load_series()
    inputs = build_inputs(raw, "2015-01-01", "2026-07-10")
    states = compute_states(inputs)

    lagged = inputs.copy()
    for col in ("vix_5d", "breadth_20d", "spy_vs_200dma", "oas"):
        lagged[col] = inputs[col].shift(1)
    lagged = lagged.dropna(subset=["vix_5d", "breadth_20d",
                                   "spy_vs_200dma", "oas"])
    states_lagged = compute_states(lagged)

    common = states.index.intersection(states_lagged.index)
    diff_share = float((states.loc[common] != states_lagged.loc[common]).mean())
    assert diff_share > 0.02, \
        f"shift test: only {diff_share:.1%} of states changed — inputs may " \
        f"not be time-aligned to T at all"

    w = states.map(BINARY_WEIGHTS)
    wl = states_lagged.map(BINARY_WEIGHTS)
    eq, _ = simulate(w.loc[common], inputs.loc[common])
    eql, _ = simulate(wl.loc[common], inputs.loc[common])
    ratio = float(eq.iloc[-1] / eql.iloc[-1])
    assert abs(ratio - 1.0) > 0.005, \
        f"shift test: equity ratio {ratio:.4f} — lag made no material difference"
    print(f"  pin 2 (shift): {diff_share:.1%} of states flip, "
          f"equity ratio {ratio:.3f} — lag matters: OK")


def test_pin3_walk_window():
    raw = load_series()
    full = build_inputs(raw, "2015-01-01", "2026-07-10")
    rng = np.random.RandomState(4)
    sample = sorted(rng.choice(len(full), size=40, replace=False))
    checked = 0
    for i in sample:
        t = full.index[i]
        # truncate every raw series AT t, rebuild, compare the row for t
        trunc = {k: df.loc[:t] for k, df in raw.items()}
        row_t = build_inputs(trunc, "2015-01-01", str(t.date())).loc[t]
        for col in ("vix_5d", "breadth_20d", "spy_vs_200dma", "oas",
                    "spy_open", "spy_close", "cash_daily"):
            a, b = float(row_t[col]), float(full.loc[t, col])
            assert np.isclose(a, b, rtol=0, atol=1e-12), \
                f"{t.date()} {col}: truncated {a} != full {b} — " \
                f"something reads past T"
        checked += 1
    print(f"  pin 3 (walk-window): {checked} sampled dates identical under "
          f"truncation — no lookahead: OK")


def test_reconstruction_anchors():
    """Known-history anchors the reconstruction must reproduce."""
    raw = load_series()
    inputs = build_inputs(raw, "2015-01-01", "2026-07-10")
    states = compute_states(inputs)

    covid = states.loc["2020-02-28":"2020-04-30"]
    assert (covid == "Risk-off").mean() > 0.9, "COVID must read Risk-off"
    assert states.loc["2022-06-01":"2022-10-31"].isin(
        ["Caution", "Risk-off"]).all(), "2022 bear must read degraded"
    # the Jul 2026 case-study closes: production printed Trending (recorded
    # in regime_history) and the reconstruction must agree
    for d in ("2026-07-01", "2026-07-02"):
        assert states.loc[d].iloc[0] == "Risk-on / Trending" \
            if hasattr(states.loc[d], "iloc") else states.loc[d] == "Risk-on / Trending", d
    print("  anchors: COVID Risk-off, 2022 degraded, Jul-2026 Trending: OK")


if __name__ == "__main__":
    print("\n=== Backtest lookahead pins (Build 4) ===")
    if not _have_cache():
        print("  SKIP: data/backtest_cache/ not populated on this machine "
              "(gitignored raw pulls) — see docs/backtest-regime.md")
        sys.exit(0)
    test_pin2_shift_test()
    test_pin3_walk_window()
    test_reconstruction_anchors()
    print("\nAll lookahead pins green.\n")
