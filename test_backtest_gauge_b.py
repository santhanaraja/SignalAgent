#!/usr/bin/env python3
"""
Gauge B campaign lookahead pins + chassis unit tests (D-008, analysis-only).

The candidate gauge adds a STATEFUL element the Build 4 parliament didn't have —
the asymmetric-hysteresis replay — so it gets its own causality pin: replaying
the chassis on data TRUNCATED at T must reproduce the full-series confirmed
state at T exactly (nothing reads past T, and the trailing HY shapes don't
either). Plus the shift test and chassis-logic unit pins.

Needs the local backtest cache (gitignored); skips if absent.
Run: python3 test_backtest_gauge_b.py
"""
import os
import sys

import numpy as np
import pandas as pd

sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
sys.path.insert(0, os.path.join(os.path.dirname(os.path.abspath(__file__)), "scripts"))

from backtest_regime import CACHE, load_series, build_inputs, simulate
from backtest_gauge_b import (
    _raw_chassis_state, replay_states, hy_shapes, warmed_hy_shapes,
    new_hysteresis_state, compute_regime_chassis, chassis_durations,
    LADDER_90_50_25_5, CHASSIS_STATES,
)


def _have_cache():
    return all(os.path.exists(os.path.join(CACHE, f"{n}.csv"))
               for n in ("SPY", "RSP", "VIX", "IRX", "OAS"))


def _synthetic(raw_states):
    """Build a tiny inputs frame + hy_stress Series that yield the requested
    raw chassis states, so the hysteresis can be tested in isolation."""
    idx = pd.date_range("2020-01-01", periods=len(raw_states), freq="D")
    trend, vix, breadth, hy = [], [], [], []
    for s in raw_states:
        if s == "In-Trend-Full":
            trend.append(1.0); vix.append(15.0); breadth.append(1.0); hy.append(False)
        elif s == "In-Trend-Throttled":
            trend.append(1.0); vix.append(25.0); breadth.append(1.0); hy.append(False)
        elif s == "Out-Defensive":
            trend.append(-1.0); vix.append(15.0); breadth.append(1.0); hy.append(False)
        elif s == "Out-Risk-off":
            trend.append(-1.0); vix.append(25.0); breadth.append(1.0); hy.append(False)
    inp = pd.DataFrame({"spy_vs_200dma": trend, "vix_5d": vix,
                        "breadth_20d": breadth}, index=idx)
    return inp, pd.Series(hy, index=idx)


def test_chassis_raw_logic():
    # in-trend: no throttle -> Full; any throttle -> Throttled (never overridden)
    assert _raw_chassis_state(True, 15.0, False, 1.0) == "In-Trend-Full"
    assert _raw_chassis_state(True, 25.0, False, 1.0) == "In-Trend-Throttled"   # vix
    assert _raw_chassis_state(True, 15.0, True, 1.0) == "In-Trend-Throttled"    # hy
    assert _raw_chassis_state(True, 15.0, False, -1.0) == "In-Trend-Throttled"  # breadth
    # out-of-trend: defensive by default, risk-off on a real stress trigger
    assert _raw_chassis_state(False, 15.0, False, 1.0) == "Out-Defensive"
    assert _raw_chassis_state(False, 25.0, False, 1.0) == "Out-Risk-off"        # vix
    assert _raw_chassis_state(False, 15.0, True, 1.0) == "Out-Risk-off"         # hy
    # breadth alone never forces risk-off out-of-trend (a throttle, not a trigger)
    assert _raw_chassis_state(False, 15.0, False, -1.0) == "Out-Defensive"
    print("  chassis raw logic: trend decides direction, throttles scale within: OK")


def test_throttle_config():
    # default require_k=1: any single throttle downgrades In-Trend (campaign rule)
    assert _raw_chassis_state(True, 25.0, False, 1.0) == "In-Trend-Throttled"
    # require_k=2: one throttle stays Full; two downgrade
    assert _raw_chassis_state(True, 25.0, False, 1.0, require_k=2) == "In-Trend-Full"
    assert _raw_chassis_state(True, 25.0, True, 1.0, require_k=2) == "In-Trend-Throttled"
    # require_k=3: only all three downgrade
    assert _raw_chassis_state(True, 25.0, True, 1.0, require_k=3) == "In-Trend-Full"
    assert _raw_chassis_state(True, 25.0, True, -1.0, require_k=3) == "In-Trend-Throttled"
    # looser vix_thr: vix 22 is not stress at threshold 25
    assert _raw_chassis_state(True, 22.0, False, 1.0, vix_thr=25.0) == "In-Trend-Full"
    print("  throttle config: require_k gating + threshold looseness (defaults=campaign): OK")


def test_hysteresis_asymmetric():
    # upgrade needs N=2 closes above confirmed; downgrade is instant (jump to raw)
    raws = ["In-Trend-Full", "In-Trend-Full", "Out-Risk-off", "Out-Risk-off",
            "In-Trend-Full", "In-Trend-Full", "In-Trend-Full"]
    inp, hy = _synthetic(raws)
    got = list(replay_states(inp, hy, n=2, mode="asymmetric"))
    assert got == ["Out-Defensive", "In-Trend-Full", "Out-Risk-off", "Out-Risk-off",
                   "Out-Risk-off", "In-Trend-Full", "In-Trend-Full"], got
    print("  hysteresis asymmetric: upgrade waits N=2, downgrade instant: OK")


def test_hysteresis_symmetric():
    # symmetric: downgrade ALSO needs N -> a 1-day dip does not confirm
    raws = ["In-Trend-Full", "In-Trend-Full", "In-Trend-Full", "Out-Risk-off",
            "In-Trend-Full", "In-Trend-Full"]
    inp, hy = _synthetic(raws)
    got = list(replay_states(inp, hy, n=2, mode="symmetric"))
    # seed Out-Defensive -> confirms Full at t2 (2 closes above), the single
    # Out-Risk-off at t3 does NOT confirm (needs 2), stays Full
    assert got == ["Out-Defensive", "In-Trend-Full", "In-Trend-Full",
                   "In-Trend-Full", "In-Trend-Full", "In-Trend-Full"], got
    # N=1 is no-hysteresis: confirmed == raw every step (both modes)
    got1 = list(replay_states(inp, hy, n=1, mode="asymmetric"))
    assert got1 == raws, got1
    print("  hysteresis symmetric: downgrade also needs N; N=1 == raw: OK")


def test_pin_shift(inputs):
    stress = hy_shapes(inputs["oas"])["pctile_60d"]
    states = replay_states(inputs, stress, n=2)
    lagged_inputs = inputs.copy()
    for c in ("vix_5d", "breadth_20d", "spy_vs_200dma", "oas"):
        lagged_inputs[c] = inputs[c].shift(1)
    lagged_inputs = lagged_inputs.dropna(subset=["vix_5d", "breadth_20d",
                                                 "spy_vs_200dma", "oas"])
    lagged_stress = hy_shapes(lagged_inputs["oas"])["pctile_60d"]
    states_lag = replay_states(lagged_inputs, lagged_stress, n=2)
    common = states.index.intersection(states_lag.index)
    diff = float((states.loc[common] != states_lag.loc[common]).mean())
    assert diff > 0.02, f"shift test: only {diff:.1%} of states changed"
    # NB: this is a SENSITIVITY/liveness check (the gauge uses time at all), not a
    # lookahead proof — a lookahead gauge would also flip. Causality is the
    # walk-window truncation pin below.
    print(f"  sensitivity (shift): {diff:.1%} of chassis states flip on a +1d lag "
          f"— gauge is time-live (causality proven by the walk-window pin): OK")


def test_pin_walk_window(raw, inputs):
    """The stateful causality pin: replay on data truncated at T reproduces the
    full-series confirmed state at T (no lookahead in the HY shapes or the
    hysteresis)."""
    full_stress = warmed_hy_shapes(raw, inputs.index)["pctile_60d"]
    full_states = replay_states(inputs, full_stress, n=2)
    rng = np.random.RandomState(8)
    # sample past the 504d percentile warmup so the trailing window is defined
    pool = [i for i in range(len(inputs)) if i > 560]
    sample = sorted(rng.choice(pool, size=25, replace=False))
    checked = 0
    for i in sample:
        t = inputs.index[i]
        trunc = {k: df.loc[:t] for k, df in raw.items()}
        tin = build_inputs(trunc, "2015-01-01", str(t.date()))
        tstress = warmed_hy_shapes(trunc, tin.index)["pctile_60d"]
        tstates = replay_states(tin, tstress, n=2)
        assert tstates.loc[t] == full_states.loc[t], \
            f"{t.date()}: truncated {tstates.loc[t]} != full {full_states.loc[t]}"
        checked += 1
    print(f"  pin (walk-window): {checked} dates — truncated replay == full-series "
          f"confirmed state, incl. stateful hysteresis: OK")


def test_ladder_and_durations(inputs):
    stress = hy_shapes(inputs["oas"])["pctile_60d"]
    states = replay_states(inputs, stress, n=2)
    assert set(states.unique()) <= set(CHASSIS_STATES)
    w = states.map(LADDER_90_50_25_5)
    assert w.between(0.05, 0.90).all(), "ladder weights out of 5-90% range"
    dur = chassis_durations(states)
    assert "In-Trend-Full" in dur and dur["In-Trend-Full"]["runs"] > 0
    print("  ladder maps to 5-90%, chassis durations populated: OK")


if __name__ == "__main__":
    print("\n=== Gauge B campaign — chassis units + lookahead pins ===")
    test_chassis_raw_logic()
    test_throttle_config()
    test_hysteresis_asymmetric()
    test_hysteresis_symmetric()
    if not _have_cache():
        print("  SKIP data pins: data/backtest_cache/ not populated (gitignored)")
        sys.exit(0)
    raw = load_series()
    inputs = build_inputs(raw, "2015-01-01", "2026-07-10")
    test_pin_shift(inputs)
    test_pin_walk_window(raw, inputs)
    test_ladder_and_durations(inputs)
    print("\nAll Gauge B pins green.\n")
