#!/usr/bin/env python3
"""
Build 4 step 0 pin — the extracted pure regime functions must reproduce
recorded production reality EXACTLY. This pin is the backtest
reconstruction's license to exist.

Two corpora:
A. framework/state/regime_history.json entries from 2026-07-05 (the 1A
   3-voter cutover) onward: per-gauge votes where the recorded value's
   basis is unambiguous (VIX, breadth), plus the full ladder + gate from
   recorded signals -> recorded state. (Pre-1A entries are old 5-voter
   states — structurally not replayable through the current ladder;
   listed in the backtest report's limitations.)
B. Every CI-committed public/framework.json since the cutover (git
   history, ~5 artifacts/day): full component replay including the HY
   fallback percentile basis and the raw-pct gate, state must match.

Run: python3 test_regime_extraction.py   (corpus B needs the git repo)
"""

import json
import os
import subprocess
import sys

sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))

from framework.regime_calculator import (breadth_ratio_vote, compute_regime,
                                         gate_from_pct, hy_oas_vote,
                                         hy_percentile_vote, ladder_state,
                                         vix_vote)

REPO = os.path.dirname(os.path.abspath(__file__))
CUTOVER = "2026-07-05"   # Build 1A: 3-voter ladder shipped (466637e)
# Gauge B chassis cut over 2026-07-17 evening (debf12f, D-008): entries/
# artifacts from 2026-07-18 on are chassis-authored — their regime no longer
# derives from the 3-voter ladder, so the 1A replay stops here BY DESIGN.
# This is a frozen PAST-era boundary (the extraction corpus this pin
# licenses), not a future-firing assertion; chassis-era output replays
# under test_gauge_chassis's own pins.
CHASSIS_CUTOVER = "2026-07-18"


def _replay_ladder(entry):
    """Recorded per-gauge signals + recorded gate -> state via the
    extracted ladder; compare to the recorded regime."""
    gauges = entry.get("gauges") or {}
    signals = [g.get("signal") for g in gauges.values()]
    base = ladder_state(signals.count("risk_on"), signals.count("risk_off"),
                        signals.count("unavailable"))
    gate = entry.get("backdrop_gate") or {}
    state = "Caution" if (not gate.get("open", True)
                          and base.startswith("Risk-on")) else base
    return state


def test_corpus_a_regime_history():
    rh = json.load(open(os.path.join(REPO, "framework", "state",
                                     "regime_history.json")))
    entries = [e for e in rh
               if CUTOVER <= e.get("date", "") < CHASSIS_CUTOVER
               and e.get("gauges")]
    chassis_era = [e.get("date") for e in rh
                   if e.get("date", "") >= CHASSIS_CUTOVER]
    if not entries:
        # regime_history is a ROLLING file — the fixed 1A-era window ages
        # out of it (~Sept 2026). That is corpus retirement, not failure:
        # the era lives on in git history and corpus B still replays the
        # committed artifacts (review finding: the empty-corpus assert was
        # a delayed suite bomb).
        assert chassis_era, "regime_history has no entries at all — load failure"
        print(f"  corpus A: 1A-era window has aged out of the rolling "
              f"history file ({len(chassis_era)} chassis-era entries "
              f"remain) — corpus retired; corpus B still replays: OK")
        return
    for e in entries:
        g = e["gauges"]
        # component votes where the recorded basis is unambiguous
        assert vix_vote(g["vix_5d_avg"]["value"]) == g["vix_5d_avg"]["signal"], \
            f"{e['date']}: vix vote diverges"
        assert breadth_ratio_vote(g["breadth"]["value"]) == g["breadth"]["signal"], \
            f"{e['date']}: breadth vote diverges"
        # full ladder + gate from recorded signals
        assert _replay_ladder(e) == e["regime"], \
            f"{e['date']}: ladder replay {_replay_ladder(e)} != {e['regime']}"
        # recorded counts must agree with recorded signals
        sigs = [x.get("signal") for x in g.values()]
        assert sigs.count("risk_on") == e["risk_on_count"], e["date"]
        assert sigs.count("risk_off") == e["risk_off_count"], e["date"]
    era_note = (f" ({len(chassis_era)} chassis-era entries excluded — the "
                f"chassis ENGINE is pinned by test_gauge_chassis; the "
                f"chassis-era history CORPUS itself is not replay-pinned, "
                f"and pre-2026-07-23 intraday entries may carry forming-bar "
                f"prints)" if chassis_era else "")
    print(f"  corpus A: {len(entries)} regime_history entries replay "
          f"exactly{era_note}: OK")


def _committed_artifacts():
    """Every committed version of public/framework.json since the cutover."""
    shas = subprocess.run(
        ["git", "log", "--format=%H", f"--since={CUTOVER}",
         "--", "public/framework.json"],
        capture_output=True, text=True, cwd=REPO).stdout.split()
    for sha in shas:
        raw = subprocess.run(["git", "show", f"{sha}:public/framework.json"],
                             capture_output=True, text=True, cwd=REPO).stdout
        try:
            fw = json.loads(raw.replace(": NaN", ": null"))
        except json.JSONDecodeError:
            continue
        # data-driven era guard: chassis-authored artifacts declare their
        # schema (Gauge B serve sentinel) — their regime is not the
        # ladder's output and replays under test_gauge_chassis instead
        if fw.get("schema") == "regime-b-chassis":
            yield sha[:9], None
            continue
        r = fw.get("regime") or {}
        if r.get("date", "") >= CUTOVER and r.get("gauges"):
            yield sha[:9], r


def test_corpus_b_committed_artifacts():
    n = comp = skipped = 0
    for sha, r in _committed_artifacts():
        if r is None:
            skipped += 1
            continue
        g, gate = r["gauges"], r.get("backdrop_gate") or {}
        # ladder + gate replay (every artifact)
        signals = [x.get("signal") for x in g.values()]
        base = ladder_state(signals.count("risk_on"),
                            signals.count("risk_off"),
                            signals.count("unavailable"))
        state = "Caution" if (not gate.get("open", True)
                              and base.startswith("Risk-on")) else base
        assert state == r["regime"], \
            f"{sha} {r.get('date')}: ladder {state} != {r['regime']}"
        # component replay on the richer artifact blocks
        v = g.get("vix_5d_avg", {})
        if v.get("value") is not None:
            assert vix_vote(v["value"]) == v["signal"], f"{sha}: vix"
            comp += 1
        b = g.get("breadth", {})
        if b.get("value") is not None and b.get("source") == "RSP/SPY fallback":
            assert breadth_ratio_vote(b["value"]) == b["signal"], f"{sha}: breadth"
            comp += 1
        h = g.get("hy_spread", {})
        if h.get("source") == "HYG/IEF fallback" and h.get("percentile") is not None:
            assert hy_percentile_vote(h["percentile"]) == h["signal"], f"{sha}: hy"
            comp += 1
        elif h.get("source") == "FRED" and h.get("value") is not None:
            assert hy_oas_vote(h["value"]) == h["signal"], f"{sha}: hy oas"
            comp += 1
        # gate replay from the raw-or-rounded pct the artifact carries
        gv = gate.get("value")
        if gv is not None:
            g_open, _ = gate_from_pct(gv)
            # rounded 0.00 can mask a hair-below close; only assert when
            # the recorded value is not at the rounding boundary
            if abs(gv) > 0.005:
                assert g_open == gate.get("open"), f"{sha}: gate"
        n += 1
    assert n >= 10, f"only {n} committed artifacts found — corpus too thin"
    era_note = (f" ({skipped} chassis-schema artifacts excluded)"
                if skipped else "")
    print(f"  corpus B: {n} committed artifacts, {comp} component votes "
          f"replay exactly{era_note}: OK")


def test_compute_regime_edges():
    # all-risk-on, gate open
    r = compute_regime(15.0, 2.8, 1.0, 5.0)
    assert r["state"] == "Risk-on / Trending" and r["gate"]["open"]
    # gate caps risk-on, fails closed on None
    r = compute_regime(15.0, 2.8, 1.0, -0.001)
    assert r["state"] == "Caution" and r["gate"]["reason"] == "below_200dma"
    r = compute_regime(15.0, 2.8, 1.0, None)
    assert r["state"] == "Caution" and r["gate"]["reason"] == "data_unavailable"
    # gate never upgrades a degraded state
    r = compute_regime(25.0, 4.5, -1.0, 5.0)
    assert r["state"] == "Risk-off" and not r["gate"]["capped"]
    # any risk_off vote caps at Caution
    r = compute_regime(15.0, 4.5, 1.0, 5.0)
    assert r["state"] == "Caution"
    # threshold boundaries are <= / > exactly as production
    assert compute_regime(18.0, 3.0, 0.51, 1.0)["state"] == "Risk-on / Trending"
    assert compute_regime(18.01, 3.0, 0.5, 1.0)["votes"]["vix_5d_avg"] == "caution"
    assert compute_regime(18.0, 3.0, 0.5, 1.0)["votes"]["breadth"] == "caution"
    # unavailable voters: 2+ unavailable -> Caution, never Risk-off
    r = compute_regime(None, None, -5.0, 1.0)
    assert r["state"] == "Caution"
    # grid overrides thread through
    r = compute_regime(19.0, 2.8, 1.0, 5.0, vix_thresholds=(20.0, 24.0))
    assert r["votes"]["vix_5d_avg"] == "risk_on"
    # NaN inputs vote unavailable — an INTENTIONAL divergence from the old
    # inline branches, where NaN fell through every <= to risk_off and two
    # NaN voters could print Risk-off on a pure data outage (violating the
    # ladder's documented contract). Pinned so the corner stays deliberate.
    nan = float("nan")
    r = compute_regime(nan, 2.8, 1.0, 5.0)
    assert r["votes"]["vix_5d_avg"] == "unavailable"
    assert r["state"] == "Risk-on / Choppy"      # 2 risk_on, 1 unavailable
    r = compute_regime(nan, nan, 1.0, 5.0)
    assert r["state"] == "Caution"               # 2+ unavailable, never Risk-off
    print("  compute_regime edges: gate, caps, boundaries, overrides, NaN: OK")


if __name__ == "__main__":
    print("\n=== Regime extraction pin (Build 4 step 0) ===")
    test_corpus_a_regime_history()
    test_corpus_b_committed_artifacts()
    test_compute_regime_edges()
    print("\nExtraction pin green — reconstruction licensed.\n")
