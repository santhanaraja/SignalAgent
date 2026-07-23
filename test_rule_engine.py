#!/usr/bin/env python3
"""
Tests for the standing-rules engine (framework/rule_engine.py).

Pins the documented behavior (docs/rules.md) and guards the 2026-07-09
dead-code cleanup: removed R2 violation branch, removed unused
elevated_on_entry set, removed unused _eval_r3 regime param, renamed
elevated_in_risk_off -> elevated_in_defensive. Behavior for every
pipeline-reachable state is unchanged.

Run: python3 test_rule_engine.py
"""

import datetime
import os
import sys
from collections import Counter

import yaml

sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))

import framework.rule_engine as re_mod
from framework.rule_engine import RuleEngine

CONFIG = yaml.safe_load(
    open(os.path.join(os.path.dirname(os.path.abspath(__file__)),
                      "framework", "config.yaml")))

TRENDING = "Risk-on / Trending"
CHOPPY = "Risk-on / Choppy"
CAUTION = "Caution"
RISK_OFF = "Risk-off"


class _FixedDate(datetime.date):
    """Pinned to a Friday: R1 flips to action_needed on Sundays
    (rule_engine reads datetime.date.today() directly), so an unpinned
    tally assertion failed every Sunday — found live 2026-07-12."""
    @classmethod
    def today(cls):
        return cls(2026, 7, 10)


class _PinnedDatetime:
    date = _FixedDate
    datetime = datetime.datetime


def _eval(regime, groups, entries=None, exits=None):
    """groups = the REAL holdings' GICS groups (D-007 Phase 2 — the
    active-state trigger; themes no longer gate anything here)."""
    rr = {"regime": regime}
    tr = {"entry_signals": entries or [], "exit_signals": exits or []}
    old_dt = re_mod.datetime
    re_mod.datetime = _PinnedDatetime
    try:
        return RuleEngine(CONFIG).evaluate(rr, tr, active_groups=groups)
    finally:
        re_mod.datetime = old_dt

# the July-6 book as it RESOLVED IN ITS ERA: ARWR/BIIB -> Biotechnology,
# IWM unmapped -> own bucket (R28 convention). A historical fixture — today's
# artifacts resolve ARWR to nothing (it aged out of every layer); the pin
# tests the engine's group semantics, not today's resolver output.
JULY6_GROUPS = ["Biotechnology", "IWM (ungrouped)"]


def _statuses(out):
    return {e["rule"]: e["status"] for e in out["evaluations"]}


def test_flat_book_live_state():
    # TODAY'S LIVE STATE (pinned): flat book, zero active groups — every
    # position-triggered reminder idles with the no-active-positions detail
    out = _eval(TRENDING, [])
    assert out["summary"] == {"total_rules": 27, "compliant": 27,
                              "action_needed": 0, "violations": 0}, out["summary"]
    assert out["active_groups_used"] == []
    msgs = {e["rule"]: e["message"] for e in out["evaluations"]}
    for r in ("R6", "R7", "R21", "R25"):
        assert msgs[r].startswith("No active positions."), (r, msgs[r])
    print("  flat book (today's live state): 27/0/0, position reminders "
          "idle with 'No active positions': OK")


def test_july6_book_groups_fire():
    # the July-6 held book: 3 holdings -> 2 groups -> the 13 active-state
    # reminders fire with GROUP names (Trending)
    out = _eval(TRENDING, JULY6_GROUPS)
    assert out["summary"] == {"total_rules": 27, "compliant": 14,
                              "action_needed": 13, "violations": 0}, out["summary"]
    st = _statuses(out)
    elevated = {r for r, s in st.items() if s == "elevated"}
    assert elevated == {"R6", "R7", "R8", "R10", "R11", "R12", "R13",
                        "R15", "R19", "R21", "R23", "R24", "R25"}, elevated
    r6 = next(e for e in out["evaluations"] if e["rule"] == "R6")
    assert "Active groups: Biotechnology, IWM (ungrouped)." in r6["message"]
    assert out["active_groups_used"] == JULY6_GROUPS
    print("  July-6 book: 3 holdings -> 2 groups -> 14/13/0, rules name the "
          "groups: OK")


def test_r9_never_elevates():
    # R9's only membership was the removed elevated_on_entry set — it must
    # now be compliant in every regime/active combination.
    for regime in (TRENDING, CHOPPY, CAUTION, RISK_OFF):
        for active in ([], ["Semis", "Biotech"]):
            out = _eval(regime, active,
                        entries=[{"theme": "X",
                                  "action": "ENTRY SIGNAL — qualified for activation"}])
            assert _statuses(out)["R9"] == "compliant", (regime, active)
    print("  R9 never elevates (display-only) in any state: OK")


def test_r2_has_no_violation_path():
    # Even a synthetic Risk-off ENTRY signal (a retired-era input, now
    # ignored entirely) yields elevated, never a violation.
    out = _eval(RISK_OFF, [],
                entries=[{"theme": "X",
                          "action": "ENTRY SIGNAL — qualified for activation"}])
    assert _statuses(out)["R2"] == "elevated"
    assert out["summary"]["violations"] == 0
    # plain Risk-off elevates R2; risk-on leaves it compliant
    assert _statuses(_eval(RISK_OFF, []))["R2"] == "elevated"
    assert _statuses(_eval(TRENDING, []))["R2"] == "compliant"
    print("  R2 has no violation branch; elevated in Risk-off only: OK")


def test_r3_r4_superseded():
    # D-007 Phase 3: R3/R4 are pointer rows — injected rotation signals
    # (the retired echo inputs) are IGNORED entirely; no signal can ever
    # produce an action_needed from either rule again
    out = _eval(TRENDING, [],
                entries=[{"theme": "X",
                          "action": "ENTRY SIGNAL — qualified for activation"}],
                exits=[{"theme": "Y", "action": "EXIT SIGNAL — fired",
                        "reason": "rank collapse"}])
    st = _statuses(out)
    assert st["R3"] == "compliant" and st["R4"] == "compliant"
    r3 = next(e for e in out["evaluations"] if e["rule"] == "R3")
    r4 = next(e for e in out["evaluations"] if e["rule"] == "R4")
    assert r3["superseded_by"] == "universe rotation (D-007)"
    assert "top-15 GICS scanner" in r3["message"] and "D-011" in r3["message"]
    assert "stops" in r4["superseded_by"]
    assert "EXIT_FIRED" in r4["message"] and "breakers" in r4["message"]
    assert out["summary"]["action_needed"] == 0
    # the pointers hold with no signals passed at all (the live call shape)
    st2 = _statuses(_eval(TRENDING, []))
    assert st2["R3"] == "compliant" and st2["R4"] == "compliant"
    print("  R3/R4 superseded pointers: injected rotation signals ignored, "
          "scanner/stops named: OK")


def test_defensive_set_fires_in_caution_and_riskoff():
    # elevated_in_defensive = {R10,R13,R15,R16,R17,R18,R26}; the four that
    # are ONLY defensive (R16,R17,R18,R26) are compliant risk-on, elevated
    # in BOTH Caution and Risk-off (the rename's whole point).
    defensive_only = {"R16", "R17", "R18", "R26"}
    for r in defensive_only:
        assert _statuses(_eval(TRENDING, []))[r] == "compliant", r
        assert _statuses(_eval(CAUTION, []))[r] == "elevated", r
        assert _statuses(_eval(RISK_OFF, []))[r] == "elevated", r
    print("  defensive set (R16/17/18/26) elevates in Caution AND Risk-off: OK")


def test_r5_superseded():
    # R5 retired (D-007 Phase 2): theme-count semantics superseded by R28's
    # per-group caps — NO violation branch remains at any group count
    for groups in ([], ["A"], ["A", "B", "C", "D", "E"]):
        out = _eval(TRENDING, groups)
        r5 = next(e for e in out["evaluations"] if e["rule"] == "R5")
        assert r5["status"] == "compliant", groups
        assert r5["superseded_by"] == "R28"
        assert "R28 per-group caps" in r5["message"]
        assert "≤20% / ≤3" in r5["message"]
        assert out["summary"]["violations"] == 0
    print("  R5 superseded -> R28 per-group caps pointer; no violation "
          "branch at any group count: OK")


def test_positions_unavailable_not_flat():
    # an outage must never render as a confident flat book (review finding):
    # active_groups=None (unknown) reads "Positions unavailable", never
    # "No active positions."; [] stays the proven-flat message
    u = _eval(TRENDING, None)
    f = _eval(TRENDING, [])
    r6u = next(e for e in u["evaluations"] if e["rule"] == "R6")["message"]
    r6f = next(e for e in f["evaluations"] if e["rule"] == "R6")["message"]
    assert r6u.startswith("Positions unavailable — active state unknown."), r6u
    assert r6f.startswith("No active positions."), r6f
    assert u["active_state_unavailable"] is True
    assert f["active_state_unavailable"] is False
    assert u["summary"]["violations"] == 0        # honest, not alarmist
    # dedupe + coercion: malformed caller input never crashes or duplicates
    d = _eval(TRENDING, ["Z", "A", "A"])
    assert d["active_groups_used"] == ["A", "Z"]
    print("  outage != flat: unknown-state message + flag, dedupe/coercion: OK")


def test_full_tally_matrix():
    # Golden per-scenario summary counts (regime x active-GROUP counts).
    # Identical numbers to the theme era — the elevation logic is
    # isomorphic; only the trigger source changed (D-007 Phase 2).
    expected = {
        (TRENDING, 0): (27, 0, 0),
        (TRENDING, 2): (14, 13, 0),
        (CHOPPY, 2): (14, 13, 0),
        (CAUTION, 0): (20, 7, 0),
        (CAUTION, 2): (10, 17, 0),
        (RISK_OFF, 0): (19, 8, 0),
        (RISK_OFF, 2): (9, 18, 0),
    }
    for (regime, n), (comp, act, viol) in expected.items():
        groups = JULY6_GROUPS[:n]
        s = _eval(regime, groups)["summary"]
        assert (s["compliant"], s["action_needed"], s["violations"]) == (comp, act, viol), \
            (regime, n, s)
    print("  full tally matrix across regimes x active-GROUP counts: OK")


if __name__ == "__main__":
    print("\n=== Rule engine tests (R1-R27, group era) ===")
    test_flat_book_live_state()
    test_july6_book_groups_fire()
    test_r9_never_elevates()
    test_r2_has_no_violation_path()
    test_r3_r4_superseded()
    test_defensive_set_fires_in_caution_and_riskoff()
    test_r5_superseded()
    test_positions_unavailable_not_flat()
    test_full_tally_matrix()
    print("\nAll rule-engine tests passed.\n")
