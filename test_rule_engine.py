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

import os
import sys
from collections import Counter

import yaml

sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))

from framework.rule_engine import RuleEngine

CONFIG = yaml.safe_load(
    open(os.path.join(os.path.dirname(os.path.abspath(__file__)),
                      "framework", "config.yaml")))

TRENDING = "Risk-on / Trending"
CHOPPY = "Risk-on / Choppy"
CAUTION = "Caution"
RISK_OFF = "Risk-off"


def _eval(regime, active, entries=None, exits=None):
    rr = {"regime": regime}
    tr = {"active_themes": active, "entry_signals": entries or [],
          "exit_signals": exits or []}
    return RuleEngine(CONFIG).evaluate(rr, tr)


def _statuses(out):
    return {e["rule"]: e["status"] for e in out["evaluations"]}


def test_current_state_tally():
    # Choppy + 2 active themes (Semis, Biotech) — the live theme-layer state
    out = _eval(CHOPPY, ["Semis", "Biotech"])
    assert out["summary"] == {"total_rules": 27, "compliant": 14,
                              "action_needed": 13, "violations": 0}, out["summary"]
    st = _statuses(out)
    elevated = {r for r, s in st.items() if s == "elevated"}
    assert elevated == {"R6", "R7", "R8", "R10", "R11", "R12", "R13",
                        "R15", "R19", "R21", "R23", "R24", "R25"}, elevated
    print("  current state (Choppy, 2 themes): 14/13/0, correct 13 elevated: OK")


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
    # Even the synthetic input that used to trip it — an ENTRY signal in
    # Risk-off, which theme_ranker cannot produce — now yields elevated.
    out = _eval(RISK_OFF, [],
                entries=[{"theme": "X",
                          "action": "ENTRY SIGNAL — qualified for activation"}])
    assert _statuses(out)["R2"] == "elevated"
    assert out["summary"]["violations"] == 0
    # plain Risk-off elevates R2; risk-on leaves it compliant
    assert _statuses(_eval(RISK_OFF, []))["R2"] == "elevated"
    assert _statuses(_eval(TRENDING, []))["R2"] == "compliant"
    print("  R2 has no violation branch; elevated in Risk-off only: OK")


def test_r3_works_without_regime_param():
    # qualified ENTRY -> action_needed; building -> compliant; none -> compliant
    q = _eval(TRENDING, [], entries=[{"theme": "X",
              "action": "ENTRY SIGNAL — qualified for activation"}])
    r3 = next(e for e in q["evaluations"] if e["rule"] == "R3")
    assert r3["status"] == "action_needed"
    b = _eval(TRENDING, [], entries=[{"theme": "X",
              "action": "Building — 1/2 weeks in top 2"}])
    assert _statuses(b)["R3"] == "compliant"
    assert _statuses(_eval(TRENDING, []))["R3"] == "compliant"
    print("  R3 evaluates from entry signals alone (no regime param): OK")


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


def test_r5_hard_limit_violation():
    # R5 is the one numeric limit that can violate — 3 active themes.
    out = _eval(TRENDING, ["A", "B", "C"])
    assert _statuses(out)["R5"] == "violation"
    assert out["summary"]["violations"] == 1
    print("  R5 flags a violation above max_active_themes: OK")


def test_full_tally_matrix():
    # Golden per-scenario summary counts (pipeline-reachable states).
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
        active = ["Semis", "Biotech"][:n]
        s = _eval(regime, active)["summary"]
        assert (s["compliant"], s["action_needed"], s["violations"]) == (comp, act, viol), \
            (regime, n, s)
    print("  full tally matrix across regimes x active-theme counts: OK")


if __name__ == "__main__":
    print("\n=== Rule engine tests (R1-R27) ===")
    test_current_state_tally()
    test_r9_never_elevates()
    test_r2_has_no_violation_path()
    test_r3_works_without_regime_param()
    test_defensive_set_fires_in_caution_and_riskoff()
    test_r5_hard_limit_violation()
    test_full_tally_matrix()
    print("\nAll rule-engine tests passed.\n")
