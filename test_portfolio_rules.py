#!/usr/bin/env python3
"""
Tests for R28 — real-dollar portfolio enforcement (PER-508 Phase 0).

Pins per the build spec: the live zero-holdings/Choppy state, the
July-6 book vs the Trending ceiling, R15 violation at 9%, the
per-group count and exposure violations, the 92%-Trending ceiling
violation vs the 60%-Caution downshift action_needed (with the
derisk-via-discipline message), the unconfigured degrade, warning
tiers, ungrouped bucketing, unknown-regime fallback, and missing-price
degradation.

Run: python3 test_portfolio_rules.py
"""

import os
import sys

sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))

from framework.portfolio_rules import (REGIME_CEILINGS, assess_portfolio,
                                       build_group_map)

CAP = 100_000.0


def _positions(*holdings):
    return {"holdings": [dict(h) for h in holdings]}


def test_live_state_zero_holdings_choppy():
    r = assess_portfolio(_positions(), {}, CAP, "Risk-on / Choppy")
    assert r["status"] == "ok"
    assert r["ceiling_pct"] == 50.0 and r["deployed_pct"] == 0.0
    assert r["cash_pct"] == 100.0
    assert r["ceiling"]["status"] == "compliant"
    assert r["summary"]["violations"] == 0
    assert r["positions"] == [] and r["groups"] == []
    print("  live pin: zero holdings, Choppy -> compliant, ceiling 50, deployed 0: OK")


def test_july6_book_vs_trending():
    """The real July-6 entries at entry prices: ~18.7% deployed."""
    pos = _positions(
        {"ticker": "IWM", "shares": 23},
        {"ticker": "ARWR", "shares": 71, "group": "Biotechnology"},
        {"ticker": "BIIB", "shares": 27, "group": "Biotechnology"},
    )
    prices = {"IWM": 299.58, "ARWR": 85.36, "BIIB": 213.47}
    r = assess_portfolio(pos, prices, CAP, "Risk-on / Trending")
    assert r["ceiling_pct"] == 90.0
    assert 18.0 < r["deployed_pct"] < 19.5, r["deployed_pct"]
    assert r["ceiling"]["status"] == "compliant"
    assert all(p["status"] == "compliant" for p in r["positions"])
    assert all(g["status"] == "compliant" for g in r["groups"])
    # IWM is ungrouped -> its own bucket; the two biotechs share one
    names = {g["group"] for g in r["groups"]}
    assert "IWM (ungrouped)" in names and "Biotechnology" in names
    assert len(names) == 2
    bio = next(g for g in r["groups"] if g["group"] == "Biotechnology")
    assert bio["count"] == 2                     # under the 3-per-group cap
    print("  July-6 book: ~18.7% vs Trending 90 -> all compliant, buckets right: OK")


def test_r15_violation_and_warning():
    r = assess_portfolio(_positions({"ticker": "AAA", "shares": 90}),
                         {"AAA": 100.0}, CAP, "Risk-on / Trending")
    p = r["positions"][0]
    assert p["pct_of_capital"] == 9.0 and p["status"] == "violation"
    assert "$9,000" in p["message"] and "8% per-position limit" in p["message"]

    r = assess_portfolio(_positions({"ticker": "AAA", "shares": 75}),
                         {"AAA": 100.0}, CAP, "Risk-on / Trending")
    assert r["positions"][0]["status"] == "warning"      # 7.5% > 7.2
    r = assess_portfolio(_positions({"ticker": "AAA", "shares": 30}),
                         {"AAA": 100.0}, CAP, "Risk-on / Trending")
    p = r["positions"][0]
    assert p["status"] == "compliant" and "starter size" in p["message"]
    print("  R15: 9% violation, 7.5% warning, 3% starter-compliant: OK")


def test_group_violations():
    """D-007 Phase 0 caps as AMENDED 2026-07-13: <=20% / <=3 positions
    (from 15%/2 — two full-size 8% names now fit with room)."""
    prices = {t: 100.0 for t in "ABCD"}
    # 4 positions, one group -> count violation (exposure 16% now fine)
    pos = _positions(
        {"ticker": "A", "shares": 40, "group": "Semis"},
        {"ticker": "B", "shares": 40, "group": "Semis"},
        {"ticker": "C", "shares": 40, "group": "Semis"},
        {"ticker": "D", "shares": 40, "group": "Semis"},
    )
    r = assess_portfolio(pos, prices, CAP, "Risk-on / Trending")
    semis = r["groups"][0]
    assert semis["status"] == "violation"
    assert "4 positions exceed the 3-per-group cap" in semis["message"]
    assert "group cap ($" not in semis["message"]        # 16% under 20

    # 3 positions, one group: at the count cap -> not a violation
    pos3 = _positions(
        {"ticker": "A", "shares": 40, "group": "Semis"},
        {"ticker": "B", "shares": 40, "group": "Semis"},
        {"ticker": "C", "shares": 40, "group": "Semis"},
    )
    r = assess_portfolio(pos3, prices, CAP, "Risk-on / Trending")
    assert r["groups"][0]["status"] != "violation"       # 12%, 3 names

    # 21% in one group -> exposure violation at the amended cap
    pos = _positions(
        {"ticker": "A", "shares": 105, "group": "Semis"},
        {"ticker": "B", "shares": 105, "group": "Semis"},
    )
    r = assess_portfolio(pos, prices, CAP, "Risk-on / Trending")
    semis = r["groups"][0]
    assert semis["status"] == "violation"
    assert "21.0% of capital exceeds the 20% group cap" in semis["message"]
    assert "$20,000" in semis["message"]

    # boundary: exactly 20.0% is NOT a violation — warning band (>18%)
    pos = _positions(
        {"ticker": "A", "shares": 100, "group": "Semis"},
        {"ticker": "B", "shares": 100, "group": "Semis"},
    )
    r = assess_portfolio(pos, prices, CAP, "Risk-on / Trending")
    semis = r["groups"][0]
    assert semis["status"] == "warning", semis
    assert "within 0.0pp of the 20% group cap" in semis["message"]
    # 16% sits under the warning band (18%) -> compliant
    pos = _positions(
        {"ticker": "A", "shares": 80, "group": "Semis"},
        {"ticker": "B", "shares": 80, "group": "Semis"},
    )
    r = assess_portfolio(pos, prices, CAP, "Risk-on / Trending")
    assert r["groups"][0]["status"] == "compliant"
    print("  groups (amended 20/3): 4-count + 21% violations, 3-count ok, "
          "20.0-exact warning, 16% compliant: OK")


def test_ceiling_violation_vs_downshift_action_needed():
    # 92% deployed at Trending: bought past the ceiling -> violation
    holdings = [{"ticker": f"T{i}", "shares": 10, "group": f"G{i}"}
                for i in range(12)]                       # 12 x 7.67% = 92%
    prices = {f"T{i}": 766.67 for i in range(12)}
    r = assess_portfolio({"holdings": holdings}, prices, CAP,
                         "Risk-on / Trending")
    assert 91.5 < r["deployed_pct"] < 92.5
    assert r["ceiling"]["status"] == "violation"
    assert "bought past the ceiling" in r["ceiling"]["message"]

    # 60% deployed and the regime drops to Caution (25): downshift ->
    # action_needed with the derisk-via-discipline message
    holdings = [{"ticker": f"T{i}", "shares": 10, "group": f"G{i}"}
                for i in range(8)]                        # 8 x 7.5% = 60%
    prices = {f"T{i}": 750.0 for i in range(8)}
    r = assess_portfolio({"holdings": holdings}, prices, CAP, "Caution")
    assert r["deployed_pct"] == 60.0 and r["ceiling_pct"] == 25.0
    assert r["ceiling"]["status"] == "action_needed"
    m = r["ceiling"]["message"]
    assert "reduce exposure below 25% via normal exit discipline" in m
    assert "no same-day forced liquidation" in m
    assert r["summary"]["action_needed"] == 1
    print("  ceiling: 92% Trending violation; 60% Caution downshift action_needed: OK")


def test_ceiling_warning_and_cash_floor_row():
    holdings = [{"ticker": f"T{i}", "shares": 10, "group": f"G{i}"}
                for i in range(6)]                        # 6 x 7.67 = 46%
    prices = {f"T{i}": 766.67 for i in range(6)}
    r = assess_portfolio({"holdings": holdings}, prices, CAP,
                         "Risk-on / Choppy")
    assert r["ceiling"]["status"] == "warning"            # 46 > 45 (90% of 50)
    cf = r["cash_floor"]
    assert cf["status"] == "info"
    assert "the ceiling's complement, not a separate limit" in cf["message"]
    assert "50" in cf["message"] or "50%" in r["ceiling"]["message"]
    print("  ceiling warning at 92% of Choppy cap; cash floor informational: OK")


def test_unconfigured_and_unknown_regime_and_no_price():
    r = assess_portfolio(_positions({"ticker": "A", "shares": 1}),
                         {"A": 100.0}, None, "Risk-on / Choppy")
    assert r["status"] == "unconfigured" and "account_capital_usd" in r["message"]
    r = assess_portfolio(_positions(), {}, 0, "Caution")
    assert r["status"] == "unconfigured"
    r = assess_portfolio(_positions(), {}, True, "Caution")
    assert r["status"] == "unconfigured"                  # bool is not capital

    # unknown regime -> Caution's ceiling (the gauge's outage convention)
    r = assess_portfolio(_positions(), {}, CAP, "Unknown")
    assert r["ceiling_state"] == "Caution" and r["ceiling_pct"] == 25.0

    # missing price: row degrades, never counted, never crashes
    r = assess_portfolio(_positions({"ticker": "GONE", "shares": 10}),
                         {}, CAP, "Risk-on / Choppy")
    p = r["positions"][0]
    assert p["status"] == "no_price" and r["deployed_pct"] == 0.0
    assert r["summary"]["no_price"] == 1
    print("  degrades: unconfigured (None/0/bool), unknown regime -> Caution, no-price row: OK")


def test_review_hardening():
    """Pins for the adversarial-review fixes."""
    # negative shares: invalid row, excluded from EVERY aggregate — never
    # nets down deployed into a green board
    pos = _positions({"ticker": "LONG", "shares": 300, "group": "G1"},
                     {"ticker": "SHRT", "shares": -200, "group": "G2"})
    prices = {"LONG": 100.0, "SHRT": 100.0}
    r = assess_portfolio(pos, prices, CAP, "Caution")
    assert r["deployed_pct"] == 30.0                     # long side only
    assert r["ceiling"]["status"] == "action_needed"     # 30 > 25 Caution
    bad = next(p for p in r["positions"] if p["ticker"] == "SHRT")
    assert bad["status"] == "invalid" and "long-only book" in bad["message"]
    assert r["summary"]["invalid"] == 1

    # string shares: invalid row, no TypeError
    r = assess_portfolio(_positions({"ticker": "A", "shares": "10"}),
                         {"A": 100.0}, CAP, "Caution")
    assert r["positions"][0]["status"] == "invalid"

    # NaN capital: unconfigured, never an all-compliant board
    r = assess_portfolio(_positions({"ticker": "A", "shares": 10}),
                         {"A": 100.0}, float("nan"), "Caution")
    assert r["status"] == "unconfigured"

    # two lots value at the RAW price, once (39.998 x 2000 = 79,996.0)
    r = assess_portfolio(_positions({"ticker": "A", "shares": 1000},
                                    {"ticker": "A", "shares": 1000}),
                         {"A": 39.998}, CAP, "Risk-on / Trending")
    assert r["deployed_usd"] == 79996.0, r["deployed_usd"]
    assert len(r["positions"]) == 1 and r["positions"][0]["shares"] == 2000

    # unpriced rows still COUNT toward the group position cap (4th member
    # unpriced must still trip the amended 3-per-group cap)
    pos = _positions({"ticker": "A", "shares": 40, "group": "Semis"},
                     {"ticker": "B", "shares": 40, "group": "Semis"},
                     {"ticker": "C", "shares": 40, "group": "Semis"},
                     {"ticker": "D", "shares": 40, "group": "Semis"})
    r = assess_portfolio(pos, {"A": 100.0, "B": 100.0, "C": 100.0}, CAP,
                         "Risk-on / Trending")
    semis = next(g for g in r["groups"] if g["group"] == "Semis")
    assert semis["count"] == 4 and semis["status"] == "violation"
    assert "unpriced: D" in semis["message"]
    assert "unpriced row excluded — exposure understated" \
        in r["ceiling"]["message"]

    # ALL prices missing: degraded, ceiling unavailable — an outage never
    # renders as a compliant flat book
    r = assess_portfolio(pos, {}, CAP, "Risk-on / Choppy")
    assert r["degraded"] is True
    assert r["ceiling"]["status"] == "unavailable"
    assert "exposure unknown" in r["ceiling"]["message"]
    # zero holdings stays NON-degraded (a genuinely flat book is fine)
    r = assess_portfolio(_positions(), {}, CAP, "Risk-on / Choppy")
    assert r["degraded"] is False and r["ceiling"]["status"] == "compliant"
    print("  review hardening: negative/string shares, NaN capital, raw-price"
          " lots, unpriced group count, degraded outage: OK")


def test_build_group_map():
    u = {"groups": {"Semis": {"tickers": ["NVDA", "AMD"]},
                    "Biotechnology": {"tickers": ["ARWR"]}}}
    m = build_group_map(u)
    assert m == {"NVDA": "Semis", "AMD": "Semis", "ARWR": "Biotechnology"}
    assert build_group_map({}) == {}
    print("  build_group_map: universe GICS mapping: OK")


if __name__ == "__main__":
    print("\n=== R28 portfolio enforcement tests (PER-508 Phase 0) ===")
    test_live_state_zero_holdings_choppy()
    test_july6_book_vs_trending()
    test_r15_violation_and_warning()
    test_group_violations()
    test_ceiling_violation_vs_downshift_action_needed()
    test_ceiling_warning_and_cash_floor_row()
    test_unconfigured_and_unknown_regime_and_no_price()
    test_review_hardening()
    test_build_group_map()
    print("\nAll R28 tests passed.\n")
