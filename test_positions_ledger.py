#!/usr/bin/env python3
"""Pins for the closed[] trade ledger (positions.json schema 1.2,
2026-08-18 ruling). Durable form: no fixed SHA, no gitignored inputs;
runs on a fresh clone.

Three things pinned:
  1. The COMMITTED positions.json obeys every ledger law (empty
     closed[] passes the laws trivially but the schema/keys are still
     checked).
  2. THE MOVE is atomic and lossless — driven against a temp copy via
     the real scripts/close_position.py, never the live file: one
     holding out, one closed entry in, entry facts preserved verbatim,
     realized fields recomputable.
  3. THE FAILURES FAIL (pin doctrine law 3): a duplicate left in both
     arrays, an estimate wearing basis actual_fill, and a realized_r
     that does not recompute must each be REFUSED, with the violation
     named.

Run: python3 test_positions_ledger.py
"""

import copy
import json
import os
import shutil
import subprocess
import sys
import tempfile

REPO = os.path.dirname(os.path.abspath(__file__))
POSITIONS = os.path.join(REPO, "framework", "state", "positions.json")
sys.path.insert(0, os.path.join(REPO, "scripts"))
import close_position as cp


def test_committed_file_obeys_the_laws():
    with open(POSITIONS) as f:
        doc = json.load(f)
    assert doc["schema_version"] == "1.2"
    assert isinstance(doc.get("closed"), list), "closed[] missing"
    cp.validate_ledger(doc)
    print(f"  (1) committed positions.json: schema 1.2, closed[] "
          f"present ({len(doc['closed'])} entries), all laws hold: OK")


def _run_close(tmp, extra=()):
    return subprocess.run(
        [sys.executable, os.path.join(REPO, "scripts",
                                      "close_position.py"),
         "--ticker", "HPQ", "--exit-date", "2026-08-19",
         "--exit-fill", "29.40", "--fees", "0.15",
         "--exit-reason", "system_stop",
         "--fill-source", "broker_statement",
         "--overrides", "ZERO — pin fixture",
         "--capital", "97500",
         "--regime-at-exit", "Risk-on / Trending",
         "--file", tmp, *extra],
        capture_output=True, text=True)


def test_the_move_is_atomic_and_lossless():
    with tempfile.TemporaryDirectory() as d:
        tmp = os.path.join(d, "positions.json")
        shutil.copy(POSITIONS, tmp)
        before = json.load(open(tmp))
        h = next(x for x in before["holdings"] if x["ticker"] == "HPQ")
        r = _run_close(tmp)
        assert r.returncode == 0, r.stderr
        after = json.load(open(tmp))
        assert len(after["holdings"]) == len(before["holdings"]) - 1
        assert len(after["closed"]) == len(before["closed"]) + 1
        assert not any(x["ticker"] == "HPQ" for x in after["holdings"])
        c = after["closed"][-1]
        for k in ("ticker", "entry_date", "entry_price", "shares",
                  "entry_stop"):
            assert c[k] == h[k], f"entry fact {k} not preserved"
        # realized fields recompute (194 sh, 29.40 fill, 24.68 entry)
        want = round(194 * (29.40 - 24.68) - 0.15, 2)
        assert c["realized_usd"] == want, (c["realized_usd"], want)
        cp.validate_ledger(after)
        # everything else in the file byte-equal apart from the move
        b2, a2 = copy.deepcopy(before), copy.deepcopy(after)
        b2["holdings"] = [x for x in b2["holdings"]
                          if x["ticker"] != "HPQ"]
        a2["closed"] = a2["closed"][:-1]
        for key in ("watching", "account", "schema_version"):
            assert b2[key] == a2[key], f"{key} changed by the move"
        assert b2["holdings"] == a2["holdings"], "other holdings changed"
    print("  (2) the move: one out, one in, entry facts verbatim, "
          "realized recomputes, nothing else touched: OK")


def test_the_failures_fail():
    with open(POSITIONS) as f:
        doc = json.load(f)
    base = {
        "ticker": "HPQ", "entry_date": "2026-07-20",
        "entry_price": 24.68, "shares": 194, "entry_stop": 23.64,
        "exit_date": "2026-08-19", "exit_fill": 29.40, "fees_usd": 0.15,
        "exit_reason": "system_stop", "overrides": "ZERO — fixture",
        "realized_usd": 915.53, "realized_r": 4.5378,
        "realized_pct_of_capital": 0.9390, "capital_usd_at_exit": 97500,
        "regime_at_entry": None, "regime_at_exit": None,
        "fill_source": "broker_statement", "basis": "actual_fill",
        "note": "fixture",
    }
    # (a) duplicate: HPQ is in holdings AND closed
    d1 = copy.deepcopy(doc)
    d1["closed"].append(copy.deepcopy(base))
    try:
        cp.validate_ledger(d1)
        raise SystemExit("duplicate in both arrays PASSED — pin broken")
    except AssertionError as e:
        assert "BOTH holdings and closed" in str(e)
    # (b) an estimate wearing basis actual_fill
    d2 = copy.deepcopy(doc)
    e2 = copy.deepcopy(base)
    e2["ticker"] = "ZZTEST"
    e2["fill_source"] = "estimate"
    try:
        cp.validate_ledger(d2 | {"closed": [e2]})
        raise SystemExit("estimate-as-actual PASSED — pin broken")
    except AssertionError as e:
        assert "rendering as a measurement" in str(e)
    # (c) a realized_r that does not recompute from the stored stop
    d3 = copy.deepcopy(doc)
    e3 = copy.deepcopy(base)
    e3["ticker"] = "ZZTEST"
    e3["realized_r"] = 2.0
    try:
        cp.validate_ledger(d3 | {"closed": [e3]})
        raise SystemExit("non-recomputable R PASSED — pin broken")
    except AssertionError as e:
        assert "does not recompute" in str(e)
    print("  (3) failures fail, naming the violation: duplicate move, "
          "estimate-as-measurement, non-recomputable R: OK")


if __name__ == "__main__":
    print("\n=== positions.json closed[] ledger pins (schema 1.2) ===")
    test_committed_file_obeys_the_laws()
    test_the_move_is_atomic_and_lossless()
    test_the_failures_fail()
    print("\nAll ledger pins passed.\n")
