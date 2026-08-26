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
     arrays, an estimate wearing basis actual_fill, a realized_r that
     does not recompute, and a system_stop_modified exit attributed to
     the system must each be REFUSED, with the violation named.

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


def _fixture_holding(path=None):
    """The pin must not name a ticker: holdings turn over, and hard-coding
    one makes the pin rot the day that position closes (HPQ closed
    2026-08-25 and took this pin with it). Drive whatever is live."""
    with open(path or POSITIONS) as f:
        doc = json.load(f)
    h = doc["holdings"][0]
    assert h["entry_price"] > h["entry_stop"], "fixture holding has no R"
    return h


def _run_close(tmp, extra=()):
    h = _fixture_holding(tmp)
    return subprocess.run(
        [sys.executable, os.path.join(REPO, "scripts",
                                      "close_position.py"),
         "--ticker", h["ticker"], "--exit-date", "2026-08-19",
         "--exit-fill", f"{h['entry_price'] * 1.05:.4f}",
         "--fees", "0.15",
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
        h = _fixture_holding(tmp)
        r = _run_close(tmp)
        assert r.returncode == 0, r.stderr
        after = json.load(open(tmp))
        assert len(after["holdings"]) == len(before["holdings"]) - 1
        assert len(after["closed"]) == len(before["closed"]) + 1
        assert not any(x["ticker"] == h["ticker"]
                       and x["entry_date"] == h["entry_date"]
                       for x in after["holdings"])
        c = after["closed"][-1]
        for k in ("ticker", "entry_date", "entry_price", "shares",
                  "entry_stop"):
            assert c[k] == h[k], f"entry fact {k} not preserved"
        fill = round(h["entry_price"] * 1.05, 4)
        want = round(h["shares"] * (fill - h["entry_price"]) - 0.15, 2)
        assert c["realized_usd"] == want, (c["realized_usd"], want)
        cp.validate_ledger(after)
        # everything else in the file byte-equal apart from the move
        b2, a2 = copy.deepcopy(before), copy.deepcopy(after)
        b2["holdings"] = [x for x in b2["holdings"]
                          if not (x["ticker"] == h["ticker"]
                                  and x["entry_date"] == h["entry_date"])]
        a2["closed"] = a2["closed"][:-1]
        for key in ("watching", "account", "schema_version"):
            assert b2[key] == a2[key], f"{key} changed by the move"
        assert b2["holdings"] == a2["holdings"], "other holdings changed"
    print("  (2) the move: one out, one in, entry facts verbatim, "
          "realized recomputes, nothing else touched: OK")


def test_the_failures_fail():
    with open(POSITIONS) as f:
        doc = json.load(f)
    _h = doc["holdings"][0]
    base = {
        "ticker": _h["ticker"], "entry_date": _h["entry_date"],
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
    # (d) THE ATTRIBUTION AXIS, both directions, across SPELLINGS.
    # DNTH 2026-08-21 is the live case: the system fired on the
    # 2026-08-20 close, the operator swapped in an intraday stop, and
    # recording that as a zero attribution would let an operator
    # decision score as doctrine. The first draft of this law tested
    # `startswith("ZERO")`; EVERY value below defeated it, which is why
    # the pin asserts the INVARIANT (the count) and not the spelling.
    d4 = copy.deepcopy(doc)
    zero_spellings = ["ZERO — the decision was the system's", "NONE — x",
                      "None", "no overrides", "0 — x", "0", 0]
    for v in zero_spellings:
        e4 = copy.deepcopy(base)
        e4["ticker"] = "ZZTEST"
        e4["exit_reason"] = "system_stop_modified"
        e4["overrides"] = v
        try:
            cp.validate_ledger(d4 | {"closed": [e4]})
            raise SystemExit(f"modified-stop attributed {v!r} PASSED "
                             "— the law is a spelling test again")
        except AssertionError as e:
            assert "cannot count 0" in str(e), (v, str(e))
    # an attribution that does not OPEN with a count is refused outright
    # rather than silently read as zero
    for v in (None, "", "   ", "banana", True):
        e4 = copy.deepcopy(base)
        e4["ticker"] = "ZZTEST"
        e4["overrides"] = v
        try:
            cp.validate_ledger(d4 | {"closed": [e4]})
            raise SystemExit(f"uncountable attribution {v!r} PASSED")
        except AssertionError as e:
            assert "countable attribution" in str(e), (v, str(e))
    # THE CONVERSE: a plain system_stop carrying a real override is
    # misfiled — that is the direction that would hide an override
    e4c = copy.deepcopy(base)
    e4c["ticker"] = "ZZTEST"
    e4c["overrides"] = "ONE — operator substituted an intraday stop"
    try:
        cp.validate_ledger(d4 | {"closed": [e4c]})
        raise SystemExit("system_stop carrying ONE override PASSED")
    except AssertionError as e:
        assert "was the system's" in str(e)
    # ...and the honest combinations are ACCEPTED (the law refuses the
    # misattribution, not the reason)
    ok_note = "scored against the doctrine's next open at 110.44"
    for reason, ov in (("system_stop_modified", "ONE — intraday stop"),
                       ("system_stop", "ZERO — the system's call"),
                       ("discretionary", "ONE — operator judgement"),
                       ("thesis_failed", "ZERO — breaker tripped")):
        e4d = copy.deepcopy(base)
        e4d["ticker"] = "ZZTEST"
        e4d["exit_reason"] = reason
        e4d["overrides"] = ov
        # only the substituted-mechanism row owes the benchmark
        e4d["note"] = ok_note if reason == "system_stop_modified" \
            else base["note"]
        cp.validate_ledger(d4 | {"closed": [e4d]})
    # (e) THE BENCHMARK LAW: a substituted mechanism whose note never
    # names the doctrine's next-open fill. This is the exact shape of
    # the DNTH first draft, which scored the override against the
    # trigger close (its own stop price) and read as free.
    e5 = copy.deepcopy(base)
    e5["ticker"] = "ZZTEST"
    e5["exit_reason"] = "system_stop_modified"
    e5["overrides"] = "ONE — intraday stop substituted"
    e5["note"] = ("within pennies of the modelled exit at the trigger "
                  "close, so the cost was epistemic, not financial")
    try:
        cp.validate_ledger(d4 | {"closed": [e5]})
        raise SystemExit("override scored against the trigger close PASSED")
    except AssertionError as e:
        assert "NEXT-OPEN FILL" in str(e), str(e)
    e5b = copy.deepcopy(e5)
    e5b["note"] = ("the doctrine's fill was the next open at 110.44; the "
                   "substitution cost $23.50 = 0.1262R")
    cp.validate_ledger(d4 | {"closed": [e5b]})
    print("  (3) failures fail, naming the violation: duplicate move, "
          "estimate-as-measurement, non-recomputable R, and the "
          f"attribution axis both ways across {len(zero_spellings)} "
          "spellings of zero, and an override scored against the trigger "
          "close instead of the next open: OK")


def test_the_real_cli_refuses_a_misattributed_override():
    """Pin doctrine law 1: DRIVE THE REAL ENTRY POINT. validate_ledger
    is reachable in-process, but the thing an operator actually runs is
    scripts/close_position.py — so the refusal is demonstrated there,
    against a temp copy, and the live file is never touched."""
    with tempfile.TemporaryDirectory() as d:
        tmp = os.path.join(d, "positions.json")
        shutil.copy(POSITIONS, tmp)
        before = open(tmp).read()
        r = _run_close(tmp, ("--exit-reason", "system_stop_modified",
                             "--overrides", "NONE — the system's call"))
        assert r.returncode != 0, \
            "the CLI ACCEPTED a modified stop attributed to the system"
        assert "cannot count 0" in (r.stderr or ""), r.stderr
        assert open(tmp).read() == before, \
            "a REFUSED close still rewrote the file"
        # an honest attribution is still refused while the note omits
        # the benchmark — the two laws are independent
        r2 = _run_close(tmp, ("--exit-reason", "system_stop_modified",
                              "--overrides", "ONE — intraday stop "
                              "substituted for the next-open fill"))
        assert r2.returncode != 0 and "NEXT-OPEN FILL" in (r2.stderr or ""), \
            r2.stderr
        assert open(tmp).read() == before, "a REFUSED close rewrote the file"
        # with BOTH — a real attribution and a note that names the
        # doctrine's benchmark — the same close succeeds
        r3 = _run_close(tmp, ("--exit-reason", "system_stop_modified",
                              "--overrides", "ONE — intraday stop "
                              "substituted for the next-open fill",
                              "--note", "the doctrine's fill was the next "
                              "open at 29.50; the substitution cost $19.40"))
        assert r3.returncode == 0, r3.stderr
    print("  (4) the REAL CLI refuses a misattributed override AND an "
          "override never scored against the open, leaves the file "
          "byte-identical both times, and accepts the honest form: OK")


if __name__ == "__main__":
    print("\n=== positions.json closed[] ledger pins (schema 1.2) ===")
    test_committed_file_obeys_the_laws()
    test_the_move_is_atomic_and_lossless()
    test_the_failures_fail()
    test_the_real_cli_refuses_a_misattributed_override()
    print("\nAll ledger pins passed.\n")
