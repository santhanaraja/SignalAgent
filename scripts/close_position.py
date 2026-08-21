#!/usr/bin/env python3
"""Close a position: the ATOMIC MOVE within positions.json (schema 1.2)
— out of holdings, into closed[]. Never a copy that leaves a duplicate,
never a delete that loses the record.

    python3 scripts/close_position.py --ticker GEN \\
        --exit-date 2026-08-18 --exit-fill 27.66 --fees 0.16 \\
        --exit-reason discretionary --fill-source broker_statement \\
        --capital 97500 --regime-at-exit "Risk-on / Trending" \\
        --overrides "ONE — ..." [--regime-at-entry ...] [--note ...] \\
        [--dry-run] [--file PATH]

THE LEDGER'S TWO LAWS, enforced here and pinned in
test_positions_ledger.py:
  RECOMPUTABLE, NOT TRUSTED. realized_r derives from the stored
  entry_stop and realized_pct from the stored capital_usd_at_exit —
  both are carried precisely so a future reader can recompute them
  (the GEN and DNTH disputes both turned on a stop nobody could
  reconstruct after the fact; capital will change).
  SOURCE IS NOT OPTIONAL. fill_source ∈ {broker_statement, estimate,
  reconstructed} and basis ∈ {actual_fill, close_estimate}, with the
  consistency law: an estimate can never carry basis actual_fill, and
  actual_fill requires a real fill (broker statement or a
  reconstruction of one). A reconstruction rendering as a measurement
  is the D-019 shape.
  A SUBSTITUTED MECHANISM IS AN OVERRIDE. exit_reason distinguishes
  WHOSE decision the exit was, and the vocabulary separates two things
  that look alike in a fill blotter:
    system_stop          — the system fired and the operator executed
                           it. Deviations of TIMING or PRICE within the
                           execution day (a mid-morning limit instead of
                           market-on-open) are execution caveats for the
                           note; the DECISION was still the system's, so
                           overrides is ZERO.
    system_stop_modified — the system fired and the operator SUBSTITUTED
                           A DIFFERENT EXIT MECHANISM for the doctrine's
                           next-open fill (D-018's rejected intraday
                           stop, a conditional order, a hold-and-see).
                           Choosing the mechanism IS the override. It
                           also destroys evidence — an intraday trigger
                           makes it unknowable whether the close-basis
                           rule would have exited at all — so it must
                           never aggregate with system_stop when live
                           results are scored against the backtest.
  The attribution is enforced BOTH WAYS, off a parsed count rather than
  a spelling (OVERRIDE_LAW + override_count): system_stop must count
  ZERO, system_stop_modified and discretionary must count at least ONE,
  and an attribution that does not OPEN with a count is refused
  outright. The first draft of this law tested
  `overrides.startswith("ZERO")` and was worthless — NONE, None, "no
  overrides", 0, null, "" and a bare space all passed at the real CLI,
  each recording an operator decision as doctrine. thesis_failed is
  deliberately unconstrained: a broken thesis can be the breaker's call
  or the operator's, and forcing one would fabricate an attribution.

  THE BENCHMARK IS THE NEXT OPEN, NOT THE TRIGGER CLOSE. A note that
  scores a fill against the signal-day close is comparing to an
  ESTIMATE, not to the doctrine. The modelled fill is
  `df["Open"].iloc[fired_i + 1]` (grade_outcomes.py), "sell next open"
  (docs/backtest-systems.md), and notify_assessment.py already renders
  the close figure as `basis: close_estimate`. Both benchmarks may be
  quoted; they must not be confused, and the override cost of a
  substituted mechanism is measured against the OPEN.

Write is validate-then-os.replace: the new object is fully validated
in memory, written to a temp file in the same directory, and swapped
in atomically. On ANY failure the original file is untouched.
"""

import argparse
import datetime
import json
import os
import re
import sys
import tempfile

REPO = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
DEFAULT_PATH = os.path.join(REPO, "framework", "state", "positions.json")

EXIT_REASONS = ("system_stop", "system_stop_modified",
                "thesis_failed", "discretionary")
FILL_SOURCES = ("broker_statement", "estimate", "reconstructed")
BASES = ("actual_fill", "close_estimate")
R_TOL = 0.005          # recomputation tolerances (rounding only)
PCT_TOL = 0.005

# exit_reason -> what the attribution count must be. thesis_failed is
# deliberately UNCONSTRAINED: a broken thesis can be the breaker's call
# or the operator's, and forcing one would fabricate an attribution.
OVERRIDE_LAW = {
    "system_stop": "zero",
    "system_stop_modified": "nonzero",
    "discretionary": "nonzero",
}

_COUNT_WORDS = {"ZERO": 0, "NONE": 0, "NO": 0, "ONE": 1, "TWO": 2,
                "THREE": 3, "FOUR": 4, "FIVE": 5, "SIX": 6}


def override_count(v):
    """The leading count of an attribution string ("ONE — ..." -> 1).

    Returns None when the field does not OPEN with a count, which the
    caller treats as a violation rather than as zero: the failure mode
    this guards is an operator decision wearing an uncountable
    attribution and thereby scoring as doctrine.
    """
    if v is None or isinstance(v, bool):
        return None
    if isinstance(v, (int, float)):
        return int(v) if float(v).is_integer() and v >= 0 else None
    head = re.split(r"[\s\u2014:,.-]", str(v).strip(), maxsplit=1)[0].upper()
    if head in _COUNT_WORDS:
        return _COUNT_WORDS[head]
    return int(head) if head.isdigit() else None


def build_closed_entry(h, a):
    """The closed-trade record from the holding row + exit facts.
    Entry facts are copied VERBATIM from the row — the move preserves
    the record, it never restates it."""
    shares = h["shares"]
    entry = h["entry_price"]
    e_stop = h["entry_stop"]
    # fees_usd None = UNMEASURED, never zero (the ledger's own D-019
    # discipline): realized_usd is then GROSS of fees and the note must
    # say so. A statement-backed fee is subtracted normally.
    realized = round(shares * (a.exit_fill - entry) - (a.fees or 0.0), 2)
    r_usd = shares * (entry - e_stop)
    if r_usd <= 0:
        raise SystemExit(f"{h['ticker']}: entry_stop {e_stop} >= entry "
                         f"{entry} — R undefined; fix the row first")
    entry_note = h.get("note", "")
    return {
        "ticker": h["ticker"],
        "entry_date": h["entry_date"],
        "entry_price": entry,
        "shares": shares,
        "entry_stop": e_stop,
        "exit_date": a.exit_date,
        "exit_fill": a.exit_fill,
        "fees_usd": a.fees,          # None = unmeasured, not zero
        "exit_reason": a.exit_reason,
        "overrides": a.overrides,
        "realized_usd": realized,
        "realized_r": round(realized / r_usd, 4),
        "realized_pct_of_capital": round(realized / a.capital * 100, 4),
        "capital_usd_at_exit": a.capital,
        "regime_at_entry": a.regime_at_entry,
        "regime_at_exit": a.regime_at_exit,
        "fill_source": a.fill_source,
        "basis": a.basis,
        "note": (a.note + " || ENTRY NOTE: " + entry_note) if a.note
                else ("ENTRY NOTE: " + entry_note),
    }


def validate_ledger(doc):
    """Schema-1.2 laws over the whole file. Raises on violation."""
    assert doc.get("schema_version") == "1.2", "schema_version != 1.2"
    live = {(h["ticker"], h["entry_date"])
            for h in doc.get("holdings", [])}
    req = {"ticker", "entry_date", "entry_price", "shares", "entry_stop",
           "exit_date", "exit_fill", "fees_usd", "exit_reason",
           "overrides", "realized_usd", "realized_r",
           "realized_pct_of_capital", "capital_usd_at_exit",
           "regime_at_entry", "regime_at_exit", "fill_source", "basis",
           "note"}
    for c in doc.get("closed", []):
        key = (c["ticker"], c["entry_date"])
        missing = req - set(c)
        assert not missing, f"{key}: missing fields {sorted(missing)}"
        assert key not in live, \
            f"{key}: present in BOTH holdings and closed — the move " \
            "left a duplicate"
        assert c["exit_reason"] in EXIT_REASONS, (key, c["exit_reason"])
        # THE ATTRIBUTION IS A COUNT, AND IT MUST PARSE. Written as a
        # denylist ("must not say ZERO") this law was worthless: NONE,
        # None, "no overrides", 0, null, "" and a bare space all sailed
        # through the real CLI and recorded an operator decision as
        # doctrine. The count is therefore parsed POSITIVELY and an
        # unparseable attribution is itself the violation — an
        # attribution nobody can count is not an attribution.
        n = override_count(c["overrides"])
        assert n is not None, \
            f"{key}: overrides must OPEN with a countable attribution " \
            f"(ZERO/ONE/TWO... or a digit), got {c['overrides']!r} — an " \
            "attribution nobody can count cannot be aggregated, and " \
            "silently reads as zero"
        want = OVERRIDE_LAW.get(c["exit_reason"])
        if want == "zero":
            assert n == 0, \
                f"{key}: exit_reason {c['exit_reason']} means the decision " \
                f"was the system's, but the attribution counts {n} " \
                "override(s) — file it as system_stop_modified (mechanism " \
                "substituted) or discretionary (decision overridden)"
        elif want == "nonzero":
            assert n >= 1, \
                f"{key}: exit_reason {c['exit_reason']} means the operator " \
                "substituted a different exit mechanism for the doctrine's " \
                "next-open fill — that IS an override; the attribution " \
                f"cannot count {n}"
        if c["exit_reason"] == "system_stop_modified":
            # THE BENCHMARK LAW, enforced rather than merely documented.
            # A substituted mechanism is the ONE class whose cost is
            # measurable, and it is measurable only against the
            # doctrine's next-open fill. The first draft of the DNTH row
            # scored itself against the TRIGGER CLOSE and concluded the
            # override was free; it had in fact cost 0.1262R. Requiring
            # the note to name the benchmark does not make it correct,
            # but it makes an unmeasured override impossible to file
            # silently — the same shape as the fees_usd caveat above.
            note = c.get("note", "")
            assert "next open" in note.lower(), \
                f"{key}: a substituted exit mechanism must be scored " \
                "against the DOCTRINE'S NEXT-OPEN FILL and the note must " \
                "say so — scoring it against the trigger close measures " \
                "the override against itself and always reads as free"
        assert c["fill_source"] in FILL_SOURCES, (key, c["fill_source"])
        assert c["basis"] in BASES, (key, c["basis"])
        assert c["fees_usd"] is None or isinstance(c["fees_usd"],
                                                   (int, float)), \
            f"{key}: fees_usd must be a number or null (unmeasured)"
        if c["fees_usd"] is None:
            assert "GROSS of fees" in c.get("note", ""), \
                f"{key}: fees_usd is null (unmeasured) but the note " \
                "does not say realized_usd is GROSS of fees"
        # D-019 consistency: an estimate never renders as a measurement
        if c["fill_source"] == "estimate":
            assert c["basis"] == "close_estimate", \
                f"{key}: an ESTIMATE carrying basis {c['basis']!r} is " \
                "a reconstruction rendering as a measurement"
        if c["basis"] == "actual_fill":
            assert c["fill_source"] in ("broker_statement",
                                        "reconstructed"), \
                f"{key}: basis actual_fill requires a real fill source"
        # recomputable, not trusted
        r_usd = c["shares"] * (c["entry_price"] - c["entry_stop"])
        assert r_usd > 0, f"{key}: non-positive R"
        want_r = c["realized_usd"] / r_usd
        assert abs(want_r - c["realized_r"]) <= R_TOL, \
            f"{key}: realized_r {c['realized_r']} does not recompute " \
            f"({want_r:.4f}) from the stored entry_stop"
        want_pct = c["realized_usd"] / c["capital_usd_at_exit"] * 100
        assert abs(want_pct - c["realized_pct_of_capital"]) <= PCT_TOL, \
            f"{key}: realized_pct {c['realized_pct_of_capital']} does " \
            f"not recompute ({want_pct:.4f}) from the stored capital"


def close_position(path, a):
    with open(path) as f:
        original = f.read()
    doc = json.loads(original)
    matches = [h for h in doc.get("holdings", [])
               if h.get("ticker") == a.ticker]
    if not matches:
        raise SystemExit(f"{a.ticker}: not in holdings — nothing to close")
    if len(matches) > 1:
        raise SystemExit(f"{a.ticker}: {len(matches)} holdings rows — "
                         "disambiguate by hand; the move must be atomic "
                         "and unambiguous")
    h = matches[0]
    entry = build_closed_entry(h, a)

    n_hold, n_closed = len(doc["holdings"]), len(doc.get("closed", []))
    doc["holdings"] = [x for x in doc["holdings"] if x is not h]
    doc.setdefault("closed", []).append(entry)
    doc["updated_at"] = datetime.datetime.now(
        datetime.timezone.utc).strftime("%Y-%m-%dT%H:%M:%SZ")

    # THE MOVE, proven before anything touches disk
    assert len(doc["holdings"]) == n_hold - 1, "holdings did not shrink by 1"
    assert len(doc["closed"]) == n_closed + 1, "closed did not grow by 1"
    for k in ("ticker", "entry_date", "entry_price", "shares",
              "entry_stop"):
        assert entry[k] == h[k], f"entry fact {k} not preserved verbatim"
    validate_ledger(doc)

    if a.dry_run:
        print(json.dumps(entry, indent=2))
        print(f"\n(dry run — {path} untouched)")
        return entry

    out = json.dumps(doc, indent=2, ensure_ascii=True) + "\n"
    d = os.path.dirname(os.path.abspath(path))
    fd, tmp = tempfile.mkstemp(dir=d, suffix=".tmp")
    try:
        with os.fdopen(fd, "w") as f:
            f.write(out)
        json.loads(open(tmp).read())        # the bytes on disk parse
        os.replace(tmp, path)               # atomic swap
    except BaseException:
        if os.path.exists(tmp):
            os.unlink(tmp)
        raise
    print(f"{a.ticker}: moved holdings -> closed "
          f"(realized ${entry['realized_usd']:,.2f} = "
          f"{entry['realized_r']:+.2f}R = "
          f"{entry['realized_pct_of_capital']:+.2f}% of capital, "
          f"fill_source={entry['fill_source']}, basis={entry['basis']})")
    return entry


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--ticker", required=True)
    ap.add_argument("--exit-date", required=True)
    ap.add_argument("--exit-fill", type=float, required=True)
    ap.add_argument("--fees", required=True,
                    help='fee in USD, or "unknown" to record it as '
                         'UNMEASURED (null) with realized gross of fees')
    ap.add_argument("--exit-reason", required=True, choices=EXIT_REASONS)
    ap.add_argument("--fill-source", required=True, choices=FILL_SOURCES)
    ap.add_argument("--basis", default="actual_fill", choices=BASES)
    ap.add_argument("--overrides", required=True,
                    help='attribution: "ZERO — ..." or "ONE — ..."')
    ap.add_argument("--capital", type=float, required=True,
                    help="account capital at exit (recomputability)")
    ap.add_argument("--regime-at-entry", default=None)
    ap.add_argument("--regime-at-exit", default=None)
    ap.add_argument("--note", default=None)
    ap.add_argument("--dry-run", action="store_true")
    ap.add_argument("--file", default=DEFAULT_PATH,
                    help="positions.json path (tests drive a copy)")
    a = ap.parse_args()
    a.fees = None if str(a.fees).lower() == "unknown" else float(a.fees)
    close_position(a.file, a)


if __name__ == "__main__":
    main()
