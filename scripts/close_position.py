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

Write is validate-then-os.replace: the new object is fully validated
in memory, written to a temp file in the same directory, and swapped
in atomically. On ANY failure the original file is untouched.
"""

import argparse
import datetime
import json
import os
import sys
import tempfile

REPO = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
DEFAULT_PATH = os.path.join(REPO, "framework", "state", "positions.json")

EXIT_REASONS = ("system_stop", "thesis_failed", "discretionary")
FILL_SOURCES = ("broker_statement", "estimate", "reconstructed")
BASES = ("actual_fill", "close_estimate")
R_TOL = 0.005          # recomputation tolerances (rounding only)
PCT_TOL = 0.005


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
