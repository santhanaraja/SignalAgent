#!/usr/bin/env python3
"""MIGRATE AN ORPHAN TRADE INTO closed[] — a trade whose only record is
a watch row, or none at all.

close_position.py moves a row OUT OF holdings[]. It refuses anything not
held, which is correct for a live book and useless for the five trades
this project ran before closed[] existed: they left holdings months ago
and their entry facts survive only in a note, in position_events.json,
or in a commit message. This script supplies those facts EXPLICITLY on
the command line and then hands them to the SAME two functions the live
closer uses — close_position.build_closed_entry and
close_position.validate_ledger — so the ledger's laws are enforced by
one implementation, not by a second copy that can drift from it.

WHAT IT REFUSES, AND WHY EACH REFUSAL EXISTS:
  * no fill               — a trade with no fill is not closed. The
                            house rule is older than this script: the
                            price reaching a limit is not evidence the
                            limit filled, and an order existing is not
                            evidence of a fill. IWM and BIIB are exactly
                            this case and they are NOT migrated.
  * a duplicate           — (ticker, entry_date) already in closed[].
                            BIIB has a SECOND, later trade already in
                            the ledger; migrating the July one must not
                            collide with it or overwrite it.
  * an existing holding   — the ticker is live in holdings[]. Then it is
                            not an orphan and close_position.py is the
                            right tool.
  * entry_stop >= entry   — inherited from build_closed_entry: R would
                            be zero or negative.
  * anything validate_ledger rejects, over the WHOLE file, after the
    append — the attribution law, the re-entry ordering law, the
    D-019 source/basis consistency law, recomputability.

--drop-watch removes the ticker's watching row IN THE SAME atomic write,
so the record is never in two places and never in neither. When the NAME
is staying on the watch list but its TRADE RECORD is moving — CRWD is the
case — --watch-pointer replaces that row's note with a pointer to the
closed row and drops its now-duplicated `overrides` field. Migrating the
record while leaving a full copy of it on the watch row is the failure
this option exists to prevent: two records of one trade drift, and the
watch-row copy carries no entry_stop, so nobody can recompute its R.

    python3 scripts/migrate_orphan_trade.py --ticker CRWD \\
        --entry-date 2026-07-20 --entry-price 200.10 --shares 24 \\
        --entry-stop 188.5061 --exit-date 2026-07-23 \\
        --exit-fill 189.085 --fees unknown --exit-reason system_stop \\
        --fill-source broker_statement --overrides "ZERO — ..." \\
        --capital 97500 --note "..." [--drop-watch] [--dry-run]
"""
import argparse
import datetime
import json
import os
import sys
import tempfile

sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
from close_position import (BASES, DEFAULT_PATH, EXIT_REASONS,  # noqa: E402
                            FILL_SOURCES, build_closed_entry,
                            validate_ledger)


def migrate(path, a):
    if not __debug__:
        raise SystemExit(
            "refusing to run under -O / PYTHONOPTIMIZE: every ledger law in "
            "close_position.validate_ledger and every pre-write guard here is "
            "an `assert`, and -O strips them all. Under -O this script will "
            "happily write an illegal row to disk.")
    with open(path) as f:
        doc = json.loads(f.read())

    watch = [w for w in doc.get("watching", []) if w.get("ticker") == a.ticker]
    if watch and not (a.drop_watch or a.watch_pointer or a.leave_watch_row):
        raise SystemExit(
            f"{a.ticker}: has a watching row and no disposition was given. "
            "Migrating the record while leaving a full copy of it on the "
            "watch row is the two-records-of-one-trade failure this script "
            "exists to prevent — and the watch copy carries no entry_stop, "
            "so nobody can recompute its R. Pass --drop-watch, "
            "--watch-pointer TEXT, or --leave-watch-row-alone REASON.")
    live = [h for h in doc.get("holdings", []) if h.get("ticker") == a.ticker]
    if live:
        raise SystemExit(
            f"{a.ticker}: is IN HOLDINGS ({live[0].get('shares')} sh, entered "
            f"{live[0].get('entry_date')}) — not an orphan. Use "
            "scripts/close_position.py, which moves the row atomically.")
    # THE DUPLICATE CHECK MUST NOT BE AN EXACT-STRING CHECK. "2026-7-06"
    # and "2026-07-06" are the same trade and would both file, and the
    # ledger's own ordering law only validates dates for tickers that are
    # currently HELD — so a second copy of a closed-and-gone trade sails
    # through. Both sides are parsed to real dates first.
    def _d(v, what):
        try:
            return datetime.date.fromisoformat(str(v))
        except ValueError:
            raise SystemExit(f"{what} must be an ISO YYYY-MM-DD date to be "
                             f"compared and ordered, got {v!r}")
    a.entry_date = _d(a.entry_date, "--entry-date").isoformat()
    a.exit_date = _d(a.exit_date, "--exit-date").isoformat()
    if a.exit_date < a.entry_date:
        raise SystemExit(f"{a.ticker}: exit {a.exit_date} precedes entry "
                         f"{a.entry_date}")
    dup = [c for c in doc.get("closed", [])
           if c["ticker"] == a.ticker
           and _d(c["entry_date"], f"closed {c['ticker']} entry_date")
           == _d(a.entry_date, "--entry-date")]
    if dup:
        raise SystemExit(
            f"{a.ticker}/{a.entry_date}: already in closed[] "
            f"(exit {dup[0]['exit_date']}, ${dup[0]['realized_usd']:,.2f}) — "
            "migrating it again would duplicate the trade.")

    # the entry facts, in the shape build_closed_entry reads
    h = {"ticker": a.ticker, "entry_date": a.entry_date,
         "entry_price": a.entry_price, "shares": a.shares,
         "entry_stop": a.entry_stop, "note": a.entry_note or ""}
    entry = build_closed_entry(h, a)

    n_closed = len(doc.get("closed", []))
    before_rows = json.dumps(doc.get("closed", []), sort_keys=True)
    n_watch = len(doc.get("watching", []))
    doc.setdefault("closed", []).append(entry)
    if a.drop_watch:
        # A WHOLE-ROW DELETE MUST PROVE THE CONTENT SURVIVED. Asserting the
        # row COUNT only proves something was removed. The operator retypes
        # the watch note into --entry-note by hand, and a hand copy that
        # drops a sentence loses it forever: the watch row was the only
        # record. Every string value on the row being deleted must appear
        # verbatim somewhere in the new closed row.
        row = watch[0]
        blob = json.dumps(entry, ensure_ascii=False)
        missing = [k for k, v in row.items()
                   if isinstance(v, str) and k not in ("status", "theme",
                                                       "entry_gate")
                   and v not in blob and k not in (a.waive_carry or [])]
        if missing:
            raise SystemExit(
                f"{a.ticker}: --drop-watch would delete the watching row but "
                f"these fields are not carried verbatim into the closed row: "
                f"{missing}. Carry them in --note/--entry-note, or waive each "
                "explicitly with --waive-carry FIELD (which records that you "
                "looked).")
        doc["watching"] = [w for w in doc.get("watching", [])
                           if w.get("ticker") != a.ticker]
    elif a.watch_pointer:
        hit = [w for w in doc.get("watching", [])
               if w.get("ticker") == a.ticker]
        if len(hit) != 1:
            raise SystemExit(f"{a.ticker}: --watch-pointer needs exactly one "
                             f"watching row, found {len(hit)}")
        hit[0]["note"] = a.watch_pointer
        hit[0].pop("overrides", None)
    doc["updated_at"] = datetime.datetime.now(
        datetime.timezone.utc).strftime("%Y-%m-%dT%H:%M:%SZ")

    # PROVEN BEFORE ANYTHING TOUCHES DISK
    assert len(doc["closed"]) == n_closed + 1, "closed did not grow by exactly 1"
    assert json.dumps(doc["closed"][:-1], sort_keys=True) == before_rows, \
        "an EXISTING closed row changed — the migration appends, it never edits"
    if a.drop_watch:
        assert len(doc["watching"]) == n_watch - 1, \
            f"{a.ticker}: --drop-watch removed {n_watch - len(doc['watching'])} " \
            "watch rows, expected exactly 1"
    else:
        assert len(doc["watching"]) == n_watch, "watch list changed unbidden"
    if a.watch_pointer:
        row = [w for w in doc["watching"] if w["ticker"] == a.ticker][0]
        assert "overrides" not in row and a.watch_pointer == row["note"], \
            "the watch row still carries its own copy of the trade record"
    for k in ("ticker", "entry_date", "entry_price", "shares", "entry_stop"):
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
        json.loads(open(tmp).read())
        os.replace(tmp, path)
    except BaseException:
        if os.path.exists(tmp):
            os.unlink(tmp)
        raise
    print(f"{a.ticker}: orphan -> closed[] "
          f"(realized ${entry['realized_usd']:,.2f} = "
          f"{entry['realized_r']:+.4f}R = "
          f"{entry['realized_pct_of_capital']:+.4f}% of capital"
          + (", watch row removed" if a.drop_watch else "") + ")")
    return entry


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--ticker", required=True)
    ap.add_argument("--entry-date", required=True)
    ap.add_argument("--entry-price", type=float, required=True)
    ap.add_argument("--shares", type=int, required=True)
    ap.add_argument("--entry-stop", type=float, required=True,
                    help="SMA20 on the confirmed close BEFORE the entry, "
                         "read from the frame — not back-solved from R")
    ap.add_argument("--exit-date", required=True)
    ap.add_argument("--exit-fill", type=float, required=True,
                    help="a real fill. There is no --no-fill: a trade "
                         "without one is not closed.")
    ap.add_argument("--fees", required=True,
                    help='fee in USD, or "unknown" for UNMEASURED (null), '
                         "which requires the note to say GROSS of fees")
    ap.add_argument("--exit-reason", required=True, choices=EXIT_REASONS)
    ap.add_argument("--fill-source", required=True, choices=FILL_SOURCES)
    ap.add_argument("--basis", default="actual_fill", choices=BASES)
    ap.add_argument("--overrides", required=True)
    ap.add_argument("--capital", type=float, required=True)
    ap.add_argument("--regime-at-entry", default=None)
    ap.add_argument("--regime-at-exit", default=None)
    ap.add_argument("--note", default=None)
    ap.add_argument("--entry-note", default=None,
                    help="the entry-side record, carried into the row's "
                         "note exactly as close_position.py carries the "
                         "holding's own note")
    ap.add_argument("--drop-watch", action="store_true")
    ap.add_argument("--leave-watch-row-alone", dest="leave_watch_row",
                    default=None, metavar="REASON",
                    help="keep the watching row untouched, on the record, "
                         "accepting that the trade is then described twice")
    ap.add_argument("--waive-carry", action="append", default=[],
                    metavar="FIELD",
                    help="a watching-row field --drop-watch may discard "
                         "without carrying it verbatim; repeatable")
    ap.add_argument("--watch-pointer", default=None,
                    help="keep the watching row but replace its note with "
                         "this pointer and drop its `overrides` field — for "
                         "a name that stays on the list after its trade "
                         "record moves into closed[]")
    ap.add_argument("--dry-run", action="store_true")
    ap.add_argument("--file", default=DEFAULT_PATH)
    a = ap.parse_args()
    a.fees = None if str(a.fees).lower() == "unknown" else float(a.fees)
    chosen = [f for f in ("drop_watch", "watch_pointer", "leave_watch_row")
              if getattr(a, f)]
    if len(chosen) > 1:
        raise SystemExit(f"pick ONE watch-row disposition, got {chosen}")
    migrate(a.file, a)


if __name__ == "__main__":
    main()
