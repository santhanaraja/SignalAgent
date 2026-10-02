#!/usr/bin/env python3
"""APPEND A HOLDING to positions.json (schema 1.2) — the entry side of
close_position.py, with the same laws and the same atomic write.

Entries were hand-edited into the file until 2026-10-01, when seven
landed in one session and hand-editing stopped being defensible. This
does the same job with the invariants enforced instead of remembered:

  * (ticker, entry_date) must not already be in holdings[] — the ZBRA
    fill was reported twice on 2026-09-28 and was nearly recorded twice.
  * entry_stop < entry_price, or R is zero or negative and the position
    cannot be sized in risk terms at all.
  * THE RE-ENTRY ORDERING LAW, checked here with a clear message rather
    than left to validate_ledger's assertion: every closed row for this
    ticker must have exited STRICTLY BEFORE this entry. FTNT is the live
    case — closed 2026-09-03, re-entered 2026-10-01.
  * the whole file must satisfy validate_ledger afterwards.

    python3 scripts/add_entry.py --ticker NVDA --shares 27 \\
        --entry-price 230.76 --entry-date 2026-10-01 \\
        --entry-stop 223.1252 --note "..." [--dry-run] [--file PATH]

stop_on_entry is always sma20_close: it is the only stop rule this book
runs, and spelling it per-call invites a typo that _stop_for() would
then report as "level not computed by engine".

ORDER MATTERS AND THIS TOOL CANNOT ENFORCE IT. Close before you add. On
2026-10-01 STX was only eligible because AAPL left the same session: with
AAPL still held, Technology Hardware already had three names and was at
its count cap. Run the adds first and nothing here would have noticed —
holding rows carry no group, so the group caps are R28's to compute, and
R28's enforcement class is COMPUTED (reporting-hard): it reports and
cannot block. The sequencing is the caller's responsibility; what this
tool guarantees is that each row it writes is internally sound and that
the file still satisfies every ledger law afterwards.
"""
import argparse
import datetime
import json
import os
import sys
import tempfile

sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
from close_position import DEFAULT_PATH, validate_ledger  # noqa: E402

KEY_ORDER = ("ticker", "shares", "entry_price", "entry_date",
             "stop_on_entry", "entry_stop", "note")


def add_entry(path, a):
    if not __debug__:
        raise SystemExit("refusing to run under -O: every law here and in "
                         "validate_ledger is an assert, and -O strips them")
    with open(path) as f:
        doc = json.loads(f.read())

    try:
        ed = datetime.date.fromisoformat(a.entry_date).isoformat()
    except ValueError:
        raise SystemExit(f"--entry-date must be ISO YYYY-MM-DD, got "
                         f"{a.entry_date!r}")
    dup = [h for h in doc.get("holdings", [])
           if h.get("ticker") == a.ticker and h.get("entry_date") == ed]
    if dup:
        raise SystemExit(
            f"{a.ticker}/{ed}: already in holdings ({dup[0].get('shares')} "
            "sh) — a fill reported twice is still one fill")
    # THE SAME R LAW close_position.build_closed_entry enforces, stated as
    # R rather than as a price comparison. An earlier version checked only
    # entry_stop < entry_price, which is the right test for a positive share
    # count and silently accepted 0 and negative ones: shares=0 gives
    # r_usd=0 and a position worth nothing, shares=-10 gives a short this
    # book has no rules for.
    if a.shares < 1:
        raise SystemExit(f"{a.ticker}: shares must be at least 1, got "
                         f"{a.shares} — a zero or negative lot is not an entry")
    if a.entry_price <= 0:
        raise SystemExit(f"{a.ticker}: entry_price must be positive, got "
                         f"{a.entry_price}")
    r_usd = a.shares * (a.entry_price - a.entry_stop)
    if r_usd <= 0:
        raise SystemExit(f"{a.ticker}: entry_stop {a.entry_stop} >= entry "
                         f"{a.entry_price} — R is {r_usd:+.4f}, so the "
                         "position cannot be sized in risk terms")
    late = [c for c in doc.get("closed", [])
            if c["ticker"] == a.ticker and str(c["exit_date"]) >= ed]
    if late:
        raise SystemExit(
            f"{a.ticker}: a closed row exits {late[0]['exit_date']}, which is "
            f"NOT strictly before this entry {ed} — a re-entry requires the "
            "exit to precede it; fix the dates before adding")

    row = {"ticker": a.ticker, "shares": a.shares,
           "entry_price": a.entry_price, "entry_date": ed,
           "stop_on_entry": "sma20_close", "entry_stop": a.entry_stop,
           "note": a.note or ""}
    assert list(row) == list(KEY_ORDER), "holding key order drifted"

    n = len(doc.get("holdings", []))
    before = json.dumps(doc.get("holdings", []), sort_keys=True)
    closed_before = json.dumps(doc.get("closed", []), sort_keys=True)
    doc.setdefault("holdings", []).append(row)
    doc["updated_at"] = datetime.datetime.now(
        datetime.timezone.utc).strftime("%Y-%m-%dT%H:%M:%SZ")

    assert len(doc["holdings"]) == n + 1, "holdings did not grow by exactly 1"
    assert json.dumps(doc["holdings"][:-1], sort_keys=True) == before, \
        "an EXISTING holding changed — this appends, it never edits"
    assert json.dumps(doc["closed"], sort_keys=True) == closed_before, \
        "closed[] changed — an entry must not touch the trade history"
    validate_ledger(doc)

    if a.dry_run:
        print(json.dumps(row, indent=2))
        print(f"\n(dry run — {path} untouched)")
        return row

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
    risk = r_usd
    print(f"{a.ticker}: added — {a.shares} @ {a.entry_price} stop "
          f"{a.entry_stop}, initial risk ${risk:,.2f} "
          f"({risk / 97500 * 100:.4f}% of 97500), position "
          f"${a.shares * a.entry_price:,.2f}")
    return row


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--ticker", required=True)
    ap.add_argument("--shares", type=int, required=True)
    ap.add_argument("--entry-price", type=float, required=True)
    ap.add_argument("--entry-date", required=True)
    ap.add_argument("--entry-stop", type=float, required=True,
                    help="SMA20 on the confirmed close BEFORE the entry, "
                         "read from the frame on two paths")
    ap.add_argument("--note", default=None)
    ap.add_argument("--dry-run", action="store_true")
    ap.add_argument("--file", default=DEFAULT_PATH)
    a = ap.parse_args()
    add_entry(a.file, a)


if __name__ == "__main__":
    main()
