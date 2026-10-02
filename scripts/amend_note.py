#!/usr/bin/env python3
"""APPEND A DECLARED AMENDMENT to a closed row's note. Nothing else.

closed[] is an append-only history and every tool here is built that way:
migrate_orphan_trade.py refuses to edit an existing row, add_entry.py
asserts closed[] did not move, and the pins check pre-existing rows
byte-for-byte. That is right almost always — and it has one hole. A LATER
RULING CAN FALSIFY A SENTENCE IN AN EARLIER ROW, and then the choice is
between leaving a known-false claim in the ledger or editing history.

This takes the third option: the amendment is APPENDED to the note, the
original words are left intact above it, and the edit REGISTERS ITSELF in
schema_notes.note_amendments with a reason — so the pins can allow
exactly this row's note to differ and nothing else, and a reader arrives
at the change from either end instead of finding a row that quietly says
something different from what it used to.

    python3 scripts/amend_note.py --ticker HPQ --entry-date 2026-09-03 \\
        --amendment "..." --reason "..." [--dry-run] [--file PATH]

WHAT IT REFUSES: a row it cannot find uniquely; an empty amendment or
reason; any attempt to amend a HOLDING (entries are not history yet —
edit them before they close); and it fails the whole write if any field
other than `note` on that one row would change.
"""
import argparse
import datetime
import json
import os
import sys
import tempfile

sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
from close_position import DEFAULT_PATH, validate_ledger  # noqa: E402


def amend(path, a):
    if not __debug__:
        raise SystemExit("refusing to run under -O: the laws here are asserts")
    with open(path) as f:
        doc = json.loads(f.read())
    if not (a.amendment or "").strip():
        raise SystemExit("--amendment is empty; an amendment with no text is "
                         "a silent rewrite with extra steps")
    if not (a.reason or "").strip():
        raise SystemExit("--reason is empty; the register exists so a reader "
                         "can see WHY, not just that something moved")
    if any(h.get("ticker") == a.ticker and h.get("entry_date") == a.entry_date
           for h in doc.get("holdings", [])):
        raise SystemExit(f"{a.ticker}/{a.entry_date}: that is a HOLDING, not "
                         "history — amend its note directly before it closes")
    hits = [c for c in doc.get("closed", [])
            if c["ticker"] == a.ticker and c["entry_date"] == a.entry_date]
    if len(hits) != 1:
        raise SystemExit(f"{a.ticker}/{a.entry_date}: found {len(hits)} closed "
                         "rows; the target must be unique")
    row = hits[0]
    before = {k: v for k, v in row.items()}
    stamp = datetime.date.today().isoformat()
    marker = f" || AMENDMENT {stamp}: "
    if marker in row["note"]:
        raise SystemExit(f"{a.ticker}/{a.entry_date}: already amended today; "
                         "a second amendment on one day should be one edit")
    row["note"] = row["note"] + marker + a.amendment.strip()

    # ONLY `note`, ONLY THIS ROW.
    changed = [k for k in before if k != "note" and before[k] != row[k]]
    assert not changed, f"fields other than note changed: {changed}"
    others = [c for c in doc["closed"] if c is not row]
    reg = doc.setdefault("schema_notes", {}).setdefault("note_amendments", [])
    reg.append({"row": f"{a.ticker}/{a.entry_date}", "date": stamp,
                "reason": a.reason.strip()})
    assert json.dumps(others, sort_keys=True) == json.dumps(
        [c for c in doc["closed"] if c is not row], sort_keys=True), \
        "another closed row moved"
    validate_ledger(doc)

    if a.dry_run:
        print(json.dumps({"note_tail": row["note"][-400:],
                          "register": reg[-1]}, indent=2))
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
    print(f"{a.ticker}/{a.entry_date}: note amended (+{len(a.amendment)} "
          f"chars) and registered in schema_notes.note_amendments "
          f"({len(reg)} entr{'y' if len(reg) == 1 else 'ies'})")
    return row


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--ticker", required=True)
    ap.add_argument("--entry-date", required=True)
    ap.add_argument("--amendment", required=True)
    ap.add_argument("--reason", required=True)
    ap.add_argument("--dry-run", action="store_true")
    ap.add_argument("--file", default=DEFAULT_PATH)
    a = ap.parse_args()
    amend(a.file, a)


if __name__ == "__main__":
    main()
