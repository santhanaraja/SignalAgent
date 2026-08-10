#!/usr/bin/env python3
"""Deliberate updater for the committed Nasdaq-100 membership list.

THE POINT OF THIS FILE IS THAT IT IS NOT THE RUNTIME PATH.

universe_source reads data/pool/nasdaq100.json and never fetches it. Index
membership changes a handful of times a year, on announced dates; a live
scrape would hand a third party the ability to change what the system can
trade, silently, between two rotations. So the list is a committed artifact
with a dated provenance header, and this script is how a human moves it:

    python scripts/refresh_nasdaq100.py          # show the diff, write nothing
    python scripts/refresh_nasdaq100.py --write  # write it, then review + commit

Run at reconstitution (annual, December) or on an announced ad-hoc change.
The diff is the deliverable: a membership change you cannot explain is a
reason to stop, not to commit.
"""

import argparse
import datetime
import json
import os
import re
import sys

BASE_DIR = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
LIST_PATH = os.path.join(BASE_DIR, "data", "pool", "nasdaq100.json")

SOURCE_URL = "https://api.nasdaq.com/api/quote/list-type/nasdaq100"
SOURCE_KIND = "Nasdaq, Inc. — the index publisher"
_HEADERS = {"User-Agent": "Mozilla/5.0 SignalAgent/1.0 (nasdaq100-refresh)"}

# Validation floors. A source that changes shape must fail loudly rather than
# quietly hand back a short list that silently shrinks the pool.
MIN_COUNT, MAX_COUNT = 95, 115
# A real reconstitution moves a handful of names. Anything larger is far more
# likely to be a wrong list that happens to validate — a most-actives decoy, a
# different index, a truncated response — than a genuine index overhaul. The
# count and symbol checks cannot tell those apart; continuity with what is
# already committed can.
MAX_CHURN = 15
SYMBOL_RE = re.compile(r"^[A-Z][A-Z.\-]{0,5}$")
# Present in every plausible Nasdaq-100: if the parse loses these, it is the
# parse that broke, not the index.
SANITY = {"AAPL", "MSFT", "NVDA", "AMZN", "META", "GOOGL", "COST", "NFLX"}


def _find_rows(obj):
    """Locate the membership row list anywhere in the response envelope.

    The endpoint nests payloads (data.data.rows) and has moved that nesting
    before, so this searches for the SHAPE rather than a fixed path.

    It returns the LONGEST shape-match, not the first. The envelope carries
    other symbol lists — `mostActive` and friends — and dict insertion order
    can put a decoy ahead of the real membership. A first-match search will
    happily return a 100-row most-actives list that passes every validation
    below and writes a plausible, wrong index.
    """
    best = None
    stack = [obj]
    while stack:
        cur = stack.pop()
        if isinstance(cur, list):
            if cur and isinstance(cur[0], dict) and "symbol" in cur[0]:
                if best is None or len(cur) > len(best):
                    best = cur
            else:
                stack.extend(cur)
        elif isinstance(cur, dict):
            stack.extend(cur.values())
    return best


def fetch_symbols():
    import requests
    r = requests.get(SOURCE_URL, headers=_HEADERS, timeout=30)
    r.raise_for_status()
    rows = _find_rows(r.json())
    if not rows:
        raise RuntimeError(
            f"{SOURCE_URL} returned no recognisable row list — the response "
            "shape changed; inspect it by hand before trusting this script")
    symbols = sorted({str(row["symbol"]).strip().upper() for row in rows
                      if row.get("symbol")})
    problems = []
    if not MIN_COUNT <= len(symbols) <= MAX_COUNT:
        problems.append(f"{len(symbols)} symbols, expected {MIN_COUNT}-{MAX_COUNT}")
    bad = [s for s in symbols if not SYMBOL_RE.match(s)]
    if bad:
        problems.append(f"malformed symbols: {bad}")
    missing = SANITY - set(symbols)
    if missing:
        problems.append(f"sanity names absent: {sorted(missing)}")
    if problems:
        raise RuntimeError("Nasdaq-100 fetch failed validation:\n  "
                           + "\n  ".join(problems))
    return symbols


def load_committed():
    if not os.path.exists(LIST_PATH):
        return None
    with open(LIST_PATH) as f:
        return json.load(f)


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--write", action="store_true",
                    help="write the file (default: diff only)")
    ap.add_argument("--allow-churn", action="store_true",
                    help=f"permit a membership change larger than "
                         f"{MAX_CHURN} names (a real reconstitution can be)")
    args = ap.parse_args()

    symbols = fetch_symbols()
    now = datetime.datetime.now(datetime.timezone.utc)

    old = load_committed()
    old_symbols = set((old or {}).get("symbols") or [])
    added = sorted(set(symbols) - old_symbols)
    removed = sorted(old_symbols - set(symbols))

    print(f"source   : {SOURCE_URL}")
    print(f"as_of    : {now.date().isoformat()}")
    print(f"symbols  : {len(symbols)}"
          + (f"  (committed: {len(old_symbols)})" if old else "  (no committed list yet)"))
    if old:
        print(f"added    : {', '.join(added) if added else '(none)'}")
        print(f"removed  : {', '.join(removed) if removed else '(none)'}")
        if not added and not removed:
            print("\nno membership change.")

    churn = len(added) + len(removed)
    if old and churn > MAX_CHURN and not args.allow_churn:
        print(f"\nREFUSING TO WRITE: {churn} membership changes "
              f"({len(added)} added, {len(removed)} removed) exceeds "
              f"MAX_CHURN={MAX_CHURN}.")
        print("A change this large is more often a wrong list that passed "
              "validation — a most-actives decoy, a different index, a "
              "truncated response — than a real reconstitution.")
        print("Check the diff above against the published index change. If it "
              "is genuine, re-run with --allow-churn.")
        return 1

    if not args.write:
        print("\n(diff only — re-run with --write to update the committed list)")
        return 0

    payload = {
        "_provenance": {
            "index": "Nasdaq-100",
            "as_of": now.date().isoformat(),
            "retrieved_at": now.isoformat(timespec="seconds"),
            "source_url": SOURCE_URL,
            "source_kind": SOURCE_KIND,
            "retrieved_by": "scripts/refresh_nasdaq100.py --write",
            "runtime_fetches_this": False,
            "note": (
                "READ FROM THIS FILE AT RUNTIME. universe_source never fetches "
                "index membership. This list is updated deliberately, by a human "
                "running the refresh script and reviewing the diff, at "
                "reconstitution (annual, December) or on an announced ad-hoc "
                "change."),
            "count_note": (
                "The count may exceed 100: the index admits multiple share "
                "classes of one company (GOOG and GOOGL are both members), so "
                "it holds ~100 companies, not ~100 tickers."),
        },
        "count": len(symbols),
        "symbols": symbols,
    }
    os.makedirs(os.path.dirname(LIST_PATH), exist_ok=True)
    tmp = LIST_PATH + ".tmp"
    with open(tmp, "w") as f:
        json.dump(payload, f, indent=2)
        f.write("\n")
    os.replace(tmp, LIST_PATH)
    print(f"\nwrote {os.path.relpath(LIST_PATH, BASE_DIR)} "
          f"({len(symbols)} symbols, as_of {now.date().isoformat()})")
    print("review the diff, then commit it deliberately.")
    return 0


if __name__ == "__main__":
    sys.exit(main())
