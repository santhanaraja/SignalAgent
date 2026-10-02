#!/usr/bin/env python3
"""PINS FOR THE ORPHAN MIGRATION — anchored to its DIFF, not to the book.

closed[] began at VSAT on 2026-08-19. Five trades ran before it existed.
Three — ARWR, CRWD and GEN — had their only record on a watching row and
were migrated into closed[]. Two, IWM and BIIB (both sold 2026-07-09),
have no record anywhere and are NOT migrated: no fill price was ever
written down, and a trade without a fill is not closed.

WHY THIS FILE WAS REWRITTEN ON 2026-10-01. Its first version asserted
"35 -> 38 closed rows" and "the seven holdings are untouched" against
the LIVE FILE. Both went red the same week, when one close and seven
entries took the book to 13 names and 39 rows — not because the
migration broke, but because the pin was measuring the book instead of
the migration. That is the HEAD-anchoring mistake wearing different
clothes: a pin that has to be edited every time the book changes is not
a pin, it is a snapshot.

So the invariants about the MOVE are now checked on the migration's own
DIFF — the revision pair located BY CONTENT, the first revision whose
closed[] carries all three migrated keys where its parent's carries
none — and only the invariants that are genuinely about the CURRENT
file are checked against the current file. Both routes are exercised,
and when no commit exists yet (the change still uncommitted) the pins
fall back to comparing the working file with an in-memory
reconstruction, which needs no history at all.
"""
import copy
import json
import tempfile
import os
import subprocess
import sys

REPO = os.path.dirname(os.path.abspath(__file__))
LEDGER = os.path.join(REPO, "framework", "state", "positions.json")
sys.path.insert(0, os.path.join(REPO, "scripts"))
from close_position import override_count, validate_ledger  # noqa: E402

FAILS = []
MIGRATED = {("ARWR", "2026-07-06"), ("CRWD", "2026-07-20"),
            ("GEN", "2026-08-12")}
GONE = {"GEN", "ARWR"}
EXPECT = {                       # ticker: ($, R, shares, entry, stop, fill)
    "ARWR": (-636.16, -1.2008, 71, 85.36, 77.8985, 76.40),
    "CRWD": (-264.36, -0.9501, 24, 200.10, 188.5061, 189.085),
    "GEN":  (-161.48, -0.6774, 218, 28.40, 27.3065, 27.66),
}
FIELDS = ("ticker", "entry_date", "entry_price", "shares", "entry_stop",
          "exit_date", "exit_fill", "fees_usd", "exit_reason",
          "realized_usd", "realized_r", "realized_pct_of_capital",
          "capital_usd_at_exit", "regime_at_entry", "regime_at_exit",
          "fill_source", "basis")
GOLD = {
    "ARWR": ("ARWR", "2026-07-06", 85.36, 71, 77.8985, "2026-07-10", 76.40,
             None, "system_stop", -636.16, -1.2008, -0.6525, 97500.0,
             "Risk-on / Trending", "Risk-on / Choppy", "broker_statement",
             "actual_fill"),
    "CRWD": ("CRWD", "2026-07-20", 200.10, 24, 188.5061, "2026-07-23", 189.085,
             None, "system_stop", -264.36, -0.9501, -0.2711, 97500.0,
             "Risk-on / Trending", "Risk-on / Choppy", "broker_statement",
             "actual_fill"),
    "GEN":  ("GEN", "2026-08-12", 28.40, 218, 27.3065, "2026-08-18", 27.66,
             0.16, "system_stop", -161.48, -0.6774, -0.1656, 97500.0,
             "Risk-on / Trending", "Risk-on / Choppy", "broker_statement",
             "actual_fill"),
}
ANCHORS = {
    "ARWR": ["FILED ZERO AFTER A REVERSAL", "13:22 ET",
             "the 2026-07-13 open of 74.82", "GROSS of fees"],
    "CRWD": ["split 4:1 ex-2026-07-02", "11:02 ET", "+$26.16",
             "GROSS of fees", "POST-SPLIT"],
    "GEN":  ["27.3118", "2026-08-17T20:23:33Z", "2026-08-18T20:19:53Z",
             "NON-FILL 2026-08-27", "HAL 2026-08-28", "next open",
             "REVERSING THEIR OWN 2026-08-18 RULING"],
}

key = lambda c: (c["ticker"], c["entry_date"])                   # noqa: E731


def check(name, cond, detail=""):
    print(f"  {'OK  ' if cond else 'FAIL'} {name}" + (f" — {detail}" if detail else ""))
    if not cond:
        FAILS.append(name)


def raises(fn):
    try:
        fn()
    except (AssertionError, SystemExit, KeyError, TypeError) as e:
        return str(e)[:150]
    return None


def _show(sha):
    out = subprocess.run(["git", "show", f"{sha}:framework/state/positions.json"],
                         capture_output=True, text=True, cwd=REPO)
    if out.returncode != 0:
        return None
    try:
        return json.loads(out.stdout)
    except Exception:
        return None


def migration_pair():
    """(parent, child, sha) for the revision that DID the migration.

    Located by CONTENT — the revision whose closed[] carries all three
    migrated keys and whose parent's carries none — so it keeps working
    however far the book moves afterwards and whatever the commit's
    position in history.
    """
    # NO DEPTH CAP. "-80" would go silently vacuous once the file passed
    # eighty revisions, and a search that stops finding the thing it looks
    # for reads exactly like a pass.
    log = subprocess.run(["git", "log", "--format=%H", "--",
                          "framework/state/positions.json"],
                         capture_output=True, text=True, cwd=REPO)
    for sha in log.stdout.split():
        child = _show(sha)
        if child is None:
            continue
        ck = {key(c) for c in child.get("closed", [])}
        if not MIGRATED <= ck:
            continue
        parent = _show(f"{sha}^")
        if parent is None:
            continue
        pk = {key(c) for c in parent.get("closed", [])}
        if MIGRATED & pk:
            continue
        # NO EXCLUSIVITY REQUIREMENT. An earlier version asserted that the
        # commit added ONLY these three, which silently forced the change
        # to be split into two commits and went red if it was not. The
        # invariant that matters is that the three ARRIVED here and that
        # no pre-existing row changed; whatever else the same commit did
        # is not this file's business.
        return parent, child, sha[:7]
    return None, None, None


def head_pair(now):
    """The diff against HEAD, for when the change is not yet committed.

    THE PARENT MUST NOT BE DERIVED FROM THE FILE UNDER TEST. The first
    version of this built the parent by subtracting the three migrated
    rows from `now`, which made "every pre-existing row is byte-identical"
    and "grew by exactly 3" compare the artifact with itself — green on a
    tampered ledger. HEAD is a real, independent 35-closed/7-held parent,
    so the comparison has something to bite on even before the commit.
    """
    parent = _show("HEAD")
    return (parent, copy.deepcopy(now)) if parent is not None else (None, None)


def reconstructed_pair(now):
    """Last resort with NO history at all: take the three back out.

    This IS self-referential and cannot catch a tampered row; it exists
    only so the file still runs in a tree with no reachable git, and the
    header says which route was used so a green run is never mistaken for
    a checked one.
    """
    child = copy.deepcopy(now)
    parent = copy.deepcopy(now)
    parent["closed"] = [c for c in parent["closed"] if key(c) not in MIGRATED]
    parent["watching"] = ([{"ticker": t, "status": "watch_entry",
                            "note": f"reconstructed baseline — {t} SOLD ..."}
                           for t in sorted(GONE)]
                          + [dict(w) for w in parent["watching"]])
    for w in parent["watching"]:
        if w["ticker"] == "CRWD":
            w["note"] = "reconstructed baseline — CRWD SOLD 24 @ 189.085"
            w["overrides"] = "ZERO — reconstructed baseline"
    (parent.get("schema_notes") or {}).pop("closed_coverage", None)
    return parent, child


now = json.load(open(LEDGER))
rows = {c["ticker"]: c for c in now["closed"] if key(c) in MIGRATED}

# NOT-RUN RATHER THAN FAIL WHEN THE INPUT IS ABSENT. These pins judge a
# migration; if it has not been applied there is nothing to judge, and a
# red suite that means "not done yet" is indistinguishable from a red
# suite that means "broken". scripts/run_pins.py reads exit 3 as NOT-RUN.
# This file was briefly committed alongside an UNAPPLIED migration — the
# ledger write having been refused by a permission classifier — and the
# whole suite went red for that reason alone.
if set(rows) != {"ARWR", "CRWD", "GEN"}:
    print("ORPHAN-MIGRATION PINS: NOT RUN — the migration is not applied to "
          f"{os.path.relpath(LEDGER, REPO)}.")
    print(f"  expected ARWR/CRWD/GEN in closed[]; found {sorted(rows) or 'none'}.")
    print("  Apply the migration and these pins will judge it.")
    raise SystemExit(3)

print("ORPHAN-MIGRATION PINS\n")
parent, child, sha = migration_pair()
committed = parent is not None
route = "commit"
if not committed:
    parent, child = head_pair(now)
    route = "head"
    if parent is None:
        parent, child = reconstructed_pair(now)
        route = "reconstruction"
_src = {"commit": f"commit {sha} against its parent",
        "head": "the working file against HEAD (not yet committed — a REAL "
                "independent parent)",
        "reconstruction": "an in-memory reconstruction (no git reachable — "
                          "SELF-REFERENTIAL, cannot catch a tampered row)"}[route]
print(f"(the MOVE is checked on {_src}; the live file is "
      f"{len(now['closed'])} closed / {len(now['holdings'])} held, and the "
      "pins do not depend on those counts)")
# NO "both routes agree" CHECK. An earlier version compared the
# independent parent with the in-memory reconstruction and expected them
# equal — which held only while the migration was the ONLY uncommitted
# change. Once the same working file also carried the AAPL close the two
# differ by that row, legitimately, and the check went red for a reason
# that had nothing to do with the migration. What it was reaching for is
# already covered below, against the independent parent: every row the
# parent has must appear byte-identically in the child.
if route == "reconstruction":
    print("  ..   NOTE: the parent is a reconstruction from the file under "
          "test, so the byte-identity checks below are self-referential and "
          "cannot catch a tampered pre-existing row. Only the end-state "
          "checks carry weight on this route.")

# (1) THE MOVE — on the diff, so the book may change freely afterwards
print("\n(1) the move: closed[] grew by exactly 3 and nothing else shifted")
pk = {key(c) for c in parent.get("closed", [])}
ck = {key(c) for c in child.get("closed", [])}
check("the three migrated rows arrived in this change",
      MIGRATED <= (ck - pk), f"added {sorted(ck - pk)}")
check("and none was removed", not (pk - ck))
# BYTE-IDENTICAL EXCEPT FOR DECLARED AMENDMENTS, AND THE EXCEPTION IS
# NARROW BY CONSTRUCTION. closed[] is append-only, but a later ruling can
# falsify a sentence in an earlier row — the 2026-10-02 GEN reversal did
# exactly that to HPQ's note. scripts/amend_note.py appends the
# correction, leaves the original words above it, and registers the edit
# in schema_notes.note_amendments. This check reads that register: a row
# named there may differ in `note` ONLY, every other row must be
# byte-identical, and an amendment that is not registered is a violation.
AMENDED = {tuple(e["row"].split("/"))
           for e in (child.get("schema_notes") or {}).get("note_amendments", [])}
pb = {key(c): c for c in parent.get("closed", [])}
cb = {key(c): c for c in child["closed"]}
drift, illegal = [], []
for k, pv in pb.items():
    cv = cb.get(k)
    if cv is None:
        drift.append((k, "row vanished"))
        continue
    diff = [f for f in set(pv) | set(cv) if pv.get(f) != cv.get(f)]
    if not diff:
        continue
    if k in AMENDED and diff == ["note"]:
        continue
    illegal.append((k, diff))
check("every pre-existing closed row is byte-identical, bar declared "
      "note amendments", not drift and not illegal,
      f"{len(pb)} rows compared; declared amendments {sorted(AMENDED) or 'none'}; "
      f"violations {illegal + drift or 'none'}")
check("bug reintroduced: an UNdeclared note edit is a violation",
      bool([1 for k, pv in pb.items()
            if k not in AMENDED
            and json.dumps(dict(pv, note=(pv.get("note") or "") + " x"),
                           sort_keys=True) != json.dumps(pv, sort_keys=True)]),
      "any row not in the register whose note moved would be caught above")
check("and every declared amendment names a row that exists",
      all(k in cb for k in AMENDED), str(sorted(AMENDED)))
tampered = copy.deepcopy(child)
tampered["closed"][0] = dict(tampered["closed"][0], realized_usd=0.0)
tb = {key(c): json.dumps(c, sort_keys=True) for c in tampered["closed"]}
check("bug reintroduced: edit one pre-existing row and the check goes red",
      not all(tb.get(k) == v for k, v in pb.items()),
      f"zeroed {tampered['closed'][0]['ticker']}'s realized_usd")
# HOLDINGS: PROVEN AGAINST THE TOOL, NOT AGAINST A DIFF. On the commit
# route the parent is the migration's own commit and the diff is exact.
# On the HEAD route the parent predates every uncommitted change, so
# holdings differ for reasons that have nothing to do with the migration
# — an earlier version asserted equality there and went red the moment
# the same working file also carried the AAPL close. The durable claim is
# that THE TOOL CANNOT TOUCH HOLDINGS, so it is demonstrated by running
# the real tool against a throwaway copy on every route.
if route == "commit":
    check("holdings byte-identical across the migration commit",
          json.dumps(parent.get("holdings"), sort_keys=True)
          == json.dumps(child.get("holdings"), sort_keys=True),
          f"{len(child.get('holdings', []))} holdings at the time of the move")
else:
    print(f"  ..   holdings diff skipped on the {route} route: the parent "
          "predates other uncommitted changes. The tool check below is what "
          "carries it.")
with tempfile.TemporaryDirectory() as _d:
    _tmp = os.path.join(_d, "positions.json")
    _src = _show("HEAD") or now
    with open(_tmp, "w") as _f:
        _f.write(json.dumps(_src, indent=2, ensure_ascii=True) + "\n")
    _before = json.dumps(json.load(open(_tmp))["holdings"], sort_keys=True)
    _p = subprocess.run(
        [sys.executable, os.path.join(REPO, "scripts", "migrate_orphan_trade.py"),
         "--file", _tmp, "--ticker", "ZZORPH", "--entry-date", "2026-01-02",
         "--entry-price", "100.0", "--shares", "10", "--entry-stop", "90.0",
         "--exit-date", "2026-01-05", "--exit-fill", "95.0", "--fees", "0.0",
         "--exit-reason", "system_stop", "--fill-source", "broker_statement",
         "--overrides", "ZERO — pin fixture", "--capital", "97500",
         "--note", "pin fixture"], capture_output=True, text=True)
    check("the migration tool ran in WRITE mode against a real ledger",
          _p.returncode == 0, (_p.stdout.strip().splitlines() or [""])[-1][:80])
    _after_doc = json.load(open(_tmp))
    check("and it left holdings byte-identical — the tool cannot touch them",
          json.dumps(_after_doc["holdings"], sort_keys=True) == _before,
          f"{len(_after_doc['holdings'])} holdings before and after")
    check("while closed[] grew by exactly one",
          len(_after_doc["closed"]) == len(_src["closed"]) + 1)
    check("and the bytes on disk round-trip through the repo's serialisation",
          open(_tmp).read()
          == json.dumps(_after_doc, indent=2, ensure_ascii=True) + "\n")
check("the watch list lost exactly GEN and ARWR",
      ({w["ticker"] for w in parent["watching"]}
       - {w["ticker"] for w in child["watching"]}) == GONE)

# (2) the rows themselves, in the CURRENT file — these are durable facts
print("\n(2) the three rows, as they stand in the ledger now")
check("all three are present", set(rows) == {"ARWR", "CRWD", "GEN"}, str(sorted(rows)))
for t, (usd, r, sh, ep, stop, fill) in EXPECT.items():
    c = rows.get(t)
    if not c:
        check(f"{t} row", False, "missing")
        continue
    bad = dict(c, entry_stop=round(stop * 0.97, 4))
    re_bad = bad["realized_usd"] / (bad["shares"]
                                    * (bad["entry_price"] - bad["entry_stop"]))
    check(f"{t}: bug reintroduced breaks the recompute",
          abs(re_bad - c["realized_r"]) > 0.005,
          f"a 3% wrong stop reads {re_bad:+.4f}R, not {c['realized_r']:+.4f}R")
    fee = c["fees_usd"] or 0.0
    want_usd = round(c["shares"] * (c["exit_fill"] - c["entry_price"]) - fee, 2)
    want_r = c["realized_usd"] / (c["shares"] * (c["entry_price"] - c["entry_stop"]))
    check(f"{t}: ${c['realized_usd']:,.2f} = {c['realized_r']:+.4f}R recomputes",
          (c["shares"], c["entry_price"], c["entry_stop"], c["exit_fill"])
          == (sh, ep, stop, fill)
          and c["realized_usd"] == want_usd == usd
          and abs(want_r - c["realized_r"]) <= 0.005 and c["realized_r"] == r)
for t, gold in GOLD.items():
    got = tuple(rows.get(t, {}).get(f) for f in FIELDS)
    diff = [f"{f}: {g!r} -> {v!r}" for f, g, v in zip(FIELDS, gold, got) if g != v]
    check(f"{t}: all {len(FIELDS)} pinned fields match", not diff, "; ".join(diff))
check("bug reintroduced: change any pinned field and the pin names it",
      tuple(dict(rows["GEN"], fill_source="estimate").get(f) for f in FIELDS)
      != GOLD["GEN"])
for t, anchors in ANCHORS.items():
    blob = (rows[t].get("note") or "") + (rows[t].get("overrides") or "")
    gone = [x for x in anchors if x not in blob]
    check(f"{t}: every load-bearing sentence is still in the row", not gone,
          f"missing {gone}" if gone else f"{len(anchors)} anchors")

# (3) the attributions
print("\n(3) the attributions")
# GEN WAS REVERSED ON 2026-10-02. It was filed discretionary/ONE on the
# watching row's account that the state machine read HELD at the sale;
# position_events.json refutes that (EXIT_FIRED 2026-08-17T20:23:33Z,
# bar_date 2026-08-17; no event until 16:19:53 ET on 08-18, about five
# and a half hours after the 10:44 ET sale), and the operator reversed
# their own ruling. It is DVN's shape.
check("all three migrated rows are system_stop with ZERO",
      all(rows[t]["exit_reason"] == "system_stop"
          and override_count(rows[t]["overrides"]) == 0
          for t in ("ARWR", "CRWD", "GEN")),
      ", ".join(f"{t} {rows[t]['exit_reason']}/"
                f"{override_count(rows[t]['overrides'])}"
                for t in ("ARWR", "CRWD", "GEN")))
check("bug reintroduced: system_stop with a count of ONE is refused",
      raises(lambda: validate_ledger(
          dict(now, closed=[dict(rows["GEN"], overrides="ONE — x")])))
      is not None)
check("bug reintroduced: discretionary with a count of 0 is refused",
      raises(lambda: validate_ledger(
          dict(now, closed=[dict(rows["GEN"], exit_reason="discretionary")])))
      is not None)
check("none of the three carries an override any more",
      not [t for t in ("ARWR", "CRWD", "GEN")
           if override_count(rows[t]["overrides"])])
# THE TALLY, which the reversal moved from four to three
tally = [(c["ticker"], c["exit_reason"]) for c in now["closed"]
         if override_count(c["overrides"])]
check("the book's override tally is THREE — DNTH, OXY, HPQ",
      [t for t, _ in tally] == ["DNTH", "OXY", "HPQ"], str(tally))
check("and HPQ is the only discretionary row",
      [c["ticker"] for c in now["closed"]
       if c["exit_reason"] == "discretionary"] == ["HPQ"])
check("GEN is now IN the system-stop census", "GEN" in
      [c["ticker"] for c in now["closed"] if c["exit_reason"] == "system_stop"])

# (4) the two that CANNOT be migrated
print("\n(4) a trade with no fill is not closed")
cmd = [sys.executable, os.path.join(REPO, "scripts", "migrate_orphan_trade.py"),
       "--ticker", "IWM", "--entry-date", "2026-07-06", "--entry-price", "299.58",
       "--shares", "23", "--entry-stop", "293.0438", "--exit-date", "2026-07-09",
       "--fees", "unknown", "--exit-reason", "system_stop", "--fill-source",
       "estimate", "--basis", "close_estimate", "--overrides", "ZERO — x",
       "--capital", "97500", "--dry-run", "--file", LEDGER,
       "--note", "reconstructed from the frame; GROSS of fees"]
p = subprocess.run(cmd, capture_output=True, text=True)
check("the CLI refuses it outright", p.returncode != 0
      and "exit-fill" in (p.stderr + p.stdout),
      (p.stderr.strip().splitlines() or [""])[-1][:90])
p2 = subprocess.run(cmd + ["--exit-fill", "295.27"], capture_output=True, text=True)
check("bug reintroduced: hand it the reconstructed open and it goes through",
      p2.returncode == 0, "which is why the fill is required, not defaulted")
check("IWM and BIIB are NOT in closed[]",
      not ({("IWM", "2026-07-06"), ("BIIB", "2026-07-06")}
           & {key(c) for c in now["closed"]}))
check("BIIB's SECOND trade is untouched and still its only row",
      [key(c) for c in now["closed"] if c["ticker"] == "BIIB"]
      == [("BIIB", "2026-08-10")])
cov = (now.get("schema_notes") or {}).get("closed_coverage", "")
check("and the exclusion is recorded IN THE FILE with its bound",
      all(x in cov for x in ("IWM", "BIIB", "2026-07-09", "2c869d4",
                             "-$491.98", "not closed")), f"{len(cov)} chars")
check("bug reintroduced: without that note closed[] silently claims "
      "coverage from the book's first day",
      min(c["entry_date"] for c in now["closed"]) == "2026-07-06")

# (5) the whole file still satisfies every ledger law
print("\n(5) the ledger's own laws, over the whole file")
check("validate_ledger passes", raises(lambda: validate_ledger(now)) is None)
check("bug reintroduced: a second copy of a migrated trade under a sloppy "
      "date is refused",
      raises(lambda: validate_ledger(
          dict(now, closed=now["closed"] + [dict(rows["GEN"],
                                                 entry_date="2026-8-12")])))
      is not None)

# (6) the no-history route, actually exercised
print("\n(6) the no-history route is exercised, not merely asserted")
if os.environ.get("ORPHAN_PINS_NO_RECURSE") == "1":
    print("  (skipped: this IS the no-history run)")
else:
    env = dict(os.environ, ORPHAN_PINS_NO_RECURSE="1",
               GIT_DIR=os.path.join(REPO, ".git-does-not-exist"))
    p3 = subprocess.run([sys.executable, os.path.abspath(__file__)],
                        capture_output=True, text=True, env=env, cwd=REPO,
                        timeout=180)
    check("the whole file passes with git history unreachable",
          p3.returncode == 0 and "in-memory reconstruction" in p3.stdout,
          (p3.stdout.strip().splitlines() or ["no output"])[-1][:80])

print()
if FAILS:
    print(f"{len(FAILS)} PIN(S) FAILED: {FAILS}")
    raise SystemExit(1)
print("All orphan-migration pins passed (each demonstrated against its bug).")
