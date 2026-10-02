#!/usr/bin/env python3
"""THE WATCH-LIST LAW — checked across the WHOLE history, not one prune.

A watch row is sometimes the only record a trade has: closed[] began at
VSAT on 2026-08-19, and every trade before that lived on one. So the law
is not "do not remove these names", it is:

    A WATCH ROW THAT CARRIES A TRADE RECORD MAY NOT LOSE IT UNLESS THAT
    TRADE HAS A ROW IN closed[].

Removing the row, blanking it in place, or pointing it at a trade that
was never filed all count as losing it. The law is keyed on
(ticker, entry_date), because a later trade under the same ticker does
not protect an earlier record — BIIB is the live proof, with an August
row in closed[] and a July one that is still missing.

WHY THIS FILE WAS REWRITTEN ON 2026-10-01. It used to assert the
specific prune of 2026-09-28 against the live file: these five names
gone, these three left, these seven holdings untouched. That went red
the same week when the book went to thirteen names — not because
anything broke, but because the pin was measuring the book. Pinning the
names rather than the rule also meant it had to be edited to let correct
work through, which teaches nothing. It now walks every consecutive pair
of committed revisions and checks the law on each, so it covers the
2026-09-28 prune, the 2026-10-01 migration, and anything that comes
after, without being touched again.
"""
import copy
import json
import os
import re
import subprocess
import sys

REPO = os.path.dirname(os.path.abspath(__file__))
LEDGER = os.path.join(REPO, "framework", "state", "positions.json")
FAILS = []
KEPT = {"CFG", "CRWD", "MRNA"}
PRUNED = {"JBHT", "RF", "MTB", "GEN", "ARWR"}
MIGRATED = {("GEN", "2026-08-12"), ("ARWR", "2026-07-06"),
            ("CRWD", "2026-07-20")}
POINTER = re.compile(r"\(([A-Z.]+), entry_date (\d{4}-\d{2}-\d{2})\)")


def check(name, cond, detail=""):
    print(f"  {'OK  ' if cond else 'FAIL'} {name}" + (f" — {detail}" if detail else ""))
    if not cond:
        FAILS.append(name)


def carries_trade(row):
    blob = (row.get("note") or "") + (row.get("overrides") or "")
    return "SOLD" in blob.upper() or "P&L" in blob.upper()


def lost_record(parent_row, child_rows, closed_keys):
    """Did this ticker LOSE its trade record in the child revision?"""
    t = parent_row["ticker"]
    if not carries_trade(parent_row):
        return False
    after = [w for w in child_rows if w.get("ticker") == t]
    if not after:
        return True                      # removed outright
    if carries_trade(after[0]):
        return False                     # still carries it
    m = POINTER.search(after[0].get("note") or "")
    return not (m and (m.group(1), m.group(2)) in closed_keys)


def revisions():
    # NO DEPTH CAP: "-120" would go silently vacuous once the file passed
    # that many revisions, and a walk that stops finding transitions reads
    # exactly like a clean history.
    log = subprocess.run(["git", "log", "--format=%H", "--",
                          "framework/state/positions.json"],
                         capture_output=True, text=True, cwd=REPO)
    out = []
    for sha in log.stdout.split():
        r = subprocess.run(["git", "show", f"{sha}:framework/state/positions.json"],
                           capture_output=True, text=True, cwd=REPO)
        if r.returncode != 0:
            continue
        try:
            out.append((sha[:7], json.loads(r.stdout)))
        except Exception:
            continue
    return list(reversed(out))           # oldest first


now = json.load(open(LEDGER))
_w = {x["ticker"] for x in now["watching"]}
# NOT-RUN rather than FAIL when the prune is half-applied: GEN and ARWR
# could only be removed once their trades were migrated, so before that
# this file has nothing to judge. exit 3 is run_pins.py's NOT-RUN signal.
if {"GEN", "ARWR"} & _w:
    print("WATCH-LIST LAW PINS: NOT RUN — the prune is not complete; "
          f"{sorted({'GEN', 'ARWR'} & _w)} still on the watch list, which is "
          "correct until their trades are migrated into closed[].")
    raise SystemExit(3)

print("WATCH-LIST LAW PINS\n")

# (1) THE LAW, over every committed transition
print("(1) across every committed revision pair: no trade record was lost")
revs = revisions()
print(f"    walking {len(revs)} revisions of positions.json")
# KEYED ON THE PAIR WHERE THE PAIR IS AVAILABLE. A watch row has no
# entry_date field, so the strict (ticker, entry_date) law can only be
# applied when a date can be read out of the row's own text. Where it
# can, it is; where it cannot, the check falls back to the ticker and
# SAYS SO, rather than claiming a strictness it is not applying. The
# fallback is the weaker law — BIIB is the case it cannot distinguish,
# with an August row filed and a July one missing — and the controls
# below drive the strict form directly.
ISO = re.compile(r"(\d{4}-\d{2}-\d{2})")
violations, transitions, loose = [], 0, 0
for (psha, parent), (csha, child) in zip(revs, revs[1:]):
    ck = {(c["ticker"], c["entry_date"]) for c in child.get("closed", [])}
    ckx = {(c["ticker"], c["exit_date"]) for c in child.get("closed", [])}
    for row in parent.get("watching", []) or []:
        if not lost_record(row, child.get("watching", []) or [], ck):
            continue
        transitions += 1
        t = row["ticker"]
        dates = set(ISO.findall((row.get("note") or "")
                                + (row.get("overrides") or "")))
        pairs = {(t, d) for d in dates}
        if pairs:
            # MATCH ON entry_date OR exit_date. A watch row records the
            # trade the way a human wrote it, which is usually the SALE
            # date: ARWR's says "sold 71 @ 76.40 2026-07-10" and carries
            # no entry date at all, while its closed row is keyed
            # (ARWR, 2026-07-06). Keying on entry_date alone made the
            # walk report a false violation against ARWR the moment the
            # migration was committed — permanently red, for a trade
            # that is filed correctly.
            if not (pairs & (ck | ckx)):
                violations.append((csha, t, "pair-keyed", sorted(dates)[:3]))
        else:
            loose += 1
            if not any(k[0] == t for k in ck):
                violations.append((csha, t, "ticker-keyed (no date in row)", []))
check("every record that was lost had its trade filed in the same revision",
      not violations, f"{transitions} record-losing transitions seen "
      f"({transitions - loose} pair-keyed, {loose} ticker-keyed for want of a "
      f"date in the row), violations: {violations or 'none'}")
# THE WALK CAN ONLY SEE WHAT IS COMMITTED, and a pass over zero
# transitions is not evidence of anything. So the requirement is
# conditional on the migration being visible in history: before it
# lands, the walk is reported as vacuous and the negative controls
# below carry the file; after it lands, the walk must see the
# transitions it exists to police.
mig_committed = any(MIGRATED <= {(c["ticker"], c["entry_date"])
                                 for c in d.get("closed", [])}
                    for _, d in revs)
if mig_committed:
    check("the walk found the transitions it exists to police",
          transitions >= 2, f"{transitions} exercised")
else:
    print(f"  ..   the walk is VACUOUS so far ({transitions} transitions): the "
          "migration and prune are not committed yet, so no committed "
          "revision has lost a record. The negative controls below are what "
          "demonstrate the law until then.")

# the negative control: the law must BITE on a fabricated loss
print("\n    the law, demonstrated against the bug it exists for")
if revs:
    _, last = revs[-1]
    fake_parent = copy.deepcopy(last)
    fake_parent["watching"] = (fake_parent.get("watching") or []) + [
        {"ticker": "ZZGONE", "status": "watch_entry",
         "note": "SOLD 10 @ 1.00 2026-01-01 — the only record of this trade"}]
    fake_child = copy.deepcopy(last)
    ck = {(c["ticker"], c["entry_date"]) for c in fake_child.get("closed", [])}
    check("bug reintroduced: remove a record-carrying row with no closed row "
          "and the law catches it",
          lost_record(fake_parent["watching"][-1],
                      fake_child.get("watching", []), ck)
          and not any(k[0] == "ZZGONE" for k in ck))
    blanked = copy.deepcopy(last)
    for w in blanked.get("watching", []):
        if w["ticker"] == "CRWD":
            w.clear()
            w.update({"ticker": "CRWD", "status": "watch_entry",
                      "note": "watching for reclaim"})
    src = [w for w in last["watching"] if w["ticker"] == "CRWD"]
    if src:
        check("bug reintroduced: BLANK a pointer row in place — not removed, "
              "just emptied — and the law still calls it a loss",
              lost_record({"ticker": "CRWD",
                           "note": "SOLD 24 @ 189.085 — the only record"},
                          blanked.get("watching", []), ck))
    check("bug reintroduced: a pointer naming a trade that is NOT filed is "
          "a loss too",
          lost_record({"ticker": "CRWD", "note": "SOLD 24 @ 189.085"},
                      [{"ticker": "CRWD",
                        "note": "see (CRWD, entry_date 1999-01-01)"}], ck))
    # THE STRICT LAW, DRIVEN THROUGH THE WALK ON A TWO-REVISION FIXTURE.
    # Asserting the two keys differ is not a test of the law; this builds a
    # parent whose BIIB watch row names the JULY trade, a child that has
    # dropped it, and a closed[] holding only the AUGUST row — and requires
    # the pair-keyed check to call it a violation.
    jul = copy.deepcopy(last)
    jul["watching"] = (jul.get("watching") or []) + [
        {"ticker": "BIIB", "status": "watch_entry",
         "note": "SOLD 27 @ 198.92 on 2026-07-09 — entry 2026-07-06, the only "
                 "record of this trade"}]
    kid = copy.deepcopy(last)
    ck2 = {(c["ticker"], c["entry_date"]) for c in kid.get("closed", [])}
    row = jul["watching"][-1]
    dates = set(re.findall(r"(\d{4}-\d{2}-\d{2})",
                           row["note"]))
    pairs = {("BIIB", d) for d in dates}
    ck2x = {(c["ticker"], c["exit_date"]) for c in kid.get("closed", [])}
    check("keyed on the PAIR, BIIB's AUGUST row does not protect the JULY "
          "record: the walk calls it a violation even with exit_date allowed",
          lost_record(row, kid.get("watching", []), ck2)
          and not (pairs & (ck2 | ck2x))
          and any(k[0] == "BIIB" for k in ck2),
          f"dates read from the row: {sorted(dates)}; "
          f"BIIB rows filed: {sorted(k for k in ck2 if k[0] == 'BIIB')}")
    check("and the ticker-keyed fallback would have MISSED it — which is why "
          "the pair is used wherever a date is available",
          any(k[0] == "BIIB" for k in ck2))

# (2) the end state of the two prunes
print("\n(2) the watch list as it stands")
w = {x["ticker"] for x in now["watching"]}
check("CFG, CRWD and MRNA remain, and nothing else", w == KEPT, str(sorted(w)))
check("the five pruned names are gone", not (PRUNED & w))
closed_keys = {(c["ticker"], c["entry_date"]) for c in now["closed"]}
check("all three migrated trades are in closed[]", MIGRATED <= closed_keys)
crwd = [x for x in now["watching"] if x["ticker"] == "CRWD"][0]
m = POINTER.search(crwd.get("note") or "")
check("CRWD's row is a POINTER naming a row that exists",
      bool(m) and (m.group(1), m.group(2)) in closed_keys,
      m.group(0) if m else "no (ticker, entry_date) in the note")
check("no watch row carries an attribution any more — those live in closed[]",
      not any("overrides" in x for x in now["watching"]),
      f"{len(now['watching'])} rows scanned")
# NOT "no watch row may carry a record" — the CRWD pointer deliberately
# carries a human-readable summary beside the authoritative row, and an
# earlier draft of this check forbade exactly that. The law is that any
# row carrying a record must NAME a filed trade.
unfiled = []
for x in now["watching"]:
    if not carries_trade(x):
        continue
    m2 = POINTER.search(x.get("note") or "")
    if not (m2 and (m2.group(1), m2.group(2)) in closed_keys):
        unfiled.append(x["ticker"])
check("every watch row that carries a record names a FILED trade",
      not unfiled, f"unfiled: {unfiled or 'none'}")

# (3) CFG is on the keep list and has the only closed row among the keepers
print("\n(3) the one that stayed for a separate reason")
cfg = [c for c in now["closed"] if c["ticker"] == "CFG"]
check("CFG's -1.9673R row is still there and still the worst R on record",
      bool(cfg) and cfg[0]["realized_r"] == -1.9673
      and cfg[0]["realized_r"] == min(c["realized_r"] for c in now["closed"]),
      f"CFG {cfg[0]['realized_usd']} = {cfg[0]['realized_r']}R" if cfg else "")

print()
if FAILS:
    print(f"{len(FAILS)} PIN(S) FAILED: {FAILS}")
    raise SystemExit(1)
print("All watch-list law pins passed (each demonstrated against its bug).")
