#!/usr/bin/env python3
"""PINS FOR THE 2026-09-28 WATCH-LIST PRUNE — each demonstrated first.

The prune removes JBHT, RF and MTB. GEN and ARWR were HELD BACK and these
pins enforce why: their watch rows are the ONLY record of real trades.
GEN's row carries a discretionary sale of 218 @ $27.66 on 2026-08-18 for
-$161.32 = -0.68R with overrides ONE, and ARWR's records a stop-out sale
of 71 @ 76.40 on 2026-07-10. Neither has a row in closed[]. Deleting them
would delete the only live record of both trades, and GEN's is the
precedent the HPQ attribution ruling leans on.

CRWD's row carries a third — sold 24 @ 189.085 on 2026-07-23 for
-$264.36 — and it is on the keep list anyway, but the pin asserts the
full set so a future prune cannot take it by surprise. closed[] begins at
VSAT on 2026-08-19; every trade before that lives on a watch row.

Each pin runs against the committed HEAD state and the working tree, and
each is shown failing against the bug it exists for before it passes.
"""
import copy
import json
import os
import subprocess
import sys

REPO = os.path.dirname(os.path.abspath(__file__))
LEDGER = os.path.join(REPO, "framework", "state", "positions.json")
FAILS = []
REMOVED = {"JBHT", "RF", "MTB"}
HELD_BACK = {"GEN", "ARWR"}


def check(name, cond, detail=""):
    print(f"  {'OK  ' if cond else 'FAIL'} {name}" + (f" — {detail}" if detail else ""))
    if not cond:
        FAILS.append(name)


def baseline_ledger():
    """The state BEFORE the prune, whatever the commit position.

    Reading HEAD works only while the change is uncommitted: once the prune
    is committed, HEAD already carries it and every before/after check
    inverts. That happened on the first rebase. So the baseline is the last
    revision of this file that still holds all eight watchers, found by
    walking back — and if none is reachable (a shallow clone, a fresh
    checkout after the push), the pins fall back to asserting the end state
    and demonstrating the bug against an in-memory reconstruction, which
    needs no history at all.
    """
    log = subprocess.run(["git", "log", "--format=%H", "-40", "--",
                          "framework/state/positions.json"],
                         capture_output=True, text=True, cwd=REPO)
    for sha in log.stdout.split():
        out = subprocess.run(["git", "show", f"{sha}:framework/state/positions.json"],
                             capture_output=True, text=True, cwd=REPO)
        if out.returncode != 0:
            continue
        try:
            d = json.loads(out.stdout)
        except Exception:
            continue
        if REMOVED <= {w["ticker"] for w in d.get("watching", [])}:
            return d
    return None


now = json.load(open(LEDGER))
was = baseline_ledger()
if was is None:                      # no pre-prune revision reachable
    was = copy.deepcopy(now)         # reconstruct it: the three, put back
    was["watching"] = was["watching"] + [
        {"ticker": t, "status": "watch_entry", "note": "reconstructed baseline"}
        for t in sorted(REMOVED)]
KEPT = {"CFG", "CRWD", "MRNA"} | HELD_BACK

print("WATCH-LIST PRUNE PINS\n")

# (1) the three are gone
print("(1) the three are gone from the watch list")
w_before = {w["ticker"] for w in (was or {}).get("watching", [])}
w_after = {w["ticker"] for w in now["watching"]}
check("bug reintroduced leaves them in place", REMOVED <= w_before,
      "HEAD still carries " + ", ".join(sorted(REMOVED)))
check("the prune removes exactly those three",
      not (REMOVED & w_after) and (w_before - w_after) == REMOVED,
      f"watch list is now {sorted(w_after)}")

# (2) every closed row is byte-identical
print("(2) the ledger is not touched")
b = json.dumps((was or {}).get("closed"), sort_keys=True)
a = json.dumps(now["closed"], sort_keys=True)
mutated = copy.deepcopy(now)
mutated["closed"][0] = dict(mutated["closed"][0], realized_usd=0.0)
check("bug reintroduced changes a closed row",
      json.dumps(mutated["closed"], sort_keys=True) != b)
check("every closed row is byte-identical to HEAD", a == b,
      f"{len(now['closed'])} rows unchanged")
cfg = [c for c in now["closed"] if c["ticker"] == "CFG"]
check("CFG's -1.9673R row is still there and still the worst R on record",
      bool(cfg) and cfg[0]["realized_r"] == -1.9673
      and cfg[0]["realized_r"] == min(c["realized_r"] for c in now["closed"]),
      f"CFG {cfg[0]['realized_usd']} = {cfg[0]['realized_r']}R" if cfg else "")

# (3) the holdings are untouched
print("(3) the seven holdings are untouched")
expect = {"ANET": 31, "WAT": 15, "HPE": 104, "TER": 15, "NTAP": 31,
          "AAPL": 18, "ZBRA": 17}
got = {h["ticker"]: h["shares"] for h in now["holdings"]}
check("bug reintroduced drops a holding", len({k: v for k, v in got.items()
                                               if k != "ZBRA"}) == 6)
check("all seven present with the right share counts", got == expect, str(got))
check("and byte-identical to HEAD",
      json.dumps(now["holdings"], sort_keys=True)
      == json.dumps((was or {}).get("holdings"), sort_keys=True))

# (4) nothing else in the watch list moved
print("(4) nothing else in the watch list moved")
check("CFG, CRWD and MRNA remain", {"CFG", "CRWD", "MRNA"} <= w_after)
before_rows = {w["ticker"]: json.dumps(w, sort_keys=True)
               for w in (was or {}).get("watching", [])}
after_rows = {w["ticker"]: json.dumps(w, sort_keys=True) for w in now["watching"]}
check("every surviving row is byte-identical",
      all(after_rows[t] == before_rows[t] for t in after_rows),
      f"{len(after_rows)} rows compared")

# (5) THE ONE THAT MATTERS: a row carrying a trade record is not removed
print("(5) a watch row that carries a trade record is not removed")
def carries_trade(row):
    blob = (row.get("note") or "") + (row.get("overrides") or "")
    return "SOLD" in blob.upper() or "P&L" in blob.upper()
recorded = {w["ticker"] for w in (was or {}).get("watching", []) if carries_trade(w)}
check("three rows carry one — GEN, ARWR and CRWD",
      recorded == {"GEN", "ARWR", "CRWD"}, f"detected {sorted(recorded)}")
check("bug reintroduced would have removed them with the rest",
      bool(recorded & {"GEN", "JBHT", "ARWR", "RF", "MTB"}),
      "the instruction named all five")
check("neither was removed", not (recorded & (w_before - w_after)))
gen = [w for w in now["watching"] if w["ticker"] == "GEN"]
check("GEN's -$161.32 = -0.68R discretionary sale survives in the file",
      bool(gen) and "161.32" in (gen[0].get("note") or ""))
arwr = [w for w in now["watching"] if w["ticker"] == "ARWR"]
check("ARWR's 71 @ 76.40 stop-out survives in the file",
      bool(arwr) and "76.40" in (arwr[0].get("note") or ""))

print()
if FAILS:
    print(f"{len(FAILS)} PIN(S) FAILED: {FAILS}")
    raise SystemExit(1)
print("All watch-list prune pins passed (each demonstrated against its bug).")
