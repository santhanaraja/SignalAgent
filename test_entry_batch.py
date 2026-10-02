#!/usr/bin/env python3
"""PINS FOR scripts/add_entry.py AND THE 2026-10-01 BATCH.

Seven entries and one close landed in one session — the largest single
update this book has had. Entries had been hand-edited into
positions.json until then; add_entry.py exists because hand-editing
seven rows is not defensible, and these pins are its laws plus the
batch's own arithmetic.

ANCHORED TO THE DIFF, NOT TO THE BOOK. Two pin files were rewritten the
same day for measuring the live book instead of the change — asserting
"35 -> 38 closed rows" and "the seven holdings", both of which went red
as soon as the book moved. So the batch is checked on the revision pair
that performed it, located BY CONTENT: the revision where AAPL leaves
holdings[] and arrives in closed[]. The current file is checked only
for what is genuinely a current-file fact.
"""
import copy
import json
import os
import subprocess
import sys

REPO = os.path.dirname(os.path.abspath(__file__))
LEDGER = os.path.join(REPO, "framework", "state", "positions.json")
sys.path.insert(0, os.path.join(REPO, "scripts"))
from close_position import override_count, validate_ledger  # noqa: E402

FAILS = []
CAP = 97500.0
#  ticker: (shares, fill, entry_stop, initial risk, position $)
BATCH = {
    "NVDA": (27, 230.76, 223.1252, 206.1396, 6230.52),
    "FFIV": (14, 449.27, 423.9295, 354.7670, 6289.78),
    "STX":  (6, 937.00, 862.6306, 446.2164, 5622.00),
    "FTNT": (35, 178.38, 168.2475, 354.6375, 6243.30),
    "PANW": (15, 392.20, 364.0160, 422.7600, 5883.00),
    "CRL":  (21, 288.30, 283.8700, 93.0300, 6054.30),
    "MCHP": (81, 78.935, 74.1445, 388.0305, 6393.735),
    # APH was reported unfilled on 2026-10-01 and withheld rather than
    # written on the strength of the range; the operator supplied the
    # placement (~13:45 ET) and fill (13:57:51) times on 2026-10-02 and the
    # tape corroborates them — the 13:57 bar is the first after 13:45 whose
    # low (86.49) crosses the 86.52 limit.
    "APH":  (75, 86.52, 81.3053, 391.1025, 6489.00),
}
AAPL = {"ticker": "AAPL", "entry_date": "2026-09-28", "entry_price": 339.86,
        "shares": 18, "entry_stop": 329.396, "exit_date": "2026-10-01",
        "exit_fill": 329.72, "fees_usd": 0.0, "exit_reason": "system_stop",
        "realized_usd": -182.52, "realized_r": -0.969,
        "capital_usd_at_exit": 97500.0, "fill_source": "broker_statement",
        "basis": "actual_fill"}
# THE SIX CARRIED OVER. Not a whole-book snapshot: an earlier version
# pinned all thirteen names and the book's total initial risk, which
# would go red the next time any position is opened or closed — the
# exact defect this file's docstring says it was written to avoid. What
# is durable is the BATCH: the seven arrived, AAPL left, the six that
# were already there were not touched.
CARRIED = {"ANET", "WAT", "HPE", "TER", "NTAP", "ZBRA"}


def check(name, cond, detail=""):
    print(f"  {'OK  ' if cond else 'FAIL'} {name}" + (f" — {detail}" if detail else ""))
    if not cond:
        FAILS.append(name)


def raises(fn):
    try:
        fn()
    except (AssertionError, SystemExit, KeyError, TypeError) as e:
        return str(e)[:160]
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


def batch_pair():
    """(parent, child, sha) for the revision that closed AAPL, by content."""
    log = subprocess.run(["git", "log", "--format=%H", "-80", "--",
                          "framework/state/positions.json"],
                         capture_output=True, text=True, cwd=REPO)
    k = ("AAPL", "2026-09-28")
    for sha in log.stdout.split():
        child = _show(sha)
        if child is None:
            continue
        if k not in {(c["ticker"], c["entry_date"]) for c in child.get("closed", [])}:
            continue
        parent = _show(f"{sha}^")
        if parent is None:
            continue
        if k not in {(h["ticker"], h["entry_date"]) for h in parent.get("holdings", [])}:
            continue
        return parent, child, sha[:7]
    return None, None, None


now = json.load(open(LEDGER))
_held = {h["ticker"] for h in now["holdings"]}
_closed = {(c["ticker"], c["entry_date"]) for c in now["closed"]}
# NOT-RUN rather than FAIL when the batch is absent — see the same note in
# test_orphan_migration.py. exit 3 is scripts/run_pins.py's NOT-RUN signal.
if not (set(BATCH) <= _held) or ("AAPL", "2026-09-28") not in _closed:
    print("ENTRY-BATCH PINS: NOT RUN — the 2026-10-01 batch is not applied.")
    print(f"  missing from holdings: {sorted(set(BATCH) - _held) or 'none'}; "
          f"AAPL closed row present: {('AAPL', '2026-09-28') in _closed}")
    raise SystemExit(3)

print("ENTRY-BATCH PINS (2026-10-01)\n")

# (1) add_entry.py's refusals — durable, independent of the book
print("(1) the tool refuses what it must")
import tempfile                                                 # noqa: E402
import shutil                                                   # noqa: E402
ADD = os.path.join(REPO, "scripts", "add_entry.py")


def run_add(tmp, extra):
    return subprocess.run([sys.executable, ADD, "--file", tmp] + extra,
                          capture_output=True, text=True)


with tempfile.TemporaryDirectory() as d:
    tmp = os.path.join(d, "positions.json")
    shutil.copy(LEDGER, tmp)
    held = json.load(open(tmp))["holdings"][0]
    r = run_add(tmp, ["--ticker", held["ticker"], "--shares", "1",
                      "--entry-price", str(held["entry_price"]),
                      "--entry-date", held["entry_date"],
                      "--entry-stop", str(held["entry_stop"]), "--dry-run"])
    check("a duplicate (ticker, entry_date) is refused", r.returncode != 0
          and "already in holdings" in r.stdout + r.stderr,
          "a fill reported twice is still one fill")
    r = run_add(tmp, ["--ticker", "ZZTEST", "--shares", "10",
                      "--entry-price", "50.0", "--entry-date", "2026-10-01",
                      "--entry-stop", "50.0", "--dry-run"])
    check("entry_stop == entry_price is refused, in R terms", r.returncode != 0
          and "cannot be sized in risk terms" in r.stdout + r.stderr,
          "the message states R, not just a price comparison")
    r = run_add(tmp, ["--ticker", "ZZTEST", "--shares", "10",
                      "--entry-price", "50.0", "--entry-date", "2026-10-1",
                      "--entry-stop", "45.0", "--dry-run"])
    check("an unorderable date is refused, not coerced", r.returncode != 0
          and "ISO" in r.stdout + r.stderr, '"2026-10-1" is not a date')
    # THE RE-ENTRY ORDERING LAW, driven through the real CLI
    doc = json.load(open(tmp))
    row = dict(doc["closed"][0], ticker="ZZORD", entry_date="2026-01-01",
               exit_date="2026-10-01")
    doc["closed"] = doc["closed"] + [row]
    open(tmp, "w").write(json.dumps(doc, indent=2, ensure_ascii=True) + "\n")
    r = run_add(tmp, ["--ticker", "ZZORD", "--shares", "10",
                      "--entry-price", "50.0", "--entry-date", "2026-10-01",
                      "--entry-stop", "45.0", "--dry-run"])
    check("a re-entry on the SAME DAY as the exit is refused — strictly before",
          r.returncode != 0 and "strictly before" in r.stdout + r.stderr)
    r = run_add(tmp, ["--ticker", "ZZORD", "--shares", "10",
                      "--entry-price", "50.0", "--entry-date", "2026-10-02",
                      "--entry-stop", "45.0", "--dry-run"])
    check("and the day AFTER is accepted — a legal re-entry", r.returncode == 0,
          "FTNT is the live case: closed 2026-09-03, re-entered 2026-10-01")
    for bad, why in ((["--shares", "0"], "a zero lot"),
                     (["--shares", "-10"], "a negative lot")):
        r = run_add(tmp, ["--ticker", "ZZQTY", "--entry-price", "50.0",
                          "--entry-date", "2026-10-02", "--entry-stop", "45.0",
                          "--dry-run"] + bad)
        check(f"{why} is refused", r.returncode != 0
              and "at least 1" in r.stdout + r.stderr)
    r = run_add(tmp, ["--ticker", "ZZPX", "--shares", "10",
                      "--entry-price", "0", "--entry-date", "2026-10-02",
                      "--entry-stop", "-5.0", "--dry-run"])
    check("a non-positive entry price is refused", r.returncode != 0
          and "must be positive" in r.stdout + r.stderr)

# (1b) THE WRITE PATH, EXERCISED. Every check above passes --dry-run, so
#      before this the write path of the tool that produced all seven of
#      today's holdings was never run by a pin at all.
print("\n(1b) the write path, actually run")
with tempfile.TemporaryDirectory() as d:
    tmp = os.path.join(d, "positions.json")
    shutil.copy(LEDGER, tmp)
    before = json.load(open(tmp))
    r = run_add(tmp, ["--ticker", "ZZWRITE", "--shares", "13",
                      "--entry-price", "77.77", "--entry-date", "2026-12-31",
                      "--entry-stop", "70.70", "--note", "write-path pin"])
    check("it writes and reports", r.returncode == 0,
          (r.stdout.strip().splitlines() or [""])[-1][:80])
    raw = open(tmp).read()
    after = json.loads(raw)
    check("the bytes on disk are exactly the repo's serialisation",
          raw == json.dumps(after, indent=2, ensure_ascii=True) + "\n")
    check("holdings grew by exactly one and closed[] did not move",
          len(after["holdings"]) == len(before["holdings"]) + 1
          and json.dumps(after["closed"], sort_keys=True)
          == json.dumps(before["closed"], sort_keys=True))
    row = after["holdings"][-1]
    check("the new row's key ORDER is the ledger's, not alphabetical",
          list(row) == ["ticker", "shares", "entry_price", "entry_date",
                        "stop_on_entry", "entry_stop", "note"], str(list(row)))
    check("stop_on_entry is set for us, not taken on trust",
          row["stop_on_entry"] == "sma20_close")
    check("every pre-existing holding is byte-identical",
          json.dumps(after["holdings"][:-1], sort_keys=True)
          == json.dumps(before["holdings"], sort_keys=True))
    check("and the result still satisfies every ledger law",
          raises(lambda: validate_ledger(after)) is None)
    r2 = run_add(tmp, ["--ticker", "ZZWRITE", "--shares", "13",
                       "--entry-price", "77.77", "--entry-date", "2026-12-31",
                       "--entry-stop", "70.70"])
    check("running it twice is refused — the file is not written again",
          r2.returncode != 0 and open(tmp).read() == raw,
          "idempotence by refusal, not by overwrite")

# (2) the batch, on its own diff
print("\n(2) the 2026-10-01 batch, checked on the revision that made it")
parent, child, sha = batch_pair()
if parent is None:
    print("  ..   not committed yet — the batch diff cannot be walked. The "
          "current-file checks below still apply, and this section will "
          "engage once it lands.")
else:
    print(f"       commit {sha} against its parent")
    ph = {h["ticker"]: h["shares"] for h in parent["holdings"]}
    ch = {h["ticker"]: h["shares"] for h in child["holdings"]}
    check("AAPL left holdings and exactly the seven arrived",
          set(ph) - set(ch) == {"AAPL"} and set(ch) - set(ph) == set(BATCH),
          f"-{sorted(set(ph)-set(ch))} +{sorted(set(ch)-set(ph))}")
    check("every carried-over holding is byte-identical",
          all(json.dumps(a, sort_keys=True) == json.dumps(b, sort_keys=True)
              for a, b in [(x, y) for x in parent["holdings"]
                           for y in child["holdings"]
                           if x["ticker"] == y["ticker"]
                           and x["ticker"] != "AAPL"]))
    pc = {(c["ticker"], c["entry_date"]) for c in parent.get("closed", [])}
    cc = {(c["ticker"], c["entry_date"]) for c in child["closed"]}
    check("closed[] grew by exactly the AAPL row",
          cc - pc == {("AAPL", "2026-09-28")} and not (pc - cc))
    check("the watch list did not move in this commit",
          json.dumps(parent.get("watching"), sort_keys=True)
          == json.dumps(child.get("watching"), sort_keys=True))

# (3) the thirteen, and every stored value recomputed
print("\n(3) the batch's own rows, and the arithmetic of the eight")
got = {h["ticker"]: h["shares"] for h in now["holdings"]}
check("the eight are held and AAPL is not",
      set(BATCH) <= set(got) and "AAPL" not in got,
      f"{len(got)} names held: {sorted(got)}")
check("the six that predate the batch are all still held", CARRIED <= set(got),
      f"missing {sorted(CARRIED - set(got)) or 'none'}")
by = {h["ticker"]: h for h in now["holdings"]}
tot_risk = 0.0
for t, (sh, fill, stop, risk, posn) in BATCH.items():
    h = by.get(t)
    if not h:
        check(f"{t} present", False, "missing")
        continue
    r = h["shares"] * (h["entry_price"] - h["entry_stop"])
    tot_risk += r
    check(f"{t}: {sh} @ {fill} stop {stop} -> risk ${risk:,.2f}",
          (h["shares"], h["entry_price"], h["entry_stop"]) == (sh, fill, stop)
          and abs(r - risk) < 0.005
          and abs(h["shares"] * h["entry_price"] - posn) < 0.005
          and h["entry_date"] == "2026-10-01"
          and h["stop_on_entry"] == "sma20_close")
check(f"the eight carry ${tot_risk:,.2f} of initial risk in total",
      abs(tot_risk - 2656.684) < 0.01, f"{tot_risk / CAP * 100:.4f}% of capital")
six = sum(h["shares"] * (h["entry_price"] - h["entry_stop"])
          for h in now["holdings"] if h["ticker"] in CARRIED)
check("the six carry $2,265.75, so the fourteen carried $4,922.43 = 5.0487% "
      "once APH was recorded",
      abs(six - 2265.7481) < 0.01 and abs(six + tot_risk - 4922.4321) < 0.02,
      f"six ${six:,.2f} + seven ${tot_risk:,.2f} = ${six + tot_risk:,.2f}; "
      "the figure is pinned on those thirteen rows, not on the book's size, "
      "so a later entry does not falsify it")

# (4) AAPL's closed row
print("\n(4) AAPL out, and the attribution that governs it")
a = [c for c in now["closed"] if (c["ticker"], c["entry_date"])
     == ("AAPL", "2026-09-28")]
check("the row exists and is the only AAPL row", len(a) == 1)
if a:
    a = a[0]
    diff = [f"{k}: {v!r} -> {a.get(k)!r}" for k, v in AAPL.items()
            if a.get(k) != v]
    check("every pinned field matches", not diff, "; ".join(diff))
    r = a["realized_usd"] / (a["shares"] * (a["entry_price"] - a["entry_stop"]))
    check("realised recomputes from the stored stop",
          a["realized_usd"] == round(18 * (329.72 - 339.86) - 0.0, 2)
          and abs(r - a["realized_r"]) <= 0.005,
          f"${a['realized_usd']:,.2f} = {a['realized_r']:+.4f}R")
    check("it is system_stop with overrides ZERO — DVN's shape",
          a["exit_reason"] == "system_stop"
          and override_count(a["overrides"]) == 0)
    check("bug reintroduced: filed discretionary it would need an override",
          raises(lambda: validate_ledger(
              dict(now, closed=[dict(a, exit_reason="discretionary")])))
          is not None)
    blob = a["note"] + a["overrides"]
    for anchor in ("bar_date 2026-09-29", "+$25.20", "-$19.44", "+$5.76",
                   "334.6991", "6.8136", "RE_ENTRY_ARMING"):
        check(f"the note still carries {anchor!r}", anchor in blob)

# (5) the laws over the whole file
print("\n(5) the ledger's laws")
check("validate_ledger passes", raises(lambda: validate_ledger(now)) is None)
check("no ticker is in both holdings and closed for the SAME entry_date",
      not ({(h["ticker"], h["entry_date"]) for h in now["holdings"]}
           & {(c["ticker"], c["entry_date"]) for c in now["closed"]}))
check("FTNT is in both arrays — a legal re-entry, exit strictly before entry",
      "FTNT" in got
      and max(c["exit_date"] for c in now["closed"] if c["ticker"] == "FTNT")
      < by["FTNT"]["entry_date"],
      "closed 2026-09-03, re-entered 2026-10-01")
check("bug reintroduced: dated the re-entry on the exit day it is refused",
      raises(lambda: validate_ledger(
          dict(now, holdings=[dict(by["FTNT"], entry_date="2026-09-03")])))
      is not None)

print()
if FAILS:
    print(f"{len(FAILS)} PIN(S) FAILED: {FAILS}")
    raise SystemExit(1)
print("All entry-batch pins passed (each demonstrated against its bug).")
