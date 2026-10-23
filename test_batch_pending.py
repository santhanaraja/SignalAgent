#!/usr/bin/env python3
"""PINS FOR THE 2026-10-23 BATCH AND THE PENDING-EXIT-DATE LAW.

Nine sales and one ten-session-late entry, recorded from PRICES ALONE:
the operator had the fills and not the dates, so every closed row here
carries `exit_date: null` plus a declared `exit_date_pending` reason.

WHY A NEW LAW RATHER THAN A LOOSENED ONE. validate_ledger already said
"a date nobody can order is not a date" and refused nulls outright, and
that paragraph exists because an unorderable date let a duplicate trade
validate twice and be double-counted in every total. Writing the TRIGGER
date instead is the conflation that has produced four errors in this
ledger. So the null is permitted only when DECLARED, and the re-entry
law is made to REFUSE a live holding of the same ticker rather than read
an unorderable date as satisfied. These pins hold that shape in place:
a bare null must still be rejected, and a pending row must still block
re-entry.

ANCHORED TO FACTS, NOT TO THE BOOK'S SIZE. Two pin files were rewritten
for measuring the live book instead of the change. So the nine are keyed
by (ticker, entry_date) and resolved across holdings[] UNION closed[],
the same discipline that let test_entry_batch survive these nine closes
green.
"""
import copy
import json
import os
import shutil
import subprocess
import sys
import tempfile

REPO = os.path.dirname(os.path.abspath(__file__))
LEDGER = os.path.join(REPO, "framework", "state", "positions.json")
sys.path.insert(0, os.path.join(REPO, "scripts"))
from close_position import override_count, validate_ledger  # noqa: E402

FAILS = []
CAP = 97500.0

#  ticker: (entry_date, shares, entry, entry_stop, fill, realised, R,
#           trigger_close, trigger_date, next_open, gap)
BATCH = {
    "ANET": ("2026-09-21", 31, 205.00, 193.8155, 203.51, -46.19, -0.1332,
             204.52, "2026-10-13", 200.43, -126.79),
    "HPE":  ("2026-09-24", 104, 61.38, 56.4186, 56.205, -538.20, -1.0431,
             63.11, "2026-10-14", 59.91, -332.80),
    "TER":  ("2026-09-28", 15, 398.56, 363.2875, 365.00, -503.40, -0.9514,
             383.85, "2026-10-12", 371.50, -185.25),
    "ZBRA": ("2026-09-28", 17, 372.52, 353.2785, 377.26, 80.58, 0.2463,
             369.01, "2026-10-16", 373.35, 73.78),
    "NVDA": ("2026-10-01", 27, 230.76, 223.1252, 224.52, -168.48, -0.8173,
             226.32, "2026-10-12", 224.34, -53.46),
    "FFIV": ("2026-10-01", 14, 449.27, 423.9295, 447.8491, -19.89, -0.0561,
             442.03, "2026-10-16", 447.72, 79.66),
    "FTNT": ("2026-10-01", 35, 178.38, 168.2475, 176.42, -68.60, -0.1934,
             181.91, "2026-10-15", 182.35, 15.40),
    "PANW": ("2026-10-01", 15, 392.20, 364.0160, 362.40, -447.00, -1.0573,
             375.04, "2026-10-14", 367.05, -119.85),
    "APH":  ("2026-10-01", 75, 86.52, 81.3053, 75.4184, -832.62, -2.1289,
             82.69, "2026-10-14", 80.38, -173.25),
}
# The two whose exit was WITHDRAWN by a return to HELD, so whose
# attribution is conditional on the fill date. Recorded, not absorbed.
WITHDRAWN = {"ZBRA": "2026-10-20", "FFIV": "2026-10-21"}
APA = ("2026-10-09", 130, 46.425, 43.9415, 322.855, 6035.25)
# the book after the write
BOOK_AFTER = {"WAT": "2026-09-21", "NTAP": "2026-09-28",
              "CRL": "2026-10-01", "APA": "2026-10-09"}


def check(label, ok, detail=""):
    print(f"  {'OK  ' if ok else 'FAIL'} {label}" + (f" — {detail}" if detail else ""))
    if not ok:
        FAILS.append(label)


def raises(fn):
    try:
        fn()
    except BaseException as e:
        return e
    return None


now = json.load(open(LEDGER))
_held = {(h["ticker"], h["entry_date"]): h for h in now["holdings"]}
_clsd = {(c["ticker"], c["entry_date"]): c for c in now["closed"]}
_all = {**_clsd, **_held}
KEYS = {(t, v[0]) for t, v in BATCH.items()}

# NOT-RUN rather than a sea of red when the batch is simply not applied,
# and FAIL when it is PARTLY applied — the three-way split that
# test_entry_batch needed and this file inherits deliberately.
_missing = sorted(k for k in KEYS if k not in _clsd)
if len(_missing) == len(KEYS) and ("APA", "2026-10-09") not in _held:
    print("BATCH-PENDING PINS: NOT RUN — the 2026-10-23 batch is not "
          "applied at all (no closed row for any of the nine, and APA is "
          "not held). These are NOT passes.")
    raise SystemExit(3)
if _missing or ("APA", "2026-10-09") not in _held:
    print("BATCH-PENDING PINS\n")
    check("the batch is applied whole, not in part", False,
          f"not closed: {_missing or 'none'}; APA held: "
          f"{('APA', '2026-10-09') in _held} — a partial batch is a "
          "half-written ledger, which must fail rather than skip")
    raise SystemExit(1)

print("(1) the nine closed rows, recomputed from their own stored fields")
tot = 0.0
for t, (ed, sh, e, st, fill, usd, R, tc, td, no, gap) in BATCH.items():
    c = _clsd[(t, ed)]
    risk = sh * (e - st)
    tot += usd
    check(f"{t}: {sh} @ {fill} -> ${usd:,.2f} = {R:+.4f}R, recomputed",
          (c["shares"], c["entry_price"], c["entry_stop"], c["exit_fill"])
          == (sh, e, st, fill)
          and abs(c["realized_usd"] - round(sh * (fill - e), 2)) < 0.005
          and abs(c["realized_usd"] - usd) < 0.005
          and abs(c["realized_r"] - round(usd / risk, 4)) < 0.0001
          and abs(c["realized_r"] - R) < 0.0001)
    check(f"{t}: the exit date is NULL and the reason is DECLARED",
          c["exit_date"] is None
          and isinstance(c.get("exit_date_pending"), str)
          and c["exit_date_pending"].strip() != "")
    check(f"{t}: the trigger date {td} was NOT written as the exit date",
          c["exit_date"] != td and td in c["note"] and str(tc) in c["note"])
    # BOTH LEGS, because the delay never needed the fill date either. An
    # earlier draft of these rows marked the delay "pending", following an
    # instruction that asserted it needed one; delay = (fill - the open
    # after this name's own trigger) x shares takes only the fill PRICE and
    # the trigger, both of which were in hand. The pin asserts the
    # decomposition CLOSES against the trigger close, which is what makes
    # a dropped or mis-signed leg visible.
    _delay = (fill - no) * sh
    check(f"{t}: gap {gap:+,.2f} and delay {_delay:+,.2f} close against "
          f"its OWN trigger close {tc}",
          abs((no - tc) * sh - gap) < 0.005
          and abs(gap + _delay - sh * (fill - tc)) < 0.005
          and f"{abs(gap):,.2f}" in c["note"]
          and f"{abs(_delay):,.2f}" in c["note"]
          # KEYED ON THE AFFIRMATIVE MARKER, NOT ON THE ABSENCE OF THE WORD
          # "pending". The corrected paragraph RETRACTS the pending framing
          # and therefore quotes it, so an absence test fails on a correct
          # row — which is exactly what it did on all nine.
          and "BOTH LEGS ARE COMPUTED" in c["note"])
# BOTH AGGREGATES NOW READ THE LEDGER. As first written they summed the
# BATCH table's own literals and compared the result to a constant derived
# from those same literals, and picked the win out of the same table —
# tautologies that read no ledger field and could not fail. The point of
# an aggregate is to catch a row that drifted individually in a way the
# per-row checks tolerate, which requires the ledger as the source.
_led_sum = round(sum(_clsd[(t, v[0])]["realized_usd"] for t, v in BATCH.items()), 2)
check(f"the nine realise ${tot:,.2f} in total, summed from the LEDGER",
      abs(_led_sum - -2543.80) < 0.01 and abs(_led_sum - tot) < 0.01,
      f"ledger ${_led_sum:,.2f} vs table ${tot:,.2f} — date-independent: "
      "shares x (fill - entry), risk fixed at entry")
_led_wins = sorted(t for t, v in BATCH.items()
                   if _clsd[(t, v[0])]["realized_usd"] > 0)
check("ZBRA is the one WIN among the nine, read from the LEDGER",
      _led_wins == ["ZBRA"],
      f"wins per the ledger: {_led_wins} — sold 1.28% below its "
      "2026-10-08 close but above its own entry")

# THE LEDGER-WIDE WORST-R RANKING, PINNED HERE BECAUSE NOTHING PINNED IT
# WHEN IT MATTERED. test_watchlist_prune.py used to assert CFG held the
# worst R, as a side-effect of a watch-list check; that claim was removed
# on 2026-10-23 when APH displaced it, leaving the ranking unpinned — and
# in the same batch ARWR 2026-07-06's committed note ("CFG's -1.9673R
# still holds the worst-R record") went false without anything noticing.
# A superlative in one row can be falsified by a write to another, so the
# ranking needs a pin of its own, in a ledger-law file rather than a
# watch-list one.
_byr = sorted(now["closed"], key=lambda c: c["realized_r"])
check("APH 2026-10-01 is the worst R in the ledger at -2.1289, ahead of "
      "CFG -1.9673",
      (_byr[0]["ticker"], _byr[0]["entry_date"]) == ("APH", "2026-10-01")
      and _byr[0]["realized_r"] == -2.1289
      and _byr[1]["realized_r"] == -1.9673,
      f"worst three: {[(c['ticker'], c['realized_r']) for c in _byr[:3]]}")
check("and no row still asserts the displaced CFG record in the present "
      "tense",
      not any("still holds the worst-R record" in (c.get("note") or "")
              for c in now["closed"]),
      "ARWR 2026-07-06 carried exactly that sentence until this batch "
      "falsified it")

print("\n(2) attribution, per name, and the override tally")
for t, (ed, *_rest) in BATCH.items():
    c = _clsd[(t, ed)]
    check(f"{t} is system_stop with a parseable ZERO",
          c["exit_reason"] == "system_stop" and override_count(c["overrides"]) == 0)
for t, held_on in WITHDRAWN.items():
    c = _clsd[(t, BATCH[t][0])]
    # CASE-INSENSITIVE ON THE PHRASE, EXACT ON THE DATES. An earlier
    # version of this pin looked for "ONE override" while the note says
    # "ONE OVERRIDE", and failed a row whose content was correct — the pin
    # was brittle, not the record. The dates stay exact because getting
    # the withdrawal date or the pivotal fill date wrong is the defect
    # this pin is for.
    _n = c["note"]
    _nl = _n.lower()
    check(f"{t} returned to HELD on {held_on} — the conditional is RECORDED",
          held_on in _n and "2026-10-22" in _n
          and "discretionary" in _nl and "one override" in _nl
          and "HPQ" in _n and "override tally" in _nl,
          "a fill dated while the ladder read HELD makes this an override; "
          "the row says so rather than absorbing it")
check("the override tally over the WHOLE ledger is still THREE",
      sum(override_count(c["overrides"]) or 0 for c in now["closed"]) == 3,
      "DNTH, OXY, HPQ")
check("every closed row's attribution parses to a count",
      all(override_count(c["overrides"]) is not None for c in now["closed"]),
      "an uncountable attribution scores an operator decision as doctrine")

print("\n(3) APA — the entry that was ten sessions late")
ed, sh, e, st, risk, posn = APA
h = _held[("APA", ed)]
check(f"APA {sh} @ {e} stop {st} -> risk ${risk:,.2f}",
      (h["shares"], h["entry_price"], h["entry_stop"]) == (sh, e, st)
      and abs(sh * (e - st) - risk) < 0.005
      and abs(sh * e - posn) < 0.005)
check("APA carries the holding key order and stop_on_entry",
      list(h) == ["ticker", "shares", "entry_price", "entry_date",
                  "stop_on_entry", "entry_stop", "note"]
      and h["stop_on_entry"] == "sma20_close", str(list(h)))
check("APA's note records the as-of trap that would have moved the stop",
      "43.7202" in h["note"] and "2026-10-22" in h["note"]
      and "43.9415" in h["note"],
      "a fetch made after the 10-22 ex-$0.25 back-applies it and returns "
      "43.7202 — $0.2213/share low")
check("APA's note records row 2 flooring the line, with both forms",
      "46.75" in h["note"] and "46.76" in h["note"]
      and "1.8010" in h["note"],
      "rounded 46.76 = 1.8010xATR is OVER the guard; floored 46.75 is not")

print("\n(4) the book after the write")
check("exactly four holdings, and they are WAT NTAP CRL APA",
      {h["ticker"] for h in now["holdings"]} == set(BOOK_AFTER)
      and len(now["holdings"]) == 4,
      str(sorted(h["ticker"] for h in now["holdings"])))
check("none of the nine is still held",
      not ({t for t in BATCH} & {h["ticker"] for h in now["holdings"]}))
check("every one of the nine is in closed[] at its own entry_date",
      all(k in _clsd for k in KEYS))

print("\n(5) THE PENDING-DATE LAW, each bug reintroduced")
check("validate_ledger passes on the ledger as written",
      raises(lambda: validate_ledger(now)) is None)


def _mutate(fn):
    d = copy.deepcopy(now)
    fn(d)
    return d


def _drop_pending(d):
    for c in d["closed"]:
        if c["ticker"] == "ANET" and c["exit_date"] is None:
            c.pop("exit_date_pending", None)


check("bug reintroduced: a BARE null exit_date with no declared reason "
      "is refused",
      raises(lambda: validate_ledger(_mutate(_drop_pending))) is not None,
      "a bare null is indistinguishable from a dropped field")


def _blank_pending(d):
    for c in d["closed"]:
        if c["ticker"] == "ANET" and c["exit_date"] is None:
            c["exit_date_pending"] = "   "


check("bug reintroduced: a whitespace-only reason is refused",
      raises(lambda: validate_ledger(_mutate(_blank_pending))) is not None)


def _reenter(d):
    d["holdings"].append({"ticker": "ANET", "shares": 1,
                          "entry_price": 200.0, "entry_date": "2026-10-23",
                          "stop_on_entry": "sma20_close",
                          "entry_stop": 190.0, "note": ""})


# ASSERTED ON THE NEW GUARD'S OWN MESSAGE, because "something raised" is
# not discriminating here: the pre-existing _iso() helper ALSO raises on a
# null, so deleting the new guard entirely left this pin green. A pin that
# passes with and without the code it exists for is not a pin. The
# message is the only thing that distinguishes which law caught it.
_re = raises(lambda: validate_ledger(_mutate(_reenter)))
check("bug reintroduced: re-entering a ticker that has a PENDING close "
      "is refused BY THE PENDING LAW, not read as satisfied",
      _re is not None
      and "cannot be ordered against a live entry" in str(_re),
      f"raised: {str(_re)[:90]}")


def _trigger_as_exit(d):
    for c in d["closed"]:
        if c["ticker"] == "ANET" and c["exit_date"] is None:
            c["exit_date"] = "2026-10-13"       # its TRIGGER date
check("a dated close still validates, so the law permits dating it later",
      raises(lambda: validate_ledger(_mutate(_trigger_as_exit))) is None,
      "the pending marker is a waypoint, not a dead end")

print("\n(6) add_entry refuses a re-entry while the close is pending")
# tempfile, NOT a path inside REPO: run_pins.py reports a suite that
# dirties a tracked-or-watched path, and test_backtest_doctrine.py's three
# hard-coded /tmp names are the standing proof that a fixed path also
# collides when two suites run at once. mkdtemp honours TMPDIR.
_td = tempfile.mkdtemp(prefix="pending_pin_")
tmp = os.path.join(_td, "positions.json")
try:
    json.dump(now, open(tmp, "w"), indent=2)
    r = subprocess.run([sys.executable,
                        os.path.join(REPO, "scripts", "add_entry.py"),
                        "--ticker", "ANET", "--shares", "1",
                        "--entry-price", "200.0", "--entry-date", "2026-10-23",
                        "--entry-stop", "190.0", "--file", tmp],
                       capture_output=True, text=True)
    # "PENDING exit_date" is the new block's wording; the pre-existing
    # `late` branch refuses too (str(None) sorts above any 2026 date) but
    # reports "a closed row exits None", so keying on the bare word PENDING
    # left this pin green with the new block deleted.
    check("add_entry.py refuses ANET with the PENDING-specific message",
          r.returncode != 0
          and "PENDING exit_date" in (r.stdout + r.stderr),
          (r.stdout + r.stderr).strip().splitlines()[-1][:120]
          if (r.stdout + r.stderr).strip() else "no output")
    check("and it did not write the file",
          json.load(open(tmp))["holdings"] == now["holdings"])
finally:
    shutil.rmtree(_td, ignore_errors=True)

print("\n(7) close_position's CLI cannot write a dateless row by accident")
for args, why in (
        ([], "neither flag"),
        (["--exit-date", "2026-10-14", "--exit-date-pending", "x"], "both flags")):
    r = subprocess.run([sys.executable,
                        os.path.join(REPO, "scripts", "close_position.py"),
                        "--ticker", "WAT", "--exit-fill", "455.0",
                        "--fees", "0", "--exit-reason", "system_stop",
                        "--fill-source", "broker_statement",
                        "--overrides", "ZERO — x", "--capital", "97500",
                        "--dry-run"] + args,
                       capture_output=True, text=True)
    check(f"refused with {why}",
          r.returncode != 0
          and "exactly one of" in (r.stdout + r.stderr))

print()
if FAILS:
    print(f"{len(FAILS)} PIN(S) FAILED: {FAILS}")
    raise SystemExit(1)
print("All batch-pending pins passed (each demonstrated against its bug).")
