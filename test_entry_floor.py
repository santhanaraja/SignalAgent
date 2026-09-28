#!/usr/bin/env python3
"""THE ENTRY-FLOOR PIN — written before the mechanism is chosen.

THE INVARIANT, which holds whatever mechanism wins: AN ENTRY THAT WOULD
EXECUTE AT OR BELOW ITS OWN STOP MUST NOT EXECUTE. A fill at or under the
stop is a stop-out on arrival, and its R denominator is zero or negative,
so the position cannot even be sized in risk terms.

WHY THE CHECK CANNOT LIVE ON THE CLOSE. Row 1 of the grade requires close
> SMA20, so EVERY A+ name passes "price above stop" by construction on
close data. The only price that can fail the test is the fill, and the
fill does not exist when an evening order is built. That is why this is a
missing MECHANISM and not a missing validation, and why the pin is written
against a fixture that SUPPLIES a fill price: it can then be pointed at
whichever mechanism is chosen without being rewritten.

NOTHING HERE IS WIRED INTO THE LIVE PATH. `entry_floor_verdict` is the
invariant expressed as a function so it can be pinned today; the live code
is untouched.

THE FIXTURES ARE REAL. LITE on 2026-09-28 graded A+ on the 09-25 close at
941.65 with a 911.6960 stop and an 8.67% slack to its 1023.2645 line, and
by 11:33 ET traded at 892.92 — $18.78 BELOW the stop. The three entries
actually taken that day are the over-reach control: every one of them
filled above its stop and inside its band.
"""
import os
import sys

REPO = os.path.dirname(os.path.abspath(__file__))
sys.path.insert(0, REPO)
sys.path.insert(0, os.path.join(REPO, "scripts"))

import auto_trader as at                      # noqa: E402
from broker_adapter import ShadowBroker       # noqa: E402

FAILS = []


def check(name, cond, detail=""):
    print(f"  {'OK  ' if cond else 'FAIL'} {name}" + (f" — {detail}" if detail else ""))
    if not cond:
        FAILS.append(name)


def entry_floor_verdict(fill, stop, line, *, floor=True, ceiling=True):
    """The invariant. Returns (allowed, reason).

    `floor` and `ceiling` exist so the pins can remove exactly one bound
    and watch the bug happen, which is the house standard: a guard that
    was never seen failing has not been demonstrated.
    """
    if fill is None or stop is None:
        return False, "no fill or no stop — an unpriced entry cannot be checked"
    if floor and fill <= stop:
        return False, (f"fill {fill:.4f} is at or below the stop {stop:.4f} — "
                       f"a stop-out on arrival, and R is {fill - stop:+.4f} "
                       f"per share, so the position cannot be sized in risk")
    if ceiling and line is not None and fill > line:
        return False, (f"fill {fill:.4f} is above the 1.8x line {line:.4f} — "
                       f"past the extension guard the grade was given under")
    return True, "fill inside [stop, line]"


# The real numbers, 2026-09-28.
LITE = dict(close=941.65, stop=911.6960, line=1023.2645, fill=892.92)
TAKEN = {          # ticker: (fill, stop, line) — all three filled inside the band
    "TER":  (398.56, 363.2875, 398.5752),
    "NTAP": (204.39, 190.7075, 204.5418),
    "AAPL": (339.86, 329.3960, 341.6605),
}

print("ENTRY-FLOOR PINS — each shown failing against the bug first\n")

# (1) the live path queues LITE, because it can only see the close
print("(1) the close-basis path cannot catch it")
art = {"generated_at": "2026-09-25T23:48:16+00:00",
       "regime": {"date": "2026-09-25",
                  "chassis": {"replay": {"end": "2026-09-25"},
                              "exposure_ceiling_pct": 50.0}},
       "r28": {"ceiling_pct": 50.0, "summary": {"no_price": 0},
               "ceiling": {"status": "compliant"}},
       "position_signals": {"tickers": {}},
       "candidate_grades": {"LITE": {"grade": "A+", "group": "Communications Equipment"}}}
frame = lambda t: {"close": LITE["close"], "sma20": LITE["stop"],
                   "atr14": (LITE["line"] - LITE["stop"]) / 1.8,
                   "basis_bar": "2026-09-25"}
ref = []
intents, _ = at.compute_entries(art, frame, {"LITE": 92}, {}, lambda t: 0.0,
                                97500.0, "2026-09-25", at.Guards(), ref,
                                venue_of=lambda t: {"exchange": "NMS", "currency": "USD"})
check("today's path queues LITE with no floor",
      len(intents) == 1 and intents[0].ticker == "LITE",
      f"{intents[0].qty} shares, limit {intents[0].limit}, risk "
      f"${intents[0].risk_usd:,.2f} — all computed on the close" if intents else "")
check("and its own reconciliation test passes, because close > stop always holds for an A+",
      LITE["stop"] < LITE["close"] < LITE["line"],
      "row 1 of the grade guarantees it — the close-basis check is vacuous")

# (2) the invariant, with the floor removed and then restored
print("\n(2) the floor, against the LITE fill")
ok, why = entry_floor_verdict(LITE["fill"], LITE["stop"], LITE["line"], floor=False)
check("bug reintroduced lets the fill through", ok,
      f"executes at {LITE['fill']} against a {LITE['stop']} stop, "
      f"${(LITE['stop'] - LITE['fill']) * 6:,.2f} underwater on 6 shares at the open")
ok, why = entry_floor_verdict(LITE["fill"], LITE["stop"], LITE["line"])
check("the floor refuses it", not ok, why)
check("and says why in R terms, not just in price", "R is" in why)

# (3) a fill exactly AT the stop is refused too — R is zero, not small
print("\n(3) a fill exactly at the stop")
ok, why = entry_floor_verdict(LITE["stop"], LITE["stop"], LITE["line"])
check("refused at equality", not ok, why)

# (4) no over-reach: the three entries actually taken must all pass
print("\n(4) over-reach control — the three fills actually taken on 2026-09-28")
for t, (fill, stop, line) in TAKEN.items():
    ok, why = entry_floor_verdict(fill, stop, line)
    check(f"{t} passes", ok, f"fill {fill} inside [{stop}, {line}]")

# (5) the ceiling half: a fill above the line is refused, and removing the
#     ceiling lets it through — the same shape as the floor
print("\n(5) the other bound, so the band is closed on both sides")
ok, _ = entry_floor_verdict(1100.00, LITE["stop"], LITE["line"], ceiling=False)
check("bug reintroduced fills above the line", ok, "past the extension guard")
ok, why = entry_floor_verdict(1100.00, LITE["stop"], LITE["line"])
check("the ceiling refuses it", not ok, why)

# (6) an unpriced entry is refused rather than assumed good
print("\n(6) no fill at all")
ok, why = entry_floor_verdict(None, LITE["stop"], LITE["line"])
check("refused, not defaulted", not ok, why)

print()
if FAILS:
    print(f"{len(FAILS)} PIN(S) FAILED: {FAILS}")
    raise SystemExit(1)
print("All entry-floor pins passed (each demonstrated against its bug).")
