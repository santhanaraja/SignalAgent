#!/usr/bin/env python3
"""PINS FOR THE US-LISTING GATE — each DEMONSTRATED against its bug.

3443.TW (Taiwan) and 6701.T (Tokyo) reached the pool because nothing in
pool construction ever asked where a security is listed. These pins run the
real functions — listing_venue.filter_us_listed, the auto-live entry path,
and universe_builder._group_composite — first with the gate removed, so the
foreign line is seen entering the pool and the order, and only then with the
gate on.

The venue fixtures are the real fields, read from yfinance on 2026-09-21:
  AAPL     exchange NMS  market us_market  currency USD  financialCurrency USD
  TSM      exchange NYQ  market us_market  currency USD  financialCurrency TWD
  3443.TW  exchange TAI  market tw_market  currency TWD
  6701.T   exchange JPX  market jp_market  currency JPY
TSM is the one that matters most: a filter that reads financialCurrency, or
that reads the country of the company rather than of the listing, throws out
every ADR in the pool — and TSM has been a real holding.
"""
import os
import statistics
import sys

REPO = os.path.dirname(os.path.abspath(__file__))
sys.path.insert(0, REPO)
sys.path.insert(0, os.path.join(REPO, "scripts"))

import listing_venue as lv                    # noqa: E402
import auto_trader as at                      # noqa: E402
import universe_builder as ub                 # noqa: E402
from broker_adapter import ShadowBroker       # noqa: E402

FAILS = []
VENUES = {
    "AAPL":    {"exchange": "NMS", "market": "us_market", "currency": "USD"},
    "TSM":     {"exchange": "NYQ", "market": "us_market", "currency": "USD"},
    "MU":      {"exchange": "NMS", "market": "us_market", "currency": "USD"},
    "QCOM":    {"exchange": "NMS", "market": "us_market", "currency": "USD"},
    "3443.TW": {"exchange": "TAI", "market": "tw_market", "currency": "TWD"},
    "6701.T":  {"exchange": "JPX", "market": "jp_market", "currency": "JPY"},
    "NOFIELD": {},
    "ODD.TW":  {"exchange": "NMS", "market": "us_market", "currency": "USD"},
}


def fetch(tk):
    return VENUES.get(tk, {})


def check(name, cond, detail=""):
    print(f"  {'OK  ' if cond else 'FAIL'} {name}" + (f" — {detail}" if detail else ""))
    if not cond:
        FAILS.append(name)


POOL = ["AAPL", "MU", "QCOM", "TSM", "3443.TW", "6701.T"]
print("PINS — the US-listing gate, each shown failing first\n")

# (a) and (b): the two known foreign lines
print("(a) 3443.TW in a source")
no_gate = sorted(POOL)                       # the pool as it is built today
check("bug reintroduced leaves 3443.TW in the pool", "3443.TW" in no_gate,
      "this is today's behaviour: nothing asks where it is listed")
kept, excluded, disagreements = lv.filter_us_listed(POOL, fetch)
row = next((e for e in excluded if e["ticker"] == "3443.TW"), None)
check("gate excludes it", "3443.TW" not in kept and row is not None)
check("with its exchange and reason logged",
      row and row["exchange"] == "TAI" and "not a US exchange" in row["reason"],
      row["reason"] if row else "")

print("(b) 6701.T in a source")
check("bug reintroduced leaves 6701.T in the pool", "6701.T" in no_gate)
row = next((e for e in excluded if e["ticker"] == "6701.T"), None)
check("gate excludes it with its reason",
      "6701.T" not in kept and row and row["exchange"] == "JPX")

# (c) the over-reach test
print("(c) TSM, a US-listed ADR, must be KEPT")
over_reach = [t for t in POOL
              if (VENUES.get(t, {}).get("currency") == "USD"
                  and lv.classify(t, lambda x: {"currency": "TWD"}).us_listed)]
check("a filter reading financialCurrency would drop TSM",
      not lv.classify("TSM", lambda t: {"exchange": "NYQ",
                                        "currency": "TWD"}).us_listed,
      "TSM reports financialCurrency TWD; reading it excludes every ADR")
check("the gate keeps TSM", "TSM" in kept)
check("and keeps the plain US names", {"AAPL", "MU", "QCOM"} <= set(kept))

# (d) missing field
print("(d) missing exchange field")
kept2, excl2, _ = lv.filter_us_listed(["AAPL", "NOFIELD"], fetch)
check("bug reintroduced would let an unresolvable name through",
      lv.classify("NOFIELD", lambda t: {"exchange": "NMS"}).us_listed,
      "with a field present it passes; the pin is about its ABSENCE")
row = next((e for e in excl2 if e["ticker"] == "NOFIELD"), None)
check("gate excludes it", "NOFIELD" not in kept2 and row is not None)
check("AND logs it rather than dropping it silently",
      row and "no exchange or market field" in row["reason"]
      and "data failure" in row["reason"], row["reason"] if row else "")

# (e) suffix versus field
print("(e) suffix and exchange field disagree")
kept3, excl3, dis = lv.filter_us_listed(["ODD.TW"], fetch)
check("the exchange field wins, so the name is KEPT", kept3 == ["ODD.TW"],
      "a suffix list as the gate would have thrown this away")
check("and the disagreement is logged, both readings recorded",
      dis and dis[0]["ticker"] == "ODD.TW" and "venue field governs" in dis[0]["note"])

# (f) defence in depth: the pool filter regressed and the foreign line got through
print("(f) a foreign listing forced past the pool filter")
art = {"generated_at": "2026-09-21T22:08:01+00:00",
       "regime": {"date": "2026-09-21",
                  "chassis": {"replay": {"end": "2026-09-21"},
                              "exposure_ceiling_pct": 50.0}},
       "r28": {"ceiling_pct": 50.0, "summary": {"no_price": 0},
               "ceiling": {"status": "compliant"}},
       "position_signals": {"tickers": {}},
       "candidate_grades": {"3443.TW": {"grade": "A+", "group": "Semiconductors"}}}
# 3443.TW quoted around TWD 1,850 — about USD 61 at roughly 30.5 TWD/USD.
# Read as dollars it is thirty times its real price, so a 6.5% slot buys 3
# shares of something worth about $182 instead of the intended $6,337.50.
frame = lambda t: {"close": 1850.0, "sma20": 1800.0, "atr14": 60.0,
                   "basis_bar": "2026-09-21"}
import datetime as dt
NOW = dt.datetime(2026, 9, 21, 22, 30, tzinfo=dt.timezone.utc)


def run(guards, venue_of=fetch):
    return at.run(artifact=art, ladder_state={}, broker=ShadowBroker(
        {"positions": {}, "prices": {}}), frame=frame, scores={"3443.TW": 81},
        capital=97500.0, target_session="2026-09-21", now_utc=NOW,
        stage="shadow", guards=guards,
        kill_reader=lambda: {"kill_switch": "off"}, venue_of=venue_of)


bug = run(at.Guards(us_listing_only=False, usd_only=False))
bought = [i for i in bug["intents"] if i["side"] == "buy"]
check("bug reintroduced buys the Taiwan listing",
      len(bought) == 1 and bought[0]["ticker"] == "3443.TW",
      f"sized {bought[0]['qty']} shares at a TWD price read as dollars"
      if bought else "")
fix = run(at.Guards())
check("the entry path refuses it even with the pool filter gone",
      not [i for i in fix["intents"] if i["side"] == "buy"]
      and any(r["code"] == "VENUE_NOT_US" for r in fix["refusals"]))

# (g) a non-USD price reaching the sizing path
print("(g) a non-USD price reaching the sizing path")
usd_venue = lv.classify("3443.TW", lambda t: {"exchange": "NMS",
                                              "currency": "TWD"})
raised = False
try:
    lv.assert_usd("3443.TW", usd_venue)
except ValueError as e:
    raised = "TWD" in str(e)
check("the sizing assertion refuses a TWD price", raised)
fix2 = run(at.Guards(us_listing_only=False))
check("and the run refuses and alerts rather than sizing it",
      not [i for i in fix2["intents"] if i["side"] == "buy"]
      and any(r["code"] == "CURRENCY_NOT_USD" for r in fix2["refusals"]))

# (h) the group recomputation
print("(h) a group's composite recomputes once its foreign members are gone")
W = {"ytd": 0.50, "r3m": 0.30, "r1m": 0.20}
members = [
    {"ticker": "MU", "ytd": 256.11, "r3m": 40.0, "r1m": 10.0},
    {"ticker": "MRVL", "ytd": 187.84, "r3m": 30.0, "r1m": 8.0},
    {"ticker": "TSM", "ytd": 44.16, "r3m": 20.0, "r1m": 6.0},
    {"ticker": "3443.TW", "ytd": 238.00, "r3m": 35.0, "r1m": 9.0},
    {"ticker": "ADI", "ytd": 39.78, "r3m": 12.0, "r1m": 4.0},
]
with_fx = ub._group_composite(members, W)
without = ub._group_composite([m for m in members if m["ticker"] != "3443.TW"], W)
hand_med = statistics.median([256.11, 187.84, 44.16, 39.78])
check("the composite moves when the foreign member is removed",
      with_fx[0] != without[0],
      f"{with_fx[0]} with 3443.TW, {without[0]} without")
check("and the recomputation matches the formula by hand",
      abs(without[1] - round(hand_med, 2)) < 0.005,
      f"median YTD without it is {without[1]}")
check("the formula itself is untouched — same function, same weights",
      ub._group_composite(members, W) == with_fx)

print()
if FAILS:
    print(f"{len(FAILS)} PIN(S) FAILED: {FAILS}")
    raise SystemExit(1)
print("All foreign-listing pins passed (each demonstrated against its bug).")
