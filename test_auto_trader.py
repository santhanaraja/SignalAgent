#!/usr/bin/env python3
"""PINS FOR THE ORDER AUTOMATION — each DEMONSTRATED, never asserted.

Every pin runs the same fixture twice. First with the guard DISABLED, which
reintroduces exactly the bug the guard exists for, and asserts the wrong
thing actually happens: the ghost sell is produced, the duplicate is placed,
the oversized order goes out. Only then does it run with the guard enabled
and assert the refusal. A pin that never saw the bug happen has not shown
the guard does anything.

Fixtures are anchored to real committed state — the 2026-09-21 post-splitter
bake in which DVN fired on its own confirmed close, the 2026-08-18 bakes
where GEN was unpriced while R28 read compliant, the 2026-09-15 SMCI sale
the artifact went on reporting as HELD — because a fixture invented to fit
the code is a fixture that passes by accident.
"""
import copy
import datetime as dt
import io
import json
import os
import sys
import tempfile

sys.path.insert(0, os.path.join(os.path.dirname(os.path.abspath(__file__)),
                                "scripts"))
import auto_trader as at                                    # noqa: E402
from broker_adapter import ShadowBroker, OpenOrder          # noqa: E402

SESSION = "2026-09-21"
NOW = dt.datetime(2026, 9, 21, 22, 30, tzinfo=dt.timezone.utc)
CAPITAL = 97500.0
FAILS = []


def check(name, cond, detail=""):
    print(f"  {'OK  ' if cond else 'FAIL'} {name}" + (f" — {detail}" if detail else ""))
    if not cond:
        FAILS.append(name)


def artifact(holdings=None, candidates=None, watchers=None, *,
             generated="2026-09-21T22:08:01.000000+00:00",
             replay_end=SESSION, no_price=0, ceiling_pct=50.0):
    tickers = {}
    for tk, row in (holdings or {}).items():
        tickers[tk] = dict(kind="holding", **row)
    for tk, row in (watchers or {}).items():
        tickers[tk] = dict(kind="watching", **row)
    return {
        "generated_at": generated,
        "regime": {"date": SESSION, "regime": "Risk-on / Choppy",
                   "chassis": {"confirmed_state": "In-Trend-Throttled",
                               "replay": {"end": replay_end} if replay_end else None,
                               "exposure_ceiling_pct": ceiling_pct}},
        "r28": {"ceiling_pct": ceiling_pct, "summary": {"no_price": no_price},
                "ceiling": {"status": "compliant"}},
        "position_signals": {"tickers": tickers},
        "candidate_grades": candidates or {},
    }


def ladder(tickers, session=SESSION):
    return {t: {"state": "HELD", "last_close_date": session} for t in tickers}


def frame_of(d):
    accessed = {"keys": set()}

    def f(tk):
        row = dict(d.get(tk, {}))
        accessed["keys"] |= set(row)
        return row
    f.accessed = accessed
    return f


# real 2026-09-21 numbers: DVN fired on its own confirmed close
DVN_ROW = dict(state="EXIT_FIRED", close=47.59, shares=45,
               stop={"level": 48.36}, group="Oil & Gas Exploration & Production")
HPQ_ROW = dict(state="HELD", close=32.95, shares=190,
               stop={"level": 31.89}, group="Technology Hardware, Storage & Peripherals")
# real 2026-09-18 frame figures for the A+ names
FRAME = {
    "A":    dict(close=156.47, sma20=151.7375, atr14=4.0829, basis_bar=SESSION),
    "AAPL": dict(close=336.13, sma20=322.6400, atr14=7.7007, basis_bar=SESSION),
    "ABBV": dict(close=263.96, sma20=259.6855, atr14=5.5493, basis_bar=SESSION),
    "ANET": dict(close=199.39, sma20=193.8155, atr14=7.7644, basis_bar=SESSION),
    "DELL": dict(close=568.06, sma20=503.6705, atr14=37.1957, basis_bar=SESSION),
    "HPE":  dict(close=60.76, sma20=55.1090, atr14=4.1052, basis_bar=SESSION),
    "TSM":  dict(close=434.67, sma20=421.5297, atr14=9.7863, basis_bar=SESSION),
    "VRSN": dict(close=303.12, sma20=292.4110, atr14=7.5486, basis_bar=SESSION),
    "WAT":  dict(close=421.63, sma20=412.8205, atr14=11.4521, basis_bar=SESSION),
}
SCORES = {"A": 80, "AAPL": 81, "ABBV": 86, "ANET": 92, "DELL": 81, "HPE": 92,
          "TSM": 92, "VRSN": 86, "WAT": 86}
GROUPS = {"A": "Life Sciences Tools & Services", "AAPL": "Technology Hardware, Storage & Peripherals",
          "ABBV": "Biotechnology", "ANET": "Communications Equipment",
          "DELL": "Technology Hardware, Storage & Peripherals",
          "HPE": "Technology Hardware, Storage & Peripherals",
          "TSM": "Semiconductors", "VRSN": "Internet Services & Infrastructure",
          "WAT": "Life Sciences Tools & Services"}
CANDS = {t: {"grade": "A+", "group": GROUPS[t]} for t in FRAME}


US_VENUE = lambda tk: {"exchange": "NMS", "currency": "USD"}


def go(art, led, broker, *, guards=None, stage="shadow", store=None,
       frame=None, kill=None, selection="risk_first", secrets=None,
       now=NOW, capital=CAPITAL, venue_of=US_VENUE):
    return at.run(artifact=art, ladder_state=led, broker=broker,
                  frame=frame or frame_of(FRAME), scores=SCORES,
                  capital=capital, target_session=SESSION, now_utc=now,
                  stage=stage, guards=guards or at.Guards(),
                  kill_reader=kill or (lambda: {"kill_switch": "off"}),
                  store=store if store is not None else {"placed_keys": [], "days": {}},
                  selection=selection, secrets=secrets, venue_of=venue_of)


def sells(rec):
    return [i for i in rec["intents"] if i["side"] == "sell"]


def buys(rec):
    return [i for i in rec["intents"] if i["side"] == "buy"]


def codes(rec):
    return {r["code"] for r in rec["refusals"]}


print("PINS — each shown failing against the reintroduced bug first\n")

# (a) ghost — the SMCI shape: sold 2026-09-15, still HELD in the artifact
print("(a) ghost: EXIT_FIRED on a ticker the broker does not hold")
art = artifact({"SMCI": dict(state="EXIT_FIRED", close=39.09, shares=167,
                             stop={"level": 37.88}, group="Technology Hardware, Storage & Peripherals")})
led = ladder(["SMCI"])
bug = go(art, led, ShadowBroker({"positions": {}, "prices": {}}),
         guards=at.Guards(broker_truth=False))
check("bug reintroduced sells a position that does not exist",
      len(sells(bug)) == 1 and sells(bug)[0]["qty"] == 167,
      "a sell with no position behind it opens a SHORT in a margin account")
fix = go(art, led, ShadowBroker({"positions": {}, "prices": {}}))
check("guard refuses and names it", not sells(fix) and "GHOST" in codes(fix))

# (b) quantity clamp
print("(b) quantity: artifact says 167, broker holds 100")
bk = ShadowBroker({"positions": {"SMCI": 100}, "prices": {"SMCI": 39.09}})
bug = go(art, led, ShadowBroker({"positions": {"SMCI": 100}, "prices": {"SMCI": 39.09}}),
         guards=at.Guards(broker_truth=False))
check("bug reintroduced sells the artifact's 167", sells(bug)[0]["qty"] == 167)
fix = go(art, led, bk)
check("guard sells 100 and logs the clamp",
      sells(fix)[0]["qty"] == 100 and "QTY_CLAMPED" in codes(fix))

# (c) double-fire
print("(c) double-fire: the same key twice")
art2 = artifact({"DVN": DVN_ROW})
led2 = ladder(["DVN"])
bkp = {"positions": {"DVN": 45}, "prices": {"DVN": 47.59}}
store = {"placed_keys": [f"DVN|{SESSION}|sell"], "days": {}}
bug = go(art2, led2, ShadowBroker(bkp), guards=at.Guards(idempotent_local=False),
         store=copy.deepcopy(store))
check("bug reintroduced re-sells an already-placed key", len(sells(bug)) == 1)
fix = go(art2, led2, ShadowBroker(bkp), store=copy.deepcopy(store))
check("guard refuses the duplicate", not sells(fix) and "DUPLICATE_LOCAL" in codes(fix))

# (d) resting order at the broker
print("(d) a resting order already at the broker for the key")
rest = dict(bkp, open_orders=[dict(ticker="DVN", side="sell", qty=45,
                                   placed_for_session=SESSION)])
bug = go(art2, led2, ShadowBroker(rest), guards=at.Guards(idempotent_broker=False))
check("bug reintroduced duplicates a resting order", len(sells(bug)) == 1)
fix = go(art2, led2, ShadowBroker(rest))
check("guard sees the broker's book", not sells(fix) and "DUPLICATE_BROKER" in codes(fix))

# (e) unpriced bake — the 2026-08-18 GEN shape
print("(e) unpriced bake (2026-08-18: GEN unpriced, R28 still 'compliant')")
art3 = artifact({"DVN": DVN_ROW, "GEN": dict(state="HELD", close=None, shares=218,
                                             stop={"level": 27.31}, group="Systems Software")},
                no_price=1)
bug = go(art3, ladder(["DVN", "GEN"]), ShadowBroker(bkp),
         guards=at.Guards(bake_priced=False))
check("bug reintroduced trades off a partially-priced bake", bug["status"] == "OK")
fix = go(art3, ladder(["DVN", "GEN"]), ShadowBroker(bkp))
check("guard refuses the whole run",
      fix["status"] == "REFUSED" and fix["code"] == "BAKE_UNPRICED")

# (f) forming-bar / pre-splitter — the OXY shape
print("(f) pre-splitter bake carrying yesterday's close (the OXY shape)")
art4 = artifact({"DVN": DVN_ROW}, generated="2026-09-21T11:58:27.000000+00:00",
                replay_end="2026-09-18")
bug = go(art4, ladder(["DVN"], "2026-09-18"), ShadowBroker(bkp),
         guards=at.Guards(bake_post_splitter=False, ladder_basis_matches=False,
                          bake_fresh=False))
check("bug reintroduced acts on a forming-bar preview", len(sells(bug)) == 1,
      "this is the pre-emption D-018 forbids — OXY was sold before its trigger")
fix = go(art4, ladder(["DVN"], "2026-09-18"), ShadowBroker(bkp))
check("guard refuses on the basis bar",
      fix["status"] == "REFUSED" and fix["code"] == "BAKE_WRONG_SESSION")

# (g) stale bake
print("(g) stale bake")
old = artifact({"DVN": DVN_ROW}, generated="2026-09-18T21:47:06.000000+00:00")
bug = go(old, led2, ShadowBroker(bkp), guards=at.Guards(bake_fresh=False))
check("bug reintroduced trades off a three-day-old bake", bug["status"] == "OK")
fix = go(old, led2, ShadowBroker(bkp))
check("guard refuses", fix["status"] == "REFUSED" and fix["code"] == "BAKE_STALE")

# (h) no bake by the deadline — silence must not read as safety
print("(h) no run by the deadline")
deadline = dt.datetime(2026, 9, 22, 0, 0, tzinfo=dt.timezone.utc)
quiet = at.watchdog_alert(None, deadline + dt.timedelta(minutes=5), deadline)
check("watchdog alerts when the log shows no run", bool(quiet))
check("watchdog is silent when a run is recorded",
      at.watchdog_alert(deadline - dt.timedelta(hours=1),
                        deadline + dt.timedelta(minutes=5), deadline) is None)

# (i) token expired
print("(i) refresh token expired (they die every seven days)")
dead = dict(bkp, token_expiry="2026-09-20T12:00:00+00:00")
bug = go(art2, led2, ShadowBroker(dead), guards=at.Guards(token_valid=False))
check("bug reintroduced keeps trading on a dead token", bug["status"] == "OK")
fix = go(art2, led2, ShadowBroker(dead))
check("guard refuses, alerts and hands back a by-hand list",
      fix["status"] == "REFUSED" and fix["code"] == "TOKEN_EXPIRED"
      and "by_hand" in fix)

# (j) live with a non-Schwab adapter
print("(j) live stage with an adapter that is not Schwab")
LIVEBK = dict(bkp, token_expiry="2026-10-01T12:00:00+00:00")
bug = go(art2, led2, ShadowBroker(LIVEBK), stage="live_full",
         guards=at.Guards(adapter_allowed=False))
check("bug reintroduced runs live against the shadow adapter", bug["status"] == "OK")
fix = go(art2, led2, ShadowBroker(LIVEBK), stage="live_full")
check("guard refuses to start",
      fix["status"] == "REFUSED" and fix["code"] == "ADAPTER_NOT_ALLOWED")

# (k) kill switch
print("(k) kill switch ON")
killbk = ShadowBroker(dict(LIVEBK, open_orders=[dict(ticker="DVN", side="sell", qty=45,
                                                     placed_for_session=SESSION)]))
fix = go(art2, led2, killbk, kill=lambda: {"kill_switch": "on"},
         stage="live_full", guards=at.Guards(adapter_allowed=False))
check("no new orders placed", not fix["placed"])
check("resting automation orders cancelled", killbk.cancelled)
check("Slack still shows what was paused",
      "KILL SWITCH ON" in at.slack_text(fix))
check("an unreadable switch reads as KILLED (fail safe)",
      at.read_kill_switch(lambda: (_ for _ in ()).throw(IOError("no net")),
                          at.Guards()) is True)
check("a typo reads as KILLED",
      at.read_kill_switch(lambda: {"kill_switch": "0ff"}, at.Guards()) is True)

# (l) single order over the 8% cap — the cap catches a SIZING BUG, since a
# correct 6.5% slot can never reach 8% on its own
print("(l) one order above 8% of capital (a sizing bug, 10x)")
oversized = at.Intent(ticker="DELL", side="buy", qty=110, order_type="limit",
                      limit=570.62, trigger_date=SESSION,
                      reason="sizing bug: 10x the slot", dollars=62486.60,
                      risk_usd=7082.80, stop=503.6705, basis_bar=SESSION)
ref = []
bug = at.apply_order_gates([oversized], capital=CAPITAL, stage="shadow",
                           placed_keys=set(), broker_open_keys=set(),
                           guards=at.Guards(order_size_cap=False), refusals=ref)
check("bug reintroduced lets a 10x order through",
      len(bug) == 1 and bug[0].dollars > CAPITAL * 0.08)
ref2 = []
fix = at.apply_order_gates([oversized], capital=CAPITAL, stage="shadow",
                           placed_keys=set(), broker_open_keys=set(),
                           guards=at.Guards(), refusals=ref2)
check("guard refuses the oversized order",
      not fix and any(r.code == "ORDER_TOO_LARGE" for r in ref2))

# (m) a ninth order in one day
print("(m) a ninth order in one day")
nine = {f"T{i}": {"grade": "A+", "group": f"G{i}"} for i in range(9)}
nineframe = frame_of({f"T{i}": dict(close=100.0, sma20=95.0, atr14=3.0,
                                    basis_bar=SESSION) for i in range(9)})
art9 = artifact({}, candidates=nine, ceiling_pct=90.0)
bug = go(art9, {}, ShadowBroker({"positions": {}, "prices": {}}),
         frame=nineframe, guards=at.Guards(order_count_cap=False))
check("bug reintroduced places all nine", len(buys(bug)) == 9)
fix = go(art9, {}, ShadowBroker({"positions": {}, "prices": {}}), frame=nineframe)
check("guard stops at eight",
      len(buys(fix)) == 8 and "ORDER_COUNT_CAP" in codes(fix))

# (n) entry that would breach the ceiling
print("(n) entry that would take deployment past the ceiling")
full = ShadowBroker({"positions": {"HPQ": 1400}, "prices": {"HPQ": 32.95}})
artn = artifact({"HPQ": HPQ_ROW}, candidates={"ANET": {"grade": "A+", "group": GROUPS["ANET"]}})
bug = go(artn, ladder(["HPQ"]), full, guards=at.Guards(ceiling=False))
check("bug reintroduced buys past the ceiling", len(buys(bug)) == 1)
fix = go(artn, ladder(["HPQ"]), full)
check("guard refuses", not buys(fix) and "NO_ROOM_OR_CAP" in codes(fix))

# (o) Stage 1 one-share ramp
print("(o) Stage 1: every order exactly 1 share")
arto = artifact({}, candidates={"ANET": {"grade": "A+", "group": GROUPS["ANET"]}})
RAMPBK = {"positions": {}, "prices": {}, "token_expiry": "2026-10-01T12:00:00+00:00"}
bug = go(arto, {}, ShadowBroker(RAMPBK),
         stage="live_1share", guards=at.Guards(stage_one_share=False,
                                               adapter_allowed=False))
check("bug reintroduced sends full size in the ramp stage", buys(bug)[0]["qty"] > 1)
fix = go(arto, {}, ShadowBroker(RAMPBK),
         stage="live_1share", guards=at.Guards(adapter_allowed=False))
check("guard forces 1 share", buys(fix)[0]["qty"] == 1)

# (p) a fourth name in one group
print("(p) a fourth name in one group")
artp = artifact({"HPQ": HPQ_ROW},
                candidates={t: {"grade": "A+", "group": GROUPS[t]}
                            for t in ("AAPL", "DELL", "HPE")})
bkp2 = ShadowBroker({"positions": {"HPQ": 190}, "prices": {"HPQ": 32.95}})
bug = go(artp, ladder(["HPQ"]), bkp2, guards=at.Guards(group_caps=False))
check("bug reintroduced makes a fourth name in Technology Hardware",
      len(buys(bug)) == 3)
fix = go(artp, ladder(["HPQ"]), bkp2)
check("guard admits at most two", len(buys(fix)) == 2)

# (q) a row that does not reconcile with itself — the SMCI shape, price and
# line transposed, which sizes off a NEGATIVE risk if nothing refuses it
print("(q) a row whose price and stop are transposed")
artq = artifact({}, candidates={"SMCI": {"grade": "A+", "group": GROUPS["AAPL"]}})
badframe = frame_of({"SMCI": dict(close=37.88, sma20=39.09, atr14=1.70,
                                  basis_bar=SESSION)})
bug = go(artq, {}, ShadowBroker({"positions": {}, "prices": {}}),
         frame=badframe, guards=at.Guards(row_reconciles=False))
check("bug reintroduced sizes off the unreconciled row",
      len(buys(bug)) == 1 and buys(bug)[0]["risk_usd"] < 0,
      "risk per share comes out negative")
fix = go(artq, {}, ShadowBroker({"positions": {}, "prices": {}}), frame=badframe)
check("guard refuses and logs both readings",
      not buys(fix) and "ROW_UNRECONCILED" in codes(fix)
      and any("reading A" in r["detail"] and "reading B" in r["detail"]
              for r in fix["refusals"]))

# (r) a name the broker already holds
print("(r) a candidate the broker already holds")
artr = artifact({}, candidates={"ANET": {"grade": "A+", "group": GROUPS["ANET"]}})
held = ShadowBroker({"positions": {"ANET": 31}, "prices": {"ANET": 199.39}})
bug = go(artr, {}, held, guards=at.Guards(broker_truth=False))
check("bug reintroduced buys it twice", len(buys(bug)) == 1)
fix = go(artr, {}, held)
check("guard skips it", not buys(fix) and "ALREADY_HELD" in codes(fix))

# (s) watcher-path-only A+
print("(s) an A+ that exists only on the watcher path")
arts = artifact({}, candidates={},
                watchers={"CRWD": dict(state="RE_ENTRY_READY", close=237.65,
                                       grade={"grade": "A+"}, group="Systems Software")})
sframe = frame_of({"CRWD": dict(close=237.65, sma20=215.8555, atr14=13.1194,
                                basis_bar=SESSION)})
fix = go(arts, {}, ShadowBroker({"positions": {}, "prices": {}}), frame=sframe)
check("the watcher path is read", [b["ticker"] for b in buys(fix)] == ["CRWD"],
      "GEN and CRWD were A+ only here; FTNT only on candidates")

# (t) lookahead
print("(t) no lookahead")
poison = dict(FRAME)
poison["ANET"] = dict(poison["ANET"], next_open=999.0)
f1 = frame_of(poison)
r1 = go(artifact({}, candidates=CANDS), {},
        ShadowBroker({"positions": {}, "prices": {}}), frame=f1)
f2 = frame_of(FRAME)
r2 = go(artifact({}, candidates=CANDS), {},
        ShadowBroker({"positions": {}, "prices": {}}), frame=f2)
check("picks are identical with the next open withheld",
      [b["ticker"] for b in buys(r1)] == [b["ticker"] for b in buys(r2)])

# (u) room from the broker, not the artifact
print("(u) room comes from broker positions")
artu = artifact({"HPQ": HPQ_ROW}, candidates={"ANET": {"grade": "A+", "group": GROUPS["ANET"]}})
thin = ShadowBroker({"positions": {"HPQ": 1400}, "prices": {"HPQ": 32.95}})
fat = ShadowBroker({"positions": {"HPQ": 190}, "prices": {"HPQ": 32.95}})
rthin = go(artu, ladder(["HPQ"]), thin)
rfat = go(artu, ladder(["HPQ"]), fat)
check("the same artifact gives different room for different broker books",
      not buys(rthin) and len(buys(rfat)) == 1,
      "the artifact said 40.2% deployed against a real 8.95% on 2026-09-18")

# (v) no secret in any output
print("(v) a fixture secret appears nowhere")
SECRET = "sk-fixture-9f3a-DO-NOT-LOG"
artv = artifact({"DVN": dict(DVN_ROW, note=f"token {SECRET}")})
rv = go(artv, led2, ShadowBroker(bkp), secrets=[SECRET])
blob = json.dumps(rv) + at.slack_text(rv)
with tempfile.TemporaryDirectory() as d:
    p = os.path.join(d, "runs.jsonl")
    at.append_log(p, rv)
    blob += open(p).read()
try:
    raise at.Refusal("X", f"detail carrying {SECRET}")
except at.Refusal as e:
    blob += at.redact(str(e), [SECRET])
check("secret absent from record, Slack text, log file and exception",
      SECRET not in blob)

print()
if FAILS:
    print(f"{len(FAILS)} PIN(S) FAILED: {FAILS}")
    raise SystemExit(1)
print("All automation pins passed (each demonstrated against its bug).")
