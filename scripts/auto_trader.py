#!/usr/bin/env python3
"""AUTOMATED ORDER PLACEMENT — Stage 0 (SHADOW), both sides.

WHAT THIS IS. One evening run that reads the post-splitter bake, asks the
BROKER what is actually held, and produces order intents: one sell per
engine EXIT_FIRED on a confirmed close, and buys for A+ candidates that
fit the room and the caps. In Stage 0 nothing is placed: the intents are
logged and Slacked. Stage 1 places 1-share orders. Stage 2 is full size.
Each stage transition is an operator action; nothing here promotes itself.

THE LAW THIS FILE EXISTS TO ENFORCE (spec section 2): SIGNALS come from
the artifact; HOLDINGS, QUANTITIES, ROOM and OPEN ORDERS come from the
broker. The artifact's holdings list has been wrong for five days at a
time — it carried five sold positions, showed SMCI HELD with a gain four
days after it was sold, and reported 40.2% deployed against a real 8.95%.
Selling SMCI off that list in a margin account would have opened a SHORT.

WHY THE GUARDS ARE FLAGS. Every refusal below is a named flag on Guards.
The pins disable exactly one flag to reintroduce exactly one bug, show the
unguarded run placing the wrong order, then show the guarded run refusing.
A guard that is only asserted has never been demonstrated.
"""
from __future__ import annotations

import argparse
import dataclasses
import datetime as dt
import json
import os
import urllib.request
from typing import Any, Callable

ET = dt.timezone(dt.timedelta(hours=-4))          # EDT; see SPLITTER_UTC
SPLITTER_UTC = dt.time(20, 10)                     # 16:10 ET — D-008's splitter
RUN_DEADLINE_UTC = dt.time(0, 0)                   # 20:00 ET, next UTC day
MAX_ORDER_PCT = 8.0                                # spec 3.5
MAX_ORDERS_PER_DAY = 8                             # spec 3.5
SLOT_PCT = 6.5                                     # spec 5
EXT_GUARD = 1.8                                    # the 1.8x line
GROUP_MAX_PCT = 20.0
GROUP_MAX_NAMES = 3
STAGES = ("shadow", "live_1share", "live_full")
RAW = "https://raw.githubusercontent.com/santhanaraja/SignalAgent"


class Refusal(Exception):
    """A refusal is a RESULT, not a crash: it is logged and alerted."""

    def __init__(self, code: str, detail: str):
        super().__init__(f"{code}: {detail}")
        self.code, self.detail = code, detail


@dataclasses.dataclass
class Guards:
    """Every guard, individually disableable so a pin can demonstrate the
    bug it prevents. Production runs use the defaults; nothing reads these
    from config, so a typo in a config file cannot switch one off."""
    bake_priced: bool = True
    bake_fresh: bool = True
    bake_post_splitter: bool = True
    ladder_basis_matches: bool = True
    broker_truth: bool = True
    idempotent_local: bool = True
    idempotent_broker: bool = True
    kill_switch: bool = True
    order_size_cap: bool = True
    order_count_cap: bool = True
    ceiling: bool = True
    group_caps: bool = True
    row_reconciles: bool = True
    token_valid: bool = True
    adapter_allowed: bool = True
    stage_one_share: bool = True
    us_listing_only: bool = True
    usd_only: bool = True


@dataclasses.dataclass
class Intent:
    ticker: str
    side: str                 # "buy" | "sell"
    qty: int
    order_type: str           # "market_on_open" | "limit"
    limit: float | None
    trigger_date: str
    reason: str
    dollars: float
    risk_usd: float | None = None
    stop: float | None = None
    basis_bar: str | None = None

    @property
    def key(self) -> str:
        return f"{self.ticker}|{self.trigger_date}|{self.side}"


@dataclasses.dataclass
class Refused:
    ticker: str
    side: str
    code: str
    detail: str


# ---------------------------------------------------------------- sources

def fetch_pinned(sha: str, path: str) -> Any:
    """Both the artifact and the ladder state are read AT THE SAME COMMIT.
    They are not always committed together — d379b42 touched the artifact
    and not the state file — so reading 'latest' twice can pair a bake with
    someone else's state."""
    url = f"{RAW}/{sha}/{path}"
    with urllib.request.urlopen(url, timeout=30) as r:      # nosec - fixed host
        return json.loads(r.read().decode())


def session_date_of(artifact: dict) -> str | None:
    return ((artifact.get("regime") or {}).get("chassis") or {}).get(
        "replay", {}).get("end")


def assert_bake_usable(artifact: dict, ladder_state: dict, *,
                       target_session: str, now_utc: dt.datetime,
                       guards: Guards, max_age_min: int = 180) -> None:
    """(b) PRICED, FRESH, POST-SPLITTER — and on the bar we think it is.

    The failures this has to catch are all real: a holding with a null
    close while R28 still read 'ok / compliant' (three bakes on 2026-08-18,
    GEN unpriced, r28.summary.no_price = 1); a pre-splitter bake carrying
    YESTERDAY's confirmed close (today's 11:58Z bake said 2026-09-18 while
    the 22:08Z bake said 2026-09-21); and bakes before 2026-08-07 that
    carry no replay block at all, where the basis bar cannot be read."""
    gen = artifact.get("generated_at")
    if not gen:
        raise Refusal("BAKE_SHAPE", "artifact has no generated_at")
    gen_dt = dt.datetime.fromisoformat(gen.replace("Z", "+00:00"))

    if guards.bake_post_splitter:
        end = session_date_of(artifact)
        if not end:
            raise Refusal("BAKE_NO_BASIS",
                          "chassis.replay is absent — the basis bar cannot "
                          "be read (bakes before 2026-08-07)")
        if end != target_session:
            raise Refusal("BAKE_WRONG_SESSION",
                          f"basis bar {end} is not the target session "
                          f"{target_session} — a pre-splitter bake carries "
                          f"the PREVIOUS confirmed close")
        if gen_dt.timetz() < SPLITTER_UTC.replace(tzinfo=dt.timezone.utc) \
                and gen_dt.date().isoformat() == target_session:
            raise Refusal("BAKE_PRE_SPLITTER",
                          f"generated_at {gen} precedes the 20:10Z splitter")

    if guards.bake_fresh:
        age = (now_utc - gen_dt).total_seconds() / 60.0
        if age > max_age_min:
            raise Refusal("BAKE_STALE",
                          f"bake is {age:.0f} min old (limit {max_age_min})")
        if age < -5:
            raise Refusal("BAKE_FUTURE", f"bake is dated ahead: {gen}")

    tickers = (artifact.get("position_signals") or {}).get("tickers") or {}
    if guards.bake_priced:
        r28 = artifact.get("r28") or {}
        no_price = (r28.get("summary") or {}).get("no_price")
        if no_price:
            raise Refusal("BAKE_UNPRICED",
                          f"R28 reports {no_price} unpriced position(s) and "
                          f"still says '{(r28.get('ceiling') or {}).get('status')}'")
        unpriced = sorted(t for t, row in tickers.items()
                          if row.get("kind") == "holding"
                          and row.get("close") is None)
        if unpriced:
            raise Refusal("BAKE_UNPRICED", f"holdings without a close: {unpriced}")

    if guards.ladder_basis_matches:
        bad = {t: (row or {}).get("last_close_date")
               for t, row in (ladder_state or {}).items()
               if isinstance(row, dict)
               and row.get("last_close_date") != target_session}
        if bad:
            raise Refusal("LADDER_BASIS_MISMATCH",
                          f"ladder rows not stepped to {target_session}: {bad}")


# ------------------------------------------------------------ kill switch

def read_kill_switch(reader: Callable[[], Any], guards: Guards) -> bool:
    """(f) Reachable from a phone: a one-key JSON file in the repo at
    ops/automation_switch.json, editable from the GitHub app or any browser,
    re-read immediately before every submit.

    FAIL SAFE: the ONLY value that lets orders through is {"kill_switch":
    "off"}. Unreadable, malformed, absent, a typo, a network failure —
    every one of them reads as KILLED. A switch that fails open is not a
    switch. The file ships as "on", so the automation is inert until the
    operator turns it off deliberately."""
    if not guards.kill_switch:
        return False
    try:
        doc = reader()
    except Exception:
        return True
    if not isinstance(doc, dict):
        return True
    return str(doc.get("kill_switch", "")).strip().lower() != "off"


# ----------------------------------------------------------------- exits

def compute_exits(artifact: dict, broker_positions: dict[str, int],
                  target_session: str, guards: Guards,
                  refusals: list[Refused]) -> list[Intent]:
    """(spec 4) One sell per EXIT_FIRED on the confirmed close, quantity
    min(signal, broker-held). The ladder read is the artifact's own state
    field — the same one the exit-drift detector uses (spec 3.11); this
    file does not re-derive HELD/EXIT_FIRED and cannot disagree with it."""
    out: list[Intent] = []
    tickers = (artifact.get("position_signals") or {}).get("tickers") or {}
    for tk, row in sorted(tickers.items()):
        if row.get("kind") != "holding" or row.get("state") != "EXIT_FIRED":
            continue
        signal_qty = int(row.get("shares") or 0)
        held = int(broker_positions.get(tk, 0)) if guards.broker_truth \
            else signal_qty
        qty = min(signal_qty, held) if guards.broker_truth else signal_qty
        if qty <= 0:
            refusals.append(Refused(tk, "sell", "GHOST",
                                    f"EXIT_FIRED but the broker holds "
                                    f"{held}; artifact said {signal_qty}"))
            continue
        if held < signal_qty:
            refusals.append(Refused(tk, "sell", "QTY_CLAMPED",
                                    f"artifact said {signal_qty}, broker "
                                    f"holds {held} — selling {qty}"))
        out.append(Intent(ticker=tk, side="sell", qty=qty,
                          order_type="market_on_open", limit=None,
                          trigger_date=target_session,
                          reason="EXIT_FIRED on the confirmed close",
                          dollars=qty * float(row.get("close") or 0.0),
                          basis_bar=target_session))
    return out


# --------------------------------------------------------------- entries

def a_plus_names(artifact: dict) -> list[tuple[str, str, str]]:
    """(e) BOTH paths. The candidate path publishes grade and group only —
    no score, no SMA20, no ATR14, no line, no runway — while the watcher
    path publishes assess_inputs and grade_inputs. Both are read here; the
    frame supplies what the candidate path does not."""
    out = []
    for tk, row in sorted((artifact.get("candidate_grades") or {}).items()):
        if (row or {}).get("grade") == "A+":
            out.append((tk, row.get("group") or "", "candidate"))
    tickers = (artifact.get("position_signals") or {}).get("tickers") or {}
    for tk, row in sorted(tickers.items()):
        if row.get("kind") != "watching":
            continue
        grade = row.get("grade")
        grade = grade.get("grade") if isinstance(grade, dict) else grade
        if grade == "A+":
            out.append((tk, row.get("group") or "", "watcher"))
    return out


def compute_entries(artifact: dict, frame: Callable[[str], dict],
                    scores: dict[str, float], broker_positions: dict[str, int],
                    broker_value: Callable[[str], float], capital: float,
                    target_session: str, guards: Guards,
                    refusals: list[Refused], selection: str = "risk_first",
                    venue_of: Callable[[str], Any] | None = None
                    ) -> tuple[list[Intent], list[str]]:
    """(spec 5) Every A+ on both paths, minus what the broker holds, sized
    at 6.5%, bought on a limit at the 1.8x line, stopped at the frame SMA20.

    ROOM AND CAPS COME FROM BROKER POSITIONS, never from the artifact."""
    deployed = sum(qty * broker_value(tk)
                   for tk, qty in broker_positions.items() if qty)
    ceiling_pct = float(((artifact.get("r28") or {}).get("ceiling_pct")) or 0.0)
    room = capital * ceiling_pct / 100.0 - deployed
    slot = capital * SLOT_PCT / 100.0

    group_val: dict[str, float] = {}
    group_n: dict[str, int] = {}
    for tk, qty in broker_positions.items():
        if not qty:
            continue
        g = ((artifact.get("position_signals") or {}).get("tickers") or {}
             ).get(tk, {}).get("group") or "(ungrouped)"
        group_val[g] = group_val.get(g, 0.0) + qty * broker_value(tk)
        group_n[g] = group_n.get(g, 0) + 1

    cands = []
    for tk, group, path in a_plus_names(artifact):
        if guards.broker_truth and broker_positions.get(tk):
            refusals.append(Refused(tk, "buy", "ALREADY_HELD",
                                    f"broker holds {broker_positions[tk]}"))
            continue
        # DEFENCE IN DEPTH, INDEPENDENT OF THE POOL FILTER. 3443.TW reached
        # the dashboard because pool construction never asked where a
        # security is listed. If that gate ever regresses, this one still
        # refuses: the automation must be unable to place an order for a
        # foreign listing, and unable to size off a price that is not in
        # dollars. A missing resolver is treated as unknown, and unknown
        # refuses — the same safe direction the pool filter takes.
        if guards.us_listing_only or guards.usd_only:
            import listing_venue as lv
            v = lv.classify(tk, venue_of) if venue_of else None
            if v is None:
                refusals.append(Refused(tk, "buy", "VENUE_UNKNOWN",
                                        "no venue resolver supplied — an "
                                        "unverified listing cannot be bought"))
                continue
            if guards.us_listing_only and not v.us_listed:
                refusals.append(Refused(tk, "buy", "VENUE_NOT_US", v.reason))
                continue
            if guards.usd_only:
                try:
                    lv.assert_usd(tk, v)
                except ValueError as e:
                    refusals.append(Refused(tk, "buy", "CURRENCY_NOT_USD",
                                            str(e)))
                    continue
        f = frame(tk)
        close, sma, atr = f.get("close"), f.get("sma20"), f.get("atr14")
        if None in (close, sma, atr) or atr <= 0 or sma <= 0:
            refusals.append(Refused(tk, "buy", "FRAME_INCOMPLETE",
                                    f"close={close} sma20={sma} atr14={atr}"))
            continue
        line = sma + EXT_GUARD * atr
        if close > line:
            refusals.append(Refused(tk, "buy", "ABOVE_THE_LINE",
                                    f"close {close:.4f} is above the 1.8x "
                                    f"line {line:.4f} — a limit there fills "
                                    f"instantly at a worse price"))
            continue
        # A row must order itself: stop BELOW price BELOW line. The SMCI row
        # that had price and line transposed satisfies none of that, and a
        # mechanical rule cannot choose between the two readings, so it
        # refuses and prints both rather than sizing off a negative risk.
        if guards.row_reconciles and not (0 < sma < close):
            refusals.append(Refused(tk, "buy", "ROW_UNRECONCILED",
                                    f"reading A: price {close:.4f} with stop "
                                    f"{sma:.4f} gives risk "
                                    f"{close - sma:+.4f}/share; reading B: "
                                    f"the stop sits at or above the price, so "
                                    f"the entry is already stopped out — "
                                    f"neither is picked"))
            continue
        if f.get("basis_bar") and f["basis_bar"] != target_session:
            refusals.append(Refused(tk, "buy", "FRAME_WRONG_BAR",
                                    f"frame basis {f['basis_bar']} != "
                                    f"{target_session}"))
            continue
        qty = int(slot // close)
        if qty <= 0:
            refusals.append(Refused(tk, "buy", "SLOT_TOO_SMALL",
                                    f"one share costs {close:.2f} > slot"))
            continue
        cands.append(dict(t=tk, group=group, path=path, close=close, sma=sma,
                          atr=atr, line=line, qty=qty, score=scores.get(tk, 0),
                          risk=qty * (close - sma), dollars=qty * close))

    risk_order = sorted(cands, key=lambda c: (c["risk"], c["t"]))
    score_order = sorted(cands, key=lambda c: (-c["score"], c["t"]))
    chosen_order = risk_order if selection == "risk_first" else score_order
    other_order = score_order if selection == "risk_first" else risk_order

    def admit(seq):
        picked, r, gv, gn = [], room, dict(group_val), dict(group_n)
        for c in seq:
            if guards.ceiling and c["dollars"] > r:
                continue
            if guards.group_caps:
                if gn.get(c["group"], 0) + 1 > GROUP_MAX_NAMES:
                    continue
                if (gv.get(c["group"], 0.0) + c["dollars"]) > \
                        capital * GROUP_MAX_PCT / 100.0:
                    continue
            picked.append(c)
            r -= c["dollars"]
            gv[c["group"]] = gv.get(c["group"], 0.0) + c["dollars"]
            gn[c["group"]] = gn.get(c["group"], 0) + 1
        return picked

    picked = admit(chosen_order)
    shadow_pick = admit(other_order)
    for c in cands:
        if c not in picked:
            refusals.append(Refused(c["t"], "buy", "NO_ROOM_OR_CAP",
                                    f"room ${room:,.2f}, group {c['group']} "
                                    f"at {gn_fmt(group_n, c['group'])}"))
    intents = [Intent(ticker=c["t"], side="buy", qty=c["qty"],
                      order_type="limit", limit=round(c["line"], 2),
                      trigger_date=target_session,
                      reason=f"A+ on the {c['path']} path",
                      dollars=c["dollars"], risk_usd=c["risk"],
                      stop=round(c["sma"], 4), basis_bar=target_session)
               for c in picked]
    return intents, [c["t"] for c in shadow_pick]


def gn_fmt(group_n: dict[str, int], g: str) -> str:
    return f"{group_n.get(g, 0)}/{GROUP_MAX_NAMES} names"


# ------------------------------------------------------------ order gates

def apply_order_gates(intents: list[Intent], *, capital: float, stage: str,
                      placed_keys: set[str], broker_open_keys: set[str],
                      guards: Guards, refusals: list[Refused]) -> list[Intent]:
    """Idempotency twice over, the hard caps, and the Stage 1 one-share
    ramp. The prompt-injection guard once protected the FILE and not the
    INSTRUCTION and fired twice; for orders that is a double trade."""
    out: list[Intent] = []
    for it in intents:
        if guards.idempotent_local and it.key in placed_keys:
            refusals.append(Refused(it.ticker, it.side, "DUPLICATE_LOCAL",
                                    f"{it.key} already placed"))
            continue
        if guards.idempotent_broker and it.key in broker_open_keys:
            refusals.append(Refused(it.ticker, it.side, "DUPLICATE_BROKER",
                                    f"a resting order already covers {it.key}"))
            continue
        if guards.stage_one_share and stage == "live_1share" and it.qty != 1:
            it = dataclasses.replace(it, qty=1,
                                     dollars=(it.dollars / max(it.qty, 1)))
        if guards.order_size_cap and it.dollars > capital * MAX_ORDER_PCT / 100:
            refusals.append(Refused(it.ticker, it.side, "ORDER_TOO_LARGE",
                                    f"${it.dollars:,.2f} exceeds "
                                    f"{MAX_ORDER_PCT}% of ${capital:,.0f}"))
            continue
        if guards.order_count_cap and len(out) >= MAX_ORDERS_PER_DAY:
            refusals.append(Refused(it.ticker, it.side, "ORDER_COUNT_CAP",
                                    f"more than {MAX_ORDERS_PER_DAY} orders "
                                    f"in one day"))
            continue
        out.append(it)
    return out


def assert_adapter_allowed(stage: str, adapter_name: str, guards: Guards):
    if not guards.adapter_allowed:
        return
    if stage in ("live_1share", "live_full") and adapter_name != "schwab":
        raise Refusal("ADAPTER_NOT_ALLOWED",
                      f"stage {stage} with adapter '{adapter_name}'")


def assert_token_fresh(expiry: dt.datetime | None, now: dt.datetime,
                       guards: Guards) -> str | None:
    """Schwab refresh tokens die every seven days and cannot be extended.
    Expired: place nothing, Slack a by-hand list, alert. Within 24h: alert
    and keep running. A job that goes quiet on day eight is worse than no
    job, because the operator would believe it was running."""
    if not guards.token_valid:
        return None
    if expiry is None:
        raise Refusal("TOKEN_UNKNOWN", "no refresh-token expiry recorded")
    if expiry <= now:
        raise Refusal("TOKEN_EXPIRED",
                      f"refresh token expired {expiry.isoformat()} — placing "
                      f"nothing; a by-hand list is Slacked instead")
    if (expiry - now) <= dt.timedelta(hours=24):
        return f"token expires {expiry.isoformat()} (within 24h)"
    return None


def redact(text: str, secrets: list[str]) -> str:
    for s in secrets:
        if s:
            text = text.replace(s, "[REDACTED]")
    return text


# ------------------------------------------------------------------- run

def run(*, artifact: dict, ladder_state: dict, broker, frame, scores: dict,
        capital: float, target_session: str, now_utc: dt.datetime,
        stage: str = "shadow", guards: Guards | None = None,
        kill_reader: Callable[[], Any] | None = None,
        store: dict | None = None, selection: str = "risk_first",
        secrets: list[str] | None = None,
        venue_of: Callable[[str], Any] | None = None) -> dict:
    """One evening run. Returns the record that is logged and Slacked.

    Order of operations is deliberate: the kill switch and the bake gates
    come BEFORE any computation that could place something, and the kill
    switch is re-read immediately before each submit."""
    guards = guards or Guards()
    store = store if store is not None else {"placed_keys": [], "days": {}}
    refusals: list[Refused] = []
    alerts: list[str] = []
    rec: dict[str, Any] = {"ran_at": now_utc.isoformat(), "stage": stage,
                           "session": target_session, "adapter": broker.name,
                           "selection": selection}

    try:
        assert_adapter_allowed(stage, broker.name, guards)
        killed = read_kill_switch(kill_reader or (lambda: {"kill_switch": "on"}),
                                  guards)
        rec["kill_switch"] = "ON" if killed else "off"
        assert_bake_usable(artifact, ladder_state, target_session=target_session,
                           now_utc=now_utc, guards=guards)
        # Stage 0 holds no broker credential at all, so "no expiry" is the
        # correct state there and only a PRESENT expiry is checked. Every
        # live stage must produce one: a live run with no recorded expiry
        # cannot tell a fresh token from a dead one.
        expiry = broker.token_expiry()
        if stage != "shadow" or expiry is not None:
            warn = assert_token_fresh(expiry, now_utc, guards)
            if warn:
                alerts.append(warn)
    except Refusal as e:
        rec.update(status="REFUSED", code=e.code, detail=e.detail,
                   intents=[], placed=[], refusals=[], alerts=[str(e)])
        if e.code == "TOKEN_EXPIRED":
            rec["by_hand"] = "placing nothing; the operator places by hand"
        return _finish(rec, secrets)

    positions = broker.positions()
    sells = compute_exits(artifact, positions, target_session, guards, refusals)
    buys, other_rule = compute_entries(
        artifact, frame, scores, positions, broker.last_price, capital,
        target_session, guards, refusals, selection=selection,
        venue_of=venue_of)
    broker_keys = {o.key for o in broker.open_orders()}
    day_count = len(store.get("days", {}).get(target_session, []))
    gated = apply_order_gates(sells + buys, capital=capital, stage=stage,
                              placed_keys=set(store.get("placed_keys", [])),
                              broker_open_keys=broker_keys, guards=guards,
                              refusals=refusals)
    if day_count:
        gated = gated[:max(0, MAX_ORDERS_PER_DAY - day_count)]

    placed = []
    if killed:
        for o in broker.open_orders():
            broker.cancel(o.key)
        alerts.append("KILL SWITCH ON — nothing placed; automation-placed "
                      "resting orders cancelled")
    else:
        for it in gated:
            if read_kill_switch(kill_reader or (lambda: {"kill_switch": "on"}),
                                guards):
                alerts.append("KILL SWITCH flipped mid-run — stopped")
                break
            if stage == "shadow":
                placed.append({"key": it.key, "shadow_only": True})
                continue
            oid = broker.place(it.ticker, it.side, it.qty, it.order_type,
                               it.limit, target_session)
            placed.append({"key": it.key, "order_id": oid})
            store.setdefault("placed_keys", []).append(it.key)
            store.setdefault("days", {}).setdefault(target_session, []).append(it.key)

    rec.update(status="OK", intents=[dataclasses.asdict(i) for i in gated],
               placed=placed,
               refusals=[dataclasses.asdict(r) for r in refusals],
               alerts=alerts,
               other_rule_would_pick=other_rule)
    return _finish(rec, secrets)


def _finish(rec: dict, secrets: list[str] | None) -> dict:
    """Nothing leaves this process carrying a credential, including inside
    an exception string that got captured into a detail field."""
    blob = redact(json.dumps(rec), secrets or [])
    return json.loads(blob)


def watchdog_alert(last_run_utc: dt.datetime | None, now_utc: dt.datetime,
                   deadline_utc: dt.datetime) -> str | None:
    """(spec 3.6, D-019) Coverage, not outcome. This runs on a DIFFERENT
    host from the job — a GitHub Actions cron reading the run log — because
    a watchdog that shares a machine with the thing it watches goes quiet
    at exactly the moment it is needed. A sleeping Mac produces no run, no
    orders and no alert; this is what notices."""
    if now_utc < deadline_utc:
        return None
    if last_run_utc is None or last_run_utc < deadline_utc - dt.timedelta(hours=12):
        return (f"NO AUTOMATION RUN recorded for the session ending "
                f"{deadline_utc.date().isoformat()} — silence is not 'nothing "
                f"to do'. Check the host and the bake.")
    return None


def append_log(path: str, rec: dict) -> None:
    """Append-only, every run including the empty ones: silence in the log
    is itself the alarm condition the watchdog looks for."""
    with open(path, "a") as f:
        f.write(json.dumps(rec, sort_keys=True) + "\n")


def slack_text(rec: dict) -> str:
    lines = [f"*SignalAgent automation — {rec['stage']} — session "
             f"{rec['session']}*"]
    if rec.get("status") == "REFUSED":
        lines.append(f"RUN REFUSED [{rec['code']}] {rec['detail']}")
        return "\n".join(lines)
    lines.append(f"kill switch: {rec.get('kill_switch')} · adapter "
                 f"{rec['adapter']} · selection {rec['selection']}")
    for i in rec["intents"]:
        lim = f" limit {i['limit']}" if i["limit"] is not None else " at the open"
        risk = f" · risk ${i['risk_usd']:,.2f}" if i.get("risk_usd") else ""
        lines.append(f"{i['side'].upper()} {i['qty']} {i['ticker']}{lim} "
                     f"(${i['dollars']:,.2f}{risk}) — {i['reason']}")
    if not rec["intents"]:
        lines.append("no orders")
    for r in rec["refusals"]:
        lines.append(f"refused {r['side']} {r['ticker']} [{r['code']}] "
                     f"{r['detail']}")
    for a in rec["alerts"]:
        lines.append(f"ALERT {a}")
    if rec.get("other_rule_would_pick"):
        lines.append("the other selection rule would have picked: "
                     + ", ".join(rec["other_rule_would_pick"]))
    lines.append("to cancel: flip the kill switch, or cancel at the broker "
                 "before the open")
    return "\n".join(lines)


def main(argv=None):  # pragma: no cover - wiring, exercised by hand
    p = argparse.ArgumentParser()
    p.add_argument("--sha", required=True, help="commit to read the bake at")
    p.add_argument("--session", required=True)
    p.add_argument("--snapshot", required=True, help="broker snapshot json")
    p.add_argument("--stage", default="shadow", choices=STAGES)
    p.add_argument("--selection", default="risk_first",
                   choices=("risk_first", "score_first"))
    p.add_argument("--log", default=os.path.expanduser(
        "~/SignalAgent-logs/automation/runs.jsonl"))
    a = p.parse_args(argv)
    from broker_adapter import ShadowBroker
    artifact = fetch_pinned(a.sha, "public/framework.json")
    ladder = fetch_pinned(a.sha, "framework/state/position_state.json")
    rec = run(artifact=artifact, ladder_state=ladder,
              broker=ShadowBroker.from_file(a.snapshot),
              frame=lambda t: {}, scores={}, capital=97500.0,
              target_session=a.session, now_utc=dt.datetime.now(dt.timezone.utc),
              stage=a.stage, selection=a.selection)
    os.makedirs(os.path.dirname(a.log), exist_ok=True)
    append_log(a.log, rec)
    print(slack_text(rec))
    return 0


if __name__ == "__main__":  # pragma: no cover
    raise SystemExit(main())
