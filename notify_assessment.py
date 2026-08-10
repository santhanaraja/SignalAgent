#!/usr/bin/env python3
"""
Post-close Slack push (PER-508 item 22) — the daily assessment, delivered.

Runs as a CI step after the framework engine and BEFORE the data commit
(the once-per-day marker in data/last_notified must ride the commit —
Actions clones fresh every run). Reads the just-computed artifacts
directly via ticker_api's assessment section helpers — byte-identical
data to /api/assessment.json, no HTTP round-trip.

Gates (computed here, never from the cron slot — GitHub throttles the
schedule, so cron timing is unreliable):
  - ASSESSMENT_WEBHOOK_URL env present (else: clean exit, nothing posted)
  - weekday, ET hour >= 16 (post-close)
  - not already notified today (data/last_notified marker, written only
    after a successful post so a Slack failure retries next run)
  - artifact schema-valid and generated today

Failure mode: NOTHING here may fail the data pipeline — main() never
raises, and the webhook URL is never printed (exception text is
scrubbed; requests' own errors embed the URL, so we never re-raise them
verbatim).
"""

import datetime
import json
import os

REPO = os.path.dirname(os.path.abspath(__file__))
MARKER_PATH = os.path.join(REPO, "data", "last_notified")
FRAMEWORK_JSON = os.path.join(REPO, "public", "framework.json")


def _now_et():
    """ET clock — module-level so tests can inject times."""
    from zoneinfo import ZoneInfo
    return datetime.datetime.now(ZoneInfo("America/New_York"))


def should_notify(now_et, marker_path=None):
    """(ok, reason) — post-close, weekday, once per trading day."""
    marker_path = marker_path or MARKER_PATH
    if now_et.weekday() >= 5:
        return False, f"weekend ({now_et.strftime('%A')}) — skipping"
    # 16:15, not 16:00 (review finding): the chassis confirms today's close
    # from 16:10 ET, and the cron grid has a slot AT the bell — a 16:0x
    # post would burn the once-per-day marker on an artifact still serving
    # YESTERDAY's close-basis regime, suppressing the real close report.
    if now_et.time() < datetime.time(16, 15):
        return False, f"pre-close ({now_et.strftime('%H:%M')} ET) — skipping"
    if os.path.exists(marker_path):
        try:
            with open(marker_path, "r") as f:
                if f.read().strip() == now_et.date().isoformat():
                    return False, "already notified today — skipping"
        except IOError:
            pass
    return True, "post-close window, not yet notified today"


def write_marker(now_et, marker_path=None):
    marker_path = marker_path or MARKER_PATH
    os.makedirs(os.path.dirname(marker_path), exist_ok=True)
    with open(marker_path, "w") as f:
        f.write(now_et.date().isoformat())


def position_risk(holdings, data):
    """Per-position and book risk from the fill facts the framework now
    emits (entry_price / shares / entry_stop).

    Conventions, fixed here so every caller shares them:
      initial risk = shares x (entry - entry_stop)   [FIXED at entry]
      risk to stop = shares x (entry - live stop)    [MOVES; negative
                     once the stop clears entry = locked profit]
      heat         = sum of POSITIVE risk-to-stop only. Netting a
                     locked-profit position against a live one would
                     understate what is actually at risk.
      book R       = sum(open P&L) / sum(initial risk) — DOLLAR-WEIGHTED
                     (5B's tiny-R ruling: a mean of per-position
                     multiples lets a near-zero denominator dominate).
    A holding missing any input is EXCLUDED and named, never counted
    as zero risk.
    """
    rows, missing = {}, []
    initial = heat = locked = pnl = 0.0
    locked_n = 0
    for t, x in (holdings or {}).items():
        entry = x.get("entry_price")
        shares = x.get("shares")
        e_stop = x.get("entry_stop")
        close = x.get("close")
        stop = (x.get("stop") or {}).get("level")
        if None in (entry, shares, e_stop) or entry <= e_stop:
            missing.append(t)
            continue
        init = shares * (entry - e_stop)
        to_stop = shares * (entry - stop) if stop is not None else init
        open_pnl = shares * (close - entry) if close is not None else 0.0
        rows[t] = {"initial_usd": init, "to_stop_usd": to_stop,
                   "open_pnl_usd": open_pnl, "open_r": open_pnl / init}
        initial += init
        pnl += open_pnl
        if to_stop > 0:
            heat += to_stop
        else:
            locked += -to_stop
            locked_n += 1
    cap = ((data or {}).get("r28") or {}).get("capital_usd")
    return {"rows": rows, "missing": missing, "initial_usd": initial,
            "heat_usd": heat, "locked_usd": locked, "locked_n": locked_n,
            "capital_usd": cap,
            "heat_pct": (heat / cap * 100.0) if cap else None,
            "book_r": (pnl / initial) if initial else 0.0}


def build_message(data, now_et):
    """
    Compact Slack mrkdwn from the framework artifact (+ sibling files via
    ticker_api's assessment helpers — the same sections the endpoint
    serves). Pure formatting; no network.
    """
    import ticker_api as api
    regime = api._assessment_regime(data)
    positions = api._assessment_positions(data)
    technicals = api._assessment_technicals(data)
    vol = api._assessment_vol(data)
    try:
        changes = api._assessment_changes(data)
    except Exception:
        changes = {}

    lines = []
    shift = (changes or {}).get("regime_shift")
    reg_txt = (f"{shift['from']} → {shift['to']}" if shift
               else regime.get("regime", "?"))
    # One-grammar retirement (Phase 3): the "(N/N/N)" parliament vote
    # counts are gone — the chassis sets the regime (D-008), and two
    # readers were misled by counts that neither explain nor set it.
    lines.append(f"*SignalAgent Daily Assessment* — "
                 f"{now_et.strftime('%Y-%m-%d %I:%M %p')} ET")
    lines.append(f"Regime: *{reg_txt}*")

    holdings = {t: x for t, x in positions.items()
                if isinstance(x, dict) and x.get("kind") == "holding"}
    watchers = {t: x for t, x in positions.items()
                if isinstance(x, dict) and x.get("kind") == "watching"}

    risk = position_risk(holdings, data)

    lines.append("*Holdings*")
    if not holdings:
        lines.append("• none — cash")
    for t, x in holdings.items():
        state = x.get("state", "?")
        flag = "🔴 " if state == "EXIT_FIRED" else ""   # the action trigger
        stop = (x.get("stop") or {}).get("level")
        cu = ((technicals.get(t) or {}).get("cushion_to_stop") or {})
        cu_txt = (f" · cushion {cu['atr_multiple']}×ATR"
                  if cu.get("atr_multiple") is not None else "")
        stop_txt = (f"${x.get('close')} vs stop ${stop}" if stop is not None
                    else f"${x.get('close')} vs SMA20 ${x.get('sma20')}")
        lines.append(f"{flag}• {t} *{state}* — {stop_txt}{cu_txt}")
        r = (risk["rows"] or {}).get(t)
        if r is None:
            lines.append("    ↳ risk unknown — no entry_stop/shares on "
                         "the row (not zero: unmeasured)")
            continue
        if r["to_stop_usd"] < 0:
            at_risk = f"locked profit ${abs(r['to_stop_usd']):,.2f}"
        else:
            at_risk = f"at risk ${r['to_stop_usd']:,.2f}"
        lines.append(
            f"    ↳ initial risk ${r['initial_usd']:,.2f} · {at_risk} "
            f"· open {r['open_r']:+.2f}R")

    if risk["rows"]:
        cap_txt = (f" = {risk['heat_pct']:.2f}% of "
                   f"${risk['capital_usd']:,.0f}"
                   if risk["capital_usd"] else "")
        lines.append(
            f"*Risk* — heat ${risk['heat_usd']:,.2f}{cap_txt} "
            f"(sum of positions still at risk to their stops)")
        extra = [f"initial risk at entry ${risk['initial_usd']:,.2f}"]
        if risk["locked_usd"]:
            extra.append(f"locked profit ${risk['locked_usd']:,.2f} "
                         f"({risk['locked_n']} stop"
                         f"{'s' if risk['locked_n'] != 1 else ''} above "
                         f"entry)")
        extra.append(f"book {risk['book_r']:+.2f}R")
        lines.append("    ↳ " + " · ".join(extra))
        lines.append("    ↳ R = open P&L ÷ INITIAL risk (fixed at entry, "
                     "never re-based); risk-to-stop is the MOVING one. "
                     "Book R is dollar-weighted sum(P&L)/sum(risk), not a "
                     "mean of per-position R.")
        if risk["missing"]:
            lines.append(f"    ↳ ⚠️ {len(risk['missing'])} holding(s) "
                         f"excluded from every total — no entry data: "
                         f"{', '.join(sorted(risk['missing']))}")

    lines.append("*Watchlist*")
    if not watchers:
        lines.append("• none")
    for t, x in watchers.items():
        ext = (f"{'+' if (x.get('extension_pct') or 0) > 0 else ''}"
               f"{x.get('extension_pct')}% ({x.get('extension_atr')}×ATR)")
        guard = f" — {x['extension_guard']}" if x.get("extension_guard") else ""
        # D-011 grade chip ("MRNA WATCHING [C]"); gate note when Choppy
        # blocks a READY without A+
        g = (x.get("grade") or {}).get("grade")
        g_txt = f" [{g}]" if g else ""
        gate = " ⛔ grade-blocked" if x.get("grade_gate") else ""
        lines.append(f"• {t} *{x.get('state', '?')}*{g_txt} — ext {ext}"
                     f"{guard}{gate}")

    # D-017 close-report line: A+ names spelled out (rare by construction
    # — when the line names one, it earns eyes), B/C stay counts. Absent
    # block (pre-emission artifact) -> no line (era-aware); ungradeable
    # rows are counted, never silently dropped. FRESHNESS GUARD (review
    # finding): the line renders only while the served signals.json still
    # carries the runner's annotation — an engine rewrite after a failed
    # framework run strips it, so stale grades can never be presented as
    # the close verdict (the page chips vanish on the same token).
    cand = data.get("candidate_grades")
    sig_annotated = False
    try:
        with open(os.path.join(api.PUBLIC_DIR, "signals.json")) as f:
            sig_annotated = bool((json.load(f) or {}).get("candidate_grades"))
    except Exception:
        sig_annotated = False
    if isinstance(cand, dict) and cand and sig_annotated:
        graded = {t: c for t, c in cand.items()
                  if isinstance(c, dict) and c.get("grade")}
        ap = sorted(t for t, c in graded.items() if c["grade"] == "A+")
        nb = sum(1 for c in graded.values() if c["grade"] == "B")
        nc = sum(1 for c in graded.values() if c["grade"] == "C")
        nu = len(cand) - len(graded)
        ap_txt = f"{len(ap)} A+ ({', '.join(ap)})" if ap else "0 A+"
        lines.append(f"Candidates: {ap_txt} · {nb} B · {nc} C"
                     + (f" · {nu} ungraded" if nu else ""))

    # D-019: if any SELECTED group's breaker could not be fully checked
    # today, the report says so ONCE, naming the groups. A degraded
    # breaker is not a clear one, and a close report that stayed silent
    # would let the reader assume every group was verified. Era-aware:
    # artifacts with no coverage fields produce no line, never a claim.
    try:
        with open(os.path.join(api.PUBLIC_DIR, "signals.json")) as f:
            _sig = json.load(f) or {}
        _deg = []
        for _g in (_sig.get("groups") or []):
            if not isinstance(_g, dict):
                continue
            _exp = _g.get("breaker_checks_expected")
            _ran = _g.get("breaker_checks_run")
            _missing = ([c for c in _exp if c not in set(_ran)]
                        if isinstance(_exp, list) and isinstance(_ran, list)
                        else [])
            # both flavours: a check that never ran, AND a check that ran
            # on a degraded input (expected==run but not certified)
            _reasons = _g.get("breaker_degraded_reasons") or []
            if (_g.get("breaker_status") == "degraded" or _missing
                    or _reasons):
                _deg.append((_g.get("name") or "?", _missing))
        if _deg:
            _names = ", ".join(n for n, _ in _deg[:4])
            _more = f" +{len(_deg) - 4}" if len(_deg) > 4 else ""
            _checks = sorted({c for _, m in _deg for c in m})
            lines.append(
                f"⚠ Breaker coverage INCOMPLETE for {len(_deg)} group(s): "
                f"{_names}{_more}"
                + (f" — checks not run: {', '.join(_checks[:3])}"
                   if _checks else "")
                + ". Not clear — unverified.")
    except Exception as _e:
        # AN OUTAGE NEVER IMPERSONATES SAFETY — including here. Silently
        # passing would delete the coverage warning and leave a report
        # that reads all-clear, which is the exact defect this build
        # exists to kill. Say that the check itself could not run.
        lines.append(f"⚠ Breaker coverage could not be checked "
                     f"({type(_e).__name__}) — treat group breakers as "
                     f"unverified for this report.")

    # D-018 pin (c): a caught-up or revised exit must read AS SUCH in the
    # close report, naming its bar — never mistaken for a plain
    # today-at-the-close transition.
    def _pos_bit(tr):
        b = f"{tr['ticker']} {tr.get('from_state') or '—'}→{tr['to_state']}"
        if tr.get("revised"):
            b += f" (⟳ {tr.get('bar_date')} close revised)"
        elif tr.get("caught_up"):
            b += f" ({tr.get('bar_date')} close, caught up)"
        return b
    bits = [_pos_bit(tr)
            for tr in (changes or {}).get("position_transitions", [])
            if isinstance(tr, dict) and tr.get("ticker")]
    bits += [f"{v['gauge']} {v['from']}→{v['to']}"
             for v in (changes or {}).get("voters_flipped", [])]
    lines.append("Changes: " + ("; ".join(bits) if bits
                                else "no changes since prior run"))

    vix = (vol or {}).get("vix") or {}
    term = (vol or {}).get("term_structure") or ""
    term_txt = term if term in ("contango", "inverted") else "n/a"
    lines.append(f"VIX {vix.get('spot', '?')} · term structure: {term_txt}")

    # DASHBOARD_URL accepts either the base origin or the full page URL
    dash = os.environ.get("DASHBOARD_URL", "").rstrip("/")
    if dash:
        if not dash.endswith("framework.html"):
            dash = f"{dash}/framework.html"
        lines.append(f"<{dash}|framework.html>")
    return "\n".join(lines)


def post_to_slack(webhook_url, text):
    """POST to the incoming webhook. Raises a SANITIZED error on failure —
    requests' own exceptions embed the URL, which must never reach logs."""
    import requests
    try:
        resp = requests.post(webhook_url, json={"text": text}, timeout=10)
    except Exception as e:
        raise RuntimeError(f"webhook POST failed: {type(e).__name__}") from None
    if resp.status_code >= 300:
        raise RuntimeError(f"webhook returned HTTP {resp.status_code}")


def main():
    webhook = os.environ.get("ASSESSMENT_WEBHOOK_URL", "")
    try:
        if not webhook:
            print("[notify] no webhook configured — skipping")
            return 0
        now = _now_et()
        ok, reason = should_notify(now)
        print(f"[notify] {reason}")
        if not ok:
            return 0

        with open(FRAMEWORK_JSON, "r") as f:
            data = json.load(f)
        import ticker_api as api
        if not api._framework_shape_valid(data):
            print("[notify] framework artifact not schema-valid — skipping")
            return 0
        gen_date = str(data.get("generated_at", ""))[:10]
        if gen_date != now.date().isoformat():
            print(f"[notify] artifact generated {gen_date or 'unknown'}, "
                  f"not today — skipping")
            return 0

        msg = build_message(data, now)
        post_to_slack(webhook, msg)
        write_marker(now)   # only after a successful post: failures retry
        print(f"[notify] posted daily assessment ({len(msg)} chars); "
              f"marker written for {now.date().isoformat()}")
        return 0
    except Exception as e:
        # NEVER fail the data pipeline; NEVER echo the webhook URL.
        text = str(e)
        if webhook:
            text = text.replace(webhook, "<webhook>")
        print(f"[notify] failed (non-fatal): {type(e).__name__}: {text}")
        return 0


if __name__ == "__main__":
    raise SystemExit(main())
