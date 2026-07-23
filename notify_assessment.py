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

    bits = [f"{tr['ticker']} {tr.get('from_state') or '—'}→{tr['to_state']}"
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
