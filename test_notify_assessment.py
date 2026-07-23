#!/usr/bin/env python3
"""
Tests for the post-close Slack push (PER-508 item 22).

Pins: message formatting (normal day, EXIT_FIRED day, no-changes day,
regime-shift header), the once-per-trading-day marker gating, the
absent-secret clean exit, and the never-echo-the-webhook /
never-fail-the-pipeline failure mode.

Run: python3 test_notify_assessment.py
"""

import contextlib
import datetime
import io
import json
import os
import shutil
import sys
import tempfile

sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))

import notify_assessment as notify
import ticker_api


def _et(iso_dt):
    from zoneinfo import ZoneInfo
    return datetime.datetime.fromisoformat(iso_dt).replace(
        tzinfo=ZoneInfo("America/New_York"))


def _framework_payload(exit_fired=False, shift=True, now_et=None):
    g = {"value": 1.0, "signal": "risk_on", "detail": "d"}
    gen = (now_et or _et(NOW)).astimezone(datetime.timezone.utc)
    return {
        # stamped from the INJECTED clock: real UTC may already be the next
        # date late in the ET evening, and the freshness guard would
        # (correctly) skip — which is a different test
        "generated_at": gen.isoformat(),
        "schema": "regime-b-chassis",
        "regime": {
            "regime": "Risk-on / Choppy", "action": "A+ only.",
            # chassis-era artifact (D-008): serve guard requires engine match
            "engine": "chassis",
            "chassis": {"engine": "chassis",
                        "raw_state": "In-Trend-Throttled",
                        "confirmed_state": "In-Trend-Throttled",
                        "regime": "Risk-on / Choppy",
                        "exposure_ceiling_pct": 50.0, "trend_in": True,
                        "throttles": {"vix": {"firing": True},
                                      "hy": {"firing": False},
                                      "breadth": {"firing": False}},
                        "throttles_firing": 1,
                        "hysteresis": {"up": 0, "down": 0, "n": 2,
                                       "mode": "asymmetric"},
                        "degraded": False, "degraded_reason": None},
            "risk_on_count": 2, "caution_count": 1, "risk_off_count": 0,
            "gauges": {
                "vix_5d_avg": {"value": 16.1, "spot": 15.93,
                               "signal": "risk_on", "detail": "d"},
                "hy_spread": dict(g),
                "breadth": {"value": -0.2, "signal": "caution", "detail": "d"},
            },
            "backdrop_gate": {"open": True, "capped": False},
            "macro_inputs": {"yield_curve": {"value": 0.8}},
        },
        "themes": {"active_themes": [], "ranked_themes": []},
        "position_signals": {
            "tickers": {
                "ARWR": {
                    "ticker": "ARWR", "kind": "holding", "theme": "Biotech",
                    "state": "EXIT_FIRED" if exit_fired else "HELD",
                    "close": 84.0, "sma20": 80.29,
                    "extension_pct": 4.62, "extension_atr": 0.95,
                    "stop": {"type": "sma20_close", "level": 80.29},
                    "conditions": {"2_confirmation": {"atr14": 3.9}},
                    "next_earnings_date": None, "days_to_earnings": None,
                },
                "MRNA": {
                    "ticker": "MRNA", "kind": "watching", "theme": "Biotech",
                    "state": "EXTENDED_HOLD", "close": 76.56, "sma20": 64.53,
                    "extension_pct": 18.65, "extension_atr": 2.03,
                    "extension_guard": "extension 2.03×ATR > 1.8× — re-entry suppressed",
                    "conditions": {}, "next_earnings_date": None,
                    "days_to_earnings": None,
                },
            },
            "transitions": [],
        },
        "_shift": shift,
    }


class _Env:
    """Sandboxed artifact tree + patched ticker_api/notify globals."""

    def __init__(self, changes=True, shift=True):
        self.tmp = tempfile.mkdtemp(prefix="notify_")
        pub = os.path.join(self.tmp, "public")
        dat = os.path.join(self.tmp, "data")
        st = os.path.join(self.tmp, "state")
        for d in (pub, dat, st):
            os.makedirs(d)
        self.olds = (ticker_api.PUBLIC_DIR, ticker_api.DATA_DIR,
                     ticker_api.FRAMEWORK_STATE_DIR, notify.MARKER_PATH,
                     notify.FRAMEWORK_JSON, notify._now_et,
                     dict(os.environ))
        ticker_api.PUBLIC_DIR = pub
        ticker_api.DATA_DIR = dat
        ticker_api.FRAMEWORK_STATE_DIR = st
        notify.MARKER_PATH = os.path.join(dat, "last_notified")
        notify.FRAMEWORK_JSON = os.path.join(pub, "framework.json")
        os.environ.pop("DASHBOARD_URL", None)

        v = lambda s: {"signal": s, "value": 1, "detail": "d"}
        hist = [
            {"date": "2026-07-08",
             "regime": "Risk-on / Trending" if shift else "Risk-on / Choppy",
             "gauges": {"vix_5d_avg": v("risk_on"), "hy_spread": v("risk_on"),
                        "breadth": v("risk_on" if shift else "caution")}},
            {"date": "2026-07-09", "regime": "Risk-on / Choppy",
             "gauges": {"vix_5d_avg": v("risk_on"), "hy_spread": v("risk_on"),
                        "breadth": v("caution")}},
        ]
        with open(os.path.join(st, "regime_history.json"), "w") as f:
            json.dump(hist, f)
        events = {"changes": ([{
            "timestamp": "2026-07-09T20:01:00+00:00", "ticker": "IWM",
            "type": "position_state_change",
            "detail": {"from_state": "WATCHING", "to_state": "RE_ENTRY_ARMING"},
        }] if changes else [])}
        with open(os.path.join(dat, "position_events.json"), "w") as f:
            json.dump(events, f)
        with open(os.path.join(pub, "signals.json"), "w") as f:
            json.dump({"indexes": {
                "^VIX": {"level": 15.93, "avg_5d": 16.1},
                "^VIX3M": {"level": 18.99, "avg_5d": 19.1},
            }, "groups": []}, f)

    def write_framework(self, payload):
        with open(notify.FRAMEWORK_JSON, "w") as f:
            json.dump(payload, f)

    def close(self):
        (ticker_api.PUBLIC_DIR, ticker_api.DATA_DIR,
         ticker_api.FRAMEWORK_STATE_DIR, notify.MARKER_PATH,
         notify.FRAMEWORK_JSON, notify._now_et, env) = self.olds
        os.environ.clear()
        os.environ.update(env)
        shutil.rmtree(self.tmp, ignore_errors=True)


NOW = "2026-07-09T16:15:00"


def test_message_formatting_normal_day():
    env = _Env()
    try:
        msg = notify.build_message(_framework_payload(), _et(NOW))
        assert "*SignalAgent Daily Assessment* — 2026-07-09 04:15 PM ET" in msg
        # one-grammar retirement (Phase 3): the regime line carries NO
        # parliament vote counts — the chassis sets the regime (D-008)
        assert "Regime: *Risk-on / Trending → Risk-on / Choppy*" in msg
        assert "(2/1/0)" not in msg and "/0)" not in msg
        assert "• ARWR *HELD* — $84.0 vs stop $80.29 · cushion 0.95×ATR" in msg
        assert ("• MRNA *EXTENDED_HOLD* — ext +18.65% (2.03×ATR) — "
                "extension 2.03×ATR > 1.8× — re-entry suppressed") in msg
        assert "Changes: IWM WATCHING→RE_ENTRY_ARMING" in msg
        assert "breadth risk_on→caution" in msg
        assert "VIX 15.93 · term structure: contango" in msg
        assert "🔴" not in msg
        assert "framework.html" not in msg          # no DASHBOARD_URL set
        os.environ["DASHBOARD_URL"] = "https://example.test"
        msg2 = notify.build_message(_framework_payload(), _et(NOW))
        assert "<https://example.test/framework.html|framework.html>" in msg2
        # full page URL (the Actions variable's actual form): no doubled path
        os.environ["DASHBOARD_URL"] = "https://example.test/framework.html"
        msg3 = notify.build_message(_framework_payload(), _et(NOW))
        assert "<https://example.test/framework.html|framework.html>" in msg3
        assert "framework.html/framework.html" not in msg3
    finally:
        env.close()
    print("  formatting: header/shift, holding+cushion, guard, changes, VIX: OK")


def test_message_exit_fired_leads_with_alarm():
    env = _Env()
    try:
        msg = notify.build_message(_framework_payload(exit_fired=True), _et(NOW))
        line = next(l for l in msg.splitlines() if "ARWR" in l)
        assert line.startswith("🔴 "), line
        assert "*EXIT_FIRED*" in line and "vs stop $80.29" in line
    finally:
        env.close()
    print("  formatting: EXIT_FIRED line leads with the alarm: OK")


def test_message_no_changes_day():
    env = _Env(changes=False, shift=False)
    try:
        msg = notify.build_message(_framework_payload(shift=False), _et(NOW))
        assert "Changes: no changes since prior run" in msg
        assert "→" not in msg.splitlines()[1]        # regime line: no arrow
    finally:
        env.close()
    print("  formatting: no-changes day renders cleanly: OK")


def test_message_candidates_line():
    """D-017 close-report line: era-aware (no block -> no line), A+ names
    spelled out, B/C counted, ungradeable counted never dropped."""
    env = _Env()
    try:
        def _annotate_signals(block):
            p = os.path.join(ticker_api.PUBLIC_DIR, "signals.json")
            with open(p) as f:
                sig = json.load(f)
            if block is None:
                sig.pop("candidate_grades", None)
            else:
                sig["candidate_grades"] = block
            with open(p, "w") as f:
                json.dump(sig, f)

        pay = _framework_payload()
        msg = notify.build_message(pay, _et(NOW))
        assert "Candidates:" not in msg      # pre-emission artifact: no line
        block = {
            "HPQ": {"grade": "A+", "reasons": "", "group": "Tech Hardware"},
            "AAA": {"grade": "A+", "reasons": "", "group": "Tech Hardware"},
            "BBB": {"grade": "B", "reasons": "RSI unavailable", "group": "T"},
            "CC1": {"grade": "C", "reasons": "conditions", "group": "T"},
            "CC2": {"grade": "C", "reasons": "conditions", "group": "T"},
            "NUL": {"grade": None, "reasons": "inputs unavailable",
                    "group": "T"},
        }
        pay["candidate_grades"] = block
        # FRESHNESS GUARD: framework carries the block but signals.json
        # is un-annotated (engine rewrote after a failed framework run)
        # -> no line — stale grades never presented as the close verdict
        msg = notify.build_message(pay, _et(NOW))
        assert "Candidates:" not in msg
        _annotate_signals(block)
        msg = notify.build_message(pay, _et(NOW))
        assert "Candidates: 2 A+ (AAA, HPQ) · 1 B · 2 C · 1 ungraded" in msg
        pay["candidate_grades"] = {"BBB": {"grade": "B"},
                                   "CC1": {"grade": "C"}}
        _annotate_signals(pay["candidate_grades"])
        msg = notify.build_message(pay, _et(NOW))
        assert "Candidates: 0 A+ · 1 B · 1 C" in msg   # zero-A+: no parens
        assert "0 A+ (" not in msg
        pay["candidate_grades"] = {}
        msg = notify.build_message(pay, _et(NOW))
        assert "Candidates:" not in msg      # empty block: nothing to say
    finally:
        env.close()
    print("  formatting: D-017 candidates line (era-aware, freshness-"
          "guarded, A+ named, ungraded counted): OK")


def test_marker_gating_once_per_trading_day():
    env = _Env()
    try:
        # pre-close
        ok, why = notify.should_notify(_et("2026-07-09T15:59:00"))
        assert not ok and "pre-close" in why
        # the settling window (review finding): a 16:0x run still serves
        # YESTERDAY's close-basis regime (chassis confirms from 16:10) —
        # posting there would burn the once-per-day marker on it
        ok, why = notify.should_notify(_et("2026-07-09T16:05:00"))
        assert not ok and "pre-close" in why
        # post-close, first run (16:15 — past the settle boundary)
        ok, _ = notify.should_notify(_et(NOW))
        assert ok
        # marker written -> same day suppressed (throttled reruns)
        notify.write_marker(_et(NOW))
        ok, why = notify.should_notify(_et("2026-07-09T17:30:00"))
        assert not ok and "already notified" in why
        # next trading day -> fires again
        ok, _ = notify.should_notify(_et("2026-07-10T16:20:00"))
        assert ok
        # weekend
        ok, why = notify.should_notify(_et("2026-07-11T16:20:00"))
        assert not ok and "weekend" in why
    finally:
        env.close()
    print("  gating: pre-close / once-per-day marker / weekend: OK")


def test_absent_secret_exits_cleanly():
    env = _Env()
    try:
        os.environ.pop("ASSESSMENT_WEBHOOK_URL", None)
        posted = []
        old = notify.post_to_slack
        notify.post_to_slack = lambda *a: posted.append(a)
        out = io.StringIO()
        with contextlib.redirect_stdout(out):
            rc = notify.main()
        notify.post_to_slack = old
        assert rc == 0
        assert "no webhook configured" in out.getvalue()
        assert posted == []
    finally:
        env.close()
    print("  absent secret: clean exit, nothing posted: OK")


def test_webhook_failure_never_fails_never_leaks():
    env = _Env()
    try:
        secret = "https://hooks.slack.com/services/T000/B000/SECRETSECRET"
        os.environ["ASSESSMENT_WEBHOOK_URL"] = secret
        notify._now_et = lambda: _et(NOW)
        env.write_framework(_framework_payload())

        old = notify.post_to_slack

        def boom(url, text):
            raise RuntimeError(f"connection refused for url: {url}")
        notify.post_to_slack = boom
        out = io.StringIO()
        with contextlib.redirect_stdout(out):
            rc = notify.main()
        notify.post_to_slack = old
        assert rc == 0, "a Slack outage must never fail the pipeline"
        logged = out.getvalue()
        assert secret not in logged, "webhook URL leaked into logs"
        assert "<webhook>" in logged
        assert not os.path.exists(notify.MARKER_PATH), \
            "marker must not be written on failure (next run retries)"

        # and the success path writes the marker
        sent = []
        notify.post_to_slack = lambda url, text: sent.append(text)
        with contextlib.redirect_stdout(io.StringIO()):
            rc = notify.main()
        notify.post_to_slack = old
        assert rc == 0 and len(sent) == 1
        assert open(notify.MARKER_PATH).read() == "2026-07-09"
    finally:
        env.close()
    print("  failure mode: rc=0, URL scrubbed, marker only on success: OK")


def test_stale_artifact_skipped():
    env = _Env()
    try:
        os.environ["ASSESSMENT_WEBHOOK_URL"] = "https://hooks.example/x"
        notify._now_et = lambda: _et(NOW)
        payload = _framework_payload()
        payload["generated_at"] = "2026-07-08T20:00:00+00:00"   # yesterday
        env.write_framework(payload)
        posted = []
        old = notify.post_to_slack
        notify.post_to_slack = lambda *a: posted.append(a)
        out = io.StringIO()
        with contextlib.redirect_stdout(out):
            rc = notify.main()
        notify.post_to_slack = old
        assert rc == 0 and posted == []
        assert "not today — skipping" in out.getvalue()
    finally:
        env.close()
    print("  stale artifact (framework failed upstream): skipped, no post: OK")


if __name__ == "__main__":
    print("\n=== Post-close notification tests (PER-508 #22) ===")
    test_message_formatting_normal_day()
    test_message_exit_fired_leads_with_alarm()
    test_message_no_changes_day()
    test_message_candidates_line()
    test_marker_gating_once_per_trading_day()
    test_absent_secret_exits_cleanly()
    test_webhook_failure_never_fails_never_leaks()
    test_stale_artifact_skipped()
    print("\nAll notification tests passed.\n")
