#!/usr/bin/env python3
"""
Tests for the intraday stop-breach alerts (PER-510-B).

Fixture is Friday 2026-07-10's real ARWR state — the case this feature
exists for: stop 80.59 (sma20_close), ATR14 4.0, price gapping below the
stop intraday while the close-based engine (correctly) waited for the
close. Pins: breach/warn tiers with ×ATR depth, the once-per-ticker-per-
tier-per-day matrix (warn→breach escalation fires, everything else
suppresses, next day resets), holdings-only scope, empty-holdings and
market-hours gates, and the items-22/23 shared contract (never-fail,
never-leak, marker-only-on-success, absent-secret clean exit).

Run: python3 test_notify_intraday.py
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

import notify_intraday as ia


def _et(iso_dt):
    from zoneinfo import ZoneInfo
    return datetime.datetime.fromisoformat(iso_dt).replace(
        tzinfo=ZoneInfo("America/New_York"))


NOW = "2026-07-10T13:05:00"     # Friday, mid-session


def _artifact():
    """Friday's ARWR shape (held, stop live) + a watchlist name that must
    never alert."""
    return {
        "generated_at": "2026-07-10T15:27:00+00:00",
        "position_signals": {"tickers": {
            "ARWR": {"ticker": "ARWR", "kind": "holding", "state": "HELD",
                     "close": 79.10, "sma20": 80.59,
                     "stop": {"type": "sma20_close", "level": 80.59},
                     "conditions": {"2_confirmation": {"atr14": 4.0}}},
            "MRNA": {"ticker": "MRNA", "kind": "watching",
                     "state": "RE_ENTRY_READY", "close": 68.27,
                     "sma20": 65.64, "conditions": {}},
        }},
    }


POSITIONS = {"holdings": [{"ticker": "ARWR", "shares": 71,
                           "stop_on_entry": "sma20_close"}],
             "watching": [{"ticker": "MRNA"}]}


def test_breach_message_format():
    tier, line = ia.evaluate_holding(
        "ARWR", _artifact()["position_signals"]["tickers"]["ARWR"],
        (76.68, -3.1), already_sent={})
    assert tier == "breach"
    assert line == ("⚠️ INTRADAY — ARWR $76.68 below stop $80.59 "
                    "(-0.98×ATR breach) · close decides, no pre-emption · "
                    "next check at the close"), line
    # graze vs deep: depth must differentiate
    _, graze = ia.evaluate_holding(
        "ARWR", _artifact()["position_signals"]["tickers"]["ARWR"],
        (80.19, -0.5), already_sent={})
    assert "(-0.10×ATR breach)" in graze
    print("  breach: exact message, ×ATR depth (graze vs deep): OK")


def test_warn_tier_and_boundaries():
    row = _artifact()["position_signals"]["tickers"]["ARWR"]
    # stop 80.59 + 0.25*4.0 = 81.59 band ceiling
    tier, line = ia.evaluate_holding("ARWR", row, (81.00, 0.2), {})
    assert tier == "warn"
    assert line == ("⚠️ INTRADAY — ARWR $81.00 approaching stop $80.59 "
                    "(+0.10×ATR above) · close decides, no pre-emption"), line
    assert ia.evaluate_holding("ARWR", row, (81.59, 0.2), {}) is None  # at ceiling
    assert ia.evaluate_holding("ARWR", row, (84.00, 1.0), {}) is None  # calm
    # exactly AT the stop: not below -> warn tier, not breach
    tier, _ = ia.evaluate_holding("ARWR", row, (80.59, 0.0), {})
    assert tier == "warn"
    # no ATR in artifact: warn tier unavailable, breach still fires (no depth)
    row_no_atr = dict(row, conditions={})
    assert ia.evaluate_holding("ARWR", row_no_atr, (81.00, 0.2), {}) is None
    tier, line = ia.evaluate_holding("ARWR", row_no_atr, (76.68, -3.1), {})
    assert tier == "breach" and "×ATR" not in line
    print("  warn tier: band edges, at-stop, missing-ATR degradation: OK")


def test_once_per_day_matrix():
    row = _artifact()["position_signals"]["tickers"]["ARWR"]
    breach_px, warn_px = (76.68, -3.1), (81.00, 0.2)
    # warn already sent: warn suppressed, breach ESCALATES
    assert ia.evaluate_holding("ARWR", row, warn_px, {"ARWR": "warn"}) is None
    tier, _ = ia.evaluate_holding("ARWR", row, breach_px, {"ARWR": "warn"})
    assert tier == "breach"
    # breach already sent: everything suppressed for the day
    assert ia.evaluate_holding("ARWR", row, breach_px, {"ARWR": "breach"}) is None
    assert ia.evaluate_holding("ARWR", row, warn_px, {"ARWR": "breach"}) is None
    print("  once-per-day matrix: warn→breach escalates, repeats suppressed: OK")


def test_marker_persistence_and_day_reset():
    tmp = tempfile.mkdtemp(prefix="ia_marker_")
    marker = os.path.join(tmp, "last_intraday_alerts")
    try:
        assert ia.load_marker(_et(NOW), marker) == {}
        ia.write_marker(_et(NOW), {"ARWR": "warn"}, marker)
        assert ia.load_marker(_et(NOW), marker) == {"ARWR": "warn"}
        # same file, next trading day: reset
        assert ia.load_marker(_et("2026-07-13T10:00:00"), marker) == {}
        # corrupt marker tolerated
        open(marker, "w").write("not json")
        assert ia.load_marker(_et(NOW), marker) == {}
    finally:
        shutil.rmtree(tmp, ignore_errors=True)
    print("  marker: per-day scoping, day reset, corrupt-file tolerance: OK")


def test_market_hours_gate():
    ok, why = ia.in_market_hours(_et("2026-07-10T09:15:00"))
    assert not ok and "outside market hours" in why
    assert ia.in_market_hours(_et("2026-07-10T09:30:00"))[0]
    assert ia.in_market_hours(_et("2026-07-10T15:59:00"))[0]
    ok, why = ia.in_market_hours(_et("2026-07-10T16:00:00"))
    assert not ok
    ok, why = ia.in_market_hours(_et("2026-07-11T13:00:00"))   # Saturday
    assert not ok and "weekend" in why
    print("  market-hours gate: 9:30/15:59 in, 9:15/16:00/weekend out: OK")


def test_build_alerts_scope():
    quotes = {"ARWR": (76.68, -3.1), "MRNA": (60.00, -8.0)}
    lines, sent = ia.build_alerts(_artifact(), POSITIONS, quotes, {})
    assert len(lines) == 1 and sent == {"ARWR": "breach"}
    assert "MRNA" not in "".join(lines)      # watchlist never alerts (v1)
    # holding with a failed quote: skipped, not crashed
    lines, sent = ia.build_alerts(_artifact(), POSITIONS, {"ARWR": None}, {})
    assert lines == [] and sent == {}
    # NaN last_price = no quote, never "no alert needed" (review finding)
    lines, sent = ia.build_alerts(_artifact(), POSITIONS,
                                  {"ARWR": (float("nan"), -3.1)}, {})
    assert lines == [] and sent == {}
    # duplicate lots as two holdings entries: one alert line, not two
    dup = {"holdings": [{"ticker": "ARWR", "shares": 40},
                        {"ticker": "ARWR", "shares": 31}], "watching": []}
    lines, sent = ia.build_alerts(_artifact(), dup, quotes, {})
    assert len(lines) == 1 and sent == {"ARWR": "breach"}
    print("  scope: holdings only, watchlist silent, failed/NaN quote, "
          "dup lots deduped: OK")


def test_exit_fired_rows_never_realert():
    """A stale artifact (framework step is continue-on-error) can carry an
    EXIT_FIRED row WITH its stop into market hours — that exit already
    signaled at a prior close; re-alerting it as 'close decides' is false
    (review finding)."""
    art = _artifact()
    art["position_signals"]["tickers"]["ARWR"]["state"] = "EXIT_FIRED"
    assert ia.evaluate_holding(
        "ARWR", art["position_signals"]["tickers"]["ARWR"],
        (76.40, -3.5), {}) is None
    print("  EXIT_FIRED stale artifact row: never re-alerts: OK")


def test_main_contract():
    old_env = dict(os.environ)
    old = (ia.post_to_slack, ia._now_et, ia.MARKER_PATH, ia.FRAMEWORK_JSON,
           ia.POSITIONS_JSON)
    old_fetch, old_traded = ia._fetch_quote, ia._traded_today
    tmp = tempfile.mkdtemp(prefix="ia_main_")
    try:
        ia._traded_today = lambda now: True
        ia.MARKER_PATH = os.path.join(tmp, "last_intraday_alerts")
        ia.FRAMEWORK_JSON = os.path.join(tmp, "framework.json")
        ia.POSITIONS_JSON = os.path.join(tmp, "positions.json")
        json.dump(_artifact(), open(ia.FRAMEWORK_JSON, "w"))
        json.dump(POSITIONS, open(ia.POSITIONS_JSON, "w"))
        ia._now_et = lambda: _et(NOW)
        ia._fetch_quote = lambda s: {"ARWR": (76.68, -3.1)}.get(s)

        # absent secret: clean exit, nothing attempted
        os.environ.pop("ASSESSMENT_WEBHOOK_URL", None)
        posted = []
        ia.post_to_slack = lambda *a: posted.append(a)
        out = io.StringIO()
        with contextlib.redirect_stdout(out):
            assert ia.main() == 0
        assert posted == [] and "no webhook configured" in out.getvalue()

        # empty holdings: silent skip, no fetch, no post
        secret = "https://hooks.slack.com/services/T0/B0/INTRADAYSECRET"
        os.environ["ASSESSMENT_WEBHOOK_URL"] = secret
        json.dump({"holdings": [], "watching": [{"ticker": "MRNA"}]},
                  open(ia.POSITIONS_JSON, "w"))
        out = io.StringIO()
        with contextlib.redirect_stdout(out):
            assert ia.main() == 0
        assert posted == [] and "no holdings" in out.getvalue()
        json.dump(POSITIONS, open(ia.POSITIONS_JSON, "w"))

        # webhook failure: rc 0, URL scrubbed, marker NOT written
        def boom(url, text):
            raise RuntimeError(f"refused for url: {url}")
        ia.post_to_slack = boom
        out = io.StringIO()
        with contextlib.redirect_stdout(out):
            assert ia.main() == 0
        assert secret not in out.getvalue() and "<webhook>" in out.getvalue()
        assert not os.path.exists(ia.MARKER_PATH)

        # success: single post, marker written with the tier
        sent_msgs = []
        ia.post_to_slack = lambda url, text: sent_msgs.append(text)
        with contextlib.redirect_stdout(io.StringIO()):
            assert ia.main() == 0
        assert len(sent_msgs) == 1 and "below stop $80.59" in sent_msgs[0]
        marker = json.load(open(ia.MARKER_PATH))
        assert marker == {"date": "2026-07-10", "sent": {"ARWR": "breach"}}

        # second run same day: suppressed by the marker, no second post
        with contextlib.redirect_stdout(io.StringIO()):
            assert ia.main() == 0
        assert len(sent_msgs) == 1

        # all fetches fail: skip cleanly, no post, rc 0
        os.remove(ia.MARKER_PATH)
        ia._fetch_quote = lambda s: None
        out = io.StringIO()
        with contextlib.redirect_stdout(out):
            assert ia.main() == 0
        assert len(sent_msgs) == 1 and "all quote fetches failed" in out.getvalue()

        # market holiday: no SPY session today -> skip before any fetch
        ia._fetch_quote = lambda s: {"ARWR": (76.68, -3.1)}.get(s)
        ia._traded_today = lambda now: False
        out = io.StringIO()
        with contextlib.redirect_stdout(out):
            assert ia.main() == 0
        assert len(sent_msgs) == 1 and "no SPY session today" in out.getvalue()
    finally:
        os.environ.clear()
        os.environ.update(old_env)
        (ia.post_to_slack, ia._now_et, ia.MARKER_PATH, ia.FRAMEWORK_JSON,
         ia.POSITIONS_JSON) = old
        ia._fetch_quote, ia._traded_today = old_fetch, old_traded
        shutil.rmtree(tmp, ignore_errors=True)
    print("  contract: absent secret / empty book / never-fail / never-leak"
          " / marker-on-success / all-fetch-fail: OK")


if __name__ == "__main__":
    print("\n=== Intraday stop-breach alert tests (PER-510-B) ===")
    test_breach_message_format()
    test_warn_tier_and_boundaries()
    test_once_per_day_matrix()
    test_marker_persistence_and_day_reset()
    test_market_hours_gate()
    test_build_alerts_scope()
    test_exit_fired_rows_never_realert()
    test_main_contract()
    print("\nAll intraday alert tests passed.\n")
