#!/usr/bin/env python3
"""
Intraday stop-breach alerts (PER-510-B) — third sibling of the post-close
report (item 22) and pre-market briefing (item 23). Same webhook, same
never-fail contract, own once-per-day marker (data/last_intraday_alerts).

HOLDINGS ONLY (money at risk, not entries — watchlist alerts are out of
scope in v1). Two tiers per ticker per day:
- breach: live price below the SMA20 stop level -> alert with depth in
  xATR (a -0.1x graze reads differently from -0.9x)
- warn: live price within 0.25xATR ABOVE the stop -> softer "approaching"
A breach supersedes/includes a warn: warn->breach escalation still fires,
but nothing repeats within a tier, and after a breach the ticker is done
for the day.

INFORMATION ONLY — nothing is executed. The sma20_close stop rule (R11)
still decides at the close; the message says so. The exit-timing rule
change itself stays gated on Build 5 evidence.

CADENCE REALITY: this rides the throttled hourly update-signals runs
(~5x/day in practice), so detection lag is up to ~an hour. It is an hourly
check, NOT a tick watcher. If that is ever insufficient, faster detection
is a separate infra decision (external pinger / self-hosted runner), not a
tweak here.

Gates (in-script, ET clock — cron slot timing is throttle-unreliable):
weekday, 9:30 <= ET < 16:00, per-ticker-per-tier-per-day marker, webhook
present. Empty holdings -> exits silently (the step costs nothing while
the book is flat).
"""

import datetime
import json
import math
import os

from notify_premarket import _fetch_quote   # same degrade-to-None fetcher

REPO = os.path.dirname(os.path.abspath(__file__))
MARKER_PATH = os.path.join(REPO, "data", "last_intraday_alerts")
FRAMEWORK_JSON = os.path.join(REPO, "public", "framework.json")
POSITIONS_JSON = os.path.join(REPO, "framework", "state", "positions.json")

WARN_BAND_ATR = 0.25   # warn when price sits within this xATR above the stop


def _now_et():
    """ET clock — module-level so tests can inject times."""
    from zoneinfo import ZoneInfo
    return datetime.datetime.now(ZoneInfo("America/New_York"))


def in_market_hours(now_et):
    """(ok, reason) — weekday, 9:30 <= ET < 16:00."""
    if now_et.weekday() >= 5:
        return False, f"weekend ({now_et.strftime('%A')}) — skipping"
    minutes = now_et.hour * 60 + now_et.minute
    if not (570 <= minutes < 960):
        return False, (f"outside market hours "
                       f"({now_et.strftime('%H:%M')} ET) — skipping")
    return True, "market hours"


def _traded_today(now_et):
    """True if SPY printed a daily bar dated today — full NYSE holidays fail
    this and skip the run (a stale last-close quote must not alert on a
    closed market). Fails OPEN on fetch trouble: an API blip must not
    silence a real breach; the per-ticker quote fetch still gates.
    NOTE: 13:00 half-day early closes still pass (SPY has a bar) — a
    breach alert between 13:00 and 16:00 on those ~4 days/yr is stale by
    hours but still truthful vs the stop; documented residual."""
    try:
        import yfinance as yf
        df = yf.Ticker("SPY").history(period="5d")
        if df is None or df.empty:
            return True
        return df.index[-1].date() == now_et.date()
    except Exception:
        return True


def load_marker(now_et, marker_path=None):
    """{ticker: tier} already alerted TODAY; stale dates reset."""
    marker_path = marker_path or MARKER_PATH
    try:
        with open(marker_path, "r") as f:
            data = json.load(f)
        if data.get("date") == now_et.date().isoformat():
            sent = data.get("sent")
            return sent if isinstance(sent, dict) else {}
    except Exception:
        pass
    return {}


def write_marker(now_et, sent, marker_path=None):
    marker_path = marker_path or MARKER_PATH
    os.makedirs(os.path.dirname(marker_path), exist_ok=True)
    with open(marker_path, "w") as f:
        json.dump({"date": now_et.date().isoformat(), "sent": sent}, f,
                  indent=2)


def evaluate_holding(ticker, art_row, quote, already_sent=None):
    """None, or (tier, line) for one holding.

    Stop level and ATR come from the last framework artifact (the engine's
    computed stop — its `close` field is yesterday's close by design, which
    is why the live quote is fetched separately). No active stop in the
    artifact means the exit already signaled at a prior close — the alarm
    for that is the post-close report's job, not intraday's.
    """
    art_row = art_row or {}
    if quote is None:
        return None
    price = quote[0]
    # NaN/garbage last_price must read as no-quote, not as "no alert needed"
    if not isinstance(price, (int, float)) or not math.isfinite(price):
        return None
    # EXIT_FIRED rows still carry their stop, and a stale artifact (the
    # framework step is continue-on-error) can survive into market hours —
    # but that exit already signaled at a prior close and the post-close
    # report owns that alarm. Re-alerting it as "close decides" would be
    # false (review finding).
    if art_row.get("state") == "EXIT_FIRED":
        return None
    stop = (art_row.get("stop") or {}).get("level")
    if stop is None:
        return None
    atr = ((art_row.get("conditions") or {}).get("2_confirmation")
           or {}).get("atr14")

    if price < stop:
        tier = "breach"
        head = f"⚠️ INTRADAY — {ticker} ${price:.2f} below stop ${stop}"
        if atr:
            head += f" ({(price - stop) / atr:.2f}×ATR breach)"
        line = " · ".join([head, "close decides, no pre-emption",
                           "next check at the close"])
    elif atr and price < stop + WARN_BAND_ATR * atr:
        tier = "warn"
        line = " · ".join([
            f"⚠️ INTRADAY — {ticker} ${price:.2f} approaching stop ${stop} "
            f"(+{(price - stop) / atr:.2f}×ATR above)",
            "close decides, no pre-emption"])
    else:
        return None

    already = (already_sent or {}).get(ticker)
    if already == "breach":
        return None                       # breach already covered the day
    if already == "warn" and tier == "warn":
        return None                       # no repeat within the tier
    return tier, line


def build_alerts(data, positions, quotes, already_sent):
    """(lines, sent_tiers) across all holdings — pure logic, no network."""
    art_tickers = ((data.get("position_signals") or {}).get("tickers") or {})
    lines, sent, seen = [], {}, set()
    for h in (positions.get("holdings") or []):
        if not isinstance(h, dict) or not h.get("ticker"):
            continue
        t = h["ticker"]
        if t in seen:            # two lots as two entries: one alert, not two
            continue
        seen.add(t)
        result = evaluate_holding(t, art_tickers.get(t), quotes.get(t),
                                  already_sent)
        if result:
            tier, line = result
            lines.append(line)
            sent[t] = tier
    return lines, sent


def post_to_slack(webhook_url, text):
    """Sanitized errors only — requests' exceptions embed the URL."""
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
            print("[intraday] no webhook configured — skipping")
            return 0
        now = _now_et()
        ok, reason = in_market_hours(now)
        print(f"[intraday] {reason}")
        if not ok:
            return 0
        if not _traded_today(now):
            print("[intraday] no SPY session today (market holiday) — skipping")
            return 0

        try:
            with open(POSITIONS_JSON, "r") as f:
                positions = json.load(f)
        except Exception:
            positions = {}
        holdings = [h["ticker"] for h in (positions.get("holdings") or [])
                    if isinstance(h, dict) and h.get("ticker")]
        if not holdings:
            print("[intraday] no holdings — nothing at risk, skipping")
            return 0

        try:
            with open(FRAMEWORK_JSON, "r") as f:
                data = json.load(f)
        except Exception:
            data = {}

        quotes = {t: _fetch_quote(t) for t in holdings}
        if all(v is None for v in quotes.values()):
            print("[intraday] all quote fetches failed — cannot evaluate, "
                  "skipping (next hourly run retries)")
            return 0

        already = load_marker(now)
        lines, sent = build_alerts(data, positions, quotes, already)
        if not lines:
            print(f"[intraday] {len(holdings)} holding(s) checked — "
                  f"no breach/warn to alert")
            return 0

        post_to_slack(webhook, "\n".join(lines))
        already.update(sent)
        write_marker(now, already)   # only after a successful post
        print(f"[intraday] posted {len(lines)} alert(s): "
              + ", ".join(f"{t}={tier}" for t, tier in sent.items()))
        return 0
    except Exception as e:
        # NEVER fail the workflow; NEVER echo the webhook URL.
        text = str(e)
        if webhook:
            text = text.replace(webhook, "<webhook>")
        print(f"[intraday] failed (non-fatal): {type(e).__name__}: {text}")
        return 0


if __name__ == "__main__":
    raise SystemExit(main())
