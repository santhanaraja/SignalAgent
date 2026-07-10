#!/usr/bin/env python3
"""
Pre-market briefing push (PER-508 item 23) — companion to the post-close
report (notify_assessment.py). Same webhook, same never-fail contract,
same once-per-day marker pattern (separate marker: data/last_notified_am).

Unlike item 22, this one FETCHES live quotes — it runs pre-market, when
the committed artifacts are stale by design (they are yesterday's close;
the regime line is labeled accordingly). Pre-market quotes are thin and
flaky: any symbol that fails resolves to "n/a" and never blocks the
message; if ALL fetches fail, the briefing still goes out using
last-close prices with a "live quotes unavailable" note — never silence.

Holdings gap check bakes the discipline into the message: a name trading
below its stop pre-market is flagged 🔴 with "close decides, no
pre-emption" — the system reminds you of R11 before you can pre-empt it.

Gates (in-script, ET clock — cron timing is throttle-unreliable):
weekday, 6:30 <= ET < 9:30, once per day, webhook present.
"""

import datetime
import json
import os

REPO = os.path.dirname(os.path.abspath(__file__))
MARKER_PATH = os.path.join(REPO, "data", "last_notified_am")
FRAMEWORK_JSON = os.path.join(REPO, "public", "framework.json")
POSITIONS_JSON = os.path.join(REPO, "framework", "state", "positions.json")

FUTURES = [("ES=F", "ES"), ("NQ=F", "NQ"), ("RTY=F", "RTY"),
           ("CL=F", "CL"), ("GC=F", "GC")]
VOL = [("^VIX", "VIX"), ("^VIX9D", "9D"), ("^VIX3M", "3M")]


def _now_et():
    """ET clock — module-level so tests can inject times."""
    from zoneinfo import ZoneInfo
    return datetime.datetime.now(ZoneInfo("America/New_York"))


def should_notify(now_et, marker_path=None):
    """(ok, reason) — weekday, pre-market window, once per day."""
    marker_path = marker_path or MARKER_PATH
    if now_et.weekday() >= 5:
        return False, f"weekend ({now_et.strftime('%A')}) — skipping"
    minutes = now_et.hour * 60 + now_et.minute
    if not (390 <= minutes < 570):          # 6:30 <= ET < 9:30
        return False, (f"outside pre-market window "
                       f"({now_et.strftime('%H:%M')} ET) — skipping")
    if os.path.exists(marker_path):
        try:
            with open(marker_path, "r") as f:
                if f.read().strip() == now_et.date().isoformat():
                    return False, "already briefed today — skipping"
        except IOError:
            pass
    return True, "pre-market window, not yet briefed today"


def write_marker(now_et, marker_path=None):
    marker_path = marker_path or MARKER_PATH
    os.makedirs(os.path.dirname(marker_path), exist_ok=True)
    with open(marker_path, "w") as f:
        f.write(now_et.date().isoformat())


def _fetch_quote(symbol):
    """(last_price, pct_vs_prior_settle_or_None) or None. Pre-market
    quotes are thin — every failure degrades to None, never raises."""
    try:
        import yfinance as yf
        tk = yf.Ticker(symbol)
        last = prev = None
        try:
            fi = tk.fast_info
            last = getattr(fi, "last_price", None)
            prev = getattr(fi, "previous_close", None)
        except Exception:
            pass
        if not last or not prev:
            df = tk.history(period="5d")
            if df is not None and not df.empty:
                closes = df["Close"]
                last = last or float(closes.iloc[-1])
                if not prev and len(closes) > 1:
                    prev = float(closes.iloc[-2])
        if not last:
            return None
        pct = ((float(last) - float(prev)) / float(prev) * 100) if prev else None
        return (float(last), None if pct is None else float(pct))
    except Exception:
        return None


def fetch_quotes(symbols):
    """{symbol: (last, pct)|None} + all_failed flag."""
    quotes = {s: _fetch_quote(s) for s in symbols}
    return quotes, all(v is None for v in quotes.values())


def _fmt_quote(label, q):
    if not q:
        return f"{label} n/a"
    last, pct = q
    pct_txt = f" {'+' if pct >= 0 else ''}{pct:.1f}%" if pct is not None else ""
    return f"{label} {last:,.2f}{pct_txt}"


def _holding_line(ticker, art_row, quote):
    """Gap check for one holding: live (or last-close) price vs the last
    computed SMA20 stop. 🔴 below the line, ⚠️ within 0.5×ATR above."""
    art_row = art_row or {}
    stop = (art_row.get("stop") or {}).get("level")
    atr = ((art_row.get("conditions") or {}).get("2_confirmation")
           or {}).get("atr14")
    state = art_row.get("state")
    if quote:
        price, src_note = quote[0], ""
    elif art_row.get("close") is not None:
        price, src_note = art_row["close"], " (last close)"
    else:
        return f"• {ticker} — no price data"
    st = f" ({state})" if state and state != "HELD" else ""
    if stop is None:
        # No active stop = the exit already signaled (stops exist only for
        # HELD/EXIT_FIRED). Reference the SMA20 reclaim line instead — no
        # alarm flag; the alarm was yesterday's close report.
        sma20 = art_row.get("sma20")
        if sma20 is not None:
            return (f"• {ticker}{st} ${price:.2f}{src_note} vs SMA20 ${sma20} "
                    f"— no active stop (exit signaled; reclaim = close above SMA20)")
        return f"• {ticker}{st} ${price:.2f}{src_note} — no computed stop in last artifact"
    if price < stop:
        return (f"🔴 • {ticker}{st} ${price:.2f}{src_note} vs stop ${stop} — "
                f"below stop pre-market — close decides, no pre-emption")
    if atr and price < stop + 0.5 * atr:
        return (f"⚠️ • {ticker}{st} ${price:.2f}{src_note} vs stop ${stop} — "
                f"within 0.5×ATR of stop")
    return f"• {ticker}{st} ${price:.2f}{src_note} vs stop ${stop}"


def build_message(data, positions, quotes, all_failed, now_et):
    """Compact pre-market briefing. Pure formatting; no network."""
    lines = [f"*SignalAgent PRE-MARKET* — "
             f"{now_et.strftime('%Y-%m-%d %I:%M %p')} ET"]
    if all_failed:
        lines.append("_live quotes unavailable — showing last-close data_")

    lines.append("Futures: " + " · ".join(
        _fmt_quote(label, quotes.get(sym)) for sym, label in FUTURES))

    vix = quotes.get("^VIX")
    vix3m = quotes.get("^VIX3M")
    if vix and vix3m:
        term = "contango" if vix3m[0] > vix[0] else "inverted"
    else:
        term = "n/a"
    lines.append("Vol: " + " · ".join(
        _fmt_quote(label, quotes.get(sym)) for sym, label in VOL)
        + f" · {term}")

    art_tickers = ((data.get("position_signals") or {}).get("tickers") or {})
    holdings = [h for h in (positions.get("holdings") or [])
                if isinstance(h, dict) and h.get("ticker")]
    lines.append("*Holdings gap check*")
    if not holdings:
        lines.append("• none — cash")
    for h in holdings:
        t = h["ticker"]
        lines.append(_holding_line(t, art_tickers.get(t), quotes.get(t)))

    watchers = [(t, x) for t, x in art_tickers.items()
                if isinstance(x, dict) and x.get("kind") == "watching"]
    if watchers:
        lines.append("Watchlist: " + " · ".join(
            f"{t} {x.get('state', '?')}" for t, x in watchers))

    regime = (data.get("regime") or {})
    gen_date = str(data.get("generated_at", ""))[:10]
    lines.append(f"Regime: {regime.get('regime', '?')} "
                 f"({regime.get('risk_on_count')}/{regime.get('caution_count')}"
                 f"/{regime.get('risk_off_count')}) — as of last close"
                 + (f" ({gen_date})" if gen_date else ""))

    dash = os.environ.get("DASHBOARD_URL", "").rstrip("/")
    if dash:
        if not dash.endswith("framework.html"):
            dash = f"{dash}/framework.html"
        lines.append(f"<{dash}|framework.html>")
    return "\n".join(lines)


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
            print("[premarket] no webhook configured — skipping")
            return 0
        now = _now_et()
        ok, reason = should_notify(now)
        print(f"[premarket] {reason}")
        if not ok:
            return 0

        try:
            with open(FRAMEWORK_JSON, "r") as f:
                data = json.load(f)
        except Exception:
            data = {}
        try:
            with open(POSITIONS_JSON, "r") as f:
                positions = json.load(f)
        except Exception:
            positions = {}

        holding_syms = [h["ticker"] for h in (positions.get("holdings") or [])
                        if isinstance(h, dict) and h.get("ticker")]
        symbols = ([s for s, _ in FUTURES] + [s for s, _ in VOL]
                   + holding_syms)
        quotes, all_failed = fetch_quotes(symbols)
        if all_failed:
            print("[premarket] all quote fetches failed — sending "
                  "last-close briefing")

        msg = build_message(data, positions, quotes, all_failed, now)
        post_to_slack(webhook, msg)
        write_marker(now)   # only after a successful post: failures retry
        print(f"[premarket] posted briefing ({len(msg)} chars); "
              f"marker written for {now.date().isoformat()}")
        return 0
    except Exception as e:
        # NEVER fail the workflow; NEVER echo the webhook URL.
        text = str(e)
        if webhook:
            text = text.replace(webhook, "<webhook>")
        print(f"[premarket] failed (non-fatal): {type(e).__name__}: {text}")
        return 0


if __name__ == "__main__":
    raise SystemExit(main())
