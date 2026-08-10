"""Grade outcome tracker — simulated spell trades over the grade series.

OBSERVES ONLY. Nothing here is a position and nothing feeds production
logic: this replays the committed grade history (data/grade_history.json,
e8a1ada line) through the production fill/stop discipline to measure
what graded names went on to do.

One trade per grade SPELL (the first day a ticker enters a grade):
entry at the next session's OPEN (D-006), stop = SMA20 close-basis —
a confirmed close NOT ABOVE the SMA20 exits at the next open (D-018's
equality law, deliberately the production rule rather than the looser
"close below"). Never a second trade on a ticker while one runs;
re-entry only after the stop's exit fill. Trades whose entry or exit
fill has not printed yet are PENDING / EXIT_FIRED — never silently
dropped, never counted as closed.

Statuses: PENDING (spell seen, next open not printed yet) · OPEN ·
EXIT_FIRED (stop fired on the latest close, exit fill pending) ·
STOPPED (closed). Aggregates are computed on STOPPED trades only and
always carry the censoring counts beside them.

Two laws added by adversarial review:
 · CONFIRMED CLOSES ONLY — every fetched frame passes through the
   D-018 splitter (framework.regime_calculator.confirmed_close_frame)
   before simulation, so an intraday rebuild can never fire a stop or
   set an anchor off the forming bar.
 · CLOSED TRADES ARE FACTS — a STOPPED trade recorded in a prior
   artifact is carried forward if a later rebuild cannot re-derive it
   (price window rolled past its spell, or a transient fetch failure);
   it is marked "carried" and never silently deleted. days_held counts
   CONFIRMED CLOSES HELD (the 6A convention); MFE includes the exit
   fill the trade actually received, so pnl can never exceed it.
"""

import datetime
import hashlib
import json
import os

import numpy as np
import pandas as pd

BASE_DIR = os.path.dirname(os.path.abspath(__file__))
GRADE_HISTORY_PATH = os.path.join(BASE_DIR, "data", "grade_history.json")
OUTCOMES_PATH = os.path.join(BASE_DIR, "data", "grade_outcomes.json")

SCHEMA = "grade-outcomes-1"
SMA_PERIOD = 20


# ------------------------------------------------------------- spells
def detect_spells(days):
    """[(date, ticker, grade, scorer_version)] — the FIRST day of each
    per-ticker grade run. A ticker absent the previous series day (not
    graded) starts a spell on reappearance even in the same grade."""
    spells = []
    prev = {}
    for day in days:
        grades = day["grades"]
        for t, g in grades.items():
            if prev.get(t) != g:
                spells.append((day["date"], t, g, day["scorer_version"]))
        prev = {t: g for t, g in grades.items()}
    return spells


# ------------------------------------------------------------- pricing
def fetch_frames(tickers, period="1y"):
    """Batch OHLC via yfinance (auto-adjusted, naive index) — injectable
    in tests; the simulator never fetches on its own."""
    import yfinance as yf
    frames = {}
    tickers = sorted(set(tickers))
    for i in range(0, len(tickers), 100):
        chunk = tickers[i:i + 100]
        df = yf.download(chunk, period=period, interval="1d",
                         group_by="ticker", auto_adjust=True,
                         threads=True, progress=False)
        if df is None or df.empty:
            continue
        for t in chunk:
            try:
                if t in df.columns.get_level_values(0):
                    sub = df[t].dropna(subset=["Close"])
                    if len(sub):
                        sub = sub.copy()
                        try:
                            sub.index = sub.index.tz_localize(None)
                        except TypeError:
                            pass
                        frames[t] = sub
            except Exception:
                continue
    return frames


# ------------------------------------------------------------ simulate
def simulate_ticker(spells, df):
    """All trades for one ticker's chronological spells. `df` is the
    OHLC frame. Returns (trades, skips)."""
    close = df["Close"]
    sma20 = close.rolling(SMA_PERIOD).mean()
    dates = [str(d)[:10] for d in df.index]
    trades, skips = [], []
    busy_until = None            # exit-fill date; eligible from that day on

    for grade_date, t, grade, era in spells:
        if busy_until is not None and grade_date < busy_until:
            skips.append({"ticker": t, "grade_date": grade_date,
                          "grade": grade, "reason": "trade_running"})
            continue
        # signal-day bar: last bar <= grade date (stop basis)
        sig_i = None
        for i in range(len(dates) - 1, -1, -1):
            if dates[i] <= grade_date:
                sig_i = i
                break
        if sig_i is None:
            # the frame does not reach the spell at all — a WINDOW
            # problem, not a data-poverty one (D-019: label the truth)
            skips.append({"ticker": t, "grade_date": grade_date,
                          "grade": grade, "reason": "price_window_rolled"})
            continue
        if sig_i < SMA_PERIOD - 1 or not np.isfinite(sma20.iloc[sig_i]):
            skips.append({"ticker": t, "grade_date": grade_date,
                          "grade": grade, "reason": "insufficient_history"})
            continue
        stop = float(sma20.iloc[sig_i])

        base = {"ticker": t, "grade": grade, "grade_date": grade_date,
                "scorer_era": era, "initial_stop": round(stop, 4)}
        if sig_i + 1 >= len(dates):          # next open not printed yet
            trades.append({**base, "status": "PENDING",
                           "entry_date": None, "entry": None,
                           "initial_r": None, "exit_date": None,
                           "exit": None, "days_held": 0,
                           "pnl_pct": None, "pnl_r": None,
                           "mfe_pct": None, "ever_profitable": None})
            busy_until = "9999-12-31"        # pending entry occupies the
            continue                         # ticker; later spells are
                                             # recorded skips, not silence
        e_i = sig_i + 1
        entry = float(df["Open"].iloc[e_i])
        r_ps = entry - stop                  # may be <=0 for below-SMA names
        status, exit_date, exit_px = "OPEN", None, None
        fired_i = None
        for i in range(e_i, len(dates)):     # confirmed closes, D-018
            c = float(close.iloc[i])
            s = sma20.iloc[i]
            if np.isfinite(s) and not c > float(s):
                fired_i = i
                break
        if fired_i is not None:
            if fired_i + 1 < len(dates):
                status = "STOPPED"
                exit_date = dates[fired_i + 1]
                exit_px = float(df["Open"].iloc[fired_i + 1])
                busy_until = exit_date
            else:
                status = "EXIT_FIRED"        # fill pending at next open
                busy_until = "9999-12-31"
        else:
            busy_until = "9999-12-31"        # OPEN through the last bar
        last_i = fired_i if fired_i is not None else len(dates) - 1
        highs = df["High"].iloc[e_i:last_i + 1]
        peak = float(highs.max()) if len(highs) else None
        if exit_px is not None:
            # the exit FILL is part of the trade's excursion — without
            # it a gap-up exit produced pnl > mfe, a definitional
            # impossibility (review finding: 13/72 closed trades)
            peak = max(peak, exit_px) if peak is not None else exit_px
        mfe = round((peak / entry - 1) * 100, 2) \
            if peak is not None and entry > 0 else None
        pnl_pct = pnl_r = None
        if status == "STOPPED" and entry > 0:
            pnl_pct = round((exit_px / entry - 1) * 100, 2)
            pnl_r = round((exit_px - entry) / r_ps, 2) if r_ps > 0 else None
        trades.append({**base, "status": status,
                       "entry_date": dates[e_i], "entry": round(entry, 4),
                       "initial_r": round(r_ps, 4),
                       "exit_date": exit_date,
                       "exit": round(exit_px, 4) if exit_px else None,
                       "days_held": last_i - e_i + 1,
                       "pnl_pct": pnl_pct, "pnl_r": pnl_r,
                       "mfe_pct": mfe,
                       "ever_profitable": (mfe is not None and mfe > 0)})
    return trades, skips


def aggregate(trades):
    """Per-grade blocks, CLOSED-ONLY statistics, censoring counts always
    beside them."""
    out = {}
    for grade in ("A+", "B", "C"):
        rows = [t for t in trades if t["grade"] == grade]
        closed = [t for t in rows if t["status"] == "STOPPED"]
        open_ = [t for t in rows if t["status"] != "STOPPED"]
        blk = {"n_closed": len(closed), "n_open": len(open_),
               "open_breakdown": {
                   s: sum(1 for t in open_ if t["status"] == s)
                   for s in ("OPEN", "EXIT_FIRED", "PENDING")
                   if any(t["status"] == s for t in open_)}}
        if closed:
            held = sorted(t["days_held"] for t in closed)
            pnl = [t["pnl_pct"] for t in closed if t["pnl_pct"] is not None]
            mfe = sorted(t["mfe_pct"] for t in closed
                         if t["mfe_pct"] is not None)
            blk.update({
                "days_held_median_closed": held[len(held) // 2],
                "days_held_mean_closed": round(sum(held) / len(held), 1),
                "pct_profitable_closed": round(
                    100 * sum(1 for t in closed
                              if (t["pnl_pct"] or 0) > 0) / len(closed), 1),
                "pnl_pct_mean_closed": round(sum(pnl) / len(pnl), 2)
                if pnl else None,
                "pnl_pct_median_closed": sorted(pnl)[len(pnl) // 2]
                if pnl else None,
                "mfe_pct_closed": {
                    "median": mfe[len(mfe) // 2] if mfe else None,
                    "p75": mfe[int(len(mfe) * 0.75)] if mfe else None,
                    "max": mfe[-1] if mfe else None},
            })
        out[grade] = blk
    return out


# ----------------------------------------------------------------- build
def series_content_key(doc):
    """Hash of the series' GRADE CONTENT only — date, era, grades.
    generated_at/source_commit churn every bake (the D-012 upsert
    refreshes the day row), so a byte-hash rebuilt and refetched every
    15 minutes; the staleness guard must key on what the simulation
    actually reads (review finding: the no-op contract was dead)."""
    canon = [(d["date"], d.get("scorer_version"),
              sorted((d.get("grades") or {}).items()))
             for d in doc.get("days", [])]
    return hashlib.sha256(
        json.dumps(canon, sort_keys=True).encode()).hexdigest()[:16]


def state_key(doc, today):
    return series_content_key(doc) + ":" + today


def build(history_path=None, frames=None, out_path=None, today=None,
          now_et=None):
    history_path = history_path or GRADE_HISTORY_PATH
    out_path = out_path or OUTCOMES_PATH
    today = today or datetime.date.today().isoformat()
    raw = open(history_path, "rb").read()
    doc = json.loads(raw)
    spells = detect_spells(doc["days"])
    tickers = sorted({t for _, t, _, _ in spells})
    if frames is None:
        frames = fetch_frames(tickers)
    # D-018 confirmed closes ONLY: strip the forming bar before any
    # stop check or anchor — an intraday rebuild must never fire on an
    # unconfirmed close (review finding, HIGH)
    from framework.regime_calculator import confirmed_close_frame
    frames = {t: confirmed_close_frame(df, now_et=now_et)[0]
              for t, df in frames.items()}
    by_ticker = {}
    for sp in spells:
        by_ticker.setdefault(sp[1], []).append(sp)
    trades, skips = [], []
    missing = []
    for t, sps in sorted(by_ticker.items()):
        df = frames.get(t)
        if df is None or len(df) < SMA_PERIOD + 1:
            missing.append(t)
            continue
        tr, sk = simulate_ticker(sorted(sps), df)
        trades.extend(tr)
        skips.extend(sk)
    # CLOSED TRADES ARE FACTS: carry forward any previously recorded
    # STOPPED trade this rebuild could not re-derive (window rolled or
    # ticker fetch failed) — marked, never silently deleted
    carried = 0
    have = {(t["ticker"], t["grade_date"]) for t in trades}
    if os.path.exists(out_path):
        try:
            with open(out_path) as f:
                prior = json.load(f)
            for pt in prior.get("trades", []):
                key = (pt["ticker"], pt["grade_date"])
                if pt.get("status") == "STOPPED" and key not in have:
                    trades.append({**pt, "carried": True})
                    have.add(key)
                    carried += 1
                    skips = [k for k in skips
                             if (k["ticker"], k["grade_date"]) != key]
        except (OSError, ValueError):
            pass
    out = {
        "schema": SCHEMA,
        "generated_at": datetime.datetime.now(
            datetime.timezone.utc).isoformat(),
        "state_key": state_key(doc, today),
        "disclaimer": ("SIMULATED outcomes on graded names — the grade "
                       "series replayed through the production fill/stop "
                       "discipline. Nothing here is a position."),
        "mechanics": {"entry": "next session open after the grade date "
                               "(D-006)",
                      "stop": "SMA20 close-basis, equality exits, fill at "
                              "next open (D-018)",
                      "overlap": "one trade per ticker at a time; "
                                 "re-entry only after the exit fill"},
        "provenance": {"grade_history_sha16":
                       hashlib.sha256(raw).hexdigest()[:16],
                       "series_days": len(doc["days"]),
                       "series_span": [doc["days"][0]["date"],
                                       doc["days"][-1]["date"]],
                       "tickers_no_price_data": missing,
                       "carried_closed_trades": carried},
        "trades": trades,
        "skipped_spells": skips,
        "aggregates": aggregate(trades),
    }
    tmp = out_path + ".tmp"
    with open(tmp, "w") as f:
        json.dump(out, f, indent=1)
    os.replace(tmp, out_path)
    return out


def refresh_if_stale(today=None):
    """The per-bake hook (framework_runner): full rebuild only when the
    grade history changed or the day rolled over — an intraday re-bake
    with an unchanged series is a fast no-op (trades only advance on
    confirmed closes)."""
    today = today or datetime.date.today().isoformat()
    try:
        with open(GRADE_HISTORY_PATH) as f:
            doc = json.load(f)
    except (OSError, ValueError):
        return "no-series"
    want = state_key(doc, today)
    if os.path.exists(OUTCOMES_PATH):
        try:
            with open(OUTCOMES_PATH) as f:
                if json.load(f).get("state_key") == want:
                    return "fresh"
        except (OSError, ValueError):
            pass
    build(today=today)
    return "rebuilt"


if __name__ == "__main__":
    out = build()
    ag = out["aggregates"]
    print(f"trades: {len(out['trades'])}, skipped spells: "
          f"{len(out['skipped_spells'])}, no-price tickers: "
          f"{len(out['provenance']['tickers_no_price_data'])}")
    for g in ("A+", "B", "C"):
        print(f"  {g}: {json.dumps(ag[g])}")
