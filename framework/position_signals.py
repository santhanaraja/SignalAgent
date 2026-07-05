#!/usr/bin/env python3
"""
Position Signal Engine (Build 1B) — per-ticker exit/re-entry state machine.

Makes exits and re-entries mechanical for tracked holdings and watchlist
names. Consumes the swing regime gauge and the Sunday-cadence theme
qualification READ-ONLY. Signals only — nothing is ever auto-executed.

The re-entry rule — ALL FIVE conditions must hold:
  1. trigger:      close above SMA20 (close basis)
  2. confirmation: 2 consecutive closes above SMA20, OR one close more
                   than atr_mult * ATR(atr_period) above SMA20
  3. regime gate:  Risk-on / Trending = full clearance;
                   Risk-on / Choppy   = conditional ("A+ only" flag);
                   Caution / Risk-off = blocked
  4. slope:        SMA20 flat-or-rising vs slope_lookback_days ago
  5. thesis:       the ticker's theme still qualified (active, or ranked
                   top-N) per the R4-fixed weekly qualification. Themes
                   with no framework mapping (e.g. "SmallCap/Broad" or an
                   explicit "external:" prefix) pass automatically but are
                   flagged no_theme_mapping — never silently true.

State machine (close basis, evaluated daily):
  HELD -> EXIT_FIRED (close below SMA20; this IS the exit signal)
       -> WATCHING (still below)
       -> RE_ENTRY_ARMING (reclaim above SMA20, not all 5 yet)
       -> RE_ENTRY_READY (all 5 true; carries "A+ only" flag under Choppy)
  A close back below SMA20 from ARMING/READY returns to WATCHING.

Every state transition is emitted as a history event
(type "position_state_change") through the existing history.json pattern.
"""

import datetime
import json
import os

import numpy as np


def _utcnow():
    """Timestamp — module-level so tests can inject replay times."""
    return datetime.datetime.now(datetime.timezone.utc)


HELD = "HELD"
EXIT_FIRED = "EXIT_FIRED"
WATCHING = "WATCHING"
RE_ENTRY_ARMING = "RE_ENTRY_ARMING"
RE_ENTRY_READY = "RE_ENTRY_READY"

_SEVERITY = {
    EXIT_FIRED: "critical",
    RE_ENTRY_READY: "high",
    HELD: "medium",
    RE_ENTRY_ARMING: "medium",
    WATCHING: "low",
}


class PositionSignalEngine:
    """Evaluate the 5-condition re-entry rule per tracked ticker."""

    STATE_DIR = os.path.join(os.path.dirname(__file__), "state")
    # history files live at the repo root (same files history_manager writes)
    DATA_DIR = os.path.join(os.path.dirname(__file__), "..", "data")
    PUBLIC_DIR = os.path.join(os.path.dirname(__file__), "..", "public")

    def __init__(self, config: dict, fetcher):
        cfg = config.get("positions", {}) or {}
        self.sma_period = cfg.get("sma_period", 20)
        self.confirmation_closes = cfg.get("confirmation_closes", 2)
        self.atr_period = cfg.get("atr_period", 14)
        self.atr_mult = cfg.get("atr_mult", 0.5)
        self.slope_lookback = cfg.get("slope_lookback_days", 5)
        self.themes_cfg = config.get("themes", {}) or {}
        self.fetcher = fetcher
        os.makedirs(self.STATE_DIR, exist_ok=True)

    # ------------------------------------------------------------------
    # Pure per-day evaluation (also used by replay harnesses/tests)
    # ------------------------------------------------------------------

    def evaluate(self, entry: dict, kind: str, df, regime_state: str,
                 theme_status: dict, prev_state: str) -> dict:
        """
        Evaluate one ticker for the last bar of df.

        entry: the positions.json record. kind: "holding" | "watching".
        df: daily OHLC history up to and including the evaluation day.
        theme_status: from _theme_status(). prev_state: persisted state or
        None on first sight.

        Pure: no I/O, no state writes — replay harnesses feed truncated
        windows day by day.
        """
        min_bars = max(self.sma_period + self.slope_lookback,
                       self.atr_period + 1) + 1
        if df is None or len(df) < min_bars:
            return {
                "state": prev_state or WATCHING,
                "insufficient_data": True,
                "detail": f"insufficient history ({0 if df is None else len(df)} bars,"
                          f" need {min_bars})",
                "conditions": {},
            }

        close = df["Close"]
        sma = close.rolling(self.sma_period).mean()
        last_close = float(close.iloc[-1])
        sma_now = float(sma.iloc[-1])
        sma_then = float(sma.iloc[-1 - self.slope_lookback])
        if not (np.isfinite(last_close) and np.isfinite(sma_now)
                and np.isfinite(sma_then)):
            return {
                "state": prev_state or WATCHING,
                "insufficient_data": True,
                "detail": "non-finite close/SMA in window",
                "conditions": {},
            }
        atr = self._atr(df, self.atr_period)

        above_now = last_close > sma_now
        # consecutive closes above their same-day SMA20
        consec_above = 0
        for i in range(1, len(close) + 1):
            s = sma.iloc[-i]
            if np.isnan(s) or not float(close.iloc[-i]) > float(s):
                break
            consec_above += 1

        atr_break = above_now and atr is not None \
            and last_close > sma_now + self.atr_mult * atr

        c1 = {
            "name": "trigger",
            "met": above_now,
            "detail": f"close {last_close:.2f} {'above' if above_now else 'below'}"
                      f" SMA20 {sma_now:.2f}",
        }
        c2_met = consec_above >= self.confirmation_closes or atr_break
        if atr_break:
            c2_how = (f"close {last_close:.2f} > SMA20 + "
                      f"{self.atr_mult}*ATR14 ({sma_now + self.atr_mult * atr:.2f})")
        else:
            c2_how = (f"{min(consec_above, self.confirmation_closes)}"
                      f"/{self.confirmation_closes} consecutive closes above SMA20")
        c2 = {"name": "confirmation", "met": c2_met, "detail": c2_how,
              "consecutive_closes_above": consec_above,
              "atr14": None if atr is None else round(atr, 2)}

        if regime_state == "Risk-on / Trending":
            c3 = {"met": True, "mode": "full",
                  "detail": "regime Risk-on / Trending — full clearance"}
        elif regime_state == "Risk-on / Choppy":
            c3 = {"met": True, "mode": "conditional",
                  "detail": "regime Risk-on / Choppy — conditional, A+ setups only"}
        else:
            c3 = {"met": False, "mode": "blocked",
                  "detail": f"regime {regime_state} — re-entry blocked"}
        c3["name"] = "regime_gate"

        slope_ok = sma_now >= sma_then
        c4 = {
            "name": "slope",
            "met": slope_ok,
            "detail": f"SMA20 {sma_now:.2f} vs {self.slope_lookback}d ago "
                      f"{sma_then:.2f} ({'flat/rising' if slope_ok else 'falling'})",
        }

        c5 = {
            "name": "thesis",
            "met": theme_status["met"],
            "detail": theme_status["detail"],
        }
        if theme_status.get("no_theme_mapping"):
            c5["no_theme_mapping"] = True

        conditions = {"1_trigger": c1, "2_confirmation": c2, "3_regime_gate": c3,
                      "4_slope": c4, "5_thesis": c5}
        all_five = all(c["met"] for c in conditions.values())

        state = self._next_state(prev_state, kind, above_now, all_five)

        # Informational extension readings (no gating — the discretion
        # layer decides what "too extended to chase" means, e.g. MRNA 33%
        # above its SMA20 printing READY)
        ext_pct = (last_close - sma_now) / sma_now * 100
        ext_atr = None if not atr else (last_close - sma_now) / atr

        result = {
            "state": state,
            "close": round(last_close, 2),
            "sma20": round(sma_now, 2),
            "extension_pct": round(ext_pct, 2),
            "extension_atr": None if ext_atr is None else round(ext_atr, 2),
            "conditions": conditions,
            "conditions_met": sum(1 for c in conditions.values() if c["met"]),
            "all_conditions_met": all_five,
        }
        if state == RE_ENTRY_READY and c3.get("mode") == "conditional":
            result["a_plus_only"] = True
        if not above_now:
            result["distance_to_sma20_pct"] = round(
                (last_close - sma_now) / sma_now * 100, 2)
        # Effective stop for held names (and the stop that just fired)
        if state in (HELD, EXIT_FIRED):
            result["stop"] = self._stop_for(entry, sma_now)
        return result

    def _next_state(self, prev, kind, above, all_five):
        if prev in (None, ""):
            if kind == "holding":
                return HELD if above else EXIT_FIRED
            return self._reclaim_state(above, all_five)
        if prev == HELD:
            return HELD if above else EXIT_FIRED
        # EXIT_FIRED / WATCHING / RE_ENTRY_ARMING / RE_ENTRY_READY
        state = self._reclaim_state(above, all_five)
        # positions.json is authoritative for what is held: a HOLDING whose
        # re-entry conditions complete returns to HELD (live stop resumes).
        # Without this, HELD is unreachable forever after one exit.
        if state == RE_ENTRY_READY and kind == "holding":
            return HELD
        return state

    @staticmethod
    def _reclaim_state(above, all_five):
        if not above:
            return WATCHING
        return RE_ENTRY_READY if all_five else RE_ENTRY_ARMING

    @staticmethod
    def _atr(df, period):
        """ATR as the simple mean of the last `period` True Ranges."""
        if len(df) < period + 1:
            return None
        high = df["High"].to_numpy(dtype=float)[-(period + 1):]
        low = df["Low"].to_numpy(dtype=float)[-(period + 1):]
        close = df["Close"].to_numpy(dtype=float)[-(period + 1):]
        prev_close = close[:-1]
        tr = np.maximum(high[1:] - low[1:],
                        np.maximum(np.abs(high[1:] - prev_close),
                                   np.abs(low[1:] - prev_close)))
        return float(tr.mean())

    def _stop_for(self, entry, sma_now):
        rule = entry.get("stop_on_entry")
        if rule == "sma20_close":
            return {"type": "sma20_close", "level": round(sma_now, 2),
                    "detail": f"exit on close below SMA20 ({sma_now:.2f})"}
        if rule:
            return {"type": rule, "level": None,
                    "detail": f"stop rule '{rule}' — level not computed by engine"}
        return {"type": None, "level": None, "detail": "no stop rule defined"}

    def _theme_status(self, theme, watchlist_names, active_names, ranks, top_n):
        """Condition 5 basis. Unmapped themes pass but are FLAGGED."""
        if not theme or theme.startswith("external:") \
                or theme not in watchlist_names:
            return {"met": True, "no_theme_mapping": True,
                    "detail": f"no framework theme mapping ('{theme}') — "
                              f"thesis gate not applicable"}
        if theme in active_names:
            return {"met": True,
                    "detail": f"theme '{theme}' active (weekly qualification)"}
        rank = ranks.get(theme)
        if rank is not None and rank <= top_n:
            return {"met": True,
                    "detail": f"theme '{theme}' ranked #{rank} (top {top_n})"}
        return {"met": False,
                "detail": f"theme '{theme}' not qualified "
                          f"(rank {rank if rank is not None else 'n/a'}, not active)"}

    # ------------------------------------------------------------------
    # Live compute: load positions + prior states, evaluate, persist, emit
    # ------------------------------------------------------------------

    def compute(self, regime_result: dict, theme_result: dict = None,
                qualified_active: list = None, emit_events: bool = True) -> dict:
        positions_path = os.path.join(self.STATE_DIR, "positions.json")
        if os.path.exists(positions_path):
            positions = self._load_json(positions_path, None)
            if positions is None or not isinstance(positions, dict):
                # Unparseable/wrong-shaped hand-edited file: skip the layer
                # WITHOUT touching persisted state (treating it as empty
                # would prune every tracked ticker's state).
                print("[framework] positions.json unreadable — position "
                      "signals skipped, state preserved")
                return {
                    "generated_at": _utcnow().isoformat(),
                    "error": "positions.json unreadable — engine skipped",
                    "tickers": {},
                    "transitions": [],
                }
        else:
            positions = {"holdings": [], "watching": []}
        prev_states = self._load_json(
            os.path.join(self.STATE_DIR, "position_state.json"), {})

        if qualified_active is None:
            qualified = self._load_json(
                os.path.join(self.STATE_DIR, "qualified_themes.json"),
                {"active": []})
            qualified_active = [a["name"] if isinstance(a, dict) else a
                                for a in qualified.get("active", [])]
        ranks = {t["name"]: t.get("rank")
                 for t in (theme_result or {}).get("ranked_themes", [])}
        top_n = self.themes_cfg.get("entry_rule", {}).get("requires_top_n_rank", 2)
        watchlist_names = {t["name"] for t in self.themes_cfg.get("watchlist", [])}
        regime_state = (regime_result or {}).get("regime", "Unknown")

        tickers, transitions = {}, []
        entries = [(h, "holding") for h in positions.get("holdings", []) or []
                   if isinstance(h, dict)] + \
                  [(w, "watching") for w in positions.get("watching", []) or []
                   if isinstance(w, dict)]
        seen = set()
        for entry, kind in entries:
            ticker = entry.get("ticker")
            if not ticker:
                continue
            if ticker in seen:
                # duplicate listing (e.g. in both holdings and watching):
                # holdings iterate first and win; flag instead of colliding
                print(f"[framework] positions.json lists {ticker} twice — "
                      f"keeping the first (holdings-first) entry")
                continue
            seen.add(ticker)
            df = self._strip_synthetic_last_bar(self.fetcher(ticker, period="6mo"))
            theme_status = self._theme_status(entry.get("theme"), watchlist_names,
                                              qualified_active, ranks, top_n)
            prev = (prev_states.get(ticker) or {}).get("state")
            result = self.evaluate(entry, kind, df, regime_state, theme_status, prev)
            result["ticker"] = ticker
            result["kind"] = kind
            result["theme"] = entry.get("theme")
            result["note"] = entry.get("note")
            tickers[ticker] = result

            if result.get("insufficient_data"):
                # No data, no verdict: leave persisted state untouched so a
                # transient fetch failure can't seed or poison the machine.
                continue
            new_state = result["state"]
            if new_state != prev:
                event = self._transition_event(ticker, entry, prev, result)
                transitions.append(event)
                prev_states[ticker] = {"state": new_state,
                                       "since": _utcnow().date().isoformat()}
            elif ticker not in prev_states:
                prev_states[ticker] = {"state": new_state,
                                       "since": _utcnow().date().isoformat()}

        # prune states for tickers no longer tracked
        tracked = {e.get("ticker") for e, _ in entries}
        prev_states = {t: s for t, s in prev_states.items() if t in tracked}

        self._save_json(os.path.join(self.STATE_DIR, "position_state.json"),
                        prev_states)
        if emit_events and transitions:
            self.emit_history_events(transitions)

        return {
            "generated_at": _utcnow().isoformat(),
            "account": positions.get("account"),
            "schema_version": positions.get("schema_version"),
            "regime_state": regime_state,
            "tickers": tickers,
            "transitions": transitions,
        }

    def _transition_event(self, ticker, entry, prev, result):
        state = result["state"]
        met = result.get("conditions_met", 0)
        detail = {
            "from_state": prev,
            "to_state": state,
            "close": result.get("close"),
            "sma20": result.get("sma20"),
            "extension_pct": result.get("extension_pct"),
            "extension_atr": result.get("extension_atr"),
            "conditions_met": f"{met}/5",
        }
        if result.get("a_plus_only"):
            detail["a_plus_only"] = True
        stop = result.get("stop")
        if state == EXIT_FIRED and stop:
            detail["stop"] = stop.get("detail")
        desc_from = prev if prev else "untracked"
        return {
            "timestamp": _utcnow().isoformat(),
            "type": "position_state_change",
            "severity": _SEVERITY.get(state, "medium"),
            "group": entry.get("theme"),
            "ticker": ticker,
            "description": f"{ticker}: {desc_from} → {state}",
            "detail": detail,
        }

    def emit_history_events(self, events):
        """
        Append events to data/position_events.json + public/ mirror —
        same event schema and {"changes": [...]} shape as history.json,
        but a SEPARATE file: history_manager rewrites history.json from
        its own in-memory copy every pipeline run, so a second writer to
        the shared file loses events in a plain read-modify-write race.
        history.html merges both files into one timeline. Sanitized.
        """
        import sys
        sys.path.insert(0, os.path.join(os.path.dirname(__file__), ".."))
        from signal_engine import sanitize_for_json
        events = sanitize_for_json(events)
        for base in (self.DATA_DIR, self.PUBLIC_DIR):
            path = os.path.join(base, "position_events.json")
            history = self._load_json(path, {"changes": []})
            history.setdefault("changes", []).extend(events)
            history["changes"] = history["changes"][-500:]
            os.makedirs(base, exist_ok=True)
            self._save_json(path, history)

    @staticmethod
    def _strip_synthetic_last_bar(df):
        """
        Drop fetch_data's live-quote append (Volume 0, O==H==L==C): the
        state machine is CLOSE-basis (R11 — never intraday wicks). Intraday
        runs therefore evaluate the last COMPLETED daily bar; the post-close
        run sees today's real bar and advances the machine.
        """
        if df is None or len(df) == 0:
            return df
        last = df.iloc[-1]
        try:
            if float(last["Volume"]) == 0 and \
                    float(last["Open"]) == float(last["High"]) \
                    == float(last["Low"]) == float(last["Close"]):
                return df.iloc[:-1]
        except (KeyError, TypeError, ValueError):
            pass
        return df

    # ------------------------------------------------------------------

    @staticmethod
    def _load_json(path, default):
        if os.path.exists(path):
            try:
                with open(path, "r") as f:
                    return json.load(f)
            except (json.JSONDecodeError, IOError):
                pass
        return default

    @staticmethod
    def _save_json(path, obj):
        with open(path, "w") as f:
            json.dump(obj, f, indent=2)
