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
  5. thesis:       the ticker's GICS group is in the CURRENT active
                   universe's selected groups (D-007 Phase 1 — the scanner
                   is the thesis; replaces theme_qualified). Group resolves
                   through the shared R28 resolver (universe mapping ->
                   signals.json fallback -> manual row override). Tickers
                   with NO resolvable group pass automatically but are
                   flagged no_group_mapping — never silently true.

The A+ grade (D-011, rides this build): grade_setup() computes the ruled
seven-row setup grade beside the 5 conditions — A+ / B (named reasons) /
C (mechanical-or-guard fail, or the approach-filter C-escalation clause:
a knife is blocked in EVERY regime). Hard gate in Choppy (RE_ENTRY_READY
without A+ renders blocked — the EXTENDED_HOLD visual law); advisory in
Trending. The old theme layer keeps computing DISPLAY-ONLY in parallel
(Phase 3 deletes it).

State machine (close basis, evaluated daily):
  HELD -> EXIT_FIRED (close below SMA20; this IS the exit signal)
       -> WATCHING (still below)
       -> RE_ENTRY_ARMING (reclaim above SMA20, not all 5 yet)
       -> RE_ENTRY_READY (all 5 true; carries "A+ only" flag under Choppy)
  A close back below SMA20 from ARMING/READY returns to WATCHING.
  EXTENDED_HOLD (PER-508 item 20): a WATCHLIST name with all 5 met but
  extension_atr above positions.extension_guard_max (default 1.8) is
  blocked from READY — too far above the mean to be an entry. Reverts
  to READY naturally when extension falls back within the ceiling.
  Holdings are exempt (never force-exited by the guard).

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
EXTENDED_HOLD = "EXTENDED_HOLD"   # all 5 met but too extended to enter

_SEVERITY = {
    EXIT_FIRED: "critical",
    RE_ENTRY_READY: "high",
    HELD: "medium",
    RE_ENTRY_ARMING: "medium",
    EXTENDED_HOLD: "medium",
    WATCHING: "low",
}


# ---------------------------------------------------------------------------
# The 5-condition assessment as a pure function (PER-508 item 24b)
#
# The ONLY implementation of the condition evaluators, the state map, and
# the extension guard. PositionSignalEngine.evaluate computes the scalar
# inputs from price history and delegates every decision here; the Position
# Lab endpoint (/api/position/simulate) feeds user inputs through the same
# function — a second copy of the rules is structurally impossible (Score
# Lab precedent). Threshold defaults mirror config.yaml positions; the
# engine passes its config values explicitly.
# ---------------------------------------------------------------------------

def _reclaim(above, all_five):
    if not above:
        return WATCHING
    return RE_ENTRY_READY if all_five else RE_ENTRY_ARMING


def next_state(prev, kind, above, all_five):
    if prev in (None, ""):
        if kind == "holding":
            return HELD if above else EXIT_FIRED
        return _reclaim(above, all_five)
    if prev == HELD:
        return HELD if above else EXIT_FIRED
    # EXIT_FIRED / WATCHING / RE_ENTRY_ARMING / RE_ENTRY_READY
    state = _reclaim(above, all_five)
    # positions.json is authoritative for what is held: a HOLDING whose
    # re-entry conditions complete returns to HELD (live stop resumes).
    # Without this, HELD is unreachable forever after one exit.
    if state == RE_ENTRY_READY and kind == "holding":
        return HELD
    return state


def assess_position(close, sma20, sma20_5d_ago, atr14,
                    consecutive_closes_above, regime_state, group_in_universe,
                    kind, *, prev_state=None, thesis_detail=None,
                    confirmation_closes=2, atr_mult=0.5,
                    extension_guard_max=1.8, slope_lookback_days=5):
    """The 5 conditions + state + extension guard on scalar inputs.

    Returns {state, conditions{1..5}, conditions_met, all_conditions_met,
    extension_pct, extension_atr[, extension_guard][, a_plus_only]
    [, distance_to_sma20_pct]} exactly as evaluate() embeds them.
    prev_state=None gives the stateless assessment the lab uses.
    """
    above_now = close > sma20
    atr_break = above_now and atr14 is not None \
        and close > sma20 + atr_mult * atr14

    c1 = {
        "name": "trigger",
        "met": above_now,
        "detail": f"close {close:.2f} {'above' if above_now else 'below'}"
                  f" SMA20 {sma20:.2f}",
    }
    c2_met = consecutive_closes_above >= confirmation_closes or atr_break
    if atr_break:
        c2_how = (f"close {close:.2f} > SMA20 + "
                  f"{atr_mult}*ATR14 ({sma20 + atr_mult * atr14:.2f})")
    else:
        c2_how = (f"{min(consecutive_closes_above, confirmation_closes)}"
                  f"/{confirmation_closes} consecutive closes above SMA20")
    c2 = {"name": "confirmation", "met": c2_met, "detail": c2_how,
          "consecutive_closes_above": consecutive_closes_above,
          "atr14": None if atr14 is None else round(atr14, 2)}

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

    slope_ok = sma20 >= sma20_5d_ago
    c4 = {
        "name": "slope",
        "met": slope_ok,
        "detail": f"SMA20 {sma20:.2f} vs {slope_lookback_days}d ago "
                  f"{sma20_5d_ago:.2f} ({'flat/rising' if slope_ok else 'falling'})",
    }

    c5 = {
        "name": "thesis",
        "met": bool(group_in_universe),
        "detail": thesis_detail if thesis_detail is not None else
                  f"group {'in' if group_in_universe else 'not in'} "
                  f"current universe",
    }

    conditions = {"1_trigger": c1, "2_confirmation": c2,
                  "3_regime_gate": c3, "4_slope": c4, "5_thesis": c5}
    all_five = all(c["met"] for c in conditions.values())

    state = next_state(prev_state, kind, above_now, all_five)

    ext_pct = (close - sma20) / sma20 * 100
    ext_atr = None if not atr14 else (close - sma20) / atr14

    # Extension guard (PER-508 item 20): a WATCHLIST name cannot go
    # RE_ENTRY_READY while extended beyond the ceiling. Conditions 1-2
    # confirm a reclaim NEAR the mean; a name that ran vertically away
    # from it passes them trivially. Holdings exempt.
    extension_guard = None
    if (kind == "watching" and state == RE_ENTRY_READY
            and ext_atr is not None and ext_atr > extension_guard_max):
        state = EXTENDED_HOLD
        extension_guard = (f"extension {round(ext_atr, 2)}×ATR > "
                           f"{extension_guard_max}× — re-entry suppressed")

    result = {
        "state": state,
        "conditions": conditions,
        "conditions_met": sum(1 for c in conditions.values() if c["met"]),
        "all_conditions_met": all_five,
        "extension_pct": round(ext_pct, 2),
        "extension_atr": None if ext_atr is None else round(ext_atr, 2),
    }
    if state == RE_ENTRY_READY and c3.get("mode") == "conditional":
        result["a_plus_only"] = True
    if extension_guard:
        result["extension_guard"] = extension_guard
    if not above_now:
        result["distance_to_sma20_pct"] = round(
            (close - sma20) / sma20 * 100, 2)
    return result


# ---------------------------------------------------------------------------
# The A+ setup grade as a pure function (D-011, PER-508 comment 11716 +
# both amendments). The ONLY implementation of the seven-row checklist and
# the grade algorithm — the engine and the Position Lab endpoint feed the
# same function (Lab law 1, D-010).
# ---------------------------------------------------------------------------

def grade_setup(*, all_conditions_met, extension_atr, close, sma5,
                up_close_since_swing_low, rsi14, quality_score,
                score_waived=False, breaker_status=None, runway_sessions=None,
                extension_guard_max=1.8, rsi_min=45.0, rsi_max=70.0,
                score_min=75.0, runway_min_sessions=15):
    """The D-011 seven-row setup grade on scalar inputs.

    Rows: (1) five mechanical conditions · (2) extension <= guard ·
    (3) approach filter: close > SMA5 AND >=1 up-close since the swing low ·
    (4) RSI in [rsi_min, rsi_max] · (5) quality score >= score_min (waived
    for index vehicles) · (6) group breaker clear · (7) earnings runway >=
    runway_min_sessions TRADING sessions STRICTLY before the print day
    (amendment 2 — the print is not runway; None runway = no known print =
    unbounded, passes).

    Grades: all seven -> "A+". Rows 1-2 fail -> "C" (mechanical/guard).
    Row 3 fail -> "C" in EVERY regime (amendment 1, the C-escalation
    clause: a B-graded knife would be a permitted knife in Trending,
    contradicting the doctrine's founding case). Rows 4-6 fail (or their
    data is unavailable — A+ must be PROVEN, never defaulted) -> "B" with
    the failing rows named.

    Returns {"grade", "rows": {seven row dicts}, "failing": [names],
    "reasons": "; ".join}.
    """
    rows = {}

    rows["1_conditions"] = {
        "met": bool(all_conditions_met),
        "detail": "five mechanical conditions "
                  + ("met" if all_conditions_met else "not met"),
    }

    ext_ok = extension_atr is not None and extension_atr <= extension_guard_max
    rows["2_extension"] = {
        "met": bool(ext_ok),
        "detail": (f"extension {extension_atr}×ATR <= {extension_guard_max}×"
                   if ext_ok else
                   f"extension {extension_atr}×ATR > {extension_guard_max}×"
                   if extension_atr is not None else
                   "extension unavailable (no ATR)"),
    }

    appr_unavailable = close is None or sma5 is None
    appr_ok = (not appr_unavailable and close > sma5
               and bool(up_close_since_swing_low))
    if appr_unavailable:
        appr_detail = "approach unavailable (no SMA5)"
    elif not close > sma5:
        appr_detail = f"close {close:.2f} below SMA5 {sma5:.2f} — arriving from above"
    elif not up_close_since_swing_low:
        appr_detail = "no up-close since the swing low — turn unconfirmed"
    else:
        appr_detail = f"close {close:.2f} > SMA5 {sma5:.2f} with up-close off the low"
    rows["3_approach"] = {"met": bool(appr_ok), "detail": appr_detail}

    rsi_ok = rsi14 is not None and rsi_min <= rsi14 <= rsi_max
    rows["4_rsi"] = {
        "met": bool(rsi_ok),
        "detail": (f"RSI {rsi14:.0f} in [{rsi_min:.0f}, {rsi_max:.0f}]"
                   if rsi_ok else
                   f"RSI {rsi14:.0f} outside [{rsi_min:.0f}, {rsi_max:.0f}]"
                   if rsi14 is not None else "RSI unavailable"),
    }

    if score_waived:
        rows["5_score"] = {"met": True, "waived": True,
                           "detail": "quality score waived (index vehicle)"}
    else:
        score_ok = quality_score is not None and quality_score >= score_min
        rows["5_score"] = {
            "met": bool(score_ok),
            "detail": (f"score {quality_score:.0f} >= {score_min:.0f}"
                       if score_ok else
                       f"score {quality_score:.0f} < {score_min:.0f}"
                       if quality_score is not None else "score unavailable"),
        }

    brk_ok = breaker_status == "clear"
    rows["6_breaker"] = {
        "met": bool(brk_ok),
        "detail": (f"group breaker {breaker_status}" if breaker_status
                   else "group breaker unknown (no group data)"),
    }

    if runway_sessions is None:
        rows["7_runway"] = {"met": True,
                            "detail": "no known earnings print — runway unbounded"}
    else:
        run_ok = runway_sessions >= runway_min_sessions
        rows["7_runway"] = {
            "met": bool(run_ok),
            "detail": f"{runway_sessions} sessions before the print "
                      f"({'≥' if run_ok else '<'} {runway_min_sessions})",
        }

    failing = [k for k, r in rows.items() if not r["met"]]
    if not rows["1_conditions"]["met"] or not rows["2_extension"]["met"]:
        grade = "C"
    elif not rows["3_approach"]["met"] and not appr_unavailable:
        # amendment 1 — C-escalation: a PROVEN knife is blocked in EVERY
        # regime. Approach data merely unavailable follows the
        # unavailable-data convention instead (B, row named) — amendment 1
        # escalates the knife, not the unknown (review finding).
        grade = "C"
    elif failing:
        grade = "B"
    else:
        grade = "A+"
    reasons = "; ".join(rows[k]["detail"] for k in failing)
    return {"grade": grade, "rows": rows, "failing": failing,
            "reasons": reasons}


def runway_sessions_before(print_date_iso, today=None):
    """Trading sessions in [today, print) — the amendment-2 convention:
    sessions strictly BEFORE the print day, counting today's (or the next)
    session as runway. np.busday_count reproduces the record's arithmetic
    exactly (MRNA evaluated 2026-07-12/13 vs a 2026-07-31 print -> 14).
    Weekday approximation (no exchange-holiday calendar — same convention
    as every other trading-day count in the repo). None in -> None out."""
    if not print_date_iso:
        return None
    today = today or datetime.date.today()
    try:
        pd_date = datetime.date.fromisoformat(str(print_date_iso))
    except (TypeError, ValueError):
        return None
    if pd_date < today:
        return None            # past print — the name re-qualified
    if pd_date == today:
        # THE PRINT DAY: zero sessions of runway — the maximal binary
        # straddle must FAIL the >=15 bar, never read as "no known print"
        # (review finding)
        return 0
    return int(np.busday_count(today.isoformat(), pd_date.isoformat()))


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
        self.extension_guard_max = cfg.get("extension_guard_max", 1.8)
        # D-011 A+ doctrine thresholds (ruled values; do not tune outside a
        # D-011 revisit). approach_swing_lookback is the one implementation
        # knob the ruling leaves open (Build 5 retests variants).
        ap = cfg.get("aplus", {}) or {}

        def _apnum(key, default):
            # a null/garbage config value must degrade to the ruled default,
            # never crash the grade (review finding: aplus.rsi_min: null
            # raised TypeError out of evaluate)
            v = ap.get(key, default)
            return float(v) if isinstance(v, (int, float)) \
                and not isinstance(v, bool) else float(default)
        self.aplus_cfg = {
            "rsi_min": _apnum("rsi_min", 45.0),
            "rsi_max": _apnum("rsi_max", 70.0),
            "score_min": _apnum("score_min", 75.0),
            "runway_min_sessions": int(_apnum("runway_min_sessions", 15)),
            "approach_swing_lookback": max(
                2, int(_apnum("approach_swing_lookback", 20))),
            "index_vehicles": set(ap.get("index_vehicles") or
                                  ["SPY", "QQQ", "IWM", "DIA", "RSP"]),
        }
        self.themes_cfg = config.get("themes", {}) or {}
        self.fetcher = fetcher
        os.makedirs(self.STATE_DIR, exist_ok=True)

    # ------------------------------------------------------------------
    # Pure per-day evaluation (also used by replay harnesses/tests)
    # ------------------------------------------------------------------

    def evaluate(self, entry: dict, kind: str, df, regime_state: str,
                 thesis_status: dict, prev_state: str,
                 grade_ctx: dict = None) -> dict:
        """
        Evaluate one ticker for the last bar of df.

        entry: the positions.json record. kind: "holding" | "watching".
        df: daily OHLC history up to and including the evaluation day.
        thesis_status: from _group_status() — condition 5's basis (D-007:
        group in current universe). prev_state: persisted state or None on
        first sight. grade_ctx (watchers): {breaker_status, next_earnings_date,
        score_waived, today} — the non-df grade inputs; when given, the
        D-011 grade is computed and attached.

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

        # All condition/state/guard decisions live in assess_position
        # (module-level pure, PER-508 item 24b) — evaluate only computes
        # the scalar inputs from price history and re-attaches df-derived
        # extras.
        result = assess_position(
            last_close, sma_now, sma_then, atr, consec_above, regime_state,
            thesis_status["met"], kind, prev_state=prev_state,
            thesis_detail=thesis_status["detail"],
            confirmation_closes=self.confirmation_closes,
            atr_mult=self.atr_mult,
            extension_guard_max=self.extension_guard_max,
            slope_lookback_days=self.slope_lookback)
        if thesis_status.get("no_group_mapping"):
            result["conditions"]["5_thesis"]["no_group_mapping"] = True

        state = result["state"]
        result.update({
            "close": round(last_close, 2),
            # short MAs (PER-508 producer amendment): display-only inputs
            # for the assessment technicals; finite whenever sma20 is
            # (their windows are subsets of sma20's)
            "sma5": round(float(close.rolling(5).mean().iloc[-1]), 2),
            "sma10": round(float(close.rolling(10).mean().iloc[-1]), 2),
            "sma20": round(sma_now, 2),
        })
        # The EXACT simulate inputs this evaluation used — the Position Lab
        # seeds from these verbatim (parity by construction; the display
        # fields above are rounded and can flip a branch at a boundary).
        result["assess_inputs"] = {
            "close": last_close,
            "sma20": sma_now,
            "sma20_5d_ago": sma_then,
            "atr14": atr,
            "consecutive_closes_above": consec_above,
            "regime_state": regime_state,
            "group_in_universe": thesis_status["met"],
            "kind": kind,
        }
        # Effective stop for held names (and the stop that just fired)
        if state in (HELD, EXIT_FIRED):
            result["stop"] = self._stop_for(entry, sma_now)

        # --- D-011 A+ grade (watchers; entry doctrine) -------------------
        # The ENTIRE grade block is guarded: a grade failure (malformed
        # aplus config, degenerate df, scoring error) must degrade to
        # grade-unavailable — it can never take down the state machine
        # (review finding).
        if grade_ctx is not None and kind == "watching":
            try:
                ap = self.aplus_cfg
                # approach filter inputs: unrounded SMA5 + at least one
                # up-close since the swing low (lowest close in the
                # trailing lookback)
                sma5_now = float(close.rolling(5).mean().iloc[-1])
                look = min(len(close), ap["approach_swing_lookback"])
                tail = close.iloc[-look:]
                low_pos = int(np.argmin(tail.to_numpy(dtype=float)))
                after = tail.iloc[low_pos:].to_numpy(dtype=float)
                up_close = bool((np.diff(after) > 0).any())
                # RSI-14 + quality score from the same df (deferred import —
                # signal_engine is heavy; same pattern as sanitize_for_json)
                rsi_v = None
                score_v = None
                try:
                    import sys
                    _parent = os.path.join(os.path.dirname(__file__), "..")
                    if _parent not in sys.path:
                        sys.path.insert(0, _parent)
                    from signal_engine import compute_rsi, score_stock
                    r = compute_rsi(close).iloc[-1]
                    rsi_v = float(r) if np.isfinite(r) else None
                    score_v, _sig, _det = score_stock(df)
                except Exception:
                    pass
                runway = runway_sessions_before(
                    grade_ctx.get("next_earnings_date"),
                    grade_ctx.get("today"))
                # extension for row 2: the UNROUNDED value the guard itself
                # compares — the rounded display field can disagree with
                # the guard exactly at the 1.8 boundary (review finding)
                ext_raw = None if not atr else (last_close - sma_now) / atr
                grade = grade_setup(
                    all_conditions_met=result["all_conditions_met"],
                    extension_atr=ext_raw,
                    close=last_close, sma5=sma5_now,
                    up_close_since_swing_low=up_close,
                    rsi14=rsi_v, quality_score=score_v,
                    score_waived=bool(grade_ctx.get("score_waived")),
                    breaker_status=grade_ctx.get("breaker_status"),
                    runway_sessions=runway,
                    extension_guard_max=self.extension_guard_max,
                    rsi_min=ap["rsi_min"], rsi_max=ap["rsi_max"],
                    score_min=ap["score_min"],
                    runway_min_sessions=ap["runway_min_sessions"])
                result["grade"] = grade
                # The EXACT grade inputs (lab seeding — parity by
                # construction)
                result["grade_inputs"] = {
                    "sma5": sma5_now,
                    "up_close_since_swing_low": up_close,
                    "rsi14": rsi_v,
                    "quality_score": score_v,
                    "score_waived": bool(grade_ctx.get("score_waived")),
                    "breaker_status": grade_ctx.get("breaker_status"),
                    "runway_sessions": runway,
                }
                # D-011 Q4 enforcement: HARD GATE under the conditional
                # regime (Choppy — READY already carries a_plus_only there;
                # Caution blocks via condition 3). Trending: advisory, B
                # permitted. No new machine state: READY *renders* blocked
                # (the EXTENDED_HOLD visual law) via this field.
                if result.get("a_plus_only") and grade["grade"] != "A+":
                    result["grade_gate"] = (
                        f"READY blocked — grade {grade['grade']} under "
                        f"Choppy (A+ required): "
                        f"{grade['reasons'] or 'see rows'}")
            except Exception as exc:
                result["grade_error"] = f"grade unavailable: {exc}"
                # Q4 fail-safe: under the conditional regime an ungradeable
                # READY must still BLOCK — A+ is proven, never presumed
                if result.get("a_plus_only"):
                    result["grade_gate"] = (
                        "READY blocked — grade unavailable under Choppy "
                        "(A+ required, cannot be proven)")
        return result

    def _next_state(self, prev, kind, above, all_five):
        """Delegates to the module-level pure state map (item 24b)."""
        return next_state(prev, kind, above, all_five)

    @staticmethod
    def _reclaim_state(above, all_five):
        return _reclaim(above, all_five)

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

    @staticmethod
    def _group_status(entry, group_map, selected_groups, weeks_by_group,
                      top_n=15):
        """Condition 5 basis (D-007 Phase 1): the ticker's GICS group is in
        the current active universe's selected groups. Resolution: manual
        'group' key on the positions.json row overrides the shared resolver
        map (same precedence as R28). Tickers with NO resolvable group pass
        but are FLAGGED no_group_mapping — never silently true (the old
        no_theme_mapping convention, carried forward)."""
        ticker = entry.get("ticker")
        group = entry.get("group") or group_map.get(ticker)
        if not group:
            if not group_map and not selected_groups:
                # TOTAL data outage (no universe AND no signals): the gate
                # must fail closed like the partial outage below, never
                # masquerade as a benign never-mapped pass (review finding —
                # the deeper outage must not OPEN the gate the shallower
                # one closes)
                return {"met": False, "group": None,
                        "weeks_in_universe": None,
                        "detail": "universe unavailable — thesis gate "
                                  "fails closed"}
            hint = entry.get("theme")
            return {"met": True, "no_group_mapping": True, "group": None,
                    "weeks_in_universe": None,
                    "detail": f"no group mapping ('{hint}') — "
                              f"thesis gate not applicable"}
        if not selected_groups:
            # universe artifact missing/empty: fail CLOSED (an outage must
            # not open the entry gate) — but say so honestly, never claim
            # the group ranked out of a universe that wasn't there
            return {"met": False, "group": group, "weeks_in_universe": None,
                    "detail": "universe unavailable — thesis gate fails "
                              "closed"}
        if group in selected_groups:
            return {"met": True, "group": group,
                    "weeks_in_universe": weeks_by_group.get(group),
                    "detail": f"group '{group}' in universe (top-{top_n})"}
        return {"met": False, "group": group,
                "weeks_in_universe": None,
                "detail": f"group '{group}' not in current universe "
                          f"(top-{top_n})"}

    # ------------------------------------------------------------------
    # Live compute: load positions + prior states, evaluate, persist, emit
    # ------------------------------------------------------------------

    def compute(self, regime_result: dict, theme_result: dict = None,
                qualified_active: list = None, emit_events: bool = True,
                universe_active: dict = None, group_map: dict = None) -> dict:
        """theme_result / qualified_active are accepted for caller
        back-compat but NO LONGER gate condition 5 (D-007 Phase 1) — the
        theme layer runs display-only in parallel. universe_active /
        group_map come from the runner (shared with R28); loaded from the
        data artifacts when absent."""
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

        # --- Condition-5 basis (D-007 Phase 1): the current universe ----
        if universe_active is None:
            universe_active = self._load_json(
                os.path.join(self.DATA_DIR, "universe_active.json"), {}) or {}
        signals = self._load_json(
            os.path.join(self.DATA_DIR, "signals.json"), {}) or {}
        if group_map is None:
            from .portfolio_rules import resolve_group_map
            group_map = resolve_group_map(universe_active, signals)
        selected_groups = set((universe_active.get("groups") or {}).keys())
        weeks_by_group = {
            name: g.get("weeks_in_universe")
            for name, g in (universe_active.get("groups") or {}).items()}
        # Group breakers (grade row 6) from the dashboard artifact — may lag
        # a Saturday rotation until Monday's signal run; unknown groups
        # grade honestly as "breaker unknown" (B-blocking, never A+).
        breaker_by_group = {
            g.get("name"): g.get("breaker_status")
            for g in signals.get("groups", []) or [] if g.get("name")}
        top_n = len(selected_groups) or 15
        regime_state = (regime_result or {}).get("regime", "Unknown")

        tickers, transitions = {}, []
        entries = [(h, "holding") for h in positions.get("holdings", []) or []
                   if isinstance(h, dict)] + \
                  [(w, "watching") for w in positions.get("watching", []) or []
                   if isinstance(w, dict)]

        # Earnings-calendar layer (PER-510, display-only): daily-cached map
        # for tracked names. Proximity matters most on HELD / RE_ENTRY_READY
        # (R8: no naked momentum exposure through binary catalysts).
        earnings_map = {}
        try:
            import sys
            _parent = os.path.join(os.path.dirname(__file__), "..")
            if _parent not in sys.path:
                sys.path.insert(0, _parent)
            from earnings_calendar import get_earnings_map, days_to_earnings
            earnings_map = get_earnings_map(
                [e.get("ticker") for e, _ in entries if e.get("ticker")])
            from earnings_calendar import _today_et
            grade_today = _today_et()
        except Exception as e:
            print(f"[framework] earnings calendar unavailable: {e}")
            days_to_earnings = lambda d: None
            grade_today = None

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
            thesis_status = self._group_status(entry, group_map,
                                               selected_groups,
                                               weeks_by_group, top_n)
            prev = (prev_states.get(ticker) or {}).get("state")
            grade_ctx = None
            if kind == "watching":
                grade_ctx = {
                    "breaker_status": breaker_by_group.get(
                        thesis_status.get("group")),
                    "next_earnings_date": earnings_map.get(ticker),
                    "score_waived": (ticker in self.aplus_cfg["index_vehicles"]
                                     or entry.get("vehicle") == "index"),
                    # ET calendar — the same clock days_to_earnings uses on
                    # the same row (review finding: date.today() on a UTC
                    # host counts runway from tomorrow between 8pm-midnight)
                    "today": grade_today,
                }
            result = self.evaluate(entry, kind, df, regime_state,
                                   thesis_status, prev, grade_ctx=grade_ctx)
            result["ticker"] = ticker
            result["kind"] = kind
            result["theme"] = entry.get("theme")
            result["group"] = thesis_status.get("group")
            result["weeks_in_universe"] = thesis_status.get("weeks_in_universe")
            result["note"] = entry.get("note")
            result["next_earnings_date"] = earnings_map.get(ticker)
            result["days_to_earnings"] = days_to_earnings(earnings_map.get(ticker))
            dte = result["days_to_earnings"]
            if dte is not None and dte <= 7 and result["state"] in (HELD, RE_ENTRY_READY):
                result["earnings_note"] = (
                    f"earnings in {dte}d — R8: binary catalyst window")
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
        if result.get("extension_guard"):
            detail["extension_guard"] = result["extension_guard"]
        stop = result.get("stop")
        if state == EXIT_FIRED and stop:
            detail["stop"] = stop.get("detail")
        desc_from = prev if prev else "untracked"
        return {
            "timestamp": _utcnow().isoformat(),
            "type": "position_state_change",
            "severity": _SEVERITY.get(state, "medium"),
            "group": result.get("group") or entry.get("theme"),
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
