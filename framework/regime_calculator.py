#!/usr/bin/env python3
"""
Regime Calculator — SWING-horizon regime gauge (weeks-long holds, checked daily).

Three fast voters (VIX 5d avg, HY credit spread, breadth) vote
risk_on / caution / risk_off. SPY vs its 200DMA is a binary backdrop GATE,
not a voter: any close below the 200DMA caps risk-on states at Caution.
The yield curve (a months-horizon macro input) is computed but does NOT
vote — its output is routed to macro_inputs for future consumers
(Gauge B, WFC-specific logic).

State names are unchanged and load-bearing downstream
(framework page, rule engine, theme entry gates):
Risk-on / Trending, Risk-on / Choppy, Caution, Risk-off.
"""

import datetime
import numpy as np


def _today():
    """Today's date — module-level so tests can inject replay dates."""
    return datetime.date.today()


# ---------------------------------------------------------------------------
# The regime decision as pure functions (Build 4 step 0)
#
# The ONLY implementation of the vote thresholds and the state ladder.
# Production (RegimeCalculator below) delegates every decision here; the
# walk-forward backtest and the Gauge Lab import these same functions —
# a second copy of the ladder must be structurally impossible (Score Lab
# precedent). Threshold defaults mirror framework/config.yaml; production
# passes its config values explicitly, and so does the backtest CLI.
# ---------------------------------------------------------------------------

def _finite(v):
    return isinstance(v, (int, float)) and np.isfinite(v)


def vix_vote(vix_5d, risk_on_max=18.0, caution_max=22.0):
    """VIX 5-day-average voter (lower = calmer = risk-on)."""
    if not _finite(vix_5d):
        return "unavailable"
    if vix_5d <= risk_on_max:
        return "risk_on"
    if vix_5d <= caution_max:
        return "caution"
    return "risk_off"


def hy_oas_vote(oas_pct, risk_on_max=3.0, caution_max=4.0):
    """HY credit voter, primary basis: FRED BAMLH0A0HYM2 OAS in pct."""
    if not _finite(oas_pct):
        return "unavailable"
    if oas_pct <= risk_on_max:
        return "risk_on"
    if oas_pct <= caution_max:
        return "caution"
    return "risk_off"


def hy_percentile_vote(pctile, risk_on_min=60.0, caution_min=40.0):
    """HY credit voter, fallback basis: HYG/IEF ratio 60d percentile."""
    if not _finite(pctile):
        return "unavailable"
    if pctile >= risk_on_min:
        return "risk_on"
    if pctile >= caution_min:
        return "caution"
    return "risk_off"


def breadth_ratio_vote(breadth_20d_pct, band=0.5):
    """Breadth voter, RSP/SPY basis: current equal/cap-weight ratio vs the
    ratio of the 20d means, as a % change. Production reality: this
    'fallback' is the live path (the S5FI index is not fetchable)."""
    if not _finite(breadth_20d_pct):
        return "unavailable"
    if breadth_20d_pct > band:
        return "risk_on"
    if breadth_20d_pct > -band:
        return "caution"
    return "risk_off"


def gate_from_pct(spy_vs_200dma_pct):
    """Backdrop gate from SPY's raw % distance to its 200DMA.
    (open, reason) — fails CLOSED on missing/non-finite data."""
    if not _finite(spy_vs_200dma_pct):
        return False, "data_unavailable"
    if spy_vs_200dma_pct < 0:
        return False, "below_200dma"
    return True, None


def ladder_state(risk_on, risk_off, unavailable=0):
    """
    Map the 3 voter signals to a regime state (before the backdrop gate).
    Pure function of current voter signals — no history anchoring.

    Ladder on the risk_on tally:
      3 -> Risk-on / Trending, 2 -> Risk-on / Choppy,
      1 -> Caution, 0 -> Risk-off.
    Qualifiers:
      - risk_off votes are heavier than caution: ANY risk_off vote caps
        the state at Caution regardless of the tally (credit
        deteriorating while vol is calm must never print risk-on).
      - Risk-off requires 0 risk_on votes with all voters reporting,
        OR 2+ risk_off votes.
      - 2+ voters unavailable -> Caution: a one-voter tally is not
        evidence, in either direction.
    Order matters: 2 confirmed risk_off dominate missing data, and the
    unavailable guard must precede the 0-risk_on floor so a data outage
    can never print Risk-off.
    """
    if risk_off >= 2:
        return "Risk-off"
    if unavailable >= 2:
        return "Caution"
    if risk_on == 0 and unavailable == 0:
        return "Risk-off"
    if risk_off >= 1:
        return "Caution"
    if risk_on == 3:
        return "Risk-on / Trending"
    if risk_on == 2:
        return "Risk-on / Choppy"
    return "Caution"


def compute_regime(vix_5d, hy_oas_pct, breadth_20d_pct, spy_vs_200dma_pct,
                   *, vix_thresholds=(18.0, 22.0), oas_thresholds=(3.0, 4.0),
                   breadth_band=0.5):
    """The full regime decision on canonical primary-source inputs.

    Returns {"state", "votes", "gate"} exactly as production would decide
    for these gauge readings (primary HY basis: OAS absolute thresholds).
    Threshold keywords exist for the backtest sensitivity grid — defaults
    are the production config values.
    """
    votes = {
        "vix_5d_avg": vix_vote(vix_5d, *vix_thresholds),
        "hy_spread": hy_oas_vote(hy_oas_pct, *oas_thresholds),
        "breadth": breadth_ratio_vote(breadth_20d_pct, breadth_band),
    }
    signals = list(votes.values())
    base = ladder_state(signals.count("risk_on"), signals.count("risk_off"),
                        signals.count("unavailable"))
    gate_open, gate_reason = gate_from_pct(spy_vs_200dma_pct)
    capped = (not gate_open) and base.startswith("Risk-on")
    return {
        "state": "Caution" if capped else base,
        "votes": votes,
        "gate": {"open": gate_open, "capped": capped, "reason": gate_reason},
    }


# ---------------------------------------------------------------------------
# Gauge B trend chassis (D-008) — pure functions, transcribed VERBATIM from
# the validated analysis code (scripts/backtest_gauge_b.py, campaign 7904bd6 +
# throttle frontier 6603862; calibration locked per PER-508 comment 11724).
#
# 200DMA trend decides direction; VIX / HY-percentile / breadth are THROTTLE
# modifiers that scale exposure within the trend and never override it.
# Asymmetric hysteresis: downgrades instant, upgrades need N consecutive
# closes. Every default below is the locked Quality-point calibration —
# production passes config values explicitly (config.yaml regime.chassis).
#
# test_gauge_chassis.py pins this transcription against the backtest
# function itself: identical state sequences on identical inputs.
# ---------------------------------------------------------------------------

CHASSIS_STATES = ["Out-Risk-off", "Out-Defensive", "In-Trend-Throttled", "In-Trend-Full"]
CHASSIS_RANK = {s: i for i, s in enumerate(CHASSIS_STATES)}   # higher = more risk-on

# Chassis state -> the four load-bearing production regime labels. The label
# then rides the EXISTING R28 ceiling map (portfolio_rules.REGIME_CEILINGS:
# Trending 90 / Choppy 50 / Caution 25 / Risk-off 5) — the D-008 Q4 ladder
# needs no second implementation.
CHASSIS_TO_REGIME = {
    "In-Trend-Full": "Risk-on / Trending",
    "In-Trend-Throttled": "Risk-on / Choppy",
    "Out-Defensive": "Caution",
    "Out-Risk-off": "Risk-off",
}

# Q4 exposure ladder (fractions) — kept for the chassis block / Gauge Lab
# display; the dollar ceiling itself comes from portfolio_rules.
CHASSIS_LADDER = {"In-Trend-Full": 0.90, "In-Trend-Throttled": 0.50,
                  "Out-Defensive": 0.25, "Out-Risk-off": 0.05}


def chassis_raw_state(trend_in, vix_5d, hy_stress, breadth_20d, *,
                      vix_thr=22.0, breadth_thr=-0.5, require_k=1):
    """Direction from the 200DMA chassis; throttles scale WITHIN it, never
    override. In-trend: Throttled once >= require_k throttles fire, else Full.
    Out-of-trend: defensive by default, Risk-off on a real stress trigger
    (elevated vol OR credit stress). Transcribed from _raw_chassis_state;
    defaults are the locked Quality point (k1 / vix22 / br-0.5)."""
    vix_stress = vix_5d >= vix_thr
    breadth_stress = breadth_20d < breadth_thr
    if trend_in:
        stress = int(vix_stress) + int(hy_stress) + int(breadth_stress)
        return "In-Trend-Full" if stress < require_k else "In-Trend-Throttled"
    return "Out-Risk-off" if (vix_stress or hy_stress) else "Out-Defensive"


def new_chassis_hysteresis(state="Out-Defensive"):
    """Fresh hysteresis carry (the backtest's replay seed)."""
    return {"confirmed": state, "up": 0, "down": 0}


def chassis_step(trend_state, vix_5d, hy_stress, breadth_20d, hysteresis_state,
                 *, n=2, mode="asymmetric", vix_thr=22.0, breadth_thr=-0.5,
                 require_k=1):
    """One close-T step of the Gauge B chassis. Returns the confirmed state,
    its ladder exposure, the raw (pre-hysteresis) state, and the updated
    carry. Transcribed from compute_regime_chassis: upgrades count consecutive
    closes ranked above the confirmed state and jump to the CURRENT raw after
    N; downgrades are instant under asymmetric mode (the crash brake).

    trend_state : SPY %-above-200DMA (in-trend iff >= 0) — the chassis.
    hy_stress   : bool — is credit stressed? (production derives it from the
                  OAS 60d trailing percentile >= hy_pctile_cut; the pure step
                  stays shape-agnostic exactly like the backtest's.)
    """
    raw = chassis_raw_state(trend_state >= 0, vix_5d, hy_stress, breadth_20d,
                            vix_thr=vix_thr, breadth_thr=breadth_thr,
                            require_k=require_k)
    raw_rank = CHASSIS_RANK[raw]

    h = dict(hysteresis_state)
    cr = CHASSIS_RANK[h["confirmed"]]
    if raw_rank > cr:                       # candidate upgrade — slow to re-risk
        h["up"] += 1
        h["down"] = 0
        if h["up"] >= n:
            h["confirmed"] = raw
            h["up"] = 0
    elif raw_rank < cr:                     # downgrade
        h["down"] += 1
        h["up"] = 0
        if mode == "asymmetric":            # instant crash brake (jump to raw)
            h["confirmed"] = raw
            h["down"] = 0
        elif h["down"] >= n:                # symmetric control: confirm the drop
            h["confirmed"] = raw
            h["down"] = 0
    else:
        h["up"] = h["down"] = 0
    return {"state": h["confirmed"], "exposure": CHASSIS_LADDER[h["confirmed"]],
            "raw_state": raw, "hysteresis_state": h}


def pctile_of_last(values, window):
    """Percentile (0-100) of the LAST value within the trailing `window`
    values inclusive — the backtest's _rolling_pctile_of_last semantics
    ((a <= a[-1]).mean() * 100, min_periods=window) for a plain sequence.
    None when fewer than `window` observations exist OR any value in the
    positional window is non-finite — a NaN inside the window must read as
    no-signal (the rolling min_periods semantics), never silently reach
    further back in time (review finding)."""
    vals = list(values)
    if len(vals) < window:
        return None
    tail = vals[-window:]
    if any(v is None or (isinstance(v, float) and not np.isfinite(v))
           for v in tail):
        return None
    last = tail[-1]
    return float(sum(1 for v in tail if v <= last) / window * 100.0)


def replay_chassis(trend_seq, vix_seq, hy_stress_seq, breadth_seq, *,
                   n=2, mode="asymmetric", vix_thr=22.0, breadth_thr=-0.5,
                   require_k=1, seed_state="Out-Defensive"):
    """Causal replay over parallel per-close sequences -> (confirmed-state
    list, final carry, final raw). The backtest's replay_states for plain
    sequences; production re-derives today's chassis state by replaying the
    trailing window every run — gap-proof, intraday-idempotent (one step per
    daily bar by construction), and seed-independent within ~3 steps
    (downgrades adopt raw instantly; upgrades converge after N closes)."""
    h = new_chassis_hysteresis(seed_state)
    states, raw = [], None
    for t, v, s, b in zip(trend_seq, vix_seq, hy_stress_seq, breadth_seq):
        r = chassis_step(t, v, bool(s), b, h, n=n, mode=mode, vix_thr=vix_thr,
                         breadth_thr=breadth_thr, require_k=require_k)
        h = r["hysteresis_state"]
        raw = r["raw_state"]
        states.append(r["state"])
    return states, h, raw


def artifact_schema(regime_cfg):
    """The framework.json schema tag for the configured engine. Serve guard
    and runner both derive from this so a flag flip forces the committed
    artifact to regenerate rather than serving the other engine's output as
    current (the R28 artifact-baking lesson)."""
    engine = (regime_cfg or {}).get("engine", "chassis")
    return "regime-b-chassis" if engine == "chassis" else "regime-1a-3voter"


def fetch_oas_series(hy_cfg, limit=250):
    """FRED OAS observation series (ascending pandas Series indexed by date)
    for the chassis percentile — the HY voter's FRED plumbing but a SERIES,
    not a scalar (pctile_60d needs 60 trailing obs). None when the key is
    absent or the fetch fails. Shared by production and the Gauge Lab's
    OAS->percentile conversion."""
    import os
    fred_key = os.environ.get("FRED_API_KEY")
    if not fred_key:
        return None
    try:
        import pandas as pd
        import requests
        series = (hy_cfg or {}).get("fred_series", "BAMLH0A0HYM2")
        resp = requests.get(
            "https://api.stlouisfed.org/fred/series/observations",
            params={"series_id": series, "api_key": fred_key,
                    "file_type": "json", "sort_order": "desc",
                    "limit": limit},
            timeout=10)
        if resp.status_code != 200:
            return None
        obs = resp.json().get("observations", [])
        dates, vals = [], []
        for o in obs:
            v = o.get("value", ".")
            if v != ".":
                try:
                    vals.append(float(v))
                    dates.append(o.get("date"))
                except (TypeError, ValueError):
                    continue
        if len(vals) < 5:
            return None
        return pd.Series(vals, index=pd.to_datetime(dates)).sort_index()
    except Exception:
        return None


class RegimeCalculator:
    """Computes the 3-voter swing regime score and outputs current regime state."""

    STATE_DIR = None  # resolved lazily so tests can redirect it

    def __init__(self, config: dict, fetcher):
        """
        Args:
            config: Full framework config dict (loaded from config.yaml)
            fetcher: function(ticker, period) -> DataFrame with OHLCV data
        """
        self.regime_cfg = config["regime"]
        self.fetcher = fetcher
        if self.STATE_DIR is None:
            import os
            self.STATE_DIR = os.path.join(
                os.path.dirname(os.path.abspath(__file__)), "state")

    def compute(self, regime_history: list = None) -> dict:
        """
        Compute the swing regime: 3 voters + 200DMA backdrop gate.

        Voters: vix_5d_avg, hy_spread, breadth.
        Gate:   spy_vs_200dma — SPY below its 200DMA caps any risk-on state
                at Caution; never upgrades Risk-off. Fails closed (cap
                applies) when SPY data is unavailable: unknown ≠ permitted.
        Macro:  yield_curve — computed, never counted in the vote.

        Args:
            regime_history: list of past regime states (for consecutive-Sunday tracking)

        Returns dict with regime state, voter gauges, backdrop_gate,
        macro_inputs, and action recommendation.
        """
        gauge_cfgs = self.regime_cfg["gauges"]

        # --- Backdrop gate input: SPY vs 200DMA (not a voter) ---
        spy_gauge = self._compute_spy_200dma(gauge_cfgs["spy_vs_200dma"])

        # --- Voters ---
        gauges = {
            "vix_5d_avg": self._compute_vix_5d(gauge_cfgs["vix_5d_avg"]),
            "hy_spread": self._compute_hy_spread(gauge_cfgs["hy_spread"]),
            "breadth": self._compute_breadth(gauge_cfgs["breadth"]),
        }

        # --- Macro inputs: computed and published, never in the tally ---
        yield_curve = self._compute_yield_curve(gauge_cfgs["yield_curve"])
        yield_curve["role"] = "macro_input"
        macro_inputs = {"yield_curve": yield_curve}

        # --- Tally the 3 voters ---
        signals = [g["signal"] for g in gauges.values()]
        risk_on_count = signals.count("risk_on")
        caution_count = signals.count("caution")
        risk_off_count = signals.count("risk_off")
        unavailable_count = signals.count("unavailable")

        engine = self.regime_cfg.get("engine", "chassis")
        chassis_block = None

        if engine == "chassis":
            # --- Gauge B trend chassis (D-008) — the live path ---
            # 200DMA trend decides direction; the state comes from a causal
            # replay of the trailing window through the pure chassis step
            # (hysteresis included). The voter tally above is still computed
            # for display/back-compat but does NOT decide the state.
            chassis_block = self._compute_chassis()
            regime_state = chassis_block["regime"]
            trend_in = chassis_block.get("trend_in")
            # Trend-data outage mirrors the parliament's fail-closed gate
            # convention: capped + data_unavailable, so the degraded-week
            # streak logic treats an outage week as transparent evidence.
            outage = bool(chassis_block.get("degraded")) and \
                chassis_block.get("degraded_reason") in (
                    "data_unavailable", "data_unavailable_stale")
            backdrop_gate = {
                "gauge": "spy_vs_200dma",
                "role": "trend_chassis",
                "open": bool(trend_in) if trend_in is not None else False,
                "capped": outage,
                "reason": ("data_unavailable" if outage else
                           (None if trend_in else "below_200dma")),
                "value": spy_gauge.get("value"),
                "detail": spy_gauge.get("detail"),
                "price": spy_gauge.get("price"),
                "ma200": spy_gauge.get("ma200"),
            }
        else:
            # --- Build 1A parliament (kept intact for reversibility) ---
            base_state = self._determine_state(risk_on_count, risk_off_count,
                                               unavailable_count)

            # --- Backdrop gate: cap risk-on states below the 200DMA ---
            # Gate on the RAW pct (display value is rounded to 2dp; a close a
            # hair below the MA must still shut the gate). Non-finite or
            # missing data fails CLOSED: unknown is not permitted.
            spy_pct = spy_gauge.get("pct_raw", spy_gauge.get("value"))
            gate_open, gate_reason = gate_from_pct(spy_pct)

            regime_state = base_state
            gate_capped = False
            if not gate_open and base_state.startswith("Risk-on"):
                regime_state = "Caution"
                gate_capped = True

            backdrop_gate = {
                "gauge": "spy_vs_200dma",
                "role": "backdrop_gate",
                "open": gate_open,
                "capped": gate_capped,
                "reason": gate_reason,
                "value": spy_gauge.get("value"),
                "detail": spy_gauge.get("detail"),
                "price": spy_gauge.get("price"),
                "ma200": spy_gauge.get("ma200"),
            }

        # --- Degraded-week streaks (R4 confirmation protocol) ---
        # Weekly close = latest entry per ISO week (ISO weeks end Sunday, so
        # the weekly close IS the Sunday review state). Consumed by
        # theme_ranker: qualification mutates only on confirmed degradation.
        degraded_incl_current, degraded_completed = self._degraded_week_streaks(
            regime_history, regime_state, backdrop_gate)

        # --- Check consecutive-Sunday history ---
        # Deduplicate by ISO week so daily runs don't inflate the count.
        # Each ISO week keeps only the latest entry.
        consecutive_weeks = 1
        regime_change_pending = False
        if regime_history:
            # Deduplicate history to one entry per ISO week (latest wins)
            weekly = {}
            for entry in regime_history:
                try:
                    d = datetime.date.fromisoformat(entry["date"])
                    iso_week = d.isocalendar()[:2]  # (year, week)
                    weekly[iso_week] = entry
                except (ValueError, KeyError):
                    continue
            # Sort by week descending
            sorted_weeks = sorted(weekly.items(), key=lambda x: x[0], reverse=True)

            if sorted_weeks:
                last_state = sorted_weeks[0][1].get("regime")
                if last_state == regime_state:
                    consecutive_weeks = 1  # this week
                    for _, entry in sorted_weeks:
                        if entry.get("regime") == regime_state:
                            consecutive_weeks += 1
                        else:
                            break
                else:
                    regime_change_pending = True
                    consecutive_weeks = 1

        confirmations_needed = self.regime_cfg["change_protocol"]["consecutive_confirmations_required"]

        # --- Check intra-week override triggers ---
        # Triggers may reference voters, the backdrop-gate input
        # (spy_vs_200dma), or macro inputs — config keeps working unchanged.
        trigger_lookup = dict(gauges)
        trigger_lookup["spy_vs_200dma"] = spy_gauge
        trigger_lookup.update(macro_inputs)
        intra_week_override = None
        # Intra-week overrides are a PARLIAMENT-era patch: the weekly-cadence
        # gauge needed an intraweek escape hatch. The chassis re-reads daily
        # with INSTANT downgrades — the override concept is redundant there
        # and its messages contradict the chassis state (e.g. "below 200DMA
        # override" while out-of-trend IS the state driver; review finding).
        triggers = [] if engine == "chassis" else \
            self.regime_cfg["change_protocol"].get(
                "intra_week_override_triggers", [])
        for trigger in triggers:
            gauge_name = trigger["gauge"]
            if gauge_name in trigger_lookup and trigger_lookup[gauge_name]["value"] is not None:
                val = trigger_lookup[gauge_name]["value"]
                cond = trigger["condition"]
                threshold = trigger["value"]
                triggered = False
                if cond == ">" and val > threshold:
                    triggered = True
                elif cond == "<" and val < threshold:
                    triggered = True
                elif cond == ">=" and val >= threshold:
                    triggered = True
                elif cond == "<=" and val <= threshold:
                    triggered = True
                if triggered:
                    intra_week_override = trigger["message"]
                    break

        # Find action text for the state
        action = ""
        color = "#8b949e"
        for state_def in self.regime_cfg["states"]:
            if state_def["name"] == regime_state:
                action = state_def["action"]
                color = state_def.get("color", "#8b949e")
                break

        return {
            "date": _today().isoformat(),
            "regime": regime_state,
            "regime_color": color,
            "engine": engine,
            "chassis": chassis_block,
            "gauges": gauges,
            "backdrop_gate": backdrop_gate,
            "macro_inputs": macro_inputs,
            "risk_on_count": risk_on_count,
            "caution_count": caution_count,
            "risk_off_count": risk_off_count,
            "unavailable_count": unavailable_count,
            "regime_change_pending": regime_change_pending,
            "consecutive_weeks_at_state": consecutive_weeks,
            "confirmations_needed": confirmations_needed,
            "consecutive_degraded_weeks": degraded_incl_current,
            "consecutive_degraded_weeks_completed": degraded_completed,
            "intra_week_override": intra_week_override,
            "action": action,
        }

    DEGRADED_STATES = ("Caution", "Risk-off")

    def _degraded_week_streaks(self, regime_history, current_state, current_gate):
        """
        Two consecutive-degraded-week counts over weekly closes (latest
        entry per ISO week; Caution/Risk-off = degraded):

        - incl_current: streak ending with the current run as this week's
          provisional close (0 if the current state is not degraded).
        - completed: streak over COMPLETED weeks only (current ISO week
          excluded) — the basis for a delayed weekly review, so a Monday
          catch-up reaches the same conclusion the missed Sunday would have.

        Weeks whose Caution came solely from a data_unavailable gate cap
        are transparent: they neither count nor break a streak (an outage
        is not evidence in either direction).
        """
        def outage_only(gate):
            return bool(gate) and gate.get("capped") \
                and gate.get("reason") == "data_unavailable"

        this_week = _today().isocalendar()[:2]
        weekly = {}
        for entry in regime_history or []:
            try:
                d = datetime.date.fromisoformat(entry["date"])
            except (KeyError, TypeError, ValueError):
                continue
            wk = d.isocalendar()[:2]
            if wk == this_week:
                continue
            cur = weekly.get(wk)
            if cur is None or entry["date"] > cur["date"]:
                weekly[wk] = entry

        completed = 0
        # Staleness guard: if the most recent completed weekly close is
        # older than the immediately-previous ISO week (pipeline outage),
        # the evidence is stale — do not confirm degradation across the
        # gap. isocalendar of "7 days ago" is exactly the previous week.
        prev_week = (_today() - datetime.timedelta(days=7)).isocalendar()[:2]
        if weekly and max(weekly.keys()) < prev_week:
            weekly = {}
        for wk in sorted(weekly.keys(), reverse=True):
            e = weekly[wk]
            if e.get("regime") in self.DEGRADED_STATES:
                if not outage_only(e.get("backdrop_gate")):
                    completed += 1
            else:
                break

        if current_state not in self.DEGRADED_STATES:
            incl_current = 0
        else:
            incl_current = completed + (0 if outage_only(current_gate) else 1)
        return incl_current, completed

    def _determine_state(self, risk_on, risk_off, unavailable=0):
        """Delegates to the module-level pure ladder (Build 4 step 0) —
        kept as a method for existing callers and tests."""
        return ladder_state(risk_on, risk_off, unavailable)

    # ==============================================================
    # Gauge B trend chassis (D-008) — data plumbing + per-run replay
    # ==============================================================

    def _chassis_cfg(self):
        cfg = self.regime_cfg.get("chassis") or {}
        return {
            "n": int(cfg.get("hysteresis_n", 2)),
            "mode": cfg.get("hysteresis_mode", "asymmetric"),
            "vix_thr": float(cfg.get("vix_throttle", 22.0)),
            "hy_cut": float(cfg.get("hy_pctile_cut", 90.0)),
            "hy_window": int(cfg.get("hy_pctile_window", 60)),
            "breadth_thr": float(cfg.get("breadth_throttle", -0.5)),
            "require_k": int(cfg.get("require_k", 1)),
            "replay_days": int(cfg.get("replay_days", 60)),
        }

    def _fetch_oas_series(self, cfg, limit=250):
        """Delegates to the module-level fetch_oas_series (shared with the
        Gauge Lab's OAS->percentile conversion). Kept as a method so tests
        can stub the instance."""
        return fetch_oas_series(cfg, limit=limit)

    @staticmethod
    def _daily_close(df):
        """Close series aligned on NAIVE normalized trading DATES. yfinance
        indexes daily bars at midnight of the EXCHANGE tz (^VIX is Chicago,
        SPY/RSP New York) — as instants those never align, so cross-series
        reindex must key on the date, exactly like the backtest's
        date-indexed cache. Intraday-appended rows normalize onto today's
        date; duplicates keep the last (latest) row."""
        c = df["Close"]
        idx = c.index
        if getattr(idx, "tz", None) is not None:
            idx = idx.normalize().tz_localize(None)
        else:
            idx = idx.normalize()
        c = c.copy()
        c.index = idx
        return c[~c.index.duplicated(keep="last")]

    def _chassis_inputs(self, ccfg):
        """Aligned per-close input series for the chassis replay, derived
        EXACTLY as the validated backtest's build_inputs derives them:
        master index = SPY trading days; VIX/RSP reindex+ffill; vix_5d =
        rolling(5) mean; breadth = (ratio / ratio-of-20d-means - 1) * 100;
        trend = % vs 200DMA; OAS as-of ffill onto SPY days then shift(1)
        (publication lag — T's print is not knowable at close T).

        Returns (frame, hy_meta) where frame has columns trend/vix_5d/
        breadth/hy_stress over the trailing valid window, or (None, meta)
        when the trend series itself is unavailable."""
        import pandas as pd
        hy_meta = {"basis": None, "pctile": None}

        spy = self.fetcher("SPY", period="2y")
        if spy is None or len(spy) < 210:
            return None, hy_meta
        # Phantom-bar guard (review finding): on market holidays / weekend
        # runs, the fetcher appends a synthetic "today" row (fast_info quote,
        # Volume 0) for a date that will never be a trading day — it must not
        # count as a hysteresis close. Real SPY bars always carry volume, and
        # a pre-market synthetic row isn't a close either (close-basis, R11).
        # Intraday partial bars during market hours have real volume and stay
        # (the documented provisional-intraday read).
        if "Volume" in spy.columns and len(spy) and \
                float(spy["Volume"].iloc[-1]) == 0.0:
            spy = spy.iloc[:-1]
        spy_c = self._daily_close(spy)
        trend = (spy_c / spy_c.rolling(200).mean() - 1.0) * 100.0

        vix = self.fetcher("^VIX", period="6mo")
        if vix is None or len(vix) < 10:
            return None, hy_meta
        vix_c = self._daily_close(vix).reindex(spy_c.index).ffill()
        vix_5d = vix_c.rolling(5).mean()

        rsp = self.fetcher("RSP", period="6mo")
        if rsp is None or len(rsp) < 25:
            return None, hy_meta
        rsp_c = self._daily_close(rsp).reindex(spy_c.index).ffill()
        ratio = rsp_c / spy_c
        ratio_20d = rsp_c.rolling(20).mean() / spy_c.rolling(20).mean()
        breadth = (ratio / ratio_20d - 1.0) * 100.0

        # --- credit stress series (Q2 pctile_60d) ---
        # NaN-transparency (review finding): an un-warmed rolling window must
        # yield NaN (the day drops out of the replay), never a silent
        # hy_stress=False — `(pct >= cut)` alone maps NaN -> False, which
        # would mask a real credit event exactly like the pre-warmup hazard
        # the backtest's warmed_hy_shapes was added to eliminate. The .where
        # keeps NaN rows NaN so the dropna below excludes them.
        hy_stress = None
        oas = self._fetch_oas_series(self.regime_cfg["gauges"]["hy_spread"])
        if oas is not None and len(oas) >= ccfg["hy_window"] + 5:
            oas_aligned = oas.sort_index().reindex(
                spy_c.index, method="ffill").shift(1)
            pct = oas_aligned.rolling(
                ccfg["hy_window"], min_periods=ccfg["hy_window"]).apply(
                lambda a: (a <= a[-1]).mean() * 100.0, raw=True)
            if pct.notna().any():
                hy_stress = (pct >= ccfg["hy_cut"]).where(pct.notna())
                valid = pct.dropna()
                hy_meta = {"basis": "fred_oas_pctile",
                           "pctile": round(float(valid.iloc[-1]), 1)}
        if hy_stress is None:
            # Fallback: HYG/IEF ratio percentile, INVERTED (low ratio = wide
            # spreads = stress) — the HY voter's own production fallback
            # basis applied to the chassis cut. NOT the basis the D-008
            # campaign validated — the block reports degraded on it. 1y
            # fetch: the 60d window needs ~120 rows to warm across the
            # 60-day replay tail (6mo left ~6 rows of margin).
            hy_cfg = self.regime_cfg["gauges"]["hy_spread"]
            hyg = self.fetcher(hy_cfg.get("fallback_long", "HYG"), period="1y")
            ief = self.fetcher(hy_cfg.get("fallback_short", "IEF"), period="1y")
            if hyg is not None and ief is not None and len(hyg) >= 30 \
                    and len(ief) >= 30:
                hyg_c = self._daily_close(hyg).reindex(spy_c.index).ffill()
                ief_c = self._daily_close(ief).reindex(spy_c.index).ffill()
                r = hyg_c / ief_c
                rp = r.rolling(ccfg["hy_window"],
                               min_periods=ccfg["hy_window"]).apply(
                    lambda a: (a <= a[-1]).mean() * 100.0, raw=True)
                if rp.notna().any():
                    hy_stress = (rp <= (100.0 - ccfg["hy_cut"])).where(rp.notna())
                    valid = rp.dropna()
                    hy_meta = {"basis": "hyg_ief_inverted_pctile",
                               "pctile": round(float(valid.iloc[-1]), 1)}

        f = pd.DataFrame({"trend": trend, "vix_5d": vix_5d,
                          "breadth": breadth})
        if hy_stress is not None:
            f["hy_stress"] = hy_stress
            f = f.dropna(subset=["trend", "vix_5d", "breadth", "hy_stress"])
        else:
            # Credit data fully unavailable: replay with hy_stress=False —
            # an outage can never *create* stress (or print Risk-off), the
            # same discipline as the parliament's unavailable votes. The
            # block carries degraded_reason so the outage is visible.
            f["hy_stress"] = False
            f = f.dropna(subset=["trend", "vix_5d", "breadth"])
            hy_meta["basis"] = None
        return f.tail(ccfg["replay_days"]), hy_meta

    def _compute_chassis(self):
        """Replay the trailing window through the pure chassis step and
        return the chassis block (state, throttles, hysteresis, replay
        provenance). On total input failure, serve the last recorded state
        from framework/state/regime_chassis_state.json, degraded."""
        import datetime as _dt
        import os
        import json as _json
        try:
            ccfg = self._chassis_cfg()
        except (TypeError, ValueError):
            # malformed chassis config must degrade, never crash the run
            ccfg = {"n": 2, "mode": "asymmetric", "vix_thr": 22.0,
                    "hy_cut": 90.0, "hy_window": 60, "breadth_thr": -0.5,
                    "require_k": 1, "replay_days": 60}
        state_path = os.path.join(self.STATE_DIR, "regime_chassis_state.json")

        try:
            frame, hy_meta = self._chassis_inputs(ccfg)
        except Exception:
            frame, hy_meta = None, {"basis": None, "pctile": None}

        if frame is None or len(frame) < 10:
            # data outage — fall back to the recorded state, degraded.
            # STALENESS BOUND (review finding): a record older than 7
            # calendar days decays to Out-Defensive — the system's
            # fail-closed outage convention (Caution ceiling), never a
            # weeks-old In-Trend-Full served as current.
            last = {}
            try:
                with open(state_path) as fh:
                    last = _json.load(fh)
            except (OSError, ValueError):
                pass
            confirmed = last.get("confirmed", "Out-Defensive")
            reason = "data_unavailable"
            if confirmed not in CHASSIS_RANK:
                confirmed = "Out-Defensive"
            try:
                age = (_today() - _dt.date.fromisoformat(
                    str(last.get("as_of")))).days
            except (TypeError, ValueError):
                age = None
            if age is None or age > 7:
                confirmed = "Out-Defensive"
                reason = "data_unavailable_stale"
            return {
                "engine": "chassis",
                "raw_state": last.get("raw_state"),
                "confirmed_state": confirmed,
                "regime": CHASSIS_TO_REGIME[confirmed],
                "exposure_ceiling_pct": CHASSIS_LADDER[confirmed] * 100.0,
                "trend_in": None,
                "throttles": None,
                "throttles_firing": None,
                "hysteresis": {"up": last.get("up", 0),
                               "down": last.get("down", 0),
                               "n": ccfg["n"], "mode": ccfg["mode"]},
                "replay": {"days_used": 0, "window": ccfg["replay_days"]},
                "hy_basis": None, "hy_pctile": None,
                "degraded": True, "degraded_reason": reason,
            }

        seqs = (
            [float(x) for x in frame["trend"].values],
            [float(x) for x in frame["vix_5d"].values],
            [bool(x) for x in frame["hy_stress"].values],
            [float(x) for x in frame["breadth"].values],
        )
        kw = dict(n=ccfg["n"], mode=ccfg["mode"], vix_thr=ccfg["vix_thr"],
                  breadth_thr=ccfg["breadth_thr"], require_k=ccfg["require_k"])
        states, carry, raw = replay_chassis(*seqs, **kw)
        confirmed = states[-1]
        # The carry BEFORE today's step — the Gauge Lab steps from THIS so
        # its untouched seed reproduces today's confirmed state instead of
        # double-stepping the current bar (review finding).
        _, carry_pre, _ = replay_chassis(*(s[:-1] for s in seqs), **kw)

        t = float(frame["trend"].iloc[-1])
        v = float(frame["vix_5d"].iloc[-1])
        b = float(frame["breadth"].iloc[-1])
        hs = bool(frame["hy_stress"].iloc[-1])
        vix_firing = v >= ccfg["vix_thr"]
        breadth_firing = b < ccfg["breadth_thr"]
        # Degraded whenever the credit throttle is NOT on the validated
        # basis (review finding): the D-008 campaign validated pctile_60d on
        # FRED OAS specifically — the HYG/IEF fallback keeps the gauge alive
        # but must be visibly flagged, never pass as healthy.
        degraded = hy_meta["basis"] != "fred_oas_pctile"
        degraded_reason = (None if not degraded else
                           ("hy_data_unavailable" if hy_meta["basis"] is None
                            else "hy_fallback_basis"))

        block = {
            "engine": "chassis",
            "raw_state": raw,
            "confirmed_state": confirmed,
            "regime": CHASSIS_TO_REGIME[confirmed],
            "exposure_ceiling_pct": CHASSIS_LADDER[confirmed] * 100.0,
            "trend_in": bool(t >= 0),
            "trend_pct": round(t, 2),
            "throttles": {
                "vix": {"firing": bool(vix_firing), "value": round(v, 2),
                        "cut": ccfg["vix_thr"]},
                "hy": {"firing": hs, "pctile": hy_meta["pctile"],
                       "cut": ccfg["hy_cut"], "basis": hy_meta["basis"]},
                "breadth": {"firing": bool(breadth_firing),
                            "value": round(b, 2),
                            "cut": ccfg["breadth_thr"]},
            },
            "throttles_firing": int(vix_firing) + int(hs) + int(breadth_firing),
            "hysteresis": {"up": int(carry["up"]), "down": int(carry["down"]),
                           "n": ccfg["n"], "mode": ccfg["mode"]},
            "replay": {"days_used": int(len(frame)),
                       "window": ccfg["replay_days"],
                       "start": str(frame.index[0].date()),
                       "end": str(frame.index[-1].date())},
            "hy_basis": hy_meta["basis"], "hy_pctile": hy_meta["pctile"],
            "degraded": degraded,
            "degraded_reason": degraded_reason,
        }

        # Record the result (observability + the outage fallback above).
        # Committed by CI's framework/state/ add — survives redeploys.
        # Atomic tmp+rename: a mid-write kill must never leave a truncated
        # record for the commit step to pick up (review finding).
        try:
            os.makedirs(self.STATE_DIR, exist_ok=True)
            tmp_path = state_path + ".tmp"
            with open(tmp_path, "w") as fh:
                _json.dump({
                    "as_of": str(frame.index[-1].date()),
                    "confirmed": confirmed,
                    "raw_state": raw,
                    "up": int(carry["up"]), "down": int(carry["down"]),
                    "carry_pre_final": {
                        "confirmed": carry_pre["confirmed"],
                        "up": int(carry_pre["up"]),
                        "down": int(carry_pre["down"]),
                    },
                    "replay_days_used": int(len(frame)),
                    "hy_basis": hy_meta["basis"],
                    "engine": "chassis",
                }, fh, indent=2)
            os.replace(tmp_path, state_path)
        except OSError:
            pass
        return block

    # ==============================================================
    # Individual gauge computations
    # ==============================================================

    def _compute_spy_200dma(self, cfg):
        """SPY price vs 200-day moving average."""
        try:
            df = self.fetcher(cfg["proxy"], period="1y")
            if df is None or len(df) < cfg["lookback_days"]:
                return {"value": None, "signal": "unavailable", "detail": "Insufficient SPY data"}

            close = df["Close"]
            current_price = float(close.iloc[-1])
            ma200 = float(close.rolling(cfg["lookback_days"]).mean().iloc[-1])
            pct_vs_ma = ((current_price - ma200) / ma200) * 100

            # A NaN close anywhere in the window poisons the rolling mean.
            # Non-finite must read as unavailable (the gate then fails
            # closed), never as an open gate or a silent risk_off vote.
            if not np.isfinite(pct_vs_ma):
                return {"value": None, "signal": "unavailable",
                        "detail": "SPY/200DMA computed non-finite value"}

            if pct_vs_ma >= cfg["risk_on_threshold"]:
                signal = "risk_on"
            elif pct_vs_ma >= cfg["caution_threshold"]:
                signal = "caution"
            else:
                signal = "risk_off"

            return {
                "value": round(pct_vs_ma, 2),
                "signal": signal,
                "detail": f"SPY ${current_price:.2f} vs 200DMA ${ma200:.2f} ({pct_vs_ma:+.1f}%)",
                "price": round(current_price, 2),
                "ma200": round(ma200, 2),
                # unrounded, for the gate comparison — round(-0.004, 2) is
                # -0.0 and would read as "at/above the MA"
                "pct_raw": float(pct_vs_ma),
            }
        except Exception as e:
            return {"value": None, "signal": "unavailable", "detail": f"Error: {e}"}

    def _compute_vix_5d(self, cfg):
        """VIX 5-day simple average."""
        try:
            df = self.fetcher(cfg["proxy"], period="1mo")
            if df is None or len(df) < cfg["lookback_days"]:
                return {"value": None, "signal": "unavailable", "detail": "Insufficient VIX data"}

            close = df["Close"]
            vix_5d = float(close.iloc[-cfg["lookback_days"]:].mean())
            current_vix = float(close.iloc[-1])

            # NaN closes in the window poison the mean. The old inline
            # branches voted risk_off on NaN (every <= comparison False) —
            # accidental, and with two NaN voters it printed Risk-off on a
            # pure data outage, violating the ladder's own contract. Vote
            # unavailable explicitly (mirrors the SPY gauge guard).
            if not np.isfinite(vix_5d):
                return {"value": None, "signal": "unavailable",
                        "detail": "VIX 5d average computed non-finite value"}

            signal = vix_vote(vix_5d, cfg["risk_on_threshold"],
                              cfg["caution_threshold"])

            return {
                "value": round(vix_5d, 2),
                "signal": signal,
                "detail": f"VIX 5d avg {vix_5d:.1f} (spot {current_vix:.1f})",
                "spot": round(current_vix, 2),
            }
        except Exception as e:
            return {"value": None, "signal": "unavailable", "detail": f"Error: {e}"}

    def _compute_hy_spread(self, cfg):
        """
        High-yield credit spread.
        Primary: FRED BAMLH0A0HYM2.
        Fallback: HYG/IEF price ratio as proxy (lower ratio ≈ wider spread).
        """
        # Try FRED first
        spread = self._try_fred_hy_spread(cfg)
        if spread is not None:
            return spread

        # Fallback: HYG/IEF ratio
        try:
            hyg_df = self.fetcher(cfg["fallback_long"], period="6mo")
            ief_df = self.fetcher(cfg["fallback_short"], period="6mo")

            if hyg_df is None or ief_df is None or len(hyg_df) < 20 or len(ief_df) < 20:
                return {"value": None, "signal": "unavailable", "detail": "HYG/IEF data unavailable"}

            hyg_price = float(hyg_df["Close"].iloc[-1])
            ief_price = float(ief_df["Close"].iloc[-1])
            ratio = hyg_price / ief_price

            # HYG/IEF ratio: higher = risk-on (HY outperforming treasuries)
            # Map to approximate spread levels
            # Typical ratio range: 0.70-0.85
            # We'll use the ratio's 60-day percentile as a signal
            hyg_closes = hyg_df["Close"].iloc[-60:].values if len(hyg_df) >= 60 else hyg_df["Close"].values
            ief_closes = ief_df["Close"].iloc[-60:].values if len(ief_df) >= 60 else ief_df["Close"].values
            min_len = min(len(hyg_closes), len(ief_closes))
            ratios = hyg_closes[-min_len:] / ief_closes[-min_len:]
            current_pctile = float(np.sum(ratios <= ratio) / len(ratios) * 100)

            signal = hy_percentile_vote(current_pctile)

            return {
                "value": round(ratio, 4),
                "signal": signal,
                "detail": f"HYG/IEF ratio {ratio:.3f} ({current_pctile:.0f}th percentile, 60d)",
                "source": "HYG/IEF fallback",
                "percentile": round(current_pctile, 1),
            }
        except Exception as e:
            return {"value": None, "signal": "unavailable", "detail": f"HY spread error: {e}"}

    def _try_fred_hy_spread(self, cfg):
        """Try fetching HY spread from FRED API."""
        import os
        fred_key = os.environ.get("FRED_API_KEY")
        if not fred_key:
            return None

        try:
            import requests
            series = cfg.get("fred_series", "BAMLH0A0HYM2")
            url = f"https://api.stlouisfed.org/fred/series/observations"
            params = {
                "series_id": series,
                "api_key": fred_key,
                "file_type": "json",
                "sort_order": "desc",
                "limit": 5,
            }
            resp = requests.get(url, params=params, timeout=10)
            if resp.status_code != 200:
                return None

            data = resp.json()
            obs = data.get("observations", [])
            if not obs:
                return None

            # Get latest non-empty value
            for o in obs:
                val = o.get("value", ".")
                if val != ".":
                    spread = float(val)
                    signal = hy_oas_vote(spread, cfg["risk_on_threshold"],
                                         cfg["caution_threshold"])

                    return {
                        "value": round(spread, 2),
                        "signal": signal,
                        "detail": f"HY spread {spread:.2f}% (FRED {series})",
                        "source": "FRED",
                        "date": o.get("date", ""),
                    }
            return None
        except Exception:
            return None

    def _compute_breadth(self, cfg):
        """S&P 500 breadth — % of stocks above 50DMA."""
        try:
            # Try S5FI (S&P 500 % above 50DMA index)
            df = self.fetcher(cfg["proxy"], period="3mo")
            if df is not None and len(df) > 5:
                breadth = float(df["Close"].iloc[-1])

                if breadth >= cfg["risk_on_threshold"]:
                    signal = "risk_on"
                elif breadth >= cfg["caution_threshold"]:
                    signal = "caution"
                else:
                    signal = "risk_off"

                return {
                    "value": round(breadth, 1),
                    "signal": signal,
                    "detail": f"{breadth:.1f}% of S&P 500 above 50DMA",
                    "source": "S5FI",
                }

            # Fallback: RSP/SPY ratio as breadth proxy
            rsp_df = self.fetcher(cfg.get("fallback_proxy", "RSP"), period="6mo")
            spy_df = self.fetcher(cfg.get("fallback_benchmark", "SPY"), period="6mo")

            if rsp_df is None or spy_df is None:
                return {"value": None, "signal": "unavailable", "detail": "Breadth data unavailable"}

            rsp_price = float(rsp_df["Close"].iloc[-1])
            spy_price = float(spy_df["Close"].iloc[-1])
            ratio = rsp_price / spy_price

            # RSP/SPY rising = broadening breadth (equal-weight outperforming cap-weight)
            rsp_20d = float(rsp_df["Close"].iloc[-20:].mean()) if len(rsp_df) >= 20 else rsp_price
            spy_20d = float(spy_df["Close"].iloc[-20:].mean()) if len(spy_df) >= 20 else spy_price
            ratio_20d = rsp_20d / spy_20d
            ratio_change = ((ratio - ratio_20d) / ratio_20d) * 100

            # same NaN discipline as the VIX gauge: unavailable, never a
            # silent risk_off from failed comparisons
            if not np.isfinite(ratio_change):
                return {"value": None, "signal": "unavailable",
                        "detail": "Breadth ratio computed non-finite value"}

            signal = breadth_ratio_vote(ratio_change)

            return {
                "value": round(ratio_change, 2),
                "signal": signal,
                "detail": f"RSP/SPY ratio change: {ratio_change:+.2f}% (20d) — breadth {'broadening' if ratio_change > 0 else 'narrowing'}",
                "source": "RSP/SPY fallback",
            }
        except Exception as e:
            return {"value": None, "signal": "unavailable", "detail": f"Breadth error: {e}"}

    def _compute_yield_curve(self, cfg):
        """
        Yield curve: 30Y-2Y Treasury spread in percentage points.
        Macro input only — computed and published, never in the swing vote.

        Primary: FRED API DGS30/DGS2 (requires FRED_API_KEY; same fetch
                 pattern as the HY spread gauge).
        Secondary: FRED public CSV (keyless; Akamai resets non-browser
                 clients as of 2026-07, kept in case it recovers).
        Fallback: yfinance ^TYX (30Y) minus ^IRX (13-week bill), labeled
                 "30Y-3mo". ^IRX is the 3-MONTH bill — it must never be
                 presented as the 2Y.

        Thresholds (in percentage points):
          > 0.50 (50bp)  = risk_on  (healthy steepening)
          0 to 0.50      = caution  (flat)
          < 0            = risk_off (inverted)
        """
        # Try FRED API first (needs key; the true DGS2 series)
        yc = self._try_fred_api_yield_curve(cfg)
        if yc is not None:
            return yc

        # Try FRED CSV (no API key needed)
        yc = self._try_fred_csv_yield_curve(cfg)
        if yc is not None:
            return yc

        # Last resort: ^TYX-^IRX from yfinance, honestly labeled
        return self._try_yfinance_yield_curve(cfg)

    def _classify_yield_spread(self, spread, long_yield, short_yield, source,
                               date_str="", short_label="2Y"):
        """Common spread classification for all yield curve sources.

        short_label names the short leg honestly: "2Y" only when the short
        leg is a true 2-year series (FRED DGS2); the ^IRX fallback passes
        "3mo". The dgs30/dgs2 keys are emitted only for true DGS data.
        """
        spread_bp = round(spread * 100)  # basis points for display
        label = f"30Y-{short_label}"

        if spread > 0.50:
            signal = "risk_on"
            trend = "positive"
        elif spread >= 0:
            signal = "caution"
            trend = "flat"
        else:
            signal = "risk_off"
            trend = "inverted"

        date_note = f", as of {date_str}" if date_str else ""
        result = {
            "value": round(spread, 4),
            "signal": signal,
            "detail": f"{label} spread: {spread_bp}bp ({long_yield:.2f}% - {short_yield:.2f}%) — {trend}{date_note}",
            "source": source,
            "spread_bp": spread_bp,
            "spread_label": label,
            "long_yield": round(long_yield, 2),
            "short_yield": round(short_yield, 2),
        }
        if short_label == "2Y":
            result["dgs30"] = round(long_yield, 2)
            result["dgs2"] = round(short_yield, 2)
        return result

    def _try_fred_csv_yield_curve(self, cfg):
        """Fetch 30Y and 2Y yields from FRED public CSV (no API key)."""
        import io
        try:
            import requests

            # FRED serves CSV at this endpoint without authentication
            url = "https://fred.stlouisfed.org/graph/fredgraph.csv"
            params = {"id": "DGS30,DGS2"}
            resp = requests.get(url, params=params, timeout=15, headers={
                "User-Agent": "Mozilla/5.0 SignalAgent/1.0",
            })
            if resp.status_code != 200:
                print(f"[framework] FRED CSV returned {resp.status_code}")
                return None

            # Parse CSV: columns are DATE, DGS30, DGS2
            import csv
            reader = csv.reader(io.StringIO(resp.text))
            header = next(reader, None)
            if not header or len(header) < 3:
                return None

            # Read rows in reverse to find the latest non-empty values
            rows = list(reader)
            dgs30_val = None
            dgs2_val = None
            date_str = ""

            for row in reversed(rows):
                if len(row) < 3:
                    continue
                date_str_candidate = row[0]
                v30 = row[1].strip() if len(row) > 1 else "."
                v2 = row[2].strip() if len(row) > 2 else "."

                if v30 != "." and v2 != "." and v30 and v2:
                    try:
                        dgs30_val = float(v30)
                        dgs2_val = float(v2)
                        date_str = date_str_candidate
                        break
                    except ValueError:
                        continue

            if dgs30_val is None or dgs2_val is None:
                print("[framework] FRED CSV: no valid yield data found")
                return None

            spread = dgs30_val - dgs2_val
            print(f"[framework] FRED CSV yields: 30Y={dgs30_val:.2f}%, 2Y={dgs2_val:.2f}%, spread={spread*100:.0f}bp ({date_str})")
            return self._classify_yield_spread(spread, dgs30_val, dgs2_val, "FRED", date_str)

        except Exception as e:
            print(f"[framework] FRED CSV yield curve failed: {e}")
            return None

    def _try_fred_api_yield_curve(self, cfg):
        """Fetch yields from FRED API (requires FRED_API_KEY env var)."""
        import os
        fred_key = os.environ.get("FRED_API_KEY")
        if not fred_key:
            return None

        try:
            import requests
            dgs30_val = None
            dgs2_val = None

            for series, target in [(cfg.get("fred_long", "DGS30"), "30y"), (cfg.get("fred_short", "DGS2"), "2y")]:
                url = "https://api.stlouisfed.org/fred/series/observations"
                params = {
                    "series_id": series,
                    "api_key": fred_key,
                    "file_type": "json",
                    "sort_order": "desc",
                    "limit": 5,
                }
                resp = requests.get(url, params=params, timeout=10)
                if resp.status_code != 200:
                    return None
                data = resp.json()
                for o in data.get("observations", []):
                    v = o.get("value", ".")
                    if v != ".":
                        if target == "30y":
                            dgs30_val = float(v)
                        else:
                            dgs2_val = float(v)
                        break

            if dgs30_val is None or dgs2_val is None:
                return None

            spread = dgs30_val - dgs2_val
            return self._classify_yield_spread(spread, dgs30_val, dgs2_val, "FRED API")

        except Exception:
            return None

    def _try_yfinance_yield_curve(self, cfg):
        """Last resort: ^TYX (30Y) minus ^IRX (13-week bill) from yfinance.

        There is no 2Y ticker on yfinance. ^IRX is the 3-MONTH bill, so this
        measures the 30Y-3mo spread — labeled as such, never as 30Y-2Y.
        """
        try:
            # ^TYX = CBOE 30-Year Treasury Yield (in %, e.g. 4.85 = 4.85%)
            tyx_df = self.fetcher("^TYX", period="1mo")
            if tyx_df is not None and len(tyx_df) >= 1:
                yield_30y = float(tyx_df["Close"].iloc[-1])

                # ^IRX = 13-week T-bill rate (3-month, NOT a 2Y proxy)
                irx_df = self.fetcher("^IRX", period="1mo")
                if irx_df is not None and len(irx_df) >= 1:
                    yield_short = float(irx_df["Close"].iloc[-1])
                    spread = yield_30y - yield_short
                    return self._classify_yield_spread(
                        spread, yield_30y, yield_short,
                        "yfinance (^TYX-^IRX)", short_label="3mo"
                    )
                else:
                    # 30Y level only — no short leg, so no spread. value
                    # must stay None: consumers read value as a SPREAD and
                    # a 30Y level (~5) would masquerade as a 500bp spread.
                    return {
                        "value": None,
                        "signal": "unavailable",
                        "detail": f"30Y yield {yield_30y:.2f}% only — no short-leg data for spread",
                        "source": "yfinance ^TYX only",
                        "long_yield": round(yield_30y, 2),
                    }

            return {"value": None, "signal": "unavailable", "detail": "Yield curve data unavailable"}
        except Exception as e:
            return {"value": None, "signal": "unavailable", "detail": f"Yield curve error: {e}"}
