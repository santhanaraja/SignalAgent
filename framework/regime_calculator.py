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


class RegimeCalculator:
    """Computes the 3-voter swing regime score and outputs current regime state."""

    def __init__(self, config: dict, fetcher):
        """
        Args:
            config: Full framework config dict (loaded from config.yaml)
            fetcher: function(ticker, period) -> DataFrame with OHLCV data
        """
        self.regime_cfg = config["regime"]
        self.fetcher = fetcher

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

        base_state = self._determine_state(risk_on_count, risk_off_count,
                                           unavailable_count)

        # --- Backdrop gate: cap risk-on states below the 200DMA ---
        # Gate on the RAW pct (display value is rounded to 2dp; a close a
        # hair below the MA must still shut the gate). Non-finite or
        # missing data fails CLOSED: unknown is not permitted.
        spy_pct = spy_gauge.get("pct_raw", spy_gauge.get("value"))
        if spy_pct is None or not np.isfinite(spy_pct):
            gate_open, gate_reason = False, "data_unavailable"
        elif spy_pct < 0:
            gate_open, gate_reason = False, "below_200dma"
        else:
            gate_open, gate_reason = True, None

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
        for trigger in self.regime_cfg["change_protocol"].get("intra_week_override_triggers", []):
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

            if vix_5d <= cfg["risk_on_threshold"]:
                signal = "risk_on"
            elif vix_5d <= cfg["caution_threshold"]:
                signal = "caution"
            else:
                signal = "risk_off"

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

            if current_pctile >= 60:
                signal = "risk_on"
            elif current_pctile >= 40:
                signal = "caution"
            else:
                signal = "risk_off"

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
                    if spread <= cfg["risk_on_threshold"]:
                        signal = "risk_on"
                    elif spread <= cfg["caution_threshold"]:
                        signal = "caution"
                    else:
                        signal = "risk_off"

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

            if ratio_change > 0.5:
                signal = "risk_on"
            elif ratio_change > -0.5:
                signal = "caution"
            else:
                signal = "risk_off"

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
