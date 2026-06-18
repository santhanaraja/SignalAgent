#!/usr/bin/env python3
"""
Regime Calculator — Computes the 5-gauge regime score and determines
the current market regime state (Risk-on Trending / Choppy / Caution / Risk-off).

Uses EOD close prices only. No intraday noise.
"""

import datetime
import numpy as np


class RegimeCalculator:
    """Computes the 5-gauge regime score and outputs current regime state."""

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
        Compute regime from all 5 gauges.

        Args:
            regime_history: list of past regime states (for consecutive-Sunday tracking)

        Returns dict with regime state, gauge details, and action recommendation.
        """
        gauges = {}
        gauge_cfgs = self.regime_cfg["gauges"]

        # --- Gauge 1: SPY vs 200DMA ---
        gauges["spy_vs_200dma"] = self._compute_spy_200dma(gauge_cfgs["spy_vs_200dma"])

        # --- Gauge 2: VIX 5-day average ---
        gauges["vix_5d_avg"] = self._compute_vix_5d(gauge_cfgs["vix_5d_avg"])

        # --- Gauge 3: HY credit spread ---
        gauges["hy_spread"] = self._compute_hy_spread(gauge_cfgs["hy_spread"])

        # --- Gauge 4: S&P 500 breadth (% above 50DMA) ---
        gauges["breadth"] = self._compute_breadth(gauge_cfgs["breadth"])

        # --- Gauge 5: Yield curve (30Y-2Y) ---
        gauges["yield_curve"] = self._compute_yield_curve(gauge_cfgs["yield_curve"])

        # --- Determine regime state ---
        signals = [g["signal"] for g in gauges.values()]
        risk_on_count = signals.count("risk_on")
        caution_count = signals.count("caution")
        risk_off_count = signals.count("risk_off")
        unavailable_count = signals.count("unavailable")

        regime_state = self._determine_state(risk_on_count, caution_count, risk_off_count)

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
        intra_week_override = None
        for trigger in self.regime_cfg["change_protocol"].get("intra_week_override_triggers", []):
            gauge_name = trigger["gauge"]
            if gauge_name in gauges and gauges[gauge_name]["value"] is not None:
                val = gauges[gauge_name]["value"]
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
            "date": datetime.date.today().isoformat(),
            "regime": regime_state,
            "regime_color": color,
            "gauges": gauges,
            "risk_on_count": risk_on_count,
            "caution_count": caution_count,
            "risk_off_count": risk_off_count,
            "unavailable_count": unavailable_count,
            "regime_change_pending": regime_change_pending,
            "consecutive_weeks_at_state": consecutive_weeks,
            "confirmations_needed": confirmations_needed,
            "intra_week_override": intra_week_override,
            "action": action,
        }

    def _determine_state(self, risk_on, caution, risk_off):
        """
        Map gauge signal counts to a regime state.
        Pure function of current gauge signals — no history anchoring.

        Priority order (check best and worst first, fall through to middle):
          1. Risk-off   — 2+ risk_off gauges (most severe, checked first)
          2. Risk-on/Trending — 4+ risk_on, 0 risk_off
          3. Risk-on/Choppy  — 2+ risk_on, 0 risk_off
          4. Caution    — everything else (default)
        """
        # 1. Risk-off: severe downturn signals dominate
        if risk_off >= 2:
            return "Risk-off"

        # 2. Risk-on / Trending: strong broad-based risk appetite
        if risk_on >= 4 and risk_off == 0:
            return "Risk-on / Trending"

        # 3. Risk-on / Choppy: moderate risk appetite, some caution
        if risk_on >= 2 and risk_off == 0:
            return "Risk-on / Choppy"

        # 4. Caution: anything with 1 risk_off or insufficient risk_on
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
        Primary: FRED CSV download (no API key required).
        Secondary: FRED API (if FRED_API_KEY is set).
        Fallback: ^TYX (30Y yield) from yfinance (2Y approximated).

        Thresholds (in percentage points):
          > 0.50 (50bp)  = risk_on  (healthy steepening)
          0 to 0.50      = caution  (flat)
          < 0            = risk_off (inverted)
        """
        # Try FRED CSV (no API key needed)
        yc = self._try_fred_csv_yield_curve(cfg)
        if yc is not None:
            return yc

        # Try FRED API (needs key)
        yc = self._try_fred_api_yield_curve(cfg)
        if yc is not None:
            return yc

        # Last resort: ^TYX from yfinance for 30Y yield
        return self._try_yfinance_yield_curve(cfg)

    def _classify_yield_spread(self, spread, dgs30, dgs2, source, date_str=""):
        """Common spread classification for all yield curve sources."""
        spread_bp = round(spread * 100)  # basis points for display

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
        return {
            "value": round(spread, 4),
            "signal": signal,
            "detail": f"30Y-2Y spread: {spread_bp}bp ({dgs30:.2f}% - {dgs2:.2f}%) — {trend}{date_note}",
            "source": source,
            "spread_bp": spread_bp,
            "dgs30": round(dgs30, 2),
            "dgs2": round(dgs2, 2),
        }

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
        """Last resort: ^TYX for 30Y yield from yfinance."""
        try:
            # ^TYX = CBOE 30-Year Treasury Yield (in %, e.g. 4.85 = 4.85%)
            tyx_df = self.fetcher("^TYX", period="1mo")
            if tyx_df is not None and len(tyx_df) >= 1:
                yield_30y = float(tyx_df["Close"].iloc[-1])

                # No direct 2Y ticker on yfinance. Estimate from short-term proxy.
                # ^IRX = 13-week T-bill rate. Rough approximation for short end.
                irx_df = self.fetcher("^IRX", period="1mo")
                if irx_df is not None and len(irx_df) >= 1:
                    yield_short = float(irx_df["Close"].iloc[-1])
                    spread = yield_30y - yield_short
                    return self._classify_yield_spread(
                        spread, yield_30y, yield_short,
                        "yfinance (^TYX-^IRX approx)"
                    )
                else:
                    # Can only report 30Y level, no spread
                    return {
                        "value": round(yield_30y, 2),
                        "signal": "caution",
                        "detail": f"30Y yield: {yield_30y:.2f}% (no 2Y data for spread calc)",
                        "source": "yfinance ^TYX only",
                    }

            return {"value": None, "signal": "unavailable", "detail": "Yield curve data unavailable"}
        except Exception as e:
            return {"value": None, "signal": "unavailable", "detail": f"Yield curve error: {e}"}
