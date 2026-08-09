#!/usr/bin/env python3
"""
Signal Engine — Pulls market data + fundamentals, computes technical indicators,
monitors thesis-breaker conditions, and generates buy/sell/hold signals.
"""

import json
import os
import sys
import datetime
import numpy as np
import pandas as pd


class NumpyEncoder(json.JSONEncoder):
    """Handle numpy types in JSON serialization."""
    def default(self, obj):
        if isinstance(obj, (np.integer,)):
            return int(obj)
        if isinstance(obj, (np.floating,)):
            return float(obj)
        if isinstance(obj, (np.bool_,)):
            return bool(obj)
        if isinstance(obj, np.ndarray):
            return obj.tolist()
        return super().default(obj)


def sanitize_for_json(obj, _path=""):
    """Recursively map non-finite floats (NaN/Inf) to None before dumping.

    Python's json module emits bare NaN/Infinity tokens by default — invalid
    JSON that strict parsers (browsers) reject outright. Applied to every
    signals.json/artifact write so no future NaN source can blank consumers.
    Each replacement is logged with its key path so new NaN sources surface
    in run logs instead of being silently nulled.
    """
    if isinstance(obj, dict):
        return {k: sanitize_for_json(v, f"{_path}.{k}" if _path else str(k))
                for k, v in obj.items()}
    if isinstance(obj, (list, tuple)):
        return [sanitize_for_json(v, f"{_path}[{i}]") for i, v in enumerate(obj)]
    if isinstance(obj, (float, np.floating)) and not np.isfinite(obj):
        print(f"[sanitize] WARN: non-finite value ({obj}) at {_path or '<root>'} -> null")
        return None
    return obj

# Try yfinance first, fall back to direct API
try:
    import yfinance as yf
    USE_YFINANCE = True
except ImportError:
    import requests
    USE_YFINANCE = False
    print("[WARN] yfinance not installed, using direct Yahoo Finance API")

DATA_DIR = os.path.join(os.path.dirname(os.path.abspath(__file__)), "data")
PUBLIC_DIR = os.path.join(os.path.dirname(os.path.abspath(__file__)), "public")

# ============================================================
# GICS SUB-INDUSTRY DEFINITIONS
# ============================================================
# The active universe is DYNAMIC: get_industry_groups() serves the
# weekly top-N built by universe_builder.py (Friday 20:00 ET rotation —
# after the weekly close, serving the following trading week — cached in
# data/universe_active.json). The hardcoded groups below are
# retained as FALLBACK_INDUSTRY_GROUPS — the safety net when no viable
# dynamic universe exists — and as the authoritative classification
# seed for gics_classifier. Names follow the GICS sub-industry standard;
# each group carries the metadata the thesis-breaker checks consume.
# ============================================================
FALLBACK_INDUSTRY_GROUPS = {
    "Technology Hardware, Storage & Peripherals": {
        "gics_code": "45202030",
        "gics_level": "Sub-Industry",
        "sector": "Information Technology",
        "industry_group": "Technology Hardware & Equipment",
        "tickers": ["SNDK", "STX", "WDC", "PSTG", "NTAP"],
        "cycle_stage": "mid",
        "sector_type": "tech_hardware",
        "commodity_proxy": None,
        "macro_sensitivities": ["sp500_drawdown", "group_momentum", "group_trend", "group_ytd"],
    },
    "Semiconductors — Memory & HBM": {
        "gics_code": "45301020",
        "gics_level": "Sub-Industry",
        "sector": "Information Technology",
        "industry_group": "Semiconductors & Semiconductor Equipment",
        "tickers": ["MU"],
        "cycle_stage": "early-mid",
        "sector_type": "semiconductor",
        "commodity_proxy": None,
        "macro_sensitivities": ["sp500_drawdown", "group_momentum", "group_trend", "group_ytd"],
    },
    "Semiconductor Materials & Equipment": {
        "gics_code": "45301010",
        "gics_level": "Sub-Industry",
        "sector": "Information Technology",
        "industry_group": "Semiconductors & Semiconductor Equipment",
        "tickers": ["ASML", "AMAT", "LRCX", "KLAC", "ENTG", "MKSI", "TER", "ALAB"],
        "cycle_stage": "mid",
        "sector_type": "semiconductor",
        "commodity_proxy": None,
        "macro_sensitivities": ["sp500_drawdown", "group_momentum", "group_trend", "group_ytd"],
    },
    "Gold": {
        "gics_code": "15104030",
        "gics_level": "Sub-Industry",
        "sector": "Materials",
        "industry_group": "Metals & Mining",
        "tickers": ["NEM", "GOLD", "AEM", "KGC", "AU", "HL", "CDE", "PAAS", "FNV", "RGLD", "EGO", "OR", "SSRM", "WPM"],
        "cycle_stage": "mid",
        "sector_type": "precious_metals",
        "commodity_proxy": "GLD",
        "macro_sensitivities": ["commodity_drop", "usd_strength", "group_momentum", "group_trend", "breadth_collapse"],
    },
    "Copper": {
        "gics_code": "15104025",
        "gics_level": "Sub-Industry",
        "sector": "Materials",
        "industry_group": "Metals & Mining",
        "tickers": ["FCX", "SCCO", "TECK", "HBM"],
        "cycle_stage": "mid",
        "sector_type": "industrial_metals",
        "commodity_proxy": None,
        "macro_sensitivities": ["group_momentum", "group_trend", "group_ytd", "sp500_drawdown"],
    },
    "Specialty Chemicals": {
        "gics_code": "15101050",
        "gics_level": "Sub-Industry",
        "sector": "Materials",
        "industry_group": "Chemicals",
        "tickers": ["TROX", "CC", "KRO", "DOW", "LYB", "PPG", "ECL"],
        "cycle_stage": "early",
        "sector_type": "chemicals",
        "commodity_proxy": None,
        "macro_sensitivities": ["group_momentum", "group_trend", "group_ytd", "energy_spike"],
    },
    "Oil & Gas Equipment & Services": {
        "gics_code": "10101020",
        "gics_level": "Sub-Industry",
        "sector": "Energy",
        "industry_group": "Energy Equipment & Services",
        "tickers": ["VAL", "RIG", "HAL", "NOV", "FTI", "LBRT"],
        "cycle_stage": "mid",
        "sector_type": "oil_services",
        "commodity_proxy": "USO",
        "macro_sensitivities": ["oil_collapse", "group_momentum", "group_trend"],
    },
    "Oil & Gas Exploration & Production": {
        "gics_code": "10102020",
        "gics_level": "Sub-Industry",
        "sector": "Energy",
        "industry_group": "Oil, Gas & Consumable Fuels",
        "tickers": ["KOS", "EQT", "COP", "FANG", "APA", "SM"],
        "cycle_stage": "mid",
        "sector_type": "oil_gas_ep",
        "commodity_proxy": "USO",
        "macro_sensitivities": ["oil_collapse", "natgas_collapse", "group_momentum", "group_trend"],
    },
    "Aerospace & Defense": {
        "gics_code": "20101010",
        "gics_level": "Sub-Industry",
        "sector": "Industrials",
        "industry_group": "Capital Goods",
        "tickers": ["RKLB", "LHX", "LMT", "KTOS", "PLTR", "RTX", "NOC", "GD", "HII"],
        "cycle_stage": "early-mid",
        "sector_type": "defense",
        "commodity_proxy": None,
        "macro_sensitivities": ["group_momentum", "group_trend", "group_ytd"],
    },
    "Independent Power Producers & Energy Traders": {
        "gics_code": "55105020",
        "gics_level": "Sub-Industry",
        "sector": "Utilities",
        "industry_group": "Independent Power and Renewable Electricity Producers",
        "tickers": ["CEG", "VST", "NRG", "OKLO", "SMR"],
        "cycle_stage": "early-mid",
        "sector_type": "power_nuclear",
        "commodity_proxy": None,
        "macro_sensitivities": ["group_momentum", "group_trend", "group_ytd", "sp500_drawdown"],
    },
    "Coal & Consumable Fuels (Uranium)": {
        "gics_code": "10102050",
        "gics_level": "Sub-Industry",
        "sector": "Energy",
        "industry_group": "Oil, Gas & Consumable Fuels",
        "tickers": ["UEC", "LEU", "CCJ"],
        "cycle_stage": "early",
        "sector_type": "uranium",
        "commodity_proxy": None,
        "macro_sensitivities": ["group_momentum", "group_trend", "group_ytd"],
    },
    "Oil & Gas Refining & Marketing": {
        "gics_code": "10102030",
        "gics_level": "Sub-Industry",
        "sector": "Energy",
        "industry_group": "Oil, Gas & Consumable Fuels",
        "tickers": ["MPC", "VLO", "PBF", "DK", "PSX"],
        "cycle_stage": "mid-late",
        "sector_type": "refining",
        "commodity_proxy": "USO",
        "macro_sensitivities": ["group_momentum", "group_trend", "group_ytd"],
    }
}

# Macro proxy tickers for thesis-breaker checks
MACRO_TICKERS = ["^GSPC", "GLD", "UUP", "USO", "UNG", "XLE"]


def get_industry_groups():
    """Active INDUSTRY_GROUPS for this run.

    Serves the cached dynamic universe if it belongs to the current rotation
    week; otherwise universe_builder rebuilds it (self-healing when a Friday
    rotation was missed). Falls back to FALLBACK_INDUSTRY_GROUPS only when
    no viable dynamic universe exists at all (fresh checkout + total data
    failure), so the dashboard never goes empty.
    """
    try:
        from universe_builder import get_active_industry_groups
        groups = get_active_industry_groups()
        if groups:
            return groups
        print("[engine] no viable dynamic universe — using FALLBACK_INDUSTRY_GROUPS")
    except Exception as e:
        print(f"[engine] dynamic universe unavailable ({e}) — using FALLBACK_INDUSTRY_GROUPS")
    return FALLBACK_INDUSTRY_GROUPS


# ============================================================
# DYNAMIC THESIS & THESIS-BREAKER GENERATION
# ============================================================
# All text is generated from live market data — nothing hardcoded.
# ============================================================

def generate_dynamic_thesis(group_name, group_info, group_stocks, macro_data):
    """
    Generate a dynamic investment thesis based on live market data.
    Returns a string describing why this group is investable right now.
    """
    total = len(group_stocks)
    if total == 0:
        return f"{group_name}: Insufficient data for thesis generation."

    # Gather live metrics
    ytd_values = [s["ytd_return"] for s in group_stocks if s.get("ytd_return") is not None]
    avg_ytd = np.mean(ytd_values) if ytd_values else 0
    rsi_values = [s["rsi"] for s in group_stocks if s.get("rsi")]
    avg_rsi = np.mean(rsi_values) if rsi_values else 50
    beating_count = sum(1 for s in group_stocks if s.get("beating_sp500"))
    pct_beating = (beating_count / total * 100) if total > 0 else 0
    above_ma50 = sum(1 for s in group_stocks if s.get("price", 0) > s.get("ma50", 0))
    pct_above_ma50 = (above_ma50 / total * 100) if total > 0 else 0

    # Best performer
    best = max(group_stocks, key=lambda s: s.get("ytd_return", -999)) if group_stocks else None
    best_name = best.get("ticker", "N/A") if best else "N/A"
    best_ytd = best.get("ytd_return", 0) if best else 0

    # Build thesis parts
    parts = []

    # Performance context
    if avg_ytd > 15:
        parts.append(f"Strong sector momentum with avg YTD return of {avg_ytd:+.1f}%")
    elif avg_ytd > 5:
        parts.append(f"Positive sector trend with avg YTD return of {avg_ytd:+.1f}%")
    elif avg_ytd > -5:
        parts.append(f"Sector consolidating with avg YTD return of {avg_ytd:+.1f}%")
    else:
        parts.append(f"Sector under pressure with avg YTD return of {avg_ytd:+.1f}%")

    # Breadth
    if pct_beating > 70:
        parts.append(f"Broad strength — {beating_count}/{total} stocks ({pct_beating:.0f}%) beating S&P 500")
    elif pct_beating > 50:
        parts.append(f"Moderate breadth with {beating_count}/{total} stocks outperforming S&P 500")
    elif pct_beating > 30:
        parts.append(f"Narrow leadership — only {beating_count}/{total} stocks beating S&P 500")
    else:
        parts.append(f"Weak breadth — just {beating_count}/{total} stocks outperforming S&P 500")

    # Trend health
    if pct_above_ma50 > 70:
        parts.append(f"{pct_above_ma50:.0f}% of stocks above 50-day MA, confirming uptrend")
    elif pct_above_ma50 > 50:
        parts.append(f"{pct_above_ma50:.0f}% above 50-day MA, trend intact but mixed")
    else:
        parts.append(f"Only {pct_above_ma50:.0f}% above 50-day MA, trend weakening")

    # Momentum context
    if avg_rsi > 65:
        parts.append(f"Group momentum strong (avg RSI {avg_rsi:.0f}) — watch for overbought conditions")
    elif avg_rsi > 50:
        parts.append(f"Healthy momentum (avg RSI {avg_rsi:.0f})")
    elif avg_rsi > 40:
        parts.append(f"Neutral momentum (avg RSI {avg_rsi:.0f}) — potential accumulation zone")
    else:
        parts.append(f"Oversold conditions (avg RSI {avg_rsi:.0f}) — potential mean-reversion setup")

    # Leader highlight
    if best and best_ytd > 10:
        parts.append(f"Led by {best_name} at {best_ytd:+.1f}% YTD")

    # Commodity proxy context
    commodity_proxy = group_info.get("commodity_proxy")
    if commodity_proxy and commodity_proxy in macro_data:
        proxy_data = macro_data[commodity_proxy]
        if proxy_data is not None and len(proxy_data) > 20:
            proxy_price = float(proxy_data["Close"].iloc[-1])
            proxy_high_60d = float(proxy_data["High"].iloc[-60:].max()) if len(proxy_data) >= 60 else float(proxy_data["High"].max())
            pct_from_high = ((proxy_price - proxy_high_60d) / proxy_high_60d) * 100
            proxy_names = {"GLD": "Gold", "USO": "Crude Oil", "UNG": "Natural Gas"}
            proxy_label = proxy_names.get(commodity_proxy, commodity_proxy)
            if pct_from_high > -3:
                parts.append(f"{proxy_label} near 60-day highs (${proxy_price:.2f}), supporting commodity thesis")
            elif pct_from_high > -10:
                parts.append(f"{proxy_label} at ${proxy_price:.2f}, {pct_from_high:.1f}% off 60-day high")
            else:
                parts.append(f"{proxy_label} under pressure at ${proxy_price:.2f} ({pct_from_high:.1f}% off 60-day high)")

    return ". ".join(parts) + "."


def generate_dynamic_thesis_breaker(group_name, group_info, group_stocks, macro_data, sp500_ytd):
    """
    Generate a dynamic thesis-breaker description based on live market data.
    Identifies the specific risks/levels that would invalidate the investment thesis.
    Returns a string with current price-based risk levels.
    """
    total = len(group_stocks)
    risks = []

    # --- Macro risk: S&P 500 drawdown ---
    sp_data = macro_data.get("^GSPC")
    if sp_data is not None and len(sp_data) > 20:
        current_year = datetime.datetime.now().year
        ytd_data = sp_data[sp_data.index.year == current_year]
        if len(ytd_data) > 0:
            sp_current = float(ytd_data["Close"].iloc[-1])
            sp_high = float(ytd_data["High"].max())
            sp_10pct_level = sp_high * 0.9
            current_dd = ((sp_current - sp_high) / sp_high) * 100
            if current_dd < -5:
                risks.append(f"S&P 500 already {current_dd:.1f}% from YTD high ({sp_high:.0f}) — critical if breaks {sp_10pct_level:.0f}")
            else:
                risks.append(f"S&P 500 correction below {sp_10pct_level:.0f} (10% off {sp_high:.0f} high)")

    # --- Commodity-specific risks ---
    sector_type = group_info.get("sector_type", "")
    commodity_proxy = group_info.get("commodity_proxy")

    if commodity_proxy == "GLD":
        gld_data = macro_data.get("GLD")
        if gld_data is not None and len(gld_data) > 20:
            gld_price = float(gld_data["Close"].iloc[-1])
            gld_high = float(gld_data["High"].iloc[-60:].max()) if len(gld_data) >= 60 else float(gld_data["High"].max())
            gld_8pct = gld_high * 0.92
            risks.append(f"Gold reversal below ${gld_8pct:.0f} (8% off ${gld_high:.0f} recent high, currently ${gld_price:.0f})")

        # USD strength check
        uup_data = macro_data.get("UUP")
        if uup_data is not None and len(uup_data) > 20:
            uup_ytd = compute_ytd_return_v1(uup_data)
            uup_price = float(uup_data["Close"].iloc[-1])
            if uup_ytd > 2:
                risks.append(f"USD already strengthening ({uup_ytd:+.1f}% YTD at ${uup_price:.2f}) — critical above +5% YTD")
            else:
                risks.append(f"Sharp USD rally (DXY proxy at ${uup_price:.2f}, {uup_ytd:+.1f}% YTD) — watch for >5% YTD surge")

    elif commodity_proxy == "USO":
        uso_data = macro_data.get("USO")
        if uso_data is not None and len(uso_data) > 20:
            uso_price = float(uso_data["Close"].iloc[-1])
            uso_high = float(uso_data["High"].iloc[-60:].max()) if len(uso_data) >= 60 else float(uso_data["High"].max())
            uso_25pct = uso_high * 0.75
            risks.append(f"Crude oil collapse below ${uso_25pct:.2f} (25% off ${uso_high:.2f} high, currently ${uso_price:.2f})")

        if sector_type == "oil_gas_ep":
            ung_data = macro_data.get("UNG")
            if ung_data is not None and len(ung_data) > 20:
                ung_price = float(ung_data["Close"].iloc[-1])
                ung_high = float(ung_data["High"].iloc[-60:].max()) if len(ung_data) >= 60 else float(ung_data["High"].max())
                ung_30pct = ung_high * 0.70
                risks.append(f"Natural gas breakdown below ${ung_30pct:.2f} (30% off ${ung_high:.2f} high, currently ${ung_price:.2f})")

    # --- Energy cost risk for chemicals ---
    if sector_type == "chemicals":
        xle_data = macro_data.get("XLE")
        if xle_data is not None and len(xle_data) > 20:
            xle_ytd = compute_ytd_return_v1(xle_data)
            xle_price = float(xle_data["Close"].iloc[-1])
            risks.append(f"Energy cost spike — XLE at ${xle_price:.2f} ({xle_ytd:+.1f}% YTD), margins squeezed if >+15% YTD")

    # --- Group-level technical risks (always computed) ---
    rsi_values = [s["rsi"] for s in group_stocks if s.get("rsi")]
    avg_rsi = np.mean(rsi_values) if rsi_values else 50
    below_ma50 = sum(1 for s in group_stocks if s.get("price", 0) < s.get("ma50", 0))
    pct_below_ma50 = (below_ma50 / total * 100) if total > 0 else 0
    ytd_values = [s["ytd_return"] for s in group_stocks if s.get("ytd_return") is not None]
    avg_ytd = np.mean(ytd_values) if ytd_values else 0

    # RSI risk
    if avg_rsi < 45:
        risks.append(f"Group momentum deteriorating (avg RSI {avg_rsi:.0f}, breaker at <40)")
    else:
        risks.append(f"Momentum breakdown if avg RSI drops below 40 (currently {avg_rsi:.0f})")

    # MA50 risk
    if pct_below_ma50 > 30:
        risks.append(f"Trend weakening — {below_ma50}/{total} ({pct_below_ma50:.0f}%) already below 50-day MA, critical if >50%")
    else:
        risks.append(f"Trend reversal if >50% of stocks break below 50-day MA (currently {pct_below_ma50:.0f}% below)")

    # YTD risk
    if avg_ytd < 3:
        risks.append(f"Group leadership fragile (avg YTD {avg_ytd:+.1f}%), at risk of turning negative")
    else:
        risks.append(f"Leadership lost if group avg YTD turns negative (currently {avg_ytd:+.1f}%)")

    # Breadth risk for broad groups
    if total >= 8:
        beating_count = sum(1 for s in group_stocks if s.get("beating_sp500"))
        pct_beating = (beating_count / total * 100) if total > 0 else 0
        if pct_beating < 60:
            risks.append(f"Breadth narrowing — only {beating_count}/{total} ({pct_beating:.0f}%) beating S&P, critical below 50%")

    return ". ".join(risks) + "."


# ============================================================
# BREAKER COVERAGE — the check<->sensitivity map (D-019)
# ============================================================
# THE INVARIANT: a breaker may report "clear" only when every check the
# group's sensitivities CALL FOR actually ran. Before this table existed,
# a swallowed macro fetch made generate_dynamic_breaker_checks silently
# omit the affected check; check_thesis_breakers then had nothing to
# trigger AND nothing to list as "not triggered", so breaker_status fell
# through to its "clear" initialiser. A group that could not be checked
# was byte-identical to a group that was checked and found healthy —
# and the search-page gate certified that fabricated clear as verified.
#
# The map is NOT 1:1 and must not be reconstructed by name-matching:
#   commodity_drop + commodity_proxy "GLD" -> gold_below_threshold
#   oil_collapse                           -> oil_below_60
#   natgas_collapse                        -> natgas_collapse (same id)
# Four sensitivities need no macro input at all (they read the group's
# own member rows), so they can never be degraded by a macro outage.
#
# THIS TABLE IS THE SOURCE OF TRUTH and is pinned complete in both
# directions (test_breaker_coverage.py): every sensitivity any group can
# declare resolves here, and every check the generator can emit is
# claimed by exactly one spec.
BREAKER_CHECK_SPECS = (
    {"sensitivity": "sp500_drawdown", "check": "sp500_drawdown_10pct",
     "macro": "^GSPC", "min_bars": 21},
    {"sensitivity": "group_momentum", "check": "group_avg_rsi_below_40",
     "macro": None, "min_bars": 0},
    {"sensitivity": "group_trend", "check": "majority_below_ma50",
     "macro": None, "min_bars": 0},
    {"sensitivity": "group_ytd", "check": "avg_ytd_negative",
     "macro": None, "min_bars": 0},
    {"sensitivity": "breadth_collapse", "check": "breadth_collapse",
     "macro": None, "min_bars": 0},
    # conditional: the branch is gated on commodity_proxy == "GLD", so a
    # group declaring commodity_drop with any other proxy gets NO check.
    # That is a CONFIG gap, not an outage — reported as its own reason
    # rather than silently expecting nothing.
    {"sensitivity": "commodity_drop", "check": "gold_below_threshold",
     "macro": "GLD", "min_bars": 21, "requires_proxy": "GLD"},
    {"sensitivity": "usd_strength", "check": "usd_strength",
     "macro": "UUP", "min_bars": 21},
    {"sensitivity": "oil_collapse", "check": "oil_below_60",
     "macro": "USO", "min_bars": 21},
    {"sensitivity": "natgas_collapse", "check": "natgas_collapse",
     "macro": "UNG", "min_bars": 21},
    {"sensitivity": "energy_spike", "check": "energy_spike",
     "macro": "XLE", "min_bars": 21},
)


def breaker_expected_checks(group_info):
    """The checks a group's declared sensitivities CALL FOR.

    Returns (expected_ids, config_gaps) where config_gaps names any
    declared sensitivity that resolves to no check at all — an
    unimplemented combination, which must degrade the group rather than
    quietly shrink what "complete coverage" means.
    """
    sensitivities = list(group_info.get("macro_sensitivities") or [])
    proxy = group_info.get("commodity_proxy")
    by_sens = {}
    for spec in BREAKER_CHECK_SPECS:
        by_sens.setdefault(spec["sensitivity"], []).append(spec)

    expected, gaps = [], []
    for sens in sensitivities:
        specs = by_sens.get(sens)
        if not specs:
            gaps.append(f"sensitivity '{sens}' has no implemented check")
            continue
        matched = False
        for spec in specs:
            need = spec.get("requires_proxy")
            if need is not None and proxy != need:
                continue
            expected.append(spec["check"])
            matched = True
        if not matched:
            gaps.append(
                f"sensitivity '{sens}' declared with commodity_proxy "
                f"{proxy!r} — no check implemented for that pairing")
    return sorted(set(expected)), gaps


def breaker_coverage(group_info, checks_run, macro_status=None,
                     degraded_inputs=None):
    """Compare the checks that RAN against the checks CALLED FOR.

    `checks_run` is the generator's own output — the ground truth for
    what was computed — so this never re-implements a guard and cannot
    drift from it. Missing checks are explained from `macro_status`
    (recorded at fetch time) where a macro input is implicated, and
    reported honestly as an uncomputable input otherwise.

    Returns {"expected": [...], "run": [...], "missing": [...],
             "reasons": [...], "complete": bool}.
    """
    macro_status = macro_status or {}
    expected, gaps = breaker_expected_checks(group_info)
    # An input can be missing WITHOUT a check going unrun: breadth_collapse
    # and the beating-S&P counts compare against sp500_ytd, which defaults
    # to 0.0 when the (separate) index fetch fails — the check still runs,
    # on a fabricated baseline. expected-vs-run cannot see that, so a
    # degraded INPUT is reported directly.
    input_gaps = []
    for check_id, reason in (degraded_inputs or {}).items():
        if check_id in expected:
            input_gaps.append(f"{check_id}: {reason}")
    run = sorted(set(checks_run or ()))
    spec_by_check = {s["check"]: s for s in BREAKER_CHECK_SPECS}

    missing = [c for c in expected if c not in set(run)]
    reasons = list(gaps) + list(input_gaps)
    for check_id in missing:
        spec = spec_by_check.get(check_id) or {}
        macro = spec.get("macro")
        st = macro_status.get(macro) if macro else None
        if macro and st and not st.get("ok"):
            reasons.append(f"{check_id}: {macro} unavailable — "
                           f"{st.get('reason') or 'fetch failed'}")
        elif macro and macro not in macro_status:
            # never fetched this run (e.g. an older engine or a partial
            # replay) — say so rather than blaming the data
            reasons.append(f"{check_id}: {macro} was not fetched this run")
        elif macro:
            # the fetch SUCCEEDED but the check still did not run: the
            # series is too short for this check's own guard (run_engine
            # accepts >5 bars; the checks need >20), or a computed input
            # was empty (e.g. the S&P year-to-date window in the first
            # days of January). Name the input and what we know about it
            # rather than implying the fetch failed.
            bars = (macro_status.get(macro) or {}).get("bars")
            need = spec.get("min_bars")
            if bars is not None and need and bars < need:
                reasons.append(f"{check_id}: {macro} returned {bars} bars, "
                               f"needs {need}")
            else:
                reasons.append(f"{check_id}: {macro} arrived but the check "
                               f"inputs were not computable")
        else:
            reasons.append(f"{check_id}: inputs insufficient to compute "
                           f"(no macro input required — group member data)")
    return {
        "expected": expected,
        "run": run,
        "missing": missing,
        "reasons": reasons,
        "complete": not missing and not gaps and not input_gaps,
    }


def resolve_breaker_status(alerts, coverage):
    """The breaker status ladder — ONE implementation.

    A TRIGGER STILL WINS: something that fired is news regardless of what
    else could not be measured. Absent a trigger, "clear" is a POSITIVE
    CLAIM about having looked, and may only be made on complete coverage
    (D-019); otherwise the group is degraded.

    run_engine and the pins both call this, so a change to the ladder
    cannot pass a test that quietly reimplemented it.
    """
    triggered = [a for a in (alerts or []) if a.get("triggered")]
    if any(a.get("severity") == "critical" for a in triggered):
        return "critical"
    if any(a.get("severity") == "high" for a in triggered):
        return "warning"
    if any(a.get("severity") == "medium" for a in triggered):
        return "watch"
    if (coverage or {}).get("complete"):
        return "clear"
    return "degraded"


def generate_dynamic_breaker_checks(group_name, group_info, group_stocks, macro_data):
    """
    Generate dynamic breaker check definitions based on the group's
    macro_sensitivities and current market data.
    Returns dict of {check_id: description_with_current_levels}.
    """
    checks = {}
    sensitivities = group_info.get("macro_sensitivities", [])
    total = len(group_stocks)
    commodity_proxy = group_info.get("commodity_proxy")
    sector_type = group_info.get("sector_type", "")

    # S&P 500 drawdown check
    if "sp500_drawdown" in sensitivities:
        sp_data = macro_data.get("^GSPC")
        if sp_data is not None and len(sp_data) > 20:
            current_year = datetime.datetime.now().year
            ytd_data = sp_data[sp_data.index.year == current_year]
            if len(ytd_data) > 0:
                sp_high = float(ytd_data["High"].max())
                sp_current = float(ytd_data["Close"].iloc[-1])
                sp_level = sp_high * 0.9
                dd_now = ((sp_current - sp_high) / sp_high) * 100
                checks["sp500_drawdown_10pct"] = f"S&P 500 drops >10% from {sp_high:.0f} YTD high (below {sp_level:.0f}) — currently {dd_now:+.1f}%"

    # Group momentum check
    if "group_momentum" in sensitivities:
        rsi_values = [s["rsi"] for s in group_stocks if s.get("rsi")]
        avg_rsi = np.mean(rsi_values) if rsi_values else 50
        checks["group_avg_rsi_below_40"] = f"Group avg RSI falls below 40 — currently {avg_rsi:.0f}"

    # Group trend check
    if "group_trend" in sensitivities:
        below_ma50 = sum(1 for s in group_stocks if s.get("price", 0) < s.get("ma50", 0))
        pct = (below_ma50 / total * 100) if total > 0 else 0
        checks["majority_below_ma50"] = f">50% of stocks fall below 50-day MA — currently {pct:.0f}% ({below_ma50}/{total}) below"

    # Group YTD check
    if "group_ytd" in sensitivities:
        ytd_vals = [s["ytd_return"] for s in group_stocks if s.get("ytd_return") is not None]
        avg_ytd = np.mean(ytd_vals) if ytd_vals else 0
        checks["avg_ytd_negative"] = f"Group avg YTD return turns negative — currently {avg_ytd:+.1f}%"

    # Breadth collapse check
    if "breadth_collapse" in sensitivities:
        beating = sum(1 for s in group_stocks if s.get("beating_sp500"))
        pct_beating = (beating / total * 100) if total > 0 else 0
        checks["breadth_collapse"] = f"<50% of stocks beating S&P 500 — currently {pct_beating:.0f}% ({beating}/{total})"

    # Commodity-specific checks
    if "commodity_drop" in sensitivities and commodity_proxy == "GLD":
        gld_data = macro_data.get("GLD")
        if gld_data is not None and len(gld_data) > 20:
            gld_high = float(gld_data["High"].iloc[-60:].max()) if len(gld_data) >= 60 else float(gld_data["High"].max())
            gld_current = float(gld_data["Close"].iloc[-1])
            gld_drop_pct = ((gld_current - gld_high) / gld_high) * 100
            gld_trigger = gld_high * 0.92
            checks["gold_below_threshold"] = f"Gold (GLD) drops >8% from ${gld_high:.0f} high (below ${gld_trigger:.0f}) — currently ${gld_current:.0f} ({gld_drop_pct:+.1f}%)"

    if "usd_strength" in sensitivities:
        uup_data = macro_data.get("UUP")
        if uup_data is not None and len(uup_data) > 20:
            uup_ytd = compute_ytd_return_v1(uup_data)
            checks["usd_strength"] = f"DXY proxy (UUP) rises >5% YTD — currently {uup_ytd:+.1f}% YTD"

    if "oil_collapse" in sensitivities:
        uso_data = macro_data.get("USO")
        if uso_data is not None and len(uso_data) > 20:
            uso_high = float(uso_data["High"].iloc[-60:].max()) if len(uso_data) >= 60 else float(uso_data["High"].max())
            uso_current = float(uso_data["Close"].iloc[-1])
            uso_drop_pct = ((uso_current - uso_high) / uso_high) * 100
            uso_trigger = uso_high * 0.75
            checks["oil_below_60"] = f"Crude (USO) drops >25% from ${uso_high:.2f} high (below ${uso_trigger:.2f}) — currently ${uso_current:.2f} ({uso_drop_pct:+.1f}%)"

    if "natgas_collapse" in sensitivities:
        ung_data = macro_data.get("UNG")
        if ung_data is not None and len(ung_data) > 20:
            ung_high = float(ung_data["High"].iloc[-60:].max()) if len(ung_data) >= 60 else float(ung_data["High"].max())
            ung_current = float(ung_data["Close"].iloc[-1])
            ung_drop_pct = ((ung_current - ung_high) / ung_high) * 100
            ung_trigger = ung_high * 0.70
            checks["natgas_collapse"] = f"Nat gas (UNG) drops >30% from ${ung_high:.2f} high (below ${ung_trigger:.2f}) — currently ${ung_current:.2f} ({ung_drop_pct:+.1f}%)"

    if "energy_spike" in sensitivities:
        xle_data = macro_data.get("XLE")
        if xle_data is not None and len(xle_data) > 20:
            xle_ytd = compute_ytd_return_v1(xle_data)
            xle_price = float(xle_data["Close"].iloc[-1])
            checks["energy_spike"] = f"Energy (XLE) surges >15% YTD — currently ${xle_price:.2f} ({xle_ytd:+.1f}% YTD)"

    return checks


# ============================================================
# DATA FETCHING
# ============================================================
def fetch_price_data_yfinance(ticker, period="6mo"):
    """Fetch historical price data using yfinance."""
    try:
        stock = yf.Ticker(ticker)
        df = stock.history(period=period)
        if df.empty:
            return None
        df.index = df.index.tz_localize(None)
        return df
    except Exception as e:
        print(f"  [ERROR] {ticker}: {e}")
        return None


def fetch_price_data_api(ticker, period="6mo"):
    """Fetch data via Yahoo Finance API directly (fallback)."""
    try:
        range_map = {"1mo": "1mo", "3mo": "3mo", "6mo": "6mo", "1y": "1y"}
        r = range_map.get(period, "6mo")
        url = f"https://query1.finance.yahoo.com/v8/finance/chart/{ticker}?range={r}&interval=1d"
        headers = {"User-Agent": "Mozilla/5.0"}
        resp = requests.get(url, headers=headers, timeout=15)
        data = resp.json()
        result = data["chart"]["result"][0]
        timestamps = result["timestamp"]
        quotes = result["indicators"]["quote"][0]
        df = pd.DataFrame({
            "Open": quotes["open"],
            "High": quotes["high"],
            "Low": quotes["low"],
            "Close": quotes["close"],
            "Volume": quotes["volume"]
        }, index=pd.to_datetime(timestamps, unit="s"))
        df.dropna(subset=["Close"], inplace=True)
        return df
    except Exception as e:
        print(f"  [ERROR] {ticker}: {e}")
        return None


def fetch_data(ticker, period="6mo"):
    if USE_YFINANCE:
        return fetch_price_data_yfinance(ticker, period)
    return fetch_price_data_api(ticker, period)


def fetch_fundamentals_yfinance(ticker):
    """Fetch fundamental data via yfinance .info and .financials."""
    fundamentals = {
        "market_cap": None,
        "forward_pe": None,
        "trailing_pe": None,
        "revenue_growth_yoy": None,
        "gross_margin": None,
        "operating_margin": None,
        "profit_margin": None,
        "eps_trailing": None,
        "eps_forward": None,
        "dividend_yield": None,
        "beta": None,
        "short_pct_float": None,
        "target_mean_price": None,
        "recommendation": None,
        "sector": None,
        "industry": None,
        "fifty_two_week_high": None,
        "fifty_two_week_low": None
    }
    if not USE_YFINANCE:
        return fundamentals
    try:
        stock = yf.Ticker(ticker)
        info = stock.info or {}
        fundamentals["market_cap"] = info.get("marketCap")
        fundamentals["forward_pe"] = info.get("forwardPE")
        fundamentals["trailing_pe"] = info.get("trailingPE")
        fundamentals["revenue_growth_yoy"] = info.get("revenueGrowth")
        fundamentals["gross_margin"] = info.get("grossMargins")
        fundamentals["operating_margin"] = info.get("operatingMargins")
        fundamentals["profit_margin"] = info.get("profitMargins")
        fundamentals["eps_trailing"] = info.get("trailingEps")
        fundamentals["eps_forward"] = info.get("forwardEps")
        fundamentals["dividend_yield"] = info.get("dividendYield")
        fundamentals["beta"] = info.get("beta")
        fundamentals["short_pct_float"] = info.get("shortPercentOfFloat")
        fundamentals["target_mean_price"] = info.get("targetMeanPrice")
        fundamentals["recommendation"] = info.get("recommendationKey")
        fundamentals["sector"] = info.get("sector")
        fundamentals["industry"] = info.get("industry")
        fundamentals["fifty_two_week_high"] = info.get("fiftyTwoWeekHigh")
        fundamentals["fifty_two_week_low"] = info.get("fiftyTwoWeekLow")
    except Exception as e:
        print(f"  [WARN] Fundamentals failed for {ticker}: {e}")
    return fundamentals


# ============================================================
# TECHNICAL INDICATORS
# ============================================================
def compute_rsi(series, period=14):
    """Compute RSI (Relative Strength Index)."""
    delta = series.diff()
    gain = delta.where(delta > 0, 0.0)
    loss = -delta.where(delta < 0, 0.0)
    avg_gain = gain.rolling(window=period, min_periods=period).mean()
    avg_loss = loss.rolling(window=period, min_periods=period).mean()
    for i in range(period, len(avg_gain)):
        avg_gain.iloc[i] = (avg_gain.iloc[i-1] * (period - 1) + gain.iloc[i]) / period
        avg_loss.iloc[i] = (avg_loss.iloc[i-1] * (period - 1) + loss.iloc[i]) / period
    rs = avg_gain / avg_loss
    rsi = 100 - (100 / (1 + rs))
    return rsi


def compute_macd(series, fast=12, slow=26, signal_period=9):
    """Compute MACD, Signal Line, and Histogram."""
    ema_fast = series.ewm(span=fast, adjust=False).mean()
    ema_slow = series.ewm(span=slow, adjust=False).mean()
    macd_line = ema_fast - ema_slow
    signal_line = macd_line.ewm(span=signal_period, adjust=False).mean()
    histogram = macd_line - signal_line
    return macd_line, signal_line, histogram


def compute_moving_averages(series):
    """Compute 20, 50, 200 day moving averages."""
    ma20 = series.rolling(window=20).mean()
    ma50 = series.rolling(window=50).mean()
    ma200 = series.rolling(window=200).mean() if len(series) >= 200 else pd.Series([np.nan]*len(series), index=series.index)
    return ma20, ma50, ma200


def compute_ytd_return_v1(df):
    """FROZEN (D-020a). The pre-D-020a YTD: anchored on the first VALID
    close of the CURRENT year present in the frame — NOT the prior
    year's last close. On a frame that does not reach back to January
    (production's 6mo fetch after early July) the anchor silently
    becomes the oldest bar in the window, i.e. a rolling return wearing
    a YTD label. Kept verbatim so committed studies keep reproducing;
    production calls compute_ytd_return_v2. The only remaining
    production callers are the UUP/XLE macro breaker inputs, scoped out
    of D-020a by name (their thresholds were calibrated on this
    construct — re-anchoring them is its own deliberation).

    Yahoo intermittently returns phantom bars with NaN Close (e.g. a Jan-2
    row on some tickers); without dropna the NaN baseline propagates through
    group averages into signals.json as invalid bare-NaN JSON.
    """
    current_year = datetime.datetime.now().year
    closes = df[df.index.year == current_year]["Close"].dropna()
    if len(closes) < 2:
        return 0.0
    first_close = closes.iloc[0]
    last_close = closes.iloc[-1]
    if not first_close:
        return 0.0
    return round(((last_close - first_close) / first_close) * 100, 2)


def compute_ytd_return_v2(df, with_basis=False):
    """Real calendar year-to-date (D-020a): anchored on the LAST CLOSE
    OF THE PRIOR CALENDAR YEAR. Needs a frame reaching into the prior
    year (production fetches period="1y" wherever this feeds).

    Fallback, recorded not hidden: a frame with no prior-year bar (a
    listing younger than the year, or a caller that could not fetch the
    wider frame) anchors on the first current-year close — v1's anchor
    — and reports basis "first_close_of_year" so the degradation is
    visible in the artifact instead of impersonating a real YTD.
    """
    current_year = datetime.datetime.now().year
    closes = df["Close"].dropna()
    cur = closes[closes.index.year == current_year]
    if len(cur) == 0:
        return (0.0, "no_current_year_bars") if with_basis else 0.0
    prior = closes[closes.index.year == current_year - 1]
    last_close = cur.iloc[-1]
    if len(prior) and prior.iloc[-1]:
        anchor, basis = prior.iloc[-1], "prior_year_close"
    elif len(cur) >= 2 and cur.iloc[0]:
        anchor, basis = cur.iloc[0], "first_close_of_year"
    else:
        return (0.0, "insufficient") if with_basis else 0.0
    ytd = round(((last_close - anchor) / anchor) * 100, 2)
    return (ytd, basis) if with_basis else ytd


def compute_volume_trend(df, lookback=20):
    """Compare recent volume to average."""
    if len(df) < lookback:
        return 1.0
    recent_avg = df["Volume"].iloc[-5:].mean()
    longer_avg = df["Volume"].iloc[-lookback:].mean()
    if longer_avg == 0:
        return 1.0
    return round(recent_avg / longer_avg, 2)


def compute_momentum_metrics(df):
    """Compute additional momentum metrics beyond RSI/MACD."""
    close = df["Close"]
    high = df["High"]
    metrics = {}

    # % off 52-week high
    if len(high) >= 20:
        high_52w = high.iloc[-min(252, len(high)):].max()
        metrics["high_52w"] = round(float(high_52w), 2)
        metrics["pct_from_52w_high"] = round(((close.iloc[-1] - high_52w) / high_52w) * 100, 1)
    else:
        metrics["high_52w"] = round(float(close.iloc[-1]), 2)
        metrics["pct_from_52w_high"] = 0.0

    # 52-week low
    if len(df) >= 20:
        low_52w = df["Low"].iloc[-min(252, len(df)):].min()
        metrics["low_52w"] = round(float(low_52w), 2)
    else:
        metrics["low_52w"] = round(float(close.iloc[-1]), 2)

    # 1-month return
    if len(close) >= 22:
        metrics["return_1m"] = round(((close.iloc[-1] - close.iloc[-22]) / close.iloc[-22]) * 100, 2)
    else:
        metrics["return_1m"] = 0.0

    # 3-month return
    if len(close) >= 63:
        metrics["return_3m"] = round(((close.iloc[-1] - close.iloc[-63]) / close.iloc[-63]) * 100, 2)
    else:
        metrics["return_3m"] = 0.0

    # Relative strength (price vs MA50 as %)
    ma50 = close.rolling(50).mean()
    if not pd.isna(ma50.iloc[-1]):
        metrics["rs_vs_ma50"] = round(((close.iloc[-1] - ma50.iloc[-1]) / ma50.iloc[-1]) * 100, 2)
    else:
        metrics["rs_vs_ma50"] = 0.0

    # Trend strength: count of last 20 days where close > MA20
    ma20 = close.rolling(20).mean()
    if len(close) >= 20 and not pd.isna(ma20.iloc[-1]):
        last_20 = close.iloc[-20:]
        ma20_last = ma20.iloc[-20:]
        above_count = int((last_20 > ma20_last).sum())
        metrics["trend_strength"] = above_count  # 0-20, higher = stronger uptrend
    else:
        metrics["trend_strength"] = 10

    return metrics


# ============================================================
# THESIS-BREAKER MONITORING
# ============================================================
def check_thesis_breakers(group_name, group_info, group_stocks, macro_data,
                          sp500_ytd, macro_status=None,
                          degraded_inputs=None):
    """
    Check thesis-breaker conditions for a group.
    All checks are dynamically generated from live market data.

    Returns (alerts, coverage) — coverage records which checks the group's
    sensitivities CALLED FOR versus which actually RAN (D-019). The
    generator's own output is the ground truth for "ran", so coverage can
    never drift from the guards that decide computability.
    """
    alerts = []
    # Generate dynamic checks from live data instead of reading static config
    checks = generate_dynamic_breaker_checks(group_name, group_info, group_stocks, macro_data)
    coverage = breaker_coverage(group_info, checks.keys(), macro_status,
                                degraded_inputs=degraded_inputs)

    # Compute group-level metrics
    rsi_values = [s["rsi"] for s in group_stocks if s.get("rsi")]
    avg_rsi = np.mean(rsi_values) if rsi_values else 50
    below_ma50_count = sum(1 for s in group_stocks if s.get("price", 0) < s.get("ma50", 0))
    total_stocks = len(group_stocks)
    pct_below_ma50 = (below_ma50_count / total_stocks * 100) if total_stocks > 0 else 0
    ytd_values = [s["ytd_return"] for s in group_stocks if s.get("ytd_return") is not None]
    avg_ytd = np.mean(ytd_values) if ytd_values else 0
    beating_count = sum(1 for s in group_stocks if s.get("beating_sp500"))
    pct_beating = (beating_count / total_stocks * 100) if total_stocks > 0 else 0

    # --- Check each condition ---
    if "sp500_drawdown_10pct" in checks:
        sp_data = macro_data.get("^GSPC")
        if sp_data is not None and len(sp_data) > 20:
            current_year = datetime.datetime.now().year
            ytd_data = sp_data[sp_data.index.year == current_year]
            if len(ytd_data) > 0:
                ytd_high = ytd_data["High"].max()
                current = ytd_data["Close"].iloc[-1]
                drawdown = ((current - ytd_high) / ytd_high) * 100
                if drawdown < -10:
                    alerts.append({
                        "check": "sp500_drawdown_10pct",
                        "severity": "critical",
                        "triggered": True,
                        "message": f"S&P 500 down {drawdown:.1f}% from YTD high — macro risk-off",
                        "description": checks["sp500_drawdown_10pct"],
                        "value": round(drawdown, 1)
                    })

    if "group_avg_rsi_below_40" in checks:
        if avg_rsi < 40:
            alerts.append({
                "check": "group_avg_rsi_below_40",
                "severity": "high",
                "triggered": True,
                "message": f"Group avg RSI at {avg_rsi:.1f} — momentum breakdown",
                "description": checks["group_avg_rsi_below_40"],
                "value": round(avg_rsi, 1)
            })

    if "majority_below_ma50" in checks:
        if pct_below_ma50 > 50:
            alerts.append({
                "check": "majority_below_ma50",
                "severity": "high",
                "triggered": True,
                "message": f"{below_ma50_count}/{total_stocks} stocks ({pct_below_ma50:.0f}%) below 50-day MA — trend reversal",
                "description": checks["majority_below_ma50"],
                "value": round(pct_below_ma50, 0)
            })

    if "avg_ytd_negative" in checks:
        if avg_ytd < 0:
            alerts.append({
                "check": "avg_ytd_negative",
                "severity": "critical",
                "triggered": True,
                "message": f"Group avg YTD return is {avg_ytd:.1f}% — leadership lost",
                "description": checks["avg_ytd_negative"],
                "value": round(avg_ytd, 1)
            })

    if "breadth_collapse" in checks:
        if pct_beating < 50:
            alerts.append({
                "check": "breadth_collapse",
                "severity": "high",
                "triggered": True,
                "message": f"Only {beating_count}/{total_stocks} ({pct_beating:.0f}%) beating S&P — breadth collapsing",
                "description": checks["breadth_collapse"],
                "value": round(pct_beating, 0)
            })

    if "gold_below_threshold" in checks:
        gld_data = macro_data.get("GLD")
        if gld_data is not None and len(gld_data) > 20:
            recent_high = gld_data["High"].iloc[-60:].max() if len(gld_data) >= 60 else gld_data["High"].max()
            current = gld_data["Close"].iloc[-1]
            gld_drop = ((current - recent_high) / recent_high) * 100
            if gld_drop < -8:
                alerts.append({
                    "check": "gold_below_threshold",
                    "severity": "critical",
                    "triggered": True,
                    "message": f"Gold (GLD) down {gld_drop:.1f}% from recent high — commodity thesis weakening",
                    "description": checks["gold_below_threshold"],
                    "value": round(gld_drop, 1)
                })

    if "usd_strength" in checks:
        uup_data = macro_data.get("UUP")
        if uup_data is not None and len(uup_data) > 20:
            uup_ytd = compute_ytd_return_v1(uup_data)
            if uup_ytd > 5:
                alerts.append({
                    "check": "usd_strength",
                    "severity": "high",
                    "triggered": True,
                    "message": f"USD (UUP) up {uup_ytd:.1f}% YTD — dollar strength crushing gold",
                    "description": checks["usd_strength"],
                    "value": round(uup_ytd, 1)
                })

    if "oil_below_60" in checks:
        uso_data = macro_data.get("USO")
        if uso_data is not None and len(uso_data) > 20:
            recent_high = uso_data["High"].iloc[-60:].max() if len(uso_data) >= 60 else uso_data["High"].max()
            current = uso_data["Close"].iloc[-1]
            uso_drop = ((current - recent_high) / recent_high) * 100
            if uso_drop < -25:
                alerts.append({
                    "check": "oil_below_60",
                    "severity": "critical",
                    "triggered": True,
                    "message": f"Crude oil (USO) down {uso_drop:.1f}% from recent high — approaching breakeven",
                    "description": checks["oil_below_60"],
                    "value": round(uso_drop, 1)
                })

    if "natgas_collapse" in checks:
        ung_data = macro_data.get("UNG")
        if ung_data is not None and len(ung_data) > 20:
            recent_high = ung_data["High"].iloc[-60:].max() if len(ung_data) >= 60 else ung_data["High"].max()
            current = ung_data["Close"].iloc[-1]
            ung_drop = ((current - recent_high) / recent_high) * 100
            if ung_drop < -30:
                alerts.append({
                    "check": "natgas_collapse",
                    "severity": "high",
                    "triggered": True,
                    "message": f"Natural gas (UNG) down {ung_drop:.1f}% from recent high",
                    "description": checks["natgas_collapse"],
                    "value": round(ung_drop, 1)
                })

    if "energy_spike" in checks:
        xle_data = macro_data.get("XLE")
        if xle_data is not None and len(xle_data) > 20:
            xle_ytd = compute_ytd_return_v1(xle_data)
            if xle_ytd > 15:
                alerts.append({
                    "check": "energy_spike",
                    "severity": "medium",
                    "triggered": True,
                    "message": f"Energy (XLE) up {xle_ytd:.1f}% YTD — cost pressure on chemical margins",
                    "description": checks["energy_spike"],
                    "value": round(xle_ytd, 1)
                })

    # Build untriggered checks list (for "all clear" display)
    triggered_ids = {a["check"] for a in alerts}
    for check_id, desc in checks.items():
        if check_id not in triggered_ids:
            alerts.append({
                "check": check_id,
                "severity": "clear",
                "triggered": False,
                "message": "Not triggered",
                "description": desc,
                "value": None
            })

    return alerts, coverage


# ============================================================
# SIGNAL SCORING
# ============================================================
# ---------------------------------------------------------------------------
# Score component functions (PER-508 item 19)
#
# The ONLY implementation of the scoring formula. score_stock_v2() computes
# indicators from price data and delegates every point decision here;
# /api/score/simulate (Score Lab) feeds user inputs through the same
# functions. The user's hand-built calculator prototype drifted from
# production before it shipped (pre-July-3 overextension branch: 81 vs 76)
# — a second copy of these branches must never exist.
# ---------------------------------------------------------------------------

SCORE_BASE = 50
QUALIFIER_GATE = 50   # universe qualifier line (see docs/scoring.md)

# Simulator vocabulary for the MACD component: (bullish, momentum_confirms).
# "confirms" means the histogram moved WITH the cross side — rising while
# bullish, falling while bearish.
MACD_STATES = {
    "bullish_rising":  (True, True),      # +13
    "bullish":         (True, False),     # +8
    "bearish":         (False, False),    # -8
    "bearish_falling": (False, True),     # -13
}


def score_rsi_points(rsi):
    if rsi < 30:
        return 15
    if rsi < 40:
        return 8
    if rsi < 60:
        return 3
    if rsi < 70:
        return -3
    if rsi < 80:
        return -8
    return -15


def score_macd_points(bullish, momentum_confirms):
    if bullish:
        return 8 + (5 if momentum_confirms else 0)
    return -8 - (5 if momentum_confirms else 0)


def score_ma_points(above_ma20, above_ma50, ma20_gt_ma50):
    return (4 if above_ma20 else -4) \
        + (6 if above_ma50 else -6) \
        + (4 if ma20_gt_ma50 else -4)


def score_ytd_points_v1(ytd_return):
    """FROZEN (D-020a). The pre-D-020a ladder: peaks at 12 on (20,50],
    tapers to 8 on (50,100], then the overextension penalties (-10
    above 100, -15 above 150). Kept verbatim for the committed studies'
    parity pins; production scores with score_ytd_points_v2."""
    if ytd_return > 50:
        pts = 8
    elif ytd_return > 20:
        pts = 12
    elif ytd_return > 5:
        pts = 6
    elif ytd_return > 0:
        pts = 2
    elif ytd_return > -10:
        pts = -4
    else:
        pts = -10

    # Overextension penalty: >150 checked first (was dead code below the
    # >100 branch — fixed 2026-07-03, mega-winners now -15 not -10)
    if ytd_return > 150:
        pts -= 15
    elif ytd_return > 100:
        pts -= 10
    return pts


def score_ytd_points_v2(ytd_return):
    """D-020a cap: rises exactly as v1 up to the ladder's PEAK (12 on
    entering >20), then holds FLAT — no taper at >50, no penalties at
    >100/>150. The prereg's plateau rule: the v1 curve was ALREADY
    declining before the 100% boundary (12 -> 8 at >50), so the plateau
    sits at the curve's PEAK value (12), not at 8, which is a point
    already on the way down. Never reverses, never resumes rising, and
    no value anywhere exceeds v1's maximum. Downside ladder unchanged.
    """
    if ytd_return > 20:
        return 12
    if ytd_return > 5:
        return 6
    if ytd_return > 0:
        return 2
    if ytd_return > -10:
        return -4
    return -10


def score_vol_points(vol_ratio):
    if vol_ratio > 1.5:
        return 3
    if vol_ratio < 0.7:
        return -3
    return 0


def compose_score(components):
    """Clamped composite from the per-component points dict."""
    return max(0, min(100, SCORE_BASE + sum(components.values())))


def score_signal_band(score):
    if score >= 75:
        return "strong-buy"
    if score >= 60:
        return "buy"
    if score >= 45:
        return "hold"
    if score >= 30:
        return "sell"
    return "strong-sell"


def simulate_score(rsi, macd_state, above_ma20, above_ma50, ma20_gt_ma50,
                   ytd_pct, vol_ratio):
    """Score Lab entry point: user inputs -> the real scoring functions.

    Returns (score, band, components) exactly as score_stock would produce
    for a stock exhibiting these indicator values.
    """
    bullish, confirms = MACD_STATES[macd_state]
    components = {
        "rsi": score_rsi_points(rsi),
        "macd": score_macd_points(bullish, confirms),
        "ma": score_ma_points(above_ma20, above_ma50, ma20_gt_ma50),
        "ytd": score_ytd_points_v2(ytd_pct),
        "vol": score_vol_points(vol_ratio),
    }
    score = compose_score(components)
    return score, score_signal_band(score), components


def simulate_score_v1(rsi, macd_state, above_ma20, above_ma50, ma20_gt_ma50,
                   ytd_pct, vol_ratio):
    """FROZEN (D-020a): replays v1-era payloads — artifacts baked before the scorer was versioned reproduce through THIS, never through the v2 simulator.

    Score Lab entry point: user inputs -> the real scoring functions.

    Returns (score, band, components) exactly as score_stock would produce
    for a stock exhibiting these indicator values.
    """
    bullish, confirms = MACD_STATES[macd_state]
    components = {
        "rsi": score_rsi_points(rsi),
        "macd": score_macd_points(bullish, confirms),
        "ma": score_ma_points(above_ma20, above_ma50, ma20_gt_ma50),
        "ytd": score_ytd_points_v1(ytd_pct),
        "vol": score_vol_points(vol_ratio),
    }
    score = compose_score(components)
    return score, score_signal_band(score), components


def score_stock_v1(df, group_info=None):
    """
    FROZEN (D-020a). The pre-D-020a scorer, verbatim: YTD anchored by
    compute_ytd_return_v1 on whatever frame it is handed (production
    fed it a 6mo frame) and pointed by the score_ytd_points_v1 ladder
    with its >100/>150 penalties. Exists SOLELY so the committed
    studies keep reproducing — Layer A's ladder-parity pin and the
    frozen-anchor pins drive it. Never called by production; production
    calls score_stock_v2. The duplication with v2 below is deliberate:
    a shared body would let future edits mutate the frozen scorer
    silently.
    """
    close = df["Close"]
    details = {}
    components = {}

    # --- RSI ---
    rsi = compute_rsi(close)
    current_rsi = rsi.iloc[-1] if not pd.isna(rsi.iloc[-1]) else 50
    details["rsi"] = round(current_rsi, 1)
    components["rsi"] = score_rsi_points(current_rsi)

    # --- MACD ---
    macd_line, signal_line, histogram = compute_macd(close)
    current_macd = macd_line.iloc[-1] if not pd.isna(macd_line.iloc[-1]) else 0
    current_signal = signal_line.iloc[-1] if not pd.isna(signal_line.iloc[-1]) else 0
    current_hist = histogram.iloc[-1] if not pd.isna(histogram.iloc[-1]) else 0
    prev_hist = histogram.iloc[-2] if len(histogram) > 1 and not pd.isna(histogram.iloc[-2]) else 0

    details["macd"] = round(current_macd, 3)
    details["macd_signal"] = round(current_signal, 3)
    details["macd_histogram"] = round(current_hist, 3)

    bullish = current_macd > current_signal
    confirms = (current_hist > prev_hist) if bullish else (current_hist < prev_hist)
    macd_state = next(k for k, v in MACD_STATES.items() if v == (bullish, confirms))
    components["macd"] = score_macd_points(bullish, confirms)

    # --- Moving Averages ---
    # NOTE (known bias, kept as-is): when MA20/MA50 are NaN (fewer than
    # 20/50 bars), they default to current_price, so all three comparisons
    # below fail and a young listing takes the full -14. In practice the
    # universe's 90-day history gate keeps such names out.
    ma20, ma50, ma200 = compute_moving_averages(close)
    current_price = close.iloc[-1]
    ma20_val = ma20.iloc[-1] if not pd.isna(ma20.iloc[-1]) else current_price
    ma50_val = ma50.iloc[-1] if not pd.isna(ma50.iloc[-1]) else current_price
    ma200_val = ma200.iloc[-1] if not pd.isna(ma200.iloc[-1]) else None

    details["price"] = round(current_price, 2)
    details["ma20"] = round(ma20_val, 2)
    details["ma50"] = round(ma50_val, 2)
    details["ma200"] = round(ma200_val, 2) if ma200_val else None

    components["ma"] = score_ma_points(current_price > ma20_val,
                                       current_price > ma50_val,
                                       ma20_val > ma50_val)

    # --- YTD Momentum (FROZEN v1: frame-anchored, penalty ladder) ---
    ytd_return = compute_ytd_return_v1(df)
    details["ytd_return"] = ytd_return
    components["ytd"] = score_ytd_points_v1(ytd_return)

    # --- Volume ---
    vol_ratio = compute_volume_trend(df)
    details["volume_ratio"] = vol_ratio
    components["vol"] = score_vol_points(vol_ratio)

    # --- Momentum Metrics ---
    momentum = compute_momentum_metrics(df)
    details.update(momentum)

    score = compose_score(components)
    details["composite_score"] = score
    details["score_components"] = components

    # The EXACT simulate_score inputs this run scored — Score Lab seeds from
    # these verbatim, so a pre-seeded ticker reproduces its score by
    # construction. The display fields above are rounded AFTER scoring
    # (rsi 1dp, price/MAs 2dp) and can flip a branch at a boundary; these
    # cannot. ytd/vol are rounded inside their compute functions pre-scoring,
    # so they are already exact.
    details["score_inputs"] = {
        "rsi": float(current_rsi),
        "macd_state": macd_state,
        "above_ma20": bool(current_price > ma20_val),
        "above_ma50": bool(current_price > ma50_val),
        "ma20_gt_ma50": bool(ma20_val > ma50_val),
        "ytd_pct": ytd_return,
        "vol_ratio": vol_ratio,
    }

    details["scorer_version"] = "score_stock_v1"
    signal = score_signal_band(score)
    details["signal"] = signal
    return score, signal, details


def score_stock_v2(df, group_info=None, ytd_return=None, ytd_basis=None):
    """
    Compute a composite score (0-100) and signal for a stock (D-020a).

    Additive from a base of 50 across five components; per-component points
    are returned in details["score_components"] ({rsi, macd, ma, ytd, vol})
    for the dashboard's score-breakdown tooltip. See docs/scoring.md.
    All point decisions live in the score_*_points functions above — shared
    with /api/score/simulate, never duplicated.

    D-020a: the YTD component is REAL calendar year-to-date (prior-year
    last close anchor) pointed by the capped v2 ladder. Callers whose df
    is an indicator frame too short to reach the prior year (production's
    6mo fetch) pass ytd_return/ytd_basis computed from a wider frame via
    compute_ytd_return_v2; every other indicator still reads the df it
    always read — the indicator windows are untouched by D-020a. When no
    override is given, the YTD is computed from df itself (correct for
    the universe builder's 1y frames) and the basis is recorded either
    way in details["ytd_basis"].
    """
    close = df["Close"]
    details = {}
    components = {}

    # --- RSI ---
    rsi = compute_rsi(close)
    current_rsi = rsi.iloc[-1] if not pd.isna(rsi.iloc[-1]) else 50
    details["rsi"] = round(current_rsi, 1)
    components["rsi"] = score_rsi_points(current_rsi)

    # --- MACD ---
    macd_line, signal_line, histogram = compute_macd(close)
    current_macd = macd_line.iloc[-1] if not pd.isna(macd_line.iloc[-1]) else 0
    current_signal = signal_line.iloc[-1] if not pd.isna(signal_line.iloc[-1]) else 0
    current_hist = histogram.iloc[-1] if not pd.isna(histogram.iloc[-1]) else 0
    prev_hist = histogram.iloc[-2] if len(histogram) > 1 and not pd.isna(histogram.iloc[-2]) else 0

    details["macd"] = round(current_macd, 3)
    details["macd_signal"] = round(current_signal, 3)
    details["macd_histogram"] = round(current_hist, 3)

    bullish = current_macd > current_signal
    confirms = (current_hist > prev_hist) if bullish else (current_hist < prev_hist)
    macd_state = next(k for k, v in MACD_STATES.items() if v == (bullish, confirms))
    components["macd"] = score_macd_points(bullish, confirms)

    # --- Moving Averages ---
    # NOTE (known bias, kept as-is): when MA20/MA50 are NaN (fewer than
    # 20/50 bars), they default to current_price, so all three comparisons
    # below fail and a young listing takes the full -14. In practice the
    # universe's 90-day history gate keeps such names out.
    ma20, ma50, ma200 = compute_moving_averages(close)
    current_price = close.iloc[-1]
    ma20_val = ma20.iloc[-1] if not pd.isna(ma20.iloc[-1]) else current_price
    ma50_val = ma50.iloc[-1] if not pd.isna(ma50.iloc[-1]) else current_price
    ma200_val = ma200.iloc[-1] if not pd.isna(ma200.iloc[-1]) else None

    details["price"] = round(current_price, 2)
    details["ma20"] = round(ma20_val, 2)
    details["ma50"] = round(ma50_val, 2)
    details["ma200"] = round(ma200_val, 2) if ma200_val else None

    components["ma"] = score_ma_points(current_price > ma20_val,
                                       current_price > ma50_val,
                                       ma20_val > ma50_val)

    # --- YTD Momentum (D-020a: real calendar YTD, capped ladder) ---
    if ytd_return is None:
        ytd_return, ytd_basis = compute_ytd_return_v2(df, with_basis=True)
    details["ytd_return"] = ytd_return
    details["ytd_basis"] = ytd_basis
    components["ytd"] = score_ytd_points_v2(ytd_return)

    # --- Volume ---
    vol_ratio = compute_volume_trend(df)
    details["volume_ratio"] = vol_ratio
    components["vol"] = score_vol_points(vol_ratio)

    # --- Momentum Metrics ---
    momentum = compute_momentum_metrics(df)
    details.update(momentum)

    score = compose_score(components)
    details["composite_score"] = score
    details["score_components"] = components

    # The EXACT simulate_score inputs this run scored — Score Lab seeds from
    # these verbatim, so a pre-seeded ticker reproduces its score by
    # construction. The display fields above are rounded AFTER scoring
    # (rsi 1dp, price/MAs 2dp) and can flip a branch at a boundary; these
    # cannot. ytd/vol are rounded inside their compute functions pre-scoring,
    # so they are already exact.
    details["score_inputs"] = {
        "rsi": float(current_rsi),
        "macd_state": macd_state,
        "above_ma20": bool(current_price > ma20_val),
        "above_ma50": bool(current_price > ma50_val),
        "ma20_gt_ma50": bool(ma20_val > ma50_val),
        "ytd_pct": ytd_return,
        "vol_ratio": vol_ratio,
    }

    details["scorer_version"] = "score_stock_v2"
    signal = score_signal_band(score)
    details["signal"] = signal
    return score, signal, details


def scorer_era(artifact):
    """Era label for a baked artifact (D-020a): artifacts written before
    the scorer was versioned carry no stamp and are v1-era BY OMISSION —
    they are never retro-relabelled as v2."""
    return artifact.get("scorer_version") or "score_stock_v1-era"


def compute_grade_inputs(df, ytd_df=None):
    """
    D-017 emission: the per-row scalars the framework runner feeds to the
    D-011 grade (grade_setup) for un-tracked candidates.

    D-018 reconciliation: stripping the SYNTHETIC bar was never the whole
    close-basis law — a live-quote row (Volume 0) went, but the day's
    FORMING bar has real volume and stayed, so an intraday signal run
    graded candidates on a partial bar. Both grade paths now apply
    `confirmed_close_frame`, the same splitter the regime hysteresis and
    the position ladder use: grades are computed on CONFIRMED closes on
    every path, on every run.

    The row's DISPLAY fields (price/rsi/score) deliberately stay live —
    the dashboard's job is the tape as it is now. Only the graded inputs
    are close-basis, and the divergence is the honest one: opinions are
    graded on closes.

    Deferred imports — the framework package imports this module the
    other way, also deferred.
    """
    from framework.position_signals import (grade_inputs_from_df,
                                            strip_synthetic_last_bar)
    from framework.regime_calculator import confirmed_close_frame
    sdf = strip_synthetic_last_bar(df)
    sdf, _forming = confirmed_close_frame(sdf)
    gi = grade_inputs_from_df(sdf)
    if gi is None:
        return None
    r = compute_rsi(sdf["Close"]).iloc[-1]
    gi["rsi14"] = float(r) if np.isfinite(r) else None
    # D-020a: the graded quality_score uses REAL YTD, and — per D-018 —
    # on CONFIRMED closes: the wider ytd frame goes through the same
    # synthetic-strip + confirmed-close splitter as the graded df. With
    # no wider frame the v2 fallback anchors on sdf and says so in the
    # recorded basis rather than silently keeping the rolling label.
    ytd_v = ytd_b = None
    if ytd_df is not None and len(ytd_df):
        sydf = strip_synthetic_last_bar(ytd_df)
        sydf, _yf = confirmed_close_frame(sydf)
        if sydf is not None and len(sydf):
            ytd_v, ytd_b = compute_ytd_return_v2(sydf, with_basis=True)
    s, _sig, _det = score_stock_v2(sdf, ytd_return=ytd_v, ytd_basis=ytd_b)
    gi["quality_score"] = None if s is None else float(s)
    gi["ytd_basis"] = _det.get("ytd_basis")
    return gi


def compute_trade_signal(details, breaker_status="clear"):
    """
    Compute an actionable trade signal and reasoning for position trading.
    Uses all available technical indicators to determine entry timing.

    Returns:
        trade_signal: "BUY NOW", "WAIT FOR PULLBACK", "ACCUMULATE ON DIP",
                      "HOLD POSITION", "REDUCE/EXIT", "AVOID"
        trade_reasoning: human-readable explanation
    """
    rsi = details.get("rsi", 50)
    macd_hist = details.get("macd_histogram", 0)
    macd = details.get("macd", 0)
    macd_sig = details.get("macd_signal", 0)
    price = details.get("price", 0)
    ma20 = details.get("ma20", price)
    ma50 = details.get("ma50", price)
    ma200 = details.get("ma200")
    score = details.get("composite_score", 50)
    signal = details.get("signal", "hold")
    ytd = details.get("ytd_return", 0)
    vol_ratio = details.get("volume_ratio", 1.0)
    pct_from_high = details.get("pct_from_52w_high", 0)
    trend_strength = details.get("trend_strength", 0)
    rs_vs_ma50 = details.get("rs_vs_ma50", 0)
    return_1m = details.get("return_1m", 0)

    reasons = []
    bullish = 0
    bearish = 0

    # --- Breaker check (overrides everything) ---
    if breaker_status == "degraded":
        reasons.append("Group breaker UNVERIFIED — some checks could not be "
                       "computed this run")
    elif breaker_status in ("critical", "warning"):
        reasons.append(f"Thesis breaker {breaker_status.upper()} — macro headwinds active")
        bearish += 3

    # --- Trend structure ---
    above_ma50 = price > ma50
    above_ma200 = ma200 is not None and price > ma200
    ma_aligned = ma20 > ma50  # short MA above long MA = uptrend

    if above_ma50 and ma_aligned:
        bullish += 2
        reasons.append("Price above MA50, MAs aligned bullish")
    elif above_ma50:
        bullish += 1
        reasons.append("Price above MA50 but MAs converging")
    else:
        bearish += 2
        reasons.append("Price below MA50 — trend weakening")

    if above_ma200:
        bullish += 1
        reasons.append("Above MA200 — long-term uptrend intact")
    elif ma200 is not None:
        bearish += 1
        reasons.append("Below MA200 — long-term trend broken")

    # --- RSI assessment ---
    if rsi >= 75:
        bearish += 2
        reasons.append(f"RSI {rsi:.0f} — overbought, high pullback risk")
    elif rsi >= 65:
        bearish += 1
        reasons.append(f"RSI {rsi:.0f} — getting extended")
    elif 40 <= rsi <= 60:
        bullish += 1
        reasons.append(f"RSI {rsi:.0f} — neutral zone, room to run")
    elif rsi <= 35:
        bullish += 2
        reasons.append(f"RSI {rsi:.0f} — oversold, potential bounce")

    # --- MACD momentum ---
    macd_bullish_cross = macd > macd_sig
    hist_increasing = macd_hist > 0

    if macd_bullish_cross and hist_increasing:
        bullish += 2
        reasons.append("MACD bullish cross, histogram expanding")
    elif macd_bullish_cross:
        bullish += 1
        reasons.append("MACD above signal but histogram fading")
    elif not macd_bullish_cross and macd_hist < 0:
        bearish += 2
        reasons.append("MACD bearish, histogram negative")

    # --- Volume confirmation ---
    if vol_ratio >= 1.5:
        bullish += 1
        reasons.append(f"Volume {vol_ratio:.1f}x avg — institutional interest")
    elif vol_ratio <= 0.7:
        bearish += 1
        reasons.append(f"Volume {vol_ratio:.1f}x avg — low conviction")

    # --- Proximity to 52W high ---
    if pct_from_high is not None:
        if pct_from_high > -3:
            bearish += 1
            reasons.append(f"{pct_from_high:.1f}% from 52W high — near resistance")
        elif pct_from_high < -20:
            bearish += 1
            reasons.append(f"{pct_from_high:.1f}% from 52W high — deep correction")
        elif -15 <= pct_from_high <= -5:
            bullish += 1
            reasons.append(f"{pct_from_high:.1f}% from 52W high — healthy pullback zone")

    # --- Trend strength ---
    if trend_strength >= 16:
        bullish += 1
        reasons.append(f"Trend strength {trend_strength}/20 — strong sustained uptrend")
    elif trend_strength <= 5:
        bearish += 1
        reasons.append(f"Trend strength {trend_strength}/20 — no uptrend present")

    # --- 1M return (recent momentum) ---
    if return_1m is not None:
        if return_1m > 15:
            bearish += 1
            reasons.append(f"1M return +{return_1m:.1f}% — parabolic, needs cooling")
        elif return_1m < -10:
            reasons.append(f"1M return {return_1m:.1f}% — sharp decline, watch for base")

    # --- Determine trade signal ---
    net = bullish - bearish

    # AN OUTAGE NEVER IMPERSONATES SAFETY, IN THE ARTIFACT TOO (D-019).
    # "degraded" means some called-for check never ran, so every signal
    # below would rest on a PRESUMED clear breaker. Withhold it here, at
    # the point of publication — the dashboard reads trade_signal straight
    # out of signals.json and would otherwise render a BUY NOW that the
    # search page correctly refuses to show. Changing the VALUE (rather
    # than adding a flag) is deliberate: an older cached page renders an
    # unfamiliar label harmlessly, whereas it would ignore a new flag and
    # print the false BUY NOW.
    if breaker_status == "degraded":
        return "SIGNAL WITHHELD", ("Group breaker unverified — some checks "
                                   "could not be computed this run, so no "
                                   "trade signal is published for this group.")
    if breaker_status == "critical":
        trade_signal = "AVOID"
    elif breaker_status == "warning" and net <= 0:
        trade_signal = "AVOID"
    elif signal in ("sell", "strong-sell"):
        if net <= -2:
            trade_signal = "AVOID"
        else:
            trade_signal = "REDUCE/EXIT"
    elif signal == "hold":
        if net >= 2:
            trade_signal = "ACCUMULATE ON DIP"
        elif net <= -1:
            trade_signal = "REDUCE/EXIT"
        else:
            trade_signal = "HOLD POSITION"
    elif signal in ("buy", "strong-buy"):
        # Strong buy/buy signal — now determine timing
        if rsi >= 70 and pct_from_high is not None and pct_from_high > -5:
            trade_signal = "WAIT FOR PULLBACK"
        elif rsi <= 60 and macd_bullish_cross and above_ma50:
            trade_signal = "BUY NOW"
        elif rsi <= 65 and hist_increasing and above_ma50:
            trade_signal = "BUY NOW"
        elif vol_ratio >= 1.5 and macd_bullish_cross:
            trade_signal = "BUY NOW"  # breakout on volume
        elif net >= 3:
            trade_signal = "BUY NOW"
        elif rsi >= 65:
            trade_signal = "WAIT FOR PULLBACK"
        else:
            trade_signal = "ACCUMULATE ON DIP"
    else:
        trade_signal = "HOLD POSITION"

    # Build concise reasoning (top 3 most relevant)
    top_reasons = reasons[:4]
    trade_reasoning = "; ".join(top_reasons)

    return trade_signal, trade_reasoning


# ------------------------------------------------------------------
# SWING TRADE SIGNAL (2-10 day holds)
# ------------------------------------------------------------------
def compute_swing_trade_signal(details, df):
    """
    Compute swing trade signal with specific entry/stop/target prices.
    Swing trading = 2-10 day holds focused on mean reversion and short-term momentum.

    Returns dict: {signal, reasoning, entry_price, stop_loss, target_price, risk_reward}
    """
    rsi = details.get("rsi", 50)
    macd_hist = details.get("macd_histogram", 0)
    macd = details.get("macd", 0)
    macd_sig = details.get("macd_signal", 0)
    price = details.get("price", 0)
    ma20 = details.get("ma20", price)
    ma50 = details.get("ma50", price)
    vol_ratio = details.get("volume_ratio", 1.0)
    pct_from_high = details.get("pct_from_52w_high", 0)
    trend_strength = details.get("trend_strength", 0)

    if price == 0 or df is None or len(df) < 10:
        return {
            "signal": "NO SETUP", "reasoning": "Insufficient data",
            "entry_price": None, "stop_loss": None, "target_price": None, "risk_reward": None
        }

    # --- Compute swing-specific price levels from recent data ---
    recent = df.tail(10)
    swing_low_5 = float(df.tail(5)["Low"].min())
    swing_low_10 = float(recent["Low"].min())
    swing_high_10 = float(recent["High"].max())
    swing_high_5 = float(df.tail(5)["High"].max())

    # ATR for buffer calculations
    if len(df) >= 14:
        highs = df["High"].values[-14:]
        lows = df["Low"].values[-14:]
        closes = df["Close"].values[-15:-1]
        tr = []
        for i in range(len(highs)):
            tr_val = max(float(highs[i]) - float(lows[i]),
                         abs(float(highs[i]) - float(closes[i])) if i < len(closes) else 0,
                         abs(float(lows[i]) - float(closes[i])) if i < len(closes) else 0)
            tr.append(tr_val)
        atr = sum(tr) / len(tr) if tr else price * 0.02
    else:
        atr = price * 0.02

    reasons = []
    bullish = 0
    bearish = 0

    # --- MA20 is primary for swing trading ---
    above_ma20 = price > ma20
    if above_ma20:
        bullish += 1
        reasons.append(f"Price above MA20 ({ma20:.2f})")
    else:
        bearish += 1
        reasons.append(f"Price below MA20 ({ma20:.2f})")

    # Price near MA20 = pullback opportunity
    ma20_dist = abs(price - ma20) / price * 100
    if ma20_dist < 2 and above_ma20:
        bullish += 1
        reasons.append("Price near MA20 support — pullback entry zone")

    # --- RSI for swing (mean reversion focus) ---
    if rsi <= 35:
        bullish += 3
        reasons.append(f"RSI {rsi:.0f} — oversold bounce setup")
    elif rsi <= 45:
        bullish += 2
        reasons.append(f"RSI {rsi:.0f} — approaching oversold, favorable entry")
    elif rsi >= 75:
        bearish += 3
        reasons.append(f"RSI {rsi:.0f} — overbought, fade/short setup")
    elif rsi >= 65:
        bearish += 1
        reasons.append(f"RSI {rsi:.0f} — extended, risk of reversal")
    else:
        reasons.append(f"RSI {rsi:.0f} — neutral zone")

    # --- MACD histogram direction (key swing trigger) ---
    if len(df) >= 3:
        # Check if histogram is turning (direction change)
        try:
            ema12 = df["Close"].ewm(span=12).mean()
            ema26 = df["Close"].ewm(span=26).mean()
            macd_line = ema12 - ema26
            sig_line = macd_line.ewm(span=9).mean()
            hist = macd_line - sig_line
            hist_today = float(hist.iloc[-1])
            hist_yesterday = float(hist.iloc[-2])
            hist_turning_up = hist_today > hist_yesterday and hist_yesterday < float(hist.iloc[-3])
            hist_turning_down = hist_today < hist_yesterday and hist_yesterday > float(hist.iloc[-3])
        except Exception:
            hist_turning_up = False
            hist_turning_down = False
    else:
        hist_turning_up = False
        hist_turning_down = False

    if hist_turning_up:
        bullish += 2
        reasons.append("MACD histogram turning up — momentum shift bullish")
    elif hist_turning_down:
        bearish += 2
        reasons.append("MACD histogram turning down — momentum fading")
    elif macd > macd_sig:
        bullish += 1
        reasons.append("MACD above signal line")
    else:
        bearish += 1
        reasons.append("MACD below signal line")

    # --- Volume spike for conviction ---
    if vol_ratio >= 1.5:
        bullish += 1
        reasons.append(f"Volume {vol_ratio:.1f}x avg — high conviction move")
    elif vol_ratio <= 0.6:
        reasons.append(f"Volume {vol_ratio:.1f}x avg — low activity, weak setup")

    # --- Determine signal ---
    net = bullish - bearish

    if net >= 4 and rsi <= 45:
        signal = "BUY SWING"
    elif net >= 3 and hist_turning_up:
        signal = "BUY SWING"
    elif net >= 2 and above_ma20:
        signal = "WAIT FOR DIP"
    elif net <= -3 and rsi >= 70:
        signal = "FADE THE RALLY"
    elif net <= -2:
        signal = "EXIT SWING"
    elif net >= 1:
        signal = "HOLD SWING"
    elif net <= -1:
        signal = "EXIT SWING"
    else:
        signal = "NO SETUP"

    # --- Compute entry/stop/target ---
    buffer = atr * 0.3

    if signal in ("BUY SWING", "WAIT FOR DIP", "HOLD SWING"):
        # Long setup
        entry_price = round(min(ma20, swing_low_5 + buffer), 2)
        if entry_price > price * 1.02:
            entry_price = round(price, 2)  # Don't set entry too far above current
        stop_loss = round(swing_low_10 - buffer, 2)
        risk = entry_price - stop_loss
        if risk <= 0:
            risk = atr
        target_price = round(entry_price + (risk * 2), 2)  # 2:1 R/R target
        # Cap target at recent swing high if it's closer
        if swing_high_10 > entry_price and swing_high_10 < target_price:
            target_price = round(swing_high_10, 2)
        rr = round((target_price - entry_price) / risk, 1) if risk > 0 else 0
        risk_reward = f"1:{rr}"
    elif signal in ("FADE THE RALLY", "EXIT SWING"):
        # Short/exit setup
        entry_price = round(max(ma20, swing_high_5 - buffer), 2)
        if entry_price < price * 0.98:
            entry_price = round(price, 2)
        stop_loss = round(swing_high_10 + buffer, 2)
        risk = stop_loss - entry_price
        if risk <= 0:
            risk = atr
        target_price = round(entry_price - (risk * 2), 2)
        if swing_low_10 < entry_price and swing_low_10 > target_price:
            target_price = round(swing_low_10, 2)
        rr = round((entry_price - target_price) / risk, 1) if risk > 0 else 0
        risk_reward = f"1:{rr}"
    else:
        entry_price = None
        stop_loss = None
        target_price = None
        risk_reward = None

    reasoning = "; ".join(reasons[:4])

    return {
        "signal": signal,
        "reasoning": reasoning,
        "entry_price": entry_price,
        "stop_loss": stop_loss,
        "target_price": target_price,
        "risk_reward": risk_reward,
    }


# ------------------------------------------------------------------
# INTRADAY TRADE SIGNAL (same-day trades)
# ------------------------------------------------------------------
def compute_intraday_trade_signal(details, df):
    """
    Compute intraday trade signal with specific entry/stop/target prices.
    Uses previous day's levels, ATR, gap analysis, and momentum indicators.

    Returns dict: {signal, reasoning, entry_price, stop_loss, target_price, risk_reward}
    """
    rsi = details.get("rsi", 50)
    macd_hist = details.get("macd_histogram", 0)
    macd = details.get("macd", 0)
    macd_sig = details.get("macd_signal", 0)
    price = details.get("price", 0)
    vol_ratio = details.get("volume_ratio", 1.0)

    if price == 0 or df is None or len(df) < 15:
        return {
            "signal": "NO SETUP", "reasoning": "Insufficient data",
            "entry_price": None, "stop_loss": None, "target_price": None, "risk_reward": None
        }

    # --- Key intraday levels from previous days ---
    prev_high = float(df["High"].iloc[-2])
    prev_low = float(df["Low"].iloc[-2])
    prev_close = float(df["Close"].iloc[-2])
    today_open = float(df["Open"].iloc[-1])
    today_high = float(df["High"].iloc[-1])
    today_low = float(df["Low"].iloc[-1])

    # --- ATR(14) for volatility-based stops/targets ---
    highs = df["High"].values[-15:-1]
    lows = df["Low"].values[-15:-1]
    closes = df["Close"].values[-16:-2]
    tr_list = []
    for i in range(len(highs)):
        tr_val = max(float(highs[i]) - float(lows[i]),
                     abs(float(highs[i]) - float(closes[i])) if i < len(closes) else 0,
                     abs(float(lows[i]) - float(closes[i])) if i < len(closes) else 0)
        tr_list.append(tr_val)
    atr = sum(tr_list) / len(tr_list) if tr_list else price * 0.02

    # --- Gap analysis ---
    gap = today_open - prev_close
    gap_pct = (gap / prev_close) * 100 if prev_close > 0 else 0

    # --- Previous day range ---
    prev_range = prev_high - prev_low
    range_ratio = prev_range / atr if atr > 0 else 1

    reasons = []
    bullish = 0
    bearish = 0

    # --- Gap direction ---
    if gap_pct > 0.5:
        bullish += 1
        reasons.append(f"Gap up +{gap_pct:.1f}% — bullish opening")
    elif gap_pct < -0.5:
        bearish += 1
        reasons.append(f"Gap down {gap_pct:.1f}% — bearish opening")
    else:
        reasons.append(f"Flat open ({gap_pct:+.1f}%) — no gap bias")

    # --- Price vs previous day levels ---
    if price > prev_high:
        bullish += 2
        reasons.append(f"Price above prev high ({prev_high:.2f}) — breakout")
    elif price > prev_close:
        bullish += 1
        reasons.append(f"Price above prev close ({prev_close:.2f}) — bullish bias")
    elif price < prev_low:
        bearish += 2
        reasons.append(f"Price below prev low ({prev_low:.2f}) — breakdown")
    elif price < prev_close:
        bearish += 1
        reasons.append(f"Price below prev close ({prev_close:.2f}) — bearish bias")

    # --- RSI for intraday momentum ---
    if rsi >= 70:
        bearish += 1
        reasons.append(f"RSI {rsi:.0f} — overbought, reversal risk")
    elif rsi <= 30:
        bullish += 1
        reasons.append(f"RSI {rsi:.0f} — oversold, bounce likely")
    elif 40 <= rsi <= 60:
        bullish += 1
        reasons.append(f"RSI {rsi:.0f} — room for directional move")

    # --- MACD momentum ---
    macd_bullish = macd > macd_sig
    if macd_bullish and macd_hist > 0:
        bullish += 1
        reasons.append("MACD bullish with expanding histogram")
    elif not macd_bullish and macd_hist < 0:
        bearish += 1
        reasons.append("MACD bearish with contracting histogram")

    # --- Volume conviction ---
    if vol_ratio >= 1.5:
        bullish += 1 if price > prev_close else 0
        bearish += 1 if price < prev_close else 0
        reasons.append(f"Volume {vol_ratio:.1f}x avg — strong conviction")
    elif vol_ratio <= 0.5:
        reasons.append(f"Volume {vol_ratio:.1f}x avg — thin, avoid")

    # --- Volatility check ---
    if atr / price * 100 < 0.5:
        reasons.append(f"Low volatility (ATR {atr:.2f}) — tight range expected")
    elif atr / price * 100 > 3:
        reasons.append(f"High volatility (ATR {atr:.2f}) — widen stops")

    # --- Determine signal ---
    net = bullish - bearish

    if vol_ratio <= 0.5:
        signal = "RANGE BOUND"
    elif net >= 3 and price > prev_high:
        signal = "LONG ENTRY"
    elif net >= 2 and macd_bullish and price > prev_close:
        signal = "LONG ENTRY"
    elif net >= 2 and price <= prev_close:
        signal = "WAIT FOR BREAKOUT"
    elif net <= -3 and price < prev_low:
        signal = "SHORT ENTRY"
    elif net <= -2 and not macd_bullish and price < prev_close:
        signal = "SHORT ENTRY"
    elif net <= -2 and price >= prev_close:
        # SHORT-side setup (trigger = prev-low break). Named BREAKDOWN
        # WATCH: the old "WAIT FOR PULLBACK" collided with the position
        # ladder's long-side patience signal of the same name (PER-509).
        signal = "BREAKDOWN WATCH"
    elif abs(net) <= 1 and range_ratio < 0.8:
        signal = "RANGE BOUND"
    elif net >= 1:
        signal = "WAIT FOR BREAKOUT"
    elif net <= -1:
        signal = "BREAKDOWN WATCH"
    else:
        signal = "NO SETUP"

    # --- Compute entry/stop/target using ATR ---
    if signal == "LONG ENTRY":
        entry_price = round(max(prev_high, price), 2)  # Breakout above prev high
        stop_loss = round(entry_price - (1.5 * atr), 2)
        target_price = round(entry_price + (2.5 * atr), 2)  # ~1.7:1 R/R
        risk = entry_price - stop_loss
        reward = target_price - entry_price
        rr = round(reward / risk, 1) if risk > 0 else 0
        risk_reward = f"1:{rr}"
    elif signal == "SHORT ENTRY":
        entry_price = round(min(prev_low, price), 2)  # Breakdown below prev low
        stop_loss = round(entry_price + (1.5 * atr), 2)
        target_price = round(entry_price - (2.5 * atr), 2)
        risk = stop_loss - entry_price
        reward = entry_price - target_price
        rr = round(reward / risk, 1) if risk > 0 else 0
        risk_reward = f"1:{rr}"
    elif signal == "WAIT FOR BREAKOUT":
        entry_price = round(prev_high, 2)  # Trigger = prev high break
        stop_loss = round(prev_high - (1.5 * atr), 2)
        target_price = round(prev_high + (2 * atr), 2)
        risk = entry_price - stop_loss
        reward = target_price - entry_price
        rr = round(reward / risk, 1) if risk > 0 else 0
        risk_reward = f"1:{rr}"
    elif signal == "BREAKDOWN WATCH":
        entry_price = round(prev_low, 2)  # Trigger = prev low break
        stop_loss = round(prev_low + (1.5 * atr), 2)
        target_price = round(prev_low - (2 * atr), 2)
        risk = stop_loss - entry_price
        reward = entry_price - target_price
        rr = round(reward / risk, 1) if risk > 0 else 0
        risk_reward = f"1:{rr}"
    else:
        entry_price = None
        stop_loss = None
        target_price = None
        risk_reward = None

    reasoning = "; ".join(reasons[:4])

    return {
        "signal": signal,
        "reasoning": reasoning,
        "entry_price": entry_price,
        "stop_loss": stop_loss,
        "target_price": target_price,
        "risk_reward": risk_reward,
    }


# ------------------------------------------------------------------
# STAGE ANALYSIS (Weinstein Method)
# ------------------------------------------------------------------
def compute_stage_analysis(details, df):
    """
    Determine the current Weinstein Stage for a stock.
    Primary trend MA: 150-day (≈30-week) when the frame has ≥150 bars —
    but on the production 6-month frame (~124 bars) the fallback
    min(len, 100)-day MA ALWAYS fires, so what ships is effectively a
    100-day MA. Fields/labels are named ma100 accordingly (honest labels,
    PER-509); ma_period reports the actual window used. Display-only:
    never feeds score or trade signals (see docs/scoring.md).

    Stages:
        Stage 1 — Basing/Accumulation: Price consolidating near flat primary MA
        Stage 2 — Advancing/Markup:    Price above rising primary MA (ideal buy zone)
        Stage 3 — Topping/Distribution: Price struggling, primary MA flattening after rise
        Stage 4 — Declining/Markdown:   Price below falling primary MA (avoid/short)

    Returns dict: {stage, stage_name, description, confidence,
                   ma100, ma100_slope, price_vs_ma100_pct, ma_period, factors}
    """
    price = details.get("price", 0)
    ma50 = details.get("ma50", price)
    rsi = details.get("rsi", 50)
    vol_ratio = details.get("volume_ratio", 1.0)
    trend_strength = details.get("trend_strength", 10)

    if price == 0 or df is None or len(df) < 50:
        return {
            "stage": 0, "stage_name": "Unknown",
            "description": "Insufficient data for stage analysis",
            "confidence": "low", "factors": []
        }

    close = df["Close"]

    # --- Compute the primary trend MA ---
    # 150-day when the frame allows; the production 6mo frame (~124 bars)
    # always takes the fallback, so ma_period is 100 in practice.
    if len(close) >= 150:
        ma_period = 150
        ma150 = close.rolling(150).mean()
        ma150_current = float(ma150.iloc[-1])
        # Slope: compare current MA to 20 days ago
        ma150_20ago = float(ma150.iloc[-21]) if len(ma150) > 20 and not pd.isna(ma150.iloc[-21]) else ma150_current
        ma150_slope = (ma150_current - ma150_20ago) / ma150_20ago * 100 if ma150_20ago != 0 else 0
    else:
        ma_period = min(len(close), 100)
        ma150_series = close.rolling(ma_period).mean()
        ma150_current = float(ma150_series.iloc[-1]) if not pd.isna(ma150_series.iloc[-1]) else price
        ma150_20ago = float(ma150_series.iloc[-21]) if len(ma150_series) > 20 and not pd.isna(ma150_series.iloc[-21]) else ma150_current
        ma150_slope = (ma150_current - ma150_20ago) / ma150_20ago * 100 if ma150_20ago != 0 else 0
    ma_label = f"MA{ma_period}"

    # --- Price position relative to MA150 ---
    price_vs_ma150_pct = (price - ma150_current) / ma150_current * 100 if ma150_current != 0 else 0

    # --- MA50 slope (shorter-term confirmation) ---
    ma50_series = close.rolling(50).mean()
    if len(ma50_series) > 20 and not pd.isna(ma50_series.iloc[-21]):
        ma50_20ago = float(ma50_series.iloc[-21])
        ma50_slope = (ma50 - ma50_20ago) / ma50_20ago * 100 if ma50_20ago != 0 else 0
    else:
        ma50_slope = 0

    # --- Volatility / Range compression for basing detection ---
    if len(close) >= 30:
        recent_30 = close.iloc[-30:]
        range_pct = (float(recent_30.max()) - float(recent_30.min())) / float(recent_30.mean()) * 100
    else:
        range_pct = 10  # default moderate

    # --- Stage determination ---
    factors = []
    stage_scores = {1: 0, 2: 0, 3: 0, 4: 0}

    # Price vs MA150
    if price_vs_ma150_pct > 5:
        stage_scores[2] += 3
        factors.append(f"Price {price_vs_ma150_pct:+.1f}% above {ma_label} — bullish positioning")
    elif price_vs_ma150_pct > 0:
        stage_scores[2] += 1
        stage_scores[1] += 1
        factors.append(f"Price {price_vs_ma150_pct:+.1f}% above {ma_label} — near support")
    elif price_vs_ma150_pct > -3:
        stage_scores[3] += 2
        stage_scores[1] += 1
        factors.append(f"Price {price_vs_ma150_pct:+.1f}% near {ma_label} — testing support")
    else:
        stage_scores[4] += 3
        factors.append(f"Price {price_vs_ma150_pct:+.1f}% below {ma_label} — bearish positioning")

    # MA150 slope (primary trend)
    if ma150_slope > 0.5:
        stage_scores[2] += 3
        factors.append(f"{ma_label} rising ({ma150_slope:+.2f}%) — uptrend confirmed")
    elif ma150_slope > -0.2:
        stage_scores[1] += 2
        stage_scores[3] += 2
        factors.append(f"{ma_label} flat ({ma150_slope:+.2f}%) — consolidation/transition")
    else:
        stage_scores[4] += 3
        factors.append(f"{ma_label} falling ({ma150_slope:+.2f}%) — downtrend confirmed")

    # MA50 slope (shorter-term momentum)
    if ma50_slope > 1.0:
        stage_scores[2] += 2
        factors.append(f"MA50 strongly rising ({ma50_slope:+.1f}%)")
    elif ma50_slope > 0:
        stage_scores[2] += 1
        factors.append(f"MA50 rising ({ma50_slope:+.1f}%)")
    elif ma50_slope > -1.0:
        stage_scores[3] += 1
        factors.append(f"MA50 flattening ({ma50_slope:+.1f}%)")
    else:
        stage_scores[4] += 2
        factors.append(f"MA50 falling ({ma50_slope:+.1f}%)")

    # Range compression (basing detection)
    if range_pct < 8 and abs(ma150_slope) < 0.5:
        stage_scores[1] += 2
        factors.append(f"Tight 30-day range ({range_pct:.1f}%) — basing pattern")

    # Volume pattern
    if vol_ratio >= 1.5 and price > ma150_current:
        stage_scores[2] += 1
        factors.append(f"High volume ({vol_ratio:.1f}x) above {ma_label} — accumulation")
    elif vol_ratio >= 1.5 and price < ma150_current:
        stage_scores[4] += 1
        factors.append(f"High volume ({vol_ratio:.1f}x) below {ma_label} — distribution")

    # Trend strength confirmation
    if trend_strength >= 16:
        stage_scores[2] += 1
    elif trend_strength <= 5:
        stage_scores[4] += 1

    # --- Determine winning stage ---
    stage = max(stage_scores, key=stage_scores.get)
    max_score = stage_scores[stage]
    total = sum(stage_scores.values()) or 1

    # Confidence based on dominance
    dominance = max_score / total
    if dominance > 0.5:
        confidence = "high"
    elif dominance > 0.35:
        confidence = "medium"
    else:
        confidence = "low"

    # Stage names and descriptions
    stage_info = {
        1: ("Basing", f"Accumulation phase — price consolidating near flat {ma_label}. Watch for breakout above {ma_label} with volume."),
        2: ("Advancing", f"Markup phase — price above rising {ma_label}. Ideal zone for long positions. Ride the trend."),
        3: ("Topping", f"Distribution phase — {ma_label} flattening after advance. Consider tightening stops or taking profits."),
        4: ("Declining", f"Markdown phase — price below falling {ma_label}. Avoid new longs. Consider short positions."),
    }

    name, desc = stage_info[stage]

    return {
        "stage": stage,
        "stage_name": name,
        "description": desc,
        "confidence": confidence,
        "ma100": round(ma150_current, 2),
        "ma100_slope": round(ma150_slope, 3),
        "price_vs_ma100_pct": round(price_vs_ma150_pct, 2),
        "ma_period": ma_period,
        "factors": factors[:4],
    }


# ============================================================
# MAIN PIPELINE
# ============================================================
INDEX_TICKERS = {
    "^GSPC": "S&P 500",
    "^IXIC": "NASDAQ",
    "^DJI": "DOW 30",
    "^RUT": "Russell 2000",
    "^VIX": "VIX",
    # VIX term structure (PER-508 producer amendment): consumed by
    # /api/assessment.json vol_complex. A flaky fetch degrades to the
    # endpoint's null+note — never blocks the run (SKIP branch below).
    "^VIX9D": "VIX 9-Day",
    "^VIX3M": "VIX 3-Month",
}


def get_index_data():
    """Fetch YTD return and current level for major indexes."""
    print("Fetching market indexes...")
    indexes = {}
    for ticker, name in INDEX_TICKERS.items():
        print(f"  {name} ({ticker})...", end=" ")
        # D-020a: 1y frame so the YTD anchor reaches the prior-year
        # close; level/day-change/avg_5d read the last bars and are
        # frame-invariant
        df = fetch_data(ticker, period="1y")
        if df is not None and len(df) > 5:
            ytd = compute_ytd_return_v2(df)
            level = round(float(df["Close"].iloc[-1]), 2)
            prev_close = round(float(df["Close"].iloc[-2]), 2) if len(df) > 1 else level
            day_change = round(level - prev_close, 2)
            day_change_pct = round((day_change / prev_close) * 100, 2) if prev_close else 0
            indexes[ticker] = {
                "name": name,
                "level": level,
                "ytd": ytd,
                "day_change": day_change,
                "day_change_pct": day_change_pct,
                # 5-day average — the vol_complex read for the VIX family,
                # uniform across all indexes
                "avg_5d": round(float(df["Close"].iloc[-5:].mean()), 2),
            }
            print(f"OK — {level:,.2f} (YTD {ytd}%)")
        else:
            print("SKIP")
    return indexes


def get_sp500_ytd():
    print("Fetching S&P 500 benchmark...")
    # D-020a: 1y frame -> real prior-year-close anchor; beating_sp500
    # compares stock YTD (v2) against this, so the anchors must match
    df = fetch_data("^GSPC", period="1y")
    if df is not None:
        return compute_ytd_return_v2(df)
    return 0.0


def run_engine():
    """Main entry point — runs the full signal pipeline."""
    timestamp = datetime.datetime.now(datetime.timezone.utc).isoformat()
    print(f"\n{'='*60}")
    print(f"Signal Engine Run — {timestamp}")
    print(f"{'='*60}\n")

    # Fetch all market indexes
    indexes = get_index_data()
    _sp_entry = indexes.get("^GSPC") or {}
    _sp_ytd_raw = _sp_entry.get("ytd")
    sp500_ytd = _sp_ytd_raw if _sp_ytd_raw is not None else 0.0
    # COVERAGE, NOT OUTCOME (D-019): this is a SEPARATE fetch from the
    # macro proxies, and its failure used to vanish into a 0.0 default —
    # every "beating the S&P" comparison then measured against a
    # fabricated baseline while breadth_collapse still reported a result.
    _sp_baseline_ok = _sp_ytd_raw is not None
    degraded_inputs = ({} if _sp_baseline_ok else
                       {"breadth_collapse":
                        "S&P 500 YTD baseline unavailable — the "
                        "beating-S&P comparison defaulted to 0%"})
    print(f"S&P 500 YTD: {sp500_ytd}%"
          + ("" if _sp_baseline_ok else "  !! BASELINE UNAVAILABLE") + "\n")

    # Fetch macro proxy tickers for thesis-breaker checks
    print("Fetching macro proxy tickers...")
    macro_data = {}
    # COVERAGE, NOT OUTCOME (D-019): every fetch outcome is RECORDED. A
    # silent SKIP here used to make the affected breaker checks vanish and
    # the group read "clear" — a group that could not be checked was
    # byte-identical to one checked and found healthy. The reason travels
    # with the run and lands in the artifact.
    macro_status = {}
    for ticker in MACRO_TICKERS:
        df = fetch_data(ticker, period="6mo")
        if df is None:
            macro_status[ticker] = {"ok": False,
                                    "reason": "fetch returned no data"}
            print(f"  {ticker}: SKIP (no data)")
        elif len(df) <= 5:
            macro_status[ticker] = {"ok": False,
                                    "reason": f"only {len(df)} bars returned"}
            print(f"  {ticker}: SKIP ({len(df)} bars)")
        else:
            macro_data[ticker] = df
            macro_status[ticker] = {"ok": True, "reason": None, "bars": len(df)}
            print(f"  {ticker}: OK")
    _degraded_macro = [t for t, v in macro_status.items() if not v["ok"]]
    if _degraded_macro:
        print(f"  !! macro coverage INCOMPLETE: {', '.join(_degraded_macro)}"
              f" — dependent group breakers will report degraded, not clear")

    # Resolve this week's active universe (dynamic top-N; falls back to the
    # hardcoded groups only if no viable dynamic universe exists)
    industry_groups = get_industry_groups()
    print(f"Active universe: {len(industry_groups)} groups")

    # Collect all unique tickers
    all_tickers = set()
    for group_name, group_info in industry_groups.items():
        all_tickers.update(group_info["tickers"])

    print(f"\nFetching data for {len(all_tickers)} unique tickers...")
    ticker_data = {}
    ticker_signals = {}

    ytd_frames = {}
    for ticker in sorted(all_tickers):
        print(f"  Fetching {ticker}...", end=" ")
        df = fetch_data(ticker, period="6mo")
        if df is not None and len(df) > 20:
            ticker_data[ticker] = df
            # D-020a: a SEPARATE 1y frame feeds ONLY the YTD anchor —
            # the 6mo indicator frame above is untouched. A failed 1y
            # fetch degrades visibly (ytd_basis in the artifact), never
            # silently.
            ydf = fetch_data(ticker, period="1y")
            if ydf is not None and len(ydf):
                ytd_frames[ticker] = ydf
                print(f"OK ({len(df)} days, ytd frame {len(ydf)})")
            else:
                print(f"OK ({len(df)} days, YTD FRAME MISSING — "
                      "falling back to the 6mo anchor, recorded)")
        else:
            print("SKIP (insufficient data)")

    # Fetch fundamentals
    print(f"\nFetching fundamentals for {len(ticker_data)} tickers...")
    ticker_fundamentals = {}
    for ticker in sorted(ticker_data.keys()):
        print(f"  {ticker}...", end=" ")
        fund = fetch_fundamentals_yfinance(ticker)
        ticker_fundamentals[ticker] = fund
        mcap = fund.get("market_cap")
        fpe = fund.get("forward_pe")
        print(f"MCap={'${:,.0f}'.format(mcap) if mcap else 'N/A'}, FwdPE={fpe or 'N/A'}")

    print(f"\nComputing signals for {len(ticker_data)} tickers...")
    for ticker, df in ticker_data.items():
        groups_for_ticker = [g for g, info in industry_groups.items() if ticker in info["tickers"]]
        group_info = industry_groups.get(groups_for_ticker[0], {}) if groups_for_ticker else {}

        _ydf = ytd_frames.get(ticker)
        _ytd_v = _ytd_b = None
        if _ydf is not None:
            _ytd_v, _ytd_b = compute_ytd_return_v2(_ydf, with_basis=True)
        score, signal, details = score_stock_v2(df, group_info,
                                                ytd_return=_ytd_v,
                                                ytd_basis=_ytd_b)
        details["beating_sp500"] = bool(details.get("ytd_return", 0) > sp500_ytd)

        # Merge fundamentals
        fund = ticker_fundamentals.get(ticker, {})
        details["fundamentals"] = {
            "market_cap": fund.get("market_cap"),
            "forward_pe": fund.get("forward_pe"),
            "trailing_pe": fund.get("trailing_pe"),
            "revenue_growth": fund.get("revenue_growth_yoy"),
            "gross_margin": fund.get("gross_margin"),
            "operating_margin": fund.get("operating_margin"),
            "profit_margin": fund.get("profit_margin"),
            "eps_trailing": fund.get("eps_trailing"),
            "eps_forward": fund.get("eps_forward"),
            "dividend_yield": fund.get("dividend_yield"),
            "beta": fund.get("beta"),
            "short_pct_float": fund.get("short_pct_float"),
            "target_price": fund.get("target_mean_price"),
            "recommendation": fund.get("recommendation"),
            "industry": fund.get("industry")
        }

        # Stage analysis
        stage = compute_stage_analysis(details, df)
        details["stage_analysis"] = stage

        # D-017: emit the D-011 grade's per-row inputs (stripped-df
        # scalars). Failure only costs this row its candidate grade
        # (unavailable-data rule) — it can never block the signal run.
        try:
            details["grade_inputs"] = compute_grade_inputs(
                df, ytd_df=ytd_frames.get(ticker))
        except Exception as e:
            details["grade_inputs"] = None
            print(f"  {ticker}: grade-input emission failed ({e})")

        ticker_signals[ticker] = {
            "score": score,
            "signal": signal,
            "details": details,
            "groups": groups_for_ticker
        }
        stage_lbl = f"S{stage['stage']} {stage['stage_name']}" if stage.get('stage') else "?"
        print(f"  {ticker}: Score={score}, Signal={signal}, Stage={stage_lbl}, YTD={details.get('ytd_return', 'N/A')}%")

    # Earnings-calendar layer (PER-510, display-only v1): daily-cached
    # next-earnings map for the active universe. Missing date = no chip;
    # a flaky calendar endpoint can never block the run.
    from earnings_calendar import get_earnings_map, days_to_earnings
    earnings_map = get_earnings_map(
        [t for gi in industry_groups.values() for t in gi["tickers"]])

    # Build group-level data
    groups_output = []
    for group_name, group_info in industry_groups.items():
        stocks_in_group = []
        ytd_returns = []

        for ticker in group_info["tickers"]:
            if ticker in ticker_signals:
                sig = ticker_signals[ticker]
                d = sig["details"]
                stocks_in_group.append({
                    "ticker": ticker,
                    "next_earnings_date": earnings_map.get(ticker),
                    "days_to_earnings": days_to_earnings(earnings_map.get(ticker)),
                    "score": sig["score"],
                    "score_components": d.get("score_components"),
                    "score_inputs": d.get("score_inputs"),
                    "grade_inputs": d.get("grade_inputs"),
                    "signal": sig["signal"],
                    "ytd_return": d.get("ytd_return", 0),
                    # D-020a: the anchor that produced ytd_return — a
                    # failed 1y fetch degrades VISIBLY in the row itself
                    "ytd_basis": d.get("ytd_basis"),
                    "price": d.get("price", 0),
                    "rsi": d.get("rsi", 50),
                    "macd": d.get("macd", 0),
                    "macd_signal": d.get("macd_signal", 0),
                    "macd_histogram": d.get("macd_histogram", 0),
                    "ma20": d.get("ma20", 0),
                    "ma50": d.get("ma50", 0),
                    "ma200": d.get("ma200"),
                    "volume_ratio": d.get("volume_ratio", 1),
                    "beating_sp500": d.get("beating_sp500", False),
                    # Momentum
                    "high_52w": d.get("high_52w", 0),
                    "low_52w": d.get("low_52w", 0),
                    "pct_from_52w_high": d.get("pct_from_52w_high", 0),
                    "return_1m": d.get("return_1m", 0),
                    "return_3m": d.get("return_3m", 0),
                    "rs_vs_ma50": d.get("rs_vs_ma50", 0),
                    "trend_strength": d.get("trend_strength", 10),
                    # Fundamentals
                    "fundamentals": d.get("fundamentals", {}),
                    # Stage Analysis
                    "stage_analysis": d.get("stage_analysis")
                })
                ytd_returns.append(d.get("ytd_return", 0))

        if not stocks_in_group:
            continue

        stocks_in_group.sort(key=lambda x: x["ytd_return"], reverse=True)
        # nan-safe mean: ignore missing/NaN member returns; 0 if none valid
        ytd_valid = [v for v in ytd_returns if v is not None and np.isfinite(v)]
        avg_ytd = round(float(np.mean(ytd_valid)), 2) if ytd_valid else 0
        avg_score = round(np.mean([s["score"] for s in stocks_in_group]), 1)
        beating_count = sum(1 for s in stocks_in_group if s["beating_sp500"])

        if avg_score >= 70:
            group_signal = "strong-buy"
        elif avg_score >= 58:
            group_signal = "buy"
        elif avg_score >= 45:
            group_signal = "hold"
        else:
            group_signal = "sell"

        # Generate dynamic thesis and thesis breaker from live data
        dynamic_thesis = generate_dynamic_thesis(
            group_name, group_info, stocks_in_group, macro_data
        )
        dynamic_thesis_breaker = generate_dynamic_thesis_breaker(
            group_name, group_info, stocks_in_group, macro_data, sp500_ytd
        )

        # Check thesis breakers (uses dynamic checks internally)
        breaker_alerts, breaker_cov = check_thesis_breakers(
            group_name, group_info, stocks_in_group, macro_data, sp500_ytd,
            macro_status=macro_status, degraded_inputs=degraded_inputs
        )
        breaker_status = resolve_breaker_status(breaker_alerts, breaker_cov)

        # Compute trade signal for each stock in the group
        for stock in stocks_in_group:
            trade_sig, trade_reason = compute_trade_signal(
                {
                    "rsi": stock.get("rsi", 50),
                    "macd_histogram": stock.get("macd_histogram", 0),
                    "macd": stock.get("macd", 0),
                    "macd_signal": stock.get("macd_signal", 0),
                    "price": stock.get("price", 0),
                    "ma20": stock.get("ma20", 0),
                    "ma50": stock.get("ma50", 0),
                    "ma200": stock.get("ma200"),
                    "composite_score": stock.get("score", 50),
                    "signal": stock.get("signal", "hold"),
                    "ytd_return": stock.get("ytd_return", 0),
                    "volume_ratio": stock.get("volume_ratio", 1.0),
                    "pct_from_52w_high": stock.get("pct_from_52w_high", 0),
                    "trend_strength": stock.get("trend_strength", 10),
                    "rs_vs_ma50": stock.get("rs_vs_ma50", 0),
                    "return_1m": stock.get("return_1m", 0),
                },
                breaker_status=breaker_status
            )
            stock["trade_signal"] = trade_sig
            stock["trade_reasoning"] = trade_reason
            # Earnings proximity note (PER-510): text only — the tally and
            # decision ladder are untouched in v1
            dte = stock.get("days_to_earnings")
            if dte is not None and dte <= 7:
                stock["trade_reasoning"] += (
                    f"; earnings in {dte}d — R8: binary catalyst window")

        groups_output.append({
            "name": group_name,
            "gics_code": group_info.get("gics_code", ""),
            "gics_level": group_info.get("gics_level", ""),
            "sector": group_info["sector"],
            "industry_group": group_info.get("industry_group", ""),
            "thesis": dynamic_thesis,
            "thesis_breaker": dynamic_thesis_breaker,
            "cycle_stage": group_info["cycle_stage"],
            "avg_ytd": avg_ytd,
            "avg_score": avg_score,
            "group_signal": group_signal,
            "stock_count": len(stocks_in_group),
            "beating_sp500_count": beating_count,
            "breaker_status": breaker_status,
            "breaker_alerts": breaker_alerts,
            # D-019 coverage record: what the sensitivities CALLED FOR vs
            # what actually RAN. A degraded group can never again be
            # byte-identical to a clear one — that identity WAS the bug.
            "breaker_checks_expected": breaker_cov["expected"],
            "breaker_checks_run": breaker_cov["run"],
            "breaker_degraded_reasons": breaker_cov["reasons"],
            # rotation-week audit: candidates that did NOT make the cut
            # (empty for fallback/legacy groups — UI omits the strip)
            "near_misses": group_info.get("near_misses", []),
            "stocks": stocks_in_group
        })

    groups_output.sort(key=lambda x: x["avg_ytd"], reverse=True)
    for i, group in enumerate(groups_output):
        group["rank"] = i + 1

    output = {
        "timestamp": timestamp,
        # D-020a: scorer era stamp — artifacts without this key were
        # baked by the v1 scorer and are never retro-relabelled
        "scorer_version": "score_stock_v2",
        "sp500_ytd": sp500_ytd,
        # per-ticker macro fetch outcomes for this run (D-019) — the
        # run-level companion to each group's coverage record
        "macro_status": macro_status,
        "sp500_baseline_ok": _sp_baseline_ok,
        "indexes": indexes,
        "total_tickers": len(ticker_signals),
        "total_groups": len(groups_output),
        "groups": groups_output
    }
    output = sanitize_for_json(output)

    signals_path = os.path.join(DATA_DIR, "signals.json")
    os.makedirs(DATA_DIR, exist_ok=True)
    with open(signals_path, "w") as f:
        json.dump(output, f, indent=2, cls=NumpyEncoder)
    print(f"\nSignals written to {signals_path}")

    public_signals_path = os.path.join(PUBLIC_DIR, "signals.json")
    os.makedirs(PUBLIC_DIR, exist_ok=True)
    with open(public_signals_path, "w") as f:
        json.dump(output, f, indent=2, cls=NumpyEncoder)

    # Print thesis-breaker summary
    print(f"\n{'='*60}")
    print("THESIS-BREAKER STATUS")
    print(f"{'='*60}")
    for g in groups_output:
        triggered = [a for a in g["breaker_alerts"] if a["triggered"]]
        icon = {"critical": "🔴", "warning": "🟠", "watch": "🟡",
                "clear": "🟢", "degraded": "⚠️"}.get(g["breaker_status"], "⚪")
        print(f"  {icon} {g['name']}: {g['breaker_status'].upper()}")
        for a in triggered:
            print(f"      ⚠ {a['message']}")

    return output


if __name__ == "__main__":
    run_engine()
