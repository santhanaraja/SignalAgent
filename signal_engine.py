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
# Names follow GICS Level 3/4 classification standard
# Each group includes thesis-breaker conditions that can be
# programmatically checked against market data
# ============================================================
INDUSTRY_GROUPS = {
    "Technology Hardware, Storage & Peripherals": {
        "gics_code": "45202030",
        "gics_level": "Sub-Industry",
        "sector": "Information Technology",
        "industry_group": "Technology Hardware & Equipment",
        "tickers": ["SNDK", "STX", "WDC", "PSTG", "NTAP"],
        "thesis": "AI infrastructure buildout driving record demand for high-capacity storage. NAND pricing stabilizing; HDD nearline demand surging from hyperscaler data centers.",
        "thesis_breaker": "AI capex slowdown, hyperscaler order deferrals, or NAND oversupply from Chinese fabs (YMTC).",
        "cycle_stage": "mid",
        "breaker_checks": {
            "sp500_drawdown_10pct": "S&P 500 drops >10% from YTD high (proxy for macro risk-off / capex freeze)",
            "group_avg_rsi_below_40": "Group avg RSI falls below 40 — momentum breakdown",
            "majority_below_ma50": ">50% of stocks fall below 50-day MA — trend reversal",
            "avg_ytd_negative": "Group avg YTD return turns negative — leadership lost"
        }
    },
    "Semiconductors — Memory & HBM": {
        "gics_code": "45301020",
        "gics_level": "Sub-Industry",
        "sector": "Information Technology",
        "industry_group": "Semiconductors & Semiconductor Equipment",
        "tickers": ["MU"],
        "thesis": "HBM is the critical bottleneck for AI training/inference. Supply fully booked through 2026; prices rising ~20%. TAM from $35B to $100B by 2028.",
        "thesis_breaker": "AI training efficiency gains reducing memory per GPU, Chinese HBM alternatives, or customer inventory digestion.",
        "cycle_stage": "early-mid",
        "breaker_checks": {
            "sp500_drawdown_10pct": "S&P 500 drops >10% from YTD high",
            "group_avg_rsi_below_40": "Group avg RSI falls below 40",
            "majority_below_ma50": ">50% of stocks fall below 50-day MA",
            "avg_ytd_negative": "Group avg YTD return turns negative"
        }
    },
    "Semiconductor Materials & Equipment": {
        "gics_code": "45301010",
        "gics_level": "Sub-Industry",
        "sector": "Information Technology",
        "industry_group": "Semiconductors & Semiconductor Equipment",
        "tickers": ["ASML", "AMAT", "LRCX", "KLAC", "ENTG", "MKSI", "TER", "ALAB"],
        "thesis": "AI capex supercycle driving record WFE spend — advanced packaging (CoWoS/HBM), EUV lithography, high-NA EUV, and AI connectivity (PCIe retimers, CXL). TSMC/Samsung/Intel all expanding capacity.",
        "thesis_breaker": "WFE spending downturn, fab project delays/cancellations, China export restrictions tightening, or customer inventory correction.",
        "cycle_stage": "mid",
        "breaker_checks": {
            "sp500_drawdown_10pct": "S&P 500 drops >10% from YTD high — macro risk-off / capex freeze",
            "group_avg_rsi_below_40": "Group avg RSI falls below 40 — momentum breakdown",
            "majority_below_ma50": ">50% of stocks fall below 50-day MA — trend reversal",
            "avg_ytd_negative": "Group avg YTD return turns negative — leadership lost"
        }
    },
    "Gold": {
        "gics_code": "15104030",
        "gics_level": "Sub-Industry",
        "sector": "Materials",
        "industry_group": "Metals & Mining",
        "tickers": ["NEM", "GOLD", "AEM", "KGC", "AU", "HL", "CDE", "PAAS", "FNV", "RGLD", "EGO", "OR", "SSRM", "WPM"],
        "thesis": "Record gold prices with massive operating leverage. Central bank buying, geopolitical risk, de-dollarization. Broadest group — 13+ stocks beating S&P.",
        "thesis_breaker": "Sharp rise in real interest rates, USD strength, gold reversal below $2,400, or mining cost inflation.",
        "cycle_stage": "mid",
        "breaker_checks": {
            "gold_below_threshold": "Gold (GLD) drops >8% from recent high — commodity thesis weakening",
            "usd_strength": "DXY (UUP) rises >5% YTD — USD strength crushing gold",
            "group_avg_rsi_below_40": "Group avg RSI falls below 40",
            "majority_below_ma50": ">50% of stocks fall below 50-day MA",
            "breadth_collapse": "<50% of stocks beating S&P 500 (was 90%+) — breadth deteriorating"
        }
    },
    "Copper": {
        "gics_code": "15104025",
        "gics_level": "Sub-Industry",
        "sector": "Materials",
        "industry_group": "Metals & Mining",
        "tickers": ["FCX", "SCCO", "TECK", "HBM"],
        "thesis": "Structural supply deficit from electrification, renewable energy buildout, and AI data center infrastructure. ICSG forecasts deficit in 2026. Mine supply growth lagging demand.",
        "thesis_breaker": "Global recession crushing industrial demand, unexpected mine supply surge, or copper substitution in key applications.",
        "cycle_stage": "mid",
        "breaker_checks": {
            "group_avg_rsi_below_40": "Group avg RSI falls below 40 — momentum breakdown",
            "majority_below_ma50": ">50% of stocks fall below 50-day MA — trend reversal",
            "avg_ytd_negative": "Group avg YTD return turns negative — leadership lost",
            "sp500_drawdown_10pct": "S&P 500 drops >10% from YTD high — macro risk-off"
        }
    },
    "Specialty Chemicals": {
        "gics_code": "15101050",
        "gics_level": "Sub-Industry",
        "sector": "Materials",
        "industry_group": "Chemicals",
        "tickers": ["TROX", "CC", "KRO", "DOW", "LYB", "PPG", "ECL"],
        "thesis": "Cyclical mean-reversion. TiO2 volumes recovering, supply being cut (plant closures). Destocking cycle ending across specialty chemicals.",
        "thesis_breaker": "China TiO2 dumping, construction/auto demand double-dip, or energy cost spikes.",
        "cycle_stage": "early",
        "breaker_checks": {
            "group_avg_rsi_below_40": "Group avg RSI falls below 40",
            "majority_below_ma50": ">50% of stocks fall below 50-day MA",
            "avg_ytd_negative": "Group avg YTD return turns negative",
            "energy_spike": "Energy (XLE) surges >15% YTD — energy cost pressure on margins"
        }
    },
    "Oil & Gas Equipment & Services": {
        "gics_code": "10101020",
        "gics_level": "Sub-Industry",
        "sector": "Energy",
        "industry_group": "Energy Equipment & Services",
        "tickers": ["VAL", "RIG", "HAL", "NOV", "FTI", "LBRT"],
        "thesis": "Structural upcycle: no new deepwater rigs since 2014, aging fleet, accelerating project sanctioning. Day rates $400K/day.",
        "thesis_breaker": "Oil below $60, OPEC+ supply surge, or rapid transition to onshore/unconventional.",
        "cycle_stage": "mid",
        "breaker_checks": {
            "oil_below_60": "Crude oil (USO) drops >25% from recent high — approaching $60 breakeven",
            "group_avg_rsi_below_40": "Group avg RSI falls below 40",
            "majority_below_ma50": ">50% of stocks fall below 50-day MA"
        }
    },
    "Oil & Gas Exploration & Production": {
        "gics_code": "10102020",
        "gics_level": "Sub-Industry",
        "sector": "Energy",
        "industry_group": "Oil, Gas & Consumable Fuels",
        "tickers": ["KOS", "EQT", "COP", "FANG", "APA", "SM"],
        "thesis": "Nat gas E&P benefiting from LNG export additions 2026-27. Oil E&P supported by OPEC+ discipline. Selective, not broad.",
        "thesis_breaker": "OPEC+ production increases, warm winters killing nat gas, or global recession.",
        "cycle_stage": "mid",
        "breaker_checks": {
            "oil_below_60": "Crude oil drops >25% from recent high",
            "natgas_collapse": "Natural gas (UNG) drops >30% from recent high",
            "group_avg_rsi_below_40": "Group avg RSI falls below 40",
            "majority_below_ma50": ">50% of stocks fall below 50-day MA"
        }
    },
    "Aerospace & Defense": {
        "gics_code": "20101010",
        "gics_level": "Sub-Industry",
        "sector": "Industrials",
        "industry_group": "Capital Goods",
        "tickers": ["RKLB", "LHX", "LMT", "KTOS", "PLTR", "RTX", "NOC", "GD", "HII"],
        "thesis": "European rearmament (NATO 3%+ GDP), Ukraine, Pacific tensions creating multi-year defense supercycle. Navy shipbuilding at $26-27B authorized for 2026.",
        "thesis_breaker": "Peace deals, budget sequestration, CR-driven spending delays, or program cancellations.",
        "cycle_stage": "early-mid",
        "breaker_checks": {
            "group_avg_rsi_below_40": "Group avg RSI falls below 40",
            "majority_below_ma50": ">50% of stocks fall below 50-day MA",
            "avg_ytd_negative": "Group avg YTD return turns negative"
        }
    },
    "Independent Power Producers & Energy Traders": {
        "gics_code": "55105020",
        "gics_level": "Sub-Industry",
        "sector": "Utilities",
        "industry_group": "Independent Power and Renewable Electricity Producers",
        "tickers": ["CEG", "VST", "NRG", "OKLO", "SMR"],
        "thesis": "AI data center power demand creating structural electricity shortage. Nuclear restarts and new builds accelerating. Capacity contracts at premium pricing. Electrification and reshoring compounding demand growth.",
        "thesis_breaker": "Data center power demand forecasts miss, natural gas price spikes crushing margins, regulatory blocks on nuclear restarts, or grid interconnection delays.",
        "cycle_stage": "early-mid",
        "breaker_checks": {
            "group_avg_rsi_below_40": "Group avg RSI falls below 40 — momentum breakdown",
            "majority_below_ma50": ">50% of stocks fall below 50-day MA — trend reversal",
            "avg_ytd_negative": "Group avg YTD return turns negative — leadership lost",
            "sp500_drawdown_10pct": "S&P 500 drops >10% from YTD high — macro risk-off"
        }
    },
    "Coal & Consumable Fuels (Uranium)": {
        "gics_code": "10102050",
        "gics_level": "Sub-Industry",
        "sector": "Energy",
        "industry_group": "Oil, Gas & Consumable Fuels",
        "tickers": ["UEC", "LEU", "CCJ"],
        "thesis": "Nuclear renaissance driven by AI data center power demand. Uranium supply structurally deficient. BofA targets $135/lb vs $81-88 current.",
        "thesis_breaker": "Uranium price pullback, SMR delays, or return of Russian/Kazakh supply.",
        "cycle_stage": "early",
        "breaker_checks": {
            "group_avg_rsi_below_40": "Group avg RSI falls below 40",
            "majority_below_ma50": ">50% of stocks fall below 50-day MA",
            "avg_ytd_negative": "Group avg YTD return turns negative"
        }
    },
    "Oil & Gas Refining & Marketing": {
        "gics_code": "10102030",
        "gics_level": "Sub-Industry",
        "sector": "Energy",
        "industry_group": "Oil, Gas & Consumable Fuels",
        "tickers": ["MPC", "VLO", "PBF", "DK", "PSX"],
        "thesis": "Crack spreads elevated as refining capacity additions lag closures. Export demand from Latin America supporting Gulf Coast economics.",
        "thesis_breaker": "Crack spread compression from new ME/Asia capacity, demand destruction.",
        "cycle_stage": "mid-late",
        "breaker_checks": {
            "group_avg_rsi_below_40": "Group avg RSI falls below 40",
            "majority_below_ma50": ">50% of stocks fall below 50-day MA",
            "avg_ytd_negative": "Group avg YTD return turns negative"
        }
    }
}

# Macro proxy tickers for thesis-breaker checks
MACRO_TICKERS = ["^GSPC", "GLD", "UUP", "USO", "UNG", "XLE"]


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


def compute_ytd_return(df):
    """Compute YTD return from first trading day of current year."""
    current_year = datetime.datetime.now().year
    year_data = df[df.index.year == current_year]
    if len(year_data) < 2:
        return 0.0
    first_close = year_data["Close"].iloc[0]
    last_close = year_data["Close"].iloc[-1]
    return round(((last_close - first_close) / first_close) * 100, 2)


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
def check_thesis_breakers(group_name, group_info, group_stocks, macro_data, sp500_ytd):
    """
    Check thesis-breaker conditions for a group.
    Returns list of triggered alerts with severity.
    """
    alerts = []
    checks = group_info.get("breaker_checks", {})

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
            uup_ytd = compute_ytd_return(uup_data)
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
            xle_ytd = compute_ytd_return(xle_data)
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

    return alerts


# ============================================================
# SIGNAL SCORING
# ============================================================
def score_stock(df, group_info=None):
    """
    Compute a composite score (0-100) and signal for a stock.
    """
    close = df["Close"]
    score = 50
    details = {}

    # --- RSI ---
    rsi = compute_rsi(close)
    current_rsi = rsi.iloc[-1] if not pd.isna(rsi.iloc[-1]) else 50
    details["rsi"] = round(current_rsi, 1)

    if current_rsi < 30:
        score += 15
    elif current_rsi < 40:
        score += 8
    elif current_rsi < 60:
        score += 3
    elif current_rsi < 70:
        score -= 3
    elif current_rsi < 80:
        score -= 8
    else:
        score -= 15

    # --- MACD ---
    macd_line, signal_line, histogram = compute_macd(close)
    current_macd = macd_line.iloc[-1] if not pd.isna(macd_line.iloc[-1]) else 0
    current_signal = signal_line.iloc[-1] if not pd.isna(signal_line.iloc[-1]) else 0
    current_hist = histogram.iloc[-1] if not pd.isna(histogram.iloc[-1]) else 0
    prev_hist = histogram.iloc[-2] if len(histogram) > 1 and not pd.isna(histogram.iloc[-2]) else 0

    details["macd"] = round(current_macd, 3)
    details["macd_signal"] = round(current_signal, 3)
    details["macd_histogram"] = round(current_hist, 3)

    if current_macd > current_signal:
        score += 8
        if current_hist > prev_hist:
            score += 5
    else:
        score -= 8
        if current_hist < prev_hist:
            score -= 5

    # --- Moving Averages ---
    ma20, ma50, ma200 = compute_moving_averages(close)
    current_price = close.iloc[-1]
    ma20_val = ma20.iloc[-1] if not pd.isna(ma20.iloc[-1]) else current_price
    ma50_val = ma50.iloc[-1] if not pd.isna(ma50.iloc[-1]) else current_price
    ma200_val = ma200.iloc[-1] if not pd.isna(ma200.iloc[-1]) else None

    details["price"] = round(current_price, 2)
    details["ma20"] = round(ma20_val, 2)
    details["ma50"] = round(ma50_val, 2)
    details["ma200"] = round(ma200_val, 2) if ma200_val else None

    if current_price > ma20_val:
        score += 4
    else:
        score -= 4
    if current_price > ma50_val:
        score += 6
    else:
        score -= 6
    if ma20_val > ma50_val:
        score += 4
    else:
        score -= 4

    # --- YTD Momentum ---
    ytd_return = compute_ytd_return(df)
    details["ytd_return"] = ytd_return

    if ytd_return > 50:
        score += 8
    elif ytd_return > 20:
        score += 12
    elif ytd_return > 5:
        score += 6
    elif ytd_return > 0:
        score += 2
    elif ytd_return > -10:
        score -= 4
    else:
        score -= 10

    if ytd_return > 100:
        score -= 10
    elif ytd_return > 150:
        score -= 15

    # --- Volume ---
    vol_ratio = compute_volume_trend(df)
    details["volume_ratio"] = vol_ratio
    if vol_ratio > 1.5:
        score += 3
    elif vol_ratio < 0.7:
        score -= 3

    # --- Momentum Metrics ---
    momentum = compute_momentum_metrics(df)
    details.update(momentum)

    score = max(0, min(100, score))
    details["composite_score"] = score

    if score >= 75:
        signal = "strong-buy"
    elif score >= 60:
        signal = "buy"
    elif score >= 45:
        signal = "hold"
    elif score >= 30:
        signal = "sell"
    else:
        signal = "strong-sell"

    details["signal"] = signal
    return score, signal, details


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
    if breaker_status in ("critical", "warning"):
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
        signal = "WAIT FOR PULLBACK"
    elif abs(net) <= 1 and range_ratio < 0.8:
        signal = "RANGE BOUND"
    elif net >= 1:
        signal = "WAIT FOR BREAKOUT"
    elif net <= -1:
        signal = "WAIT FOR PULLBACK"
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
    elif signal == "WAIT FOR PULLBACK":
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
    Uses 150-day MA (≈30-week MA) as primary, with slope analysis,
    price position, and volume confirmation.

    Stages:
        Stage 1 — Basing/Accumulation: Price consolidating near flat MA150
        Stage 2 — Advancing/Markup:    Price above rising MA150 (ideal buy zone)
        Stage 3 — Topping/Distribution: Price struggling, MA150 flattening after rise
        Stage 4 — Declining/Markdown:   Price below falling MA150 (avoid/short)

    Returns dict: {stage, stage_name, description, confidence, factors}
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

    # --- Compute 150-day MA (≈30-week MA) ---
    if len(close) >= 150:
        ma150 = close.rolling(150).mean()
        ma150_current = float(ma150.iloc[-1])
        # Slope: compare current MA150 to 20 days ago
        ma150_20ago = float(ma150.iloc[-21]) if len(ma150) > 20 and not pd.isna(ma150.iloc[-21]) else ma150_current
        ma150_slope = (ma150_current - ma150_20ago) / ma150_20ago * 100 if ma150_20ago != 0 else 0
    else:
        # Fallback to MA50 if not enough data for MA150
        ma150_series = close.rolling(min(len(close), 100)).mean()
        ma150_current = float(ma150_series.iloc[-1]) if not pd.isna(ma150_series.iloc[-1]) else price
        ma150_20ago = float(ma150_series.iloc[-21]) if len(ma150_series) > 20 and not pd.isna(ma150_series.iloc[-21]) else ma150_current
        ma150_slope = (ma150_current - ma150_20ago) / ma150_20ago * 100 if ma150_20ago != 0 else 0

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
        factors.append(f"Price {price_vs_ma150_pct:+.1f}% above MA150 — bullish positioning")
    elif price_vs_ma150_pct > 0:
        stage_scores[2] += 1
        stage_scores[1] += 1
        factors.append(f"Price {price_vs_ma150_pct:+.1f}% above MA150 — near support")
    elif price_vs_ma150_pct > -3:
        stage_scores[3] += 2
        stage_scores[1] += 1
        factors.append(f"Price {price_vs_ma150_pct:+.1f}% near MA150 — testing support")
    else:
        stage_scores[4] += 3
        factors.append(f"Price {price_vs_ma150_pct:+.1f}% below MA150 — bearish positioning")

    # MA150 slope (primary trend)
    if ma150_slope > 0.5:
        stage_scores[2] += 3
        factors.append(f"MA150 rising ({ma150_slope:+.2f}%) — uptrend confirmed")
    elif ma150_slope > -0.2:
        stage_scores[1] += 2
        stage_scores[3] += 2
        factors.append(f"MA150 flat ({ma150_slope:+.2f}%) — consolidation/transition")
    else:
        stage_scores[4] += 3
        factors.append(f"MA150 falling ({ma150_slope:+.2f}%) — downtrend confirmed")

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
        factors.append(f"High volume ({vol_ratio:.1f}x) above MA150 — accumulation")
    elif vol_ratio >= 1.5 and price < ma150_current:
        stage_scores[4] += 1
        factors.append(f"High volume ({vol_ratio:.1f}x) below MA150 — distribution")

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
        1: ("Basing", "Accumulation phase — price consolidating near flat MA150. Watch for breakout above MA150 with volume."),
        2: ("Advancing", "Markup phase — price above rising MA150. Ideal zone for long positions. Ride the trend."),
        3: ("Topping", "Distribution phase — MA150 flattening after advance. Consider tightening stops or taking profits."),
        4: ("Declining", "Markdown phase — price below falling MA150. Avoid new longs. Consider short positions."),
    }

    name, desc = stage_info[stage]

    return {
        "stage": stage,
        "stage_name": name,
        "description": desc,
        "confidence": confidence,
        "ma150": round(ma150_current, 2),
        "ma150_slope": round(ma150_slope, 3),
        "price_vs_ma150_pct": round(price_vs_ma150_pct, 2),
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
}


def get_index_data():
    """Fetch YTD return and current level for major indexes."""
    print("Fetching market indexes...")
    indexes = {}
    for ticker, name in INDEX_TICKERS.items():
        print(f"  {name} ({ticker})...", end=" ")
        df = fetch_data(ticker, period="6mo")
        if df is not None and len(df) > 5:
            ytd = compute_ytd_return(df)
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
            }
            print(f"OK — {level:,.2f} (YTD {ytd}%)")
        else:
            print("SKIP")
    return indexes


def get_sp500_ytd():
    print("Fetching S&P 500 benchmark...")
    df = fetch_data("^GSPC", period="6mo")
    if df is not None:
        return compute_ytd_return(df)
    return 0.0


def run_engine():
    """Main entry point — runs the full signal pipeline."""
    timestamp = datetime.datetime.now().isoformat()
    print(f"\n{'='*60}")
    print(f"Signal Engine Run — {timestamp}")
    print(f"{'='*60}\n")

    # Fetch all market indexes
    indexes = get_index_data()
    sp500_ytd = indexes.get("^GSPC", {}).get("ytd", 0.0)
    print(f"S&P 500 YTD: {sp500_ytd}%\n")

    # Fetch macro proxy tickers for thesis-breaker checks
    print("Fetching macro proxy tickers...")
    macro_data = {}
    for ticker in MACRO_TICKERS:
        df = fetch_data(ticker, period="6mo")
        if df is not None and len(df) > 5:
            macro_data[ticker] = df
            print(f"  {ticker}: OK")
        else:
            print(f"  {ticker}: SKIP")

    # Collect all unique tickers
    all_tickers = set()
    for group_name, group_info in INDUSTRY_GROUPS.items():
        all_tickers.update(group_info["tickers"])

    print(f"\nFetching data for {len(all_tickers)} unique tickers...")
    ticker_data = {}
    ticker_signals = {}

    for ticker in sorted(all_tickers):
        print(f"  Fetching {ticker}...", end=" ")
        df = fetch_data(ticker, period="6mo")
        if df is not None and len(df) > 20:
            ticker_data[ticker] = df
            print(f"OK ({len(df)} days)")
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
        groups_for_ticker = [g for g, info in INDUSTRY_GROUPS.items() if ticker in info["tickers"]]
        group_info = INDUSTRY_GROUPS.get(groups_for_ticker[0], {}) if groups_for_ticker else {}

        score, signal, details = score_stock(df, group_info)
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

        ticker_signals[ticker] = {
            "score": score,
            "signal": signal,
            "details": details,
            "groups": groups_for_ticker
        }
        stage_lbl = f"S{stage['stage']} {stage['stage_name']}" if stage.get('stage') else "?"
        print(f"  {ticker}: Score={score}, Signal={signal}, Stage={stage_lbl}, YTD={details.get('ytd_return', 'N/A')}%")

    # Build group-level data
    groups_output = []
    for group_name, group_info in INDUSTRY_GROUPS.items():
        stocks_in_group = []
        ytd_returns = []

        for ticker in group_info["tickers"]:
            if ticker in ticker_signals:
                sig = ticker_signals[ticker]
                d = sig["details"]
                stocks_in_group.append({
                    "ticker": ticker,
                    "score": sig["score"],
                    "signal": sig["signal"],
                    "ytd_return": d.get("ytd_return", 0),
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
        avg_ytd = round(np.mean(ytd_returns), 2) if ytd_returns else 0
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

        # Check thesis breakers
        breaker_alerts = check_thesis_breakers(
            group_name, group_info, stocks_in_group, macro_data, sp500_ytd
        )
        triggered_alerts = [a for a in breaker_alerts if a["triggered"]]
        breaker_status = "clear"
        if any(a["severity"] == "critical" for a in triggered_alerts):
            breaker_status = "critical"
        elif any(a["severity"] == "high" for a in triggered_alerts):
            breaker_status = "warning"
        elif any(a["severity"] == "medium" for a in triggered_alerts):
            breaker_status = "watch"

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

        groups_output.append({
            "name": group_name,
            "gics_code": group_info.get("gics_code", ""),
            "gics_level": group_info.get("gics_level", ""),
            "sector": group_info["sector"],
            "industry_group": group_info.get("industry_group", ""),
            "thesis": group_info["thesis"],
            "thesis_breaker": group_info["thesis_breaker"],
            "cycle_stage": group_info["cycle_stage"],
            "avg_ytd": avg_ytd,
            "avg_score": avg_score,
            "group_signal": group_signal,
            "stock_count": len(stocks_in_group),
            "beating_sp500_count": beating_count,
            "breaker_status": breaker_status,
            "breaker_alerts": breaker_alerts,
            "stocks": stocks_in_group
        })

    groups_output.sort(key=lambda x: x["avg_ytd"], reverse=True)
    for i, group in enumerate(groups_output):
        group["rank"] = i + 1

    output = {
        "timestamp": timestamp,
        "sp500_ytd": sp500_ytd,
        "indexes": indexes,
        "total_tickers": len(ticker_signals),
        "total_groups": len(groups_output),
        "groups": groups_output
    }

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
        icon = {"critical": "🔴", "warning": "🟠", "watch": "🟡", "clear": "🟢"}.get(g["breaker_status"], "⚪")
        print(f"  {icon} {g['name']}: {g['breaker_status'].upper()}")
        for a in triggered:
            print(f"      ⚠ {a['message']}")

    return output


if __name__ == "__main__":
    run_engine()
