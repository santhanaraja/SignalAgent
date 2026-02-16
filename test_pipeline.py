#!/usr/bin/env python3
"""
Test the pipeline with synthetic data to verify signal computation,
history tracking, and JSON output — no API calls needed.
"""

import json
import os
import sys
import datetime
import numpy as np
import pandas as pd


class NumpyEncoder(json.JSONEncoder):
    def default(self, obj):
        if isinstance(obj, (np.integer,)): return int(obj)
        if isinstance(obj, (np.floating,)): return float(obj)
        if isinstance(obj, (np.bool_,)): return bool(obj)
        if isinstance(obj, np.ndarray): return obj.tolist()
        return super().default(obj)

# Add parent dir to path
sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
from signal_engine import compute_rsi, compute_macd, compute_moving_averages, score_stock, compute_ytd_return, DATA_DIR, PUBLIC_DIR
from history_manager import detect_changes, save_snapshot, run_history_manager

def generate_synthetic_price_data(ticker, trend="up", volatility=0.02, days=130):
    """Generate realistic-looking price data for testing."""
    np.random.seed(hash(ticker) % 2**31)
    dates = pd.bdate_range(end=datetime.datetime.now(), periods=days)

    base_price = 50 + np.random.rand() * 150  # Random starting price $50-$200

    if trend == "up":
        drift = 0.002
    elif trend == "down":
        drift = -0.001
    else:
        drift = 0.0005

    returns = np.random.normal(drift, volatility, days)
    prices = base_price * np.cumprod(1 + returns)

    df = pd.DataFrame({
        "Open": prices * (1 + np.random.uniform(-0.005, 0.005, days)),
        "High": prices * (1 + np.abs(np.random.normal(0, 0.01, days))),
        "Low": prices * (1 - np.abs(np.random.normal(0, 0.01, days))),
        "Close": prices,
        "Volume": np.random.randint(500000, 50000000, days).astype(float)
    }, index=dates)

    return df


def test_indicators():
    """Test technical indicator calculations."""
    print("=" * 60)
    print("TEST 1: Technical Indicators")
    print("=" * 60)

    df = generate_synthetic_price_data("TEST_UP", trend="up", days=130)
    close = df["Close"]

    # RSI
    rsi = compute_rsi(close)
    current_rsi = rsi.iloc[-1]
    print(f"\n  RSI(14): {current_rsi:.1f}")
    assert 0 <= current_rsi <= 100, f"RSI out of range: {current_rsi}"
    print("  ✓ RSI in valid range [0, 100]")

    # MACD
    macd, signal, hist = compute_macd(close)
    print(f"  MACD: {macd.iloc[-1]:.3f}, Signal: {signal.iloc[-1]:.3f}, Hist: {hist.iloc[-1]:.3f}")
    assert not pd.isna(macd.iloc[-1]), "MACD is NaN"
    print("  ✓ MACD computed successfully")

    # Moving Averages
    ma20, ma50, ma200 = compute_moving_averages(close)
    print(f"  MA20: {ma20.iloc[-1]:.2f}, MA50: {ma50.iloc[-1]:.2f}")
    assert not pd.isna(ma50.iloc[-1]), "MA50 is NaN"
    print("  ✓ Moving averages computed")

    # YTD Return
    ytd = compute_ytd_return(df)
    print(f"  YTD Return: {ytd:.2f}%")
    print("  ✓ YTD return computed")

    print("\n  ✅ All indicator tests passed!\n")


def test_scoring():
    """Test scoring across different market conditions."""
    print("=" * 60)
    print("TEST 2: Signal Scoring")
    print("=" * 60)

    scenarios = [
        ("BULL_STOCK", "up", 0.015),
        ("BEAR_STOCK", "down", 0.02),
        ("FLAT_STOCK", "flat", 0.01),
        ("VOLATILE_BULL", "up", 0.04),
    ]

    for name, trend, vol in scenarios:
        df = generate_synthetic_price_data(name, trend=trend, volatility=vol, days=130)
        score, signal, details = score_stock(df, {})
        print(f"\n  {name}: Score={score}, Signal={signal.upper()}")
        print(f"    RSI={details['rsi']:.1f}, MACD_Hist={details['macd_histogram']:.3f}, YTD={details['ytd_return']:.1f}%")
        assert 0 <= score <= 100, f"Score out of range for {name}"
        assert signal in ["strong-buy", "buy", "hold", "sell", "strong-sell"], f"Invalid signal: {signal}"

    print("\n  ✅ All scoring tests passed!\n")


def test_full_pipeline():
    """Test the full pipeline with synthetic data and generate real output."""
    print("=" * 60)
    print("TEST 3: Full Pipeline (Synthetic Data)")
    print("=" * 60)

    # Simulated industry groups with synthetic data
    test_groups = {
        "Gold Mining / Precious Metals": {
            "tickers": {"NEM": "up", "GOLD": "up", "AEM": "up", "KGC": "up"},
            "thesis": "Record gold prices with massive operating leverage.",
            "thesis_breaker": "Sharp rise in real interest rates or USD strength.",
            "cycle_stage": "mid", "sector": "Materials"
        },
        "Data Storage — NAND / HDD": {
            "tickers": {"STX": "up", "WDC": "up", "MU": "up"},
            "thesis": "AI infrastructure buildout driving record storage demand.",
            "thesis_breaker": "AI capex slowdown or NAND oversupply.",
            "cycle_stage": "mid", "sector": "Technology"
        },
        "Aerospace & Defense": {
            "tickers": {"LMT": "up", "LHX": "up", "KTOS": "up", "RKLB": "up"},
            "thesis": "European rearmament and defense supercycle.",
            "thesis_breaker": "Peace deals or budget sequestration.",
            "cycle_stage": "early-mid", "sector": "Industrials"
        },
        "TiO2 / Specialty Chemicals": {
            "tickers": {"TROX": "up", "CC": "flat", "DOW": "flat"},
            "thesis": "Cyclical mean-reversion, supply cuts, destocking end.",
            "thesis_breaker": "China dumping or demand double-dip.",
            "cycle_stage": "early", "sector": "Materials"
        },
    }

    sp500_ytd = 2.5  # simulated
    groups_output = []

    for group_name, group_info in test_groups.items():
        stocks = []
        ytd_returns = []

        for ticker, trend in group_info["tickers"].items():
            df = generate_synthetic_price_data(ticker, trend=trend, days=130)
            score, signal, details = score_stock(df, group_info)
            stocks.append({
                "ticker": ticker,
                "score": score,
                "signal": signal,
                "ytd_return": details["ytd_return"],
                "price": details["price"],
                "rsi": details["rsi"],
                "macd": details["macd"],
                "macd_signal": details["macd_signal"],
                "macd_histogram": details["macd_histogram"],
                "ma20": details["ma20"],
                "ma50": details["ma50"],
                "ma200": details.get("ma200"),
                "volume_ratio": details["volume_ratio"],
                "beating_sp500": bool(details["ytd_return"] > sp500_ytd),
                # Momentum
                "high_52w": details.get("high_52w", 0),
                "low_52w": details.get("low_52w", 0),
                "pct_from_52w_high": details.get("pct_from_52w_high", 0),
                "return_1m": details.get("return_1m", 0),
                "return_3m": details.get("return_3m", 0),
                "rs_vs_ma50": details.get("rs_vs_ma50", 0),
                "trend_strength": details.get("trend_strength", 10),
                # Fundamentals (synthetic)
                "fundamentals": {
                    "market_cap": int(np.random.uniform(1e9, 100e9)),
                    "forward_pe": round(np.random.uniform(5, 40), 1),
                    "trailing_pe": round(np.random.uniform(5, 50), 1),
                    "revenue_growth": round(np.random.uniform(-0.1, 0.3), 3),
                    "operating_margin": round(np.random.uniform(-0.05, 0.35), 3),
                    "profit_margin": round(np.random.uniform(-0.1, 0.25), 3),
                    "eps_forward": round(np.random.uniform(0.5, 15), 2),
                    "eps_trailing": round(np.random.uniform(0.3, 12), 2),
                    "beta": round(np.random.uniform(0.5, 2.5), 2),
                    "short_pct_float": round(np.random.uniform(0.01, 0.15), 3),
                    "target_price": round(details["price"] * np.random.uniform(0.9, 1.3), 2),
                    "recommendation": np.random.choice(["buy", "hold", "strong_buy", "sell"]),
                    "dividend_yield": round(np.random.uniform(0, 0.04), 3),
                    "industry": "Synthetic Test"
                }
            })
            ytd_returns.append(details["ytd_return"])

        stocks.sort(key=lambda x: x["ytd_return"], reverse=True)
        avg_ytd = round(np.mean(ytd_returns), 2)
        avg_score = round(np.mean([s["score"] for s in stocks]), 1)

        group_signal = "strong-buy" if avg_score >= 70 else "buy" if avg_score >= 58 else "hold" if avg_score >= 45 else "sell"

        groups_output.append({
            "name": group_name,
            "gics_code": "00000000",
            "gics_level": "Sub-Industry",
            "sector": group_info["sector"],
            "industry_group": group_info["sector"],
            "thesis": group_info["thesis"],
            "thesis_breaker": group_info["thesis_breaker"],
            "cycle_stage": group_info["cycle_stage"],
            "avg_ytd": avg_ytd,
            "avg_score": avg_score,
            "group_signal": group_signal,
            "stock_count": len(stocks),
            "beating_sp500_count": sum(1 for s in stocks if s["beating_sp500"]),
            "breaker_status": "clear",
            "breaker_alerts": [
                {"check": "group_avg_rsi_below_40", "severity": "clear", "triggered": False,
                 "message": "Not triggered", "description": "Group avg RSI falls below 40", "value": None},
                {"check": "majority_below_ma50", "severity": "clear", "triggered": False,
                 "message": "Not triggered", "description": ">50% of stocks fall below 50-day MA", "value": None}
            ],
            "stocks": stocks
        })

    groups_output.sort(key=lambda x: x["avg_ytd"], reverse=True)
    for i, g in enumerate(groups_output):
        g["rank"] = i + 1

    output = {
        "timestamp": datetime.datetime.now().isoformat(),
        "sp500_ytd": sp500_ytd,
        "total_tickers": sum(len(g["stocks"]) for g in groups_output),
        "total_groups": len(groups_output),
        "groups": groups_output
    }

    # Save to data and public dirs
    os.makedirs(DATA_DIR, exist_ok=True)
    os.makedirs(PUBLIC_DIR, exist_ok=True)

    signals_path = os.path.join(DATA_DIR, "signals.json")
    with open(signals_path, "w") as f:
        json.dump(output, f, indent=2, cls=NumpyEncoder)

    public_signals_path = os.path.join(PUBLIC_DIR, "signals.json")
    with open(public_signals_path, "w") as f:
        json.dump(output, f, indent=2, cls=NumpyEncoder)

    print(f"\n  Generated {len(groups_output)} groups with {output['total_tickers']} tickers")
    for g in groups_output:
        print(f"  #{g['rank']} {g['name']}: Avg YTD {g['avg_ytd']}%, Signal={g['group_signal'].upper()}, Score={g['avg_score']}")
        for s in g["stocks"]:
            print(f"      {s['ticker']}: YTD {s['ytd_return']}%, Score={s['score']}, Signal={s['signal']}")

    print(f"\n  signals.json written to {signals_path}")
    print(f"  signals.json written to {public_signals_path}")
    print("\n  ✅ Full pipeline test passed!\n")

    return output


def test_history():
    """Test history detection by running pipeline twice with changes."""
    print("=" * 60)
    print("TEST 4: History Manager")
    print("=" * 60)

    # First run — creates baseline
    print("\n  [Run 1] Creating baseline...")
    run_history_manager()

    # Modify signals.json to simulate changes
    signals_path = os.path.join(DATA_DIR, "signals.json")
    with open(signals_path) as f:
        data = json.load(f)

    # Simulate changes
    if data["groups"]:
        # Change a group signal
        data["groups"][0]["group_signal"] = "sell"
        data["groups"][0]["avg_score"] = 35

        # Change a stock signal
        if data["groups"][0]["stocks"]:
            data["groups"][0]["stocks"][0]["signal"] = "sell"
            data["groups"][0]["stocks"][0]["score"] = 25

        # Remove a stock from group 1
        if len(data["groups"]) > 1 and data["groups"][1]["stocks"]:
            removed = data["groups"][1]["stocks"].pop()
            print(f"  Simulated removing {removed['ticker']} from {data['groups'][1]['name']}")

        # Add a new group
        data["groups"].append({
            "name": "TEST — New Group",
            "sector": "Test",
            "thesis": "Testing",
            "thesis_breaker": "Testing",
            "cycle_stage": "test",
            "avg_ytd": 50.0,
            "avg_score": 65,
            "group_signal": "buy",
            "stock_count": 1,
            "beating_sp500_count": 1,
            "rank": len(data["groups"]) + 1,
            "stocks": [{"ticker": "TEST", "score": 65, "signal": "buy", "ytd_return": 50, "price": 100, "rsi": 55, "macd": 0.5, "macd_signal": 0.3, "macd_histogram": 0.2, "ma20": 95, "ma50": 90, "volume_ratio": 1.3, "beating_sp500": True}]
        })

    data["timestamp"] = datetime.datetime.now().isoformat()

    with open(signals_path, "w") as f:
        json.dump(data, f, indent=2)

    # Also update public copy
    with open(os.path.join(PUBLIC_DIR, "signals.json"), "w") as f:
        json.dump(data, f, indent=2)

    # Second run — should detect changes
    print("\n  [Run 2] Detecting changes...")
    changes = run_history_manager()

    print(f"\n  Detected {len(changes)} changes total")
    assert len(changes) > 0, "No changes detected — history manager failed"

    # Verify history.json exists and has content
    history_path = os.path.join(DATA_DIR, "history.json")
    assert os.path.exists(history_path), "history.json not created"
    with open(history_path) as f:
        history = json.load(f)
    assert len(history["changes"]) > 0, "No changes in history.json"
    assert len(history["snapshots"]) >= 2, "Less than 2 snapshots"

    # Verify public copy
    public_history = os.path.join(PUBLIC_DIR, "history.json")
    assert os.path.exists(public_history), "public/history.json not created"

    print(f"\n  history.json has {len(history['changes'])} changes and {len(history['snapshots'])} snapshots")
    print("\n  ✅ History manager tests passed!\n")


if __name__ == "__main__":
    test_indicators()
    test_scoring()
    test_full_pipeline()
    test_history()

    print("=" * 60)
    print("  ALL TESTS PASSED ✅")
    print("=" * 60)
    print(f"\n  Dashboard:  public/index.html")
    print(f"  History:    public/history.html")
    print(f"  Data:       data/signals.json")
    print(f"  Changes:    data/history.json")
    print(f"  Snapshots:  data/snapshots/\n")
