#!/usr/bin/env python3
"""
Framework Runner — Daily orchestrator that runs Regime → Theme → Rules
pipeline and writes output to framework/output/latest.json.
"""

import datetime
import json
import os
import traceback

import yaml
import yfinance as yf
import pandas as pd

from .regime_calculator import RegimeCalculator
from .theme_ranker import ThemeRanker
from .rule_engine import RuleEngine


BASE_DIR = os.path.dirname(os.path.abspath(__file__))
CONFIG_PATH = os.path.join(BASE_DIR, "config.yaml")
OUTPUT_DIR = os.path.join(BASE_DIR, "output")
STATE_DIR = os.path.join(BASE_DIR, "state")


def load_config() -> dict:
    """Load framework config from YAML."""
    with open(CONFIG_PATH, "r") as f:
        return yaml.safe_load(f)


def fetch_data(ticker: str, period: str = "1y") -> pd.DataFrame:
    """
    Fetch OHLCV data via yfinance.
    Returns DataFrame or None on failure.
    """
    try:
        tk = yf.Ticker(ticker)
        df = tk.history(period=period, auto_adjust=True)
        if df is None or df.empty:
            return None
        # Flatten multi-index columns if present
        if isinstance(df.columns, pd.MultiIndex):
            df.columns = df.columns.get_level_values(0)
        return df
    except Exception as e:
        print(f"[framework] fetch_data({ticker}, {period}) failed: {e}")
        return None


def load_regime_history() -> list:
    """Load regime_history.json."""
    path = os.path.join(STATE_DIR, "regime_history.json")
    if os.path.exists(path):
        try:
            with open(path, "r") as f:
                return json.load(f)
        except (json.JSONDecodeError, IOError):
            pass
    return []


def save_regime_history(history: list, new_entry: dict):
    """Append new regime result to history and save."""
    os.makedirs(STATE_DIR, exist_ok=True)
    # Avoid duplicate dates
    history = [h for h in history if h.get("date") != new_entry.get("date")]
    history.append({
        "date": new_entry.get("date"),
        "regime": new_entry.get("regime"),
        "risk_on_count": new_entry.get("risk_on_count"),
        "caution_count": new_entry.get("caution_count"),
        "risk_off_count": new_entry.get("risk_off_count"),
    })
    # Keep last 52 weeks
    history = history[-52:]
    path = os.path.join(STATE_DIR, "regime_history.json")
    with open(path, "w") as f:
        json.dump(history, f, indent=2)


def run_framework(force_fetch: bool = False) -> dict:
    """
    Run the full framework pipeline.

    Args:
        force_fetch: If True, fetch fresh data even if recently run.

    Returns:
        Complete framework result dict.
    """
    print(f"[framework] Starting framework run at {datetime.datetime.now(datetime.timezone.utc).isoformat()}")

    config = load_config()

    # --- Layer 1: Regime ---
    print("[framework] Computing regime...")
    regime_history = load_regime_history()
    regime_calc = RegimeCalculator(config, fetch_data)
    regime_result = regime_calc.compute(regime_history)
    print(f"[framework] Regime: {regime_result['regime']} ({regime_result['risk_on_count']} risk-on, "
          f"{regime_result['caution_count']} caution, {regime_result['risk_off_count']} risk-off)")

    # Save regime history
    save_regime_history(regime_history, regime_result)

    # --- Layer 2: Theme Rotation ---
    print("[framework] Computing theme rankings...")
    theme_ranker = ThemeRanker(config, fetch_data)
    theme_result = theme_ranker.compute(regime_result["regime"])

    # Save theme history snapshot
    theme_ranker.save_weekly_snapshot(theme_result)

    top_themes = [t for t in theme_result["ranked_themes"] if t.get("rank") is not None]
    for t in top_themes[:3]:
        print(f"[framework]   #{t['rank']} {t['name']} ({t['proxy']}): "
              f"4w {t.get('return_4w', 'N/A')}%, 12w {t.get('return_12w', 'N/A')}%")

    # --- Layer 3: Rules ---
    print("[framework] Evaluating rules...")
    rule_engine = RuleEngine(config)
    rules_result = rule_engine.evaluate(regime_result, theme_result)
    summary = rules_result["summary"]
    print(f"[framework] Rules: {summary['compliant']} compliant, "
          f"{summary['action_needed']} action needed, {summary['violations']} violations")

    # --- Assemble output ---
    output = {
        "generated_at": datetime.datetime.now(datetime.timezone.utc).isoformat(),
        "framework_version": config.get("framework", {}).get("version", "1.0"),
        "regime": regime_result,
        "themes": theme_result,
        "rules": rules_result,
        "standing_rules_text": config.get("standing_rules", {}),
    }

    # --- Write output ---
    os.makedirs(OUTPUT_DIR, exist_ok=True)

    # Latest
    latest_path = os.path.join(OUTPUT_DIR, "latest.json")
    with open(latest_path, "w") as f:
        json.dump(output, f, indent=2, default=str)
    print(f"[framework] Written to {latest_path}")

    # Date-stamped archive
    date_str = datetime.date.today().isoformat()
    archive_path = os.path.join(OUTPUT_DIR, f"framework_{date_str}.json")
    with open(archive_path, "w") as f:
        json.dump(output, f, indent=2, default=str)
    print(f"[framework] Archived to {archive_path}")

    # Also write to public dir for frontend access
    public_dir = os.path.join(BASE_DIR, "..", "public")
    if os.path.isdir(public_dir):
        public_path = os.path.join(public_dir, "framework.json")
        with open(public_path, "w") as f:
            json.dump(output, f, indent=2, default=str)
        print(f"[framework] Written to {public_path}")

    print(f"[framework] Framework run complete.")
    return output


if __name__ == "__main__":
    try:
        result = run_framework(force_fetch=True)
        print(json.dumps(result, indent=2, default=str)[:2000])
    except Exception as e:
        print(f"[framework] ERROR: {e}")
        traceback.print_exc()
