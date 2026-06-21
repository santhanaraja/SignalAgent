#!/usr/bin/env python3
"""
Theme Ranker — Ranks 8 theme proxies by composite momentum (4w + 12w return),
tracks consecutive-Sunday qualification, and determines entry/exit signals.
"""

import datetime
import json
import os
import numpy as np


class ThemeRanker:
    """Rank themes by composite momentum and manage qualification state."""

    STATE_DIR = os.path.join(os.path.dirname(__file__), "state")

    def __init__(self, config: dict, fetcher):
        """
        Args:
            config: Full framework config dict
            fetcher: function(ticker, period) -> DataFrame
        """
        self.themes_cfg = config["themes"]
        self.fetcher = fetcher
        os.makedirs(self.STATE_DIR, exist_ok=True)

    def _compute_theme_returns(self, proxy: str, ranking_cfg: dict) -> dict:
        """Compute 4w and 12w returns + composite rank score for a proxy."""
        try:
            df = self.fetcher(proxy, period="6mo")
            if df is None or len(df) < ranking_cfg.get("lookback_12w", 60):
                return {
                    "return_4w": None,
                    "return_12w": None,
                    "composite": None,
                    "detail": f"{proxy}: insufficient data ({len(df) if df is not None else 0} days)",
                }

            close = df["Close"]
            lookback_4w = ranking_cfg.get("lookback_4w", 20)
            lookback_12w = ranking_cfg.get("lookback_12w", 60)

            current = float(close.iloc[-1])
            price_4w_ago = float(close.iloc[-lookback_4w]) if len(close) >= lookback_4w else float(close.iloc[0])
            price_12w_ago = float(close.iloc[-lookback_12w]) if len(close) >= lookback_12w else float(close.iloc[0])

            ret_4w = ((current - price_4w_ago) / price_4w_ago) * 100
            ret_12w = ((current - price_12w_ago) / price_12w_ago) * 100

            return {
                "return_4w": round(ret_4w, 2),
                "return_12w": round(ret_12w, 2),
                "composite": None,  # Will be set after ranking
                "current_price": round(current, 2),
                "detail": f"{proxy}: 4w {ret_4w:+.1f}%, 12w {ret_12w:+.1f}%",
            }
        except Exception as e:
            return {
                "return_4w": None,
                "return_12w": None,
                "composite": None,
                "detail": f"{proxy}: error — {e}",
            }

    def compute(self, regime_state: str) -> dict:
        """
        Rank all themes, evaluate entry/exit, return full result.
        """
        watchlist = self.themes_cfg["watchlist"]
        ranking_cfg = self.themes_cfg["ranking"]
        entry_cfg = self.themes_cfg["entry_rule"]
        exit_cfg = self.themes_cfg["exit_rule"]

        # --- Compute returns ---
        theme_data = []
        for theme in watchlist:
            name = theme["name"]
            proxy = theme["proxy"]
            data = self._compute_theme_returns(proxy, ranking_cfg)
            data["name"] = name
            data["proxy"] = proxy
            theme_data.append(data)

        # --- Composite ranking (rank average method) ---
        available = [t for t in theme_data if t.get("return_4w") is not None and t.get("return_12w") is not None]
        unavailable = [t for t in theme_data if t.get("return_4w") is None or t.get("return_12w") is None]

        if len(available) > 1:
            # Rank by 4w return (descending)
            sorted_4w = sorted(available, key=lambda x: x["return_4w"], reverse=True)
            for i, t in enumerate(sorted_4w):
                t["rank_4w"] = i + 1

            # Rank by 12w return (descending)
            sorted_12w = sorted(available, key=lambda x: x["return_12w"], reverse=True)
            for i, t in enumerate(sorted_12w):
                t["rank_12w"] = i + 1

            # Composite = average of two ranks (lower is better)
            for t in available:
                t["composite"] = (t["rank_4w"] + t["rank_12w"]) / 2.0

            # Sort by composite ascending (rank 1 = best)
            available.sort(key=lambda x: x["composite"])
        elif len(available) == 1:
            available[0]["rank_4w"] = 1
            available[0]["rank_12w"] = 1
            available[0]["composite"] = 1.0

        # Assign final rank
        for i, t in enumerate(available):
            t["rank"] = i + 1
        for t in unavailable:
            t["rank"] = None
            t["rank_4w"] = None
            t["rank_12w"] = None

        ranked = available + unavailable

        # --- Load state ---
        theme_history = self._load_theme_history()
        qualified = self._load_qualified_themes()
        active_records = qualified.get("active", [])
        # The active list may hold legacy name-strings or {name, proxy, ...}
        # records; normalize to a list of names for all in-memory logic.
        active_themes = [self._active_name(a) for a in active_records]
        price_by_name = {t["name"]: t.get("current_price") for t in theme_data}

        # Consecutive-week counts must look only at PAST weeks. compute() runs
        # before save_weekly_snapshot(), so on a same-day re-run today's snapshot
        # is already in history — exclude it so this week isn't double-counted.
        today_iso = datetime.date.today().isoformat()
        past_history = [h for h in theme_history if h.get("date") != today_iso]

        # --- Entry signals ---
        entry_signals = []
        allowed_regimes = entry_cfg.get("regime_required", [])
        regime_ok = regime_state in allowed_regimes
        consec_needed = entry_cfg.get("consecutive_sundays_required", 2)
        top_n = entry_cfg.get("requires_top_n_rank", 2)
        max_active = ranking_cfg.get("max_active_themes", 2)

        for t in available:
            consec = self._count_consecutive_top_n(t["name"], past_history, top_n)
            t["consecutive_top_n_weeks"] = consec + 1 if t["rank"] <= top_n else 0

            if t["rank"] <= top_n and t["name"] not in active_themes:
                if consec + 1 >= consec_needed and regime_ok and len(active_themes) < max_active:
                    entry_signals.append({
                        "theme": t["name"],
                        "proxy": t["proxy"],
                        "rank": t["rank"],
                        "consecutive_weeks": consec + 1,
                        "action": "ENTRY SIGNAL — qualified for activation",
                        "requires_discretionary_review": entry_cfg.get("discretionary_review_required", True),
                    })
                elif consec + 1 < consec_needed:
                    entry_signals.append({
                        "theme": t["name"],
                        "proxy": t["proxy"],
                        "rank": t["rank"],
                        "consecutive_weeks": consec + 1,
                        "action": f"Building — {consec + 1}/{consec_needed} weeks in top {top_n}",
                        "requires_discretionary_review": False,
                    })

        # --- Exit signals ---
        exit_signals = []
        exit_rank_threshold = exit_cfg.get("drops_below_rank", 3)
        exit_consec_needed = exit_cfg.get("consecutive_sundays_required", 3)

        for active_name in active_themes:
            theme_ranked = next((t for t in ranked if t["name"] == active_name), None)
            if theme_ranked is None:
                exit_signals.append({
                    "theme": active_name,
                    "action": "EXIT — theme data unavailable",
                    "reason": "unavailable",
                })
                continue

            if theme_ranked["rank"] is not None and theme_ranked["rank"] > exit_rank_threshold:
                consec_below = self._count_consecutive_below_rank(active_name, past_history, exit_rank_threshold)
                if consec_below + 1 >= exit_consec_needed:
                    exit_signals.append({
                        "theme": active_name,
                        "proxy": theme_ranked["proxy"],
                        "rank": theme_ranked["rank"],
                        "consecutive_weeks_below": consec_below + 1,
                        "action": f"EXIT SIGNAL — below #{exit_rank_threshold} for {consec_below + 1} weeks",
                        "reason": "rank_degradation",
                    })
                else:
                    exit_signals.append({
                        "theme": active_name,
                        "proxy": theme_ranked["proxy"],
                        "rank": theme_ranked["rank"],
                        "consecutive_weeks_below": consec_below + 1,
                        "action": f"Warning — {consec_below + 1}/{exit_consec_needed} weeks below #{exit_rank_threshold}",
                        "reason": "rank_warning",
                    })

            if exit_cfg.get("regime_degradation_triggers_exit", True) and regime_state in ("Caution", "Risk-off"):
                exit_signals.append({
                    "theme": active_name,
                    "proxy": theme_ranked.get("proxy", ""),
                    "action": f"EXIT SIGNAL — regime degraded to {regime_state}",
                    "reason": "regime_degradation",
                })

        # --- Update active membership in qualified_themes.json ---
        # Entry signal -> add to active (with entry date + proxy price).
        # Exit signal  -> remove from active. This closes the lifecycle so
        # future runs can evaluate exit conditions against real positions.
        self._update_active_state(qualified, active_records, entry_signals,
                                  exit_signals, price_by_name)

        # Concentration limits
        conc = self.themes_cfg.get("concentration_limits", {})
        concentration_notes = []
        max_per = conc.get("max_per_theme_pct_of_book", 15)
        max_total = conc.get("max_total_themes_pct_of_book", 25)
        concentration_notes.append(f"Max per theme: {max_per}% of book")
        concentration_notes.append(f"Max total themes: {max_total}% of book")

        return {
            "date": datetime.date.today().isoformat(),
            "ranked_themes": ranked,
            "entry_signals": entry_signals,
            "exit_signals": exit_signals,
            "active_themes": active_themes,
            "regime_state_used": regime_state,
            "regime_allows_entry": regime_ok,
            "concentration_limits": concentration_notes,
        }

    # ==============================================================
    # History helpers
    # ==============================================================

    def _count_consecutive_top_n(self, theme_name: str, history: list, top_n: int) -> int:
        """Count how many consecutive past weeks this theme was in top N."""
        count = 0
        for entry in reversed(history):
            themes = entry.get("rankings", {})
            rank = themes.get(theme_name, {}).get("rank")
            if rank is not None and rank <= top_n:
                count += 1
            else:
                break
        return count

    def _count_consecutive_below_rank(self, theme_name: str, history: list, threshold: int) -> int:
        """Count consecutive past weeks this theme was below the rank threshold."""
        count = 0
        for entry in reversed(history):
            themes = entry.get("rankings", {})
            rank = themes.get(theme_name, {}).get("rank")
            if rank is not None and rank > threshold:
                count += 1
            else:
                break
        return count

    def _load_theme_history(self) -> list:
        """Load theme_history.json from state dir."""
        path = os.path.join(self.STATE_DIR, "theme_history.json")
        if os.path.exists(path):
            try:
                with open(path, "r") as f:
                    return json.load(f)
            except (json.JSONDecodeError, IOError):
                pass
        return []

    def _load_qualified_themes(self) -> dict:
        """Load qualified_themes.json from state dir."""
        path = os.path.join(self.STATE_DIR, "qualified_themes.json")
        if os.path.exists(path):
            try:
                with open(path, "r") as f:
                    return json.load(f)
            except (json.JSONDecodeError, IOError):
                pass
        return {"active": [], "pending": []}

    @staticmethod
    def _active_name(record) -> str:
        """Return the theme name from an active record (dict) or legacy string."""
        return record["name"] if isinstance(record, dict) else record

    def _update_active_state(self, qualified: dict, active_records: list,
                             entry_signals: list, exit_signals: list,
                             price_by_name: dict):
        """
        Update + persist the qualified_themes.json "active" list for this run.

        Entry signal -> add the theme (with entry date + proxy entry price).
        Exit signal  -> remove the theme. Writes the file only when membership
        actually changes, so idempotent re-runs leave it untouched.
        """
        updated = list(active_records)
        changed = False
        max_active = self.themes_cfg.get("ranking", {}).get("max_active_themes", 2)

        # Remove any theme that fired a confirmed EXIT SIGNAL.
        exited = {s["theme"] for s in exit_signals if "EXIT SIGNAL" in s.get("action", "")}
        if exited:
            pruned = [a for a in updated if self._active_name(a) not in exited]
            if len(pruned) != len(updated):
                updated = pruned
                changed = True

        # Add any theme that fired a confirmed ENTRY SIGNAL.
        today_iso = datetime.date.today().isoformat()
        current_names = {self._active_name(a) for a in updated}
        for s in entry_signals:
            if "ENTRY SIGNAL" not in s.get("action", ""):
                continue
            if len(updated) >= max_active:
                break  # never persist more than max_active_themes positions
            name = s["theme"]
            if name in current_names:
                continue
            updated.append({
                "name": name,
                "proxy": s.get("proxy"),
                "entry_date": today_iso,
                "entry_price": price_by_name.get(name),
            })
            current_names.add(name)
            changed = True

        if changed:
            qualified["active"] = updated
            self._save_qualified_themes(qualified)

    def _save_qualified_themes(self, qualified: dict):
        """Persist qualified_themes.json (active/pending lifecycle state)."""
        path = os.path.join(self.STATE_DIR, "qualified_themes.json")
        with open(path, "w") as f:
            json.dump(qualified, f, indent=2)

    def save_weekly_snapshot(self, result: dict):
        """Append this week's ranking to theme_history.json."""
        history = self._load_theme_history()
        snapshot = {
            "date": result["date"],
            "rankings": {},
        }
        for t in result["ranked_themes"]:
            snapshot["rankings"][t["name"]] = {
                "rank": t.get("rank"),
                "return_4w": t.get("return_4w"),
                "return_12w": t.get("return_12w"),
                "composite": t.get("composite"),
            }

        # Avoid duplicate dates
        history = [h for h in history if h.get("date") != result["date"]]
        history.append(snapshot)

        # Keep last 52 weeks
        history = history[-52:]

        path = os.path.join(self.STATE_DIR, "theme_history.json")
        with open(path, "w") as f:
            json.dump(history, f, indent=2)
