#!/usr/bin/env python3
"""
Rule Engine — Evaluates standing rules R1-R27 against current regime and theme state.
Produces a rule log with compliance status, violations, and action items.
"""

import datetime


class RuleEngine:
    """Evaluate standing rules against current framework state."""

    def __init__(self, config: dict):
        self.rules = config.get("standing_rules", {})
        self.themes_cfg = config.get("themes", {})

    def evaluate(self, regime_result: dict, theme_result: dict) -> dict:
        """
        Evaluate all 27 standing rules.

        Args:
            regime_result: Output from RegimeCalculator.compute()
            theme_result: Output from ThemeRanker.compute()

        Returns dict with rule evaluations, violations, and summary.
        """
        regime = regime_result.get("regime", "Unknown")
        active_themes = theme_result.get("active_themes", [])
        entry_signals = theme_result.get("entry_signals", [])
        exit_signals = theme_result.get("exit_signals", [])

        evaluations = []
        violations = []
        action_items = []

        today = datetime.date.today()
        is_sunday = today.weekday() == 6

        # --- R1: Weekly Sunday review ---
        r1 = self._eval_r1(is_sunday)
        evaluations.append(r1)
        if r1["status"] == "action_needed":
            action_items.append(r1)

        # --- R2: Risk-off → no new aggressive ---
        r2 = self._eval_r2(regime, entry_signals)
        evaluations.append(r2)
        if r2["status"] == "violation":
            violations.append(r2)

        # --- R3: New theme entry requirements ---
        r3 = self._eval_r3(entry_signals, regime)
        evaluations.append(r3)
        if r3["status"] == "action_needed":
            action_items.append(r3)

        # --- R4: Theme exit conditions ---
        r4 = self._eval_r4(exit_signals)
        evaluations.append(r4)
        if r4["status"] in ("violation", "action_needed"):
            action_items.append(r4)

        # --- R5: Maximum 2 themes ---
        r5 = self._eval_r5(active_themes)
        evaluations.append(r5)
        if r5["status"] == "violation":
            violations.append(r5)

        # --- R6-R27: Behavioral / structural rules (evaluated as reminders) ---
        behavioral_rules = {
            "R6": self._eval_behavioral(regime, active_themes, "R6", "invalidation"),
            "R7": self._eval_behavioral(regime, active_themes, "R7", "sizing"),
            "R8": self._eval_behavioral(regime, active_themes, "R8", "binary_catalyst"),
            "R9": self._eval_behavioral(regime, active_themes, "R9", "compliance"),
            "R10": self._eval_behavioral(regime, active_themes, "R10", "stop_management"),
            "R11": self._eval_behavioral(regime, active_themes, "R11", "close_based"),
            "R12": self._eval_behavioral(regime, active_themes, "R12", "timeframe"),
            "R13": self._eval_behavioral(regime, active_themes, "R13", "stop_override"),
            "R14": self._eval_behavioral(regime, active_themes, "R14", "profit_taking"),
            "R15": self._eval_behavioral(regime, active_themes, "R15", "position_size"),
            "R16": self._eval_behavioral(regime, active_themes, "R16", "theme_concentration"),
            "R17": self._eval_behavioral(regime, active_themes, "R17", "total_theme_exposure"),
            "R18": self._eval_behavioral(regime, active_themes, "R18", "cash_reserve"),
            "R19": self._eval_behavioral(regime, active_themes, "R19", "cost_basis"),
            "R20": self._eval_behavioral(regime, active_themes, "R20", "cross_funding"),
            "R21": self._eval_behavioral(regime, active_themes, "R21", "same_day_entry"),
            "R22": self._eval_behavioral(regime, active_themes, "R22", "hot_tape"),
            "R23": self._eval_behavioral(regime, active_themes, "R23", "conviction_sizing"),
            "R24": self._eval_behavioral(regime, active_themes, "R24", "target_hit"),
            "R25": self._eval_behavioral(regime, active_themes, "R25", "adding_to_position"),
            "R26": self._eval_behavioral(regime, active_themes, "R26", "margin"),
            "R27": self._eval_behavioral(regime, active_themes, "R27", "theme_switching"),
        }

        for rule_id, eval_result in behavioral_rules.items():
            evaluations.append(eval_result)
            if eval_result["status"] == "elevated":
                action_items.append(eval_result)

        # Summary
        compliant_count = sum(1 for e in evaluations if e["status"] == "compliant")
        action_count = sum(1 for e in evaluations if e["status"] in ("action_needed", "elevated"))
        violation_count = sum(1 for e in evaluations if e["status"] == "violation")

        return {
            "date": datetime.date.today().isoformat(),
            "evaluations": evaluations,
            "violations": violations,
            "action_items": action_items,
            "summary": {
                "total_rules": len(evaluations),
                "compliant": compliant_count,
                "action_needed": action_count,
                "violations": violation_count,
            },
            "regime_used": regime,
            "active_themes_used": active_themes,
        }

    # ==============================================================
    # Individual rule evaluators
    # ==============================================================

    def _eval_r1(self, is_sunday: bool) -> dict:
        """R1: Weekly Sunday review."""
        if is_sunday:
            return {
                "rule": "R1",
                "text": self.rules.get("R1", ""),
                "status": "action_needed",
                "message": "Today is Sunday — weekly review required. Complete regime + theme analysis.",
            }
        return {
            "rule": "R1",
            "text": self.rules.get("R1", ""),
            "status": "compliant",
            "message": "Not Sunday. Next review on Sunday.",
        }

    def _eval_r2(self, regime: str, entry_signals: list) -> dict:
        """R2: Risk-off → no new aggressive positions."""
        if regime == "Risk-off" and any(s.get("action", "").startswith("ENTRY") for s in entry_signals):
            return {
                "rule": "R2",
                "text": self.rules.get("R2", ""),
                "status": "violation",
                "message": f"VIOLATION: Regime is {regime} but entry signals exist. Block all new aggressive positions.",
            }
        if regime == "Risk-off":
            return {
                "rule": "R2",
                "text": self.rules.get("R2", ""),
                "status": "elevated",
                "message": f"Regime is {regime}. No new aggressive positions allowed.",
            }
        return {
            "rule": "R2",
            "text": self.rules.get("R2", ""),
            "status": "compliant",
            "message": f"Regime is {regime}. Rule not triggered.",
        }

    def _eval_r3(self, entry_signals: list, regime: str) -> dict:
        """R3: New theme entry requirements."""
        qualified = [s for s in entry_signals if "ENTRY SIGNAL" in s.get("action", "")]
        building = [s for s in entry_signals if "Building" in s.get("action", "")]

        if qualified:
            names = ", ".join(s["theme"] for s in qualified)
            return {
                "rule": "R3",
                "text": self.rules.get("R3", ""),
                "status": "action_needed",
                "message": f"Entry signals qualified: {names}. Requires discretionary catalyst review before activation.",
            }
        if building:
            names = ", ".join(f"{s['theme']} ({s['action']})" for s in building)
            return {
                "rule": "R3",
                "text": self.rules.get("R3", ""),
                "status": "compliant",
                "message": f"Themes building qualification: {names}.",
            }
        return {
            "rule": "R3",
            "text": self.rules.get("R3", ""),
            "status": "compliant",
            "message": "No new entry signals.",
        }

    def _eval_r4(self, exit_signals: list) -> dict:
        """R4: Theme exit conditions."""
        exits = [s for s in exit_signals if "EXIT SIGNAL" in s.get("action", "")]
        warnings = [s for s in exit_signals if "Warning" in s.get("action", "")]

        if exits:
            names = ", ".join(f"{s['theme']}: {s.get('reason', '')}" for s in exits)
            return {
                "rule": "R4",
                "text": self.rules.get("R4", ""),
                "status": "action_needed",
                "message": f"EXIT SIGNALS FIRED: {names}. Close positions per rule.",
            }
        if warnings:
            names = ", ".join(f"{s['theme']} ({s['action']})" for s in warnings)
            return {
                "rule": "R4",
                "text": self.rules.get("R4", ""),
                "status": "compliant",
                "message": f"Exit warnings: {names}.",
            }
        return {
            "rule": "R4",
            "text": self.rules.get("R4", ""),
            "status": "compliant",
            "message": "No exit conditions triggered.",
        }

    def _eval_r5(self, active_themes: list) -> dict:
        """R5: Maximum 2 active themes."""
        max_themes = self.themes_cfg.get("ranking", {}).get("max_active_themes", 2)
        if len(active_themes) > max_themes:
            return {
                "rule": "R5",
                "text": self.rules.get("R5", ""),
                "status": "violation",
                "message": f"VIOLATION: {len(active_themes)} active themes exceeds max {max_themes}. Reduce to {max_themes}.",
            }
        return {
            "rule": "R5",
            "text": self.rules.get("R5", ""),
            "status": "compliant",
            "message": f"{len(active_themes)}/{max_themes} active themes. Cash is the default.",
        }

    def _eval_behavioral(self, regime: str, active_themes: list, rule_id: str, category: str) -> dict:
        """
        Evaluate behavioral/structural rules.
        These are elevated to 'action_needed' in certain regime contexts.
        """
        text = self.rules.get(rule_id, "")

        # Certain rules become elevated based on regime or active state
        elevated_in_risk_off = {
            "R10", "R13", "R15", "R16", "R17", "R18", "R26",
        }
        elevated_when_active = {
            "R6", "R7", "R8", "R10", "R11", "R12", "R13", "R15",
            "R19", "R21", "R23", "R24", "R25",
        }
        elevated_on_entry = {
            "R6", "R7", "R8", "R9", "R15",
        }

        is_elevated = False
        reason = ""

        if rule_id in elevated_in_risk_off and regime in ("Risk-off", "Caution"):
            is_elevated = True
            reason = f"Elevated — regime is {regime}."
        elif rule_id in elevated_when_active and len(active_themes) > 0:
            is_elevated = True
            reason = f"Active themes: {', '.join(active_themes)}."

        if is_elevated:
            return {
                "rule": rule_id,
                "text": text,
                "status": "elevated",
                "message": f"{reason} {text}",
                "category": category,
            }

        return {
            "rule": rule_id,
            "text": text,
            "status": "compliant",
            "message": text,
            "category": category,
        }
