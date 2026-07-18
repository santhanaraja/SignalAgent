#!/usr/bin/env python3
"""
Rule Engine — Evaluates standing rules R1-R27 against the current regime and
REAL portfolio state. Produces a rule log with compliance status, violations,
and action items.

D-007 Phase 2: the active-state trigger for the 13 behavioral reminder rules
derives from REAL HOLDINGS' GICS groups (positions.json resolved through the
shared resolver — the same map condition 5 and R28 consume), not from
qualified themes. R5's theme-count semantics are superseded by R28's
per-group caps. theme_result is retained ONLY for the display-only rotation
entry/exit signals R3/R4 echo until Phase 3 deletes the layer — no
qualified-theme STATE is consumed here anymore.
"""

import datetime


class RuleEngine:
    """Evaluate standing rules against current framework state."""

    def __init__(self, config: dict):
        self.rules = config.get("standing_rules", {})

    def evaluate(self, regime_result: dict, theme_result: dict = None,
                 active_groups: list = None) -> dict:
        """
        Evaluate all 27 standing rules.

        Args:
            regime_result: Output from RegimeCalculator.compute()
            theme_result: ThemeRanker output — display-only rotation
                entry/exit signals for R3/R4 (Phase 3 removes)
            active_groups: the REAL holdings' GICS groups (runner resolves
                positions.json through the shared resolver; unmapped
                holdings appear as "TICKER (ungrouped)" — R28's own bucket
                convention). Empty/None = flat book.

        Returns dict with rule evaluations, violations, and summary.
        """
        regime = regime_result.get("regime", "Unknown")
        theme_result = theme_result or {}
        entry_signals = theme_result.get("entry_signals", [])
        exit_signals = theme_result.get("exit_signals", [])
        # None = active state UNKNOWN (positions unreadable — the runner's
        # degrade path); [] = genuinely flat. An outage must never render
        # as a confident flat book (review finding). str() coercion +
        # dedupe: malformed caller input must never TypeError at sorted/join.
        active_unknown = active_groups is None
        active_groups = sorted({str(g) for g in (active_groups or []) if g})

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
        r2 = self._eval_r2(regime)
        evaluations.append(r2)
        if r2["status"] == "violation":
            violations.append(r2)

        # --- R3: New theme entry requirements ---
        r3 = self._eval_r3(entry_signals)
        evaluations.append(r3)
        if r3["status"] == "action_needed":
            action_items.append(r3)

        # --- R4: Theme exit conditions ---
        r4 = self._eval_r4(exit_signals)
        evaluations.append(r4)
        if r4["status"] in ("violation", "action_needed"):
            action_items.append(r4)

        # --- R5: SUPERSEDED by R28's per-group caps (D-007 Phase 2) ---
        evaluations.append(self._eval_r5())

        # --- R6-R27: Behavioral / structural rules (evaluated as reminders) ---
        behavioral_rules = {
            "R6": self._eval_behavioral(regime, active_groups, "R6", "invalidation", active_unknown),
            "R7": self._eval_behavioral(regime, active_groups, "R7", "sizing", active_unknown),
            "R8": self._eval_behavioral(regime, active_groups, "R8", "binary_catalyst", active_unknown),
            "R9": self._eval_behavioral(regime, active_groups, "R9", "compliance", active_unknown),
            "R10": self._eval_behavioral(regime, active_groups, "R10", "stop_management", active_unknown),
            "R11": self._eval_behavioral(regime, active_groups, "R11", "close_based", active_unknown),
            "R12": self._eval_behavioral(regime, active_groups, "R12", "timeframe", active_unknown),
            "R13": self._eval_behavioral(regime, active_groups, "R13", "stop_override", active_unknown),
            "R14": self._eval_behavioral(regime, active_groups, "R14", "profit_taking", active_unknown),
            "R15": self._eval_behavioral(regime, active_groups, "R15", "position_size", active_unknown),
            "R16": self._eval_behavioral(regime, active_groups, "R16", "theme_concentration", active_unknown),
            "R17": self._eval_behavioral(regime, active_groups, "R17", "total_theme_exposure", active_unknown),
            "R18": self._eval_behavioral(regime, active_groups, "R18", "cash_reserve", active_unknown),
            "R19": self._eval_behavioral(regime, active_groups, "R19", "cost_basis", active_unknown),
            "R20": self._eval_behavioral(regime, active_groups, "R20", "cross_funding", active_unknown),
            "R21": self._eval_behavioral(regime, active_groups, "R21", "same_day_entry", active_unknown),
            "R22": self._eval_behavioral(regime, active_groups, "R22", "hot_tape", active_unknown),
            "R23": self._eval_behavioral(regime, active_groups, "R23", "conviction_sizing", active_unknown),
            "R24": self._eval_behavioral(regime, active_groups, "R24", "target_hit", active_unknown),
            "R25": self._eval_behavioral(regime, active_groups, "R25", "adding_to_position", active_unknown),
            "R26": self._eval_behavioral(regime, active_groups, "R26", "margin", active_unknown),
            "R27": self._eval_behavioral(regime, active_groups, "R27", "theme_switching", active_unknown),
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
            "active_groups_used": active_groups,
            "active_state_unavailable": active_unknown,
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

    def _eval_r2(self, regime: str) -> dict:
        """R2: Risk-off → no new aggressive positions.

        No violation branch: theme_ranker only emits ENTRY signals when the
        regime is risk-on (its entry gate requires regime in Trending/
        Choppy), so a risk-off ENTRY signal can never reach here. R2
        degrades to an elevated reminder in Risk-off; the actual block is
        enforced upstream in theme_ranker.
        """
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

    def _eval_r3(self, entry_signals: list) -> dict:
        """R3: New theme entry requirements.

        The "risk-on regime" clause of the rule is enforced upstream in
        theme_ranker (which only produces ENTRY signals in risk-on), so
        this reads only the entry signals it is given.
        """
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
        """R4: exit conditions — echoes the display-only rotation exit
        signals (Phase 3 removes). No qualified-theme state consumed."""
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
            "message": "No exit signals.",
        }

    def _eval_r5(self) -> dict:
        """R5 retired (D-007): theme-count concentration is superseded by
        R28's computed per-group caps — the same pointer treatment R17/R18
        received. No count logic remains."""
        return {
            "rule": "R5",
            "text": self.rules.get("R5", ""),
            "status": "compliant",
            "superseded_by": "R28",
            "message": "Superseded → R28 per-group caps: ≤20% / ≤3 positions "
                       "per GICS group (computed, Layer 3). Concentration is "
                       "enforced in dollars, not theme counts (D-007).",
        }

    def _eval_behavioral(self, regime: str, active_groups: list, rule_id: str,
                         category: str, active_unknown: bool = False) -> dict:
        """
        Evaluate behavioral/structural rules.
        These are elevated to 'action_needed' in certain regime contexts.
        """
        text = self.rules.get(rule_id, "")

        # Certain rules become elevated based on regime or active state.
        # "Defensive" = the risk-reducing regimes Risk-off AND Caution
        # (both trigger this set — the name reflects the actual behavior).
        elevated_in_defensive = {
            "R10", "R13", "R15", "R16", "R17", "R18", "R26",
        }
        elevated_when_active = {
            "R6", "R7", "R8", "R10", "R11", "R12", "R13", "R15",
            "R19", "R21", "R23", "R24", "R25",
        }

        is_elevated = False
        reason = ""

        if rule_id in elevated_in_defensive and regime in ("Risk-off", "Caution"):
            is_elevated = True
            reason = f"Elevated — regime is {regime}."
        elif rule_id in elevated_when_active and len(active_groups) > 0:
            is_elevated = True
            reason = f"Active groups: {', '.join(active_groups)}."

        if is_elevated:
            return {
                "rule": rule_id,
                "text": text,
                "status": "elevated",
                "message": f"{reason} {text}",
                "category": category,
            }

        if rule_id in elevated_when_active and not active_groups:
            if active_unknown:
                # positions unreadable: say so — never assert flatness the
                # data cannot prove (review finding; R28/position-engine
                # report the same outage on their own surfaces)
                return {
                    "rule": rule_id,
                    "text": text,
                    "status": "compliant",
                    "message": f"Positions unavailable — active state "
                               f"unknown. {text}",
                    "category": category,
                }
            # flat book: the position-triggered reminder is idle (today's
            # live state — pinned)
            return {
                "rule": rule_id,
                "text": text,
                "status": "compliant",
                "message": f"No active positions. {text}",
                "category": category,
            }
        return {
            "rule": rule_id,
            "text": text,
            "status": "compliant",
            "message": text,
            "category": category,
        }
