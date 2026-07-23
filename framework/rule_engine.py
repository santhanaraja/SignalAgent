#!/usr/bin/env python3
"""
Rule Engine — Evaluates standing rules R1-R27 against the current regime and
REAL portfolio state. Produces a rule log with compliance status, violations,
and action items.

D-007 Phase 2: the active-state trigger for the 13 behavioral reminder rules
derives from REAL HOLDINGS' GICS groups (positions.json resolved through the
shared resolver — the same map condition 5 and R28 consume), not from
qualified themes. R5's theme-count semantics are superseded by R28's
per-group caps.

D-007 Phase 3: the theme layer is decommissioned. R3/R4 are pointer rows
(superseded by the universe rotation + the position engine's stops and
breakers); theme_result is accepted for back-compat but never read.
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
            theme_result: back-compat only — ignored (D-007 Phase 3)
            active_groups: the REAL holdings' GICS groups (runner resolves
                positions.json through the shared resolver; unmapped
                holdings appear as "TICKER (ungrouped)" — R28's own bucket
                convention). Empty/None = flat book.

        Returns dict with rule evaluations, violations, and summary.
        """
        regime = regime_result.get("regime", "Unknown")
        # theme_result is accepted for caller back-compat but NO LONGER
        # consumed (D-007 Phase 3): the rotation-signal echoes R3/R4 used
        # to read are retired with the theme layer itself.
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

        # --- R3/R4: SUPERSEDED by the universe rotation + the position
        # engine (D-007 Phase 3) — pointer rows, no signal consumption ---
        evaluations.append(self._eval_r3())
        evaluations.append(self._eval_r4())

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
                "message": "Today is Sunday — weekly review required. "
                           "Complete regime + universe review.",
            }
        return {
            "rule": "R1",
            "text": self.rules.get("R1", ""),
            "status": "compliant",
            "message": "Not Sunday. Next review on Sunday.",
        }

    def _eval_r2(self, regime: str) -> dict:
        """R2: Risk-off → no new aggressive positions.

        Reminder-only: the actual entry block is enforced upstream — the
        position engine's condition 3 blocks re-entry outside the risk-on
        regimes, and the A+ hard gate governs Choppy (D-011).
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

    def _eval_r3(self) -> dict:
        """R3 retired (D-007 Phase 3): theme entry signals no longer exist.
        Entries qualify through the weekly universe rotation (top-15 GICS
        scanner) + the A+ doctrine (D-011) — the R5→R28 pointer treatment."""
        return {
            "rule": "R3",
            "text": self.rules.get("R3", ""),
            "status": "compliant",
            "superseded_by": "universe rotation (D-007)",
            "message": "Superseded → universe rotation + A+ doctrine: entries "
                       "qualify via the weekly top-15 GICS scanner and the "
                       "D-011 grade — theme entry signals retired (D-007 "
                       "Phase 3).",
        }

    def _eval_r4(self) -> dict:
        """R4 retired (D-007 Phase 3): theme exit warnings no longer exist.
        Exits are governed by the position engine's close-basis stops (R11)
        and the per-group breakers — the R5→R28 pointer treatment."""
        return {
            "rule": "R4",
            "text": self.rules.get("R4", ""),
            "status": "compliant",
            "superseded_by": "position engine stops + group breakers",
            "message": "Superseded → close-basis stops (R10/R11, the 1B "
                       "engine's EXIT_FIRED) + per-group breakers: theme "
                       "exit warnings retired (D-007 Phase 3).",
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
