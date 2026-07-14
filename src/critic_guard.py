"""
Critic Guard — lightweight pre-execution validation of Planner intents.

Runs after the Planner proposes an intent but BEFORE the Executor acts on it.
Uses rule-based checks (fast, no API call) and optionally an LLM-based text-only check.
Inspired by the SPIRAL paper (AAAI 2026) which uses a Critic agent alongside the
Planner for dense feedback.

Design: hybrid rule-based + optional LLM.
- Rule checks always run (fast, deterministic).
- LLM check is a lightweight text-only call (no image) for ambiguous cases.
- When rejected, the reason + suggestion are fed back to the Planner for 1 re-gen.
"""

from __future__ import annotations
import json
import logging
from dataclasses import dataclass, field
from typing import Optional

logger = logging.getLogger("critic_guard")

# ---------------------------------------------------------------------------
# System prompt for LLM-based critic (text-only, no image)
# ---------------------------------------------------------------------------

CRITIC_SYSTEM = """You are a CRITIC that validates proposed intents from an embodied agent Planner.
Your job is to decide whether a proposed intent is REASONABLE given what the agent already knows.

Key questions to evaluate:
1. Has the agent already searched {target}? If the intent history shows the agent already approached/opened
   this target and found nothing, the intent is wasteful.
2. Is {target} currently visible and accessible? If yes, approaching it is reasonable.
3. Is there a better unexplored location based on spatial memory? If the target has never been seen,
   and the agent keeps approaching the same place, suggest an alternative.

RULES:
- APPROVE if the intent is making forward progress (approaching a new location, picking up a visible target).
- REJECT if the intent repeats a pattern that has already failed 3+ times.
- REJECT if the agent proposes "approach X" where X has already been opened/searched and found empty.
- If rejecting, provide a CONCRETE suggestion for what the Planner should do instead.

OUTPUT — valid JSON only. { first char, } last char. No markdown.

{
  "approved": true/false,
  "reason": "<1-2 sentences explaining why approved or rejected>",
  "suggestion": "<concrete alternative intent if rejected, or empty string if approved>"
}"""


@dataclass
class CriticStats:
    """Statistics for critic guard evaluation."""
    total_checks: int = 0
    approved: int = 0
    rejected: int = 0
    rule_rejected: int = 0
    llm_rejected: int = 0
    regen_attempted: int = 0
    regen_approved: int = 0
    regen_rejected: int = 0
    rejection_reasons: dict = field(default_factory=dict)
    # Accuracy tracking
    correct_rejections: int = 0      # rejection was correct (intent was indeed wasteful)
    incorrect_rejections: int = 0    # rejection was wrong (intent would have been useful)
    false_approvals: int = 0         # approved but intent led to wasted step

    def to_dict(self) -> dict:
        return {
            "total_checks": self.total_checks,
            "approved": self.approved,
            "rejected": self.rejected,
            "rule_rejected": self.rule_rejected,
            "llm_rejected": self.llm_rejected,
            "approval_rate": round(self.approved / max(1, self.total_checks), 3),
            "regen_attempted": self.regen_attempted,
            "regen_approved": self.regen_approved,
            "regen_rejected": self.regen_rejected,
            "regen_success_rate": round(
                self.regen_approved / max(1, self.regen_attempted), 3
            ),
            "correct_rejections": self.correct_rejections,
            "incorrect_rejections": self.incorrect_rejections,
            "false_approvals": self.false_approvals,
            "rejection_reasons": dict(self.rejection_reasons),
        }


class CriticGuard:
    """Validates Planner intents before Executor execution.

    Hybrid approach:
    1. Rule-based checks (always run, zero latency)
    2. LLM-based check (text-only, configurable)

    Usage:
        guard = CriticGuard(use_llm=False, vlm_client=client)
        result = guard.check(intent, target, intent_history, visible_objects, memory_text)
        if not result["approved"]:
            # feed result["reason"] + result["suggestion"] back to Planner
    """

    # Intent patterns that suggest repeated/wasteful behavior
    _REPEATED_INTENT_THRESHOLD = 3

    def __init__(
        self,
        use_llm: bool = False,
        vlm_client=None,
        max_regen_attempts: int = 1,
    ):
        self.use_llm = use_llm
        self.vlm_client = vlm_client
        self.max_regen_attempts = max_regen_attempts
        self.stats = CriticStats()

        # Track which receptacle types have been opened/searched
        self._searched_receptacles: set[str] = set()
        # Track which specific object types have been interacted with and their outcomes
        self._object_outcomes: dict[str, str] = {}  # objectType -> "found" | "empty" | "failed"

    # ------------------------------------------------------------------
    # Public API
    # ------------------------------------------------------------------

    def check(
        self,
        intent: str,
        target: str,
        intent_history: list[dict],
        visible_objects: list[dict],
        memory_text: str = "",
        action_history: list[dict] | None = None,
    ) -> dict:
        """Validate a proposed intent. Returns {approved, reason, suggestion}."""
        self.stats.total_checks += 1

        # Phase 1: Rule-based checks (fast, always run)
        rule_result = self._rule_check(
            intent, target, intent_history, visible_objects, action_history
        )
        if rule_result is not None:
            self.stats.rejected += 1
            self.stats.rule_rejected += 1
            reason_key = rule_result.get("reason_key", "unknown_rule")
            self.stats.rejection_reasons[reason_key] = (
                self.stats.rejection_reasons.get(reason_key, 0) + 1
            )
            logger.info(
                "Critic REJECTED (rule) intent=%r target=%r reason=%s",
                intent, target, rule_result.get("reason", "")[:100],
            )
            return rule_result

        # Phase 2: LLM-based check (optional)
        if self.use_llm and self.vlm_client:
            llm_result = self._llm_check(
                intent, target, intent_history, visible_objects, memory_text
            )
            if llm_result is not None and not llm_result.get("approved", True):
                self.stats.rejected += 1
                self.stats.llm_rejected += 1
                reason_key = "llm_rejected"
                self.stats.rejection_reasons[reason_key] = (
                    self.stats.rejection_reasons.get(reason_key, 0) + 1
                )
                logger.info(
                    "Critic REJECTED (LLM) intent=%r target=%r reason=%s",
                    intent, target, llm_result.get("reason", "")[:100],
                )
                return llm_result

        # Approved
        self.stats.approved += 1
        logger.debug("Critic APPROVED intent=%r target=%r", intent, target)
        return {"approved": True, "reason": "ok", "suggestion": ""}

    def mark_receptacle_searched(self, receptacle_type: str):
        """Mark a receptacle type as having been opened/searched."""
        if receptacle_type:
            self._searched_receptacles.add(receptacle_type)

    def mark_object_outcome(self, object_type: str, outcome: str):
        """Track outcome of interacting with an object type.
        outcome: 'found', 'empty', 'failed'
        """
        if object_type:
            self._object_outcomes[object_type] = outcome

    def get_stats(self) -> dict:
        """Return current statistics as a dict."""
        return self.stats.to_dict()

    def reset_stats(self):
        """Reset statistics counters."""
        self.stats = CriticStats()

    def feedback_for_planner(self, reason: str, suggestion: str) -> str:
        """Build feedback text to feed to the Planner for re-generation."""
        parts = ["Your previous intent was REJECTED by the Critic."]
        if reason:
            parts.append(f"Reason: {reason}")
        if suggestion:
            parts.append(f"Suggestion: {suggestion}")
        parts.append("Propose a DIFFERENT intent that addresses this feedback.")
        return "\n".join(parts)

    # ------------------------------------------------------------------
    # Rule-based checks
    # ------------------------------------------------------------------

    def _rule_check(
        self,
        intent: str,
        target: str,
        intent_history: list[dict],
        visible_objects: list[dict],
        action_history: list[dict] | None,
    ) -> dict | None:
        """Run rule-based checks. Returns rejection dict or None if all pass."""

        # Rule 1: Exact intent+target repeated 3+ times
        rejection = self._check_repeated_intent(intent, target, intent_history)
        if rejection:
            return rejection

        # Rule 2: "approach X" where X is an already-searched receptacle
        rejection = self._check_searched_receptacle(intent, target)
        if rejection:
            return rejection

        # Rule 3: "approach X" where X is known to be empty from prior interaction
        rejection = self._check_known_empty(intent, target)
        if rejection:
            return rejection

        # Rule 4: "approach X" when X is already visible and within reach
        rejection = self._check_already_accessible(intent, target, visible_objects)
        if rejection:
            return rejection

        # All rules passed
        return None

    def _check_repeated_intent(
        self,
        intent: str,
        target: str,
        intent_history: list[dict],
    ) -> dict | None:
        """Check if the same intent+target has been tried too many times."""
        key = (intent, target)
        count = 0
        last_outcome = None
        for ih in intent_history:
            ih_key = (ih.get("intent", ""), ih.get("target", ""))
            if ih_key == key:
                count += 1
                last_outcome = "OK" if ih.get("completed") else "INCOMPLETE"

        if count >= self._REPEATED_INTENT_THRESHOLD:
            return {
                "approved": False,
                "reason": (
                    f"Intent '{intent}' targeting '{target}' has been tried "
                    f"{count} times already (last outcome: {last_outcome}). "
                    f"Repeating it again is unlikely to make progress."
                ),
                "suggestion": (
                    f"Instead of '{intent} {target}', try a different approach. "
                    f"Look at unexplored areas in spatial memory or scan the room "
                    f"for alternative locations."
                ),
                "reason_key": "repeated_intent",
            }

        return None

    def _check_searched_receptacle(
        self,
        intent: str,
        target: str,
    ) -> dict | None:
        """Check if intent is 'approach X' where X is already searched."""
        if not (intent.startswith("approach") and target):
            return None

        if target in self._searched_receptacles:
            return {
                "approved": False,
                "reason": (
                    f"Receptacle '{target}' has already been opened/searched. "
                    f"Approaching it again without new information is wasteful."
                ),
                "suggestion": (
                    f"'{target}' was already searched. Look for the target object "
                    f"in unexplored receptacles from spatial memory. If no unexplored "
                    f"receptacles remain, consider scanning the room."
                ),
                "reason_key": "searched_receptacle",
            }

        return None

    def _check_known_empty(
        self,
        intent: str,
        target: str,
    ) -> dict | None:
        """Check if intent targets an object known to be empty/failed."""
        if not target:
            return None

        outcome = self._object_outcomes.get(target)
        if outcome == "empty":
            return {
                "approved": False,
                "reason": (
                    f"Object '{target}' was previously opened and found to be empty. "
                    f"Re-approaching it will not help."
                ),
                "suggestion": (
                    f"'{target}' is known to be empty. Search other locations. "
                    f"Check spatial memory for unexplored receptacles."
                ),
                "reason_key": "known_empty",
            }

        return None

    def _check_already_accessible(
        self,
        intent: str,
        target: str,
        visible_objects: list[dict],
    ) -> dict | None:
        """Check if target is already visible and within interaction range.
        If so, 'approach' is unnecessary — should be 'pickup' or 'open' instead.
        """
        if not (intent.startswith("approach") and target):
            return None

        visible = [o for o in visible_objects if o.get("visibleBounds2D")]
        for obj in visible:
            if obj.get("objectType") == target:
                # Object is visible. Check if it's close enough for interaction (< 0.5m).
                # We don't have exact distance here, but we can flag this for the LLM critic
                # or conservatively warn. For now, let's not block — the Executor handles
                # distance. We only block if intent is clearly redundant.
                pass

        return None  # Let approach go through; Executor handles distance

    # ------------------------------------------------------------------
    # LLM-based check (text-only, no image)
    # ------------------------------------------------------------------

    def _llm_check(
        self,
        intent: str,
        target: str,
        intent_history: list[dict],
        visible_objects: list[dict],
        memory_text: str,
    ) -> dict | None:
        """Run LLM-based text-only critic check."""
        if not self.vlm_client:
            return None

        lines = []
        lines.append(f"Proposed intent: {intent}")
        if target:
            lines.append(f"Target: {target}")
        lines.append("")

        # Intent history summary
        if intent_history:
            lines.append("Intent history (what the agent has tried):")
            for ih in intent_history[-10:]:
                status = "OK" if ih.get("completed") else "INCOMPLETE"
                desc = ih.get("intent", "?")
                tgt = ih.get("target", "")
                if tgt:
                    desc += f" ({tgt})"
                lines.append(f"  {desc}: {status}")
            lines.append("")

        # Visible objects
        visible = [o for o in visible_objects if o.get("visibleBounds2D")]
        if visible:
            lines.append("Currently visible objects:")
            for o in visible[:12]:
                tags = []
                if o.get("receptacle"):
                    tags.append("receptacle")
                if o.get("openable"):
                    tags.append("open")
                tag_str = f" ({','.join(tags)})" if tags else ""
                lines.append(f"  {o['objectType']}{tag_str}")
        else:
            lines.append("(No objects currently visible)")
        lines.append("")

        # Searched receptacles
        if self._searched_receptacles:
            lines.append("Already searched receptacles:")
            for r in sorted(self._searched_receptacles):
                lines.append(f"  - {r}")
            lines.append("")

        # Spatial memory
        if memory_text:
            lines.append(memory_text)

        lines.append("\nEvaluate the proposed intent. Is it reasonable? Output JSON only.")

        prompt = "\n".join(lines)
        system = CRITIC_SYSTEM.replace("{target}", target or "the target")

        try:
            result = self.vlm_client.chat_text_json(
                system_prompt=system,
                user_text=prompt,
                required_fields=("approved", "reason", "suggestion"),
                max_tokens=512,
                temperature=0.1,
            )
            return result
        except Exception as exc:
            logger.warning("Critic LLM check failed: %s. Falling back to approve.", exc)
            return None  # On error, don't block — let the intent through
