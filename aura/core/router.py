"""Model router with tiers and budget enforcement for Aura Dev CLI.

Routes to optimal model based on tier (local/balanced/max) and budget cap.
Uses existing Config model chains and routing_stats for performance data.
"""

import logging
from typing import Optional

logger = logging.getLogger(__name__)

# Routing table: task_category -> {tier: model_name}
ROUTING_TABLE = {
    "orchestrator": {
        "max": "qwen3.5:397b-cloud",
        "balanced": "qwen3.5:397b-cloud",
        "local": "qwen3:8b",
    },
    "code_gen": {
        "max": "minimax-m2.5:cloud",
        "balanced": "kimi-k2.5:cloud",
        "local": "qwen2.5-coder:7b",
    },
    "small_edit": {
        "max": "qwen3-coder:480b-cloud",
        "balanced": "qwen2.5-coder:7b",
        "local": "qwen2.5-coder:7b",
    },
    "reasoning": {
        "max": "cogito-2.1:671b-cloud",
        "balanced": "kimi-k2-thinking:cloud",
        "local": "deepseek-r1:8b",
    },
    "tool_dispatch": {
        "max": "qwen3:8b",
        "balanced": "qwen3:8b",
        "local": "qwen3:8b",
    },
    "long_context": {
        "max": "gemini-3-flash-preview:cloud",
        "balanced": "nemotron-3-nano:30b-cloud",
        "local": "hermes3:8b",
    },
    "vision": {
        "max": "qwen3.5:397b-cloud",
        "balanced": "qwen3-vl:235b-cloud",
        "local": "gemma3:4b",
    },
    "throughput": {
        "max": "nemotron-3-super:cloud",
        "balanced": "qwen3-coder:480b-cloud",
        "local": "phi4-mini",
    },
}

VALID_TIERS = ("local", "balanced", "max")


class ModelRouter:
    """Selects models based on tier and budget constraints."""

    def __init__(self, tier: str = "balanced", budget_usd: Optional[float] = None):
        if tier not in VALID_TIERS:
            logger.warning(f"[Router] Unknown tier '{tier}', defaulting to 'balanced'")
            tier = "balanced"
        self.tier = tier
        self.budget_usd = budget_usd

    def select(self, task_category: str = "orchestrator") -> str:
        """Pick model based on tier for the given task category."""
        entry = ROUTING_TABLE.get(task_category)
        if not entry:
            logger.warning(f"[Router] Unknown category '{task_category}', using orchestrator")
            entry = ROUTING_TABLE["orchestrator"]

        model = entry.get(self.tier, entry.get("balanced"))
        logger.debug(f"[Router] {task_category}/{self.tier} -> {model}")
        return model

    def select_agentic(self) -> str:
        """Select the best model for the agentic loop (orchestrator role).

        For the main agentic loop, we want the best tool-calling model
        available at the current tier.
        """
        return self.select("orchestrator")

    def check_budget(self, brain) -> bool:
        """Return True if budget is OK, False if exhausted."""
        if self.budget_usd is None:
            return True
        stats = brain.get_session_stats()
        remaining = self.budget_usd - stats["cost_usd"]
        if remaining <= 0:
            logger.info(f"[Router] Budget exhausted: ${stats['cost_usd']:.4f} >= ${self.budget_usd:.2f}")
            return False
        return True

    def get_routing_info(self) -> dict:
        """Return current routing configuration for display."""
        return {
            "tier": self.tier,
            "budget_usd": self.budget_usd,
            "models": {
                cat: entry.get(self.tier, "?")
                for cat, entry in ROUTING_TABLE.items()
            },
        }
