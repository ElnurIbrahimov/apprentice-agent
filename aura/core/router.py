"""Model router with tiers and budget enforcement for Aura Dev CLI.

Routes to optimal model based on tier (local/balanced/max) and budget cap.
Uses existing Config model chains and routing_stats for performance data.
"""

import logging
from typing import Optional

logger = logging.getLogger(__name__)

# Routing table: task_category -> {tier: model_name}
# Updated 2026-03-13 — based on verified benchmarks (SWE-bench, AIME, MMLU-Pro)
#
# Models available (cloud): minimax-m2.7, minimax-m2.5, kimi-k2.5, qwen3.5:397b,
#   qwen3.5, deepseek-v3.2, qwen3-coder:480b, qwen3-coder-next, gpt-oss:120b,
#   glm-5, nemotron-3-super
# Utility (local): nomic-embed-text, glm-ocr
#
ROUTING_TABLE = {
    "orchestrator": {
        "max": "kimi-k2.5:cloud",              # MMLU-Pro 86.4%, SWE 76.8%, AIME 96.1%
        "balanced": "glm-5:cloud",              # SWE 77.8%, hallucination 34% (lowest)
        "local": "nemotron-3-super:cloud",      # fast cloud fallback
    },
    "code_gen": {
        "max": "minimax-m2.5:cloud",           # SWE 80.2%, Multi-SWE 51.3%, BFCL 76.8%
        "balanced": "glm-5:cloud",              # SWE 77.8%, HumanEval 90%
        "local": "qwen3-coder-next:cloud",      # efficient code MoE
    },
    "small_edit": {
        "max": "qwen3-coder-next:cloud",       # SWE 70.6%, 3B active — fast for edits
        "balanced": "qwen3-coder-next:cloud",
        "local": "nemotron-3-super:cloud",      # fast cloud fallback
    },
    "reasoning": {
        "max": "deepseek-v3.2:cloud",          # AIME 94.2%, IMO gold, MMLU-Pro 85%
        "balanced": "kimi-k2.5:cloud",          # AIME 96.1%, MMLU-Pro 86.4%
        "local": "qwen3.5:397b-cloud",          # deep planning fallback
    },
    "tool_dispatch": {
        "max": "nemotron-3-super:cloud",       # fast, sufficient for routing
        "balanced": "nemotron-3-super:cloud",
        "local": "nemotron-3-super:cloud",
    },
    "long_context": {
        "max": "minimax-m2.7:cloud",            # 1M context, self-evolving
        "balanced": "nemotron-3-super:cloud",   # 1M context, 2.2x throughput
        "local": "minimax-m2.5:cloud",          # 196K context
    },
    "vision": {
        "max": "kimi-k2.5:cloud",              # MMMU-Pro 78.5%, OCR 92.3%, MathVision 84.2%
        "balanced": "qwen3.5:cloud",            # 397B MoE, multimodal capable
        "local": "kimi-k2.5:cloud",             # native multimodal
    },
    "throughput": {
        "max": "nemotron-3-super:cloud",       # 2.2x faster than GPT-OSS, PinchBench 85.6%
        "balanced": "gpt-oss:120b-cloud",       # 5.1B active, fast, MMLU 90%
        "local": "nemotron-3-super:cloud",      # fast cloud fallback
    },
    "frontend": {
        "local": "kimi-k2.5:cloud",
        "balanced": "kimi-k2.5:cloud",
        "max": "kimi-k2.5:cloud",
    },
}

VALID_TIERS = ("local", "balanced", "max")

# Keyword patterns for task classification
TASK_KEYWORDS = {
    "frontend": {"landing page", "dashboard", "website", "webapp", "web app",
                 "pricing page", "signup page", "login page", "settings page",
                 "frontend", "user interface", "ui design", "react component",
                 "tailwind"},
    "code_gen": {"implement", "create", "build", "add feature", "new file", "write a", "scaffold"},
    "small_edit": {"fix", "change", "rename", "update", "modify", "replace", "typo", "tweak"},
    "reasoning": {"explain", "why", "how does", "analyze", "review", "what is", "understand", "compare"},
    "long_context": {"summarize this file", "entire codebase", "all files", "full project"},
    "vision": {"screenshot", "image", "diagram", "visual"},
    "tool_dispatch": {"search for", "find", "list", "show me", "grep", "look for"},
}


def classify_task(prompt: str) -> str:
    """Classify a prompt into a task category using keyword matching.

    Frontend keywords are checked first so that prompts like
    "build a landing page" route to "frontend" instead of "code_gen".
    """
    prompt_lower = prompt.lower()

    # Priority check: frontend keywords win over generic code_gen matches
    frontend_kws = TASK_KEYWORDS.get("frontend", set())
    if any(kw in prompt_lower for kw in frontend_kws):
        return "frontend"

    scores = {}
    for category, keywords in TASK_KEYWORDS.items():
        if category == "frontend":
            continue  # already handled above
        score = sum(1 for kw in keywords if kw in prompt_lower)
        if score > 0:
            scores[category] = score

    if not scores:
        return "orchestrator"

    return max(scores, key=scores.get)


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

    def select_agentic(self, prompt: str = None) -> str:
        """Select the best model for the agentic loop.

        If a prompt is provided, classifies the task and routes to the
        best model for that category. Otherwise defaults to orchestrator.
        """
        if prompt:
            category = classify_task(prompt)
            logger.debug(f"[Router] Task classified as '{category}' for: {prompt[:60]}")
            return self.select(category)
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
