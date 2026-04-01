"""Model router with tiers and budget enforcement for Aura Dev CLI.

Routes to optimal model based on tier (fast/balanced/max) and budget cap.
Uses existing Config model chains and routing_stats for performance data.

Ollama cloud ($20/mo) supports 3 concurrent model slots. The routing table
is designed to maximize parallelism by using at most 3 distinct models per
tier — an orchestrator, a coder, and a fast model — so all three slots
stay warm and requests don't queue behind each other.
"""

import logging
from typing import Optional, Tuple

logger = logging.getLogger(__name__)

# The 3 concurrent model slots for Ollama cloud ($20/mo plan).
# Each tier picks 3 models that cover all task categories.
# This avoids model-switching latency — all 3 stay loaded.
CONCURRENT_SLOTS = {
    "max": {
        "orchestrator": "kimi-k2.5:cloud",         # reasoning + orchestration + vision
        "coder": "minimax-m2.5:cloud",              # code gen (SWE 80.2%)
        "fast": "nemotron-3-super:cloud",           # tool dispatch + throughput
    },
    "balanced": {
        "orchestrator": "glm-5:cloud",              # orchestration + code (SWE 77.8%)
        "coder": "qwen3-coder-next:cloud",          # code gen + small edits
        "fast": "nemotron-3-super:cloud",            # tool dispatch + throughput
    },
    "fast": {
        "orchestrator": "nemotron-3-super:cloud",   # everything fast
        "coder": "qwen3-coder-next:cloud",          # code
        "fast": "nemotron-3-super:cloud",            # same (1 slot)
    },
}

# Map task categories to slot roles
_CATEGORY_TO_SLOT = {
    "orchestrator": "orchestrator",
    "code_gen": "coder",
    "small_edit": "fast",
    "reasoning": "orchestrator",
    "tool_dispatch": "fast",
    "long_context": "orchestrator",
    "vision": "orchestrator",
    "throughput": "fast",
    "frontend": "coder",
}

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
        "fast": "nemotron-3-super:cloud",       # fast cloud
    },
    "code_gen": {
        "max": "minimax-m2.5:cloud",           # SWE 80.2%, Multi-SWE 51.3%, BFCL 76.8%
        "balanced": "glm-5:cloud",              # SWE 77.8%, HumanEval 90%
        "fast": "qwen3-coder-next:cloud",       # efficient code MoE
    },
    "small_edit": {
        "max": "qwen3-coder-next:cloud",       # SWE 70.6%, 3B active — fast for edits
        "balanced": "qwen3-coder-next:cloud",
        "fast": "nemotron-3-super:cloud",       # fast cloud
    },
    "reasoning": {
        "max": "deepseek-v3.2:cloud",          # AIME 94.2%, IMO gold, MMLU-Pro 85%
        "balanced": "kimi-k2.5:cloud",          # AIME 96.1%, MMLU-Pro 86.4%
        "fast": "qwen3.5:397b-cloud",           # deep planning
    },
    "tool_dispatch": {
        "max": "nemotron-3-super:cloud",       # fast, sufficient for routing
        "balanced": "nemotron-3-super:cloud",
        "fast": "nemotron-3-super:cloud",
    },
    "long_context": {
        "max": "minimax-m2.7:cloud",            # 1M context, self-evolving
        "balanced": "nemotron-3-super:cloud",   # 1M context, 2.2x throughput
        "fast": "minimax-m2.5:cloud",           # 196K context
    },
    "vision": {
        "max": "kimi-k2.5:cloud",              # MMMU-Pro 78.5%, OCR 92.3%, MathVision 84.2%
        "balanced": "qwen3.5:cloud",            # 397B MoE, multimodal capable
        "fast": "kimi-k2.5:cloud",              # native multimodal
    },
    "throughput": {
        "max": "nemotron-3-super:cloud",       # 2.2x faster than GPT-OSS, PinchBench 85.6%
        "balanced": "gpt-oss:120b-cloud",       # 5.1B active, fast, MMLU 90%
        "fast": "nemotron-3-super:cloud",       # fast cloud
    },
    "frontend": {
        "fast": "kimi-k2.5:cloud",
        "balanced": "kimi-k2.5:cloud",
        "max": "kimi-k2.5:cloud",
    },
}

VALID_TIERS = ("fast", "balanced", "max")

# Weighted keyword patterns for task classification.
# Each entry is (keyword, weight). Higher weight = stronger signal.
TASK_KEYWORDS: dict[str, list[tuple[str, int]]] = {
    "frontend": [
        ("landing page", 3), ("dashboard", 3), ("website", 2), ("webapp", 2),
        ("web app", 2), ("pricing page", 3), ("signup page", 3), ("login page", 3),
        ("settings page", 3), ("frontend", 3), ("user interface", 2), ("ui design", 2),
        ("react component", 2), ("tailwind", 2),
    ],
    "code_gen": [
        ("implement", 2), ("create", 1), ("build", 2), ("add feature", 2),
        ("new file", 2), ("write a", 1), ("scaffold", 2),
    ],
    "small_edit": [
        ("fix", 2), ("change", 1), ("rename", 2), ("update", 1), ("modify", 1),
        ("replace", 1), ("typo", 3), ("tweak", 2),
    ],
    "reasoning": [
        ("explain", 2), ("why", 1), ("how does", 2), ("analyze", 2), ("review", 2),
        ("what is", 1), ("understand", 1), ("compare", 2),
    ],
    "long_context": [
        ("summarize this file", 3), ("entire codebase", 3), ("all files", 2),
        ("full project", 2),
    ],
    "vision": [
        ("screenshot", 3), ("image", 2), ("diagram", 2), ("visual", 2),
    ],
    "tool_dispatch": [
        ("search for", 2), ("find", 1), ("list", 1), ("show me", 1),
        ("grep", 2), ("look for", 1),
    ],
}


def classify_task(prompt: str) -> Tuple[str, float]:
    """Classify a prompt into a task category using weighted keyword scoring.

    Returns (category, confidence) where confidence is 0.0-1.0.
    Falls back to 'orchestrator' when no keywords match or confidence is too low.
    """
    prompt_lower = prompt.lower()

    scores: dict[str, int] = {}
    for category, keywords in TASK_KEYWORDS.items():
        score = sum(weight for kw, weight in keywords if kw in prompt_lower)
        if score > 0:
            scores[category] = score

    if not scores:
        return ("orchestrator", 0.0)

    best = max(scores, key=scores.get)
    total = sum(scores.values())
    confidence = scores[best] / total if total > 0 else 0.0

    # Low confidence → fall back to orchestrator
    if confidence < 0.4:
        return ("orchestrator", confidence)

    return (best, confidence)


class ModelRouter:
    """Selects models based on tier and budget constraints."""

    def __init__(self, tier: str = "balanced", budget_usd: Optional[float] = None):
        # Backward compat: "local" → "fast"
        if tier == "local":
            logger.info("[Router] Tier 'local' is deprecated, using 'fast'")
            tier = "fast"
        if tier not in VALID_TIERS:
            logger.warning(f"[Router] Unknown tier '{tier}', defaulting to 'balanced'")
            tier = "balanced"
        self.tier = tier
        self.budget_usd = budget_usd

    def select(self, task_category: str = "orchestrator") -> str:
        """Pick model based on tier for the given task category.

        Uses the 3-slot concurrent model system to avoid model switching.
        Maps task categories to slot roles (orchestrator/coder/fast),
        then looks up which model is assigned to that slot for the current tier.
        """
        # Use concurrent slot mapping for efficient 3-model utilization
        slot = _CATEGORY_TO_SLOT.get(task_category, "orchestrator")
        tier_slots = CONCURRENT_SLOTS.get(self.tier, CONCURRENT_SLOTS["balanced"])
        model = tier_slots.get(slot, tier_slots["orchestrator"])

        # Fall back to ROUTING_TABLE for categories not in slot mapping
        if not model:
            entry = ROUTING_TABLE.get(task_category, ROUTING_TABLE["orchestrator"])
            model = entry.get(self.tier, entry.get("balanced"))

        logger.debug(f"[Router] {task_category}/{self.tier} -> slot:{slot} -> {model}")
        return model

    def select_agentic(self, prompt: str = None) -> str:
        """Select the best model for the agentic loop.

        If a prompt is provided, classifies the task and routes to the
        best model for that category. Otherwise defaults to orchestrator.
        """
        if prompt:
            category, confidence = classify_task(prompt)
            logger.debug(f"[Router] Task classified as '{category}' (conf={confidence:.2f}) for: {prompt[:60]}")
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
        tier_slots = CONCURRENT_SLOTS.get(self.tier, CONCURRENT_SLOTS["balanced"])
        return {
            "tier": self.tier,
            "budget_usd": self.budget_usd,
            "concurrent_slots": tier_slots,
            "models": {
                cat: self.select(cat)
                for cat in ROUTING_TABLE.keys()
            },
        }
