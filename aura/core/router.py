"""Model router with tiers and budget enforcement for Aura Dev CLI.

Routes to optimal model based on tier (fast/balanced/max) and budget cap.
Uses existing Config model chains and routing_stats for performance data.
Learns from outcomes: tracks success/failure per (category, model) pair
and consults the strategy bandit for category-level performance signals.

Ollama cloud ($20/mo) supports 3 concurrent model slots. The routing table
is designed to maximize parallelism by using at most 3 distinct models per
tier — an orchestrator, a coder, and a fast model — so all three slots
stay warm and requests don't queue behind each other.
"""

import logging
import threading
from collections import defaultdict
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
        "coder": "glm-5.1:cloud",                   # code gen + SWE 77.8%, +28% coding
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
#   glm-5, glm-5.1, nemotron-3-super
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
        "balanced": "glm-5.1:cloud",            # SWE 77.8%, +28% coding over GLM-5
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

# ── Embedding-based task classification ──
# Exemplar phrases per category, used to compute centroid embeddings.
_CATEGORY_EXEMPLARS: dict[str, list[str]] = {
    "code_gen": [
        "implement a new feature", "create a REST API endpoint", "build a function that",
        "scaffold the project", "write a class for", "generate code to handle",
    ],
    "small_edit": [
        "fix the typo", "rename the variable", "change the color to",
        "update the import", "modify the return value", "swap the order of",
    ],
    "reasoning": [
        "explain how this works", "why does this fail", "analyze the performance",
        "compare these approaches", "what would happen if", "review the architecture",
    ],
    "frontend": [
        "build a landing page", "style the component", "add a button that",
        "create a form with", "fix the CSS layout", "responsive design for",
    ],
    "long_context": [
        "summarize this document", "read the entire file", "analyze all the logs",
        "review the full codebase", "compare these two large files",
    ],
    "tool_dispatch": [
        "run the tests", "search for files", "find the definition",
        "execute the script", "check git status", "list the directory",
    ],
    "throughput": [
        "process all files", "batch update", "migrate the database",
        "refactor across the codebase", "update all imports",
    ],
}

# Cache for computed centroids
_category_centroids: dict[str, list[float]] = {}
_centroids_computed: bool = False
_centroids_lock = threading.Lock()


def _ensure_centroids() -> None:
    """Lazily compute category centroids from exemplar embeddings."""
    global _category_centroids, _centroids_computed
    if _centroids_computed:
        return
    with _centroids_lock:
        if _centroids_computed:
            return
        try:
            import numpy as np
            import ollama

            for cat, exemplars in _CATEGORY_EXEMPLARS.items():
                embeddings = []
                for ex in exemplars:
                    try:
                        resp = ollama.embed(model="nomic-embed-text:latest", input=ex)
                        if resp and "embeddings" in resp and resp["embeddings"]:
                            embeddings.append(resp["embeddings"][0])
                    except Exception:
                        continue
                if embeddings:
                    _category_centroids[cat] = np.mean(embeddings, axis=0).tolist()
        except ImportError:
            pass  # numpy or ollama not available
        except Exception:
            pass  # Ollama not running, etc.
        finally:
            _centroids_computed = True


def classify_task_embedding(prompt: str) -> "tuple[str, float] | None":
    """Classify a prompt using embedding cosine similarity.

    Returns (category, confidence) or None if embeddings unavailable.
    """
    _ensure_centroids()
    if not _category_centroids:
        return None

    try:
        import numpy as np
        import ollama

        resp = ollama.embed(model="nomic-embed-text:latest", input=prompt)
        if not resp or "embeddings" not in resp or not resp["embeddings"]:
            return None

        prompt_vec = np.array(resp["embeddings"][0])

        best_cat, best_sim = None, -1.0
        for cat, centroid in _category_centroids.items():
            centroid_vec = np.array(centroid)
            # Cosine similarity
            dot = float(np.dot(prompt_vec, centroid_vec))
            norm = float(np.linalg.norm(prompt_vec) * np.linalg.norm(centroid_vec))
            sim = dot / (norm + 1e-8)
            if sim > best_sim:
                best_sim = sim
                best_cat = cat

        if best_cat and best_sim > 0.5:
            return (best_cat, best_sim)
    except ImportError:
        pass  # numpy or ollama not available
    except Exception:
        pass  # Embedding call failed

    return None


# ── Strategy bandit integration (optional) ──
# Maps bandit ProblemCategory values → router task categories
_BANDIT_TO_ROUTER_CATEGORY = {
    "math": "reasoning",
    "code": "code_gen",
    "analysis": "reasoning",
    "creative": "orchestrator",
    "planning": "orchestrator",
    "debug": "small_edit",
}

# Maps router categories → bandit ProblemCategory values (reverse)
_ROUTER_TO_BANDIT_CATEGORY = {
    "code_gen": "code",
    "small_edit": "debug",
    "reasoning": "math",       # closest match
    "orchestrator": "planning",
    "frontend": "code",
    "tool_dispatch": None,     # no bandit equivalent
    "long_context": None,
    "vision": None,
    "throughput": None,
}


def classify_task(prompt: str) -> Tuple[str, float]:
    """Classify a prompt into a task category.

    Tries embedding-based classification first (more accurate),
    falls back to keyword matching if embeddings unavailable.

    Returns (category, confidence) where confidence is 0.0-1.0.
    Falls back to 'orchestrator' when no keywords match or confidence is too low.
    """
    # Try embedding-based classification (more accurate)
    emb_result = classify_task_embedding(prompt)
    if emb_result and emb_result[1] > 0.6:
        return emb_result

    # Fall back to keyword-based classification
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
    """Selects models based on tier and budget constraints.

    Learns from outcomes: tracks success/failure per (category, model)
    and consults the strategy bandit for category-level performance data.
    """

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

        # Outcome tracking: {(category, model): {"successes": int, "failures": int, "total_iters": int, "count": int}}
        self._outcome_stats: dict[tuple, dict] = defaultdict(
            lambda: {"successes": 0, "failures": 0, "total_iters": 0, "count": 0}
        )
        self._stats_lock = threading.Lock()

        # Lazy bandit reference (loaded on first use)
        self._bandit = None
        self._bandit_loaded = False

    def _get_bandit(self):
        """Lazy-load the strategy bandit singleton. Returns None if unavailable."""
        if not self._bandit_loaded:
            self._bandit_loaded = True
            try:
                from aura.consciousness.strategy_bandit import get_strategy_bandit
                self._bandit = get_strategy_bandit()
                logger.debug("[Router] Strategy bandit connected")
            except Exception as e:
                logger.debug(f"[Router] Strategy bandit unavailable: {e}")
                self._bandit = None
        return self._bandit

    def record_outcome(
        self,
        task_category: str,
        model_used: str,
        success: bool,
        iterations: int = 1,
    ):
        """Record a routing outcome for learning.

        Args:
            task_category: The task category that was routed.
            model_used: The model that handled the task.
            success: Whether the task completed successfully.
            iterations: Number of agentic loop iterations used.
        """
        key = (task_category, model_used)
        with self._stats_lock:
            stats = self._outcome_stats[key]
            if success:
                stats["successes"] += 1
            else:
                stats["failures"] += 1
            stats["total_iters"] += iterations
            stats["count"] += 1

        logger.debug(
            f"[Router] Outcome: {task_category}/{model_used} "
            f"success={success} iters={iterations} "
            f"(total: {stats['successes']}S/{stats['failures']}F)"
        )

    def _should_override_model(self, task_category: str, model: str) -> bool:
        """Check if a model has consistently failed for a category.

        Returns True if the model has >3 failures and 0 successes,
        meaning the caller should use the fallback instead.
        """
        key = (task_category, model)
        with self._stats_lock:
            stats = self._outcome_stats.get(key)
            if not stats:
                return False
            return stats["failures"] > 3 and stats["successes"] == 0

    def _get_fallback_model(self, task_category: str, failed_model: str) -> Optional[str]:
        """Get the next model in the tier chain for a category.

        Tries the next tier up (fast -> balanced -> max) to find a
        different model for the same category.
        """
        tier_order = ["fast", "balanced", "max"]
        current_idx = tier_order.index(self.tier) if self.tier in tier_order else 1

        # Try higher tiers first, then lower
        for offset in [1, 2, -1]:
            try_idx = current_idx + offset
            if 0 <= try_idx < len(tier_order):
                try_tier = tier_order[try_idx]
                entry = ROUTING_TABLE.get(task_category, ROUTING_TABLE.get("orchestrator", {}))
                candidate = entry.get(try_tier)
                if candidate and candidate != failed_model:
                    return candidate

        return None

    def _bandit_tier_hint(self, task_category: str) -> Optional[str]:
        """Consult the strategy bandit for a tier hint.

        If the bandit shows high reward for MCTS in a category that maps
        to this router category, suggest the "max" tier (reasoning model).
        If it shows chain_of_thought dominates, suggest "fast" tier.

        Returns None if no strong signal.
        """
        bandit = self._get_bandit()
        if not bandit:
            return None

        bandit_cat = _ROUTER_TO_BANDIT_CATEGORY.get(task_category)
        if not bandit_cat:
            return None

        try:
            arm_stats = bandit.get_arm_stats()
            cat_arms = arm_stats.get(bandit_cat, [])
            if not cat_arms:
                return None

            # Only act on data with sufficient pulls
            total_pulls = sum(a.get("total_pulls", 0) for a in cat_arms)
            if total_pulls < 5:
                return None

            # Find the dominant strategy
            best_arm = max(cat_arms, key=lambda a: a.get("mean_reward", 0.5))
            best_strategy = best_arm.get("strategy", "")
            mean_reward = best_arm.get("mean_reward", 0.5)

            # Strong signal threshold
            if mean_reward < 0.6:
                return None

            # MCTS dominates → complex task, use max tier
            if best_strategy == "mcts" and mean_reward > 0.65:
                logger.debug(
                    f"[Router] Bandit hint: {bandit_cat} favors MCTS "
                    f"(reward={mean_reward:.3f}) → suggesting 'max' tier"
                )
                return "max"

            # Chain-of-thought dominates with high reward → simpler task, fast is fine
            if best_strategy == "chain_of_thought" and mean_reward > 0.75:
                logger.debug(
                    f"[Router] Bandit hint: {bandit_cat} favors CoT "
                    f"(reward={mean_reward:.3f}) → suggesting 'fast' tier"
                )
                return "fast"

        except Exception as e:
            logger.debug(f"[Router] Bandit hint failed: {e}")

        return None

    def select(self, task_category: str = "orchestrator") -> str:
        """Pick model based on tier for the given task category.

        Uses the 3-slot concurrent model system to avoid model switching.
        Maps task categories to slot roles (orchestrator/coder/fast),
        then looks up which model is assigned to that slot for the current tier.

        Enhanced with outcome learning and bandit-informed tier hints.
        """
        # Step 1: Check if the bandit suggests a different tier for this category
        effective_tier = self.tier
        bandit_hint = self._bandit_tier_hint(task_category)
        if bandit_hint and bandit_hint != self.tier:
            effective_tier = bandit_hint
            logger.debug(f"[Router] Bandit override: {task_category} tier {self.tier} -> {effective_tier}")

        # Step 2: Standard slot-based model selection
        slot = _CATEGORY_TO_SLOT.get(task_category, "orchestrator")
        tier_slots = CONCURRENT_SLOTS.get(effective_tier, CONCURRENT_SLOTS["balanced"])
        model = tier_slots.get(slot, tier_slots["orchestrator"])

        # Fall back to ROUTING_TABLE for categories not in slot mapping
        if not model:
            entry = ROUTING_TABLE.get(task_category, ROUTING_TABLE["orchestrator"])
            model = entry.get(effective_tier, entry.get("balanced"))

        # Step 3: Check outcome history — override if model consistently fails
        if self._should_override_model(task_category, model):
            fallback = self._get_fallback_model(task_category, model)
            if fallback:
                logger.info(
                    f"[Router] Overriding {model} for {task_category} "
                    f"(consistent failures) -> {fallback}"
                )
                model = fallback

        logger.debug(f"[Router] {task_category}/{effective_tier} -> slot:{slot} -> {model}")
        return model

    def select_agentic(self, prompt: str | None = None) -> str:
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
        # Include outcome stats summary
        with self._stats_lock:
            overrides = {
                f"{cat}/{model}": f"{s['successes']}S/{s['failures']}F"
                for (cat, model), s in self._outcome_stats.items()
                if s["count"] > 0
            }
        return {
            "tier": self.tier,
            "budget_usd": self.budget_usd,
            "concurrent_slots": tier_slots,
            "models": {
                cat: self.select(cat)
                for cat in ROUTING_TABLE.keys()
            },
            "outcome_stats": overrides,
            "bandit_connected": self._get_bandit() is not None,
        }
