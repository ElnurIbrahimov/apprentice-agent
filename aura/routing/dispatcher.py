"""Real dispatcher — turns the model lineup into actual specialty dispatching.

Philosophy:
- Classify the prompt into a task class (code, reasoning, math, vision, simple,
  longctx, tool).
- Pick the best model by catalog specialty for that class, respecting the
  current cost budget and warm-slot preference.
- Cheap tasks go cheap (deepseek-v3.2 / nemotron-3-super). Hard tasks go deep
  (kimi-k2.6 / qwen3.5:397b / minimax-m2.5). No more "nemotron handles 80% of
  traffic because the router has no data."

This sits alongside the neural Router (`aura/routing/router.py`). The neural
router scores on continuous feature vectors; this dispatcher scores on a coarse
task class. They converge most of the time; the dispatcher is faster and the
default choice for speed-sensitive paths.
"""
from __future__ import annotations

import logging
import re
import time
from dataclasses import dataclass, field
from typing import Optional

from aura import models_catalog as catalog
from aura.models_catalog import (
    SPECIALTY_CHEAP,
    SPECIALTY_CODE,
    SPECIALTY_FAST,
    SPECIALTY_LONGCTX,
    SPECIALTY_MATH,
    SPECIALTY_REASONING,
    SPECIALTY_TOOL,
    SPECIALTY_VISION,
    ModelProfile,
)

logger = logging.getLogger(__name__)


# ---------------------------------------------------------------------------
# Task classification
# ---------------------------------------------------------------------------

# Task class → primary specialty pick
TASK_CODE     = "code"
TASK_REASON   = "reason"
TASK_MATH     = "math"
TASK_VISION   = "vision"
TASK_LONGCTX  = "longctx"
TASK_TOOL     = "tool"
TASK_SIMPLE   = "simple"
TASK_GENERAL  = "general"

_CODE_PATTERNS = (
    r"\b(fix|refactor|implement|debug|edit|write|add|remove)\b.*\b(function|method|class|file|module|bug)\b",
    r"\b(pytest|npm|cargo|build)\b",
    r"```[a-z]*\n",
    r"\.py\b|\.ts\b|\.rs\b|\.go\b|\.java\b|\.cpp\b|\.js\b",
)
_REASON_PATTERNS = (
    r"\b(why|explain|reason|compare|tradeoff|plan|design|architect)\b",
    r"\b(prove|derive|analyze)\b",
)
_MATH_PATTERNS = (
    r"\b(solve|compute|calculate|integral|derivative|equation|theorem)\b",
    r"\d+\s*[+\-*/^=<>]\s*\d+",
)
_VISION_PATTERNS = (
    r"\[image:",
    r"\b(screenshot|png|jpg|jpeg|diagram|chart)\b",
)
_SIMPLE_PATTERNS = (
    r"^\s*(hi|hello|hey|thanks|thank you|ok|cool|nice|yes|no)\b",
    r"^\s*what('?s| is) \d+\s*[+\-*/]\s*\d+",
)


def classify_task(prompt: str, *, has_attachment: bool = False,
                  prompt_length: Optional[int] = None) -> str:
    """Classify a prompt into one of TASK_* constants.

    Cheap — pure regex + length heuristics. The neural router handles finer
    cases; this is the fast first-pass.
    """
    if not prompt:
        return TASK_SIMPLE

    text = prompt.strip()
    length = prompt_length if prompt_length is not None else len(text)

    if has_attachment:
        return TASK_VISION

    # Simple/greeting short-circuit
    if length < 40:
        for pat in _SIMPLE_PATTERNS:
            if re.search(pat, text, re.IGNORECASE):
                return TASK_SIMPLE

    for pat in _VISION_PATTERNS:
        if re.search(pat, text, re.IGNORECASE):
            return TASK_VISION

    for pat in _MATH_PATTERNS:
        if re.search(pat, text, re.IGNORECASE):
            return TASK_MATH

    for pat in _CODE_PATTERNS:
        if re.search(pat, text, re.IGNORECASE):
            return TASK_CODE

    for pat in _REASON_PATTERNS:
        if re.search(pat, text, re.IGNORECASE):
            return TASK_REASON

    # Long prompts likely want reasoning or longctx
    if length > 2500:
        return TASK_LONGCTX
    if length > 600:
        return TASK_REASON

    return TASK_GENERAL


# ---------------------------------------------------------------------------
# Budget tier
# ---------------------------------------------------------------------------

BUDGET_FREE      = "free"       # user is cost-blind — pick best-for-task
BUDGET_BALANCED  = "balanced"   # default — cost-aware but quality-first
BUDGET_CHEAP     = "cheap"      # cost-first


def _resolve_budget(preference: Optional[str]) -> str:
    if preference in (BUDGET_FREE, BUDGET_BALANCED, BUDGET_CHEAP):
        return preference
    return BUDGET_BALANCED


# ---------------------------------------------------------------------------
# Dispatch result
# ---------------------------------------------------------------------------

@dataclass
class DispatchResult:
    """Output of `dispatch()`. Cheap to construct."""
    task_class: str
    model: str
    fallback_models: list[str] = field(default_factory=list)
    reason: str = ""
    confidence: float = 0.7  # 0-1; used by shadow-mode auto-trigger
    latency_ms: float = 0.0

    def to_log(self) -> dict:
        return {
            "task_class": self.task_class,
            "model": self.model,
            "fallbacks": self.fallback_models[:3],
            "reason": self.reason,
            "confidence": round(self.confidence, 3),
            "latency_ms": round(self.latency_ms, 2),
        }


# ---------------------------------------------------------------------------
# Task-class → primary specialty mapping
# ---------------------------------------------------------------------------

_TASK_TO_SPECIALTY = {
    TASK_CODE:    SPECIALTY_CODE,
    TASK_REASON:  SPECIALTY_REASONING,
    TASK_MATH:    SPECIALTY_MATH,
    TASK_VISION:  SPECIALTY_VISION,
    TASK_LONGCTX: SPECIALTY_LONGCTX,
    TASK_TOOL:    SPECIALTY_TOOL,
    TASK_SIMPLE:  SPECIALTY_CHEAP,
    TASK_GENERAL: SPECIALTY_FAST,
}


# ---------------------------------------------------------------------------
# Dispatch
# ---------------------------------------------------------------------------

def dispatch(
    prompt: str,
    *,
    has_attachment: bool = False,
    budget: Optional[str] = None,
    warm_models: Optional[list[str]] = None,
    override_task_class: Optional[str] = None,
) -> DispatchResult:
    """Pick the best model for this prompt. Returns a `DispatchResult`.

    Args:
        prompt: User message text
        has_attachment: True if an image is attached
        budget: "free" / "balanced" / "cheap" — controls cost sensitivity
        warm_models: currently-warm models; prefer these when scores tie
        override_task_class: bypass classification (for testing / explicit routing)
    """
    t0 = time.perf_counter()

    task = override_task_class or classify_task(prompt, has_attachment=has_attachment)
    specialty = _TASK_TO_SPECIALTY[task]
    budget_tier = _resolve_budget(budget)
    warm = set(warm_models or [])

    candidates = catalog.models_by_specialty(specialty)

    if not candidates:
        # Shouldn't happen if catalog is correct, but degrade gracefully
        candidates = list(catalog.MODELS.values())

    # Cost filter
    if budget_tier == BUDGET_CHEAP:
        # keep only the cheapest third
        sorted_by_cost = sorted(candidates, key=lambda p: p.cost_in_per_1k)
        k = max(1, len(sorted_by_cost) // 3)
        candidates = sorted_by_cost[:k]
    elif budget_tier == BUDGET_BALANCED:
        # For quality-sensitive tasks keep the full specialty ranking. The
        # $0.001/1K gap between tiers is worth +2-5% benchmark on hard work.
        # Only trim the expensive tail on genuinely simple / general / tool
        # dispatch, where quality is saturated well before the top model.
        if task in (TASK_SIMPLE, TASK_GENERAL, TASK_TOOL):
            sorted_by_cost = sorted(candidates, key=lambda p: p.cost_in_per_1k, reverse=True)
            if len(sorted_by_cost) > 2:
                most_expensive = sorted_by_cost[0]
                candidates = [c for c in candidates if c.name != most_expensive.name]

    # Warm-slot preference: if a warm model is in the top-3 candidates, promote it
    top3 = candidates[:3]
    primary = None
    warm_hit = None
    for c in top3:
        if c.name in warm:
            warm_hit = c
            break
    primary = warm_hit or candidates[0]

    # Confidence: high when top candidate has a benchmark score for the specialty
    # and is clearly ahead of #2; lower when scores are similar / missing.
    confidence = _compute_confidence(candidates, specialty)

    # Fallbacks: next 3 candidates, skipping the primary
    fallbacks = [c.name for c in candidates if c.name != primary.name][:3]

    reason_bits = [f"task={task}", f"spec={specialty}", f"budget={budget_tier}"]
    if warm_hit:
        reason_bits.append("warm_hit")

    elapsed_ms = (time.perf_counter() - t0) * 1000.0

    return DispatchResult(
        task_class=task,
        model=primary.name,
        fallback_models=fallbacks,
        reason=" ".join(reason_bits),
        confidence=confidence,
        latency_ms=elapsed_ms,
    )


def _compute_confidence(candidates: list[ModelProfile], specialty: str) -> float:
    """Return a confidence score in [0, 1] for the top candidate.

    High (~0.9) when the top model has a benchmark for this specialty AND
    is clearly ahead of #2. Low (~0.55) when scores are similar or missing.
    """
    if not candidates:
        return 0.5
    if len(candidates) == 1:
        return 0.8

    top = candidates[0]
    runner = candidates[1]

    # Look up the relevant benchmark by specialty
    score_attr = {
        SPECIALTY_CODE:      "swe_bench",
        SPECIALTY_REASONING: "mmlu_pro",
        SPECIALTY_MATH:      "aime",
    }.get(specialty)

    if not score_attr:
        # Confidence comes from specialty-primary tag strength
        if top.specialties and top.specialties[0] == specialty:
            return 0.8
        return 0.65

    top_score = getattr(top, score_attr, None)
    runner_score = getattr(runner, score_attr, None)

    if top_score is None:
        return 0.6  # no benchmark data
    if runner_score is None:
        return 0.85

    delta = top_score - runner_score
    # Benchmarks are 0-100; delta of 5+ points = clear lead
    if delta >= 5:
        return 0.92
    if delta >= 2:
        return 0.8
    return 0.65  # effectively tied → good candidate for shadow mode


# ---------------------------------------------------------------------------
# Shadow-mode trigger heuristic
# ---------------------------------------------------------------------------

SHADOW_CONFIDENCE_THRESHOLD = 0.7


def should_shadow(result: DispatchResult, *, auto_shadow: bool = True) -> bool:
    """Decide whether to run shadow mode (2-parallel) on this dispatch.

    - auto_shadow=False → never auto-trigger (user must request via /shadow)
    - Low confidence (<0.7) → shadow on
    - Reasoning/code tasks with close benchmark margins → shadow on
    - Simple/greeting tasks → always no
    """
    if not auto_shadow:
        return False
    if result.task_class == TASK_SIMPLE:
        return False
    if result.confidence < SHADOW_CONFIDENCE_THRESHOLD:
        return True
    return False


__all__ = [
    "BUDGET_BALANCED",
    "BUDGET_CHEAP",
    "BUDGET_FREE",
    "SHADOW_CONFIDENCE_THRESHOLD",
    "TASK_CODE",
    "TASK_GENERAL",
    "TASK_LONGCTX",
    "TASK_MATH",
    "TASK_REASON",
    "TASK_SIMPLE",
    "TASK_TOOL",
    "TASK_VISION",
    "DispatchResult",
    "classify_task",
    "dispatch",
    "should_shadow",
]
