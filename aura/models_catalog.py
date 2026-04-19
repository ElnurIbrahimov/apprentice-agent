"""Model profile catalog — single source of truth for Aura's cloud model lineup.

Every model that Aura can dispatch to gets a `ModelProfile` with real metadata:
cost, context length, published benchmark scores, specialty tags, and a latency
tier. The router and cost tracker both derive from this — no more parallel
dictionaries drifting out of sync.

Benchmark numbers are from each model's publisher release notes as of 2026-04.
"""
from __future__ import annotations

from dataclasses import dataclass, field
from typing import Optional


# Specialty tags — used by the dispatcher to pick the right tool for the job.
SPECIALTY_REASONING   = "reasoning"     # math, planning, multi-step logic
SPECIALTY_CODE        = "code"          # SWE-bench, code gen, refactor
SPECIALTY_VISION      = "vision"        # multimodal image input
SPECIALTY_LONGCTX     = "longctx"       # very long documents
SPECIALTY_TOOL        = "tool_dispatch" # fast tool-call selection
SPECIALTY_FAST        = "fast"          # throughput / chat / simple tasks
SPECIALTY_CHEAP       = "cheap"         # cost-sensitive bulk
SPECIALTY_MATH        = "math"          # AIME / competition math


# Latency tiers used by the warm-slot policy.
LATENCY_FAST     = "fast"      # <1s first-token
LATENCY_MEDIUM   = "medium"    # 1-3s first-token
LATENCY_SLOW     = "slow"      # >3s first-token (big MoE)


@dataclass(frozen=True)
class ModelProfile:
    """Typed metadata for one model."""
    name: str                                  # Ollama identifier, e.g. "kimi-k2.5:cloud"
    provider: str                              # "ollama_cloud", "chatgpt", "anthropic"
    cost_in_per_1k: float                      # USD per 1K input tokens
    cost_out_per_1k: float                     # USD per 1K output tokens
    context_length: int                        # tokens
    specialties: tuple[str, ...]               # ordered — first is primary
    latency_tier: str = LATENCY_MEDIUM
    swe_bench: Optional[float] = None          # % on SWE-Bench Verified (0-100)
    mmlu_pro: Optional[float] = None           # %
    aime: Optional[float] = None               # %
    notes: str = ""
    active_params_b: Optional[float] = None    # active params (B) for MoE
    total_params_b: Optional[float] = None     # total params (B)


# ---------------------------------------------------------------------------
# Catalog — Ollama Cloud lineup (Pro plan: 3 concurrent models, 2 is the sweet spot)
# ---------------------------------------------------------------------------

MODELS: dict[str, ModelProfile] = {
    # ── Speed / tool dispatch / throughput ──────────────────────────────
    "nemotron-3-super:cloud": ModelProfile(
        name="nemotron-3-super:cloud",
        provider="ollama_cloud",
        cost_in_per_1k=0.0004, cost_out_per_1k=0.0004,
        context_length=1_000_000,
        specialties=(SPECIALTY_FAST, SPECIALTY_TOOL, SPECIALTY_CHEAP),
        latency_tier=LATENCY_FAST,
        active_params_b=12, total_params_b=120,
        notes="Mamba-MoE 120B/12B, 1M ctx, 449 tok/s. The workhorse.",
    ),
    "deepseek-v3.2:cloud": ModelProfile(
        name="deepseek-v3.2:cloud",
        provider="ollama_cloud",
        cost_in_per_1k=0.00014, cost_out_per_1k=0.0003,
        context_length=128_000,
        specialties=(SPECIALTY_CHEAP, SPECIALTY_FAST, SPECIALTY_REASONING),
        latency_tier=LATENCY_FAST,
        active_params_b=37, total_params_b=685,
        notes="MoE 685B/37B, cheapest cloud model, solid general-purpose.",
    ),

    # ── Reasoning / planning ────────────────────────────────────────────
    "kimi-k2.5:cloud": ModelProfile(
        name="kimi-k2.5:cloud",
        provider="ollama_cloud",
        cost_in_per_1k=0.003, cost_out_per_1k=0.003,
        context_length=256_000,
        specialties=(SPECIALTY_REASONING, SPECIALTY_VISION, SPECIALTY_LONGCTX),
        latency_tier=LATENCY_SLOW,
        mmlu_pro=86.4, swe_bench=76.8, aime=96.1,
        active_params_b=32, total_params_b=1000,
        notes="1T/32B MoE, top general-purpose. Use for hard reasoning + vision.",
    ),
    "qwen3.5:397b-cloud": ModelProfile(
        name="qwen3.5:397b-cloud",
        provider="ollama_cloud",
        cost_in_per_1k=0.004, cost_out_per_1k=0.004,
        context_length=262_000,
        specialties=(SPECIALTY_REASONING, SPECIALTY_LONGCTX),
        latency_tier=LATENCY_SLOW,
        active_params_b=17, total_params_b=397,
        notes="Hybrid thinking mode, 201 languages. Strong reasoning.",
    ),
    "qwen3.5:cloud": ModelProfile(
        name="qwen3.5:cloud",
        provider="ollama_cloud",
        cost_in_per_1k=0.0008, cost_out_per_1k=0.0008,
        context_length=262_000,
        specialties=(SPECIALTY_FAST, SPECIALTY_REASONING, SPECIALTY_VISION),
        latency_tier=LATENCY_MEDIUM,
        active_params_b=9.65,
        notes="Dense 9.65B, multimodal, 262K ctx. Cheap reasoning tier.",
    ),
    "gpt-oss:120b-cloud": ModelProfile(
        name="gpt-oss:120b-cloud",
        provider="ollama_cloud",
        cost_in_per_1k=0.003, cost_out_per_1k=0.003,
        context_length=128_000,
        specialties=(SPECIALTY_MATH, SPECIALTY_REASONING),
        latency_tier=LATENCY_MEDIUM,
        aime=97.9,
        active_params_b=5.1, total_params_b=117,
        notes="117B MoE, 97.9% AIME. Top-tier math.",
    ),

    # ── General-purpose tier between fast and reasoning ─────────────────
    "glm-5:cloud": ModelProfile(
        name="glm-5:cloud",
        provider="ollama_cloud",
        cost_in_per_1k=0.003, cost_out_per_1k=0.003,
        context_length=200_000,
        specialties=(SPECIALTY_REASONING, SPECIALTY_CODE),
        latency_tier=LATENCY_MEDIUM,
        swe_bench=77.8,
        active_params_b=40, total_params_b=744,
        notes="744B/40B MoE, lowest-hallucination model in lineup. Orchestration default.",
    ),
    "gemma4:31b-cloud": ModelProfile(
        name="gemma4:31b-cloud",
        provider="ollama_cloud",
        cost_in_per_1k=0.001, cost_out_per_1k=0.001,
        context_length=256_000,
        specialties=(SPECIALTY_FAST, SPECIALTY_VISION),
        latency_tier=LATENCY_MEDIUM,
        mmlu_pro=85.2,
        active_params_b=31,
        notes="31B dense, multimodal+audio. Mid-tier all-rounder.",
    ),

    # ── Coding specialists ──────────────────────────────────────────────
    "minimax-m2.5:cloud": ModelProfile(
        name="minimax-m2.5:cloud",
        provider="ollama_cloud",
        cost_in_per_1k=0.004, cost_out_per_1k=0.004,
        context_length=196_000,
        specialties=(SPECIALTY_CODE, SPECIALTY_REASONING),
        latency_tier=LATENCY_MEDIUM,
        swe_bench=80.2,
        active_params_b=10, total_params_b=229,
        notes="229B MoE, SWE-Bench 80.2%. The code king.",
    ),
    "minimax-m2.7:cloud": ModelProfile(
        name="minimax-m2.7:cloud",
        provider="ollama_cloud",
        cost_in_per_1k=0.004, cost_out_per_1k=0.004,
        context_length=205_000,
        specialties=(SPECIALTY_CODE, SPECIALTY_REASONING),
        latency_tier=LATENCY_MEDIUM,
        swe_bench=56.2,
        active_params_b=10, total_params_b=230,
        notes="Self-evolving successor. SWE-Pro 56.2%.",
    ),
    "glm-5.1:cloud": ModelProfile(
        name="glm-5.1:cloud",
        provider="ollama_cloud",
        cost_in_per_1k=0.003, cost_out_per_1k=0.003,
        context_length=200_000,
        specialties=(SPECIALTY_CODE, SPECIALTY_REASONING),
        latency_tier=LATENCY_MEDIUM,
        swe_bench=77.8,
        active_params_b=40, total_params_b=744,
        notes="GLM-5 + 28% coding. Balanced code default.",
    ),
    "qwen3-coder:480b-cloud": ModelProfile(
        name="qwen3-coder:480b-cloud",
        provider="ollama_cloud",
        cost_in_per_1k=0.004, cost_out_per_1k=0.004,
        context_length=256_000,
        specialties=(SPECIALTY_CODE, SPECIALTY_LONGCTX),
        latency_tier=LATENCY_SLOW,
        swe_bench=69.6,
        active_params_b=35, total_params_b=480,
        notes="Big refactors, long-context code surgery.",
    ),
    "qwen3-coder-next:cloud": ModelProfile(
        name="qwen3-coder-next:cloud",
        provider="ollama_cloud",
        cost_in_per_1k=0.003, cost_out_per_1k=0.003,
        context_length=256_000,
        specialties=(SPECIALTY_CODE, SPECIALTY_FAST),
        latency_tier=LATENCY_FAST,
        swe_bench=71.3,
        active_params_b=3, total_params_b=80,
        notes="Efficient code MoE, 172 tok/s. Fast code tier.",
    ),
}


# Public helpers ------------------------------------------------------------

def get(name: str) -> Optional[ModelProfile]:
    """Return a model's profile, or None if unknown."""
    return MODELS.get(name)


def cost_per_1k(name: str) -> tuple[float, float]:
    """(input_rate, output_rate) per 1K tokens. Falls back to sensible default."""
    p = MODELS.get(name)
    if p is not None:
        return (p.cost_in_per_1k, p.cost_out_per_1k)
    return (0.003, 0.003)  # generic cloud default


def models_by_specialty(specialty: str) -> list[ModelProfile]:
    """All models whose specialties include the given tag, ordered best-first.

    Ordering: models that list `specialty` first in their tuple come first,
    then models that include it later. Within a group, prefer the one with the
    best benchmark for the specialty when one applies.
    """
    primary, secondary = [], []
    for p in MODELS.values():
        if not p.specialties:
            continue
        if p.specialties[0] == specialty:
            primary.append(p)
        elif specialty in p.specialties:
            secondary.append(p)

    def _score(p: ModelProfile) -> float:
        if specialty == SPECIALTY_CODE and p.swe_bench is not None:
            return -p.swe_bench  # desc
        if specialty == SPECIALTY_REASONING and p.mmlu_pro is not None:
            return -p.mmlu_pro
        if specialty == SPECIALTY_MATH and p.aime is not None:
            return -p.aime
        if specialty == SPECIALTY_CHEAP:
            return p.cost_in_per_1k
        return 0.0

    primary.sort(key=_score)
    secondary.sort(key=_score)
    return primary + secondary


def cheapest(min_specialty: Optional[str] = None) -> ModelProfile:
    """Return the cheapest model, optionally filtered by specialty."""
    pool = list(MODELS.values())
    if min_specialty:
        pool = [p for p in pool if min_specialty in p.specialties]
    if not pool:
        pool = list(MODELS.values())
    return min(pool, key=lambda p: p.cost_in_per_1k)


def all_cloud_names() -> set[str]:
    return {name for name, p in MODELS.items() if p.provider == "ollama_cloud"}


# Ollama Pro concurrency constraint — 3 hard cap, 2 sweet spot, 1 baseline.
OLLAMA_PRO_MAX_CONCURRENT = 3
OLLAMA_PRO_SWEET_SPOT    = 2


__all__ = [
    "ModelProfile",
    "MODELS",
    "SPECIALTY_REASONING", "SPECIALTY_CODE", "SPECIALTY_VISION",
    "SPECIALTY_LONGCTX", "SPECIALTY_TOOL", "SPECIALTY_FAST",
    "SPECIALTY_CHEAP", "SPECIALTY_MATH",
    "LATENCY_FAST", "LATENCY_MEDIUM", "LATENCY_SLOW",
    "get", "cost_per_1k", "models_by_specialty", "cheapest",
    "all_cloud_names",
    "OLLAMA_PRO_MAX_CONCURRENT", "OLLAMA_PRO_SWEET_SPOT",
]
