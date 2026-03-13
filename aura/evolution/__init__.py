"""
GEPA Evolution — Self-improving skill evolution for Aura.

Based on GEPA (Genetic-Pareto Prompt Evolution), adapted for Aura's
skill library and multi-agent architecture.

Usage:
    # From Python
    from aura.evolution.runner import run_evolution
    result = run_evolution()

    # From CLI
    python -m aura.evolution.runner
    python -m aura.evolution.runner --skill skill_abc --iterations 5
    python -m aura.evolution.runner --dry-run
"""

from .types import GEPAConfig, GEPAResult, Candidate
from .engine import GEPAEngine
from .adapter import AuraSkillAdapter
from .runner import run_evolution

__all__ = [
    "GEPAConfig",
    "GEPAResult",
    "GEPAEngine",
    "AuraSkillAdapter",
    "Candidate",
    "run_evolution",
]
