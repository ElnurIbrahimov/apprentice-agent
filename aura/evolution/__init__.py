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

from .adapter import AuraSkillAdapter
from .engine import GEPAEngine
from .runner import run_evolution
from .types import Candidate, GEPAConfig, GEPAResult

__all__ = [
    "AuraSkillAdapter",
    "Candidate",
    "GEPAConfig",
    "GEPAEngine",
    "GEPAResult",
    "run_evolution",
]
