"""Shared path constants for Aura.

Anchored on the package location — not the current working directory — so data
writes land in the same place regardless of where Aura is invoked from.
"""

from pathlib import Path

AURA_ROOT: Path = Path(__file__).resolve().parent.parent
AURA_DATA_DIR: Path = AURA_ROOT / "aura_data"
SKILL_LIBRARY_DIR: Path = AURA_DATA_DIR / "skill_library"
EVOLUTION_RUNS_DIR: Path = AURA_DATA_DIR / "evolution_runs"
KNOWLEDGE_GRAPH_DIR: Path = AURA_DATA_DIR / "knowledge_graph"


def ensure_data_dirs() -> None:
    """Create core data directories if they don't exist. Safe to call repeatedly."""
    for d in (AURA_DATA_DIR, SKILL_LIBRARY_DIR, EVOLUTION_RUNS_DIR, KNOWLEDGE_GRAPH_DIR):
        d.mkdir(parents=True, exist_ok=True)
