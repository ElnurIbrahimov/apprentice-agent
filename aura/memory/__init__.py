"""Memory package for AURA — consolidated SQLite store + retrieval pipeline.

Phase 2 Memory Consolidation:
  store.py         — Unified SQLite + FTS5 store (primary backend)
  retrieval.py     — BM25 + semantic + RRF + cross-encoder reranker
  fade_mem.py      — Exponential memory decay with spaced repetition
  user_profile.py  — Persistent user model
  unified_memory.py — Public API (backward-compatible query/store/store_gated)
  write_gate.py    — Memory write worthiness scoring (unchanged)
  context_budget.py — Token budget allocation (unchanged)
"""

# Backward compatibility: re-export MemorySystem from old memory.py
from .memory_system import MemorySystem

# Consolidated store (Phase 2)
from .store import MemoryStore, MemoryRecord, get_memory_store

# Unified memory interface (public API)
from .unified_memory import UnifiedMemory, UnifiedResult, get_unified_memory

# Retrieval pipeline
from .retrieval import retrieve, RetrievalResult

# FadeMem decay
from .fade_mem import batch_decay_and_prune, reinforce

# User profile
from .user_profile import UserProfile, load_profile, save_profile

__all__ = [
    # Legacy
    "MemorySystem",
    # Consolidated store
    "MemoryStore",
    "MemoryRecord",
    "get_memory_store",
    # Unified interface
    "UnifiedMemory",
    "UnifiedResult",
    "get_unified_memory",
    # Retrieval
    "retrieve",
    "RetrievalResult",
    # FadeMem
    "batch_decay_and_prune",
    "reinforce",
    # User profile
    "UserProfile",
    "load_profile",
    "save_profile",
]
