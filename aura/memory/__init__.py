"""Memory package for AURA — consolidated SQLite store + retrieval pipeline.

Phase 2 Memory Consolidation:
  store.py         — Unified SQLite + FTS5 store (primary backend)
  retrieval.py     — BM25 + semantic + RRF + cross-encoder reranker
  fade_mem.py      — Exponential memory decay with spaced repetition
  user_profile.py  — Persistent user model
  unified_memory.py — Public API (backward-compatible query/store/store_gated)
  write_gate.py    — Memory write worthiness scoring (unchanged)
  context_budget.py — Token budget allocation (unchanged)
  embedding.py     — Shared Ollama nomic-embed-text helper
"""

# Shared embedding utility
from .embedding import get_embedding

# FadeMem decay
from .fade_mem import batch_decay_and_prune, reinforce

# Retrieval pipeline
from .retrieval import RetrievalResult, retrieve

# Consolidated store (Phase 2)
from .store import MemoryRecord, MemoryStore, get_memory_store

# Unified memory interface (public API)
from .unified_memory import UnifiedMemory, UnifiedResult, get_unified_memory

# User profile
from .user_profile import UserProfile, load_profile, save_profile, update_profile_from_memories

__all__ = [
    "MemoryRecord",
    # Consolidated store
    "MemoryStore",
    "RetrievalResult",
    # Unified interface
    "UnifiedMemory",
    "UnifiedResult",
    # User profile
    "UserProfile",
    # FadeMem
    "batch_decay_and_prune",
    # Embedding
    "get_embedding",
    "get_memory_store",
    "get_unified_memory",
    "load_profile",
    "reinforce",
    # Retrieval
    "retrieve",
    "save_profile",
    "update_profile_from_memories",
]
