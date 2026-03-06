"""Memory package for AURA — backward-compatible re-export + unified interface."""

# Backward compatibility: re-export MemorySystem from old memory.py
from .memory_system import MemorySystem

# New unified memory interface (Phase 4C)
from .unified_memory import UnifiedMemory, UnifiedResult, get_unified_memory

__all__ = [
    "MemorySystem",
    "UnifiedMemory",
    "UnifiedResult",
    "get_unified_memory",
]
