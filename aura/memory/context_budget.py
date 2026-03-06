"""Unified token budget for memory injection into LLM context."""
from dataclasses import dataclass, field


@dataclass
class ContextBudget:
    """Allocates a fixed token budget across memory systems.

    Usage:
        budget = ContextBudget(total_tokens=3000)
        amem_limit = budget.allocate("amem", requested=800)
        kg_limit = budget.allocate("kg", requested=600)
        # Each allocate() returns how many tokens this system can use
    """
    total_tokens: int = 3000
    _allocated: dict = field(default_factory=dict, repr=False)

    def allocate(self, system: str, requested: int) -> int:
        """Allocate up to `requested` tokens for `system`. Returns actual allocation.

        If `system` has already been allocated, returns the existing allocation.
        Negative `requested` values are treated as 0.
        """
        if system in self._allocated:
            return self._allocated[system]
        requested = max(0, requested)
        used = sum(self._allocated.values())
        available = max(0, self.total_tokens - used)
        granted = min(requested, available)
        self._allocated[system] = granted
        return granted

    @property
    def remaining(self) -> int:
        return max(0, self.total_tokens - sum(self._allocated.values()))

    def reset(self) -> None:
        self._allocated.clear()
