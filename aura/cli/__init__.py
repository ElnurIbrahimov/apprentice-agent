"""AURA CLI interface — rich display + prompt_toolkit input."""
from __future__ import annotations

# Sentinel strings that indicate an LLM error response (not real content).
ERROR_SENTINELS = ["I'm having trouble processing", "[LLM Error]"]
