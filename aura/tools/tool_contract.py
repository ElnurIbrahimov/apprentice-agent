"""
Tool Contract — Phase 3.

Standardized metadata schema for AURA tools.
Provides ToolSpec dataclass + a registry so Brain/Parliament can inspect
tool capabilities before routing.

Priority tools annotated first; pattern is extensible to all 100+ tools.

Author: Aura reliability upgrade (2026-03)
"""

from __future__ import annotations

import logging
from dataclasses import dataclass, field
from enum import Enum
from typing import Any, Dict, List, Optional

logger = logging.getLogger(__name__)


# ---------------------------------------------------------------------------
# Enums
# ---------------------------------------------------------------------------

class ToolSafety(str, Enum):
    SAFE        = "safe"         # Read-only, idempotent
    SENSITIVE   = "sensitive"    # Reads external services / user data
    MUTATING    = "mutating"     # Writes files, sends messages, modifies state
    DESTRUCTIVE = "destructive"  # Deletes, overwrites, executes arbitrary code


class LatencyTier(str, Enum):
    INSTANT = "instant"    # < 50 ms
    FAST    = "fast"       # 50–500 ms
    MEDIUM  = "medium"     # 500 ms – 5 s
    SLOW    = "slow"       # 5 – 30 s
    VERY_SLOW = "very_slow"  # > 30 s


# ---------------------------------------------------------------------------
# ToolSpec
# ---------------------------------------------------------------------------

@dataclass
class ToolSpec:
    """
    Standardized contract for one AURA tool.

    All fields except name/description have sensible defaults so existing
    tools need minimal annotation.
    """
    # Identity
    name: str
    description: str

    # Schema hints (informal — not JSON Schema, just descriptive)
    input_schema: Dict[str, Any] = field(default_factory=dict)
    output_schema: Dict[str, Any] = field(default_factory=dict)

    # Side effects
    side_effects: List[str] = field(default_factory=list)
    mutates_state: bool = False
    idempotent: bool = True

    # Cost / latency
    latency_tier: LatencyTier = LatencyTier.MEDIUM
    cost_tier: int = 1   # 1=free/local, 2=API-cheap, 3=API-moderate, 4=API-expensive

    # Safety
    safety: ToolSafety = ToolSafety.SAFE

    # Failure modes
    failure_modes: List[str] = field(default_factory=list)
    safety_notes: str = ""

    # Usage guidance
    when_to_use: str = ""
    when_not_to_use: str = ""
    examples: List[str] = field(default_factory=list)

    # Microtask routing hint
    preferred_categories: List[str] = field(default_factory=list)

    def to_dict(self) -> Dict[str, Any]:
        return {
            "name": self.name,
            "description": self.description,
            "mutates_state": self.mutates_state,
            "idempotent": self.idempotent,
            "latency_tier": self.latency_tier.value,
            "cost_tier": self.cost_tier,
            "safety": self.safety.value,
            "side_effects": self.side_effects,
            "failure_modes": self.failure_modes,
            "safety_notes": self.safety_notes,
            "when_to_use": self.when_to_use,
            "when_not_to_use": self.when_not_to_use,
            "preferred_categories": self.preferred_categories,
        }

    def as_prompt_snippet(self) -> str:
        """Compact one-liner for injecting into LLM context."""
        safety_tag = f"[{self.safety.value}]"
        lat_tag    = f"[{self.latency_tier.value}]"
        return (
            f"tool:{self.name} {safety_tag}{lat_tag} — {self.description}"
            + (f" | when_to_use: {self.when_to_use}" if self.when_to_use else "")
        )


# ---------------------------------------------------------------------------
# Registry
# ---------------------------------------------------------------------------

class ToolRegistry:
    """Global registry of ToolSpec objects."""

    def __init__(self) -> None:
        self._specs: Dict[str, ToolSpec] = {}

    def register(self, spec: ToolSpec) -> None:
        self._specs[spec.name] = spec
        logger.debug("[ToolRegistry] Registered: %s", spec.name)

    def get(self, name: str) -> Optional[ToolSpec]:
        return self._specs.get(name)

    def all(self) -> List[ToolSpec]:
        return list(self._specs.values())

    def by_safety(self, safety: ToolSafety) -> List[ToolSpec]:
        return [s for s in self._specs.values() if s.safety == safety]

    def by_latency(self, tier: LatencyTier) -> List[ToolSpec]:
        return [s for s in self._specs.values() if s.latency_tier == tier]

    def prompt_summary(self, names: Optional[List[str]] = None) -> str:
        """Return compact summary lines for LLM context injection."""
        specs = [self._specs[n] for n in names if n in self._specs] if names else self.all()
        return "\n".join(s.as_prompt_snippet() for s in specs)

    def to_dict(self) -> Dict[str, Any]:
        return {name: spec.to_dict() for name, spec in self._specs.items()}


# ---------------------------------------------------------------------------
# Singleton
# ---------------------------------------------------------------------------

_registry: Optional[ToolRegistry] = None

def get_tool_registry() -> ToolRegistry:
    global _registry
    if _registry is None:
        _registry = ToolRegistry()
        _register_builtin_tools(_registry)
    return _registry


# ---------------------------------------------------------------------------
# Built-in annotations — priority tools
# ---------------------------------------------------------------------------

def _register_builtin_tools(reg: ToolRegistry) -> None:
    """Annotate the most important / high-risk tools."""

    from aura.reliability.routing_stats import MicrotaskCategory

    # --- Memory tools ---
    reg.register(ToolSpec(
        name="memory_search",
        description="Search across all memory systems (A-MEM, KG, RAG, Episodic) for relevant context.",
        input_schema={"query": "str", "k": "int=10", "sources": "list[str]|None"},
        output_schema={"results": "list[UnifiedResult]"},
        mutates_state=False, idempotent=True,
        latency_tier=LatencyTier.FAST, cost_tier=1,
        safety=ToolSafety.SAFE,
        failure_modes=["empty results when memory systems unavailable"],
        when_to_use="When you need to recall past user preferences, facts, or conversation context.",
        when_not_to_use="Don't call more than 3× per turn — check loop guard.",
        preferred_categories=[MicrotaskCategory.MEMORY_SUMMARIZATION, MicrotaskCategory.TOOL_SELECTION],
    ))

    reg.register(ToolSpec(
        name="memory_save",
        description="Write a memory to unified memory with gate scoring (may be discarded if low value).",
        input_schema={"content": "str", "importance": "float", "explicit_save": "bool"},
        output_schema={"decision": "str", "score": "float", "ids": "dict"},
        mutates_state=True, idempotent=False,
        latency_tier=LatencyTier.FAST, cost_tier=1,
        safety=ToolSafety.MUTATING,
        side_effects=["writes to A-MEM", "writes to Episodic memory"],
        failure_modes=["gate discards low-value content", "backends unavailable"],
        safety_notes="Gate checks for duplicates and noise. Explicit saves bypass threshold.",
        when_to_use="To persist a user preference, corrected fact, or key learning.",
        when_not_to_use="Don't call after every message — gate should handle filtering.",
        preferred_categories=[MicrotaskCategory.MEMORY_SUMMARIZATION, MicrotaskCategory.KG_MERGE],
    ))

    # --- Browser tools ---
    reg.register(ToolSpec(
        name="browser_navigate",
        description="Navigate to a URL in the managed browser session.",
        input_schema={"url": "str", "session_id": "str"},
        output_schema={"success": "bool", "title": "str", "status_code": "int"},
        mutates_state=True, idempotent=False,
        latency_tier=LatencyTier.SLOW, cost_tier=1,
        safety=ToolSafety.SENSITIVE,
        side_effects=["loads external URL", "may set cookies"],
        failure_modes=["timeout", "domain drift", "blocked URL"],
        safety_notes="BLOCKED_PATTERNS enforced. Domain drift causes hard abort.",
        when_to_use="First step of any browser task.",
        when_not_to_use="Don't navigate to login/payment/bank URLs.",
        preferred_categories=[MicrotaskCategory.BROWSER_PLANNING],
    ))

    reg.register(ToolSpec(
        name="browser_click",
        description="Click a DOM element by CSS selector with retry + postcondition check.",
        input_schema={"selector": "str", "session_id": "str", "success_signals": "list[str]"},
        output_schema={"success": "bool", "postcondition_passed": "bool|None", "retry_count": "int"},
        mutates_state=True, idempotent=False,
        latency_tier=LatencyTier.MEDIUM, cost_tier=1,
        safety=ToolSafety.MUTATING,
        side_effects=["may submit forms", "may navigate pages"],
        failure_modes=["element not found", "element not clickable", "unexpected navigation"],
        safety_notes="Destructive actions (confirm/delete/purchase) are hard-blocked.",
        when_to_use="To interact with page elements (buttons, links, form controls).",
        when_not_to_use="Do not click on checkout/payment without explicit user permission.",
        preferred_categories=[MicrotaskCategory.BROWSER_PLANNING, MicrotaskCategory.BROWSER_RECOVERY],
    ))

    reg.register(ToolSpec(
        name="browser_type",
        description="Fill a text input or textarea by CSS selector.",
        input_schema={"selector": "str", "text": "str", "session_id": "str"},
        output_schema={"success": "bool"},
        mutates_state=True, idempotent=False,
        latency_tier=LatencyTier.FAST, cost_tier=1,
        safety=ToolSafety.MUTATING,
        side_effects=["sets form field value", "may trigger JS events"],
        failure_modes=["element not found", "element not writable"],
        when_to_use="To enter text in search boxes, forms, textareas.",
        preferred_categories=[MicrotaskCategory.BROWSER_PLANNING],
    ))

    # --- Search tools ---
    reg.register(ToolSpec(
        name="web_search",
        description="Search the web via Brave/Tavily and return ranked results.",
        input_schema={"query": "str", "max_results": "int=10"},
        output_schema={"results": "list[dict]"},
        mutates_state=False, idempotent=True,
        latency_tier=LatencyTier.SLOW, cost_tier=2,
        safety=ToolSafety.SENSITIVE,
        failure_modes=["API quota exceeded", "network error", "empty results"],
        when_to_use="When user asks for current info or facts not in memory.",
        when_not_to_use="Don't search for things already in memory — check memory first.",
        preferred_categories=[MicrotaskCategory.LONG_DOC_EXTRACTION],
    ))

    # --- Filesystem ---
    reg.register(ToolSpec(
        name="filesystem_read",
        description="Read a file from the local filesystem.",
        input_schema={"path": "str"},
        output_schema={"content": "str"},
        mutates_state=False, idempotent=True,
        latency_tier=LatencyTier.INSTANT, cost_tier=1,
        safety=ToolSafety.SAFE,
        failure_modes=["file not found", "permission denied", "binary file"],
        when_to_use="To read source code, config files, data files.",
        preferred_categories=[MicrotaskCategory.CODE_EDIT, MicrotaskCategory.LONG_DOC_EXTRACTION],
    ))

    reg.register(ToolSpec(
        name="filesystem_write",
        description="Write or overwrite a file on the local filesystem.",
        input_schema={"path": "str", "content": "str"},
        output_schema={"success": "bool"},
        mutates_state=True, idempotent=False,
        latency_tier=LatencyTier.INSTANT, cost_tier=1,
        safety=ToolSafety.MUTATING,
        side_effects=["overwrites file on disk"],
        failure_modes=["permission denied", "disk full"],
        safety_notes="Always confirm path before writing. Never write to system paths.",
        when_to_use="To save generated code, configs, or documents.",
        when_not_to_use="Don't overwrite without reading existing content first.",
        preferred_categories=[MicrotaskCategory.CODE_EDIT],
    ))

    # --- Code execution ---
    reg.register(ToolSpec(
        name="code_executor",
        description="Execute Python code in a sandboxed subprocess.",
        input_schema={"code": "str", "timeout": "int=30"},
        output_schema={"stdout": "str", "stderr": "str", "exit_code": "int"},
        mutates_state=True, idempotent=False,
        latency_tier=LatencyTier.SLOW, cost_tier=1,
        safety=ToolSafety.DESTRUCTIVE,
        side_effects=["runs subprocess", "may modify filesystem", "may call network"],
        failure_modes=["timeout", "syntax error", "runtime exception", "import error"],
        safety_notes="Only execute code you generated or the user explicitly provided.",
        when_to_use="For calculations, data transformations, or testing generated code.",
        when_not_to_use="Don't execute untrusted user-provided shell commands.",
        preferred_categories=[MicrotaskCategory.CODE_EDIT],
    ))

    # --- PDF / OCR ---
    reg.register(ToolSpec(
        name="pdf_reader",
        description="Extract text and metadata from a PDF file.",
        input_schema={"path": "str", "max_pages": "int|None"},
        output_schema={"text": "str", "pages": "int", "metadata": "dict"},
        mutates_state=False, idempotent=True,
        latency_tier=LatencyTier.MEDIUM, cost_tier=1,
        safety=ToolSafety.SAFE,
        failure_modes=["scanned/image-only PDF needs OCR", "password protected", "corrupt file"],
        when_to_use="To extract content from uploaded PDF documents.",
        preferred_categories=[MicrotaskCategory.LONG_DOC_EXTRACTION, MicrotaskCategory.OCR_CLEANUP],
    ))

    # --- Model compare ---
    reg.register(ToolSpec(
        name="model_compare",
        description="Send the same prompt to multiple models and return all responses.",
        input_schema={"prompt": "str", "models": "list[str]"},
        output_schema={"results": "list[{model, response, latency_ms}]"},
        mutates_state=False, idempotent=True,
        latency_tier=LatencyTier.VERY_SLOW, cost_tier=4,
        safety=ToolSafety.SAFE,
        failure_modes=["model unavailable", "timeout", "API quota"],
        when_to_use="When user explicitly asks for model comparison or best-of-N.",
        when_not_to_use="Don't compare for every query — expensive. Use at most 1× per turn.",
        preferred_categories=[MicrotaskCategory.GENERAL],
    ))

    # --- Knowledge graph ---
    reg.register(ToolSpec(
        name="knowledge_graph_search",
        description="Search the local knowledge graph for entity relationships.",
        input_schema={"query": "str", "limit": "int=10"},
        output_schema={"nodes": "list[KGNode]"},
        mutates_state=False, idempotent=True,
        latency_tier=LatencyTier.FAST, cost_tier=1,
        safety=ToolSafety.SAFE,
        failure_modes=["KG locked by concurrent write", "no matching entities"],
        when_to_use="To retrieve structured facts about entities the user has mentioned.",
        preferred_categories=[MicrotaskCategory.KG_MERGE],
    ))

    reg.register(ToolSpec(
        name="knowledge_graph_add",
        description="Add or update a node/edge in the knowledge graph.",
        input_schema={"label": "str", "type": "str", "properties": "dict"},
        output_schema={"node_id": "str"},
        mutates_state=True, idempotent=False,
        latency_tier=LatencyTier.FAST, cost_tier=1,
        safety=ToolSafety.MUTATING,
        side_effects=["modifies KG database"],
        safety_notes="Check for contradictions before adding conflicting facts.",
        when_to_use="To persist a new fact or relationship about the user.",
        preferred_categories=[MicrotaskCategory.KG_MERGE],
    ))

    logger.debug("[ToolRegistry] %d built-in tools registered", len(reg._specs))


@dataclass
class ToolResult:
    """Standardized return type for tool executions.

    Provides a consistent shape so callers never have to guess
    whether a tool returned a dict, raised, or returned None.
    """
    success: bool
    result: Any = None
    error: str = ""

    def to_dict(self) -> Dict[str, Any]:
        d: Dict[str, Any] = {"success": self.success}
        if self.result is not None:
            d["result"] = self.result
        if self.error:
            d["error"] = self.error
        return d


__all__ = [
    "LatencyTier",
    "ToolRegistry",
    "ToolResult",
    "ToolSafety",
    "ToolSpec",
    "get_tool_registry",
]
