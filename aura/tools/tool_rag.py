"""Tool RAG — Embedding-based dynamic tool selection for AURA.

Embeds all tool descriptions at startup using nomic-embed-text,
then retrieves the top-K most relevant tools per query via cosine similarity.
"""

import json
import logging
import math
from typing import Optional

logger = logging.getLogger(__name__)

# Core tools always available as fallback
CORE_TOOLS = [
    "read_file", "grep", "glob", "list_dir",
    "shell", "search_web", "edit_file", "write_file",
]


def _cosine_similarity(a: list[float], b: list[float]) -> float:
    """Compute cosine similarity between two vectors."""
    dot = sum(x * y for x, y in zip(a, b))
    norm_a = math.sqrt(sum(x * x for x in a))
    norm_b = math.sqrt(sum(x * x for x in b))
    if norm_a == 0 or norm_b == 0:
        return 0.0
    return dot / (norm_a * norm_b)


class ToolRAG:
    """Embedding-based tool selection. Lazy-initialized on first query."""

    def __init__(self):
        self._tool_embeddings: dict[str, list[float]] = {}
        self._tool_schemas: dict[str, dict] = {}
        self._tool_descriptions: dict[str, str] = {}
        self._initialized = False
        self._schemas_loaded = False
        self._client = None
        # Stored references for lazy embedding
        self._agent_tools: dict = {}
        self._agentic_tool_schemas: list[dict] = []

    def initialize(self, agent_tools: dict, agentic_tool_schemas: list[dict]):
        """Register tools for lazy embedding. Schemas are loaded now,
        but embeddings are deferred to the first select_tools() call.

        Args:
            agent_tools: The agent's self.tools dict {name: tool_instance}
            agentic_tool_schemas: AGENTIC_TOOLS list from tool_schemas.py
        """
        try:
            import ollama
            from aura.config import Config
            self._client = ollama.Client(host=Config.OLLAMA_HOST)
        except Exception as e:
            logger.warning(f"[ToolRAG] Ollama client init failed: {e}")
            return

        # 1. Import dev tools that already have schemas
        for schema in agentic_tool_schemas:
            name = schema["function"]["name"]
            desc = schema["function"]["description"]
            self._tool_schemas[name] = schema
            self._tool_descriptions[name] = desc

        # 2. Generate schemas for agent tools not already covered
        for tool_name, tool_instance in agent_tools.items():
            if tool_name in self._tool_schemas:
                continue
            schema = self._generate_schema(tool_name, tool_instance)
            if schema:
                self._tool_schemas[tool_name] = schema
                self._tool_descriptions[tool_name] = schema["function"]["description"]

        self._schemas_loaded = True
        logger.info(f"[ToolRAG] Schemas loaded: {len(self._tool_descriptions)} tools (embeddings deferred)")

    def _ensure_embeddings(self):
        """Lazy-embed all tool descriptions on first query using batch embedding."""
        if self._initialized or not self._schemas_loaded or not self._client:
            return

        # Batch embed all descriptions in a single API call
        names = list(self._tool_descriptions.keys())
        texts = [self._tool_descriptions[n][:2000] for n in names]

        try:
            resp = self._client.embed(model="nomic-embed-text", input=texts)
            embeddings = resp.get("embeddings") or []
            for i, name in enumerate(names):
                if i < len(embeddings) and embeddings[i]:
                    self._tool_embeddings[name] = embeddings[i]
            logger.info(f"[ToolRAG] Batch embedded {len(self._tool_embeddings)}/{len(names)} tools")
        except Exception as e:
            logger.warning(f"[ToolRAG] Batch embed failed, falling back to individual: {e}")
            for name, desc in self._tool_descriptions.items():
                emb = self._embed(desc)
                if emb:
                    self._tool_embeddings[name] = emb

        self._initialized = True

    def select_tools(
        self,
        query: str,
        k: int = 8,
        always_include: Optional[list[str]] = None,
    ) -> list[dict]:
        """Return top-k tool schemas most relevant to the query.

        Args:
            query: The user's goal/query
            k: Number of tools to return
            always_include: Tool names to always include regardless of score
        """
        # Lazy-initialize embeddings on first query
        self._ensure_embeddings()

        if not self._initialized or not self._tool_embeddings:
            return self._fallback_schemas()

        query_emb = self._embed(query)
        if not query_emb:
            return self._fallback_schemas()

        # Score all tools
        scores = {}
        for name, tool_emb in self._tool_embeddings.items():
            scores[name] = _cosine_similarity(query_emb, tool_emb)

        # Sort by score, take top k
        ranked = sorted(scores.items(), key=lambda x: x[1], reverse=True)[:k]
        selected = [name for name, _ in ranked]

        # Always include forced tools
        if always_include:
            for name in always_include:
                if name not in selected and name in self._tool_schemas:
                    selected.append(name)

        return [self._tool_schemas[name] for name in selected if name in self._tool_schemas]

    def _fallback_schemas(self) -> list[dict]:
        """Return core tool schemas when embedding fails."""
        return [
            self._tool_schemas[name]
            for name in CORE_TOOLS
            if name in self._tool_schemas
        ]

    def _embed(self, text: str) -> Optional[list[float]]:
        """Compute embedding using nomic-embed-text."""
        if not self._client:
            return None
        try:
            resp = self._client.embeddings(model="nomic-embed-text", prompt=text[:2000])
            return resp.get("embedding") or (resp.get("embeddings") or [None])[0]
        except Exception:
            return None

    def _generate_schema(self, tool_name: str, tool_instance) -> Optional[dict]:
        """Generate Ollama tool-calling schema from a tool instance."""
        desc = (
            getattr(tool_instance, "description", "")
            or getattr(tool_instance, "__doc__", "")
            or tool_instance.__class__.__doc__
            or tool_name
        )
        if not desc or desc == tool_name:
            return None

        return {
            "type": "function",
            "function": {
                "name": tool_name,
                "description": str(desc)[:500],
                "parameters": {
                    "type": "object",
                    "properties": {
                        "action": {
                            "type": "string",
                            "description": "The action to perform with this tool",
                        }
                    },
                    "required": ["action"],
                },
            },
        }
