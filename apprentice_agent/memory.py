"""Memory system using JSON persistence with text similarity search.

This is a lightweight alternative to ChromaDB that works on all Python versions.
Can be swapped for ChromaDB when it supports your Python version.
"""

import json
import re
from collections import Counter
from datetime import datetime
from pathlib import Path
from typing import Optional

from .config import Config


class MemorySystem:
    """Manages long-term memory storage and retrieval using JSON persistence."""

    def __init__(self, collection_name: Optional[str] = None):
        self.collection_name = collection_name or Config.MEMORY_COLLECTION_NAME
        Config.CHROMADB_PATH.mkdir(parents=True, exist_ok=True)
        self.storage_path = Config.CHROMADB_PATH / f"{self.collection_name}.json"
        self.memories: list[dict] = []
        self._load()

    def _load(self) -> None:
        """Load memories from disk."""
        if self.storage_path.exists():
            try:
                self.memories = json.loads(self.storage_path.read_text(encoding="utf-8"))
            except (json.JSONDecodeError, IOError):
                self.memories = []

    def _save(self) -> None:
        """Save memories to disk."""
        self.storage_path.write_text(
            json.dumps(self.memories, indent=2, default=str),
            encoding="utf-8"
        )

    def remember(
        self,
        content: str,
        memory_type: str = "episodic",
        metadata: Optional[dict] = None
    ) -> str:
        """Store a new memory with optional metadata."""
        memory_id = f"{memory_type}_{datetime.now().strftime('%Y%m%d_%H%M%S_%f')}"

        memory = {
            "id": memory_id,
            "content": content,
            "type": memory_type,
            "timestamp": datetime.now().isoformat(),
            "metadata": metadata or {}
        }

        self.memories.append(memory)
        self._save()
        return memory_id

    def recall(
        self,
        query: str,
        n_results: Optional[int] = None,
        memory_type: Optional[str] = None
    ) -> list[dict]:
        """Retrieve relevant memories based on text similarity."""
        n_results = n_results or Config.MAX_MEMORY_RESULTS

        candidates = self.memories
        if memory_type:
            candidates = [m for m in candidates if m.get("type") == memory_type]

        # Score memories by text similarity
        scored = []
        for memory in candidates:
            score = self._similarity_score(query, memory["content"])
            scored.append((score, memory))

        # Sort by score descending
        scored.sort(key=lambda x: x[0], reverse=True)

        results = []
        for score, memory in scored[:n_results]:
            results.append({
                "id": memory["id"],
                "content": memory["content"],
                "metadata": {
                    "type": memory["type"],
                    "timestamp": memory["timestamp"],
                    **memory.get("metadata", {})
                },
                "distance": 1.0 - score  # Convert similarity to distance
            })
        return results

    def _similarity_score(self, query: str, text: str) -> float:
        """Calculate simple text similarity using word overlap (Jaccard-like)."""
        def tokenize(s: str) -> set[str]:
            words = re.findall(r'\b\w+\b', s.lower())
            return set(words)

        query_tokens = tokenize(query)
        text_tokens = tokenize(text)

        if not query_tokens or not text_tokens:
            return 0.0

        intersection = len(query_tokens & text_tokens)
        union = len(query_tokens | text_tokens)

        return intersection / union if union > 0 else 0.0

    def forget(self, memory_id: str) -> None:
        """Remove a specific memory by ID."""
        self.memories = [m for m in self.memories if m["id"] != memory_id]
        self._save()

    def get_recent(self, n: int = 10, memory_type: Optional[str] = None) -> list[dict]:
        """Get the most recent memories."""
        candidates = self.memories
        if memory_type:
            candidates = [m for m in candidates if m.get("type") == memory_type]

        # Sort by timestamp descending
        sorted_memories = sorted(
            candidates,
            key=lambda m: m.get("timestamp", ""),
            reverse=True
        )

        results = []
        for memory in sorted_memories[:n]:
            results.append({
                "id": memory["id"],
                "content": memory["content"],
                "metadata": {
                    "type": memory["type"],
                    "timestamp": memory["timestamp"],
                    **memory.get("metadata", {})
                }
            })
        return results

    def clear(self) -> None:
        """Clear all memories."""
        self.memories = []
        self._save()

    def count(self) -> int:
        """Return the number of stored memories."""
        return len(self.memories)
