"""
A-MEM Tool Interface for Aura Agent

Provides natural language interface to the A-MEM agentic memory system.

Usage:
    "remember: <content>" - Store a new memory
    "recall: <query>" - Search memories
    "what do you remember about X?" - Semantic search
    "link memory X to Y" - Create manual link
    "memory stats" - Show statistics
    "consolidate memories" - Run memory consolidation

Author: Aura Development Team
Created: 2026-02-03

ARCHITECTURE NOTE — A-MEM layer chain:
  - aura/tools/amem.py: Core AMEMSystem (atomic notes, embeddings, Qdrant search).
  - THIS FILE (aura/tools/amem_tool.py): Wraps AMEMSystem with a natural-language
    action interface. ApprenticeAgent.tools["amem"] → AMEMTool → AMEMSystem.
  - aura/tools/hybrid_amem.py: Optional unified store that adds KG entity extraction
    on top of both AMEMSystem and KnowledgeGraph simultaneously.
"""

import logging
from typing import Dict, Any, Optional, List

from .amem import AMEMSystem, MemoryNote, get_amem

logger = logging.getLogger(__name__)


class AMEMTool:
    """
    Tool wrapper for A-MEM agentic memory system.

    Provides natural language interface for the Aura agent.
    """

    name = "amem"
    description = """Agentic Memory System (A-MEM) - Zettelkasten-style memory with automatic linking.

Commands:
- remember: <content> - Store a new memory note
- recall: <query> - Search for related memories
- link: <note_id> to <note_id> - Create a link between notes
- evolve: <note_id> - Re-analyze and re-link a note
- box: <name> - List notes in a box/category
- stats - Show memory statistics
- consolidate - Merge duplicates, prune weak links

Examples:
- "remember: User prefers dark mode and compact layouts"
- "recall: what does user prefer for UI?"
- "remember: Solved CUDA error by updating drivers [tag:debugging, tag:cuda]"
"""

    def __init__(self, llm_func: Optional[callable] = None):
        """
        Initialize A-MEM tool.

        Args:
            llm_func: Function to call LLM for attribute extraction
        """
        self._amem: Optional[AMEMSystem] = None
        self._llm_func = llm_func

    @property
    def amem(self) -> AMEMSystem:
        """Lazy-load A-MEM system."""
        if self._amem is None:
            self._amem = get_amem(llm_func=self._llm_func)
        return self._amem

    def set_llm_func(self, llm_func: callable):
        """Set the LLM function for attribute extraction."""
        self._llm_func = llm_func
        if self._amem:
            self._amem.llm_func = llm_func

    def execute(self, action: str, **kwargs) -> Dict[str, Any]:
        """
        Execute A-MEM action.

        Args:
            action: The action string to parse and execute

        Returns:
            Dict with success status and results
        """
        action_lower = action.lower().strip()

        # Parse special syntax
        # Format: "remember: content [tag:x, tag:y] [category:episodic]"
        tags = []
        category = "general"

        # Extract tags
        if "[tag:" in action:
            import re
            tag_matches = re.findall(r'\[tag:([^\]]+)\]', action)
            tags = [t.strip() for t in tag_matches]
            action = re.sub(r'\[tag:[^\]]+\]', '', action).strip()

        # Extract category
        if "[category:" in action:
            import re
            cat_match = re.search(r'\[category:([^\]]+)\]', action)
            if cat_match:
                category = cat_match.group(1).strip()
                action = re.sub(r'\[category:[^\]]+\]', '', action).strip()

        action_lower = action.lower().strip()

        # REMEMBER: Store new memory
        if action_lower.startswith("remember:") or action_lower.startswith("store:"):
            content = action.split(":", 1)[1].strip()
            if not content:
                return {"success": False, "error": "No content provided"}

            note = self.amem.add(
                content=content,
                tags=tags,
                category=category,
                source="agent"
            )

            return {
                "success": True,
                "message": f"Stored memory with {len(note.links)} auto-links",
                "note_id": note.id,
                "keywords": note.keywords,
                "links": len(note.links),
                "boxes": note.boxes
            }

        # RECALL: Search memories
        if action_lower.startswith("recall:") or action_lower.startswith("search:"):
            query = action.split(":", 1)[1].strip()
            return self._search(query)

        # Natural language recall patterns
        recall_patterns = [
            "what do you remember about",
            "what do you know about",
            "recall anything about",
            "search for",
            "find memories about",
            "memories about",
            "memory of"
        ]
        for pattern in recall_patterns:
            if pattern in action_lower:
                query = action_lower.split(pattern)[-1].strip().rstrip("?")
                return self._search(query)

        # LINK: Manual link
        if action_lower.startswith("link:") or " to " in action_lower and "link" in action_lower:
            return self._parse_link(action)

        # EVOLVE: Re-analyze note
        if action_lower.startswith("evolve:"):
            note_id = action.split(":", 1)[1].strip()
            success = self.amem.update(note_id, re_extract=True)
            if success:
                note = self.amem.read(note_id)
                return {
                    "success": True,
                    "message": f"Re-analyzed note {note_id[:8]}",
                    "keywords": note.keywords if note else [],
                    "links": len(note.links) if note else 0
                }
            return {"success": False, "error": f"Note {note_id} not found"}

        # BOX: List notes in box
        if action_lower.startswith("box:"):
            box_name = action.split(":", 1)[1].strip()
            notes = self.amem.get_box(box_name)
            return {
                "success": True,
                "box": box_name,
                "count": len(notes),
                "notes": [
                    {"id": n.id, "content": n.content[:100], "tags": n.tags}
                    for n in notes[:20]
                ]
            }

        if action_lower == "boxes" or action_lower == "list boxes":
            boxes = self.amem.list_boxes()
            return {
                "success": True,
                "boxes": boxes
            }

        # STATS
        if action_lower in ("stats", "status", "memory stats"):
            stats = self.amem.get_stats()
            return {"success": True, **stats}

        # CONSOLIDATE
        if action_lower in ("consolidate", "consolidate memories", "cleanup"):
            result = self.amem.consolidate()
            return {"success": True, **result, "message": "Memory consolidation complete"}

        # GET: Get specific note
        if action_lower.startswith("get:") or action_lower.startswith("note:"):
            note_id = action.split(":", 1)[1].strip()
            note = self.amem.read(note_id)
            if note:
                linked = self.amem.get_linked(note_id)
                return {
                    "success": True,
                    "note": {
                        "id": note.id,
                        "content": note.content,
                        "keywords": note.keywords,
                        "tags": note.tags,
                        "context": note.context,
                        "category": note.category,
                        "importance": note.importance,
                        "links": len(note.links),
                        "created": note.created_at
                    },
                    "linked_notes": [
                        {"id": n.id, "content": n.content[:50], "strength": s}
                        for n, s in linked[:5]
                    ]
                }
            return {"success": False, "error": f"Note {note_id} not found"}

        # DELETE
        if action_lower.startswith("delete:") or action_lower.startswith("forget:"):
            note_id = action.split(":", 1)[1].strip()
            success = self.amem.delete(note_id)
            if success:
                return {"success": True, "message": f"Deleted note {note_id[:8]}"}
            return {"success": False, "error": f"Note {note_id} not found"}

        # RECENT
        if action_lower in ("recent", "recent memories", "latest"):
            notes = self.amem.get_recent_notes(limit=10)
            return {
                "success": True,
                "count": len(notes),
                "notes": [
                    {
                        "id": n.id,
                        "content": n.content[:100],
                        "keywords": n.keywords,
                        "created": n.created_at
                    }
                    for n in notes
                ]
            }

        # Default: treat as search
        return self._search(action)

    def _search(self, query: str) -> Dict[str, Any]:
        """Perform semantic search."""
        results = self.amem.search_agentic(query, k=5, follow_links=True)

        if not results:
            return {
                "success": True,
                "count": 0,
                "message": "No memories found",
                "results": []
            }

        return {
            "success": True,
            "count": len(results),
            "results": [
                {
                    "id": r["id"],
                    "content": r["content"],
                    "keywords": r.get("keywords", []),
                    "tags": r.get("tags", []),
                    "context": r.get("context", ""),
                    "relevance": round(r["relevance"], 2),
                    "hop": r.get("hop", 0)
                }
                for r in results
            ]
        }

    def _parse_link(self, action: str) -> Dict[str, Any]:
        """Parse and execute link command."""
        import re

        # Try pattern: "link <id1> to <id2>"
        match = re.search(r'link[:\s]+(\S+)\s+to\s+(\S+)', action, re.IGNORECASE)
        if match:
            source_id = match.group(1)
            target_id = match.group(2)

            success = self.amem.link(source_id, target_id)
            if success:
                return {
                    "success": True,
                    "message": f"Linked {source_id[:8]} -> {target_id[:8]}"
                }
            return {"success": False, "error": "Could not create link. Check note IDs."}

        return {"success": False, "error": "Invalid link syntax. Use: link <id1> to <id2>"}

    # =========================================================================
    # CONVENIENCE METHODS FOR AGENT
    # =========================================================================

    def remember(
        self,
        content: str,
        tags: Optional[List[str]] = None,
        category: str = "general",
        importance: float = 0.5
    ) -> MemoryNote:
        """Direct method to store a memory."""
        return self.amem.add(
            content=content,
            tags=tags or [],
            category=category,
            importance=importance,
            source="agent"
        )

    def recall(self, query: str, k: int = 5) -> List[Dict[str, Any]]:
        """Direct method to search memories."""
        return self.amem.search_agentic(query, k=k)

    def learn_from_interaction(
        self,
        user_message: str,
        aura_response: str,
        outcome: str = "success"
    ) -> Optional[MemoryNote]:
        """
        Learn from a user interaction.

        Automatically extracts and stores relevant knowledge.
        """
        # Don't store trivial interactions
        if len(user_message) < 20 and len(aura_response) < 50:
            return None

        # Create memory content
        content = f"User asked: {user_message[:200]}"
        if outcome == "success":
            content += f" | Resolution: {aura_response[:200]}"

        return self.amem.add(
            content=content,
            category="episodic",
            source="conversation",
            importance=0.6 if outcome == "success" else 0.4,
            tags=["interaction", outcome]
        )

    def learn_from_tool_use(
        self,
        tool_name: str,
        tool_input: str,
        tool_output: str,
        success: bool
    ) -> Optional[MemoryNote]:
        """
        Learn from tool execution.

        Stores procedural knowledge about tool usage patterns.
        """
        if not success and len(tool_output) < 10:
            return None

        content = f"Tool '{tool_name}' used with input: {tool_input[:150]}"
        if success:
            content += f" | Result: {tool_output[:150]}"
        else:
            content += f" | Failed: {tool_output[:100]}"

        return self.amem.add(
            content=content,
            category="procedural",
            source="tool",
            importance=0.7 if success else 0.5,
            tags=["tool-use", tool_name, "success" if success else "failure"]
        )

    def get_context_for_query(self, query: str, max_tokens: int = 500) -> str:
        """
        Get relevant memory context for a query.

        Returns formatted string suitable for injection into LLM prompt.
        """
        results = self.amem.search_agentic(query, k=3, follow_links=True)

        if not results:
            return ""

        context_parts = ["Relevant memories:"]
        total_length = 0

        for r in results:
            entry = f"- {r['content']}"
            if r.get('context'):
                entry += f" (Context: {r['context']})"

            if total_length + len(entry) > max_tokens:
                break

            context_parts.append(entry)
            total_length += len(entry)

        return "\n".join(context_parts)


# Singleton instance
_tool_instance: Optional[AMEMTool] = None


def get_amem_tool(llm_func: Optional[callable] = None) -> AMEMTool:
    """Get or create the global A-MEM tool instance."""
    global _tool_instance
    if _tool_instance is None:
        _tool_instance = AMEMTool(llm_func=llm_func)
    elif llm_func and not _tool_instance._llm_func:
        _tool_instance.set_llm_func(llm_func)
    return _tool_instance


# Export
__all__ = [
    "AMEMTool",
    "get_amem_tool"
]
