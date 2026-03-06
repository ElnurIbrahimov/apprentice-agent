# A-MEM - Zettelkasten Agentic Memory

## Overview
Inspired by the Zettelkasten note-taking method. Each memory is a "note" with content, tags, links to other notes, and metadata. Enables associative recall.

## Key Concept
Unlike traditional vector-store memory, A-MEM creates **interconnected notes** that reference each other, forming an organic knowledge network.

## Data Model
```python
@dataclass
class MemoryNote:
    id: str
    content: str
    tags: List[str]
    links: List[str]          # IDs of related notes
    source: str               # conversation, tool, auto
    created_at: str
    access_count: int
    last_accessed: str
    importance: float         # 0-1
    embedding: List[float]    # For semantic search
```

## Operations
- `add_note(content, tags, links)` - Create note
- `recall(query, limit)` - Semantic search
- `link_notes(id1, id2)` - Create bidirectional link
- `get_related(note_id)` - Follow links
- `strengthen(note_id)` - Increase importance on access

## Hybrid A-MEM
`HybridAMEMSystem` combines A-MEM with Knowledge Graph:
- A-MEM for unstructured, associative memory
- KG for structured, relationship-based memory
- Queries search both and merge results

## Files
- `aura/tools/amem.py` - Core A-MEM system
- `aura/tools/amem_tool.py` - Tool wrapper
- `aura/tools/hybrid_amem.py` - KG + A-MEM hybrid
