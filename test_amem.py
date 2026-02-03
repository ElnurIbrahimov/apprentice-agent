#!/usr/bin/env python3
"""
Test script for A-MEM Agentic Memory System

Demonstrates:
- Adding memories with auto-linking
- Semantic search
- Memory evolution
- Box organization
- Link traversal
"""

import sys
sys.path.insert(0, '.')

from apprentice_agent.tools.amem import AMEMSystem, get_amem
from apprentice_agent.tools.amem_tool import AMEMTool, get_amem_tool


def test_basic_operations():
    """Test basic A-MEM operations."""
    print("\n" + "="*60)
    print("A-MEM: Agentic Memory System Test")
    print("="*60)

    # Create fresh instance for testing
    amem = AMEMSystem(
        db_path="data/amem_test/",
        evolution_enabled=False  # Disable for testing without LLM
    )

    # Add some memories
    print("\n[1] Adding memories...")

    note1 = amem.add(
        content="User prefers dark mode for all applications",
        tags=["preference", "ui"],
        category="semantic",
        importance=0.8
    )
    print(f"  Added: {note1.id[:12]} - {note1.content[:40]}...")
    print(f"  Keywords: {note1.keywords}")

    note2 = amem.add(
        content="Fixed CUDA out of memory error by reducing batch size to 4",
        tags=["debugging", "cuda", "ml"],
        category="procedural",
        importance=0.9
    )
    print(f"  Added: {note2.id[:12]} - {note2.content[:40]}...")
    print(f"  Keywords: {note2.keywords}")

    note3 = amem.add(
        content="CUDA errors often occur when GPU memory is exhausted during training",
        tags=["knowledge", "cuda", "gpu"],
        category="semantic",
        importance=0.7
    )
    print(f"  Added: {note3.id[:12]} - {note3.content[:40]}...")
    print(f"  Keywords: {note3.keywords}")
    print(f"  Auto-links: {len(note3.links)} (should link to note2)")

    note4 = amem.add(
        content="User's favorite programming language is Python",
        tags=["preference", "coding"],
        category="semantic",
        importance=0.6
    )
    print(f"  Added: {note4.id[:12]} - {note4.content[:40]}...")

    note5 = amem.add(
        content="The RTX 4060 has 8GB VRAM which limits batch sizes for large models",
        tags=["hardware", "gpu", "ml"],
        category="fact",
        importance=0.8
    )
    print(f"  Added: {note5.id[:12]} - {note5.content[:40]}...")
    print(f"  Auto-links: {len(note5.links)}")

    # Test search
    print("\n[2] Testing search...")

    print("\n  Search: 'CUDA memory problems'")
    results = amem.search("CUDA memory problems", k=3)
    for note, score in results:
        print(f"    [{score:.2f}] {note.content[:50]}...")

    print("\n  Search: 'user preferences'")
    results = amem.search("user preferences", k=3)
    for note, score in results:
        print(f"    [{score:.2f}] {note.content[:50]}...")

    # Test agentic search with link traversal
    print("\n[3] Agentic search with link traversal...")
    results = amem.search_agentic("GPU issues", k=5, follow_links=True)
    for r in results:
        hop_indicator = f" (via link, hop {r['hop']})" if r['hop'] > 0 else ""
        print(f"    [{r['relevance']:.2f}]{hop_indicator} {r['content'][:50]}...")

    # Test boxes
    print("\n[4] Testing boxes (soft clustering)...")
    boxes = amem.list_boxes()
    print(f"  Total boxes: {len(boxes)}")
    for box_name, count in list(boxes.items())[:5]:
        print(f"    {box_name}: {count} notes")

    # Test linked notes
    print("\n[5] Getting linked notes...")
    linked = amem.get_linked(note3.id)
    print(f"  Notes linked to '{note3.content[:30]}...':")
    for linked_note, strength in linked:
        print(f"    [{strength:.2f}] {linked_note.content[:50]}...")

    # Stats
    print("\n[6] Memory statistics...")
    stats = amem.get_stats()
    for key, value in stats.items():
        print(f"  {key}: {value}")

    # Save
    amem.save()
    print("\n[7] Saved to disk.")

    print("\n" + "="*60)
    print("Test complete!")
    print("="*60)


def test_tool_interface():
    """Test the tool interface."""
    print("\n" + "="*60)
    print("A-MEM Tool Interface Test")
    print("="*60)

    tool = AMEMTool()

    # Test remember command
    print("\n[1] Testing 'remember:' command...")
    result = tool.execute("remember: Aura was created by Elnur in 2025 [tag:history, tag:aura] [category:fact]")
    print(f"  Result: {result}")

    # Test recall command
    print("\n[2] Testing 'recall:' command...")
    result = tool.execute("recall: who created Aura")
    print(f"  Found {result.get('count', 0)} memories")
    for r in result.get('results', [])[:3]:
        print(f"    - {r['content'][:50]}...")

    # Test natural language recall
    print("\n[3] Testing natural language recall...")
    result = tool.execute("what do you remember about Aura?")
    print(f"  Found {result.get('count', 0)} memories")

    # Test stats
    print("\n[4] Testing stats command...")
    result = tool.execute("stats")
    print(f"  Stats: {result}")

    # Test boxes
    print("\n[5] Testing boxes command...")
    result = tool.execute("boxes")
    print(f"  Boxes: {result.get('boxes', {})}")

    print("\n" + "="*60)
    print("Tool interface test complete!")
    print("="*60)


if __name__ == "__main__":
    # Run tests
    test_basic_operations()
    test_tool_interface()

    print("\n\nA-MEM is ready for use!")
    print("Import with: from apprentice_agent.tools.amem_tool import get_amem_tool")
    print("Or use directly: from apprentice_agent.tools.amem import get_amem")
