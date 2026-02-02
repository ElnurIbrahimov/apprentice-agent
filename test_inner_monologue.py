#!/usr/bin/env python3
"""Test script for Inner Monologue System."""

import sys
from pathlib import Path

# Add project to path
sys.path.insert(0, str(Path(__file__).parent))

from apprentice_agent.tools.inner_monologue import InnerMonologueTool, get_monologue, THOUGHT_TYPES

def test_inner_monologue():
    """Test the InnerMonologueTool functionality."""
    print("=" * 60)
    print("Testing Inner Monologue System")
    print("=" * 60)

    # Get singleton instance
    monologue = get_monologue()

    # Test 1: Start session
    print("\n1. Starting session...")
    session_id = monologue.start_session()
    print(f"   Session ID: {session_id}")

    # Test 2: Emit various thought types
    print("\n2. Emitting thoughts...")
    thoughts = [
        ("perceive", "Received user query about weather", None),
        ("recall", "Searching memory for weather-related queries...", None),
        ("reason", "User wants current weather. Should use web_search tool.", 85),
        ("decide", "Selected tool: web_search", 92),
        ("execute", "Running web search for weather...", None),
        ("reflect", "Got results from DuckDuckGo. Looks accurate.", 88),
        ("uncertain", "Low confidence on temperature unit preference.", 45),
        ("eureka", "Found user's location preference in memory!", 95),
    ]

    for thought_type, content, confidence in thoughts:
        t = monologue.think(thought_type, content, confidence=confidence)
        if t:
            print(f"   {t.format_display()}")

    # Test 3: Get recent thoughts
    print("\n3. Getting recent thoughts...")
    recent = monologue.get_recent_thoughts(5)
    print(f"   Found {len(recent)} recent thoughts")

    # Test 4: Get reasoning chain
    print("\n4. Getting reasoning chain...")
    chain = monologue.get_reasoning_chain(5)
    print(chain)

    # Test 5: Test verbosity levels
    print("\n5. Testing verbosity levels...")
    for level in [0, 1, 2, 3]:
        result = monologue.set_verbosity(level)
        print(f"   {result}")
    monologue.set_verbosity(2)  # Reset to default

    # Test 6: Execute interface
    print("\n6. Testing execute interface...")
    status = monologue.execute("status")
    print(f"   Status: {status}")

    # Test 7: End session
    print("\n7. Ending session...")
    result = monologue.end_session()
    print(f"   Session ended: {result.get('success')}")
    if result.get("summary"):
        summary = result["summary"]
        print(f"   Total thoughts: {summary.get('total_thoughts')}")
        print(f"   Eureka moments: {summary.get('eureka_moments')}")
        print(f"   Uncertain moments: {summary.get('uncertain_moments')}")

    print("\n" + "=" * 60)
    print("All tests passed!")
    print("=" * 60)

if __name__ == "__main__":
    test_inner_monologue()
