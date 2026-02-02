#!/usr/bin/env python3
"""Standalone test for Inner Monologue System - no external dependencies."""

import sys
import json
import io
from pathlib import Path
from datetime import datetime

# Fix Windows console encoding for emoji
if sys.platform == 'win32':
    sys.stdout = io.TextIOWrapper(sys.stdout.buffer, encoding='utf-8', errors='replace')

# Direct import (bypass package __init__.py)
sys.path.insert(0, str(Path(__file__).parent / "apprentice_agent" / "tools"))

from inner_monologue import InnerMonologueTool, Thought, MonologueSession, THOUGHT_TYPES, THOUGHT_ICONS

def test_inner_monologue():
    """Test the InnerMonologueTool functionality."""
    print("=" * 60)
    print("Testing Inner Monologue System (Standalone)")
    print("=" * 60)

    # Create instance directly
    monologue = InnerMonologueTool()

    # Test 1: Start session
    print("\n1. Starting session...")
    session_id = monologue.start_session()
    print(f"   Session ID: {session_id}")
    assert session_id is not None
    assert len(session_id) == 8

    # Test 2: Emit various thought types
    print("\n2. Emitting thoughts...")
    thoughts_data = [
        ("perceive", "Received user query about weather", None),
        ("recall", "Searching memory for weather-related queries...", None),
        ("reason", "User wants current weather. Should use web_search tool.", 85),
        ("decide", "Selected tool: web_search", 92),
        ("execute", "Running web search for weather...", None),
        ("reflect", "Got results from DuckDuckGo. Looks accurate.", 88),
        ("uncertain", "Low confidence on temperature unit preference.", 45),
        ("eureka", "Found user's location preference in memory!", 95),
    ]

    for thought_type, content, confidence in thoughts_data:
        t = monologue.think(thought_type, content, confidence=confidence)
        if t:
            print(f"   {t.format_display()}")
            assert t.type == thought_type
            assert t.content == content
            if confidence:
                assert t.confidence == confidence

    # Test 3: Get recent thoughts
    print("\n3. Getting recent thoughts...")
    recent = monologue.get_recent_thoughts(5)
    print(f"   Found {len(recent)} recent thoughts")
    assert len(recent) == 5

    # Test 4: Get session log
    print("\n4. Getting session log...")
    log = monologue.get_session_log()
    print(f"   Session has {len(log)} thoughts")
    assert len(log) == len(thoughts_data)

    # Test 5: Get reasoning chain
    print("\n5. Getting reasoning chain...")
    chain = monologue.get_reasoning_chain(5)
    print(chain[:200] + "...")
    assert "**" in chain  # Check that formatted thought headers exist

    # Test 6: Test verbosity levels
    print("\n6. Testing verbosity levels...")
    for level in [0, 1, 2, 3]:
        result = monologue.set_verbosity(level)
        print(f"   {result}")
        assert f"verbosity set to {level}" in result.lower()
    monologue.set_verbosity(2)  # Reset to default

    # Test 7: Execute interface - status
    print("\n7. Testing execute interface - status...")
    status = monologue.execute("status")
    print(f"   Status: success={status.get('success')}, thoughts={status.get('thought_count')}")
    assert status.get("success") == True
    assert status.get("thought_count") == len(thoughts_data)

    # Test 8: Execute interface - show thoughts
    print("\n8. Testing execute interface - show thoughts...")
    thoughts_result = monologue.execute("show thoughts")
    print(f"   Found {thoughts_result.get('count')} thoughts")
    assert thoughts_result.get("success") == True

    # Test 9: Execute interface - why
    print("\n9. Testing execute interface - why...")
    why_result = monologue.execute("why did you do that")
    print(f"   Reasoning chain available: {len(why_result.get('reasoning_chain', ''))} chars")
    assert why_result.get("success") == True

    # Test 10: End session
    print("\n10. Ending session...")
    result = monologue.end_session()
    print(f"   Session ended: {result.get('success')}")
    assert result.get("success") == True

    if result.get("summary"):
        summary = result["summary"]
        print(f"   Total thoughts: {summary.get('total_thoughts')}")
        print(f"   Eureka moments: {summary.get('eureka_moments')}")
        print(f"   Uncertain moments: {summary.get('uncertain_moments')}")
        print(f"   Thoughts by type: {summary.get('thoughts_by_type')}")
        assert summary.get("total_thoughts") == len(thoughts_data)
        assert summary.get("eureka_moments") == 1
        assert summary.get("uncertain_moments") == 1

    # Test 11: Verify thought types and icons
    print("\n11. Verifying thought types and icons...")
    for ttype, label in THOUGHT_TYPES.items():
        icon = THOUGHT_ICONS.get(ttype)
        print(f"   {icon} {ttype}: {label}")
        assert ttype in THOUGHT_ICONS

    # Test 12: Check log files were created
    print("\n12. Checking log files...")
    logs_dir = monologue.logs_dir
    sessions_dir = monologue.sessions_dir
    summaries_dir = monologue.summaries_dir
    print(f"   Logs dir: {logs_dir}")
    print(f"   Sessions dir: {sessions_dir}")
    print(f"   Summaries dir: {summaries_dir}")

    # Check if session file was created
    today = datetime.now().strftime('%Y-%m-%d')
    session_files = list(sessions_dir.glob(f"{today}_session_*.jsonl"))
    summary_files = list(summaries_dir.glob(f"{today}_summary.json"))
    print(f"   Session files today: {len(session_files)}")
    print(f"   Summary files today: {len(summary_files)}")

    print("\n" + "=" * 60)
    print("All tests passed!")
    print("=" * 60)
    return True

if __name__ == "__main__":
    success = test_inner_monologue()
    sys.exit(0 if success else 1)
