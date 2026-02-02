#!/usr/bin/env python3
"""Test all fixes: tools, deep_research, memory persistence."""

import sys
import io

if sys.platform == 'win32':
    sys.stdout = io.TextIOWrapper(sys.stdout.buffer, encoding='utf-8', errors='replace')

print("=" * 60)
print("TEST 1: Count Tools")
print("=" * 60)
from apprentice_agent.agent import ApprenticeAgent
agent = ApprenticeAgent()
print(f"Tools loaded: {len(agent.tools)}")
print(f"Tools: {list(agent.tools.keys())}")

print()
print("=" * 60)
print("TEST 2: Deep Research")
print("=" * 60)
from apprentice_agent.tools.deep_research import DeepResearchTool
dr = DeepResearchTool()
result = dr.research('Python programming', depth='quick')
print(f"Success: {result.get('success')}")
print(f"Queries run: {result.get('queries_run')}")
print(f"URLs found: {result.get('urls_found')}")
print(f"Time: {result.get('time_seconds')}s")
if result.get('sources'):
    print(f"First source: {result['sources'][0]['title'][:50]}")

print()
print("=" * 60)
print("TEST 3: Memory Persistence - Store")
print("=" * 60)
from apprentice_agent.memory_retriever import MemoryRetriever
mr = MemoryRetriever()
mr.store_fact('name', 'Elnur')
mr.store_fact('relationship', 'older brother')
print(f"Stored: name=Elnur, relationship=older brother")
print(f"Profile path: {mr.user_profile_path}")

print()
print("=" * 60)
print("TEST 4: Memory Persistence - Reload")
print("=" * 60)
mr2 = MemoryRetriever()  # New instance
print(f"Name: {mr2.get_fact('name')}")
print(f"Relationship: {mr2.get_fact('relationship')}")
print(f"Profile has {len(mr2.user_profile)} facts")

print()
print("=" * 60)
print("TEST 5: Chat works")
print("=" * 60)
response = agent.chat('hello')
print(f"Chat works: {len(response) > 0}")
print(f"Response: {response[:100]}...")

print()
print("=" * 60)
print("ALL TESTS COMPLETE")
print("=" * 60)
