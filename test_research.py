#!/usr/bin/env python3
"""Test deep research with timeout protection."""

import sys
import io

if sys.platform == 'win32':
    sys.stdout = io.TextIOWrapper(sys.stdout.buffer, encoding='utf-8', errors='replace')

print("Testing Deep Research with Timeout Protection")
print("=" * 50)

from apprentice_agent.tools.deep_research import DeepResearchTool

dr = DeepResearchTool()

# Set progress callback to print
dr.set_progress_callback(lambda msg: print(f"  PROGRESS: {msg}"))

print("\nRunning quick research on 'machine learning'...")
result = dr.research('machine learning', depth='quick')

print(f"\nResults:")
print(f"  Success: {result.get('success')}")
print(f"  Time: {result.get('time_seconds')}s")
print(f"  Timed out: {result.get('timed_out')}")
print(f"  Queries: {result.get('queries_run')}")
print(f"  URLs found: {result.get('urls_found')}")
print(f"  Pages read: {result.get('pages_read')}")
print(f"  Summary: {result.get('summary')}")

if result.get('sources'):
    print(f"\nTop 3 sources:")
    for s in result['sources'][:3]:
        print(f"  - {s['title'][:50]}")
