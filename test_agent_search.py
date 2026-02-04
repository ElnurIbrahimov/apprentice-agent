#!/usr/bin/env python3
"""Test web search through the full agent tool system."""

import sys
import io

if sys.platform == 'win32':
    sys.stdout = io.TextIOWrapper(sys.stdout.buffer, encoding='utf-8', errors='replace')

print('=== Testing Full Agent Web Search ===')
print()

# Test web search directly through WebSearchTool
from apprentice_agent.tools import WebSearchTool
tool = WebSearchTool()

print(f'Primary instance: {tool.PRIMARY_INSTANCE}')
print(f'Fallback instances: {tool.FALLBACK_INSTANCES}')
print()

# Execute web search
result = tool.search('latest AI news February 2026', num_results=3)
print('Web Search Result:')
print(f'  Success: {result.get("success")}')
print(f'  Source: {result.get("source", "N/A")}')
if result.get('results'):
    for i, r in enumerate(result['results'][:3], 1):
        title = r.get("title", "No title")[:70]
        url = r.get("url", "No URL")[:80]
        print(f'  {i}. {title}')
        print(f'     {url}')
else:
    print(f'  Error: {result.get("error", "No results")}')

print()
print('=== Test Complete ===')
