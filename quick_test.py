#!/usr/bin/env python3
"""Quick test for web search and crypto tools."""

import sys
import io

if sys.platform == 'win32':
    sys.stdout = io.TextIOWrapper(sys.stdout.buffer, encoding='utf-8', errors='replace')

print("Testing Web Search...")
from apprentice_agent.tools.web_search import WebSearchTool
t = WebSearchTool()
r = t.search('bitcoin price', 3)
print(f"  Success: {r.get('success')}")
print(f"  Source: {r.get('source', 'N/A')}")
print(f"  Results: {r.get('num_results', 0)}")
if r.get('results'):
    for x in r['results'][:3]:
        print(f"    - {x['title'][:60]}")
else:
    print(f"  Error: {r.get('error', 'Unknown')}")

print()
print("Testing Crypto Price...")
from apprentice_agent.tools.crypto_price import CryptoPriceTool
c = CryptoPriceTool()
r = c.get_price('bitcoin')
if r.get('success'):
    print(f"  Bitcoin: {r['price_formatted']} ({r['change_24h_formatted']})")
else:
    print(f"  Error: {r.get('error')}")

print()
print("Done!")
