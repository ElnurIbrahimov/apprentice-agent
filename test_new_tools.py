#!/usr/bin/env python3
"""Test DuckDuckGo web search and CoinGecko crypto price tools."""

import sys
import io

# Fix Windows Unicode
if sys.platform == 'win32':
    sys.stdout = io.TextIOWrapper(sys.stdout.buffer, encoding='utf-8', errors='replace')
    sys.stderr = io.TextIOWrapper(sys.stderr.buffer, encoding='utf-8', errors='replace')

print("=" * 60)
print("TESTING TOOLS: DuckDuckGo Search + CoinGecko Crypto Price")
print("=" * 60)
print()

# Test 1: Crypto Price (CoinGecko)
print("TEST 1: Crypto Price Tool (CoinGecko API)")
print("-" * 40)

try:
    from apprentice_agent.tools.crypto_price import CryptoPriceTool

    crypto_tool = CryptoPriceTool()

    # Test Bitcoin
    print("Getting Bitcoin price...")
    result = crypto_tool.get_price("bitcoin")

    if result.get("success"):
        print(f"  Bitcoin: {result['price_formatted']}")
        print(f"  24h Change: {result['change_24h_formatted']}")
        print(f"  Market Cap: {result['market_cap_formatted']}")
        print("  [PASS] Bitcoin price retrieved!")
    else:
        print(f"  [FAIL] Error: {result.get('error')}")

    print()

    # Test Ethereum
    print("Getting Ethereum price...")
    result = crypto_tool.get_price("eth")

    if result.get("success"):
        print(f"  Ethereum: {result['price_formatted']}")
        print(f"  24h Change: {result['change_24h_formatted']}")
        print("  [PASS] Ethereum price retrieved!")
    else:
        print(f"  [FAIL] Error: {result.get('error')}")

except Exception as e:
    print(f"  [FAIL] Crypto tool error: {e}")

print()
print()

# Test 2: Web Search (DuckDuckGo)
print("TEST 2: Web Search Tool (DuckDuckGo)")
print("-" * 40)

try:
    from apprentice_agent.tools.web_search import WebSearchTool

    search_tool = WebSearchTool()

    # Test search
    print("Searching for 'Python programming'...")
    result = search_tool.search("Python programming", num_results=3)

    if result.get("success"):
        print(f"  Source: {result['source']}")
        print(f"  Results: {result['num_results']}")
        for i, r in enumerate(result["results"][:3], 1):
            title = r["title"][:50] + "..." if len(r["title"]) > 50 else r["title"]
            print(f"  {i}. {title}")
        print("  [PASS] Web search working!")
    else:
        print(f"  [FAIL] Error: {result.get('error')}")

    print()

    # Test Bitcoin price search
    print("Searching for 'bitcoin price'...")
    result = search_tool.search("bitcoin current price", num_results=3)

    if result.get("success"):
        print(f"  Results: {result['num_results']}")
        for i, r in enumerate(result["results"][:3], 1):
            title = r["title"][:50] + "..." if len(r["title"]) > 50 else r["title"]
            print(f"  {i}. {title}")
        print("  [PASS] Bitcoin search working!")
    else:
        print(f"  [FAIL] Error: {result.get('error')}")

except Exception as e:
    print(f"  [FAIL] Web search error: {e}")

print()
print("=" * 60)
print("TOOL TESTS COMPLETE")
print("=" * 60)
