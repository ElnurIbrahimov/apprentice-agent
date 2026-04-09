"""Comprehensive stress test for Aura brain — 100 messages across all query types.

Tests:
- Conversation continuity (doesn't die after N messages)
- Different query types (simple, complex, code, search, reasoning)
- Response quality (non-empty, reasonable length)
- Timeout handling
- History management
- System prompt size
- Background task resilience
- Rapid-fire messages
- Edge cases (empty, very long, special chars)
"""

import sys
import os
import time
import threading
import traceback

import pytest

# Requires a live Ollama instance — skip in automated CI.
# Guard the import to prevent collection errors when Ollama is not running.
if os.environ.get("AURA_STRESS_TESTS") != "1":
    pytest.skip("Stress test requires live Ollama (set AURA_STRESS_TESTS=1)", allow_module_level=True)

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

# Suppress noisy logs during test
import logging
logging.basicConfig(level=logging.WARNING, format="%(levelname)s %(name)s: %(message)s")
# But keep brain logs at INFO so we can see issues
logging.getLogger("aura.brain").setLevel(logging.INFO)

from aura.brain import OllamaBrain, _BG_EXECUTOR

# ============================================================================
# TEST MESSAGE SETS
# ============================================================================

# Category 1: Simple greetings / one-liners (should be fast)
SIMPLE = [
    "hi",
    "hello",
    "how are you?",
    "what's your name?",
    "thanks",
    "ok",
    "yes",
    "tell me a joke",
    "what time is it?",
    "good morning",
]

# Category 2: Normal conversation (medium complexity)
CONVERSATION = [
    "What's the difference between Python and JavaScript?",
    "Can you explain what an API is in simple terms?",
    "What are the best practices for writing clean code?",
    "How does a database work?",
    "What's the difference between SQL and NoSQL?",
    "Explain what Docker does and why people use it",
    "What is version control and why is it important?",
    "How do web servers work?",
    "What's the difference between frontend and backend?",
    "Explain what cloud computing means",
]

# Category 3: Complex / analytical queries
COMPLEX = [
    "Compare React, Vue, and Svelte. Which should I learn first and why?",
    "Explain the CAP theorem and its implications for distributed systems",
    "What are microservices? How do they compare to monolithic architecture?",
    "Describe the key differences between REST and GraphQL APIs",
    "What is event-driven architecture and when should you use it?",
    "Explain SOLID principles with examples",
    "How does garbage collection work in different programming languages?",
    "What are design patterns? Name the most useful ones",
    "Explain the difference between concurrency and parallelism",
    "What is the actor model and how does Erlang/Elixir use it?",
]

# Category 4: Code-related queries
CODE = [
    "Write a Python function to check if a string is a palindrome",
    "How do I sort a list of dictionaries by a specific key in Python?",
    "What's the difference between a list and a tuple in Python?",
    "Write a simple REST API endpoint in FastAPI",
    "How do I handle errors properly in async Python code?",
    "Explain Python decorators with an example",
    "What are Python generators and when should I use them?",
    "How do I use list comprehensions effectively?",
    "Write a function that finds duplicates in a list",
    "Explain the GIL in Python and its implications",
]

# Category 5: Follow-up conversation (tests history continuity)
FOLLOWUP = [
    "Let's talk about machine learning",
    "What are the main types of ML?",
    "Can you go deeper on supervised learning?",
    "What about neural networks specifically?",
    "How does backpropagation work?",
    "What's the difference between CNN and RNN?",
    "Going back to what you said about supervised learning - what are common algorithms?",
    "Which of those algorithms is best for classification?",
    "Can you summarize everything we discussed about ML?",
    "One more thing - what's transfer learning?",
]

# Category 6: Edge cases
EDGE_CASES = [
    "",  # empty
    " ",  # whitespace only
    "a",  # single char
    "?" * 100,  # repeated chars
    "Hello! 🌍🎉🚀 How are you doing today? 😊",  # emojis
    "SELECT * FROM users WHERE 1=1; DROP TABLE users;--",  # SQL injection attempt
    "<script>alert('xss')</script>",  # XSS attempt
    "a " * 500,  # very long (1000 chars)
    "What is " + "very " * 50 + "important?",  # long but valid
    "tell me about " + ", ".join([f"topic{i}" for i in range(50)]),  # many topics
]

# Category 7: Rapid-fire (same message repeated - tests rate limiting)
RAPID_FIRE = ["quick question: what is 2+2?"] * 10

# Category 8: Context-dependent (tests memory across messages)
CONTEXT_CHAIN = [
    "My name is TestUser and I live in Baku",
    "What's my name?",
    "Where do I live?",
    "I'm working on a project called TestProject using Python and FastAPI",
    "What project am I working on?",
    "What tech stack did I mention?",
    "Recommend a database for my project",
    "Should I deploy on AWS or GCP for my use case?",
    "Summarize what you know about me and my project",
    "Thanks for the help!",
]

ALL_TESTS = [
    ("Simple", SIMPLE),
    ("Conversation", CONVERSATION),
    ("Complex", COMPLEX),
    ("Code", CODE),
    ("Follow-up", FOLLOWUP),
    ("Edge Cases", EDGE_CASES),
    ("Rapid Fire", RAPID_FIRE),
    ("Context Chain", CONTEXT_CHAIN),
]

# ============================================================================
# TEST RUNNER
# ============================================================================

class TestResult:
    def __init__(self, category, index, prompt, response, elapsed, error=None):
        self.category = category
        self.index = index
        self.prompt = prompt[:80]
        self.response = response
        self.elapsed = elapsed
        self.error = error

    @property
    def success(self):
        if self.error:
            return False
        if not self.response or not self.response.strip():
            return False
        if self.response.strip() == "I'm having trouble processing that right now. Please try again.":
            return False
        return True

    @property
    def status(self):
        if self.error:
            return "ERROR"
        if not self.response or not self.response.strip():
            return "EMPTY"
        if "trouble processing" in (self.response or ""):
            return "TIMEOUT"
        return "OK"


def run_stress_test():
    print("=" * 70)
    print("  AURA BRAIN STRESS TEST — 100 Messages")
    print("=" * 70)

    # Initialize brain
    print("\n[INIT] Creating OllamaBrain (warmup=False for speed)...")
    t0 = time.time()
    try:
        brain = OllamaBrain(warmup=False)
    except Exception as e:
        print(f"[FATAL] Cannot create OllamaBrain: {e}")
        return
    print(f"[INIT] Brain ready in {time.time()-t0:.1f}s")
    print(f"[INIT] Model: {brain.model}")
    print(f"[INIT] History limit: {brain._max_history}")

    results = []
    total_messages = 0
    total_start = time.time()

    for category_name, messages in ALL_TESTS:
        print(f"\n{'─'*60}")
        print(f"  Category: {category_name} ({len(messages)} messages)")
        print(f"{'─'*60}")

        for i, msg in enumerate(messages):
            total_messages += 1
            display_msg = msg[:60].replace('\n', ' ') if msg else "(empty)"

            sys.stdout.write(f"  [{total_messages:3d}] {display_msg:60s} ... ")
            sys.stdout.flush()

            start = time.time()
            response = None
            error = None

            try:
                # Use think() for non-streaming test (same core path)
                response = brain.think(msg, use_history=True)
            except Exception as e:
                error = str(e)
                traceback.print_exc()

            elapsed = time.time() - start
            result = TestResult(category_name, i, msg, response, elapsed, error)
            results.append(result)

            # Print result
            resp_preview = ""
            if response:
                resp_preview = response[:40].replace('\n', ' ')

            if result.status == "OK":
                print(f"{result.status:7s} {elapsed:5.1f}s  [{len(response):4d} chars] {resp_preview}")
            else:
                print(f"{result.status:7s} {elapsed:5.1f}s  {error or '(no response)'}")

            # Brief pause between messages to simulate real usage
            # (but not for rapid-fire category)
            if category_name != "Rapid Fire":
                time.sleep(0.3)

    total_elapsed = time.time() - total_start

    # ========================================================================
    # RESULTS SUMMARY
    # ========================================================================
    print(f"\n{'='*70}")
    print("  RESULTS SUMMARY")
    print(f"{'='*70}")

    ok_count = sum(1 for r in results if r.status == "OK")
    empty_count = sum(1 for r in results if r.status == "EMPTY")
    timeout_count = sum(1 for r in results if r.status == "TIMEOUT")
    error_count = sum(1 for r in results if r.status == "ERROR")
    total = len(results)

    print(f"\n  Total messages:  {total}")
    print(f"  OK:              {ok_count:3d} ({ok_count/total*100:.0f}%)")
    print(f"  Empty:           {empty_count:3d} ({empty_count/total*100:.0f}%)")
    print(f"  Timeout:         {timeout_count:3d} ({timeout_count/total*100:.0f}%)")
    print(f"  Error:           {error_count:3d} ({error_count/total*100:.0f}%)")
    print(f"  Total time:      {total_elapsed:.1f}s ({total_elapsed/total:.1f}s avg)")

    # Per-category breakdown
    print("\n  Per Category:")
    for cat_name, _ in ALL_TESTS:
        cat_results = [r for r in results if r.category == cat_name]
        cat_ok = sum(1 for r in cat_results if r.status == "OK")
        cat_avg = sum(r.elapsed for r in cat_results) / len(cat_results) if cat_results else 0
        cat_total = len(cat_results)
        status = "PASS" if cat_ok == cat_total else f"FAIL ({cat_total - cat_ok} failures)"
        print(f"    {cat_name:20s}  {cat_ok}/{cat_total}  avg {cat_avg:.1f}s  {status}")

    # Response time distribution
    times = sorted(r.elapsed for r in results)
    print("\n  Response times:")
    print(f"    Min:     {times[0]:.1f}s")
    print(f"    Median:  {times[len(times)//2]:.1f}s")
    print(f"    P90:     {times[int(len(times)*0.9)]:.1f}s")
    print(f"    P99:     {times[int(len(times)*0.99)]:.1f}s")
    print(f"    Max:     {times[-1]:.1f}s")

    # Check history state
    print("\n  Brain state:")
    print(f"    History length:     {len(brain.conversation_history)}")
    print(f"    Query count:        {brain._query_count}")
    print(f"    Total query count:  {brain._total_query_count}")

    # Check background executor health
    print("\n  Background executor:")
    pending = _BG_EXECUTOR._work_queue.qsize() if hasattr(_BG_EXECUTOR, '_work_queue') else '?'
    print(f"    Pending tasks: {pending}")

    # System prompt size check
    try:
        sys_prompt = brain._build_full_system_prompt("test", None, None)
        print(f"    System prompt size: {len(sys_prompt)} chars")
    except Exception as e:
        print(f"    System prompt size: ERROR ({e})")

    # Failures detail
    failures = [r for r in results if not r.success]
    if failures:
        print(f"\n  FAILURES ({len(failures)}):")
        for r in failures:
            print(f"    [{r.category}#{r.index}] {r.status}: {r.prompt[:50]}  error={r.error or 'empty response'}")

    # Final verdict
    print(f"\n{'='*70}")
    success_rate = ok_count / total * 100
    if success_rate >= 95:
        verdict = "EXCELLENT"
    elif success_rate >= 85:
        verdict = "GOOD"
    elif success_rate >= 70:
        verdict = "NEEDS WORK"
    else:
        verdict = "FAILING"
    print(f"  VERDICT: {verdict} ({success_rate:.0f}% success rate)")
    # Edge cases with empty input are expected to potentially fail
    [r for r in failures if r.prompt.strip()]
    real_total = sum(1 for r in results if r.prompt.strip())
    real_rate = sum(1 for r in results if r.success and r.prompt.strip()) / real_total * 100 if real_total else 0
    print(f"  (Excluding empty-input edge cases: {real_rate:.0f}% success)")
    print(f"{'='*70}")


if __name__ == "__main__":
    run_stress_test()
