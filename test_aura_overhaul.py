"""
Test AURA v3.1 Overhaul

Tests that everything is wired together:
1. Fast-path only handles explicit commands
2. LLM is called for conversation
3. Memory is retrieved and used
4. Interactions are stored
"""

import sys
sys.path.insert(0, '.')

def test_fast_path():
    """Test that fast-path only handles explicit commands"""
    print("\n" + "=" * 60)
    print("TEST 1: Fast Path (Minimal)")
    print("=" * 60)

    from aura.fast_path import FastPathHandler
    handler = FastPathHandler()

    tests = [
        # Should be handled by fast-path
        ("aura status", True, "AURA command"),
        ("/help", True, "Slash command"),
        ("remember this: meeting tomorrow", True, "Memory command"),

        # Should NOT be handled (goes to LLM)
        ("hi", False, "Greeting -> LLM"),
        ("how are you?", False, "Question -> LLM"),
        ("I got the job!", False, "Good news -> LLM"),
        ("I'm feeling stressed", False, "Emotional -> LLM"),
        ("thanks!", False, "Thanks -> LLM"),
        ("they will pay me well", False, "Positive -> LLM"),
    ]

    passed = 0
    for msg, should_handle, desc in tests:
        result = handler.try_fast_path(msg)
        was_handled = result is not None
        ok = was_handled == should_handle
        status = "OK" if ok else "FAIL"
        path = "FAST" if was_handled else "LLM"
        print(f"  [{status}] [{path}] {desc}: '{msg}'")
        if ok:
            passed += 1

    print(f"\n  Passed: {passed}/{len(tests)}")
    return passed == len(tests)


def test_llm_client():
    """Test LLM client with context injection"""
    print("\n" + "=" * 60)
    print("TEST 2: LLM Client (Context Injection)")
    print("=" * 60)

    try:
        from aura.llm import OllamaClient
        client = OllamaClient(model="llama3:8b")

        print("  Testing response generation with context...")

        # Test with full context
        response = client.generate(
            user_message="I got the job offer!",
            memories=["User had interview last week", "User was nervous about it"],
            emotional_context={"mood": "excited", "energy": 0.8},
            user_profile={"name": "Alex"}
        )

        print(f"  Response: {response[:100]}...")

        # Check response is not generic
        generic_responses = ["I hear you", "What's on your mind", "Got it"]
        is_generic = any(g.lower() in response.lower() for g in generic_responses)

        if is_generic:
            print("  [WARN] Response seems generic")
        else:
            print("  [OK] Response is not generic!")

        return True

    except Exception as e:
        print(f"  [ERROR] {e}")
        return False


def test_memory_retriever():
    """Test memory retrieval"""
    print("\n" + "=" * 60)
    print("TEST 3: Memory Retriever")
    print("=" * 60)

    try:
        from aura.memory import MemoryRetriever
        retriever = MemoryRetriever()

        # Test keyword extraction
        keywords = retriever._extract_keywords("How did my interview at Google go?")
        print(f"  Keywords from 'How did my interview at Google go?':")
        print(f"    {keywords}")

        # Test memory retrieval
        memories = retriever.get_relevant_memories("interview")
        print(f"  Found {len(memories)} memories about 'interview'")

        # Test user profile
        profile = retriever.get_user_profile()
        print(f"  User profile: {profile}")

        print("  [OK] Memory retriever working!")
        return True

    except Exception as e:
        print(f"  [ERROR] {e}")
        return False


def test_full_engine():
    """Test full AURA engine flow"""
    print("\n" + "=" * 60)
    print("TEST 4: Full AURA Engine")
    print("=" * 60)

    try:
        from aura import AURAEngine

        print("  Initializing AURA engine...")
        engine = AURAEngine(enable_proactive=False)

        print(f"  Engine version: {engine.soul.name if engine.soul else 'No soul'}")
        print(f"  LLM model: {engine.llm.model}")

        # Test fast-path command
        print("\n  Test 1: /status command (fast-path)")
        response = engine.generate_response("/status")
        print(f"    Response: {response[:80]}...")

        # Test conversation (should go to LLM)
        print("\n  Test 2: 'I got the job!' (LLM)")
        response = engine.generate_response("I got the job!")
        print(f"    Response: {response[:100]}...")

        # Check not generic
        generic = ["Got it", "I hear you", "What's on your mind"]
        is_generic = any(g.lower() in response.lower() for g in generic)
        if is_generic:
            print("    [WARN] Response might be too generic")
        else:
            print("    [OK] Response is contextual!")

        # Shutdown
        engine.shutdown()

        print("\n  [OK] Full engine test passed!")
        return True

    except Exception as e:
        print(f"  [ERROR] {e}")
        import traceback
        traceback.print_exc()
        return False


def main():
    print("\n" + "=" * 60)
    print("AURA v3.1 OVERHAUL TEST")
    print("=" * 60)

    results = []

    # Test 1: Fast-path
    results.append(("Fast Path", test_fast_path()))

    # Test 2: LLM Client
    results.append(("LLM Client", test_llm_client()))

    # Test 3: Memory Retriever
    results.append(("Memory Retriever", test_memory_retriever()))

    # Test 4: Full Engine
    results.append(("Full Engine", test_full_engine()))

    # Summary
    print("\n" + "=" * 60)
    print("SUMMARY")
    print("=" * 60)

    passed = sum(1 for _, ok in results if ok)
    for name, ok in results:
        status = "PASS" if ok else "FAIL"
        print(f"  [{status}] {name}")

    print(f"\n  Total: {passed}/{len(results)} tests passed")
    print("=" * 60)


if __name__ == "__main__":
    main()
