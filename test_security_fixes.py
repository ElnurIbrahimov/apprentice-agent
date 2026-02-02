"""Test security fixes applied to AURA."""

import sys
sys.path.insert(0, '.')

def test_code_executor_ast_safety():
    """Test that code executor blocks bypass attempts."""
    from apprentice_agent.tools.code_executor import CodeExecutorTool

    executor = CodeExecutorTool()

    # Test 1: Direct import block
    result = executor.execute("import os")
    assert not result.get("success"), "Should block 'import os'"
    print("  [PASS] Blocks 'import os'")

    # Test 2: Exec bypass attempt
    result = executor.execute("exec('import os')")
    assert not result.get("success"), "Should block exec()"
    print("  [PASS] Blocks exec() bypass")

    # Test 3: __import__ bypass
    result = executor.execute("__import__('os')")
    assert not result.get("success"), "Should block __import__"
    print("  [PASS] Blocks __import__ bypass")

    # Test 4: Attribute access bypass
    result = executor.execute("x = ().__class__.__bases__[0]")
    assert not result.get("success"), "Should block __class__ access"
    print("  [PASS] Blocks __class__ access")

    # Test 5: Valid code should work
    result = executor.execute("print(2 + 2)")
    assert result.get("success"), "Valid code should work"
    print("  [PASS] Valid code executes")

    return True


def test_input_validation():
    """Test input validation."""
    from apprentice_agent.tools.validation import validate_path, validate_query

    # Test null byte blocking
    try:
        validate_path("/tmp/test\x00.txt")
        assert False, "Should block null bytes"
    except ValueError:
        print("  [PASS] Blocks null bytes in paths")

    # Test empty query
    try:
        validate_query("")
        assert False, "Should block empty queries"
    except ValueError:
        print("  [PASS] Blocks empty queries")

    # Test valid inputs
    assert validate_path("test.txt") == "test.txt"
    print("  [PASS] Valid path accepted")

    assert validate_query("hello world") == "hello world"
    print("  [PASS] Valid query accepted")

    return True


def test_config_thread_safety():
    """Test thread-safe config."""
    from apprentice_agent.config import Config

    # Test getter
    model = Config.get_model('fast')
    assert model is not None
    print(f"  [PASS] Thread-safe getter: {model}")

    # Test all models
    models = Config.get_all_models()
    assert 'fast' in models
    assert 'reason' in models
    print(f"  [PASS] All models accessible: {len(models)} configs")

    return True


def test_secure_logging():
    """Test log sanitization."""
    from apprentice_agent.secure_logging import sanitize_text

    # Test password redaction
    text = "user password=secret123 logged in"
    sanitized = sanitize_text(text)
    assert "secret123" not in sanitized
    assert "[REDACTED]" in sanitized or "REDACTED" in sanitized.upper()
    print("  [PASS] Passwords redacted")

    # Test API key redaction
    text = "api_key=sk-1234567890abcdef"
    sanitized = sanitize_text(text)
    assert "sk-1234567890" not in sanitized
    print("  [PASS] API keys redacted")

    # Test token redaction
    text = "bearer eyJhbGciOiJIUzI1NiJ9.test"
    sanitized = sanitize_text(text)
    assert "eyJhbGciOiJIUzI1NiJ9" not in sanitized
    print("  [PASS] Bearer tokens redacted")

    return True


def test_web_search_rate_limiting():
    """Test web search has rate limiting."""
    from apprentice_agent.tools.web_search import WebSearchTool, _rate_limiter

    # Check rate limiter exists
    assert _rate_limiter is not None
    assert _rate_limiter.calls_per_minute == 30
    print(f"  [PASS] Rate limiter configured: {_rate_limiter.calls_per_minute} calls/min")

    # Test validation
    search = WebSearchTool()
    result = search.search("", num_results=5)  # Empty query
    assert not result.get("success")
    print("  [PASS] Empty query rejected")

    return True


if __name__ == "__main__":
    print("=" * 50)
    print("AURA Security Fixes Test Suite")
    print("=" * 50)
    print()

    tests = [
        ("Code Executor AST Safety", test_code_executor_ast_safety),
        ("Input Validation", test_input_validation),
        ("Config Thread Safety", test_config_thread_safety),
        ("Secure Logging", test_secure_logging),
        ("Web Search Rate Limiting", test_web_search_rate_limiting),
    ]

    passed = 0
    failed = 0

    for name, test_func in tests:
        print(f"\n[TEST] {name}")
        try:
            if test_func():
                passed += 1
                print(f"  [OK] {name} PASSED")
        except Exception as e:
            failed += 1
            print(f"  [FAIL] {name} FAILED: {e}")

    print()
    print("=" * 50)
    print(f"Results: {passed} passed, {failed} failed")
    print("=" * 50)
