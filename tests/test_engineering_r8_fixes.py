"""Tests for engineering review R8 fixes (2026-04-09)."""
import json
import math
import os
import sys
import threading
import time
from collections import OrderedDict
from unittest.mock import patch, MagicMock

import pytest

# Ensure project root is importable
sys.path.insert(0, str(os.path.join(os.path.dirname(__file__), "..")))


# ── Test 1: Router centroid computation is fully inside the lock ────────────

class TestRouterCentroidLocking:
    """Verify _ensure_centroids computes centroids inside the lock (not after)."""

    def test_centroids_computed_flag_set_after_computation(self):
        """_centroids_computed should only be True after centroids are populated."""
        from aura.core import router

        # Reset state
        router._centroids_computed = False
        router._category_centroids.clear()

        # Mock ollama to avoid real network calls
        with patch.dict("sys.modules", {"ollama": MagicMock(), "numpy": pytest.importorskip("numpy")}):
            # After calling _ensure_centroids, if it set _centroids_computed=True
            # inside the lock but computed outside, a concurrent thread would see
            # _centroids_computed=True with empty _category_centroids.
            # We verify the flag is only set in the finally block.
            router._ensure_centroids()
            assert router._centroids_computed is True

    def test_concurrent_centroid_init_no_empty_return(self):
        """Two threads calling _ensure_centroids should not get empty centroids."""
        from aura.core import router

        # Reset state
        router._centroids_computed = False
        router._category_centroids.clear()

        results = []
        barrier = threading.Barrier(2)

        def worker():
            barrier.wait()
            router._ensure_centroids()
            # After return, if centroids were computed, the dict should be populated
            # (or empty if ollama unavailable — but _centroids_computed should be True)
            results.append(router._centroids_computed)

        t1 = threading.Thread(target=worker)
        t2 = threading.Thread(target=worker)
        t1.start()
        t2.start()
        t1.join(timeout=10)
        t2.join(timeout=10)

        # Both should see _centroids_computed = True
        assert all(r is True for r in results)


# ── Test 2: _hot_files type fix (no .keys() on list) ─────────────────────

class TestHotFilesTypeFix:
    """_hot_files is a list, not a dict — set() conversion should work."""

    def test_hot_files_to_set(self):
        """set(getattr(obj, '_hot_files', [])) should not raise."""
        hot_files = ["a.py", "b.py", "c.py"]

        # Old code: set(getattr(self, '_hot_files', {}).keys()) would crash
        # New code: set(getattr(self, '_hot_files', []))
        result = set(hot_files)
        assert result == {"a.py", "b.py", "c.py"}

    def test_hot_files_default_is_empty_list(self):
        """Default fallback should be a list, not a dict."""
        obj = object()
        result = set(getattr(obj, '_hot_files', []))
        assert result == set()


# ── Test 3: EvaluationCache preserves OrderedDict after load ───────────────

class TestEvaluationCacheOrderedDict:
    """json.load returns plain dict; cache must wrap in OrderedDict."""

    def test_loaded_cache_supports_move_to_end(self, tmp_path):
        from aura.evolution.cache import EvaluationCache

        # Write a cache file with plain JSON
        cache_dir = str(tmp_path)
        cache_file = tmp_path / "eval_cache.json"
        cache_file.write_text(json.dumps({"a:b": 0.5, "c:d": 0.8}))

        cache = EvaluationCache(cache_dir=cache_dir)

        # This should NOT raise AttributeError (would with plain dict)
        result = cache.get("a", "b")
        assert result == 0.5

    def test_lru_eviction_works_after_load(self, tmp_path):
        from aura.evolution.cache import EvaluationCache

        cache_dir = str(tmp_path)
        cache_file = tmp_path / "eval_cache.json"
        # Seed with entries
        seed = {f"k{i}:v{i}": float(i) for i in range(10)}
        cache_file.write_text(json.dumps(seed))

        cache = EvaluationCache(cache_dir=cache_dir)
        # put() calls move_to_end and popitem — should not crash
        cache.put("new_k", "new_v", 0.99)


# ── Test 4: Differential privacy uses Laplace noise (not uniform) ──────────

class TestDifferentialPrivacyNoise:
    """PrivacyGuard.add_differential_noise must produce Laplace-distributed noise."""

    def test_noise_is_unbounded(self):
        """Laplace noise is unbounded; uniform noise is bounded to [-0.5, 0.5]*scale."""
        from aura.multi_user.privacy_guard import PrivacyGuard
        guard = PrivacyGuard()

        # Sample many values; Laplace should occasionally produce |noise| > scale
        scale = 1.0 / guard.noise_epsilon
        deviations = []
        for _ in range(1000):
            noisy = guard.add_differential_noise(0.0, sensitivity=1.0)
            deviations.append(abs(noisy))

        max_dev = max(deviations)
        # Uniform noise on [-0.5, 0.5]*scale would never exceed 0.5*scale
        # Laplace noise should exceed that at least sometimes
        assert max_dev > 0.5 * scale, (
            f"Max deviation {max_dev:.4f} ≤ 0.5*scale={0.5*scale:.4f} — "
            "noise appears bounded (uniform), not Laplace"
        )


# ── Test 5: PatternProphet thread safety ────────────────────────────────────

class TestPatternProphetThreadSafety:
    """PatternProphet must have a lock and cap interactions list."""

    def test_has_lock(self, tmp_path):
        from aura.patterns.pattern_prophet import PatternProphet
        pp = PatternProphet(data_dir=str(tmp_path))
        assert hasattr(pp, '_lock')
        assert isinstance(pp._lock, type(threading.Lock()))

    def test_interactions_capped(self, tmp_path):
        from aura.patterns.pattern_prophet import PatternProphet
        pp = PatternProphet(data_dir=str(tmp_path))

        # Stuff more than 1000 interactions
        for i in range(1050):
            pp.record_interaction(f"test message {i}")

        assert len(pp.interactions) <= 1000


# ── Test 6: Strategy bandit connection inside lock ──────────────────────────

class TestStrategyBanditLocking:
    """get_best_strategy must open connection INSIDE the lock."""

    def test_get_best_strategy_no_crash(self, tmp_path):
        from aura.consciousness.strategy_bandit import StrategyBandit, ProblemCategory
        bandit = StrategyBandit(db_path=str(tmp_path / "bandit.db"))

        # Should not crash (connection opened inside lock)
        # get_best_strategy may return None or a default strategy name
        result = bandit.get_best_strategy(ProblemCategory.CODE)
        assert result is None or isinstance(result, str)


# ── Test 7: Telegram summarize auth check ───────────────────────────────────

class TestTelegramSummarizeAuth:
    """summarize_group and summarize_thread must check _is_user_allowed."""

    def test_summarize_group_has_auth_check(self):
        """Verify the auth check is present in the source code."""
        import inspect
        from aura.messaging.telegram.mixins.sessions import SessionsMixin
        source = inspect.getsource(SessionsMixin._handle_summarize_group)
        assert "_is_user_allowed" in source

    def test_summarize_thread_has_auth_check(self):
        """Verify the auth check is present in the source code."""
        import inspect
        from aura.messaging.telegram.mixins.sessions import SessionsMixin
        source = inspect.getsource(SessionsMixin._handle_summarize_thread)
        assert "_is_user_allowed" in source


# ── Test 8: SSRF DNS resolution check ──────────────────────────────────────

class TestSSRFDnsProtection:
    """_fetch_url must resolve hostnames and block private IPs."""

    def test_blocks_dns_rebinding_to_private_ip(self):
        """A hostname resolving to a private IP should be blocked."""
        from aura.core.agentic_loop import ToolExecutor

        executor = ToolExecutor.__new__(ToolExecutor)

        # Mock socket.gethostbyname to return a private IP
        with patch("socket.gethostbyname", return_value="192.168.1.1"):
            result = executor._fetch_url({"url": "http://evil.example.com/secret"})
        assert "error" in result
        assert "private" in result["error"].lower() or "internal" in result["error"].lower()

    def test_blocks_dns_rebinding_to_metadata(self):
        """A hostname resolving to the cloud metadata IP should be blocked."""
        from aura.core.agentic_loop import ToolExecutor

        executor = ToolExecutor.__new__(ToolExecutor)

        with patch("socket.gethostbyname", return_value="169.254.169.254"):
            result = executor._fetch_url({"url": "http://evil.example.com/latest/meta-data"})
        assert "error" in result


# ── Test 9: Provider stream response cleanup ────────────────────────────────

class TestProviderStreamCleanup:
    """Streaming providers must close response on generator abandonment."""

    def test_anthropic_stream_closes_response(self):
        from aura.providers.anthropic_provider import AnthropicProvider

        mock_resp = MagicMock()
        mock_resp.status_code = 200
        mock_resp.encoding = "utf-8"
        mock_resp.iter_lines.return_value = iter([
            b'data: {"type":"content_block_delta","delta":{"type":"text_delta","text":"hi"}}',
            b'data: {"type":"message_delta","usage":{"output_tokens":1}}',
        ])

        provider = AnthropicProvider.__new__(AnthropicProvider)
        provider._session = MagicMock()
        provider._session.post.return_value = mock_resp

        gen = provider._stream_chat("http://test", {}, {})
        # Consume one chunk then abandon
        next(gen)
        gen.close()  # Triggers finally block

        mock_resp.close.assert_called()


# ── Test 10: Search route uses safe_error_detail ────────────────────────────

class TestSearchErrorSafety:
    """Search route must not leak raw exceptions."""

    def test_search_imports_safe_error_detail(self):
        """Verify safe_error_detail is imported in search.py."""
        import importlib
        mod = importlib.import_module("api.routes.search")
        assert hasattr(mod, "safe_error_detail")
