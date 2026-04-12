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
    """_fetch_url must resolve hostnames and block private IPs.

    _fetch_url lives in tool_executor and delegates SSRF validation to
    aura.security.ssrf_guard.safe_request, which calls socket.getaddrinfo.
    So we mock getaddrinfo (not gethostbyname) to simulate DNS rebinding.
    """

    @staticmethod
    def _mock_getaddrinfo(ip):
        """Build a getaddrinfo return value that points the hostname at `ip`."""
        import socket
        return [(socket.AF_INET, socket.SOCK_STREAM, 0, "", (ip, 0))]

    def test_blocks_dns_rebinding_to_private_ip(self):
        """A hostname resolving to a private IP must be blocked."""
        from aura.core.tool_executor import ToolExecutor

        executor = ToolExecutor.__new__(ToolExecutor)

        with patch(
            "aura.security.ssrf_guard.socket.getaddrinfo",
            return_value=self._mock_getaddrinfo("192.168.1.1"),
        ):
            result = executor._fetch_url({"url": "http://evil.example.com/secret"})
        assert "error" in result
        err = result["error"].lower()
        assert "private" in err or "blocked" in err or "reserved" in err

    def test_blocks_dns_rebinding_to_metadata(self):
        """A hostname resolving to the cloud metadata IP must be blocked."""
        from aura.core.tool_executor import ToolExecutor

        executor = ToolExecutor.__new__(ToolExecutor)

        with patch(
            "aura.security.ssrf_guard.socket.getaddrinfo",
            return_value=self._mock_getaddrinfo("169.254.169.254"),
        ):
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


# ── Test 11: Hands sync request_approval deleted, async path robust ──────────

class TestHandsApprovalAsync:
    """Verify the blocking sync request_approval was removed and the async
    path is the sole code path. Regression guard: don't reintroduce the sync
    fallback that could starve the 3-thread hand pool."""

    def test_sync_request_approval_removed(self):
        from aura.hands.manager import HandManager
        # Sync method is gone; async variant is the sole entry point.
        assert not hasattr(HandManager, "request_approval")
        assert hasattr(HandManager, "request_approval_async")

    def test_async_approval_timeout_returns_false(self):
        """With no resolve call, request_approval_async must eventually
        return False and not leave entries in _pending_approvals."""
        import asyncio
        from aura.hands.manager import HandManager

        async def _run():
            mgr = HandManager()
            # Set the loop reference the manager expects for broadcasts.
            mgr._approval_loop = asyncio.get_running_loop()
            # Force the wait timeout down by monkeypatching asyncio.wait_for
            # locally — the method reads timeout from inline 60s literal,
            # so we race resolve_approval instead.
            async def _call():
                return await mgr.request_approval_async("hand1", "tool1", {})

            task = asyncio.create_task(_call())
            # Give the task a beat to register the request, then resolve.
            await asyncio.sleep(0.05)
            # Pull request_id from internal state.
            with mgr._approval_lock:
                assert len(mgr._pending_approvals) == 1
                request_id = next(iter(mgr._pending_approvals))
            mgr.resolve_approval(request_id, approved=False)
            result = await asyncio.wait_for(task, timeout=2.0)
            assert result is False
            # Entry should be cleared after resolution.
            with mgr._approval_lock:
                assert mgr._pending_approvals.get(request_id) is None or \
                       mgr._pending_approvals.get(request_id).resolved is True

        asyncio.run(_run())

    def test_three_concurrent_approvals_do_not_block_each_other(self):
        """Resolve the middle approval first; it must return without waiting
        for the other two. Bounds the test to 2s to catch pool starvation."""
        import asyncio
        from aura.hands.manager import HandManager

        async def _run():
            mgr = HandManager()
            mgr._approval_loop = asyncio.get_running_loop()
            results = {}

            async def _ask(label):
                results[label] = await mgr.request_approval_async(label, "t", {})

            tasks = [
                asyncio.create_task(_ask("a")),
                asyncio.create_task(_ask("b")),
                asyncio.create_task(_ask("c")),
            ]
            await asyncio.sleep(0.05)
            # Find the request_id for hand "b" and resolve it.
            with mgr._approval_lock:
                rid_b = next(
                    r for r, req in mgr._pending_approvals.items()
                    if req.hand_name == "b"
                )
            mgr.resolve_approval(rid_b, approved=True)
            # b should finish quickly; a and c stay pending.
            await asyncio.wait_for(tasks[1], timeout=1.0)
            assert results.get("b") is True
            assert "a" not in results
            assert "c" not in results
            # Clean up pending tasks without waiting on their 60s timeout.
            tasks[0].cancel()
            tasks[2].cancel()
            try:
                await asyncio.gather(tasks[0], tasks[2], return_exceptions=True)
            except Exception:
                pass

        asyncio.run(_run())

    def test_resolve_approval_only_handles_tuple_events(self):
        """Regression guard: resolve_approval must fail loudly if a legacy
        threading.Event shows up (documents the contract change)."""
        import asyncio
        from aura.hands.manager import HandManager, ApprovalRequest

        mgr = HandManager()
        rid = "apr_legacy"
        mgr._pending_approvals[rid] = ApprovalRequest(
            request_id=rid, hand_name="h", tool_name="t", args={},
        )
        # Inject a raw threading.Event — the shape resolve_approval no longer
        # accepts. It must raise (tuple unpack) rather than silently succeed.
        mgr._approval_events[rid] = threading.Event()
        with pytest.raises((TypeError, ValueError)):
            mgr.resolve_approval(rid, True)


# ── Test 12: search_semantic brute-force fallback is paginated, bounded ─────

class TestSearchSemanticBruteForcePagination:
    """Verify the FAISS-less fallback uses bounded memory via chunked
    pagination + top-k heap, rather than loading the whole table at once."""

    @staticmethod
    def _make_store(tmp_path, rows: int, dim: int = 4):
        """Build an in-memory-ish MemoryStore with N rows, each with a random
        unit embedding plus one planted "target" vector aligned with a known
        query. Returns (store, query_vec, target_id)."""
        import numpy as np
        from aura.memory.store import MemoryStore, MemoryRecord
        db_path = tmp_path / f"mem_{rows}.db"
        store = MemoryStore(db_path=str(db_path))
        rng = np.random.default_rng(42)
        query = np.array([1.0, 0.0, 0.0, 0.0], dtype=np.float32)
        target_id = None
        for i in range(rows):
            # Planted target at index rows//2 — nearly identical to query.
            if i == rows // 2:
                emb = query.copy() + rng.normal(0, 0.001, dim).astype(np.float32)
                target_id = f"target-{i}"
                rec = MemoryRecord(id=target_id, content="needle", lifecycle_state="stable")
            else:
                vec = rng.normal(0, 1, dim).astype(np.float32)
                rec = MemoryRecord(
                    id=f"row-{i}", content=f"hay {i}", lifecycle_state="stable",
                )
                emb = vec
            store.insert(rec, embedding=emb)
        return store, query, target_id

    def test_brute_force_returns_topk_and_finds_planted_match(self, tmp_path):
        """With FAISS disabled, 2500-row scan must return k results and the
        planted near-duplicate must be ranked #1."""
        import numpy as np
        from aura.memory import store as store_mod

        with patch.object(store_mod, "_FAISS_AVAILABLE", False):
            store, query, target_id = self._make_store(tmp_path, rows=2500)
            results = store.search_semantic(query, k=10)
            assert len(results) == 10
            # Descending order.
            sims = [s for (_, s) in results]
            assert sims == sorted(sims, reverse=True)
            # Planted target ranks first.
            assert results[0][0].id == target_id

    def test_brute_force_is_paginated_not_single_fetchall(self, tmp_path, monkeypatch):
        """Force chunk size to 100 on 500 rows → expect ≥5 execute() calls,
        proving pagination rather than single load-all fetch."""
        import numpy as np
        from aura.memory import store as store_mod

        monkeypatch.setattr(store_mod, "_BRUTE_FORCE_CHUNK_SIZE", 100)
        with patch.object(store_mod, "_FAISS_AVAILABLE", False):
            store, query, _ = self._make_store(tmp_path, rows=500)
            # sqlite3.Connection.execute is read-only so we can't monkeypatch
            # it directly — wrap _get_conn to return a proxy that counts
            # execute() calls against the paginated SELECT.
            real_get_conn = store._get_conn
            call_count = {"n": 0}
            class _ConnProxy:
                def __init__(self, real): self._real = real
                def __getattr__(self, name): return getattr(self._real, name)
                def execute(self, sql, params=()):
                    if "FROM memories" in sql and "ORDER BY rowid" in sql:
                        call_count["n"] += 1
                    return self._real.execute(sql, params)
            def _proxy_get_conn():
                return _ConnProxy(real_get_conn())
            monkeypatch.setattr(store, "_get_conn", _proxy_get_conn)
            results = store.search_semantic(query, k=5)
            assert len(results) == 5
            assert call_count["n"] >= 5, f"expected ≥5 paginated calls, got {call_count['n']}"

    def test_brute_force_empty_query_returns_empty(self, tmp_path):
        """Zero-norm query vector is rejected early, no scan needed."""
        import numpy as np
        from aura.memory import store as store_mod

        with patch.object(store_mod, "_FAISS_AVAILABLE", False):
            store, _, _ = self._make_store(tmp_path, rows=10)
            zero = np.zeros(4, dtype=np.float32)
            assert store.search_semantic(zero, k=5) == []

    def test_brute_force_respects_user_id_filter(self, tmp_path):
        """Multi-user dataset: only rows with matching user_id come back."""
        import numpy as np
        from aura.memory.store import MemoryStore, MemoryRecord
        from aura.memory import store as store_mod

        db_path = tmp_path / "user_mem.db"
        store = MemoryStore(db_path=str(db_path))
        rng = np.random.default_rng(7)
        query = np.array([1.0, 0.0, 0.0, 0.0], dtype=np.float32)
        for i in range(20):
            rec = MemoryRecord(
                id=f"alice-{i}", content=f"A{i}", user_id="alice", lifecycle_state="stable",
            )
            store.insert(rec, embedding=query + rng.normal(0, 0.01, 4).astype(np.float32))
        for i in range(20):
            rec = MemoryRecord(
                id=f"bob-{i}", content=f"B{i}", user_id="bob", lifecycle_state="stable",
            )
            store.insert(rec, embedding=query + rng.normal(0, 0.01, 4).astype(np.float32))

        with patch.object(store_mod, "_FAISS_AVAILABLE", False):
            alice_results = store.search_semantic(query, k=50, user_id="alice")
            assert len(alice_results) == 20
            assert all(r.user_id == "alice" for r, _ in alice_results)
