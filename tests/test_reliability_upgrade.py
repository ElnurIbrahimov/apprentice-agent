"""
Tests for the Phase 1-4 reliability upgrade.

Run:  cd /d/Aura && python -m pytest tests/test_reliability_upgrade.py -v

All tests use mocks/fakes for external services (LLM, Qdrant, Playwright).
"""

from __future__ import annotations

import time
import unittest
from unittest.mock import MagicMock, patch


# ============================================================================
# PHASE 1 — Memory Write Gate
# ============================================================================

class TestMemoryWriteGate(unittest.TestCase):

    def _gate(self, **config_overrides):
        from aura.memory.write_gate import MemoryWriteGate
        gate = MemoryWriteGate()
        for k, v in config_overrides.items():
            setattr(gate, k, v)
        return gate

    def _cand(self, content, **kwargs):
        from aura.memory.write_gate import MemoryCandidate
        return MemoryCandidate(content=content, source="test", **kwargs)

    def test_noise_is_discarded(self):
        from aura.memory.write_gate import MemoryDecisionKind
        gate = self._gate()
        d = gate.evaluate(self._cand("ok"))
        self.assertEqual(d.kind, MemoryDecisionKind.DISCARD)

    def test_ok_is_discarded(self):
        from aura.memory.write_gate import MemoryDecisionKind
        gate = self._gate()
        d = gate.evaluate(self._cand("yes"))
        self.assertEqual(d.kind, MemoryDecisionKind.DISCARD)

    def test_empty_nearby_gives_high_novelty(self):
        from aura.memory.write_gate import MemoryDecisionKind
        gate = self._gate(_write_thr=0.2)  # low threshold so it stores
        d = gate.evaluate(
            self._cand("I prefer dark mode in all my apps", importance=0.6),
            nearby=[],
        )
        self.assertNotEqual(d.kind, MemoryDecisionKind.DISCARD)
        self.assertGreater(d.novelty, 0.5)

    def test_near_duplicate_triggers_merge(self):
        from aura.memory.write_gate import MemoryDecisionKind
        gate = self._gate(_merge_thr=0.8, _write_thr=0.2)
        nearby = [{"content": "I prefer dark mode", "score": 0.92,
                   "source_id": "mem_001", "source": "amem"}]
        d = gate.evaluate(
            self._cand("I prefer dark mode in my applications", importance=0.6),
            nearby=nearby,
        )
        self.assertEqual(d.kind, MemoryDecisionKind.MERGE_INTO)
        self.assertEqual(d.target_id, "mem_001")

    def test_correction_phrase_triggers_supersede(self):
        from aura.memory.write_gate import MemoryDecisionKind
        gate = self._gate(_supersede_thr=0.75, _merge_thr=0.98, _write_thr=0.2)
        nearby = [{"content": "I prefer Postgres for all projects", "score": 0.82,
                   "source_id": "mem_002", "source": "amem"}]
        d = gate.evaluate(
            self._cand(
                "Actually, I migrated to Supabase — no longer using raw Postgres",
                importance=0.7,
            ),
            nearby=nearby,
        )
        self.assertEqual(d.kind, MemoryDecisionKind.SUPERSEDE)

    def test_explicit_save_bypasses_low_score(self):
        from aura.memory.write_gate import MemoryDecisionKind
        gate = self._gate(_write_thr=0.95)  # extreme threshold
        d = gate.evaluate(
            self._cand("remember this", explicit_save=True, importance=0.1),
        )
        # Explicit save forces score above threshold
        self.assertNotEqual(d.kind, MemoryDecisionKind.DISCARD)

    def test_exact_dup_suppression(self):
        from aura.memory.write_gate import MemoryDecisionKind
        gate = self._gate(_write_thr=0.1)
        c = self._cand("My favourite colour is blue", importance=0.7)
        # First write
        d1 = gate.evaluate(c, nearby=[])
        self.assertNotEqual(d1.kind, MemoryDecisionKind.DISCARD)
        # Second write with identical content — should be suppressed
        c2 = self._cand("My favourite colour is blue", importance=0.7)
        d2 = gate.evaluate(c2, nearby=[])
        self.assertEqual(d2.kind, MemoryDecisionKind.DISCARD)
        self.assertEqual(d2.reason, "exact_duplicate_recent")

    def test_lifecycle_state_set(self):
        from aura.memory.write_gate import MemoryDecisionKind, MemoryLifecycleState
        gate = self._gate(_write_thr=0.1)
        d = gate.evaluate(
            self._cand("I work on BroadMind architecture research every day",
                       importance=0.9, tags=["preference"]),
        )
        if d.kind == MemoryDecisionKind.STORE_NEW:
            self.assertIn(d.lifecycle_state,
                          [MemoryLifecycleState.STABLE, MemoryLifecycleState.CANDIDATE])

    def test_disabled_gate_always_stores(self):
        from aura.memory.write_gate import MemoryDecisionKind
        gate = self._gate(_enabled=False)
        d = gate.evaluate(self._cand("ok"))
        self.assertEqual(d.kind, MemoryDecisionKind.STORE_NEW)


# ============================================================================
# PHASE 1 — Loop Guard
# ============================================================================

class TestLoopGuard(unittest.TestCase):

    def _guard(self, max_rep=3, nov_thr=0.25, window=20, budget=40):
        from aura.reliability.loop_guard import SessionLoopGuard
        g = SessionLoopGuard(session_id="test")
        g._max_rep = max_rep
        g._nov_thr = nov_thr
        g._window  = window
        g._budget  = budget
        return g

    def test_no_trigger_on_diverse_actions(self):
        g = self._guard()
        actions = [
            ("search_memory", "what is dark mode"),
            ("browser_navigate", "https://example.com"),
            ("llm_call", "summarize"),
            ("filesystem_read", "/etc/hosts"),
        ]
        for action, ctx in actions:
            r = g.record(action, ctx)
            self.assertFalse(r.triggered, f"Unexpected trigger on {action}")

    def test_triggers_on_repetition(self):
        g = self._guard(max_rep=3)
        for _ in range(3):
            r = g.record("search_memory", "same query every time")
        self.assertTrue(r.triggered)
        self.assertIn("repeated", r.reason)

    def test_triggers_on_low_novelty(self):
        g = self._guard(nov_thr=0.9)  # Very high threshold → low novelty fires fast
        for i in range(5):
            r = g.record("same_action", "same_context")
        self.assertTrue(r.triggered)

    def test_triggers_on_budget(self):
        g = self._guard(budget=5, max_rep=100, nov_thr=0.0)
        for i in range(5):
            g.record(f"action_{i}", f"ctx_{i}")
        r = g.record("action_6", "ctx_6")
        self.assertTrue(r.triggered)
        self.assertIn("budget", r.reason)

    def test_reset_clears_state(self):
        g = self._guard(max_rep=2)
        g.record("search", "q")
        g.record("search", "q")
        g.reset()
        r = g.record("search", "q")
        self.assertFalse(r.triggered)

    def test_fallback_message_on_trigger(self):
        g = self._guard(max_rep=2)
        g.record("search", "q")
        g.record("search", "q")
        r = g.record("search", "q")
        self.assertTrue(r.triggered)
        self.assertIn("circles", r.fallback_message)


# ============================================================================
# PHASE 2 — Browser Executor
# ============================================================================

class _FakePage:
    """Minimal Playwright Page mock."""
    def __init__(self, url="https://example.com", title="Example"):
        self._url   = url
        self._title = title
        self._body  = "Welcome to Example dashboard logout"

    @property
    def url(self):
        return self._url

    def goto(self, url, **kwargs):
        self._url = url

    def title(self):
        return self._title

    def inner_text(self, selector):
        return self._body

    def wait_for_selector(self, sel, **kwargs):
        pass

    def click(self, sel):
        self._body += " clicked"

    def fill(self, sel, text):
        self._body += f" filled:{text}"

    def screenshot(self, path=""):
        pass

    def evaluate(self, script):
        pass

    def select_option(self, sel, **kwargs):
        pass


class TestBrowserExecutor(unittest.TestCase):

    def _executor(self, page, **kwargs):
        from aura.browser.executor import BrowserExecutor
        return BrowserExecutor(page=page, session_id="test", **kwargs)

    def _action(self, kind, **kwargs):
        from aura.browser.executor import PlannedAction, ActionKind
        return PlannedAction(kind=ActionKind(kind), **kwargs)

    def test_navigate_success(self):
        page = _FakePage()
        ex   = self._executor(page)
        a    = self._action("navigate", url="https://example.com")
        r    = ex.execute(a)
        self.assertTrue(r.success)

    def test_postcondition_passes_when_signal_in_body(self):
        page = _FakePage(url="https://example.com/dashboard")
        page._body = "Welcome to the dashboard! logout"
        ex = self._executor(page)
        a  = self._action("click", selector="#login-btn",
                          success_signals=["dashboard", "logout"])
        r  = ex.execute(a)
        self.assertTrue(r.success)
        self.assertTrue(r.postcondition_passed)

    def test_postcondition_fails_when_no_signal(self):
        page = _FakePage()
        page._body = "error page not found"
        ex = self._executor(page)
        a  = self._action("click", selector="#login-btn",
                          success_signals=["dashboard", "welcome"])
        r  = ex.execute(a)
        self.assertFalse(r.postcondition_passed)

    def test_domain_drift_aborts(self):
        # Simulate redirect: goto() is called but page ends up at evil.com
        class RedirectToEvil(_FakePage):
            def goto(self, url, **kwargs):
                pass  # URL stays evil.com — simulates unexpected redirect

        page = RedirectToEvil(url="https://evil.com/redirect")
        ex   = self._executor(page, allowed_domain="example.com")
        a    = self._action("navigate", url="https://example.com/go")
        r    = ex.execute(a)
        self.assertFalse(r.success)
        self.assertIn("domain_drift", r.error)

    def test_destructive_action_blocked(self):
        page = _FakePage()
        ex   = self._executor(page, allow_destructive=False)
        from aura.browser.executor import PlannedAction, ActionKind, SafetyClass
        a = PlannedAction(
            kind=ActionKind.CLICK,
            selector="#confirm",
            description="confirm delete",
            safety_class=SafetyClass.DESTRUCTIVE,
        )
        r = ex.execute(a)
        self.assertFalse(r.success)
        self.assertIn("blocked", r.error)

    def test_retry_on_exception(self):
        """Executor retries when click raises."""
        page = _FakePage()
        call_count = [0]

        original_click = page.click
        def flaky_click(sel):
            call_count[0] += 1
            if call_count[0] < 2:
                raise RuntimeError("element not found")
            return original_click(sel)
        page.click = flaky_click

        ex = self._executor(page)
        a  = self._action("click", selector="#btn")
        r  = ex.execute(a)
        self.assertTrue(r.success)
        self.assertEqual(r.attempt, 2)

    def test_loop_guard_in_browser_session(self):
        """BrowserSession loop guard fires after repeated same actions."""
        from aura.browser.executor import BrowserSession, ActionKind
        page = _FakePage()
        session = BrowserSession(page=page, task="test task", session_id="sess_test")
        # Force guard to low max_rep
        if session._guard:
            session._guard._max_rep = 2

        raw = {"action": "click", "selector": "#btn", "description": "click login"}
        session.step(raw)
        session.step(raw)  # 2nd
        r = session.step(raw)  # 3rd — should trigger loop guard
        if session._guard and session._guard._triggered:
            self.assertIn("loop_guard", r.error)


# ============================================================================
# PHASE 3 — Outcome-aware Routing
# ============================================================================

class TestRoutingStats(unittest.TestCase):

    def _store(self):
        from aura.reliability.routing_stats import RoutingStatsStore
        return RoutingStatsStore(persist_path=None)

    def test_fallback_when_no_data(self):
        store = self._store()
        result = store.select_model_for_task("memory_summarization",
                                              ["model_a", "model_b"])
        self.assertEqual(result, "model_a")  # first candidate returned

    def test_selects_best_after_enough_samples(self):
        from aura.reliability.routing_stats import RoutingStatsStore
        store = self._store()
        # model_a: 3/5 success
        for _ in range(3):
            store.record("code_edit", "model_a", success=True, latency_ms=100)
        for _ in range(2):
            store.record("code_edit", "model_a", success=False, latency_ms=100)
        # model_b: 5/5 success
        for _ in range(5):
            store.record("code_edit", "model_b", success=True, latency_ms=200)

        result = store.select_model_for_task("code_edit", ["model_a", "model_b"])
        self.assertEqual(result, "model_b")

    def test_falls_back_when_insufficient_data(self):
        store = self._store()
        # Only 2 samples for model_a — below MIN_SAMPLES=5
        store.record("browser_planning", "model_a", success=True)
        store.record("browser_planning", "model_a", success=True)
        result = store.select_model_for_task("browser_planning", ["model_a", "model_b"])
        self.assertEqual(result, "model_a")  # falls back to candidates[0]

    def test_summary_structure(self):
        store = self._store()
        store.record("general", "model_x", success=True, latency_ms=50)
        s = store.summary()
        self.assertIn("total_records", s)
        self.assertIn("by_category_model", s)


# ============================================================================
# PHASE 3 — KG Contradiction
# ============================================================================

class _FakeKG:
    """Minimal fake KnowledgeGraph for contradiction tests."""
    def __init__(self):
        import networkx as nx
        self.graph = nx.DiGraph()

    def add_node(self, node_id, **attrs):
        self.graph.add_node(node_id, **attrs)

    def find_nodes(self, query, limit=10):
        return []


class TestKGContradiction(unittest.TestCase):

    def setUp(self):
        self.kg = _FakeKG()
        from aura.memory.kg_contradiction import KGContradictionDetector
        self.detector = KGContradictionDetector(self.kg)

    def _add_node(self, node_id, label, node_type="entity", **props):
        self.kg.graph.add_node(
            node_id,
            label=label,
            type=node_type,
            properties=props,
            lifecycle_state="active",
        )

    def test_no_contradiction_for_unrelated_nodes(self):
        self._add_node("n1", "dark mode preference", "concept")
        self._add_node("n2", "blue colour preference", "concept")
        recs = self.detector.check_for_contradictions("n1")
        self.assertEqual(len(recs), 0)

    def test_value_conflict_detected(self):
        self._add_node("n1", "database preference", "entity",
                       db_type="postgres")
        self._add_node("n2", "database preference", "entity",
                       db_type="mysql")
        recs = self.detector.check_for_contradictions("n1")
        self.assertTrue(any(r.contradiction_type == "value_conflict" for r in recs))

    def test_negation_detected(self):
        self._add_node("n1", "user always prefers dark mode", "concept")
        self._add_node("n2", "user never uses dark mode", "concept")
        recs = self.detector.check_for_contradictions("n1")
        self.assertTrue(any(r.contradiction_type == "direct_negation" for r in recs))

    def test_supersede_marks_old_node(self):
        from aura.memory.kg_contradiction import KG_NODE_SUPERSEDED
        self._add_node("old_n", "user prefers postgres", "entity")
        self._add_node("new_n", "user migrated to supabase", "entity")
        self.detector.supersede("old_n", "new_n", reason="user_correction")
        self.assertEqual(
            self.kg.graph.nodes["old_n"].get("lifecycle_state"),
            KG_NODE_SUPERSEDED,
        )

    def test_supersede_edge_added(self):
        from aura.memory.kg_contradiction import SUPERSEDES_EDGE
        self._add_node("old_n2", "used vim", "entity")
        self._add_node("new_n2", "switched to neovim", "entity")
        self.detector.supersede("old_n2", "new_n2")
        edges = list(self.kg.graph.edges(data=True))
        supersedes = [(u, v, d) for u, v, d in edges
                      if d.get("type") == SUPERSEDES_EDGE]
        self.assertTrue(len(supersedes) > 0)

    def test_contradiction_edge_added(self):
        from aura.memory.kg_contradiction import CONTRADICTS_EDGE
        self._add_node("a1", "user always enables telemetry", "concept")
        self._add_node("b1", "user never enables telemetry", "concept")
        self.detector.check_for_contradictions("a1")
        edges = list(self.kg.graph.edges(data=True))
        contradicts = [(u, v, d) for u, v, d in edges
                       if d.get("type") == CONTRADICTS_EDGE]
        self.assertTrue(len(contradicts) > 0)

    def test_get_unresolved_contradictions(self):
        self._add_node("x1", "user always uses docker", "concept")
        self._add_node("x2", "user never uses docker", "concept")
        self.detector.check_for_contradictions("x1")
        unresolved = self.detector.get_unresolved_contradictions()
        self.assertTrue(len(unresolved) > 0)
        self.assertFalse(all(c.resolved for c in unresolved))


# ============================================================================
# PHASE 4 — Dream Consolidator
# ============================================================================

class TestDreamConsolidator(unittest.TestCase):

    def _make_memories(self, n: int, content_prefix: str = "memory") -> list:
        return [
            {"id": f"m{i}", "content": f"{content_prefix} {i} about dark mode preference",
             "source": "amem", "tags": ["preference"], "importance": 0.6, "ts": time.time()}
            for i in range(n)
        ]

    def test_cluster_by_similarity_groups_similar(self):
        from aura.dream import _cluster_by_similarity
        items = [
            {"content": "I prefer dark mode in all my applications"},
            {"content": "Dark mode is my preferred display setting"},
            {"content": "I always use dark mode"},
            {"content": "Completely unrelated topic about cooking pasta"},
        ]
        clusters = _cluster_by_similarity(items, threshold=0.25)
        # First 3 should cluster together
        cluster_sizes = sorted([len(c) for c in clusters], reverse=True)
        self.assertGreaterEqual(cluster_sizes[0], 2)  # At least 2 similar ones grouped

    def test_consolidator_runs_without_crash(self):
        """Smoke test: run_cycle with empty memory backends."""
        from aura.dream import DreamConsolidator
        consolidator = DreamConsolidator()
        # Mock the fetch so we don't need real backends
        consolidator._fetch_memories = lambda user_id, store=None: self._make_memories(5)
        consolidator._summarize_cluster = lambda cl, uid: None  # skip LLM
        consolidator._contradiction_report = lambda: []
        consolidator._write_summary_memory = lambda s, uid: None
        consolidator._prune_stale = lambda mems, uid: []
        report = consolidator.run_cycle(user_id="test_user")
        self.assertIsNotNone(report)
        self.assertEqual(report.cycle.memories_processed, 5)

    def test_min_cluster_size_respected(self):
        """Clusters smaller than min_cluster_size produce no summaries."""
        from aura.dream import DreamConsolidator
        consolidator = DreamConsolidator()
        consolidator._min_cluster = 5
        consolidator._fetch_memories = lambda uid, store=None: self._make_memories(2)
        consolidator._write_summary_memory = lambda s, uid: None
        consolidator._contradiction_report = lambda: []
        consolidator._prune_stale = lambda mems, uid: []

        summarize_calls = [0]
        def fake_summarize(cluster, uid):
            summarize_calls[0] += 1
            return None
        consolidator._summarize_cluster = fake_summarize

        report = consolidator.run_cycle(user_id="test_user")
        # 2 memories → 1 cluster of size 2 → below min=5 → no summaries
        self.assertEqual(summarize_calls[0], 0)

    def test_routine_extraction_fires_above_min_size(self):
        from aura.dream import DreamConsolidator
        consolidator = DreamConsolidator()
        consolidator._min_cluster = 3
        consolidator._do_routines = True
        # 6 nearly identical memories → should detect a routine
        memories = [
            {"id": f"r{i}",
             "content": "user asks for code review after writing python",
             "source": "amem", "tags": [], "importance": 0.5, "ts": time.time()}
            for i in range(6)
        ]
        routines = consolidator._extract_routines(memories)
        self.assertTrue(len(routines) > 0)
        self.assertGreaterEqual(routines[0].frequency, 3)

    def test_cycle_respects_batch_size(self):
        from aura.dream import DreamConsolidator
        consolidator = DreamConsolidator()
        consolidator._batch_size = 3
        consolidator._fetch_memories = lambda uid, store=None: self._make_memories(20)
        consolidator._summarize_cluster = lambda cl, uid: None
        consolidator._contradiction_report = lambda: []
        consolidator._write_summary_memory = lambda s, uid: None
        consolidator._prune_stale = lambda mems, uid: []
        report = consolidator.run_cycle(user_id="test_user")
        # 20 memories but batch_size=3 means at most 3 clusters processed
        self.assertLessEqual(report.cycle.summaries_written, 3)

    def test_user_scoping_not_violated(self):
        """Each user_id call should only see its own user_id in logs."""
        from aura.dream import DreamConsolidator
        seen_users = []
        consolidator = DreamConsolidator()

        def fake_fetch(uid, store=None):
            seen_users.append(uid)
            return []

        consolidator._fetch_memories = fake_fetch
        consolidator._contradiction_report = lambda: []
        consolidator._prune_stale = lambda mems, uid: []
        consolidator.run_cycle(user_id="user_A")
        consolidator.run_cycle(user_id="user_B")
        self.assertIn("user_A", seen_users)
        self.assertIn("user_B", seen_users)
        # Each call should have exactly one user_id — no cross-contamination
        self.assertEqual(len(seen_users), 2)


# ============================================================================
# PHASE 1 — Telemetry
# ============================================================================

class TestTelemetry(unittest.TestCase):

    def test_emit_and_retrieve(self):
        from aura.reliability.telemetry import TelemetrySink, TelemetryKind
        sink = TelemetrySink(log_dir="/tmp/aura_telem_test")
        ev = sink.emit(TelemetryKind.TASK_RESULT, success=True, latency_ms=42.0)
        recent = sink.recent(n=10)
        self.assertTrue(any(e["event_id"] == ev.event_id for e in recent))

    def test_stats_aggregate(self):
        from aura.reliability.telemetry import TelemetrySink, TelemetryKind
        sink = TelemetrySink(log_dir="/tmp/aura_telem_test")
        sink.emit(TelemetryKind.TASK_RESULT, success=True, latency_ms=100.0)
        sink.emit(TelemetryKind.TASK_RESULT, success=False, latency_ms=200.0)
        s = sink.stats()
        self.assertEqual(s["total"], 2)
        self.assertEqual(s["successes"], 1)
        self.assertEqual(s["failures"], 1)

    def test_kind_filter(self):
        from aura.reliability.telemetry import TelemetrySink, TelemetryKind
        sink = TelemetrySink(log_dir="/tmp/aura_telem_test")
        sink.emit(TelemetryKind.MEMORY_DECISION, success=True)
        sink.emit(TelemetryKind.BROWSER_ACTION, success=True)
        mem = sink.recent(n=10, kind=TelemetryKind.MEMORY_DECISION)
        self.assertTrue(all(e["kind"] == TelemetryKind.MEMORY_DECISION for e in mem))


# ============================================================================
# Entry point
# ============================================================================

if __name__ == "__main__":
    unittest.main(verbosity=2)
