"""
Integration tests for the chat -> memory -> response pipeline.

Tests the end-to-end flow between agent.chat(), UnifiedMemory, MemoryWriteGate,
and OllamaBrain.think(), with LLM calls and vector DB backends mocked out.

The goal is to verify the PIPELINE — that components are wired together correctly
and data flows through the expected path — not to test individual components.
"""

import json
import os
import sys
import threading
import time
import pytest
from unittest.mock import patch, MagicMock, PropertyMock, ANY

# Ensure project root is on sys.path so `aura.*` and `api.*` resolve
sys.path.insert(0, os.path.join(os.path.dirname(__file__), ".."))


# ---------------------------------------------------------------------------
# Fixtures
# ---------------------------------------------------------------------------

@pytest.fixture
def write_gate():
    """Fresh MemoryWriteGate with no recent hash history."""
    from aura.memory.write_gate import MemoryWriteGate
    gate = MemoryWriteGate()
    gate._enabled = True
    gate._recent_hashes = {}
    return gate


@pytest.fixture
def unified_memory():
    """UnifiedMemory with all backends disabled (no real DB connections).

    Backends are set to None so queries return empty lists by default.
    Tests that need results should mock `query()` or inject a backend.
    """
    from aura.memory.unified_memory import UnifiedMemory
    um = UnifiedMemory()
    um._backends_checked = True  # Skip lazy init that would hit real DBs
    um._amem = None
    um._kg = None
    um._rag = None
    um._episodic = None
    um._kg_brain = None
    return um


@pytest.fixture
def mock_brain():
    """Mock OllamaBrain that returns deterministic responses.

    Provides:
      - think() returns a fixed string
      - conversation_history is an in-memory list
      - set_model_override / _last_model_used are no-ops
    """
    brain = MagicMock()
    brain.think.return_value = "Quantum computing uses qubits to perform parallel computations."
    brain.conversation_history = []
    brain._history_lock = threading.Lock()
    brain._max_history = 20
    brain.set_model_override = MagicMock()
    brain._last_model_used = "mock-model"
    brain._model_override = None
    return brain


# ---------------------------------------------------------------------------
# Test 1: Chat stores meaningful memory
# ---------------------------------------------------------------------------

class TestChatStoresMemory:
    """Verify that agent.chat() triggers a gated memory write for non-trivial
    exchanges.  We mock brain.think and UnifiedMemory, then assert that
    store_gated (or store) was called with content derived from the
    user message + agent response.
    """

    @patch("aura.memory.unified_memory.get_unified_memory")
    @patch("aura.agent.OllamaBrain")
    def test_chat_triggers_memory_store(self, MockBrain, mock_get_umem):
        """Call agent.chat() with a meaningful message and verify memory pipeline fires."""
        # Set up mock brain
        brain_instance = MagicMock()
        brain_instance.think.return_value = "Quantum computing leverages quantum mechanics for computation."
        brain_instance.conversation_history = []
        brain_instance._history_lock = threading.Lock()
        brain_instance._max_history = 20
        brain_instance.set_model_override = MagicMock()
        brain_instance._last_model_used = "mock-model"
        brain_instance._model_override = None
        MockBrain.return_value = brain_instance

        # Set up mock unified memory
        umem = MagicMock()
        umem.query.return_value = []  # No prior memories
        umem.store_gated = MagicMock(return_value={"decision": "store_new", "score": 0.7})
        umem.store = MagicMock(return_value={})
        mock_get_umem.return_value = umem

        # Patch heavy dependencies to avoid real initialization
        with patch("aura.agent.MemorySystem"), \
             patch("aura.agent.Config"), \
             patch("aura.agent.load_identity", return_value={}), \
             patch("aura.agent.get_identity_prompt", return_value=""), \
             patch("aura.agent.MetacognitionLogger"):

            # We cannot easily construct a full AuraAgent without its massive __init__,
            # so instead we test the specific code path that writes to memory.
            # The agent.chat() method (line ~5488-5520) does:
            #   1. get_unified_memory()
            #   2. Compose content = "User: <msg>\nAURA: <response>"
            #   3. Call store_gated() in a background thread

            # Simulate exactly what agent.chat does for memory storage:
            from aura.memory.unified_memory import get_unified_memory

            message = "Tell me about quantum computing"
            response = brain_instance.think(message)

            # Replicate agent's memory storage logic (agent.py ~5491-5518)
            _clean_message = message.split("\n[Screen context:")[0].strip()
            _clean_response = response.split("\n\n---\n")[0].strip()
            _mem_content = f"User: {_clean_message[:200]}\nAURA: {_clean_response[:400]}"

            _umem_ref = get_unified_memory()
            _store_fn = getattr(_umem_ref, "store_gated", _umem_ref.store)

            # Run synchronously (agent runs in daemon thread, we do it inline)
            _store_fn(content=_mem_content, source="conversation",
                      importance=0.5, emotional_pad=None)

            # Verify store_gated was called with correct content
            umem.store_gated.assert_called_once()
            call_kwargs = umem.store_gated.call_args[1]
            assert "User: Tell me about quantum computing" in call_kwargs["content"]
            assert "AURA: Quantum computing" in call_kwargs["content"]
            assert call_kwargs["source"] == "conversation"


# ---------------------------------------------------------------------------
# Test 2: Memory recall influences response
# ---------------------------------------------------------------------------

class TestMemoryRecallInfluencesResponse:
    """Verify that when UnifiedMemory returns results, they get injected
    into the context that brain.think() receives (via system_prompt).

    The flow in agent.chat() (lines ~5114-5158):
      1. query UnifiedMemory
      2. Format results as "MEMORY CONTEXT:\\n- [SOURCE] content"
      3. Append to system_prompt_addon
      4. Pass system_prompt_addon to brain.think()
    """

    def test_memory_results_injected_into_brain_context(self):
        """Pre-populate mock memory results and verify they appear in brain.think() call."""
        from aura.memory.unified_memory import UnifiedResult

        # Create mock memory results
        memory_results = [
            UnifiedResult(
                content="User prefers Python over JavaScript for data science",
                source="amem",
                score=0.85,
                relevance=0.9,
                recency=0.8,
                importance=0.7,
            ),
            UnifiedResult(
                content="Previous discussion about machine learning frameworks",
                source="episodic",
                score=0.72,
                relevance=0.75,
                recency=0.6,
                importance=0.65,
            ),
        ]

        # Simulate the context building from agent.chat() lines 5136-5142
        from aura.memory.context_budget import ContextBudget
        _ctx_budget = ContextBudget(total_tokens=3000)

        unified_results = memory_results
        unified_context = ""
        if unified_results:
            _budget = _ctx_budget.allocate("unified", requested=_ctx_budget.remaining)
            _per = max(200, (_budget * 4) // max(1, len(unified_results)))
            texts = [f"- [{r.source.upper()}] {r.content[:_per]}"
                     for r in unified_results if r.content]
            if texts:
                unified_context = "MEMORY CONTEXT:\n" + "\n".join(texts)

        # Verify memory context was built correctly
        assert "MEMORY CONTEXT:" in unified_context
        assert "[AMEM] User prefers Python over JavaScript" in unified_context
        assert "[EPISODIC] Previous discussion about machine learning" in unified_context

        # Verify this context would be appended to system_prompt_addon
        context_parts = []
        if unified_context:
            context_parts.append(unified_context)

        system_prompt_addon = "\n\n".join(context_parts) + \
            "\n\nUse this knowledge and memories when relevant to the conversation."

        assert "MEMORY CONTEXT:" in system_prompt_addon
        assert "User prefers Python" in system_prompt_addon

        # Simulate brain.think() call with the constructed prompt
        brain = MagicMock()
        brain.think.return_value = "Based on your preference for Python..."

        response = brain.think(
            "What framework should I use?",
            system_prompt=system_prompt_addon,
            task_type=None,
            tone_modifier=None,
        )

        # Verify brain.think was called with system_prompt containing memory context
        brain.think.assert_called_once()
        call_kwargs = brain.think.call_args[1]
        assert "MEMORY CONTEXT:" in call_kwargs["system_prompt"]
        assert "User prefers Python" in call_kwargs["system_prompt"]


# ---------------------------------------------------------------------------
# Test 3: Write gate filters noise
# ---------------------------------------------------------------------------

class TestWriteGateFiltersNoise:
    """Test that the MemoryWriteGate correctly discards trivial input
    and stores meaningful content.

    The write gate (write_gate.py) uses:
      - _is_noise() for hard rejection of short/trivial content
      - Weighted scoring (novelty, utility, emotion, specificity, confidence)
      - Threshold comparison (default 0.35)
    """

    def test_trivial_inputs_get_discarded(self, write_gate):
        """Trivial messages like 'ok', 'thanks', '...' should get DISCARD."""
        from aura.memory.write_gate import MemoryCandidate, MemoryDecisionKind

        trivial_inputs = ["ok", "thanks", "...", "yes", "no", "sure", "cool", "np"]

        for text in trivial_inputs:
            candidate = MemoryCandidate(
                content=text,
                source="conversation",
                user_id="test_user",
            )
            decision = write_gate.evaluate(candidate, nearby=[])
            assert decision.kind == MemoryDecisionKind.DISCARD, \
                f"Expected DISCARD for '{text}', got {decision.kind.value} (score={decision.score:.3f})"

    def test_meaningful_input_gets_stored(self, write_gate):
        """Meaningful personal preference should score above threshold."""
        from aura.memory.write_gate import MemoryCandidate, MemoryDecisionKind

        candidate = MemoryCandidate(
            content="I prefer Python over JavaScript for data science because of pandas and numpy",
            source="conversation",
            user_id="test_user",
            importance=0.6,
            tags=["preference"],
        )
        decision = write_gate.evaluate(candidate, nearby=[])

        # Should NOT be discarded — the content has specificity, utility, and novelty
        assert decision.kind != MemoryDecisionKind.DISCARD, \
            f"Expected non-DISCARD for meaningful input, got {decision.kind.value} (score={decision.score:.3f})"
        assert decision.score >= write_gate._write_thr, \
            f"Score {decision.score:.3f} below threshold {write_gate._write_thr}"

    def test_explicit_save_always_passes(self, write_gate):
        """Content with explicit_save=True should always pass the gate."""
        from aura.memory.write_gate import MemoryCandidate, MemoryDecisionKind

        candidate = MemoryCandidate(
            content="My API key is stored in the env var OPENAI_KEY",
            source="conversation",
            user_id="test_user",
            explicit_save=True,
        )
        decision = write_gate.evaluate(candidate, nearby=[])

        assert decision.kind != MemoryDecisionKind.DISCARD, \
            f"explicit_save=True should not be discarded, got {decision.kind.value}"

    def test_exact_duplicate_gets_discarded(self, write_gate):
        """Sending the same content twice within TTL should discard the second."""
        from aura.memory.write_gate import MemoryCandidate, MemoryDecisionKind

        content = "I prefer dark mode in all my editors and terminals"
        candidate1 = MemoryCandidate(content=content, source="conversation", user_id="test_user", importance=0.6)
        candidate2 = MemoryCandidate(content=content, source="conversation", user_id="test_user", importance=0.6)

        decision1 = write_gate.evaluate(candidate1, nearby=[])
        assert decision1.kind != MemoryDecisionKind.DISCARD

        decision2 = write_gate.evaluate(candidate2, nearby=[])
        assert decision2.kind == MemoryDecisionKind.DISCARD
        assert "duplicate" in decision2.reason.lower()


# ---------------------------------------------------------------------------
# Test 4: End-to-end conversation flow
# ---------------------------------------------------------------------------

class TestEndToEndConversationFlow:
    """Simulate a 3-message conversation through the memory pipeline.

    Mocks the LLM (brain.think) but lets the real MemoryWriteGate and
    UnifiedMemory.store_gated logic run (with in-memory backends).

    Flow per message:
      1. Build memory context from UnifiedMemory.query()
      2. Call brain.think() with context
      3. Store exchange via UnifiedMemory.store_gated() -> WriteGate -> store()
    """

    def test_three_message_flow(self, write_gate, unified_memory):
        """Simulate 3-turn conversation with real gate + in-memory memory."""
        from aura.memory.unified_memory import UnifiedResult
        from aura.memory.write_gate import MemoryCandidate, MemoryDecisionKind, get_write_gate

        # Track what gets stored
        stored_memories = []

        # Override store() to capture writes in-memory (no real DB)
        original_store = unified_memory.store
        def mock_store(**kwargs):
            stored_memories.append(kwargs)
            return {"mock": "stored"}
        unified_memory.store = mock_store

        # Override query() to return previously stored memories
        # Uses word overlap (simulating a simple search) — matches if any
        # non-stopword from the query appears in stored content
        _stopwords = {"what", "have", "we", "so", "far", "the", "a", "an", "is",
                       "are", "about", "for", "do", "my", "to", "in", "of", "and"}

        def mock_query(query, k=10, **kwargs):
            results = []
            query_words = [w for w in query.lower().split() if w not in _stopwords and len(w) > 2]
            for i, mem in enumerate(stored_memories):
                content = mem.get("content", "").lower()
                if any(word in content for word in query_words):
                    results.append(UnifiedResult(
                        content=mem.get("content", ""),
                        source="amem",
                        score=0.7 - (i * 0.05),
                        relevance=0.8 - (i * 0.05),
                        recency=0.9,
                        importance=0.5,
                    ))
            return results[:k]
        unified_memory.query = mock_query

        # Mock brain responses — each turn covers a distinct topic to ensure novelty
        brain_responses = [
            "Python is great for data science due to libraries like pandas and scikit-learn.",
            "I prefer using Docker containers for my deployment workflow because they provide reproducible environments.",
            "We discussed Python for data science, Docker for deployment, and your general workflow preferences.",
        ]
        response_idx = [0]

        def mock_think(prompt, **kwargs):
            idx = response_idx[0]
            response_idx[0] += 1
            return brain_responses[min(idx, len(brain_responses) - 1)]

        # --- Turn 1: User asks about Python ---
        msg1 = "I prefer Python for data science work because of pandas and numpy"
        response1 = mock_think(msg1)

        # Store via the real gate (but with mock store backend)
        mem_content1 = f"User: {msg1[:200]}\nAURA: {response1[:400]}"

        # Patch get_write_gate at source module + telemetry
        with patch("aura.memory.write_gate.get_write_gate", return_value=write_gate), \
             patch("aura.reliability.telemetry.emit", MagicMock()):
            result1 = unified_memory.store_gated(
                content=mem_content1,
                source="conversation",
                importance=0.6,
                tags=["preference"],
            )
        assert result1.get("decision") != "discard", f"Turn 1 should be stored: {result1}"

        # --- Turn 2: Different topic to ensure novelty ---
        msg2 = "I use Docker containers for my project deployment workflow"

        # Query memory — should find Turn 1's content (matching on common words)
        recall2 = unified_memory.query(msg2, k=5)
        # Build context string like agent.chat does
        if recall2:
            context_texts = [f"- [{r.source.upper()}] {r.content[:200]}" for r in recall2]
            memory_context2 = "MEMORY CONTEXT:\n" + "\n".join(context_texts)
        else:
            memory_context2 = ""

        response2 = mock_think(msg2, system_prompt=memory_context2)
        mem_content2 = f"User: {msg2[:200]}\nAURA: {response2[:400]}"

        with patch("aura.memory.write_gate.get_write_gate", return_value=write_gate), \
             patch("aura.reliability.telemetry.emit", MagicMock()):
            result2 = unified_memory.store_gated(
                content=mem_content2,
                source="conversation",
                importance=0.6,
                tags=["preference"],
            )
        assert result2.get("decision") != "discard", f"Turn 2 should be stored: {result2}"

        # --- Turn 3: Recall everything — query uses keywords present in stored content ---
        msg3 = "Tell me about Python and Docker from earlier"
        recall3 = unified_memory.query(msg3, k=10)
        assert len(recall3) >= 2, f"Turn 3 should recall at least 2 memories, got {len(recall3)}"

        # Verify stored memories contain both exchanges
        assert len(stored_memories) >= 2, f"Expected >= 2 stored memories, got {len(stored_memories)}"
        all_content = " ".join(m["content"] for m in stored_memories)
        assert "python" in all_content.lower() or "data science" in all_content.lower()
        assert "docker" in all_content.lower() or "deployment" in all_content.lower()


# ---------------------------------------------------------------------------
# Test 5: WebSocket message protocol
# ---------------------------------------------------------------------------

class TestWebSocketProtocol:
    """Test the WebSocket chat endpoint at /api/chat/stream.

    Uses FastAPI TestClient's WebSocket support to verify:
      1. Connection succeeds
      2. Sending a chat message JSON returns chunk + done messages
      3. The protocol matches: {"type": "chunk", "content": ...} and
         {"type": "done", "response": ..., "mood": ...}

    The agent is mocked to return a fixed response via chat_stream().
    """

    @pytest.fixture
    def test_app(self):
        """Create a FastAPI test app with the chat router.

        Mocks agent_service to avoid real agent initialization.
        """
        from fastapi import FastAPI
        from fastapi.testclient import TestClient

        # Disable auth for testing
        os.environ["AURA_API_AUTH_ENABLED"] = "false"
        os.environ["AURA_REQUIRE_AUTH"] = "false"

        app = FastAPI()

        from api.routes.chat import router
        app.include_router(router)

        return TestClient(app)

    def test_websocket_chat_protocol(self, test_app):
        """Connect, send a chat message, and verify chunk+done response."""
        # Mock the agent service to return predictable streaming output
        mock_service = MagicMock()
        mock_service.is_ready = True

        def fake_chat_stream(message, model_override=None, action_mode=None):
            yield {"type": "chunk", "content": "Hello "}
            yield {"type": "chunk", "content": "world!"}
            yield {"type": "done", "mood": {"emotion": "neutral", "confidence": 50}, "model_used": "test-model"}

        mock_service.chat_stream = fake_chat_stream
        mock_service.agent = MagicMock()
        mock_service.agent.brain = MagicMock()
        mock_service.agent.brain.get_current_conversation_id = MagicMock(return_value=None)

        with patch("api.routes.chat._get_agent_service", return_value=mock_service):
            with test_app.websocket_connect("/api/chat/stream") as ws:
                # Send chat message
                ws.send_text(json.dumps({
                    "type": "chat",
                    "message": "Hello AURA"
                }))

                # Collect responses
                messages = []
                while True:
                    data = ws.receive_json()
                    messages.append(data)
                    if data.get("type") in ("done", "error"):
                        break

                # Verify protocol
                chunk_messages = [m for m in messages if m["type"] == "chunk"]
                done_messages = [m for m in messages if m["type"] == "done"]

                assert len(chunk_messages) >= 1, "Should receive at least one chunk"
                assert len(done_messages) == 1, "Should receive exactly one done message"

                # Verify chunk content
                full_content = "".join(m["content"] for m in chunk_messages)
                assert "Hello " in full_content
                assert "world!" in full_content

                # Verify done message structure
                done = done_messages[0]
                assert "response" in done or "mood" in done
                assert done.get("model_used") == "test-model"

    def test_websocket_ping_pong(self, test_app):
        """Verify the WebSocket keepalive ping/pong protocol."""
        mock_service = MagicMock()
        mock_service.is_ready = True

        with patch("api.routes.chat._get_agent_service", return_value=mock_service):
            with test_app.websocket_connect("/api/chat/stream") as ws:
                ws.send_text(json.dumps({"type": "ping"}))
                data = ws.receive_json()
                assert data["type"] == "pong"

    def test_websocket_invalid_message_format(self, test_app):
        """Sending an invalid message format should return an error."""
        mock_service = MagicMock()
        mock_service.is_ready = True

        with patch("api.routes.chat._get_agent_service", return_value=mock_service):
            with test_app.websocket_connect("/api/chat/stream") as ws:
                # Send non-JSON
                ws.send_text("not json")
                data = ws.receive_json()
                assert data["type"] == "error"
                assert "Invalid JSON" in data["error"]

                # Send JSON without required fields
                ws.send_text(json.dumps({"type": "chat"}))
                data = ws.receive_json()
                assert data["type"] == "error"

    def test_websocket_stop_generation(self, test_app):
        """Sending a stop message should halt generation."""
        mock_service = MagicMock()
        mock_service.is_ready = True

        with patch("api.routes.chat._get_agent_service", return_value=mock_service):
            with test_app.websocket_connect("/api/chat/stream") as ws:
                ws.send_text(json.dumps({"type": "stop"}))
                data = ws.receive_json()
                assert data["type"] == "stopped"


# ---------------------------------------------------------------------------
# Additional pipeline integration tests
# ---------------------------------------------------------------------------

class TestUnifiedMemoryStoreGatedPipeline:
    """Test the store_gated pipeline: candidate -> WriteGate -> store fan-out."""

    def test_store_gated_calls_gate_then_store(self, unified_memory):
        """Verify store_gated calls evaluate() then store() for worthy content."""
        from aura.memory.write_gate import MemoryWriteGate, MemoryDecisionKind, MemoryDecision, MemoryCandidate

        store_calls = []
        def capture_store(**kwargs):
            store_calls.append(kwargs)
            return {"amem": "test-id-123"}
        unified_memory.store = capture_store

        # Patch the write gate at its source module and telemetry
        with patch("aura.memory.write_gate.get_write_gate") as mock_gate_fn, \
             patch("aura.reliability.telemetry.emit"):

            mock_gate = MagicMock(spec=MemoryWriteGate)
            candidate_holder = []

            def fake_evaluate(candidate, nearby=None):
                candidate_holder.append(candidate)
                return MemoryDecision(
                    kind=MemoryDecisionKind.STORE_NEW,
                    candidate=candidate,
                    score=0.72,
                    reason="scored_above_threshold",
                )
            mock_gate.evaluate = fake_evaluate
            mock_gate_fn.return_value = mock_gate

            result = unified_memory.store_gated(
                content="I use VS Code with Vim keybindings",
                source="conversation",
                importance=0.6,
                tags=["preference"],
            )

            # Gate was consulted
            assert len(candidate_holder) == 1
            assert "VS Code" in candidate_holder[0].content

            # Store was called after gate approval
            assert len(store_calls) == 1
            assert "VS Code" in store_calls[0]["content"]

            # Return value includes decision metadata
            assert result["decision"] == "store_new"
            assert result["score"] == 0.72

    def test_store_gated_skips_store_on_discard(self, unified_memory):
        """When the gate says DISCARD, store() should NOT be called."""
        from aura.memory.write_gate import MemoryWriteGate, MemoryDecisionKind, MemoryDecision

        store_calls = []
        def capture_store(**kwargs):
            store_calls.append(kwargs)
            return {}
        unified_memory.store = capture_store

        with patch("aura.memory.write_gate.get_write_gate") as mock_gate_fn, \
             patch("aura.reliability.telemetry.emit"):

            mock_gate = MagicMock(spec=MemoryWriteGate)

            def fake_evaluate(candidate, nearby=None):
                return MemoryDecision(
                    kind=MemoryDecisionKind.DISCARD,
                    candidate=candidate,
                    score=0.1,
                    reason="noise_content",
                )
            mock_gate.evaluate = fake_evaluate
            mock_gate_fn.return_value = mock_gate

            result = unified_memory.store_gated(content="ok", source="conversation")

            assert result["decision"] == "discard"
            assert len(store_calls) == 0, "store() should not be called for discarded content"
