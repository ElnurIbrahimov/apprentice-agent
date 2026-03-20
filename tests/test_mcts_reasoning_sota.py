"""Smoke tests for SOTA mcts_reasoning upgrades."""
import pytest
import tempfile
from pathlib import Path

from aura.tools.mcts_reasoning import (
    MCTSReasoning,
    MCTSConfig,
    MCTSResult,
    MCTSNode,
    ThoughtType,
    NodeState,
    TreeCache,
    mcts_reason,
)


def mock_llm(prompt, system=None):
    if "Generate" in prompt or "candidate" in prompt.lower():
        return '{"candidates": [{"thought": "Let me think about this", "rationale": "Starting analysis", "confidence": 0.7, "type": "reasoning"}, {"thought": "The answer is 4", "rationale": "Simple addition", "confidence": 0.9, "type": "conclusion"}]}'
    elif "Evaluate" in prompt or "evaluate" in prompt.lower():
        return '{"score": 0.8, "is_correct": true, "is_complete": true, "reasoning": "Good reasoning"}'
    elif "reflect" in prompt.lower() or "Analyze" in prompt:
        return '{"critique": "Could be better", "lessons": ["Try harder"], "alternatives": ["Different approach"]}'
    return '{"score": 0.5}'


# --- 1. MCTSConfig defaults ---
def test_config_defaults():
    cfg = MCTSConfig()
    assert cfg.max_iterations == 30
    assert cfg.max_depth == 10
    assert cfg.branching_factor == 5
    assert cfg.exploration_weight == pytest.approx(1.414)
    assert cfg.max_token_budget == 50000
    assert cfg.beam_width == 3
    assert cfg.max_json_retries == 2
    assert cfg.stagnation_window == 5


# --- 2. MCTSConfig with new SOTA params ---
def test_config_sota_params():
    cfg = MCTSConfig(max_token_budget=10000, beam_width=2)
    assert cfg.max_token_budget == 10000
    assert cfg.beam_width == 2
    # other defaults still intact
    assert cfg.max_iterations == 30


# --- 3. mcts_reason convenience function ---
def test_mcts_reason_returns_result():
    result = mcts_reason("What is 2+2?", mock_llm, max_iterations=2)
    assert isinstance(result, MCTSResult)
    assert result.iterations <= 2
    assert result.best_answer  # non-empty
    assert result.time_taken >= 0


# --- 4. MCTSResult metadata has total_tokens_used ---
def test_result_metadata_tokens():
    result = mcts_reason("What is 2+2?", mock_llm, max_iterations=2)
    assert "total_tokens_used" in result.metadata
    assert result.metadata["total_tokens_used"] > 0


# --- 5. MCTSReasoning instantiates with just llm_func ---
def test_reasoning_init_minimal():
    mcts = MCTSReasoning(llm_func=mock_llm)
    assert mcts.llm is mock_llm
    assert isinstance(mcts.config, MCTSConfig)
    assert mcts.tool_executor is None


# --- 6. get_current_best before and after search ---
def test_get_current_best():
    mcts = MCTSReasoning(llm_func=mock_llm, config=MCTSConfig(max_iterations=2))
    # Before search
    assert mcts.get_current_best() is None
    # After search
    mcts.search("What is 2+2?")
    best = mcts.get_current_best()
    assert best is not None
    assert isinstance(best, MCTSResult)
    assert best.confidence >= 0.0


# --- 7. TreeCache save and load ---
def test_tree_cache_roundtrip():
    with tempfile.TemporaryDirectory() as tmpdir:
        cache = TreeCache()
        cache.CACHE_DIR = Path(tmpdir) / "cache"
        cache.CACHE_DIR.mkdir(parents=True, exist_ok=True)

        from aura.tools.mcts_reasoning import Thought
        root_thought = Thought(type=ThoughtType.ROOT, content="test problem", confidence=1.0)
        root = MCTSNode(thought=root_thought)
        root.visits = 5
        root.value = 3.5

        cache.save_tree("test problem", root)
        loaded = cache.load_tree("test problem")

        assert loaded is not None
        assert loaded["problem"] == "test problem"
        assert loaded["tree"]["visits"] == 5
        assert loaded["tree"]["thought"]["content"] == "test problem"

        # Non-existent problem returns None
        assert cache.load_tree("nonexistent") is None


# --- 8. _estimate_tokens returns reasonable value ---
def test_estimate_tokens():
    mcts = MCTSReasoning(llm_func=mock_llm)
    tokens = mcts._estimate_tokens("hello world")
    # "hello world" = 11 chars -> ~2-3 tokens at 4 chars/token
    assert 1 <= tokens <= 5
    # Longer text scales proportionally
    long_tokens = mcts._estimate_tokens("a" * 400)
    assert long_tokens == 100


# --- 9. Old callback signature (2 args) still works ---
def test_old_callback_signature():
    calls = []

    def old_callback(iteration, root):
        calls.append((iteration, root))

    mcts = MCTSReasoning(llm_func=mock_llm, config=MCTSConfig(max_iterations=2))
    mcts.on_iteration_complete = old_callback
    result = mcts.search("What is 2+2?")
    # Should not raise TypeError, and callback should have been called
    assert len(calls) > 0
    assert calls[0][0] == 1  # first iteration


# --- 10. All imports work (already verified by top-level import) ---
def test_all_imports_exist():
    assert MCTSReasoning is not None
    assert MCTSConfig is not None
    assert MCTSResult is not None
    assert MCTSNode is not None
    assert ThoughtType is not None
    assert NodeState is not None
    assert mcts_reason is not None
    # Verify enum members
    assert ThoughtType.ROOT.value == "root"
    assert NodeState.PRUNED.value == "pruned"
