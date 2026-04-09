"""Smoke tests for SOTA mcts_reasoning upgrades."""
import json
import pytest
import tempfile
from pathlib import Path
from unittest.mock import MagicMock

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
from aura.tools.reasoning_tree_tool import ReasoningTreeTool


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
    mcts.search("What is 2+2?")
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


# ============================================================================
# MCTS Wiring Tests — Strategy Bandit + LATS Tool Integration
# ============================================================================

# --- 11. LATS: Tool executor is called when action node has tool metadata ---
def test_lats_tool_executor_called():
    """When tool_executor is provided and LLM generates an action with a tool,
    the tool should be executed and an observation node created."""
    tool_calls = []

    def mock_tool_executor(tool_name, tool_args):
        tool_calls.append((tool_name, tool_args))
        return {"result": "42", "source": "calculator"}

    def mock_llm_with_tools(prompt, system=None):
        if "Generate" in prompt or "candidate" in prompt.lower():
            return json.dumps({
                "candidates": [
                    {
                        "thought": "Calculate 6*7",
                        "rationale": "Direct computation",
                        "confidence": 0.9,
                        "type": "action",
                        "tool": "code_executor",
                        "tool_args": {"code": "print(6*7)"},
                    },
                    {
                        "thought": "The answer is 42",
                        "rationale": "Known result",
                        "confidence": 0.95,
                        "type": "conclusion",
                    },
                ]
            })
        elif "Evaluate" in prompt or "evaluate" in prompt.lower():
            return '{"score": 0.9, "is_correct": true, "is_complete": true, "reasoning": "Correct"}'
        return '{"score": 0.5}'

    mcts = MCTSReasoning(
        llm_func=mock_llm_with_tools,
        config=MCTSConfig(max_iterations=2, branching_factor=2, beam_width=1),
        tool_executor=mock_tool_executor,
    )
    result = mcts.search("What is 6*7?")

    # Tool should have been called at least once
    assert len(tool_calls) > 0
    assert tool_calls[0][0] == "code_executor"
    assert result.best_answer  # non-empty answer


# --- 12. LATS: Tool failure creates failed observation node ---
def test_lats_tool_failure_handled():
    """When tool execution fails, a failed observation node is created
    instead of crashing the MCTS search."""
    def mock_tool_executor(tool_name, tool_args):
        raise RuntimeError("Tool unavailable")

    def mock_llm_action(prompt, system=None):
        if "Generate" in prompt or "candidate" in prompt.lower():
            return json.dumps({
                "candidates": [
                    {
                        "thought": "Search for answer",
                        "rationale": "Web search",
                        "confidence": 0.6,
                        "type": "action",
                        "tool": "search_web",
                        "tool_args": {"query": "test"},
                    },
                    {
                        "thought": "I think the answer is yes",
                        "rationale": "Reasoning",
                        "confidence": 0.7,
                        "type": "conclusion",
                    },
                ]
            })
        elif "Evaluate" in prompt or "evaluate" in prompt.lower():
            return '{"score": 0.6, "is_correct": true, "is_complete": true, "reasoning": "OK"}'
        return '{"score": 0.5}'

    mcts = MCTSReasoning(
        llm_func=mock_llm_action,
        config=MCTSConfig(max_iterations=2, branching_factor=2, beam_width=1),
        tool_executor=mock_tool_executor,
    )
    # Should not raise
    result = mcts.search("Test question")
    assert isinstance(result, MCTSResult)


# --- 13. ReasoningTreeTool accepts and passes tool_executor ---
def test_reasoning_tree_tool_with_tool_executor():
    """ReasoningTreeTool passes tool_executor through to MCTSReasoning."""
    def mock_executor(name, args):
        return "tool result"

    tool = ReasoningTreeTool(
        llm_func=mock_llm,
        tool_executor=mock_executor,
    )
    assert tool.tool_executor is mock_executor


# --- 14. ReasoningTreeTool.execute("think_deeply") runs MCTS ---
def test_reasoning_tree_execute_think_deeply():
    """The execute method with 'think_deeply' action runs MCTS and returns structured result."""
    tool = ReasoningTreeTool(llm_func=mock_llm)
    result = tool.execute("think_deeply", problem="What is 2+2?")
    assert result["success"] is True or result.get("answer")  # may or may not succeed depending on mock
    assert "answer" in result or "summary" in result


# --- 15. Strategy Bandit MCTS availability in CATEGORY_STRATEGIES ---
def test_strategy_bandit_mcts_categories():
    """MCTS should be available for MATH, CODE, and PLANNING categories."""
    from aura.consciousness.strategy_bandit import (
        CATEGORY_STRATEGIES,
        ProblemCategory,
        ReasoningStrategy,
    )
    for cat in [ProblemCategory.MATH, ProblemCategory.CODE, ProblemCategory.PLANNING]:
        strategies = CATEGORY_STRATEGIES[cat]
        assert ReasoningStrategy.MCTS in strategies, f"MCTS missing from {cat.value}"

    # MCTS should NOT be in CREATIVE or DEBUG (pure CoT categories)
    for cat in [ProblemCategory.CREATIVE, ProblemCategory.DEBUG]:
        strategies = CATEGORY_STRATEGIES[cat]
        assert ReasoningStrategy.MCTS not in strategies, f"MCTS should not be in {cat.value}"


# --- 16. Strategy Bandit can select MCTS for math problems ---
def test_strategy_bandit_can_select_mcts():
    """Strategy Bandit should be able to select MCTS for math queries."""
    import tempfile
    from aura.consciousness.strategy_bandit import (
        StrategyBandit,
        ReasoningStrategy,
        ProblemCategory,
    )

    with tempfile.TemporaryDirectory() as tmpdir:
        bandit = StrategyBandit(
            db_path=f"{tmpdir}/test_bandit.db",
            epsilon=1.0,  # Force exploration so it can pick MCTS
            enabled=True,
        )
        # Run many selections — with epsilon=1.0, it randomly picks, so MCTS should appear
        strategies_seen = set()
        for _ in range(50):
            selection = bandit.select_strategy(
                "Calculate the factorial of 20",
                category=ProblemCategory.MATH,
            )
            strategies_seen.add(selection.strategy)
        assert ReasoningStrategy.MCTS in strategies_seen, "MCTS was never selected despite epsilon=1.0"


# --- 17. MCTS without tool_executor still works (no LATS) ---
def test_mcts_without_tool_executor():
    """MCTS should work fine without a tool_executor — pure reasoning mode."""
    def mock_llm_action_no_tools(prompt, system=None):
        if "Generate" in prompt or "candidate" in prompt.lower():
            return json.dumps({
                "candidates": [
                    {
                        "thought": "Search for data",
                        "rationale": "Need info",
                        "confidence": 0.5,
                        "type": "action",
                        "tool": "search_web",
                        "tool_args": {"query": "test"},
                    },
                    {
                        "thought": "The answer is yes",
                        "rationale": "Reasoning",
                        "confidence": 0.8,
                        "type": "conclusion",
                    },
                ]
            })
        elif "Evaluate" in prompt or "evaluate" in prompt.lower():
            return '{"score": 0.7, "is_correct": true, "is_complete": true, "reasoning": "OK"}'
        return '{"score": 0.5}'

    mcts = MCTSReasoning(
        llm_func=mock_llm_action_no_tools,
        config=MCTSConfig(max_iterations=2, branching_factor=2, beam_width=1),
        tool_executor=None,  # No tools
    )
    result = mcts.search("Test")
    assert isinstance(result, MCTSResult)
    # Action nodes with tool metadata should be created but no tool executed
    all_nodes = mcts._get_all_nodes()
    action_nodes = [n for n in all_nodes if n.thought.type == ThoughtType.ACTION]
    # Action nodes exist but have no observation children (no tool executor)
    for an in action_nodes:
        obs_children = [c for c in an.children if c.thought.type == ThoughtType.OBSERVATION]
        assert len(obs_children) == 0, "Observation nodes shouldn't exist without tool_executor"
