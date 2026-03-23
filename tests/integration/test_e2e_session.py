"""End-to-end interactive session tests for Aura CLI.

Exercises the full flow: multi-turn chat, command dispatch, conversation
forking, chain execution, cancellation, debate, and plan generation —
all against the mock Ollama server so no real LLM calls are made.
"""

import os
import sys
import threading
from pathlib import Path

import pytest

# Ensure project root is on sys.path
_PROJECT_ROOT = os.path.dirname(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
if _PROJECT_ROOT not in sys.path:
    sys.path.insert(0, _PROJECT_ROOT)


# ---------------------------------------------------------------------------
# Test 1: Full multi-turn chat session
# ---------------------------------------------------------------------------
class TestFullChatSession:
    def test_multi_turn_conversation(self, mock_brain):
        """Simulate a multi-turn chat session through AgenticLoop."""
        from aura.core.agentic_loop import AgenticLoop

        loop = AgenticLoop(brain=mock_brain, max_iterations=5)

        # Turn 1: Simple prompt
        result1 = loop.run("Hello, what can you do?")
        assert isinstance(result1, dict)
        assert result1.get("response")
        assert result1.get("success") is not False

        # Turn 2: Follow-up (history should carry over)
        result2 = loop.run("Tell me more about that")
        assert isinstance(result2, dict)
        assert result2.get("response")

        # Verify conversation history grew (2 user + 2 assistant = 4)
        assert len(loop._conversation_history) >= 4

    def test_conversation_history_persists_across_turns(self, mock_brain):
        """Each turn appends user + assistant to _conversation_history."""
        from aura.core.agentic_loop import AgenticLoop

        loop = AgenticLoop(brain=mock_brain, max_iterations=2)

        loop.run("First message")
        assert len(loop._conversation_history) == 2  # user + assistant

        loop.run("Second message")
        assert len(loop._conversation_history) == 4

        loop.run("Third message")
        assert len(loop._conversation_history) == 6

        # Verify message roles alternate correctly
        roles = [m["role"] for m in loop._conversation_history]
        assert roles == ["user", "assistant", "user", "assistant", "user", "assistant"]


# ---------------------------------------------------------------------------
# Test 2: Command dispatch in session context
# ---------------------------------------------------------------------------
class TestCommandsInSession:
    def test_help_does_not_crash(self, mock_agent):
        """/help should run without error."""
        from aura.cli.commands import handle_command
        handle_command(mock_agent, "/help", speak=False)

    def test_context_does_not_crash(self, mock_agent):
        """/context should run without error (may say 'not available')."""
        from aura.cli.commands import handle_command
        handle_command(mock_agent, "/context", speak=False)

    def test_cost_does_not_crash(self, mock_agent, capsys):
        """/cost should print session cost info."""
        from aura.cli.commands import handle_command
        handle_command(mock_agent, "/cost", speak=False)
        captured = capsys.readouterr()
        assert "Session Cost" in captured.out

    def test_model_with_arg_does_not_crash(self, mock_agent):
        """/model with an explicit model name should work without interactive prompt."""
        from aura.cli.commands import handle_command
        handle_command(mock_agent, "/model test-model:latest", speak=False)
        # Verify it actually changed
        assert mock_agent.brain._model_override == "test-model:latest"

    def test_unknown_command_prints_error(self, mock_agent, capsys):
        """Unknown command should print error, not crash."""
        from aura.cli.commands import handle_command
        handle_command(mock_agent, "/nonexistent_xyz_123", speak=False)
        captured = capsys.readouterr()
        assert "Unknown command" in captured.out


# ---------------------------------------------------------------------------
# Test 3: Fork → checkout → merge flow
# ---------------------------------------------------------------------------
class TestForkCheckoutMerge:
    def test_fork_creates_branch(self, tmp_path):
        """Forking creates a new branch with a copy of history."""
        from aura.core.conversation_fork import ConversationTree

        tree = ConversationTree(session_dir=tmp_path)
        # Seed main with some history
        tree.branches["main"].history = [
            {"role": "user", "content": "Hello"},
            {"role": "assistant", "content": "Hi there"},
        ]

        branch = tree.fork("experiment")
        assert branch.name == "experiment"
        assert branch.parent_id == "main"
        assert len(branch.history) == 2
        assert tree.current_branch == branch.id

    def test_fork_is_independent(self, tmp_path):
        """Changes on fork don't affect main."""
        from aura.core.conversation_fork import ConversationTree

        tree = ConversationTree(session_dir=tmp_path)
        tree.branches["main"].history = [
            {"role": "user", "content": "msg1"},
        ]

        branch = tree.fork("experiment")
        branch.history.append({"role": "user", "content": "fork-only msg"})

        # Main should still have only 1 message
        main = tree.branches["main"]
        assert len(main.history) == 1
        assert len(branch.history) == 2

    def test_merge_brings_new_messages_to_parent(self, tmp_path):
        """Merging appends fork-only messages back to parent."""
        from aura.core.conversation_fork import ConversationTree

        tree = ConversationTree(session_dir=tmp_path)
        tree.branches["main"].history = [
            {"role": "user", "content": "original"},
        ]

        branch = tree.fork("experiment")
        branch.history.append({"role": "user", "content": "new on fork"})

        result = tree.merge_to_parent()
        assert result["merged"] == 1
        assert result["target"] == "main"

        # Main should now have both messages
        main = tree.branches["main"]
        assert len(main.history) == 2
        assert main.history[1]["content"] == "new on fork"

        # Should have switched back to main
        assert tree.current_branch == "main"

    def test_checkout_switches_branch(self, tmp_path):
        """switch() moves to the target branch."""
        from aura.core.conversation_fork import ConversationTree

        tree = ConversationTree(session_dir=tmp_path)
        tree.fork("experiment")
        assert tree.current_branch != "main"

        tree.switch("main")
        assert tree.current_branch == "main"

    def test_full_fork_work_checkout_merge_flow(self, mock_brain, tmp_path):
        """End-to-end: run on main, fork, add work, checkout main, merge."""
        from aura.core.agentic_loop import AgenticLoop
        from aura.core.conversation_fork import ConversationTree

        loop = AgenticLoop(brain=mock_brain, max_iterations=2)

        # Send a message on main
        loop.run("Initial message")
        initial_len = len(loop._conversation_history)
        assert initial_len == 2  # user + assistant

        # Create tree and sync main's history
        tree = ConversationTree(session_dir=tmp_path)
        tree.branches["main"].history = list(loop._conversation_history)

        # Fork
        branch = tree.fork("experiment")
        assert len(branch.history) == initial_len

        # Simulate work on fork
        branch.history.append({"role": "user", "content": "Fork work"})
        branch.history.append({"role": "assistant", "content": "Fork response"})
        assert len(branch.history) == initial_len + 2

        # Main unchanged
        main = tree.branches["main"]
        assert len(main.history) == initial_len

        # Merge back
        result = tree.merge_to_parent()
        assert result["merged"] == 2
        assert len(main.history) == initial_len + 2


# ---------------------------------------------------------------------------
# Test 4: Chain execution flow
# ---------------------------------------------------------------------------
class TestChainExecution:
    def test_parse_chain_basic(self):
        """parse_chain splits on -> correctly."""
        from aura.cli.chain_mode import parse_chain

        steps = parse_chain("step one -> step two -> step three")
        assert len(steps) == 3
        assert steps[0].prompt_template == "step one"
        assert steps[1].prompt_template == "step two"
        assert steps[2].prompt_template == "step three"

    def test_parse_chain_with_model_override(self):
        """parse_chain extracts @model suffixes."""
        from aura.cli.chain_mode import parse_chain

        steps = parse_chain("research X @kimi-k2.5:cloud -> summarize @nemotron-3-super:cloud")
        assert len(steps) == 2
        assert steps[0].model == "kimi-k2.5:cloud"
        assert steps[1].model == "nemotron-3-super:cloud"

    def test_run_chain_executes_all_steps(self, mock_brain):
        """run_chain should produce a result for each step."""
        from aura.cli.chain_mode import parse_chain, run_chain

        steps = parse_chain("step one -> step two -> step three")
        assert len(steps) == 3

        result = run_chain(mock_brain, steps)
        assert len(result.step_results) == 3
        assert all(r["response"] for r in result.step_results)
        assert result.success is True

    def test_chain_context_forwarding(self, mock_brain):
        """Each step should receive prior output as context."""
        from aura.cli.chain_mode import parse_chain, run_chain

        steps = parse_chain("first -> second with {prev}")
        result = run_chain(mock_brain, steps)
        assert len(result.step_results) == 2
        # The second step's prompt should have been expanded
        # (mock returns same response regardless, but the chain ran)
        assert result.success is True

    def test_chain_on_step_callback(self, mock_brain):
        """on_step callback is called for each step."""
        from aura.cli.chain_mode import parse_chain, run_chain

        steps = parse_chain("a -> b -> c")
        callback_calls = []

        def on_step(step_num, total, result_dict):
            callback_calls.append((step_num, total))

        run_chain(mock_brain, steps, on_step=on_step)
        assert callback_calls == [(1, 3), (2, 3), (3, 3)]


# ---------------------------------------------------------------------------
# Test 5: Cancellation flow
# ---------------------------------------------------------------------------
class TestCancellation:
    def test_cancel_before_run_stops_immediately(self, mock_brain):
        """Setting cancel before run() should produce a cancelled response."""
        from aura.core.agentic_loop import AgenticLoop

        loop = AgenticLoop(brain=mock_brain, max_iterations=50)
        # Cancel is cleared at the start of run(), so we need to cancel from a thread
        # after run() has started but before it completes the first iteration.
        cancel_timer = threading.Timer(0.1, loop.cancel)
        cancel_timer.start()
        try:
            result = loop.run("This should be cancelled eventually")
        finally:
            cancel_timer.cancel()

        assert isinstance(result, dict)
        # Should have stopped well before 50 iterations
        assert result.get("iterations", 0) < 50

    def test_cancel_event_mechanism(self, mock_brain):
        """cancel() sets the internal event, run() clears it at start."""
        from aura.core.agentic_loop import AgenticLoop

        loop = AgenticLoop(brain=mock_brain, max_iterations=5)
        assert not loop._cancel_event.is_set()
        loop.cancel()
        assert loop._cancel_event.is_set()

        # run() should clear the event at start and proceed normally
        result = loop.run("normal run after cancel was set")
        assert result.get("response")
        assert result.get("success") is not False


# ---------------------------------------------------------------------------
# Test 6: Debate flow
# ---------------------------------------------------------------------------
class TestDebateFlow:
    def test_debate_completes_with_synthesis(self, mock_brain, monkeypatch):
        """Debate should complete with positions and synthesis."""
        from aura.cli.debate_mode import run_debate

        # Mock _chatgpt_client as None so it uses fallback models
        monkeypatch.setattr(mock_brain, "_chatgpt_client", None, raising=False)

        # Override the debater models to use our test model
        import aura.cli.debate_mode as dm
        monkeypatch.setattr(dm, "FALLBACK_MODELS", {
            "advocate": "test-model:latest",
            "critic": "test-model:latest",
        })

        result = run_debate(mock_brain, "SQLite vs Postgres?")
        assert result.synthesis
        assert len(result.positions) >= 2
        assert all(p.done for p in result.positions)

    def test_debate_parse_args(self):
        """parse_debate_args handles various input formats."""
        from aura.cli.debate_mode import parse_debate_args

        q, m = parse_debate_args("SQLite vs Postgres?")
        assert q == "SQLite vs Postgres?"
        assert m is None

        q, m = parse_debate_args("--models kimi,deepseek which is better?")
        assert q == "which is better?"
        assert m == "kimi,deepseek"


# ---------------------------------------------------------------------------
# Test 7: Plan-first flow
# ---------------------------------------------------------------------------
class TestPlanFirst:
    def test_plan_generation(self, mock_brain):
        """plan_first should return a dict with plan_text."""
        from aura.core.agentic_loop import AgenticLoop

        loop = AgenticLoop(brain=mock_brain, max_iterations=5)
        result = loop.plan_first("Refactor the auth module")

        assert isinstance(result, dict)
        assert "plan_text" in result
        assert "plan" in result
        assert "prompt" in result
        assert result["prompt"] == "Refactor the auth module"
        # plan_text should be non-empty (the mock returns a response)
        assert result["plan_text"]

    def test_plan_contains_parsed_plan(self, mock_brain):
        """The parsed plan object should be returned."""
        from aura.core.agentic_loop import AgenticLoop

        loop = AgenticLoop(brain=mock_brain, max_iterations=5)
        result = loop.plan_first("Build a REST API")

        # The mock returns a canned string that won't perfectly parse,
        # but parse_plan_from_llm should still return an ExecutionPlan
        assert result["plan"] is not None
