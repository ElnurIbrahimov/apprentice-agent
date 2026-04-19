"""Coverage for multi-agent orchestrator collaboration modes.

This fills a gap flagged in the multi-agent audit: modes were only tested via
Telegram integration tests, not the orchestrator directly.
"""

from unittest.mock import MagicMock

from aura.multi_agent.orchestrator import MultiAgentOrchestrator
from aura.multi_agent.protocol import (
    AgentMessage,
    AgentResult,
    CollaborationMode,
    RoutingDecision,
)


def _orch_with_routing(routing: RoutingDecision):
    """Create an orchestrator with mocked router returning the given decision."""
    llm = MagicMock(return_value="SYNTHESIZED")
    orch = MultiAgentOrchestrator(tool_registry={}, llm_func=llm)
    orch.router = MagicMock()
    orch.router.route = MagicMock(return_value=routing)
    orch.router.sanitize_for_prompt = lambda s: s  # passthrough
    return orch, llm


def _result(agent: str, text: str = "ok", success: bool = True) -> AgentResult:
    return AgentResult(success=success, response=text, agent=agent)


def test_single_mode_returns_agent_response_directly():
    orch, llm = _orch_with_routing(RoutingDecision(
        agents=["analyst"], mode=CollaborationMode.SINGLE,
        reasoning="", confidence=1.0,
    ))
    orch._execute_single = MagicMock(return_value=_result("analyst", "one-shot answer"))

    response = orch.chat("test query")

    assert response == "one-shot answer"
    llm.assert_not_called()  # No synthesis for single mode


def test_parallel_mode_synthesizes_via_llm():
    orch, llm = _orch_with_routing(RoutingDecision(
        agents=["analyst", "creative"], mode=CollaborationMode.PARALLEL,
        reasoning="", confidence=1.0,
    ))

    def fake_exec(name, msg):
        return _result(name, f"{name}-view")

    orch._execute_single = MagicMock(side_effect=fake_exec)

    response = orch.chat("what is truth")

    assert response == "SYNTHESIZED"
    llm.assert_called_once()
    # Synthesis prompt should contain both agent responses
    synthesis_user_prompt = llm.call_args[0][1]
    assert "analyst-view" in synthesis_user_prompt
    assert "creative-view" in synthesis_user_prompt


def test_sequential_mode_stops_on_first_failure():
    orch, _ = _orch_with_routing(RoutingDecision(
        agents=["searcher", "coder"], mode=CollaborationMode.SEQUENTIAL,
        reasoning="", confidence=1.0,
    ))

    call_order = []

    def fake_exec(name, msg):
        call_order.append(name)
        return _result(name, "err", success=False) if name == "searcher" else _result(name)

    orch._execute_single = MagicMock(side_effect=fake_exec)

    orch.chat("find and fix")

    assert call_order == ["searcher"], "coder should not run after searcher fails"


def test_debate_mode_returns_final_revision_when_three_rounds_succeed():
    orch, _ = _orch_with_routing(RoutingDecision(
        agents=["analyst", "creative"], mode=CollaborationMode.DEBATE,
        reasoning="", confidence=1.0,
    ))

    responses = iter([
        _result("analyst", "proposal"),
        _result("creative", "critique"),
        _result("analyst", "revised"),
    ])
    orch._execute_single = MagicMock(side_effect=lambda n, m: next(responses))

    response = orch.chat("weigh options")

    assert response == "revised"


def test_synthesize_handles_all_failures():
    """If every agent fails in PARALLEL mode, return the errors joined."""
    orch, _ = _orch_with_routing(RoutingDecision(
        agents=["analyst", "coder"], mode=CollaborationMode.PARALLEL,
        reasoning="", confidence=1.0,
    ))
    orch._execute_single = MagicMock(side_effect=lambda n, m: _result(n, "boom", success=False))

    response = orch.chat("q")

    assert "All agents encountered errors" in response
    assert "boom" in response
