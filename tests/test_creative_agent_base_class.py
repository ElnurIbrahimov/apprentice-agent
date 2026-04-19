"""Validate the CreativeAgent base class consolidation (#5 fix)."""


def test_creative_extends_simple_specialist_not_tool_using():
    """CreativeAgent has no tools, so it should extend SimpleSpecialist."""
    from aura.multi_agent.base_agent import SimpleSpecialist, ToolUsingSpecialist
    from aura.multi_agent.specialists import CreativeAgent

    assert issubclass(CreativeAgent, SimpleSpecialist)
    # SimpleSpecialist is itself a BaseSpecialist, ToolUsingSpecialist is a sibling.
    # CreativeAgent should NOT be a ToolUsingSpecialist anymore.
    assert not issubclass(CreativeAgent, ToolUsingSpecialist)


def test_creative_agent_has_no_tools():
    from aura.multi_agent.specialists import CreativeAgent
    agent = CreativeAgent(tool_registry={})
    # The class-level tools list is inherited from BaseSpecialist (empty default)
    assert agent.tools == []
    assert agent._available_tools == {}


def test_creative_execute_works_without_tools():
    """Creative agent should just call the LLM directly."""
    from aura.multi_agent.protocol import AgentMessage
    from aura.multi_agent.specialists import CreativeAgent

    agent = CreativeAgent(tool_registry={})
    msg = AgentMessage(content="brainstorm 3 ideas for a dog biscuit brand", sender="user")
    result = agent.execute(msg, llm_func=lambda sys, user: "1. Paw & Order\n2. Bark Bites\n3. Doggo")
    assert result.success
    assert "Paw" in result.response
    assert result.tools_used == []
