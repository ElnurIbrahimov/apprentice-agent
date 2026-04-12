from __future__ import annotations

from unittest.mock import MagicMock


def test_run_returns_status_for_model_timeout():
    from aura.core.agentic_loop import AgenticLoop

    brain = MagicMock()
    brain.get_session_stats.return_value = {"cost_usd": 0.0}
    brain.think_with_tools_stream.side_effect = TimeoutError("too slow")
    brain._model_override = None

    loop = AgenticLoop(brain=brain, max_iterations=2)
    loop._build_system_prompt = lambda prompt: "system"
    loop._inject_smart_context = lambda prompt, system_prompt: system_prompt
    loop._get_active_tools = lambda: []

    result = loop.run("Do something")

    assert "status" in result
    assert result["status"] == "model_timeout"
    assert result["success"] is False
