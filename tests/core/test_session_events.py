from __future__ import annotations

from unittest.mock import MagicMock

from aura.core.agentic_loop import AgenticLoop
from aura.core.session import AgenticSession


def test_session_saves_and_loads_bounded_event_trace(tmp_path):
    session = AgenticSession(sessions_dir=str(tmp_path))
    session.new(project_root=str(tmp_path), model="test-model")

    for idx in range(205):
        session.append_event(
            {
                "type": "tool_result",
                "run_id": f"run_{idx // 10}",
                "iteration": idx,
                "payload": {"tool_result": "x" * 2505, "index": idx},
            }
        )

    session.save()

    loaded = AgenticSession(sessions_dir=str(tmp_path))
    messages = loaded.load(session.session_id)

    assert messages == []
    assert len(loaded.events) == 200
    assert loaded.events[0]["iteration"] == 5
    assert loaded.events[0]["run_id"] == "run_0"
    assert len(loaded.events[-1]["payload"]["tool_result"]) == 2000
    summary = loaded.list_sessions(limit=1)[0]
    assert summary["event_count"] == 200


def test_agentic_loop_persists_structured_events_without_chunk_noise(tmp_path):
    brain = MagicMock()
    brain.get_session_stats.return_value = {"cost_usd": 0.0}
    brain._model_override = None
    brain.think_with_tools_stream.return_value = iter(
        [
            ("content", "Hello"),
            ("done", {"model": "test-model", "content": "Hello"}),
        ]
    )

    session = AgenticSession(sessions_dir=str(tmp_path))
    session.new(project_root=str(tmp_path), model="test-model")

    loop = AgenticLoop(brain=brain, project_root=str(tmp_path), max_iterations=1, session=session)
    loop._build_system_prompt = lambda prompt: "system"
    loop._inject_smart_context = lambda prompt, system_prompt: system_prompt
    loop._get_active_tools = lambda: []

    seen = []
    result = loop.run("Say hello", on_event=seen.append)

    assert result["status"] == "completed"
    assert [event.type for event in seen] == ["chunk", "response", "run_finished"]
    assert len({event.run_id for event in seen}) == 1
    assert [event["type"] for event in session.events] == ["response", "run_finished"]
    assert len({event["run_id"] for event in session.events}) == 1
    assert session.events[-1]["payload"]["status"] == "completed"
