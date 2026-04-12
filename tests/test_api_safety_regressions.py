"""Focused regressions for API helper safety and model defaults."""

from api.routes.auth import _pkce_redirect_put, _pkce_redirect_store
from api.routes.knowledge import SaveRequest
from api.routes.memory import RecallEventResponse
from api.routes.proactive import ProactiveMessageResponse


def test_pkce_redirect_store_evicts_at_capacity():
    _pkce_redirect_store.clear()
    for i in range(105):
        _pkce_redirect_put(f"state-{i}", f"https://example.com/{i}")

    assert len(_pkce_redirect_store) <= 100
    _pkce_redirect_store.clear()


def test_knowledge_save_request_tags_are_not_shared():
    first = SaveRequest(text="one")
    second = SaveRequest(text="two")

    first.tags.append("important")
    assert second.tags == []


def test_memory_recall_event_response_metadata_is_not_shared():
    first = RecallEventResponse(
        id="a",
        source="amem",
        count=1,
        query="hello",
        memories=[],
        timestamp="2026-01-01T00:00:00",
    )
    second = RecallEventResponse(
        id="b",
        source="rag",
        count=2,
        query="world",
        memories=[],
        timestamp="2026-01-01T00:00:01",
    )

    first.metadata["seen"] = True
    assert second.metadata == {}


def test_proactive_message_response_metadata_is_not_shared():
    first = ProactiveMessageResponse(
        action="notify",
        content="hello",
        priority="normal",
        timestamp="2026-01-01T00:00:00",
    )
    second = ProactiveMessageResponse(
        action="notify",
        content="world",
        priority="normal",
        timestamp="2026-01-01T00:00:01",
    )

    first.metadata["source"] = "test"
    assert second.metadata == {}
