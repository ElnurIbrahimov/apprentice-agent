"""Functional smoke tests for api/routes/tools_new.py.

All 38 endpoints exercised at least once. For happy paths we mock
``api.routes.tools_new.call_tool`` and ``api.routes.tools_new.get_agent``
so no real tool I/O happens. Validation-failure paths assert 422.
"""

from __future__ import annotations

import os
import tempfile
from pathlib import Path
from unittest.mock import MagicMock, patch

import pytest

pytest.importorskip("fastapi")

from fastapi.testclient import TestClient


# ---------------------------------------------------------------------------
# Fixtures
# ---------------------------------------------------------------------------

@pytest.fixture
def client():
    """Auth-disabled TestClient hitting the real FastAPI app."""
    env = {
        "AURA_API_AUTH_ENABLED": "false",
        "AURA_ENV": "development",
    }
    with patch.dict(os.environ, env, clear=False):
        from api.main import app
        yield TestClient(app, raise_server_exceptions=False)


@pytest.fixture
def mock_agent():
    """A MagicMock shaped like ``agent`` — every tool access returns another MagicMock
    that in turn returns ``{"success": True}`` from every call."""
    agent = MagicMock()
    # agent.tools acts like a dict where every key returns a MagicMock tool.
    tool_store: dict[str, MagicMock] = {}

    def _tool_factory(name):
        if name not in tool_store:
            tool = MagicMock()
            # Explicitly set common methods so MagicMock returns dicts.
            # (Default MagicMock attribute access returns another MagicMock,
            # which is truthy but not a dict — routes call .get('success').)
            for meth in (
                "today", "upcoming", "add_event", "remove_event",
                "review", "answer", "list_decks", "add_card",
                "get_config_status", "fetch_emails", "send_email",
                "read_screen", "get_active_window",
                "run", "list_sessions",
                "list_tasks", "board", "add_task", "update_task",
                "remove_task", "overdue",
                "request", "history",
                "query", "schema", "import_csv",
                "transcribe", "list_transcripts", "status",
                "list_history", "capture", "search", "stats",
                "list_research", "save",
            ):
                setattr(tool, meth, MagicMock(return_value={"success": True, "tool": name, "method": meth}))
            tool_store[name] = tool
        return tool_store[name]

    # dict-like tools mapping
    tools_dict = MagicMock()
    tools_dict.__contains__ = lambda self, key: True  # every tool "exists"
    tools_dict.__getitem__ = lambda self, key: _tool_factory(key)
    agent.tools = tools_dict
    return agent


@pytest.fixture
def patched_tools(mock_agent):
    """Patch both call_tool and get_agent for the duration of a test."""
    with patch("api.routes.tools_new.call_tool", return_value={"success": True, "tool": "call_tool"}) as ct, \
         patch("api.routes.tools_new.get_agent", return_value=mock_agent) as ga:
        yield ct, ga


# ---------------------------------------------------------------------------
# Parametrized GET smoke tests — one case per no-body GET endpoint
# ---------------------------------------------------------------------------

_GET_PATHS = [
    "/api/calendar/today",
    "/api/calendar/upcoming",
    "/api/calendar/upcoming?days=14",
    "/api/flashcards/due",
    "/api/flashcards/stats",
    "/api/email/status",
    "/api/email/inbox",
    "/api/email/inbox?limit=5",
    "/api/screen/read",
    "/api/screen/active-window",
    "/api/shell/sessions",
    "/api/tasks/list",
    "/api/tasks/list?status=open",
    "/api/tasks/list?project=aura",
    "/api/tasks/board",
    "/api/tasks/overdue",
    "/api/api-tester/history",
    "/api/api-tester/history?limit=50",
    "/api/database/schema",
    "/api/database/schema?db=default",
    "/api/audio/transcripts",
    "/api/audio/status",
    "/api/clipboard/history",
    "/api/clipboard/history?limit=5",
    "/api/clipboard/search?query=foo",
    "/api/clipboard/stats",
    "/api/research/list",
    "/api/research/list?category=tools",
    "/api/research/search?query=aura",
    "/api/research/stats",
    "/api/research/skills",
]


@pytest.mark.parametrize("path", _GET_PATHS)
def test_get_endpoints_return_200(client, patched_tools, path):
    """Every GET smoke-call returns 200 with a dict body."""
    response = client.get(path)
    assert response.status_code == 200, f"{path} -> {response.status_code}: {response.text}"
    body = response.json()
    assert isinstance(body, dict)


# ---------------------------------------------------------------------------
# CALENDAR
# ---------------------------------------------------------------------------

class TestCalendarRoutes:
    def test_calendar_add_happy_path(self, client, patched_tools):
        response = client.post(
            "/api/calendar/add",
            json={"title": "Standup", "start": "2026-04-13T09:00"},
        )
        assert response.status_code == 200
        assert response.json().get("success") is True

    def test_calendar_add_missing_title_returns_422(self, client, patched_tools):
        response = client.post("/api/calendar/add", json={"start": "2026-04-13T09:00"})
        assert response.status_code == 422

    def test_calendar_remove_valid_id(self, client, patched_tools):
        response = client.delete("/api/calendar/evt_abc_123")
        assert response.status_code == 200

    def test_calendar_remove_invalid_id_is_rejected(self, client, patched_tools):
        response = client.delete("/api/calendar/bad id with spaces")
        # Route either rejects with 4xx or returns {"success": False}
        assert response.status_code in (200, 404, 422)
        if response.status_code == 200:
            assert response.json().get("success") is False


# ---------------------------------------------------------------------------
# FLASHCARDS
# ---------------------------------------------------------------------------

class TestFlashcardRoutes:
    def test_flashcards_answer_happy_path(self, client, patched_tools):
        response = client.post(
            "/api/flashcards/answer",
            json={"card_id": "card_1", "quality": 4},
        )
        assert response.status_code == 200
        assert response.json().get("success") is True

    def test_flashcards_answer_missing_fields_returns_422(self, client, patched_tools):
        response = client.post("/api/flashcards/answer", json={"card_id": "c1"})
        assert response.status_code == 422

    def test_flashcards_add_happy_path(self, client, patched_tools):
        response = client.post(
            "/api/flashcards/add",
            json={"front": "Q", "back": "A", "deck": "default"},
        )
        assert response.status_code == 200


# ---------------------------------------------------------------------------
# EMAIL
# ---------------------------------------------------------------------------

class TestEmailRoutes:
    def test_email_send_happy_path(self, client, patched_tools):
        response = client.post(
            "/api/email/send",
            json={"to": "a@b.com", "subject": "hi", "body": "hello"},
        )
        assert response.status_code == 200

    def test_email_send_missing_to_returns_422(self, client, patched_tools):
        response = client.post(
            "/api/email/send", json={"subject": "hi", "body": "hello"},
        )
        assert response.status_code == 422


# ---------------------------------------------------------------------------
# SHELL
# ---------------------------------------------------------------------------

class TestShellRoutes:
    def test_shell_run_allowed_command(self, client, patched_tools):
        response = client.post(
            "/api/shell/run",
            json={"command": "ls -la", "timeout": 5},
        )
        assert response.status_code == 200
        body = response.json()
        # Allowlist passes "ls"; mocked shell_executor.run returns success.
        assert isinstance(body, dict)

    def test_shell_run_blocked_interpreter(self, client, patched_tools):
        response = client.post(
            "/api/shell/run",
            json={"command": "python -c 'print(1)'", "timeout": 5},
        )
        assert response.status_code == 200
        assert response.json().get("success") is False

    def test_shell_run_missing_command_returns_422(self, client, patched_tools):
        response = client.post("/api/shell/run", json={"timeout": 5})
        assert response.status_code == 422


# ---------------------------------------------------------------------------
# TASKS
# ---------------------------------------------------------------------------

class TestTaskRoutes:
    def test_tasks_add_happy_path(self, client, patched_tools):
        response = client.post(
            "/api/tasks/add",
            json={"title": "Fix bug", "priority": "high"},
        )
        assert response.status_code == 200

    def test_tasks_add_missing_title_returns_422(self, client, patched_tools):
        response = client.post("/api/tasks/add", json={"priority": "high"})
        assert response.status_code == 422

    def test_tasks_update_happy_path(self, client, patched_tools):
        response = client.put(
            "/api/tasks/update",
            json={"task_id": "task_1", "status": "done"},
        )
        assert response.status_code == 200

    def test_tasks_update_missing_task_id_returns_422(self, client, patched_tools):
        response = client.put("/api/tasks/update", json={"status": "done"})
        assert response.status_code == 422

    def test_tasks_remove_valid_id(self, client, patched_tools):
        response = client.delete("/api/tasks/task_42")
        assert response.status_code == 200


# ---------------------------------------------------------------------------
# API TESTER
# ---------------------------------------------------------------------------

class TestApiTesterRoutes:
    def test_api_tester_run_public_url_happy_path(self, client, patched_tools):
        """A public HTTPS URL should pass the SSRF guard and invoke the tool."""
        response = client.post(
            "/api/api-tester/run",
            json={"method": "GET", "url": "https://example.com"},
        )
        # The SSRF validator may 422 if it can't resolve example.com in a restricted env.
        assert response.status_code in (200, 422)
        if response.status_code == 200:
            assert response.json().get("success") is True

    def test_api_tester_run_rejects_localhost(self, client, patched_tools):
        response = client.post(
            "/api/api-tester/run",
            json={"method": "GET", "url": "http://localhost:8000"},
        )
        assert response.status_code == 422

    def test_api_tester_run_rejects_bad_scheme(self, client, patched_tools):
        response = client.post(
            "/api/api-tester/run",
            json={"method": "GET", "url": "file:///etc/passwd"},
        )
        assert response.status_code == 422


# ---------------------------------------------------------------------------
# DATABASE
# ---------------------------------------------------------------------------

class TestDatabaseRoutes:
    def test_database_query_select_allowed(self, client, patched_tools):
        response = client.post(
            "/api/database/query",
            json={"sql": "SELECT 1", "db": "default"},
        )
        assert response.status_code == 200

    def test_database_query_drop_blocked(self, client, patched_tools):
        response = client.post(
            "/api/database/query",
            json={"sql": "DROP TABLE users", "db": "default"},
        )
        assert response.status_code == 200
        assert response.json().get("success") is False

    def test_database_query_multi_statement_blocked(self, client, patched_tools):
        response = client.post(
            "/api/database/query",
            json={"sql": "SELECT 1; DROP TABLE users", "db": "default"},
        )
        assert response.status_code == 200
        assert response.json().get("success") is False

    def test_database_import_csv_outside_data_dir_rejected(self, client, patched_tools, tmp_path):
        bogus = tmp_path / "outside.csv"
        bogus.write_text("a,b\n1,2\n")
        response = client.post(
            "/api/database/import-csv",
            json={"csv_path": str(bogus), "table": "t1", "db": "default"},
        )
        # Either the route enforces the sandbox (200 with success=False) or
        # returns 422 from validator — both prove the path check works.
        assert response.status_code in (200, 422)
        if response.status_code == 200:
            assert response.json().get("success") is False


# ---------------------------------------------------------------------------
# AUDIO
# ---------------------------------------------------------------------------

class TestAudioRoutes:
    def test_audio_transcribe_outside_data_dir_rejected(self, client, patched_tools, tmp_path):
        bogus = tmp_path / "clip.wav"
        bogus.write_bytes(b"RIFF")
        response = client.post(
            "/api/audio/transcribe",
            json={"file_path": str(bogus)},
        )
        assert response.status_code == 200
        assert response.json().get("success") is False


# ---------------------------------------------------------------------------
# CLIPBOARD
# ---------------------------------------------------------------------------

class TestClipboardRoutes:
    def test_clipboard_capture_happy_path(self, client, patched_tools):
        response = client.post("/api/clipboard/capture")
        assert response.status_code == 200


# ---------------------------------------------------------------------------
# RESEARCH
# ---------------------------------------------------------------------------

class TestResearchRoutes:
    def test_research_save_happy_path(self, client, patched_tools):
        response = client.post(
            "/api/research/save",
            json={"title": "T", "content": "C", "category": "tools"},
        )
        assert response.status_code == 200

    def test_research_save_missing_content_returns_422(self, client, patched_tools):
        response = client.post(
            "/api/research/save",
            json={"title": "T", "category": "tools"},
        )
        assert response.status_code == 422
