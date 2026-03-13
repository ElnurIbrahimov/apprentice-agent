"""Tests for engineering fixes applied 2026-03-08."""
import pytest
import os
import sys
import uuid

# Ensure project root is importable
sys.path.insert(0, "D:/Aura")


# ── Test 1: write_gate merge target picks best match (not first) ─────────

class TestWriteGateMergeTarget:
    def test_merge_picks_highest_similarity(self):
        from aura.memory.write_gate import MemoryWriteGate, MemoryCandidate

        gate = MemoryWriteGate()
        candidate = MemoryCandidate(
            content="The user prefers dark mode for all editors",
            source="conversation",
            importance=0.7,
        )

        # Three nearby memories: first=0.89, second=0.95 (best), third=0.91
        nearby = [
            {"content": "User likes dark themes",   "source_id": "mem_first",  "score": 0.89, "source": "amem"},
            {"content": "User prefers dark mode",    "source_id": "mem_best",   "score": 0.95, "source": "amem"},
            {"content": "Dark UI preference noted",  "source_id": "mem_third",  "score": 0.91, "source": "amem"},
        ]

        target = gate._find_merge_target(candidate, nearby)
        assert target is not None
        assert target["source_id"] == "mem_best", (
            f"Expected best match (0.95), got {target['source_id']} with score {target['score']}"
        )

    def test_merge_returns_none_below_threshold(self):
        from aura.memory.write_gate import MemoryWriteGate, MemoryCandidate

        gate = MemoryWriteGate()
        candidate = MemoryCandidate(content="Something new", source="conversation")
        nearby = [
            {"content": "Unrelated", "source_id": "low", "score": 0.50, "source": "amem"},
        ]
        assert gate._find_merge_target(candidate, nearby) is None


# ── Test 2: safe_error_detail returns generic in production ──────────────

class TestSafeErrorDetail:
    def test_production_hides_detail(self):
        from api.utils import safe_error_detail

        old = os.environ.get("AURA_ENV")
        try:
            os.environ["AURA_ENV"] = "production"
            result = safe_error_detail(ValueError("secret db info"))
            assert result == "Internal server error"
        finally:
            if old is None:
                os.environ.pop("AURA_ENV", None)
            else:
                os.environ["AURA_ENV"] = old

    def test_dev_shows_detail(self):
        from api.utils import safe_error_detail

        old = os.environ.get("AURA_ENV")
        try:
            os.environ.pop("AURA_ENV", None)
            result = safe_error_detail(ValueError("secret db info"))
            assert result == "secret db info"
        finally:
            if old is None:
                os.environ.pop("AURA_ENV", None)
            else:
                os.environ["AURA_ENV"] = old


# ── Test 3: ToolResult dataclass ─────────────────────────────────────────

class TestToolResult:
    def test_success_result(self):
        from aura.tools.tool_contract import ToolResult

        tr = ToolResult(success=True, result=42)
        d = tr.to_dict()
        assert d["success"] is True
        assert d["result"] == 42
        # Empty error string is falsy → key omitted by implementation
        assert "error" not in d

    def test_failure_result(self):
        from aura.tools.tool_contract import ToolResult

        tr = ToolResult(success=False, error="boom")
        d = tr.to_dict()
        assert d["success"] is False
        # result is None → key omitted by implementation
        assert "result" not in d
        assert d["error"] == "boom"

    def test_both_result_and_error(self):
        from aura.tools.tool_contract import ToolResult

        tr = ToolResult(success=False, result="partial", error="warning")
        d = tr.to_dict()
        assert d["success"] is False
        assert d["result"] == "partial"
        assert d["error"] == "warning"


# ── Test 4: Config centralized thresholds exist ──────────────────────────

class TestConfigThresholds:
    def test_salience_filter_threshold(self):
        from aura.config import Config
        assert Config.SALIENCE_FILTER_THRESHOLD == 0.3

    def test_history_limit(self):
        from aura.config import Config
        assert Config.HISTORY_LIMIT == 20

    def test_budget_small(self):
        from aura.config import Config
        assert Config.BUDGET_SMALL == 150

    def test_budget_medium(self):
        from aura.config import Config
        assert Config.BUDGET_MEDIUM == 400

    def test_budget_large(self):
        from aura.config import Config
        assert Config.BUDGET_LARGE == 800


# ── Test 5: RequestIDMiddleware adds X-Request-ID header ─────────────────

class TestRequestIDMiddleware:
    def test_response_has_valid_uuid(self):
        from fastapi import FastAPI
        from fastapi.testclient import TestClient
        from api.middleware import RequestIDMiddleware

        app = FastAPI()
        app.add_middleware(RequestIDMiddleware)

        @app.get("/ping")
        def ping():
            return {"ok": True}

        client = TestClient(app)
        resp = client.get("/ping")

        assert resp.status_code == 200
        rid = resp.headers.get("X-Request-ID")
        assert rid is not None, "X-Request-ID header missing"
        # Must be a valid UUID4
        parsed = uuid.UUID(rid, version=4)
        assert str(parsed) == rid

    def test_client_provided_id_is_reused(self):
        from fastapi import FastAPI
        from fastapi.testclient import TestClient
        from api.middleware import RequestIDMiddleware

        app = FastAPI()
        app.add_middleware(RequestIDMiddleware)

        @app.get("/ping")
        def ping():
            return {"ok": True}

        client = TestClient(app)
        custom_id = "my-custom-request-id-12345"
        resp = client.get("/ping", headers={"X-Request-ID": custom_id})

        assert resp.headers.get("X-Request-ID") == custom_id
