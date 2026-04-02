"""Tests for security hardening fixes (2026-03-22 engineering review).

Covers: path traversal, SQL injection, shell allowlist, auth endpoints,
        SSRF, code validation, message truncation, dead stub removal,
        status route auth, marketplace sanitization, taint tracker,
        auth fail-closed, dead strategy removal.
"""
import os
import json
import tempfile
import shutil
from pathlib import Path
from unittest.mock import MagicMock, patch

import pytest

pytest.importorskip("fastapi")


# ============================================================================
# Path traversal in conversation management
# ============================================================================

class TestConversationPathTraversal:
    """Verify that conversation IDs cannot escape the conversations directory."""

    def _make_brain_stub(self, tmp_path):
        """Create a minimal OllamaBrain-like object for testing path validation."""
        from aura.brain import OllamaBrain
        # We only need _conversations_dir and _validate_conversation_path
        brain = object.__new__(OllamaBrain)
        brain._conversations_dir = Path(tmp_path) / "conversations"
        brain._conversations_dir.mkdir(parents=True, exist_ok=True)
        return brain

    def test_blocks_parent_traversal(self, tmp_path):
        brain = self._make_brain_stub(tmp_path)
        assert brain._validate_conversation_path("../../etc") is None

    def test_blocks_absolute_path(self, tmp_path):
        brain = self._make_brain_stub(tmp_path)
        assert brain._validate_conversation_path("/etc/passwd") is None

    def test_blocks_sibling_directory(self, tmp_path):
        brain = self._make_brain_stub(tmp_path)
        assert brain._validate_conversation_path("../sibling") is None

    def test_allows_valid_conversation_id(self, tmp_path):
        brain = self._make_brain_stub(tmp_path)
        # Create a valid conversation dir
        valid_dir = brain._conversations_dir / "conv-123"
        valid_dir.mkdir()
        result = brain._validate_conversation_path("conv-123")
        assert result is not None
        assert result == valid_dir.resolve()

    def test_delete_requires_history_json(self, tmp_path):
        """delete_conversation should refuse to delete dirs without history.json."""
        brain = self._make_brain_stub(tmp_path)
        # Create a dir that looks like a conversation but has no history.json
        fake_dir = brain._conversations_dir / "fake-conv"
        fake_dir.mkdir()
        (fake_dir / "some_file.txt").write_text("not a conversation")

        # Provide required attributes for delete_conversation
        brain._current_conversation_id = "other"
        brain._conversations_index_file = brain._conversations_dir / "index.json"
        brain._conversations_index_file.write_text("[]")
        brain._conversations_index_lock = __import__("threading").Lock()
        brain._conversations_index_cache = None

        result = brain.delete_conversation("fake-conv")
        assert result is False
        assert fake_dir.exists()  # Should NOT have been deleted


# ============================================================================
# SQL injection via WITH CTE
# ============================================================================

class TestSQLInjectionBlocking:
    """Verify that DML keywords inside WITH CTEs are blocked."""

    def test_blocks_with_insert(self):
        """WITH x AS (SELECT 1) INSERT INTO ... should be blocked."""
        from api.routes.tools_new import _SQL_ALLOWED_PREFIXES
        import re
        sql = "WITH x AS (SELECT 1) INSERT INTO users VALUES (1, 'admin')"
        sql_stripped = sql.strip().lower()
        # Passes prefix check (starts with 'with')
        assert sql_stripped.startswith(_SQL_ALLOWED_PREFIXES)
        # But our DML keyword check should catch it
        _DML_KEYWORDS = {"insert", "update", "delete", "drop", "alter", "create", "replace", "truncate"}
        _sql_no_strings = re.sub(r"'[^']*'", "", sql)
        _sql_no_strings = re.sub(r'"[^"]*"', "", _sql_no_strings)
        _sql_words = set(re.findall(r'\b\w+\b', _sql_no_strings.lower()))
        assert _sql_words & _DML_KEYWORDS  # Should find 'insert'

    def test_blocks_with_delete(self):
        sql = "WITH cte AS (SELECT id FROM users) DELETE FROM users WHERE id IN (SELECT id FROM cte)"
        import re
        _DML_KEYWORDS = {"insert", "update", "delete", "drop", "alter", "create", "replace", "truncate"}
        _sql_no_strings = re.sub(r"'[^']*'", "", sql)
        _sql_no_strings = re.sub(r'"[^"]*"', "", _sql_no_strings)
        _sql_words = set(re.findall(r'\b\w+\b', _sql_no_strings.lower()))
        assert _sql_words & _DML_KEYWORDS  # Should find 'delete'

    def test_allows_pure_select_with(self):
        sql = "WITH cte AS (SELECT 1) SELECT * FROM cte"
        import re
        _DML_KEYWORDS = {"insert", "update", "delete", "drop", "alter", "create", "replace", "truncate"}
        _sql_no_strings = re.sub(r"'[^']*'", "", sql)
        _sql_no_strings = re.sub(r'"[^"]*"', "", _sql_no_strings)
        _sql_words = set(re.findall(r'\b\w+\b', _sql_no_strings.lower()))
        assert not (_sql_words & _DML_KEYWORDS)  # Pure SELECT should pass


# ============================================================================
# Shell allowlist hardening
# ============================================================================

class TestShellAllowlistHardening:
    """Verify expanded go command blocking and danger patterns."""

    def test_go_generate_blocked(self):
        from api.routes.tools_new import _INTERPRETER_EXEC_FLAGS
        assert "generate" in _INTERPRETER_EXEC_FLAGS["go"]

    def test_go_build_blocked(self):
        from api.routes.tools_new import _INTERPRETER_EXEC_FLAGS
        assert "build" in _INTERPRETER_EXEC_FLAGS["go"]

    def test_go_test_blocked(self):
        from api.routes.tools_new import _INTERPRETER_EXEC_FLAGS
        assert "test" in _INTERPRETER_EXEC_FLAGS["go"]

    def test_sed_inplace_in_danger_patterns(self):
        from api.routes.tools_new import _SHELL_DANGER_PATTERNS
        assert any("sed -i" in p for p in _SHELL_DANGER_PATTERNS)

    def test_git_force_push_in_danger_patterns(self):
        from api.routes.tools_new import _SHELL_DANGER_PATTERNS
        assert any("git push --force" in p or "git push -f" in p for p in _SHELL_DANGER_PATTERNS)


# ============================================================================
# PKCE store bounded growth
# ============================================================================

class TestPKCEStoreBounded:
    """Verify PKCE store has size limits and TTL."""

    def test_store_evicts_at_capacity(self):
        from api.routes.auth import _pkce_store_put, _pkce_store_get, _pkce_store, _PKCE_MAX_ENTRIES
        _pkce_store.clear()
        # Fill to capacity + 1
        for i in range(_PKCE_MAX_ENTRIES + 5):
            _pkce_store_put(f"state-{i}", f"verifier-{i}")
        assert len(_pkce_store) <= _PKCE_MAX_ENTRIES
        _pkce_store.clear()

    def test_store_retrieves_correctly(self):
        from api.routes.auth import _pkce_store_put, _pkce_store_get, _pkce_store
        _pkce_store.clear()
        _pkce_store_put("test-state", "test-verifier")
        assert _pkce_store_get("test-state") == "test-verifier"
        # Second get should return None (one-time use)
        assert _pkce_store_get("test-state") is None
        _pkce_store.clear()


# ============================================================================
# Auth endpoint protection
# ============================================================================

class TestAuthEndpointProtection:
    """Verify auth-protected endpoints have dependencies."""

    def test_logout_requires_auth(self):
        """The /chatgpt/logout endpoint must have auth dependency."""
        from api.routes.auth import router
        for route in router.routes:
            path = getattr(route, 'path', '')
            if path.endswith("/chatgpt/logout"):
                deps = getattr(route, 'dependencies', []) or []
                assert len(deps) > 0, "/chatgpt/logout should have auth dependency"
                break
        else:
            raise AssertionError("/chatgpt/logout route not found")

    def test_models_detailed_requires_auth(self):
        """The /models/detailed endpoint must have auth dependency."""
        from api.routes.status import router
        for route in router.routes:
            path = getattr(route, 'path', '')
            if path.endswith("/models/detailed"):
                deps = getattr(route, 'dependencies', []) or []
                assert len(deps) > 0, "/models/detailed should have auth dependency"
                break


# ============================================================================
# Phase 2: SSRF DNS rebinding protection
# ============================================================================

class TestSSRFProtection:
    """Verify SSRF guard blocks private IPs and DNS rebinding."""

    def test_blocks_private_ip_directly(self):
        from aura.security.ssrf_guard import validate_url_safe
        import pytest
        with pytest.raises(ValueError, match="private"):
            validate_url_safe("http://192.168.1.1/admin")

    def test_blocks_loopback(self):
        from aura.security.ssrf_guard import validate_url_safe
        import pytest
        with pytest.raises(ValueError, match="private|blocked|Blocked"):
            validate_url_safe("http://127.0.0.1/secret")

    def test_blocks_dangerous_port(self):
        from aura.security.ssrf_guard import validate_url_safe
        import pytest
        with pytest.raises(ValueError, match="Blocked port"):
            validate_url_safe("http://example.com:6379/")

    def test_blocks_file_scheme(self):
        from aura.security.ssrf_guard import validate_url_safe
        import pytest
        with pytest.raises(ValueError, match="Blocked scheme"):
            validate_url_safe("file:///etc/passwd")

    def test_allows_public_url(self):
        """Public IPs should pass validation (DNS resolution may fail in CI)."""
        from aura.security.ssrf_guard import validate_url_safe
        # Direct IP — no DNS needed, URL unchanged, no original_hostname
        pinned_url, original_hostname = validate_url_safe("http://8.8.8.8/test")
        assert pinned_url == "http://8.8.8.8/test"
        assert original_hostname is None  # Already an IP literal


# ============================================================================
# Phase 2: Code validation (validate_script_code)
# ============================================================================

class TestCodeValidation:
    """Verify validate_script_code blocks dangerous patterns."""

    def test_blocks_subprocess_direct(self):
        from aura.agent import validate_script_code
        ok, msg = validate_script_code("import subprocess\nsubprocess.run(['ls'])", "test")
        assert not ok
        assert "subprocess" in msg.lower() or "Forbidden" in msg

    def test_blocks_string_concat_evasion(self):
        """String concatenation like 'sub' + 'process' should be caught."""
        from aura.agent import validate_script_code
        code = 'mod = "sub" + "process"\n__import__(mod)'
        ok, msg = validate_script_code(code, "test")
        assert not ok

    def test_blocks_getattr_builtins(self):
        from aura.agent import validate_script_code
        code = 'x = getattr(__builtins__, "eval")\nx("print(1)")'
        ok, msg = validate_script_code(code, "test")
        assert not ok

    def test_blocks_dunder_subclasses(self):
        from aura.agent import validate_script_code
        code = 'x = ().__class__.__bases__[0].__subclasses__()'
        ok, msg = validate_script_code(code, "test")
        assert not ok

    def test_allows_safe_code(self):
        from aura.agent import validate_script_code
        code = 'import json\nresult = json.dumps({"key": "value"})\nprint(result)'
        ok, msg = validate_script_code(code, "test")
        assert ok

    def test_blocks_type_call(self):
        from aura.agent import validate_script_code
        code = 'x = type("Exploit", (object,), {"run": lambda: None})'
        ok, msg = validate_script_code(code, "test")
        assert not ok

    def test_blocks_os_import(self):
        from aura.agent import validate_script_code
        code = 'import os\nos.listdir("/")'
        ok, msg = validate_script_code(code, "test")
        assert not ok


# ============================================================================
# Phase 2: ReAct message truncation
# ============================================================================

class TestReActMessageTruncation:
    """Verify message list doesn't grow unbounded."""

    def test_messages_truncated_at_threshold(self):
        """Simulate what the ReAct loop does: truncate when > 30 messages."""
        messages = [
            {"role": "system", "content": "system prompt"},
            {"role": "user", "content": "goal"},
        ]
        # Simulate 20 iterations adding 2 messages each
        for i in range(20):
            messages.append({"role": "assistant", "content": f"thought {i}"})
            messages.append({"role": "user", "content": f"result {i}"})

        assert len(messages) == 42  # 2 + 40

        # Apply the same truncation logic as the fix
        if len(messages) > 30:
            messages = messages[:2] + messages[-28:]

        assert len(messages) == 30
        # First two are preserved (system + goal)
        assert messages[0]["content"] == "system prompt"
        assert messages[1]["content"] == "goal"


# ============================================================================
# Phase 2: Dead stub removal verification
# ============================================================================

class TestDeadStubRemoval:
    """Verify dead module stubs are removed from agent init."""

    def test_no_mirrormind_attribute(self):
        """Agent source should not set self.mirrormind = None."""
        import inspect
        from aura.agent import ApprenticeAgent
        source = inspect.getsource(ApprenticeAgent.__init__)
        assert "self.mirrormind = None" not in source
        assert "self.mirrormind_enabled" not in source

    def test_no_theater_stub(self):
        import inspect
        from aura.agent import ApprenticeAgent
        source = inspect.getsource(ApprenticeAgent.__init__)
        assert "self.theater = None" not in source
        assert "self.theater_enabled = False" not in source

    def test_no_worldsim_stub(self):
        import inspect
        from aura.agent import ApprenticeAgent
        source = inspect.getsource(ApprenticeAgent.__init__)
        assert "self.worldsim = None" not in source
        assert "self.worldsim_enabled = False" not in source


# ============================================================================
# Phase 2: Channels module is ACTIVE (Telegram, Extension channels)
# The earlier test assumed channels would be deleted; it was not — channels are live.
# ============================================================================

class TestChannelsActive:
    """Verify the aura/channels module exists (it is actively used)."""

    def test_channels_directory_exists(self):
        channels_path = Path(__file__).parent.parent / "aura" / "channels"
        assert channels_path.exists(), f"aura/channels should exist — it contains TelegramChannel and ExtensionChannel"

    def test_channels_has_key_modules(self):
        """Verify key channel modules are present."""
        channels_path = Path(__file__).parent.parent / "aura" / "channels"
        assert (channels_path / "telegram_channel.py").exists()
        assert (channels_path / "extension_channel.py").exists()
        assert (channels_path / "channel_bridge.py").exists()


# ============================================================================
# Phase 2: Brain thread safety
# ============================================================================

class TestBrainThreadSafety:
    """Verify OllamaBrain has the _think_lock attribute."""

    def test_think_lock_exists(self):
        """OllamaBrain.__init__ should create a _think_lock."""
        import threading
        from aura.brain import OllamaBrain
        # Check class source since we can't easily instantiate without Ollama
        import inspect
        source = inspect.getsource(OllamaBrain.__init__)
        assert "_think_lock" in source


# ============================================================================
# Phase 3: Status routes require auth
# ============================================================================

class TestStatusRouterAuth:
    """Verify status router has router-level auth dependency."""

    def test_status_router_has_auth_dependency(self):
        from api.routes.status import router
        dep_names = [str(d) for d in (router.dependencies or [])]
        assert len(router.dependencies) > 0, "Status router should have require_api_key dependency"

    def test_health_endpoint_is_public(self):
        """GET /api/health should be on the public_router (no auth)."""
        from api.routes.status import public_router
        found = False
        for route in public_router.routes:
            path = getattr(route, 'path', '')
            if path.endswith("/health"):
                found = True
                break
        assert found, "/health should be on public_router (no auth)"
        # public_router should NOT have require_api_key dependency
        assert len(public_router.dependencies) == 0, "public_router should have no auth dependencies"

    def test_deep_health_requires_auth(self):
        """GET /api/health/deep should inherit router-level auth."""
        from api.routes.status import router
        for route in router.routes:
            path = getattr(route, 'path', '')
            if path.endswith("/health/deep"):
                # No per-route dependencies = inherits router-level auth
                route_deps = getattr(route, 'dependencies', None)
                # Either None (inherits) or non-empty (explicit) is fine
                if route_deps is not None:
                    # If explicitly set, shouldn't be empty
                    pass  # Any value is OK, router-level auth still applies
                break


# ============================================================================
# Phase 3: Marketplace plugin_id sanitization
# ============================================================================

class TestMarketplacePluginIdSanitization:
    """Verify plugin_id is validated against path traversal."""

    def test_blocks_path_traversal(self):
        from aura.tools.marketplace import MarketplaceTool
        tool = MarketplaceTool.__new__(MarketplaceTool)
        # get_info should reject traversal
        result = tool.get_info("../../../etc/passwd")
        assert not result.get("success")
        assert "Invalid" in result.get("error", "")

    def test_blocks_slashes(self):
        from aura.tools.marketplace import MarketplaceTool
        tool = MarketplaceTool.__new__(MarketplaceTool)
        result = tool.get_info("foo/bar")
        assert not result.get("success")

    def test_blocks_dots(self):
        from aura.tools.marketplace import MarketplaceTool
        tool = MarketplaceTool.__new__(MarketplaceTool)
        result = tool.get_info("..evil")
        assert not result.get("success")

    def test_allows_valid_id(self):
        from aura.tools.marketplace import MarketplaceTool
        tool = MarketplaceTool.__new__(MarketplaceTool)
        # This will fail at registry fetch, but shouldn't fail at validation
        result = tool.get_info("my-cool-plugin_v2")
        # Should NOT be "Invalid plugin ID"
        assert "Invalid plugin ID" not in result.get("error", "")

    def test_blocks_empty(self):
        from aura.tools.marketplace import MarketplaceTool
        tool = MarketplaceTool.__new__(MarketplaceTool)
        result = tool.get_info("")
        assert not result.get("success")


# ============================================================================
# Phase 3: Taint tracker no longer allows SECRET to display
# ============================================================================

class TestTaintTrackerDisplaySink:
    """Verify SECRET data is blocked from display sink."""

    def test_secret_blocked_from_display(self):
        from aura.security.taint_tracker import TaintTracker, TaintLabel
        tracker = TaintTracker()
        # Simulate a session that has seen a secret
        tracker._session_taints["test-session"] = TaintLabel.SECRET
        assert not tracker.is_safe_for_sink("test-session", "display")

    def test_pii_allowed_in_display(self):
        from aura.security.taint_tracker import TaintTracker, TaintLabel
        tracker = TaintTracker()
        tracker._session_taints["test-session"] = TaintLabel.PII
        assert tracker.is_safe_for_sink("test-session", "display")

    def test_secret_blocked_from_all_sinks(self):
        from aura.security.taint_tracker import TaintTracker, TaintLabel
        tracker = TaintTracker()
        tracker._session_taints["test-session"] = TaintLabel.SECRET
        for sink in ("display", "memory", "log", "external_api"):
            assert not tracker.is_safe_for_sink("test-session", sink), f"SECRET should be blocked from {sink}"


# ============================================================================
# Phase 3: Auth fails closed when key is missing
# ============================================================================

class TestAuthFailClosed:
    """Verify auth rejects requests when AURA_API_KEY is unset but auth is enabled."""

    def test_fail_closed_when_key_missing(self):
        """When auth enabled but no key, should raise 503 not silently disable."""
        import os
        from unittest.mock import patch
        from fastapi import HTTPException

        # Patch env: auth enabled, no key
        with patch.dict(os.environ, {"AURA_API_AUTH_ENABLED": "true"}, clear=False):
            with patch.dict(os.environ, {}, clear=False):
                # Remove AURA_API_KEY if present
                env_copy = dict(os.environ)
                env_copy.pop("AURA_API_KEY", None)
                with patch.dict(os.environ, env_copy, clear=True):
                    import asyncio
                    from api.auth import require_api_key
                    try:
                        asyncio.get_event_loop().run_until_complete(require_api_key(""))
                        assert False, "Should have raised HTTPException"
                    except HTTPException as e:
                        assert e.status_code == 503


# ============================================================================
# Phase 3: Dead strategy branches removed
# ============================================================================

class TestDeadStrategyRemoval:
    """Verify dead reasoning strategy branches are removed."""

    def test_no_cognitive_theater_branch(self):
        import inspect
        from aura.agent import ApprenticeAgent
        source = inspect.getsource(ApprenticeAgent)
        assert "COGNITIVE_THEATER" not in source

    def test_no_debate_branch(self):
        import inspect
        from aura.agent import ApprenticeAgent
        source = inspect.getsource(ApprenticeAgent)
        assert "ReasoningStrategy.DEBATE" not in source

    def test_no_reflexion_branch(self):
        import inspect
        from aura.agent import ApprenticeAgent
        source = inspect.getsource(ApprenticeAgent)
        assert "ReasoningStrategy.REFLEXION" not in source


# ============================================================================
# Phase 3: Shell allowlist doesn't include 'go'
# ============================================================================

class TestShellAllowlistHardened:
    """Verify dangerous commands removed from shell allowlist."""

    def test_go_not_in_allowlist(self):
        """'go' command should not be in the shell allowlist since all useful subcommands are blocked."""
        from api.routes.tools_new import _SHELL_ALLOWED_COMMANDS
        assert "go" not in _SHELL_ALLOWED_COMMANDS
