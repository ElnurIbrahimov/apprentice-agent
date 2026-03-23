"""Tests for aura.security — SSRF guard, taint tracker, audit chain, tool signing."""
import hashlib
import json
import os
import tempfile
import time

import pytest


# ── SSRF Guard ──────────────────────────────────────────────────────────────

class TestSSRFGuard:
    def test_private_ips_blocked(self):
        from aura.security.ssrf_guard import is_private_ip
        for ip in ("127.0.0.1", "10.0.0.1", "172.16.0.1", "192.168.1.1",
                    "169.254.1.1", "0.0.0.0", "::1", "fc00::1", "fe80::1"):
            assert is_private_ip(ip), f"{ip} should be blocked"

    def test_public_ips_allowed(self):
        from aura.security.ssrf_guard import is_private_ip
        for ip in ("8.8.8.8", "1.1.1.1", "104.18.26.120", "2606:4700::"):
            assert not is_private_ip(ip), f"{ip} should be allowed"

    def test_invalid_ip_blocked(self):
        from aura.security.ssrf_guard import is_private_ip
        assert is_private_ip("not-an-ip")

    def test_blocked_scheme(self):
        from aura.security.ssrf_guard import validate_url_safe
        with pytest.raises(ValueError, match="Blocked scheme"):
            validate_url_safe("ftp://evil.com")

    def test_blocked_private_ip_in_url(self):
        from aura.security.ssrf_guard import validate_url_safe
        with pytest.raises(ValueError, match="private"):
            validate_url_safe("http://127.0.0.1/admin")

    def test_blocked_port(self):
        from aura.security.ssrf_guard import validate_url_safe
        with pytest.raises(ValueError, match="Blocked port"):
            validate_url_safe("http://example.com:6379/")

    def test_empty_url(self):
        from aura.security.ssrf_guard import validate_url_safe
        with pytest.raises(ValueError):
            validate_url_safe("")

    def test_too_long_url(self):
        from aura.security.ssrf_guard import validate_url_safe
        with pytest.raises(ValueError, match="max length"):
            validate_url_safe("http://example.com/" + "a" * 5000)

    def test_no_hostname(self):
        from aura.security.ssrf_guard import validate_url_safe
        with pytest.raises(ValueError, match="no hostname"):
            validate_url_safe("http:///path")

    def test_valid_url_returns_pinned_url(self):
        from aura.security.ssrf_guard import validate_url_safe
        pinned_url, original_hostname = validate_url_safe("https://example.com/api")
        # URL should now contain the resolved IP, not the original hostname
        assert "example.com" not in pinned_url
        assert original_hostname == "example.com"
        # The pinned URL should still be valid https
        assert pinned_url.startswith("https://")

    def test_ipv6_brackets_in_pin(self):
        from aura.security.ssrf_guard import _pin_url
        from urllib.parse import urlparse
        parsed = urlparse("https://example.com:443/path")
        result = _pin_url(parsed, "2001:db8::1")
        assert "[2001:db8::1]" in result

    def test_ipv4_no_brackets(self):
        from aura.security.ssrf_guard import _pin_url
        from urllib.parse import urlparse
        parsed = urlparse("https://example.com/path")
        result = _pin_url(parsed, "1.2.3.4")
        assert "[" not in result
        assert "1.2.3.4" in result

    def test_dns_timeout(self):
        from aura.security.ssrf_guard import _resolve_hostname
        # Valid host should resolve within timeout
        ips = _resolve_hostname("example.com", timeout=5.0)
        assert len(ips) > 0

    def test_dns_failure(self):
        from aura.security.ssrf_guard import _resolve_hostname
        with pytest.raises(ValueError, match="DNS resolution failed"):
            _resolve_hostname("this-domain-definitely-does-not-exist-xyz123.invalid")


# ── Taint Tracker ───────────────────────────────────────────────────────────

class TestTaintTracker:
    def test_detect_openai_key(self):
        from aura.security.taint_tracker import scan_for_secrets
        matches = scan_for_secrets("key: sk-abcdefghijklmnopqrstuvwxyz1234")
        assert len(matches) == 1
        assert matches[0].description == "OpenAI API key"

    def test_detect_anthropic_key(self):
        from aura.security.taint_tracker import scan_for_secrets
        matches = scan_for_secrets("sk-ant-api3X9kL2mN5oP7qR1sT3uV5wX9z")
        assert len(matches) == 1
        assert matches[0].description == "Anthropic API key"

    def test_detect_aws_key(self):
        from aura.security.taint_tracker import scan_for_secrets
        matches = scan_for_secrets("AKIAIOSFODNN7EXAMPLE")
        assert len(matches) == 1
        assert matches[0].description == "AWS access key"

    def test_detect_github_pat(self):
        from aura.security.taint_tracker import scan_for_secrets
        matches = scan_for_secrets("ghp_abcdefghijklmnopqrstuvwxyz1234567890")
        assert len(matches) == 1
        assert matches[0].description == "GitHub PAT"

    def test_detect_password(self):
        from aura.security.taint_tracker import scan_for_secrets
        matches = scan_for_secrets("password=hunter2abc!")
        assert len(matches) >= 1
        assert any(m.description == "password" for m in matches)

    def test_password_no_false_positive_on_prose(self):
        from aura.security.taint_tracker import scan_for_secrets
        matches = scan_for_secrets("password=this is a long sentence with spaces and stuff")
        password_matches = [m for m in matches if m.description == "password"]
        assert len(password_matches) == 0

    def test_detect_email(self):
        from aura.security.taint_tracker import scan_for_secrets
        matches = scan_for_secrets("contact john.doe@example.com please", check_pii=True)
        assert any(m.description == "email address" for m in matches)

    def test_detect_ssn(self):
        from aura.security.taint_tracker import scan_for_secrets
        matches = scan_for_secrets("SSN: 123-45-6789")
        assert any(m.description == "SSN" for m in matches)

    def test_no_false_positives_on_clean_text(self):
        from aura.security.taint_tracker import scan_for_secrets
        matches = scan_for_secrets("Hello, how are you doing today? The weather is nice.")
        assert len(matches) == 0

    def test_redact_secrets(self):
        from aura.security.taint_tracker import redact
        text = "Use key AKIAIOSFODNN7EXAMPLE for AWS"
        result = redact(text)
        assert "AKIAIOSFODNN7EXAMPLE" not in result
        assert "[REDACTED:" in result

    def test_redact_preserves_clean_text(self):
        from aura.security.taint_tracker import redact
        text = "This is perfectly clean text"
        assert redact(text) == text

    def test_overlap_dedup(self):
        from aura.security.taint_tracker import scan_for_secrets
        # sk-ant matches both OpenAI and Anthropic patterns — should dedup
        matches = scan_for_secrets("sk-ant-abcdefghijklmnop1234567890")
        assert len(matches) == 1

    def test_highest_taint(self):
        from aura.security.taint_tracker import TaintLabel, TaintMatch, highest_taint
        matches = [
            TaintMatch(TaintLabel.PII, "email", "***", 0, 10),
            TaintMatch(TaintLabel.SECRET, "key", "***", 20, 40),
        ]
        assert highest_taint(matches) == TaintLabel.SECRET

    def test_tracker_session_elevation(self):
        from aura.security.taint_tracker import TaintTracker, TaintLabel
        t = TaintTracker()
        t.check_and_track("clean text", session_id="s1")
        assert t.is_safe_for_sink("s1", "external_api")
        t.check_and_track("key: sk-abcdefghijklmnopqrstuvwxyz1234", session_id="s1")
        assert not t.is_safe_for_sink("s1", "external_api")
        assert not t.is_safe_for_sink("s1", "memory")

    def test_tracker_clear_session(self):
        from aura.security.taint_tracker import TaintTracker
        t = TaintTracker()
        t.check_and_track("sk-abcdefghijklmnopqrstuvwxyz1234", session_id="s2")
        assert not t.is_safe_for_sink("s2", "external_api")
        t.clear_session("s2")
        assert t.is_safe_for_sink("s2", "external_api")  # back to default (PUBLIC)

    def test_private_key_detection(self):
        from aura.security.taint_tracker import scan_for_secrets
        matches = scan_for_secrets("-----BEGIN RSA PRIVATE KEY-----\nMIIE...")
        assert any(m.description == "private key" for m in matches)

    def test_connection_string_detection(self):
        from aura.security.taint_tracker import scan_for_secrets
        matches = scan_for_secrets("postgres://user:pass@host:5432/db")
        assert any(m.description == "database connection string" for m in matches)


# ── Audit Chain ─────────────────────────────────────────────────────────────

class TestAuditChain:
    @pytest.fixture
    def chain(self, tmp_path):
        from aura.security.audit_chain import AuditChain
        return AuditChain(db_path=str(tmp_path / "test_audit.db"))

    def test_append_and_count(self, chain):
        assert chain.count() == 0
        chain.append("tool_call", {"tool": "search"})
        chain.append("tool_call", {"tool": "read"})
        assert chain.count() == 2

    def test_chain_integrity(self, chain):
        for i in range(10):
            chain.append("test", {"i": i})
        valid, count, err = chain.verify()
        assert valid
        assert count == 10
        assert err is None

    def test_tail(self, chain):
        for i in range(5):
            chain.append("test", {"i": i}, agent_id=f"agent_{i}")
        entries = chain.tail(3)
        assert len(entries) == 3
        # Should be in chronological order
        assert entries[0].timestamp <= entries[1].timestamp <= entries[2].timestamp

    def test_search_by_action_type(self, chain):
        chain.append("tool_call", {"t": "a"})
        chain.append("hand_start", {"h": "researcher"})
        chain.append("tool_call", {"t": "b"})
        results = chain.search(action_type="hand_start")
        assert len(results) == 1
        assert "researcher" in results[0].action_data

    def test_search_by_agent_id(self, chain):
        chain.append("test", {}, agent_id="main")
        chain.append("test", {}, agent_id="hand:guardian")
        chain.append("test", {}, agent_id="main")
        results = chain.search(agent_id="hand:guardian")
        assert len(results) == 1

    def test_genesis_hash(self, chain):
        chain.append("first", {})
        entries = chain.tail(1)
        assert entries[0].prev_hash == "0" * 64

    def test_hash_chain_linkage(self, chain):
        chain.append("a", {})
        chain.append("b", {})
        entries = chain.tail(2)
        assert entries[1].prev_hash == entries[0].entry_hash

    def test_empty_chain_valid(self, chain):
        valid, count, err = chain.verify()
        assert valid
        assert count == 0


# ── Tool Signing ────────────────────────────────────────────────────────────

class TestToolSigning:
    def test_sign_and_verify(self, tmp_path):
        from aura.security.tool_signing import sign_tool, verify_tool
        tool = tmp_path / "my_tool.py"
        tool.write_text("class T:\n    def execute(self): pass")
        sig_path = sign_tool(str(tool))
        assert os.path.exists(sig_path)
        valid, err = verify_tool(str(tool))
        assert valid
        assert err is None

    def test_tamper_detected(self, tmp_path):
        from aura.security.tool_signing import sign_tool, verify_tool
        tool = tmp_path / "tamper_tool.py"
        tool.write_text("x = 1")
        sign_tool(str(tool))
        tool.write_text("x = 2  # tampered")
        valid, err = verify_tool(str(tool))
        assert not valid
        assert "modified" in err.lower()

    def test_missing_sig_file(self, tmp_path):
        from aura.security.tool_signing import verify_tool
        tool = tmp_path / "no_sig.py"
        tool.write_text("x = 1")
        valid, err = verify_tool(str(tool))
        assert not valid
        assert "No signature" in err

    def test_missing_tool_file(self, tmp_path):
        from aura.security.tool_signing import verify_tool
        valid, err = verify_tool(str(tmp_path / "ghost.py"))
        assert not valid

    def test_is_tool_signed(self, tmp_path):
        from aura.security.tool_signing import sign_tool, is_tool_signed
        tool = tmp_path / "check.py"
        tool.write_text("y = 1")
        assert not is_tool_signed(str(tool))
        sign_tool(str(tool))
        assert is_tool_signed(str(tool))

    def test_sig_file_has_metadata(self, tmp_path):
        from aura.security.tool_signing import sign_tool
        tool = tmp_path / "meta.py"
        tool.write_text("z = 1")
        sig_path = sign_tool(str(tool))
        with open(sig_path) as f:
            data = json.load(f)
        assert data["version"] == 1
        assert data["signed_at"] > 0
        assert data["algorithm"] in ("ed25519", "hmac-sha256")


# ── Hands ───────────────────────────────────────────────────────────────────

class TestHands:
    def test_hand_manifest(self):
        from aura.hands.researcher import ResearcherHand
        h = ResearcherHand()
        m = h.manifest
        assert m.name == "researcher"
        assert m.idle_only is True
        assert m.trigger_on_drive == "curiosity"

    def test_hand_states(self):
        from aura.hands.base import HandState
        from aura.hands.guardian import GuardianHand
        g = GuardianHand()
        assert g.state == HandState.INACTIVE
        g.state = HandState.ACTIVE
        assert g.state == HandState.ACTIVE

    def test_hand_can_run_idle_check(self):
        from aura.hands.researcher import ResearcherHand
        r = ResearcherHand()
        r.state = r.state.__class__("active")
        # Researcher needs 600s idle
        assert not r.can_run(idle_seconds=0)
        assert r.can_run(idle_seconds=700)

    def test_guardian_runs_without_idle(self):
        from aura.hands.guardian import GuardianHand
        g = GuardianHand()
        g.state = g.state.__class__("active")
        assert g.can_run(idle_seconds=0)

    def test_hand_circuit_breaker(self):
        from aura.hands.base import HandResult
        from aura.hands.researcher import ResearcherHand
        r = ResearcherHand()
        r.state = r.state.__class__("active")
        # Simulate 3 consecutive failures
        for _ in range(3):
            r.record_run(HandResult(hand_name="researcher", success=False, summary="fail", error="test"))
        assert r._consecutive_failures == 3
        # Should not run even with enough idle (circuit breaker)
        assert not r.can_run(idle_seconds=9999)

    def test_drive_trigger(self):
        from aura.hands.researcher import ResearcherHand
        r = ResearcherHand()
        r.state = r.state.__class__("active")
        # Drive above threshold should trigger regardless of interval
        drives = {"curiosity": 0.9}
        assert r.can_run(idle_seconds=700, drive_urgencies=drives)

    def test_drive_below_threshold_no_trigger(self):
        from aura.hands.researcher import ResearcherHand
        r = ResearcherHand()
        r.state = r.state.__class__("active")
        r._last_run = time.time()  # Just ran — interval not elapsed
        drives = {"curiosity": 0.3}  # Below 0.7 threshold
        assert not r.can_run(idle_seconds=700, drive_urgencies=drives)

    def test_hand_stats_enriched(self):
        from aura.hands.guardian import GuardianHand
        g = GuardianHand()
        stats = g.get_stats()
        assert "description" in stats
        assert "model_preference" in stats
        assert "trigger_on_drive" in stats
        assert stats["last_run"] is None

    def test_manager_register_and_list(self):
        from aura.hands.manager import HandManager
        from aura.hands.guardian import GuardianHand
        from aura.hands.researcher import ResearcherHand
        m = HandManager()
        m.register(GuardianHand())
        m.register(ResearcherHand())
        hands = m.list_hands()
        assert len(hands) == 2
        names = {h["name"] for h in hands}
        assert names == {"guardian", "researcher"}

    def test_manager_activate_deactivate(self):
        from aura.hands.manager import HandManager
        from aura.hands.guardian import GuardianHand
        m = HandManager()
        m.register(GuardianHand())
        assert m.activate("guardian")
        assert m.get_hand("guardian").state.value == "active"
        assert m.deactivate("guardian")
        assert m.get_hand("guardian").state.value == "inactive"

    def test_manager_unknown_hand(self):
        from aura.hands.manager import HandManager
        m = HandManager()
        assert not m.activate("nonexistent")

    def test_record_run_resets_failures_on_success(self):
        from aura.hands.base import HandResult
        from aura.hands.guardian import GuardianHand
        g = GuardianHand()
        g.record_run(HandResult(hand_name="guardian", success=False, summary="", error="e"))
        g.record_run(HandResult(hand_name="guardian", success=False, summary="", error="e"))
        assert g._consecutive_failures == 2
        g.record_run(HandResult(hand_name="guardian", success=True, summary="ok"))
        assert g._consecutive_failures == 0
