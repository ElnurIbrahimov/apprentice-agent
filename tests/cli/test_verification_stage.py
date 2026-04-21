"""Tests for aura/core/verification_stage.py."""
from __future__ import annotations

from unittest.mock import MagicMock, patch

from aura.core.verification_stage import (
    VerificationOutcome,
    VerificationStage,
    _EDIT_TOOL_NAMES,
)


# ── Mode selection ───────────────────────────────────────────────────────

def test_default_mode_is_typecheck():
    vs = VerificationStage(project_root=".", aura_config={})
    assert vs.mode == "typecheck"


def test_explicit_mode_respected():
    vs = VerificationStage(project_root=".", aura_config={"verification": {"on_edit": "both"}})
    assert vs.mode == "both"


def test_legacy_auto_test_flag_maps_to_both():
    vs = VerificationStage(project_root=".", aura_config={"auto_test": True})
    assert vs.mode == "both"


def test_invalid_mode_falls_back_to_typecheck(caplog):
    vs = VerificationStage(
        project_root=".",
        aura_config={"verification": {"on_edit": "bogus"}},
    )
    assert vs.mode == "typecheck"


def test_mode_none_disables_should_run():
    vs = VerificationStage(project_root=".", aura_config={"verification": {"on_edit": "none"}})
    assert vs.should_run("edit_file") is False


def test_should_run_only_for_edit_tools():
    vs = VerificationStage(project_root=".", aura_config={})
    # Every edit tool triggers
    for tool in _EDIT_TOOL_NAMES:
        assert vs.should_run(tool) is True
    # Non-edit tools don't
    assert vs.should_run("read_file") is False
    assert vs.should_run("grep") is False


# ── run() dispatch ───────────────────────────────────────────────────────

def test_run_with_none_mode_short_circuits():
    vs = VerificationStage(project_root=".", aura_config={"verification": {"on_edit": "none"}})
    outcome = vs.run(["foo.py"])
    assert outcome.success is True
    assert outcome.stages == []
    assert outcome.skipped_reason == "mode=none"


def test_run_with_no_changed_files_short_circuits():
    vs = VerificationStage(project_root=".", aura_config={"verification": {"on_edit": "typecheck"}})
    outcome = vs.run([])
    assert outcome.success is True
    assert outcome.stages == []


def test_run_typecheck_mode_calls_typecheck_only():
    vs = VerificationStage(project_root=".", aura_config={"verification": {"on_edit": "typecheck"}})
    with patch.object(vs, "_run_typecheck", return_value={"name": "typecheck", "runner": "none", "success": True, "duration_s": 0.0, "failures": []}) as tc, \
         patch.object(vs, "_run_tests") as tt:
        outcome = vs.run(["foo.py"])
    tc.assert_called_once()
    tt.assert_not_called()
    assert outcome.success is True


def test_run_both_mode_calls_both_when_typecheck_passes():
    vs = VerificationStage(project_root=".", aura_config={"verification": {"on_edit": "both"}})
    with patch.object(vs, "_run_typecheck", return_value={"name": "typecheck", "runner": "mypy", "success": True, "duration_s": 0.1, "failures": []}) as tc, \
         patch.object(vs, "_run_tests", return_value={"name": "tests", "runner": "pytest", "success": True, "duration_s": 0.3, "failures": []}) as tt:
        outcome = vs.run(["foo.py"])
    tc.assert_called_once()
    tt.assert_called_once()
    assert outcome.success is True


def test_run_both_mode_skips_tests_on_typecheck_failure():
    """Fail fast: if typecheck is red, skip tests to save time."""
    vs = VerificationStage(project_root=".", aura_config={"verification": {"on_edit": "both"}})
    with patch.object(vs, "_run_typecheck", return_value={"name": "typecheck", "runner": "mypy", "success": False, "duration_s": 0.1, "failures": [{"file": "foo.py", "line": 1, "message": "type error"}]}) as tc, \
         patch.object(vs, "_run_tests") as tt:
        outcome = vs.run(["foo.py"])
    tc.assert_called_once()
    tt.assert_not_called()
    assert outcome.success is False


def test_run_emits_events_when_emitter_provided():
    vs = VerificationStage(project_root=".", aura_config={"verification": {"on_edit": "typecheck"}})
    emitter = MagicMock()
    with patch.object(vs, "_run_typecheck", return_value={"name": "typecheck", "runner": "mypy", "success": True, "duration_s": 0.1, "failures": []}):
        vs.run(["foo.py"], emitter=emitter)
    # Should have fired at least start + passed
    emitted_types = [call.args[0] for call in emitter.emit.call_args_list]
    assert "verification_start" in emitted_types
    assert "verification_passed" in emitted_types


def test_run_emits_failed_event_on_failure():
    vs = VerificationStage(project_root=".", aura_config={"verification": {"on_edit": "typecheck"}})
    emitter = MagicMock()
    with patch.object(vs, "_run_typecheck", return_value={"name": "typecheck", "runner": "mypy", "success": False, "duration_s": 0.2, "failures": [{"file": "f", "line": 1, "message": "err"}]}):
        vs.run(["foo.py"], emitter=emitter)
    emitted_types = [call.args[0] for call in emitter.emit.call_args_list]
    assert "verification_failed" in emitted_types


# ── Outcome formatting ────────────────────────────────────────────────────

def test_outcome_success_produces_empty_conversation_message():
    oc = VerificationOutcome(mode="typecheck", success=True, duration_s=0.1)
    assert oc.to_conversation_message() == ""


def test_outcome_failure_formats_file_line_message():
    oc = VerificationOutcome(
        mode="typecheck", success=False, duration_s=0.2,
        stages=[{
            "name": "typecheck",
            "runner": "mypy",
            "success": False,
            "duration_s": 0.2,
            "failures": [
                {"file": "src/foo.py", "line": 42, "message": "incompatible type"},
            ],
        }],
    )
    msg = oc.to_conversation_message()
    assert "Verification failed" in msg
    assert "src/foo.py:42" in msg
    assert "incompatible type" in msg
    assert "Fix these before continuing" in msg


def test_outcome_failure_truncates_many_failures():
    failures = [
        {"file": f"f{i}.py", "line": i, "message": f"err{i}"}
        for i in range(15)
    ]
    oc = VerificationOutcome(
        mode="typecheck", success=False, duration_s=0.2,
        stages=[{"name": "typecheck", "runner": "mypy", "success": False,
                 "failures": failures, "duration_s": 0.2}],
    )
    msg = oc.to_conversation_message()
    # Only first 10 should be inline; the rest summarized.
    assert "and 5 more" in msg
