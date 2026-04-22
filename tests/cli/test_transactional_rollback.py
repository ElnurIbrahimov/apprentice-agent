"""Tests for transactional turn-scoped rollback in AgenticLoop."""
from __future__ import annotations

from pathlib import Path
from unittest.mock import MagicMock


def _make_loop_stub():
    """Minimal stand-in for AgenticLoop's rollback surface. We instantiate the
    real class methods via `types.MethodType` on a bare object, so the tests
    exercise the actual _ensure_turn_checkpoint / _rollback_turn / _clear_turn_checkpoint
    implementations without pulling the whole 1900-line loop into scope."""
    import types
    from aura.core.agentic_loop import AgenticLoop

    obj = types.SimpleNamespace()
    obj._turn_checkpoint_ids = []
    obj._turn_snapshotted_paths = set()
    obj._current_run_id = "test_run"
    obj._checkpoint_mgr = None
    obj._ensure_turn_checkpoint = types.MethodType(AgenticLoop._ensure_turn_checkpoint, obj)
    obj._rollback_turn = types.MethodType(AgenticLoop._rollback_turn, obj)
    obj._clear_turn_checkpoint = types.MethodType(AgenticLoop._clear_turn_checkpoint, obj)
    return obj


def test_ensure_turn_checkpoint_no_manager_is_noop():
    loop = _make_loop_stub()
    loop._ensure_turn_checkpoint(["/some/path.py"])
    assert loop._turn_checkpoint_ids == []


def test_ensure_turn_checkpoint_snapshots_new_paths(tmp_path):
    from aura.cli.checkpoint import CheckpointManager

    f = tmp_path / "foo.py"
    f.write_text("x = 1", encoding="utf-8")

    loop = _make_loop_stub()
    loop._checkpoint_mgr = CheckpointManager(checkpoint_dir=tmp_path / ".cp")

    loop._ensure_turn_checkpoint([str(f)])
    assert len(loop._turn_checkpoint_ids) == 1
    assert str(f) in loop._turn_snapshotted_paths


def test_ensure_turn_checkpoint_deduplicates(tmp_path):
    """Calling _ensure_turn_checkpoint for the same path twice should only
    snapshot once — the pre-turn content is captured by the first call."""
    from aura.cli.checkpoint import CheckpointManager

    f = tmp_path / "foo.py"
    f.write_text("x = 1", encoding="utf-8")

    loop = _make_loop_stub()
    loop._checkpoint_mgr = CheckpointManager(checkpoint_dir=tmp_path / ".cp")

    loop._ensure_turn_checkpoint([str(f)])
    first_len = len(loop._turn_checkpoint_ids)
    loop._ensure_turn_checkpoint([str(f)])
    # No new checkpoint because the path was already tracked.
    assert len(loop._turn_checkpoint_ids) == first_len


def test_ensure_turn_checkpoint_adds_new_paths_across_iterations(tmp_path):
    from aura.cli.checkpoint import CheckpointManager

    a = tmp_path / "a.py"
    a.write_text("a", encoding="utf-8")
    b = tmp_path / "b.py"
    b.write_text("b", encoding="utf-8")

    loop = _make_loop_stub()
    loop._checkpoint_mgr = CheckpointManager(checkpoint_dir=tmp_path / ".cp")

    loop._ensure_turn_checkpoint([str(a)])
    loop._ensure_turn_checkpoint([str(b)])  # different path — should snapshot
    assert len(loop._turn_checkpoint_ids) == 2
    assert str(a) in loop._turn_snapshotted_paths
    assert str(b) in loop._turn_snapshotted_paths


def test_rollback_restores_pre_turn_content(tmp_path):
    """Golden path: snapshot → edit → rollback → original content restored."""
    from aura.cli.checkpoint import CheckpointManager

    f = tmp_path / "foo.py"
    f.write_text("original", encoding="utf-8")

    loop = _make_loop_stub()
    loop._checkpoint_mgr = CheckpointManager(checkpoint_dir=tmp_path / ".cp")

    loop._ensure_turn_checkpoint([str(f)])
    # Simulate edits happening
    f.write_text("edited", encoding="utf-8")
    assert f.read_text(encoding="utf-8") == "edited"

    result = loop._rollback_turn()
    assert result["attempted"] == 1
    assert result["restored"] == 1
    assert result["partial"] is False
    assert f.read_text(encoding="utf-8") == "original"


def test_rollback_deletes_files_that_did_not_exist(tmp_path):
    """Agent writes a NEW file, verification fails → file should be deleted."""
    from aura.cli.checkpoint import CheckpointManager

    new_path = tmp_path / "newfile.py"  # doesn't exist yet

    loop = _make_loop_stub()
    loop._checkpoint_mgr = CheckpointManager(checkpoint_dir=tmp_path / ".cp")

    # Snapshot BEFORE the new file exists (capturing "did not exist" state).
    loop._ensure_turn_checkpoint([str(new_path)])
    # Agent creates the file.
    new_path.write_text("brand new", encoding="utf-8")
    assert new_path.is_file()

    loop._rollback_turn()
    assert not new_path.exists()


def test_rollback_restores_multiple_files_across_checkpoints(tmp_path):
    """Turn touches 3 files across 2 iterations; one rollback restores all."""
    from aura.cli.checkpoint import CheckpointManager

    a = tmp_path / "a.py"
    a.write_text("A", encoding="utf-8")
    b = tmp_path / "b.py"
    b.write_text("B", encoding="utf-8")
    c = tmp_path / "c.py"
    c.write_text("C", encoding="utf-8")

    loop = _make_loop_stub()
    loop._checkpoint_mgr = CheckpointManager(checkpoint_dir=tmp_path / ".cp")

    loop._ensure_turn_checkpoint([str(a), str(b)])  # iter 1
    a.write_text("A2", encoding="utf-8")
    b.write_text("B2", encoding="utf-8")

    loop._ensure_turn_checkpoint([str(c)])  # iter 2 adds c
    c.write_text("C2", encoding="utf-8")

    loop._rollback_turn()
    assert a.read_text(encoding="utf-8") == "A"
    assert b.read_text(encoding="utf-8") == "B"
    assert c.read_text(encoding="utf-8") == "C"


def test_rollback_emits_event_with_correct_payload(tmp_path):
    from aura.cli.checkpoint import CheckpointManager

    f = tmp_path / "foo.py"
    f.write_text("pre", encoding="utf-8")

    loop = _make_loop_stub()
    loop._checkpoint_mgr = CheckpointManager(checkpoint_dir=tmp_path / ".cp")
    loop._ensure_turn_checkpoint([str(f)])
    f.write_text("post", encoding="utf-8")

    emitter = MagicMock()
    loop._rollback_turn(event_emitter=emitter)

    emitter.emit.assert_called_once()
    event_type = emitter.emit.call_args.args[0]
    payload = emitter.emit.call_args.kwargs
    assert event_type == "turn_rolled_back"
    assert payload["restored"] == 1
    assert payload["attempted"] == 1
    assert payload["partial"] is False
    assert len(payload["paths"]) == 1


def test_clear_turn_checkpoint_leaves_files_alone(tmp_path):
    """On verification success, clearing state must NOT touch files."""
    from aura.cli.checkpoint import CheckpointManager

    f = tmp_path / "foo.py"
    f.write_text("pre", encoding="utf-8")

    loop = _make_loop_stub()
    loop._checkpoint_mgr = CheckpointManager(checkpoint_dir=tmp_path / ".cp")
    loop._ensure_turn_checkpoint([str(f)])
    f.write_text("post", encoding="utf-8")

    loop._clear_turn_checkpoint()
    assert loop._turn_checkpoint_ids == []
    assert loop._turn_snapshotted_paths == set()
    assert f.read_text(encoding="utf-8") == "post"  # unchanged


def test_rollback_with_no_checkpoints_is_safe(tmp_path):
    """Rollback called with empty state should not raise."""
    loop = _make_loop_stub()
    result = loop._rollback_turn()
    assert result["attempted"] == 0
    assert result["restored"] == 0
