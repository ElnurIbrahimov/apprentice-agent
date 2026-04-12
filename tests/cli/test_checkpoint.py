"""Tests for checkpoint system."""
import pytest
from pathlib import Path
from aura.cli.checkpoint import CheckpointManager

def test_snapshot_and_restore(tmp_path):
    mgr = CheckpointManager(checkpoint_dir=tmp_path / ".aura_checkpoints")
    test_file = tmp_path / "test.py"
    test_file.write_text("original content")

    cp_id = mgr.snapshot(str(test_file), label="before edit")
    assert cp_id is not None

    test_file.write_text("modified content")
    assert test_file.read_text() == "modified content"

    mgr.restore(cp_id)
    assert test_file.read_text() == "original content"

def test_list_checkpoints(tmp_path):
    mgr = CheckpointManager(checkpoint_dir=tmp_path / ".aura_checkpoints")
    test_file = tmp_path / "test.py"
    test_file.write_text("v1")
    mgr.snapshot(str(test_file), label="version 1")
    test_file.write_text("v2")
    mgr.snapshot(str(test_file), label="version 2")

    cps = mgr.list_checkpoints()
    assert len(cps) == 2
    assert cps[0]["label"] == "version 2"  # Most recent first

def test_prune_old_checkpoints(tmp_path):
    mgr = CheckpointManager(checkpoint_dir=tmp_path / ".aura_checkpoints", max_checkpoints=3)
    test_file = tmp_path / "test.py"
    for i in range(5):
        test_file.write_text(f"version {i}")
        mgr.snapshot(str(test_file), label=f"v{i}")
    cps = mgr.list_checkpoints()
    assert len(cps) <= 3

def test_multi_file_snapshot(tmp_path):
    mgr = CheckpointManager(checkpoint_dir=tmp_path / ".aura_checkpoints")
    f1 = tmp_path / "a.py"
    f2 = tmp_path / "b.py"
    f1.write_text("file a")
    f2.write_text("file b")

    cp_id = mgr.snapshot_multi([str(f1), str(f2)], label="multi edit")
    f1.write_text("changed a")
    f2.write_text("changed b")

    mgr.restore(cp_id)
    assert f1.read_text() == "file a"
    assert f2.read_text() == "file b"

def test_restore_nonexistent():
    import tempfile
    mgr = CheckpointManager(checkpoint_dir=Path(tempfile.mkdtemp()) / ".cp")
    assert mgr.restore("nonexistent_id") is False

def test_clear(tmp_path):
    mgr = CheckpointManager(checkpoint_dir=tmp_path / ".aura_checkpoints")
    test_file = tmp_path / "test.py"
    test_file.write_text("data")
    mgr.snapshot(str(test_file), label="test")
    assert len(mgr.list_checkpoints()) == 1
    mgr.clear()
    assert len(mgr.list_checkpoints()) == 0


def test_restore_removes_new_file_created_after_snapshot(tmp_path):
    mgr = CheckpointManager(checkpoint_dir=tmp_path / ".aura_checkpoints")
    new_file = tmp_path / "new_file.py"

    cp_id = mgr.snapshot(str(new_file), label="before create")
    new_file.write_text("generated later")
    assert new_file.exists()

    assert mgr.restore(cp_id) is True
    assert not new_file.exists()


def test_restore_mixed_existing_and_new_files(tmp_path):
    mgr = CheckpointManager(checkpoint_dir=tmp_path / ".aura_checkpoints")
    existing = tmp_path / "existing.py"
    created = tmp_path / "created.py"
    existing.write_text("before")

    cp_id = mgr.snapshot_multi([str(existing), str(created)], label="mixed")
    existing.write_text("after")
    created.write_text("brand new")

    assert mgr.restore(cp_id) is True
    assert existing.read_text() == "before"
    assert not created.exists()
