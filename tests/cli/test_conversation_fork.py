"""Tests for git-style conversation branching."""
import json
import pytest
from pathlib import Path

from aura.core.conversation_fork import Branch, ConversationTree


# ── Branch data class ────────────────────────────────────────────────

def test_branch_round_trip():
    b = Branch(
        id="fork-1", parent_id="main", name="Test Branch",
        created_at=1000.0, history=[{"role": "user", "content": "hi"}],
        fork_point=0, metadata={"tag": "test"},
    )
    d = b.to_dict()
    b2 = Branch.from_dict(d)
    assert b2.id == b.id
    assert b2.parent_id == b.parent_id
    assert b2.name == b.name
    assert b2.history == b.history
    assert b2.fork_point == b.fork_point
    assert b2.metadata == b.metadata


def test_branch_from_dict_defaults():
    b = Branch.from_dict({"id": "x"})
    assert b.parent_id is None
    assert b.name == "x"
    assert b.history == []
    assert b.fork_point == 0


# ── ConversationTree basics ──────────────────────────────────────────

def test_init_creates_main_branch():
    tree = ConversationTree()
    assert "main" in tree.branches
    assert tree.current_branch == "main"


def test_fork_creates_new_branch():
    tree = ConversationTree()
    main = tree.branches["main"]
    main.history = [{"role": "user", "content": "hello"}]

    new_branch = tree.fork(name="experiment")
    assert new_branch.id == "fork-1"
    assert new_branch.name == "experiment"
    assert new_branch.parent_id == "main"
    assert tree.current_branch == "fork-1"
    # History copied
    assert new_branch.history == [{"role": "user", "content": "hello"}]
    # Fork point at end of parent
    assert new_branch.fork_point == 1


def test_fork_copies_history():
    tree = ConversationTree()
    main = tree.branches["main"]
    main.history = [
        {"role": "user", "content": "msg1"},
        {"role": "assistant", "content": "resp1"},
    ]

    fork = tree.fork()
    assert len(fork.history) == 2
    assert fork.history[0]["content"] == "msg1"


def test_history_isolation():
    """Mutating one branch's history must not affect another."""
    tree = ConversationTree()
    main = tree.branches["main"]
    main.history = [{"role": "user", "content": "original"}]

    fork = tree.fork()

    # Append to fork
    fork.history.append({"role": "assistant", "content": "fork-only"})

    # Main should be untouched
    assert len(main.history) == 1
    assert main.history[0]["content"] == "original"

    # Fork should have the new message
    assert len(fork.history) == 2


def test_switch_branch():
    tree = ConversationTree()
    tree.fork(name="branch-a")
    assert tree.current_branch == "fork-1"

    branch = tree.switch("main")
    assert tree.current_branch == "main"
    assert branch.id == "main"


def test_switch_numeric_shorthand():
    tree = ConversationTree()
    tree.fork()
    tree.switch("main")

    branch = tree.switch("1")  # numeric shorthand for fork-1
    assert branch.id == "fork-1"


def test_switch_nonexistent_raises():
    tree = ConversationTree()
    with pytest.raises(KeyError, match="not found"):
        tree.switch("nonexistent-branch")


def test_merge_brings_new_messages():
    tree = ConversationTree()
    main = tree.branches["main"]
    main.history = [{"role": "user", "content": "base"}]

    tree.fork()
    fork = tree.get_current()
    fork.history.append({"role": "assistant", "content": "new in fork"})
    fork.history.append({"role": "user", "content": "follow-up"})

    result = tree.merge_to_parent()
    assert result["merged"] == 2
    assert result["target"] == "main"
    assert tree.current_branch == "main"

    # Main should now have the new messages
    assert len(main.history) == 3
    assert main.history[-1]["content"] == "follow-up"


def test_merge_nothing_to_merge():
    tree = ConversationTree()
    tree.fork()
    # Don't add anything new to fork
    result = tree.merge_to_parent()
    assert result["merged"] == 0
    assert tree.current_branch == "main"


def test_merge_main_fails():
    tree = ConversationTree()
    result = tree.merge_to_parent()
    assert "error" in result


def test_double_merge():
    """After merge, trying to merge the same fork again should be a no-op
    (we switched to parent after first merge)."""
    tree = ConversationTree()
    main = tree.branches["main"]
    main.history = [{"role": "user", "content": "base"}]

    tree.fork()
    fork = tree.get_current()
    fork.history.append({"role": "assistant", "content": "new"})

    result1 = tree.merge_to_parent()
    assert result1["merged"] == 1
    assert tree.current_branch == "main"

    # Now on main — merging main has no parent
    result2 = tree.merge_to_parent()
    assert "error" in result2


def test_list_branches():
    tree = ConversationTree()
    tree.fork(name="A")
    tree.switch("main")
    tree.fork(name="B")

    branches = tree.list_branches()
    assert branches[0].id == "main"  # main first
    assert len(branches) == 3
    names = [b.name for b in branches]
    assert "A" in names
    assert "B" in names


def test_get_children():
    tree = ConversationTree()
    tree.fork(name="child1")
    tree.switch("main")
    tree.fork(name="child2")

    children = tree.get_children("main")
    assert len(children) == 2


def test_sync_history():
    tree = ConversationTree()
    external = [{"role": "user", "content": "synced"}]
    tree.sync_history(external)
    assert tree.get_current().history == external


def test_fork_with_empty_history():
    tree = ConversationTree()
    fork = tree.fork()
    assert fork.history == []
    assert fork.fork_point == 0


# ── Save / Load round-trip ───────────────────────────────────────────

def test_save_load_round_trip(tmp_path):
    tree = ConversationTree(session_dir=tmp_path)
    main = tree.branches["main"]
    main.history = [{"role": "user", "content": "hello"}]

    tree.fork(name="experiment")
    fork = tree.get_current()
    fork.history.append({"role": "assistant", "content": "world"})

    tree.save()
    assert (tmp_path / "branches.json").exists()

    # Load into new tree
    tree2 = ConversationTree(session_dir=tmp_path)
    loaded = tree2.load()
    assert loaded is True
    assert tree2.current_branch == "fork-1"
    assert "fork-1" in tree2.branches
    assert tree2.branches["fork-1"].name == "experiment"
    assert len(tree2.branches["fork-1"].history) == 2


def test_save_no_session_dir():
    """save() with no session_dir is a no-op, should not crash."""
    tree = ConversationTree(session_dir=None)
    tree.save()  # Should not raise


def test_load_no_file(tmp_path):
    tree = ConversationTree(session_dir=tmp_path)
    assert tree.load() is False


def test_load_corrupt_json(tmp_path):
    (tmp_path / "branches.json").write_text("not valid json", encoding="utf-8")
    tree = ConversationTree(session_dir=tmp_path)
    assert tree.load() is False
    # main should still exist after failed load
    assert "main" in tree.branches
