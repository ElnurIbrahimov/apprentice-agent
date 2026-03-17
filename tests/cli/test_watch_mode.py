"""Tests for watch mode."""
import pytest
from pathlib import Path
from aura.cli.watch_mode import (
    FileWatcher, WatchHit, WATCH_PATTERNS, remove_ai_comment,
    create_watch_indicator,
)

def test_scan_python_file(tmp_path):
    f = tmp_path / "test.py"
    f.write_text("x = 1\n# AURA: fix this variable name\ny = 2\n")
    watcher = FileWatcher(root=str(tmp_path))
    hits = watcher.scan_file(f)
    assert len(hits) == 1
    assert hits[0].instruction == "fix this variable name"
    assert hits[0].line_number == 2

def test_scan_js_file(tmp_path):
    f = tmp_path / "app.js"
    f.write_text("const x = 1;\n// AI: optimize this function\nfunction slow() {}\n")
    watcher = FileWatcher(root=str(tmp_path))
    hits = watcher.scan_file(f)
    assert len(hits) == 1
    assert "optimize" in hits[0].instruction

def test_scan_no_hits(tmp_path):
    f = tmp_path / "clean.py"
    f.write_text("# Normal comment\nx = 1\n")
    watcher = FileWatcher(root=str(tmp_path))
    hits = watcher.scan_file(f)
    assert len(hits) == 0

def test_scan_all(tmp_path):
    (tmp_path / "a.py").write_text("# AURA: fix a\n")
    (tmp_path / "b.py").write_text("# AURA: fix b\n")
    (tmp_path / "c.txt").write_text("# AURA: ignored extension\n")  # .txt not watched
    watcher = FileWatcher(root=str(tmp_path))
    hits = watcher.scan_all()
    assert len(hits) == 2

def test_dedup_hits(tmp_path):
    f = tmp_path / "test.py"
    f.write_text("# AURA: fix this\n")
    watcher = FileWatcher(root=str(tmp_path))
    hits1 = watcher.scan_file(f)
    hits2 = watcher.scan_file(f)  # Second scan — deduped via _seen
    assert len(hits1) == 1
    assert len(hits2) == 0  # No duplicates

def test_mark_resolved(tmp_path):
    f = tmp_path / "test.py"
    f.write_text("# AURA: fix this\n")
    watcher = FileWatcher(root=str(tmp_path))
    hits = watcher.scan_file(f)
    watcher._hits.extend(hits)
    assert len(watcher.get_unresolved()) == 1
    watcher.mark_resolved(hits[0], response="Fixed!")
    assert len(watcher.get_unresolved()) == 0

def test_remove_ai_comment(tmp_path):
    f = tmp_path / "test.py"
    f.write_text("x = 1  # AURA: rename this\ny = 2\n")
    success = remove_ai_comment(str(f), 1)
    assert success
    content = f.read_text()
    assert "AURA" not in content
    assert "x = 1" in content

def test_remove_standalone_comment(tmp_path):
    f = tmp_path / "test.py"
    f.write_text("# AURA: add docstring\ndef hello():\n    pass\n")
    success = remove_ai_comment(str(f), 1)
    assert success
    content = f.read_text()
    assert "AURA" not in content

def test_watch_indicator_not_running():
    watcher = FileWatcher()
    assert create_watch_indicator(watcher) == ""

def test_watch_indicator_running_no_hits(tmp_path):
    watcher = FileWatcher(root=str(tmp_path))
    watcher._running = True  # Simulate running
    indicator = create_watch_indicator(watcher)
    assert "watching" in indicator

def test_skip_dirs(tmp_path):
    node_modules = tmp_path / "node_modules"
    node_modules.mkdir()
    (node_modules / "lib.js").write_text("// AURA: should be ignored\n")
    (tmp_path / "app.js").write_text("// AURA: should be found\n")
    watcher = FileWatcher(root=str(tmp_path))
    hits = watcher.scan_all()
    assert len(hits) == 1
    assert "app.js" in hits[0].file_path

def test_clear(tmp_path):
    f = tmp_path / "test.py"
    f.write_text("# AURA: fix\n")
    watcher = FileWatcher(root=str(tmp_path))
    watcher.scan_file(f)
    watcher._hits.extend(watcher.scan_file(f) or [])
    watcher.clear()
    assert len(watcher.get_hits()) == 0
