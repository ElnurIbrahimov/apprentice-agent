"""Tests for the crash-safe atomic_write_json helper in aura.paths."""
from __future__ import annotations

import json
from pathlib import Path

import pytest

from aura.paths import atomic_write_json


def test_atomic_write_basic_roundtrip(tmp_path: Path):
    target = tmp_path / "data.json"
    atomic_write_json(target, {"hello": "world", "n": 7})
    loaded = json.loads(target.read_text(encoding="utf-8"))
    assert loaded == {"hello": "world", "n": 7}


def test_atomic_write_creates_parent_dirs(tmp_path: Path):
    target = tmp_path / "nested" / "deep" / "data.json"
    assert not target.parent.exists()
    atomic_write_json(target, {"ok": True})
    assert target.exists()


def test_atomic_write_overwrites_existing(tmp_path: Path):
    target = tmp_path / "data.json"
    target.write_text('{"old": true}', encoding="utf-8")
    atomic_write_json(target, {"new": True})
    assert json.loads(target.read_text(encoding="utf-8")) == {"new": True}


def test_atomic_write_no_tempfile_left_behind(tmp_path: Path):
    target = tmp_path / "data.json"
    atomic_write_json(target, {"x": 1})
    # No sibling .tmp files
    tmps = list(tmp_path.glob("*.tmp"))
    assert tmps == [], f"tempfiles leaked: {tmps}"


def test_atomic_write_cleans_tempfile_on_error(tmp_path: Path, monkeypatch):
    target = tmp_path / "data.json"

    class Unpicklable:
        pass

    # json.dumps on an unserialisable object raises TypeError inside atomic_write_json.
    with pytest.raises(TypeError):
        atomic_write_json(target, {"bad": Unpicklable()})

    # No tempfile should survive the failed write.
    tmps = list(tmp_path.glob("*.tmp"))
    assert tmps == []
    assert not target.exists()


def test_atomic_write_unicode_roundtrip(tmp_path: Path):
    target = tmp_path / "data.json"
    payload = {"name": "Aura", "note": "こんにちは 🌊"}
    atomic_write_json(target, payload)
    assert json.loads(target.read_text(encoding="utf-8")) == payload
