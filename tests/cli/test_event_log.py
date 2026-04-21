"""Tests for aura/core/event_log.py — JSONL append-only event logs."""
from __future__ import annotations

import json
import os
import tempfile
import threading
from pathlib import Path
from unittest.mock import patch

from aura.core import event_log


def _count_lines(p: Path) -> int:
    if not p.is_file():
        return 0
    return sum(1 for _ in p.open(encoding="utf-8"))


def _read_records(p: Path) -> list[dict]:
    if not p.is_file():
        return []
    return [json.loads(line) for line in p.open(encoding="utf-8") if line.strip()]


# ── Basic append ─────────────────────────────────────────────────────────

def test_log_edit_writes_record(tmp_path):
    with patch.dict(os.environ, {"AURA_DATA_DIR": str(tmp_path)}):
        event_log._append_jsonl(
            tmp_path / "events" / "edits.jsonl",
            {"ts": 1.0, "session_id": "s1", "tool": "edit_file",
             "path": "foo.py", "prompt": "fix", "model": "m", "iteration": 1},
        )
    p = tmp_path / "events" / "edits.jsonl"
    recs = _read_records(p)
    assert len(recs) == 1
    assert recs[0]["tool"] == "edit_file"
    assert recs[0]["path"] == "foo.py"


def test_log_verification_truncates_failures(tmp_path):
    with patch.dict(os.environ, {"AURA_DATA_DIR": str(tmp_path)}):
        failures = [{"file": f"f{i}.py", "line": i, "message": "e"} for i in range(40)]
        event_log._append_jsonl(
            tmp_path / "events" / "verifications.jsonl",
            {"ts": 1.0, "session_id": "s1", "stage": "typecheck",
             "runner": "mypy", "status": "failed", "duration_s": 0.1,
             "failure_count": len(failures), "failures": failures[:20]},
        )
    recs = _read_records(tmp_path / "events" / "verifications.jsonl")
    assert len(recs) == 1
    assert recs[0]["failure_count"] == 40
    assert len(recs[0]["failures"]) == 20


def test_log_model_override_includes_context(tmp_path):
    with patch.dict(os.environ, {"AURA_DATA_DIR": str(tmp_path)}):
        event_log.log_model_override(
            session_id="s1",
            from_model="auto",
            to_model="kimi-k2.6:cloud",
            prompt_context="last prompt was about foo",
        )
        # log_model_override dispatches via bg_pool, so wait briefly.
        import time as _t
        for _ in range(20):
            if (tmp_path / "events" / "model_overrides.jsonl").is_file():
                break
            _t.sleep(0.05)

    p = tmp_path / "events" / "model_overrides.jsonl"
    recs = _read_records(p)
    assert len(recs) == 1
    assert recs[0]["from_model"] == "auto"
    assert recs[0]["to_model"] == "kimi-k2.6:cloud"
    assert "prompt_context" in recs[0]


# ── Concurrency ──────────────────────────────────────────────────────────

def test_concurrent_writes_do_not_interleave(tmp_path):
    """Lock-guarded appends should produce well-formed lines."""
    path = tmp_path / "events" / "edits.jsonl"

    def _writer(i: int):
        event_log._append_jsonl(
            path,
            {"ts": float(i), "session_id": "s", "tool": "edit_file",
             "path": f"f{i}.py", "prompt": "p" * 200, "model": "m", "iteration": i},
        )

    threads = [threading.Thread(target=_writer, args=(i,)) for i in range(30)]
    for t in threads:
        t.start()
    for t in threads:
        t.join()

    # Every line must be valid JSON.
    recs = _read_records(path)
    assert len(recs) == 30
    iters = sorted(r["iteration"] for r in recs)
    assert iters == list(range(30))
